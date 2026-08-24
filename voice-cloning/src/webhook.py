"""Webhook service — the whole n8n "LIveAI_Audio" flow as one process.

Same contract as the old webhook (POST a script, get `{"status":"success"}`
back immediately, receive the finished audio URL on `callback_url` later), but
the twenty nodes in between collapse into a single in-process job:

    prepare (src/thai_text) -> synth every chunk -> merge -> upload -> callback

What that buys over the n8n version: no 3-second polling granularity, no HTTP
hop per chunk, and the reference voice is encoded once per job instead of
re-read on every chunk.

The synth step no longer happens here. This process used to load VoxCPM2 itself,
which meant a second copy of the model whenever the tone studio was also running —
on one GPU that is a copy nobody can afford. Generation now goes to the shared GPU
service (src/gpu_service.py, :8020) and this process keeps everything else: the n8n
contract, Thai text preparation, chunking, the merge, the upload, the callback and
the job dashboard. Nothing a caller can see changed.

The chunk WAVs come back as files in the shared work directory rather than over
HTTP, because the very next thing that happens to them is ffmpeg — so the two
services must share SIANGTTS_WORK_DIR, which is also why they belong on one host.

Run (after src/gpu_service.py is up):
    uv run uvicorn src.webhook:app --host 0.0.0.0 --port 8010

Config (env):
    SIANGTTS_GPU_URL        shared GPU service   (default http://127.0.0.1:8020)
    SIANGTTS_REF_DIR        reference clips      (default ref/)
                            Only for listing voices — the GPU service is what
                            actually reads them, so point both at the same place.
    SIANGTTS_WORK_DIR       job scratch          (default work/)
                            Must match the GPU service's work dir.
    SIANGTTS_UPLOAD_URL     upload endpoint
    SIANGTTS_UPLOAD_TOKEN   bearer for upload   (was n8n credential "VR_live Auth")
    SIANGTTS_DEFAULT_CALLBACK
    SIANGTTS_KEEP_WORK      "1" keeps job scratch dirs for debugging

    Model settings (SIANGTTS_BASE_MODEL / _ADAPTER / _DEVICE / _CACHE_DIR) moved to
    the GPU service. Setting them here does nothing.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import pipeline
from .gpu_client import DEFAULT_URL as GPU_URL_DEFAULT
from .gpu_client import GPUClient
from .thai_text import Chunk, chunk_text, prepare_prompt

pipeline._load_dotenv()

GPU_URL = os.environ.get("SIANGTTS_GPU_URL", GPU_URL_DEFAULT)
REF_DIR = Path(os.environ.get("SIANGTTS_REF_DIR", "ref"))
KEEP_WORK = os.environ.get("SIANGTTS_KEEP_WORK", "") == "1"

AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")

# Defaults lifted from the flow's `set` node and its send_prompt_new body.
DEFAULT_VOICE = "thai_female"
DEFAULT_CALLBACK = os.environ.get(
    "SIANGTTS_DEFAULT_CALLBACK",
    "https://test.looklike.ai/api/v1/live-gpt/n8n/audio-callback",
)
DEFAULT_REF_TEXT = (
    "ริ้วรอยลดเลือน จุดด่างดำจางลง ผิวไร้ปัญหาสิว "
    "บำรุงล้ำลึกจากภายในเซลล์ผิว ผิวกระจ่างใสเนียนสวยแบบนี้ได้ใจเลยค่ะ"
)
# n8n: num_step. The flow inherited 32 from IndexTTS, where a step is cheap.
# Here every LM step runs the flow-matching DiT this many times, doubled again by
# CFG — 32 costs 64 DiT forwards per step and put a single chunk at ~1s/step on an
# uncompiled build. voxcpm's own default is 10, and the difference is inaudible.
NUM_STEP = int(os.environ.get("SIANGTTS_NUM_STEP", "10"))
GUIDANCE = float(os.environ.get("SIANGTTS_GUIDANCE", "2"))         # n8n: guidance_scale

# Finished jobs kept for /jobs. State is in memory, so this is the only thing
# stopping a long-lived process from growing without bound.
MAX_HISTORY = int(os.environ.get("SIANGTTS_MAX_HISTORY", "500"))


# ---------------------------------------------------------------------------
# Request / job model
# ---------------------------------------------------------------------------

class WebhookBody(BaseModel):
    """Body the old webhook received. Unknown fields (session_id, section,
    label, and the IndexTTS-only knobs) are accepted and ignored."""

    prompt: str = ""
    job_id: str = ""
    queue_id: str = ""
    voice_id: str = ""
    voice_text: str = ""
    ref_text: str = ""
    audio_speed: float = 1.0
    country_code: str = "th"
    callback_url: str = ""


@dataclass
class Job:
    job_id: str
    queue_id: str
    voice_id: str
    ref_text: str
    speed: float
    callback_url: str
    chunks: list[Chunk]
    prompt: str = ""
    status: str = "queued"              # queued | running | completed | failed
    error: str | None = None
    file_url: str | None = None
    done: int = 0                       # chunks synthesised so far
    gpu_job_id: str | None = None       # the render job on the GPU service
    gpu_position: int | None = None     # its place in the shared queue, while waiting
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def as_dict(self, position: int | None = None) -> dict:
        now = time.time()
        waited = (self.started or now) - self.created
        ran = ((self.finished or now) - self.started) if self.started else None
        local_mp3 = pipeline.WORK_ROOT / self.queue_id / f"{self.queue_id}.mp3"
        local_wav = pipeline.WORK_ROOT / self.queue_id / f"{self.queue_id}_000.wav"
        has_audio = local_mp3.exists() or local_wav.exists()
        audio_src = self.file_url or (f"/audio/{self.queue_id}" if has_audio else None)
        return {
            "job_id": self.job_id,
            "queue_id": self.queue_id,
            "status": self.status,
            "voice_id": self.voice_id,
            "prompt": self.prompt,
            "speed": self.speed,
            "callback_url": self.callback_url,
            "progress": f"{self.done}/{len(self.chunks)}",
            "chunks_total": len(self.chunks),
            "chunks_done": self.done,
            "position": position,       # place in line; None once it starts
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created)),
            "created_ts": self.created,
            "started_ts": self.started,
            "finished_ts": self.finished,
            "waited_s": round(waited, 1),
            "elapsed_s": round(ran, 1) if ran is not None else None,
            "file_url": self.file_url,
            "audio_src": audio_src,
            "has_local_audio": has_audio,
            "gpu_job_id": self.gpu_job_id,
            "gpu_position": self.gpu_position,
            "error": self.error,
        }



# ---------------------------------------------------------------------------
# App + worker
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["gpu"] = GPUClient(GPU_URL)
    _state["jobs"] = {}
    _state["running"] = None
    _state["queue"] = asyncio.Queue()

    # First thaisum/newmm call loads its model; do it now so job #1 isn't slower.
    await asyncio.to_thread(prepare_prompt, "อุ่นเครื่องนะคะ", "th")

    # A warning, not a failure. The GPU service takes 30–60 s to load the model, so
    # on a cold boot it is normal for this process to come up first; jobs submitted
    # meanwhile wait it out (see GPUClient.await_job) rather than being rejected.
    health = await _state["gpu"].health()
    if health is None:
        print(f"[webhook] WARNING: GPU service at {GPU_URL} is not answering yet")
    elif health.get("stub"):
        print(f"[webhook] WARNING: GPU service at {GPU_URL} is in STUB MODE — output is a test tone")
    else:
        print(f"[webhook] GPU service ok — {health.get('model')} adapter={health.get('adapter')}")

    worker = asyncio.create_task(_worker())
    pipeline.WORK_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[webhook] ready — gpu={GPU_URL} work={pipeline.WORK_ROOT}")
    try:
        yield
    finally:
        worker.cancel()
        _state.clear()


app = FastAPI(title="SiangTTS webhook", version="1.1.0", lifespan=lifespan)


async def _worker() -> None:
    """Single consumer: one GPU, so jobs run strictly one at a time. Queueing
    rather than gating with a semaphore also gives the caller backpressure for
    free — bursts wait instead of thrashing VRAM."""
    queue: asyncio.Queue = _state["queue"]
    while True:
        job: Job = await queue.get()
        try:
            await _run_job(job)
        except Exception:                                        # noqa: BLE001
            traceback.print_exc()
        finally:
            queue.task_done()


async def _run_job(job: Job) -> None:
    gpu: GPUClient = _state["gpu"]
    work = pipeline.WORK_ROOT / job.queue_id
    job.status = "running"
    job.started = time.time()
    _state["running"] = job.job_id

    try:
        work.mkdir(parents=True, exist_ok=True)

        has_style_tags = any(re.match(r"^\s*[\[(]", ch.text) for ch in job.chunks)
        use_ref_text = "" if has_style_tags else job.ref_text
        use_sidecar = (job.ref_text == DEFAULT_REF_TEXT) and not has_style_tags

        # The GPU service writes the chunk WAVs straight into this job's scratch dir,
        # under the same `<queue_id>_NNN.wav` names the in-process version produced —
        # which is what keeps /audio/{queue_id} and the merge unchanged.
        submitted = await gpu.submit_render({
            "chunks": [ch.text for ch in job.chunks],
            "voice": {
                "speaker_id": job.voice_id,
                "ref_text": use_ref_text,
                "allow_sidecar": use_sidecar,
            },
            "cfg_value": GUIDANCE,
            "timesteps": NUM_STEP,
            # Use 'tones' mode when emotion tags are present, otherwise shipped strength.
            "lora": "tones" if has_style_tags else "shipped",
            "output": {
                "mode": "files",
                "job_dir": job.queue_id,
                "names": [ch.filename for ch in job.chunks],
            },
            "lane": "batch",
            "client": "webhook",
        })
        job.gpu_job_id = submitted["job_id"]
        job.gpu_position = submitted.get("position")
        print(f"[{job.queue_id}] gpu job {job.gpu_job_id} queued at position "
              f"{job.gpu_position}")

        def _progress(remote: dict) -> None:
            if remote["chunks_done"] != job.done:
                job.done = remote["chunks_done"]
                print(f"[{job.queue_id}] chunk {job.done}/{len(job.chunks)} ok")
            job.gpu_position = remote.get("position")

        remote = await gpu.await_job(job.gpu_job_id, on_progress=_progress)
        if remote["status"] != "completed":
            raise RuntimeError(remote.get("error") or f"gpu job {remote['status']}")

        wav_paths = [Path(p) for p in remote["result"]["files"]]
        job.done = len(wav_paths)

        merged = work / f"{job.queue_id}.mp3"
        await asyncio.to_thread(
            pipeline.merge_chunks,
            wav_paths,
            merged,
            pipeline.MergeOptions(speed=job.speed),
        )

        size_kb = round(merged.stat().st_size / 1024) if merged.exists() else 0
        disp_action = (
            "kept" if KEEP_WORK else "deleted after upload — SIANGTTS_KEEP_WORK=1 to keep"
        )
        print(f"[{job.queue_id}] merged -> {merged.resolve()} ({size_kb} KB)  [{disp_action}]")

        job.file_url = await pipeline.upload(merged)
        job.status = "completed"
        await pipeline.post_callback(
            job.callback_url,
            job_id=job.job_id,
            queue_id=job.queue_id,
            file_url=job.file_url,
            error=None,
        )
        print(f"[{job.queue_id}] done -> {job.file_url}")

    except Exception as exc:                                     # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)
        traceback.print_exc()
        try:
            await pipeline.post_callback(
                job.callback_url,
                job_id=job.job_id,
                queue_id=job.queue_id,
                file_url=None,
                error=job.error,
            )
        except Exception as cb_exc:                               # noqa: BLE001
            print(f"[{job.queue_id}] error callback failed: {cb_exc}")

    finally:
        job.finished = time.time()
        _state["running"] = None
        # Only a delivered job is safe to delete. A failure usually means the
        # upload or callback broke *after* the GPU work was already done, and
        # wiping the scratch dir there throws away minutes of synthesis and the
        # only copy of the audio — leaving nothing to inspect or re-upload.
        if job.status == "completed" and not KEEP_WORK:
            pipeline.cleanup(work)
        elif job.status != "completed" and work.exists():
            print(f"[{job.queue_id}] failed - audio kept at {work.resolve()}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

async def _accept(body: WebhookBody) -> JSONResponse:
    """Validate + enqueue. Text prep runs here, not in the worker, so a bad
    script is rejected in the HTTP response instead of only via callback."""
    queue_id = body.queue_id or str(uuid.uuid4())
    callback_url = body.callback_url or DEFAULT_CALLBACK

    prompt = await asyncio.to_thread(prepare_prompt, body.prompt, body.country_code)
    if not prompt:
        # The n8n chunker returned [] here and nothing downstream fired, so the
        # webhook hung until timeout. Fail out loud instead.
        return JSONResponse(
            {"status": "error", "error": "prompt is empty after normalize"}, status_code=400
        )

    chunks = chunk_text(prompt, prefix=queue_id)
    job = Job(
        job_id=body.job_id or queue_id,
        queue_id=queue_id,
        voice_id=body.voice_id.strip() or DEFAULT_VOICE,
        ref_text=(body.ref_text or body.voice_text).strip() or DEFAULT_REF_TEXT,
        speed=body.audio_speed or 1.0,
        callback_url=callback_url,
        chunks=chunks,
        prompt=body.prompt,
    )
    jobs: dict[str, Job] = _state["jobs"]
    jobs[job.job_id] = job
    # Bounded history — the dict is insertion-ordered, so drop the oldest
    # finished jobs first and never evict anything still queued or running.
    if len(jobs) > MAX_HISTORY:
        for jid, j in list(jobs.items()):
            if len(jobs) <= MAX_HISTORY:
                break
            if j.status in ("completed", "failed"):
                del jobs[jid]

    await _state["queue"].put(job)
    print(f"[{queue_id}] queued — {len(chunks)} chunk(s), voice={job.voice_id}")
    return JSONResponse({"status": "success", "job_id": job.job_id, "chunks": len(chunks)})


@app.post("/webhook/live-ai-create-new")
async def webhook(body: WebhookBody) -> JSONResponse:
    return await _accept(body)


# The n8n webhook lived at /webhook/<path>; keep the bare path too so callers
# that were pointed straight at the node still resolve.
@app.post("/live-ai-create-new")
async def webhook_bare(body: WebhookBody) -> JSONResponse:
    return await _accept(body)


def _positions() -> dict[str, int]:
    """Place in line for everything still waiting, 1-based. The queue itself is
    opaque, but jobs are enqueued in creation order, so ordering the waiting
    ones by `created` reproduces it."""
    waiting = sorted(
        (j for j in _state.get("jobs", {}).values() if j.status == "queued"),
        key=lambda j: j.created,
    )
    return {j.job_id: i for i, j in enumerate(waiting, 1)}


@app.get("/jobs")
def list_jobs(status: str | None = None, limit: int = 100) -> JSONResponse:
    """Everything the service remembers, newest first. `?status=failed` to
    filter — this is the surface that replaces scrolling n8n's execution list."""
    jobs: list[Job] = list(_state.get("jobs", {}).values())
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.status] = counts.get(j.status, 0) + 1

    pos = _positions()
    jobs.sort(key=lambda j: j.created, reverse=True)
    if status:
        jobs = [j for j in jobs if j.status == status]
    return JSONResponse({
        "counts": counts,
        "running": _state.get("running"),
        "total": len(jobs),
        "jobs": [j.as_dict(pos.get(j.job_id)) for j in jobs[:limit]],
    })


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job: Job | None = _state.get("jobs", {}).get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return JSONResponse(job.as_dict(_positions().get(job_id)))


@app.get("/health")
async def health() -> JSONResponse:
    """Status of this service *and* the engine behind it.

    The model-shaped fields still exist and still say what model is answering — they
    are just reported by the GPU service now rather than read off a local object, so
    anything that was watching them keeps working. `gpu` is where to look when they
    come back null: it is null exactly when the engine is unreachable.
    """
    gpu: GPUClient = _state["gpu"]
    remote = await gpu.health()
    queue: asyncio.Queue | None = _state.get("queue")
    jobs: list[Job] = list(_state.get("jobs", {}).values())
    return JSONResponse({
        "status": "ok" if remote else "degraded",
        "model": (remote or {}).get("model"),
        "adapter": (remote or {}).get("adapter"),
        "device": (remote or {}).get("device"),
        "sample_rate": (remote or {}).get("sample_rate", 48000),
        "num_step": NUM_STEP,
        "guidance": GUIDANCE,
        "waiting": queue.qsize() if queue else 0,
        "running": _state.get("running"),
        "completed": sum(1 for j in jobs if j.status == "completed"),
        "failed": sum(1 for j in jobs if j.status == "failed"),
        "total": len(jobs),
        "voices_cached": ((remote or {}).get("voices") or {}).get("in_memory", 0),
        "upload_token": bool(
            os.environ.get("SIANGTTS_UPLOAD_TOKEN", "").strip() or pipeline.UPLOAD_TOKEN
        ),
        "gpu": {
            "url": GPU_URL,
            "reachable": remote is not None,
            "stub": (remote or {}).get("stub"),
            "waiting": (remote or {}).get("waiting"),
            "running": (remote or {}).get("running"),
        },
    })


@app.get("/voices")
async def list_voices() -> JSONResponse:
    """List available reference voices and cached prompt caches.

    Answered by the GPU service, which is what actually reads the reference
    directories and owns the prompt caches. Falls back to a local directory scan when
    it is unreachable, so the dashboard still shows something useful during a restart
    — without the `cached` flags, which only the engine knows.
    """
    remote = await _state["gpu"].list_voices()
    if remote is not None:
        voices = remote.get("voices", [])
    else:
        voices = []
        seen = set()
        dirs_to_check = [REF_DIR]
        fallback_dir = Path("C:/temp/tts_jobs/voices")
        if fallback_dir.exists() and fallback_dir.resolve() != REF_DIR.resolve():
            dirs_to_check.append(fallback_dir)
        for d in dirs_to_check:
            if d.exists():
                for f in d.iterdir():
                    if f.suffix.lower() in AUDIO_EXTS and f.stem not in seen:
                        seen.add(f.stem)
                        voices.append({"id": f.stem, "file": f.name, "cached": None})

    # If the default voice is not in a reference directory, list it anyway.
    if not any(v["id"] == DEFAULT_VOICE for v in voices):
        voices.insert(0, {"id": DEFAULT_VOICE, "file": f"{DEFAULT_VOICE}.mp3", "cached": False})
    return JSONResponse({"voices": voices, "default": DEFAULT_VOICE})


@app.get("/audio/{queue_id}")
async def get_audio(queue_id: str):
    """Serve locally generated audio for direct dashboard preview."""
    mp3_path = pipeline.WORK_ROOT / queue_id / f"{queue_id}.mp3"
    if mp3_path.exists():
        return FileResponse(mp3_path, media_type="audio/mpeg", filename=f"{queue_id}.mp3")
    wav_path = pipeline.WORK_ROOT / queue_id / f"{queue_id}_000.wav"
    if wav_path.exists():
        return FileResponse(wav_path, media_type="audio/wav", filename=f"{queue_id}.wav")
    return JSONResponse({"error": "audio not found in local scratch"}, status_code=404)


# ---------------------------------------------------------------------------
# Queue Monitoring Dashboard UI
# ---------------------------------------------------------------------------

QUEUE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SiangTTS · Voice Cloning Queue</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>
  :root {
    color-scheme: light dark;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Sarabun", Roboto, Helvetica, Arial, sans-serif;
    --font-mono: ui-monospace, "SF Mono", "Cascadia Code", "Cascadia Mono", Menlo, Consolas, monospace;
    
    --bg: #090d16;
    --surface: #111726;
    --surface-elevated: #1a2236;
    --surface-hover: #222d47;
    --border: #263352;
    --border-subtle: #1c2740;
    
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    
    --primary: #38bdf8;
    --primary-glow: rgba(56, 189, 248, 0.15);
    --run: #38bdf8;
    --run-bg: rgba(56, 189, 248, 0.12);
    --ok: #34d399;
    --ok-bg: rgba(52, 211, 153, 0.12);
    --fail: #f87171;
    --fail-bg: rgba(248, 113, 113, 0.12);
    --wait: #fbbf24;
    --wait-bg: rgba(251, 191, 36, 0.12);
  }

  [data-theme="light"] {
    --bg: #f8fafc;
    --surface: #ffffff;
    --surface-elevated: #f1f5f9;
    --surface-hover: #e2e8f0;
    --border: #cbd5e1;
    --border-subtle: #e2e8f0;
    
    --text: #0f172a;
    --text-muted: #475569;
    --text-dim: #94a3b8;
    
    --primary: #0284c7;
    --primary-glow: rgba(2, 132, 199, 0.12);
    --run: #0284c7;
    --run-bg: #e0f2fe;
    --ok: #16a34a;
    --ok-bg: #dcfce7;
    --fail: #dc2626;
    --fail-bg: #fee2e2;
    --wait: #d97706;
    --wait-bg: #fef3c7;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
    padding: 1.5rem 2rem 3rem;
  }

  /* Header */
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1.5rem;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .brand-logo {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0284c7, #38bdf8);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 800;
    font-size: 1.15rem;
    box-shadow: 0 4px 12px var(--primary-glow);
  }

  .brand-title h1 {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .brand-title p {
    font-size: 0.8rem;
    color: var(--text-muted);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--ok);
    box-shadow: 0 0 8px var(--ok);
    display: inline-block;
    animation: pulse 2s infinite ease-in-out;
  }
  .status-dot.disconnected { background: var(--fail); box-shadow: 0 0 8px var(--fail); }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
  }

  /* System Info Badges */
  .sys-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 0.3rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 500;
  }

  .pill b { color: var(--text); }
  .pill.success { border-color: var(--ok); color: var(--ok); }
  .pill.warning { border-color: var(--wait); color: var(--wait); }

  /* Controls */
  .header-actions {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.45rem 0.85rem;
    border-radius: 8px;
    font-size: 0.825rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    font-family: inherit;
  }
  .btn:hover { background: var(--surface-hover); border-color: var(--primary); }
  .btn:active { transform: scale(0.98); }
  .btn-primary { background: var(--primary); color: #090d16; border-color: var(--primary); font-weight: 600; }
  .btn-primary:hover { opacity: 0.9; color: #090d16; }
  .btn-icon { padding: 0.45rem; width: 34px; height: 34px; }

  select.select-control {
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.45rem 0.75rem;
    border-radius: 8px;
    font-size: 0.825rem;
    cursor: pointer;
    font-family: inherit;
  }

  /* Metric KPI Grid */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.85rem;
    margin-bottom: 1.5rem;
  }

  .kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    cursor: pointer;
    transition: all 0.15s ease;
    position: relative;
    overflow: hidden;
  }
  .kpi-card:hover { transform: translateY(-2px); border-color: var(--text-dim); }
  .kpi-card.active { border-color: var(--primary); box-shadow: 0 0 0 1px var(--primary); }

  .kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
    font-variant-numeric: tabular-nums;
  }

  .kpi-sub {
    font-size: 0.725rem;
    color: var(--text-dim);
    margin-top: 0.25rem;
  }

  .kpi-card.running .kpi-value { color: var(--run); }
  .kpi-card.queued .kpi-value { color: var(--wait); }
  .kpi-card.completed .kpi-value { color: var(--ok); }
  .kpi-card.failed .kpi-value { color: var(--fail); }

  /* Active Running Spotlight */
  .running-banner {
    display: none;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(17, 23, 38, 0.9));
    border: 1px solid var(--run);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    position: relative;
    box-shadow: 0 4px 20px var(--primary-glow);
  }
  .running-banner.active { display: block; }
  .running-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
  .running-title { font-weight: 600; font-size: 0.9rem; color: var(--run); display: flex; align-items: center; gap: 0.5rem; }
  .running-timer { font-family: var(--font-mono); font-size: 1.1rem; font-weight: 700; color: var(--run); }
  .running-prompt { color: var(--text); font-size: 0.875rem; margin-bottom: 0.75rem; font-style: italic; }
  .progress-bar-bg { height: 6px; background: rgba(255,255,255,0.1); border-radius: 9999px; overflow: hidden; }
  .progress-bar-fill { height: 100%; width: 50%; background: var(--run); border-radius: 9999px; animation: shimmer 1.5s infinite linear; }
  @keyframes shimmer { 0% { opacity: 0.6; } 50% { opacity: 1; } 100% { opacity: 0.6; } }

  /* Toolbar */
  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .tabs {
    display: flex;
    gap: 0.35rem;
    background: var(--surface);
    padding: 0.25rem;
    border-radius: 10px;
    border: 1px solid var(--border);
  }

  .tab {
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-muted);
    background: transparent;
    border: none;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }
  .tab:hover { color: var(--text); }
  .tab.active { background: var(--surface-elevated); color: var(--text); font-weight: 600; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
  .tab-badge {
    background: var(--surface-hover);
    padding: 0.1rem 0.4rem;
    border-radius: 999px;
    font-size: 0.7rem;
  }

  .search-box {
    position: relative;
    min-width: 240px;
  }
  .search-input {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.75rem 0.4rem 2rem;
    color: var(--text);
    font-size: 0.825rem;
    font-family: inherit;
  }
  .search-input:focus { outline: none; border-color: var(--primary); }
  .search-icon {
    position: absolute;
    left: 0.65rem;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-dim);
    pointer-events: none;
  }

  /* Table Container */
  .table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.825rem;
    text-align: left;
  }

  th {
    background: var(--surface-elevated);
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.725rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }

  td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border-subtle);
    vertical-align: middle;
  }

  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--surface-hover); }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .badge.running { background: var(--run-bg); color: var(--run); border: 1px solid var(--run); }
  .badge.queued { background: var(--wait-bg); color: var(--wait); border: 1px solid var(--wait); }
  .badge.completed { background: var(--ok-bg); color: var(--ok); border: 1px solid var(--ok); }
  .badge.failed { background: var(--fail-bg); color: var(--fail); border: 1px solid var(--fail); }

  .mono { font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-muted); }
  .num { font-variant-numeric: tabular-nums; }
  
  .copy-btn {
    background: transparent;
    border: none;
    color: var(--text-dim);
    cursor: pointer;
    padding: 2px 4px;
    border-radius: 4px;
    transition: all 0.15s ease;
  }
  .copy-btn:hover { color: var(--primary); background: var(--surface-hover); }

  .prompt-snippet {
    max-width: 240px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--text);
  }

  .tag-voice {
    display: inline-block;
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.15rem 0.45rem;
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  /* Audio player inline */
  .audio-action {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .btn-play {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    color: var(--text);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn-play:hover { background: var(--primary); color: #090d16; border-color: var(--primary); }
  .btn-play.playing { background: var(--ok); color: #090d16; border-color: var(--ok); }

  .empty-state {
    padding: 3rem;
    text-align: center;
    color: var(--text-muted);
  }
  .empty-state svg { width: 42px; height: 42px; margin-bottom: 0.75rem; color: var(--text-dim); }

  /* Modal */
  .modal-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.65);
    backdrop-filter: blur(4px);
    z-index: 100;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
  }
  .modal-overlay.active { display: flex; }

  .modal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    max-width: 600px;
    width: 100%;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    overflow: hidden;
    animation: modalPop 0.15s ease-out;
  }
  @keyframes modalPop {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
  }

  .modal-header {
    padding: 1.25rem 1.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .modal-header h2 { font-size: 1.1rem; font-weight: 700; }
  .modal-body { padding: 1.5rem; overflow-y: auto; display: flex; flex-direction: column; gap: 1.25rem; }
  .modal-footer {
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
  }

  .form-group { display: flex; flex-direction: column; gap: 0.4rem; }
  .form-group label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); }
  .form-control {
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    color: var(--text);
    font-size: 0.85rem;
    font-family: inherit;
  }
  .form-control:focus { outline: none; border-color: var(--primary); }
  textarea.form-control { min-height: 80px; resize: vertical; }

  .preset-pills { display: flex; gap: 0.35rem; flex-wrap: wrap; margin-top: 0.25rem; }
  .preset-pill {
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
    font-size: 0.725rem;
    cursor: pointer;
    color: var(--text-muted);
  }
  .preset-pill:hover { border-color: var(--primary); color: var(--primary); }

  /* Toast Notification */
  .toast {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1.25rem;
    color: var(--text);
    font-size: 0.85rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    z-index: 200;
    display: none;
    animation: toastSlide 0.2s ease-out;
  }
  .toast.active { display: block; }
  @keyframes toastSlide { from { transform: translateY(100%); opacity: 0; } to { transform: translateY(0); opacity: 1; } }

  pre.json-view {
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
    overflow-x: auto;
    max-height: 220px;
  }

  @media (max-width: 768px) {
    body { padding: 1rem; }
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-logo">S</div>
    <div class="brand-title">
      <h1>SiangTTS Queue <span class="status-dot" id="status-dot" title="Connected"></span></h1>
      <p>Real-time Voice Synthesis &amp; Webhook Monitor</p>
    </div>
  </div>

  <div class="sys-info">
    <div class="pill" id="pill-model">Model: <b>VoxCPM2</b></div>
    <div class="pill" id="pill-lora">LoRA: <b>siangtts-v1</b></div>
    <div class="pill" id="pill-sr">Audio: <b>48 kHz</b></div>
    <div class="pill" id="pill-steps">Steps: <b>10</b></div>
    <div class="pill" id="pill-upload">Upload: <b id="val-upload">—</b></div>
  </div>

  <div class="header-actions">
    <button class="btn btn-primary" onclick="openTestModal()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 5v14M5 12h14"/></svg>
      New Test Job
    </button>
    <select class="select-control" id="refresh-interval" onchange="updateRefreshInterval(this.value)">
      <option value="1000">Refresh 1s</option>
      <option value="2000" selected>Refresh 2s</option>
      <option value="5000">Refresh 5s</option>
      <option value="0">Paused</option>
    </select>
    <button class="btn btn-icon" title="Refresh Now" onclick="tick()">
      <svg id="refresh-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
    </button>
    <button class="btn btn-icon" title="Toggle Dark/Light Mode" onclick="toggleTheme()">
      <svg id="theme-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    </button>
  </div>
</header>

<!-- KPI Stats Grid -->
<div class="kpi-grid">
  <div class="kpi-card running" id="card-running" onclick="setFilter('running')">
    <div class="kpi-label">Running <span class="badge running" style="font-size:0.65rem;">Active</span></div>
    <div class="kpi-value" id="kpi-running">0</div>
    <div class="kpi-sub">GPU Synthesis</div>
  </div>
  <div class="kpi-card queued" id="card-queued" onclick="setFilter('queued')">
    <div class="kpi-label">In Queue <span class="badge queued" style="font-size:0.65rem;">Waiting</span></div>
    <div class="kpi-value" id="kpi-queued">0</div>
    <div class="kpi-sub">FIFO Line</div>
  </div>
  <div class="kpi-card completed" id="card-completed" onclick="setFilter('completed')">
    <div class="kpi-label">Completed <span class="badge completed" style="font-size:0.65rem;">Done</span></div>
    <div class="kpi-value" id="kpi-completed">0</div>
    <div class="kpi-sub" id="kpi-avg-time">Avg: —</div>
  </div>
  <div class="kpi-card failed" id="card-failed" onclick="setFilter('failed')">
    <div class="kpi-label">Failed <span class="badge failed" style="font-size:0.65rem;">Errors</span></div>
    <div class="kpi-value" id="kpi-failed">0</div>
    <div class="kpi-sub">Needs Attention</div>
  </div>
  <div class="kpi-card active" id="card-all" onclick="setFilter('all')">
    <div class="kpi-label">Total Jobs</div>
    <div class="kpi-value" id="kpi-total">0</div>
    <div class="kpi-sub" id="meta-updated">Connecting…</div>
  </div>
</div>

<!-- Active Running Spotlight Banner -->
<div class="running-banner" id="running-banner">
  <div class="running-header">
    <div class="running-title">
      <span class="status-dot"></span>
      Synthesizing on GPU: <b id="run-job-id">—</b> &bull; Voice: <span class="tag-voice" id="run-voice">—</span> &bull; <span id="run-chunk">Chunk 1/1</span>
    </div>
    <div class="running-timer" id="run-timer">0.0s</div>
  </div>
  <div class="running-prompt" id="run-prompt">"..."</div>
  <div class="progress-bar-bg"><div class="progress-bar-fill"></div></div>
</div>

<!-- Toolbar -->
<div class="toolbar">
  <div class="tabs">
    <button class="tab active" id="tab-all" onclick="setFilter('all')">All <span class="tab-badge" id="tab-badge-all">0</span></button>
    <button class="tab" id="tab-running" onclick="setFilter('running')">Running <span class="tab-badge" id="tab-badge-running">0</span></button>
    <button class="tab" id="tab-queued" onclick="setFilter('queued')">Queued <span class="tab-badge" id="tab-badge-queued">0</span></button>
    <button class="tab" id="tab-completed" onclick="setFilter('completed')">Completed <span class="tab-badge" id="tab-badge-completed">0</span></button>
    <button class="tab" id="tab-failed" onclick="setFilter('failed')">Failed <span class="tab-badge" id="tab-badge-failed">0</span></button>
  </div>

  <div class="search-box">
    <svg class="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
    <input type="text" class="search-input" id="search-input" placeholder="Search Job ID, voice, prompt, error…" oninput="renderTable()">
  </div>
</div>

<!-- Table -->
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th style="width: 40px;">#</th>
        <th style="width: 100px;">Status</th>
        <th style="width: 170px;">Job ID</th>
        <th>Prompt Script</th>
        <th style="width: 110px;">Voice</th>
        <th style="width: 70px;">Chunks</th>
        <th style="width: 110px;">Created</th>
        <th style="width: 80px;">Waited</th>
        <th style="width: 80px;">Elapsed</th>
        <th style="width: 120px;">Result / Audio</th>
      </tr>
    </thead>
    <tbody id="table-body">
      <tr><td colspan="10" class="empty-state">Loading queue data…</td></tr>
    </tbody>
  </table>
</div>

<!-- Job Details Modal -->
<div class="modal-overlay" id="modal-details" onclick="if(event.target===this)closeDetailsModal()">
  <div class="modal">
    <div class="modal-header">
      <h2 id="det-title">Job Details</h2>
      <button class="copy-btn" onclick="closeDetailsModal()">&times;</button>
    </div>
    <div class="modal-body" id="det-body">
      <!-- Injected by JS -->
    </div>
    <div class="modal-footer">
      <button class="btn" onclick="copyDetailsJSON()">Copy JSON</button>
      <button class="btn btn-primary" onclick="closeDetailsModal()">Close</button>
    </div>
  </div>
</div>

<!-- Quick Test Job Modal -->
<div class="modal-overlay" id="modal-test" onclick="if(event.target===this)closeTestModal()">
  <div class="modal">
    <div class="modal-header">
      <h2>Create Test Voice Job</h2>
      <button class="copy-btn" onclick="closeTestModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div class="form-group">
        <label>Thai Prompt Script</label>
        <textarea class="form-control" id="test-prompt" placeholder="ใส่ข้อความภาษาไทยที่ต้องการสร้างเสียง...">ถ้าพี่กำลังหาหูฟังที่ใส่แล้วเสียงแน่น เบสมา แต่ยังฟังรายละเอียดชัด สนใจสินค้าตัวไหนกดที่ตะกร้าได้เลยนะคะ</textarea>
        <div class="preset-pills">
          <span class="preset-pill" onclick="setPreset('ecommerce')">🛍️ E-Commerce Live</span>
          <span class="preset-pill" onclick="setPreset('service')">💬 Customer Service</span>
          <span class="preset-pill" onclick="setPreset('short')">⚡ Quick Test</span>
        </div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem;">
        <div class="form-group">
          <label>Voice ID</label>
          <select class="form-control" id="test-voice">
            <option value="demo_female">demo_female</option>
            <option value="thai_female">thai_female</option>
            <option value="demo_male">demo_male</option>
          </select>
        </div>

        <div class="form-group">
          <label>Audio Speed (<span id="test-speed-val">1.0x</span>)</label>
          <input type="range" class="form-control" id="test-speed" min="0.8" max="1.5" step="0.05" value="1.0" oninput="document.getElementById('test-speed-val').textContent=this.value+'x'">
        </div>
      </div>

      <div class="form-group">
        <label>Optional Callback URL</label>
        <input type="text" class="form-control" id="test-callback" placeholder="https://webhook.site/... (Leave empty for default)">
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn" onclick="closeTestModal()">Cancel</button>
      <button class="btn btn-primary" id="btn-submit-test" onclick="submitTestJob()">
        Submit to Queue
      </button>
    </div>
  </div>
</div>

<!-- Global Audio Element -->
<audio id="global-audio" style="display:none;"></audio>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
let currentData = { counts: {}, running: null, total: 0, jobs: [] };
let currentFilter = 'all';
let pollTimer = null;
let currentPlayingUrl = null;
let selectedJob = null;
let runningStartLocal = null;

// Helpers
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const secs = v => v == null ? "—" : v < 60 ? v.toFixed(1) + "s" : Math.floor(v / 60) + "m " + Math.round(v % 60) + "s";

function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("active");
  setTimeout(() => t.classList.remove("active"), 2500);
}

function copyText(txt, label="Copied!") {
  navigator.clipboard.writeText(txt).then(() => showToast(label));
}

// Filter Handling
function setFilter(f) {
  currentFilter = f;
  ['all', 'running', 'queued', 'completed', 'failed'].forEach(s => {
    const tab = document.getElementById(`tab-${s}`);
    const card = document.getElementById(`card-${s}`);
    if (tab) tab.classList.toggle('active', s === f);
    if (card) card.classList.toggle('active', s === f);
  });
  renderTable();
}

// Table Rendering
function renderTable() {
  const query = (document.getElementById("search-input").value || "").trim().toLowerCase();
  let list = currentData.jobs || [];

  if (currentFilter !== 'all') {
    list = list.filter(j => j.status === currentFilter);
  }

  if (query) {
    list = list.filter(j => 
      (j.job_id && j.job_id.toLowerCase().includes(query)) ||
      (j.voice_id && j.voice_id.toLowerCase().includes(query)) ||
      (j.prompt && j.prompt.toLowerCase().includes(query)) ||
      (j.error && j.error.toLowerCase().includes(query))
    );
  }

  const tbody = document.getElementById("table-body");
  if (!list.length) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg>
      <div>No jobs found ${query ? 'matching search query' : `in status '${currentFilter}'`}</div>
    </td></tr>`;
    return;
  }

  tbody.innerHTML = list.map((j, idx) => {
    const audioUrl = j.file_url || (j.has_local_audio ? `/audio/${j.queue_id}` : null);
    const isPlaying = currentPlayingUrl && currentPlayingUrl === audioUrl;
    const shortJob = j.job_id ? (j.job_id.length > 18 ? j.job_id.slice(0, 16) + "…" : j.job_id) : "—";
    const promptSnippet = j.prompt ? esc(j.prompt) : `<span style="color:var(--text-dim);">—</span>`;

    return `<tr onclick="openDetailsModal('${esc(j.job_id)}')" style="cursor:pointer;">
      <td class="num" style="color:var(--text-dim); font-weight:600;">${j.position ? '#' + j.position : idx + 1}</td>
      <td><span class="badge ${esc(j.status)}">${esc(j.status)}</span></td>
      <td>
        <span class="mono" title="${esc(j.job_id)}">${esc(shortJob)}</span>
        <button class="copy-btn" title="Copy Job ID" onclick="event.stopPropagation();copyText('${esc(j.job_id)}', 'Job ID copied!')">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        </button>
      </td>
      <td><div class="prompt-snippet" title="${esc(j.prompt || '')}">${promptSnippet}</div></td>
      <td><span class="tag-voice">${esc(j.voice_id || 'default')}</span></td>
      <td class="num">${esc(j.progress || '0/1')}</td>
      <td class="num" style="color:var(--text-muted); font-size:0.75rem;">${esc(j.created || '').split(' ')[1] || esc(j.created || '')}</td>
      <td class="num">${secs(j.waited_s)}</td>
      <td class="num">${secs(j.elapsed_s)}</td>
      <td onclick="event.stopPropagation();">
        <div class="audio-action">
          ${audioUrl ? `
            <button class="btn-play ${isPlaying ? 'playing' : ''}" title="${isPlaying ? 'Pause' : 'Play Audio'}" onclick="toggleAudio('${esc(audioUrl)}')">
              ${isPlaying ? 
                `<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>` : 
                `<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`}
            </button>
            <a href="${esc(audioUrl)}" target="_blank" class="btn" style="padding:0.25rem 0.5rem; font-size:0.75rem;" title="Download / Open Audio">URL</a>
          ` : (j.error ? `<span style="color:var(--fail); font-size:0.75rem;">Error</span>` : `<span style="color:var(--text-dim);">—</span>`)}
        </div>
      </td>
    </tr>`;
  }).join("");
}

// UI State Update
function updateUI(d) {
  currentData = d;

  // Counts
  const counts = d.counts || {};
  const running = counts.running || 0;
  const queued = counts.queued || 0;
  const completed = counts.completed || 0;
  const failed = counts.failed || 0;
  const total = d.total || 0;

  document.getElementById("kpi-running").textContent = running;
  document.getElementById("kpi-queued").textContent = queued;
  document.getElementById("kpi-completed").textContent = completed;
  document.getElementById("kpi-failed").textContent = failed;
  document.getElementById("kpi-total").textContent = total;

  document.getElementById("tab-badge-all").textContent = total;
  document.getElementById("tab-badge-running").textContent = running;
  document.getElementById("tab-badge-queued").textContent = queued;
  document.getElementById("tab-badge-completed").textContent = completed;
  document.getElementById("tab-badge-failed").textContent = failed;

  // Avg time for completed
  const completedJobs = (d.jobs || []).filter(j => j.status === 'completed' && j.elapsed_s != null);
  if (completedJobs.length > 0) {
    const avg = completedJobs.reduce((acc, j) => acc + j.elapsed_s, 0) / completedJobs.length;
    document.getElementById("kpi-avg-time").textContent = `Avg: ${avg.toFixed(1)}s`;
  } else {
    document.getElementById("kpi-avg-time").textContent = "Avg: —";
  }

  // Running Spotlight
  const banner = document.getElementById("running-banner");
  const activeJob = (d.jobs || []).find(j => j.status === 'running');
  if (activeJob) {
    banner.classList.add("active");
    document.getElementById("run-job-id").textContent = activeJob.job_id;
    document.getElementById("run-voice").textContent = activeJob.voice_id;
    document.getElementById("run-chunk").textContent = `Chunk ${activeJob.progress || '1/1'}`;
    document.getElementById("run-prompt").textContent = `"${activeJob.prompt || 'Synthesizing voice chunk...'}"`;
    document.getElementById("run-timer").textContent = secs(activeJob.elapsed_s);
  } else {
    banner.classList.remove("active");
  }

  document.getElementById("meta-updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
  document.getElementById("status-dot").classList.remove("disconnected");
  renderTable();
}

// Audio Player
function toggleAudio(url) {
  const player = document.getElementById("global-audio");
  if (currentPlayingUrl === url && !player.paused) {
    player.pause();
    currentPlayingUrl = null;
  } else {
    player.src = url;
    player.play().catch(e => showToast("Audio play failed: " + e.message));
    currentPlayingUrl = url;
  }
  renderTable();
}

document.getElementById("global-audio").addEventListener("ended", () => {
  currentPlayingUrl = null;
  renderTable();
});

// Modal Details
function openDetailsModal(jobId) {
  const job = (currentData.jobs || []).find(j => j.job_id === jobId);
  if (!job) return;
  selectedJob = job;
  const modal = document.getElementById("modal-details");
  document.getElementById("det-title").textContent = `Job: ${job.job_id}`;

  const audioUrl = job.file_url || (job.has_local_audio ? `/audio/${job.queue_id}` : null);

  let html = `
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <span class="badge ${esc(job.status)}" style="font-size:0.85rem;">${esc(job.status)}</span>
      <span class="mono">Queue ID: ${esc(job.queue_id)}</span>
    </div>

    ${job.prompt ? `
      <div class="form-group">
        <label>Thai Prompt Script</label>
        <div style="background:var(--surface-elevated); border:1px solid var(--border); border-radius:8px; padding:0.75rem; font-size:0.9rem; line-height:1.6;">
          ${esc(job.prompt)}
        </div>
      </div>
    ` : ''}

    ${audioUrl ? `
      <div class="form-group">
        <label>Audio Playback</label>
        <audio controls style="width:100%; border-radius:8px;" src="${esc(audioUrl)}"></audio>
      </div>
    ` : ''}

    ${job.error ? `
      <div class="form-group">
        <label style="color:var(--fail);">Error Message</label>
        <div style="background:var(--fail-bg); border:1px solid var(--fail); color:var(--fail); border-radius:8px; padding:0.75rem; font-family:var(--font-mono); font-size:0.75rem; white-space:pre-wrap;">
          ${esc(job.error)}
        </div>
      </div>
    ` : ''}

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; font-size:0.8rem;">
      <div class="pill">Voice: <b>${esc(job.voice_id)}</b></div>
      <div class="pill">Speed: <b>${job.speed || 1.0}x</b></div>
      <div class="pill">Progress: <b>${esc(job.progress)} chunks</b></div>
      <div class="pill">Elapsed: <b>${secs(job.elapsed_s)}</b></div>
    </div>

    <div class="form-group">
      <label>Raw Job JSON</label>
      <pre class="json-view">${esc(JSON.stringify(job, null, 2))}</pre>
    </div>
  `;

  document.getElementById("det-body").innerHTML = html;
  modal.classList.add("active");
}

function closeDetailsModal() {
  document.getElementById("modal-details").classList.remove("active");
}

function copyDetailsJSON() {
  if (selectedJob) copyText(JSON.stringify(selectedJob, null, 2), "Job JSON copied!");
}

// Quick Test Modal
function openTestModal() {
  document.getElementById("modal-test").classList.add("active");
  fetchVoices();
}
function closeTestModal() {
  document.getElementById("modal-test").classList.remove("active");
}

function setPreset(type) {
  const p = document.getElementById("test-prompt");
  if (type === 'ecommerce') {
    p.value = "ถ้าพี่กำลังหาหูฟังที่ใส่แล้วเสียงแน่น เบสมา แต่ยังฟังรายละเอียดชัด สนใจสินค้าตัวไหนกดที่ตะกร้าได้เลยนะคะ";
  } else if (type === 'service') {
    p.value = "สวัสดีค่ะ ยินดีต้อนรับเข้าสู่ระบบ SiangTTS หากมีข้อสงสัยสอบถามเพิ่มเติมได้ตลอดเวลาค่ะ";
  } else {
    p.value = "ทดสอบระบบเสียงภาษาไทย SiangTTS หนึ่ง สอง สาม";
  }
}

async function fetchVoices() {
  try {
    const res = await fetch("/voices");
    if (res.ok) {
      const data = await res.json();
      const sel = document.getElementById("test-voice");
      if (data.voices && data.voices.length > 0) {
        sel.innerHTML = data.voices.map(v => 
          `<option value="${esc(v.id)}" ${v.id === data.default ? 'selected' : ''}>${esc(v.id)} ${v.cached ? '(cached)' : ''}</option>`
        ).join("");
      }
    }
  } catch (e) {}
}

async function submitTestJob() {
  const btn = document.getElementById("btn-submit-test");
  const prompt = document.getElementById("test-prompt").value.trim();
  const voice_id = document.getElementById("test-voice").value;
  const speed = parseFloat(document.getElementById("test-speed").value) || 1.0;
  const callback_url = document.getElementById("test-callback").value.trim();

  if (!prompt) {
    showToast("กรุณาใส่ข้อความ Prompt");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Enqueuing...";

  try {
    const res = await fetch("/webhook/live-ai-create-new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        voice_id: voice_id,
        audio_speed: speed,
        callback_url: callback_url
      })
    });
    const d = await res.json();
    if (res.ok) {
      showToast(`Job queued: ${d.job_id || 'Success'}`);
      closeTestModal();
      setFilter('all');
      tick();
    } else {
      showToast("Error: " + (d.error || res.statusText));
    }
  } catch (e) {
    showToast("Request failed: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Submit to Queue";
  }
}

// Fetch loop
async function fetchHealth() {
  try {
    const r = await fetch("/health");
    if (r.ok) {
      const h = await r.json();
      if (h.upload_token) {
        document.getElementById("val-upload").textContent = "Bearer Set";
        document.getElementById("pill-upload").className = "pill success";
      } else {
        document.getElementById("val-upload").textContent = "Local Mode";
        document.getElementById("pill-upload").className = "pill warning";
      }
      if (h.num_step) document.getElementById("pill-steps").innerHTML = `Steps: <b>${h.num_step}</b>`;
    }
  } catch (e) {}
}

async function tick() {
  const icon = document.getElementById("refresh-icon");
  if (icon) icon.style.transform = "rotate(180deg)";
  try {
    const r = await fetch("/jobs?limit=100", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    updateUI(await r.json());
  } catch (e) {
    document.getElementById("meta-updated").textContent = "Disconnected — " + e.message;
    document.getElementById("status-dot").classList.add("disconnected");
  } finally {
    if (icon) setTimeout(() => icon.style.transform = "none", 200);
  }
}

function updateRefreshInterval(ms) {
  clearInterval(pollTimer);
  const val = parseInt(ms, 10);
  if (val > 0) {
    pollTimer = setInterval(tick, val);
  }
}

function toggleTheme() {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  if (isLight) {
    document.documentElement.removeAttribute("data-theme");
    localStorage.setItem("theme", "dark");
  } else {
    document.documentElement.setAttribute("data-theme", "light");
    localStorage.setItem("theme", "light");
  }
}

// Init
if (localStorage.getItem("theme") === "light") {
  document.documentElement.setAttribute("data-theme", "light");
}
fetchHealth();
tick();
pollTimer = setInterval(tick, 2000);
</script>
</body>
</html>
"""


@app.get("/queue", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def queue_page() -> HTMLResponse:
    """Live view of the queue - monitoring dashboard."""
    return HTMLResponse(QUEUE_PAGE)
