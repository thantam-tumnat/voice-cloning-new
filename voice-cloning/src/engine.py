"""The one process that owns the model: queue, worker, and job execution.

Everything that needs the GPU happens here and nowhere else. The webhook service
(:8010) and the tone studio (:8011) are clients over HTTP — see `src/gpu_service.py`
for the wire format and `src/gpu_client.py` for the caller side.

Two things make a single worker the right shape rather than a semaphore:

* there is one GPU, so generation is serial no matter how it is expressed; and
* LoRA strength is *global mutable state* on the model (`src/lora.py`), and the two
  pipelines want different values. Serialising jobs is what makes it safe to set the
  scale per job instead of freezing one pipeline's preference into the process.

Lanes exist because the two clients have different patience. A webhook script is
minutes of batch work nobody is watching; a studio click is a person waiting. The
interactive lane jumps the queue, with a cap so a busy studio cannot starve
production traffic.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import lora as lora_mod
from .voices import UnknownVoice, VoiceStore

LANES = ("interactive", "batch")

# How many interactive jobs may jump ahead of a waiting batch job before one batch
# job goes through regardless. Production traffic is never starved; the studio still
# gets served first in the common case where the batch lane is empty or shallow.
INTERACTIVE_BURST = int(os.environ.get("SIANGTTS_INTERACTIVE_BURST", "3"))

# Finished jobs kept for /v2/jobs. State is in memory, so this is the only thing
# stopping a long-lived process from growing without bound.
MAX_HISTORY = int(os.environ.get("SIANGTTS_GPU_MAX_HISTORY", "500"))


@dataclass
class RenderJob:
    chunks: list[str]
    voice: Optional[dict] = None
    cfg_value: float = 2.0
    timesteps: int = 10
    lora: Any = None
    output: dict = field(default_factory=lambda: {"mode": "npz"})
    lane: str = "batch"
    client: str = ""
    job_id: str = field(default_factory=lambda: f"g_{uuid.uuid4().hex[:10]}")

    status: str = "queued"              # queued | running | completed | failed | cancelled
    done: int = 0
    error: Optional[str] = None
    result: Optional[dict] = None       # metadata; bytes live in `payload`
    payload: Optional[bytes] = None
    voice_handle: Optional[str] = None
    lora_applied: Optional[dict] = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    def as_dict(self, position: Optional[int] = None) -> dict:
        now = time.time()
        ran = ((self.finished or now) - self.started) if self.started else None
        return {
            "job_id": self.job_id,
            "client": self.client,
            "lane": self.lane,
            "status": self.status,
            "position": position,
            "chunks_total": len(self.chunks),
            "chunks_done": self.done,
            "chunks": self.chunks,
            "progress": f"{self.done}/{len(self.chunks)}",
            "voice_handle": self.voice_handle,
            "lora": self.lora_applied,
            "created_ts": self.created,
            "started_ts": self.started,
            "finished_ts": self.finished,
            "waited_s": round((self.started or now) - self.created, 1),
            "elapsed_s": round(ran, 1) if ran is not None else None,
            "result": self.result,
            "error": self.error,
        }


class Engine:
    """Model + voice store + queue. One instance per process."""

    def __init__(
        self,
        synth: Any,
        voices: VoiceStore,
        work_root: Path,
        *,
        default_lora: Any = None,
    ) -> None:
        self.synth = synth
        self.voices = voices
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.default_lora = default_lora

        self.jobs: dict[str, RenderJob] = {}
        self.queues: dict[str, asyncio.Queue] = {lane: asyncio.Queue() for lane in LANES}
        self.running: Optional[str] = None
        self._worker: Optional[asyncio.Task] = None
        self._interactive_streak = 0
        # What the model is currently scaled to, so an unchanged scale costs nothing.
        self._lora_state: Optional[tuple[float, float]] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_forever())

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.synth, "sample_rate", 48000))

    @property
    def is_stub(self) -> bool:
        return bool(getattr(self.synth, "is_stub", False))

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #

    def submit(self, job: RenderJob) -> RenderJob:
        if job.lane not in self.queues:
            job.lane = "batch"
        self.jobs[job.job_id] = job
        self._evict_history()
        self.queues[job.lane].put_nowait(job)
        return job

    def cancel(self, job_id: str) -> bool:
        """Cancel a job that has not started. A running job is left alone — there is
        no way to interrupt a generation mid-chunk, and killing the worker would take
        every other queued job with it."""
        job = self.jobs.get(job_id)
        if job is None or job.status != "queued":
            return False
        job.status = "cancelled"
        job.finished = time.time()
        job._event.set()
        return True

    async def wait(self, job_id: str, timeout: float) -> Optional[RenderJob]:
        """Block until the job finishes, or return None on timeout.

        This is what makes the studio's synchronous UI work without a polling loop:
        it posts a job, waits out the generation on the same connection, and gets
        audio back. A timeout is not an error — the job keeps running and the caller
        falls back to polling.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.terminal:
            return job
        try:
            await asyncio.wait_for(job._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return job

    def positions(self) -> dict[str, int]:
        """Place in line, 1-based, in the order the worker will actually take them.

        Reproduced from the job records rather than read off the queues, which are
        opaque — jobs are enqueued in creation order within a lane, and the worker
        prefers interactive, so interleaving the two by that rule gives the true
        order barring an anti-starvation swap.
        """
        waiting = [j for j in self.jobs.values() if j.status == "queued"]
        inter = sorted((j for j in waiting if j.lane == "interactive"), key=lambda j: j.created)
        batch = sorted((j for j in waiting if j.lane == "batch"), key=lambda j: j.created)
        return {j.job_id: i for i, j in enumerate(inter + batch, 1)}

    def _evict_history(self) -> None:
        """Bounded history — insertion-ordered, so drop the oldest finished jobs
        first and never evict anything still queued or running."""
        if len(self.jobs) <= MAX_HISTORY:
            return
        for jid, j in list(self.jobs.items()):
            if len(self.jobs) <= MAX_HISTORY:
                break
            if j.terminal:
                del self.jobs[jid]

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #

    async def _next_job(self) -> RenderJob:
        while True:
            inter, batch = self.queues["interactive"], self.queues["batch"]

            if not inter.empty() and (batch.empty() or self._interactive_streak < INTERACTIVE_BURST):
                self._interactive_streak += 1
                return inter.get_nowait()
            if not batch.empty():
                self._interactive_streak = 0
                return batch.get_nowait()
            if not inter.empty():
                self._interactive_streak += 1
                return inter.get_nowait()

            # Both empty: wait on whichever fills first. No lane preference to apply
            # — with nothing queued there is nothing to jump ahead of. If both fire
            # at once, take one and put the other back for the next round.
            getters = [asyncio.ensure_future(q.get()) for q in (inter, batch)]
            done, pending = await asyncio.wait(getters, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
            first = done.pop()
            for extra in done:                    # both fired at once; re-queue the loser
                job = extra.result()
                self.queues[job.lane].put_nowait(job)
            return first.result()

    async def _run_forever(self) -> None:
        while True:
            job = await self._next_job()
            if job.status == "cancelled":
                continue
            try:
                await self._execute(job)
            except Exception:                                        # noqa: BLE001
                traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def _apply_lora(self, spec: Any) -> dict:
        lm, dit = lora_mod.resolve(spec if spec is not None else self.default_lora)
        if self._lora_state == (lm, dit):
            return {"lm": lm, "dit": dit, "unchanged": True}
        applied = lora_mod.set_lora_strength(getattr(self.synth, "tts_model", None), lm, dit)
        self._lora_state = (lm, dit)
        return applied

    def _resolve_voice(self, spec: Optional[dict], client: str = "") -> Optional[str]:
        """Voice spec -> handle. Returns None for an unconditioned generation."""
        if not spec:
            return None
        if spec.get("handle"):
            self.voices.get(spec["handle"])        # raises UnknownVoice -> 410
            return spec["handle"]
        if spec.get("speaker_id"):
            allow_sidecar = spec.get("allow_sidecar")
            if allow_sidecar is None:
                allow_sidecar = False if client == "tone-studio" else True
            return self.voices.resolve_speaker(
                spec["speaker_id"],
                spec.get("ref_text") or "",
                allow_sidecar=bool(allow_sidecar),
            )
        if spec.get("seed"):
            return self.voices.seed(
                lambda text: self.synth.synth(text, cfg_value=2.0, inference_timesteps=10)
            )
        return None

    def _generate(self, text: str, cache: Any, job: RenderJob):
        if cache is None:
            return self.synth.synth(
                text, cfg_value=job.cfg_value, inference_timesteps=job.timesteps
            )
        return self.synth.synth_cached(
            text, cache, cfg_value=job.cfg_value, inference_timesteps=job.timesteps
        )

    async def _execute(self, job: RenderJob) -> None:
        job.status = "running"
        job.started = time.time()
        self.running = job.job_id
        try:
            job.voice_handle = await asyncio.to_thread(self._resolve_voice, job.voice, job.client)
            cache = self.voices.get(job.voice_handle) if job.voice_handle else None
            job.lora_applied = await asyncio.to_thread(self._apply_lora, job.lora)

            client_label = job.client or "unknown"
            print(f"\n[gpu] >>> Running Job {job.job_id} from '{client_label}' (lane={job.lane}, voice={job.voice_handle or 'unpinned'}, {len(job.chunks)} chunk(s)):")
            for idx, chunk_text in enumerate(job.chunks):
                print(f"[gpu]     [{idx+1}/{len(job.chunks)}] {chunk_text!r}")

            # `files` mode writes each chunk as it is generated rather than at the
            # end. Two reasons: a long script would otherwise hold every chunk's
            # audio in memory until the last one lands, and a job that dies partway
            # leaves nothing behind to inspect or re-merge — which is what the
            # in-process version used to give the webhook for free.
            sink = self._open_sink(job)
            for i, text in enumerate(job.chunks):
                wav = await asyncio.to_thread(self._generate, text, cache, job)
                await asyncio.to_thread(sink.write, i, wav)
                job.done = i + 1

            job.result, job.payload = await asyncio.to_thread(sink.finish)
            job.status = "completed"

        except UnknownVoice as exc:
            job.status = "failed"
            job.error = f"unknown voice: {exc}"
        except Exception as exc:                                     # noqa: BLE001
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            job.finished = time.time()
            self.running = None
            job._event.set()

    # -- delivery -------------------------------------------------------- #

    def _job_dir(self, name: str) -> Path:
        """Resolve a client-supplied directory *name* under the work root.

        A name, never a path: the client picks what the folder is called, the service
        decides where it lives. Otherwise `job_dir` would be an instruction to write
        anywhere on the host's disk.

        Anything with a separator or a `..` in it is rejected rather than trimmed
        down to its last component. Silently writing to a different directory than
        the one asked for would leave the caller looking for files that are not
        there, which is a worse failure than a clear one.
        """
        safe = str(name).strip()
        if not safe or safe in (".", "..") or safe != Path(safe).name:
            raise ValueError(f"invalid job_dir {name!r} — must be a plain folder name")
        out = (self.work_root / safe).resolve()
        if not str(out).startswith(str(self.work_root.resolve())):
            raise ValueError(f"job_dir {name!r} escapes the work root")
        return out

    def _open_sink(self, job: RenderJob) -> "_Sink":
        mode = (job.output or {}).get("mode", "npz")
        if mode == "files":
            return _FileSink(
                self._job_dir(job.output.get("job_dir") or job.job_id),
                job.output.get("names") or [],
                self.sample_rate,
            )
        if mode == "npz":
            return _NpzSink(self.sample_rate)
        raise ValueError(f"unknown output mode {mode!r}")


class _Sink:
    """Where a job's chunks go as they are generated."""

    def write(self, index: int, wav) -> None:                        # pragma: no cover
        raise NotImplementedError

    def finish(self) -> tuple[dict, Optional[bytes]]:                # pragma: no cover
        raise NotImplementedError


class _FileSink(_Sink):
    """WAVs in a directory both services can see. The caller's next step is ffmpeg,
    so putting the audio on disk saves pushing it through HTTP only to be written
    out again at the other end."""

    def __init__(self, out_dir: Path, names: list[str], sample_rate: int) -> None:
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.names = names
        self.sample_rate = sample_rate
        self.paths: list[str] = []

    def write(self, index: int, wav) -> None:
        import soundfile as sf

        stem = (
            Path(self.names[index]).name
            if index < len(self.names)
            else f"{self.dir.name}_{index:03d}"
        )
        path = self.dir / f"{stem}.wav"
        sf.write(str(path), wav, self.sample_rate)
        self.paths.append(str(path))

    def finish(self) -> tuple[dict, Optional[bytes]]:
        return {
            "mode": "files",
            "dir": str(self.dir),
            "files": self.paths,
            "sample_rate": self.sample_rate,
        }, None


class _NpzSink(_Sink):
    """float32 arrays in one bundle, for a caller assembling audio in memory."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.wavs: list = []

    def write(self, index: int, wav) -> None:
        self.wavs.append(wav)

    def finish(self) -> tuple[dict, Optional[bytes]]:
        import numpy as np

        buf = io.BytesIO()
        np.savez(
            buf,
            sample_rate=np.asarray(self.sample_rate),
            count=np.asarray(len(self.wavs)),
            **{f"chunk_{i:03d}": np.asarray(w, dtype="float32") for i, w in enumerate(self.wavs)},
        )
        data = buf.getvalue()
        return {
            "mode": "npz",
            "chunks": len(self.wavs),
            "sample_rate": self.sample_rate,
            "bytes": len(data),
        }, data


__all__ = ["Engine", "RenderJob", "LANES", "INTERACTIVE_BURST", "MAX_HISTORY"]
