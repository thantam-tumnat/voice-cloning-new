"""The webhook service now that generation lives in another process.

The caller-visible contract must not have moved: same endpoints, same immediate
`{"status":"success"}`, same callback, same job dashboard fields. What changed is
underneath — a render job on the GPU service instead of an in-process synth loop —
so these tests drive the webhook against a fake engine and assert on both halves:
what it sends, and what it does with what comes back.

ffmpeg is real here. The merge is the reason the chunk WAVs are handed over as files
in the first place, so stubbing it would skip the point of that decision.
"""

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src import pipeline
from src.gpu_client import GPUUnavailable


class FakeGPU:
    """Stands in for the GPU service. Records submissions, writes the WAVs a real
    render would have written, and can be told to misbehave."""

    def __init__(self):
        self.submissions = []
        self.health_response = {"model": "openbmb/VoxCPM2", "adapter": "checkpoints/x",
                                "device": "cuda", "sample_rate": 48000, "stub": True,
                                "waiting": {"interactive": 0, "batch": 0}, "running": None,
                                "voices": {"in_memory": 3}}
        self.voices_response = {"voices": [{"id": "demo_female", "file": "demo_female.wav",
                                            "cached": True}]}
        self.fail_with = None            # exception to raise from await_job
        self.job_status = "completed"
        self.write_files = True

    async def health(self):
        return self.health_response

    async def list_voices(self):
        return self.voices_response

    async def submit_render(self, body):
        self.submissions.append(body)
        out = body["output"]
        if self.write_files:
            work = pipeline.WORK_ROOT / out["job_dir"]
            work.mkdir(parents=True, exist_ok=True)
            self.files = []
            for name in out["names"]:
                p = work / f"{name}.wav"
                tone = 0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.5, 24000))
                sf.write(str(p), tone.astype("float32"), 48000)
                self.files.append(str(p))
        else:
            self.files = []
        return {"job_id": "g_fake", "status": "queued", "position": 2}

    async def await_job(self, job_id, *, on_progress=None, **kw):
        if self.fail_with is not None:
            raise self.fail_with
        n = len(self.submissions[-1]["chunks"])
        if on_progress is not None:
            for i in range(1, n + 1):
                on_progress({"chunks_done": i, "position": None, "status": "running"})
        return {
            "job_id": job_id,
            "status": self.job_status,
            "chunks_done": n,
            "error": None if self.job_status == "completed" else "engine exploded",
            "result": {"mode": "files", "files": self.files, "sample_rate": 48000},
        }


@pytest.fixture()
def webhook(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "WORK_ROOT", tmp_path / "work")

    from src import webhook as webhook_mod

    gpu = FakeGPU()
    monkeypatch.setattr(webhook_mod, "GPUClient", lambda url: gpu)

    callbacks, uploads = [], []

    async def fake_upload(path, **kw):
        uploads.append(path)
        return f"https://cdn.test/{path.name}"

    async def fake_callback(url, **kw):
        callbacks.append({"url": url, **kw})

    monkeypatch.setattr(pipeline, "upload", fake_upload)
    monkeypatch.setattr(pipeline, "post_callback", fake_callback)

    with TestClient(webhook_mod.app) as client:
        client.gpu = gpu
        client.callbacks = callbacks
        client.uploads = uploads
        client.work = tmp_path / "work"
        yield client


def post(client, **body):
    body.setdefault("prompt", "สนใจสินค้าตัวไหน กดที่ตะกร้าได้เลยนะคะ")
    body.setdefault("voice_id", "demo_female")
    body.setdefault("callback_url", "https://callback.test/hook")
    return client.post("/webhook/live-ai-create-new", json=body)


def drain(client, queue_id, tries=100):
    """The job runs on the app's own worker task; give it the loop until it lands."""
    for _ in range(tries):
        job = client.get(f"/jobs/{queue_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        client.get("/health")          # yields to the event loop
    return job


# --------------------------------------------------------------------------- #
# The contract callers see
# --------------------------------------------------------------------------- #

def test_the_old_endpoints_both_still_answer(webhook):
    assert post(webhook, queue_id="a").json()["status"] == "success"
    assert webhook.post("/live-ai-create-new",
                        json={"queue_id": "b", "prompt": "ทดสอบ"}).json()["status"] == "success"


def test_accepts_immediately_and_reports_the_chunk_count(webhook):
    body = post(webhook, queue_id="j1").json()
    assert body == {"status": "success", "job_id": "j1", "chunks": 1}


def test_an_empty_script_is_still_rejected_synchronously(webhook):
    res = post(webhook, queue_id="j2", prompt="   ")
    assert res.status_code == 400
    assert res.json()["error"] == "prompt is empty after normalize"


def test_a_delivered_job_uploads_and_calls_back(webhook):
    post(webhook, queue_id="j3")
    job = drain(webhook, "j3")

    assert job["status"] == "completed"
    assert job["file_url"] == "https://cdn.test/j3.mp3"
    assert len(webhook.uploads) == 1
    assert webhook.callbacks[-1]["queue_id"] == "j3"
    assert webhook.callbacks[-1]["error"] is None


def test_the_dashboard_still_gets_every_field_it_reads(webhook):
    post(webhook, queue_id="j4")
    job = drain(webhook, "j4")
    for key in ("job_id", "queue_id", "status", "voice_id", "prompt", "speed",
                "callback_url", "progress", "chunks_total", "chunks_done", "position",
                "created", "created_ts", "started_ts", "finished_ts", "waited_s",
                "elapsed_s", "file_url", "audio_src", "has_local_audio", "error"):
        assert key in job, key
    # New, additive: where the work actually happened.
    assert job["gpu_job_id"] == "g_fake"


def test_progress_tracks_the_remote_job(webhook):
    post(webhook, queue_id="j5", prompt="ประโยคหนึ่ง\nประโยคสอง\nประโยคสาม")
    job = drain(webhook, "j5")
    assert job["chunks_done"] == job["chunks_total"] > 1
    assert job["progress"] == f"{job['chunks_total']}/{job['chunks_total']}"


# --------------------------------------------------------------------------- #
# What it asks the engine for
# --------------------------------------------------------------------------- #

def test_chunks_are_prepared_here_and_sent_ready_to_speak(webhook):
    post(webhook, queue_id="j6", prompt="ราคา 250 บาท")
    drain(webhook, "j6")
    sent = webhook.gpu.submissions[-1]
    # src/thai_text expands numerals before the model ever sees them.
    assert "250" not in sent["chunks"][0]
    assert "บาท" in sent["chunks"][0]


def test_output_lands_in_this_jobs_scratch_dir_under_the_old_names(webhook):
    post(webhook, queue_id="j7", prompt="ประโยคหนึ่ง\nประโยคสอง")
    drain(webhook, "j7")
    out = webhook.gpu.submissions[-1]["output"]
    assert out["mode"] == "files"
    assert out["job_dir"] == "j7"
    assert out["names"] == ["j7_000", "j7_001"]


def test_the_webhook_path_keeps_the_adapters_shipped_lora_strength(webhook):
    """This path has never scaled the LoRA. Sharing a model with the tone studio,
    which runs the DiT side at zero, is exactly why it now has to say so."""
    post(webhook, queue_id="j8")
    drain(webhook, "j8")
    assert webhook.gpu.submissions[-1]["lora"] == "shipped"


def test_it_queues_in_the_batch_lane(webhook):
    post(webhook, queue_id="j9")
    drain(webhook, "j9")
    assert webhook.gpu.submissions[-1]["lane"] == "batch"
    assert webhook.gpu.submissions[-1]["client"] == "webhook"


def test_a_caller_supplied_transcript_suppresses_the_sidecar(webhook):
    """The in-process rule, preserved: a `<clip>.txt` is only consulted when the
    caller had no transcript of its own, because the cache key depends on it."""
    post(webhook, queue_id="ja", ref_text="ข้อความอ้างอิงของผู้เรียก")
    drain(webhook, "ja")
    voice = webhook.gpu.submissions[-1]["voice"]
    assert voice["ref_text"] == "ข้อความอ้างอิงของผู้เรียก"
    assert voice["allow_sidecar"] is False

    post(webhook, queue_id="jb")
    drain(webhook, "jb")
    assert webhook.gpu.submissions[-1]["voice"]["allow_sidecar"] is True


def test_voice_text_is_accepted_as_an_alias_for_ref_text(webhook):
    post(webhook, queue_id="jc", voice_text="สคริปต์อ้างอิง")
    drain(webhook, "jc")
    assert webhook.gpu.submissions[-1]["voice"]["ref_text"] == "สคริปต์อ้างอิง"


# --------------------------------------------------------------------------- #
# When the engine misbehaves
# --------------------------------------------------------------------------- #

def test_a_failed_render_becomes_an_error_callback_not_a_crash(webhook):
    webhook.gpu.job_status = "failed"
    post(webhook, queue_id="jd")
    job = drain(webhook, "jd")

    assert job["status"] == "failed"
    assert "exploded" in job["error"]
    assert webhook.callbacks[-1]["file_url"] is None
    assert webhook.callbacks[-1]["error"] == job["error"]


def test_an_engine_restart_mid_job_fails_that_job_and_explains_why(webhook):
    """The engine's job table is in memory. After a restart the work is gone, and
    waiting out the reconnect budget would only delay the same answer."""
    webhook.gpu.fail_with = GPUUnavailable(
        "GPU service no longer knows job g_fake — it restarted while the job was "
        "queued or running")
    post(webhook, queue_id="je")
    job = drain(webhook, "je")

    assert job["status"] == "failed"
    assert "restarted" in job["error"]
    assert webhook.callbacks[-1]["error"] == job["error"]


def test_failed_jobs_keep_their_scratch_audio(webhook):
    """A failure usually means the upload or callback broke after the GPU work was
    done; wiping the directory there throws away the only copy."""
    webhook.gpu.job_status = "failed"
    post(webhook, queue_id="jf")
    drain(webhook, "jf")
    assert (webhook.work / "jf").exists()


def test_delivered_jobs_clean_up_after_themselves(webhook):
    post(webhook, queue_id="jg")
    drain(webhook, "jg")
    assert not (webhook.work / "jg").exists()


def test_one_bad_job_does_not_stop_the_next_one(webhook):
    webhook.gpu.job_status = "failed"
    post(webhook, queue_id="jh")
    drain(webhook, "jh")

    webhook.gpu.job_status = "completed"
    post(webhook, queue_id="ji")
    assert drain(webhook, "ji")["status"] == "completed"


# --------------------------------------------------------------------------- #
# Status surfaces
# --------------------------------------------------------------------------- #

def test_health_reports_the_engine_it_is_using(webhook):
    body = webhook.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "openbmb/VoxCPM2", "model fields still answer, from the engine"
    assert body["gpu"]["reachable"] is True
    assert body["gpu"]["stub"] is True


def test_health_is_degraded_when_the_engine_is_gone(webhook):
    async def no_health():
        return None

    webhook.gpu.health = no_health
    body = webhook.get("/health").json()
    assert body["status"] == "degraded"
    assert body["gpu"]["reachable"] is False
    assert body["model"] is None


def test_voices_come_from_the_engine(webhook):
    body = webhook.get("/voices").json()
    assert body["default"] == "thai_female"
    assert {"demo_female", "thai_female"} <= {v["id"] for v in body["voices"]}


def test_voices_fall_back_to_a_local_scan_when_the_engine_is_down(webhook):
    async def no_voices():
        return None

    webhook.gpu.list_voices = no_voices
    body = webhook.get("/voices").json()
    assert body["default"] == "thai_female"
    assert isinstance(body["voices"], list)
