"""The studio's side of the split: what it sends to the shared GPU service.

These run against a fake service rather than a live one, so they check the contract
(what goes on the wire, and what the service's answers turn into) without needing a
model or a port.
"""
import io

import numpy as np
import pytest

from app.services.queue_client import QueueSynthesizer, RemoteSynthesisError


class FakeResponse:
    def __init__(self, status_code=200, content=b"", payload=None):
        self.status_code = status_code
        self.content = content
        self._payload = payload
        self.text = "" if payload is None else str(payload)

    def json(self):
        return self._payload


class FakeClient:
    """Records every call and replays canned answers."""

    def __init__(self, log, answers):
        self.log = log
        self.answers = answers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _answer(self, verb, url, **kw):
        self.log.append({"verb": verb, "url": url, **kw})
        for pattern, resp in self.answers:
            if pattern in url:
                return resp
        return FakeResponse(404, payload={"error": "no canned answer"})

    def get(self, url, **kw):
        return self._answer("GET", url, **kw)

    def post(self, url, **kw):
        return self._answer("POST", url, **kw)

    def delete(self, url, **kw):
        return self._answer("DELETE", url, **kw)


def npz_bytes(chunks, sample_rate=48000):
    buf = io.BytesIO()
    np.savez(
        buf,
        sample_rate=np.asarray(sample_rate),
        count=np.asarray(len(chunks)),
        **{f"chunk_{i:03d}": np.asarray(c, dtype="float32") for i, c in enumerate(chunks)},
    )
    return buf.getvalue()


@pytest.fixture
def wired(monkeypatch):
    log = []
    answers = []
    synth = QueueSynthesizer("http://gpu.test")
    monkeypatch.setattr(synth, "_client", lambda timeout=None: FakeClient(log, answers))
    return synth, log, answers


def test_health_picks_up_sample_rate_and_stub_flag(wired):
    synth, log, answers = wired
    answers.append(("/health", FakeResponse(200, payload={
        "sample_rate": 24000, "adapter": "checkpoints/x", "stub": True})))
    assert synth.check_health() is True
    assert synth.sample_rate == 24000
    assert synth.lora_loaded is True
    assert synth.is_stub is True


def test_health_failure_is_not_an_exception(wired):
    synth, log, answers = wired
    answers.append(("/health", FakeResponse(503, payload={})))
    assert synth.check_health() is False


def test_a_run_of_chunks_goes_as_one_job(wired):
    synth, log, answers = wired
    answers.append(("/v2/jobs/render", FakeResponse(200, content=npz_bytes(
        [np.zeros(10), np.ones(10), np.zeros(10)]))))

    chunks, sr = synth.render_batch(["หนึ่ง", "สอง", "สาม"], prompt_cache="vh_abc")

    renders = [c for c in log if "render" in c["url"]]
    assert len(renders) == 1, "one job, not one request per chunk"
    body = renders[0]["json"]
    assert len(body["chunks"]) == 3
    assert body["voice"] == {"handle": "vh_abc"}, "every chunk shares one voice"
    assert body["output"]["mode"] == "npz"
    assert len(chunks) == 3 and sr == 48000


def test_lora_scales_travel_with_the_request(wired):
    from app.config import settings

    synth, log, answers = wired
    answers.append(("/v2/jobs/render", FakeResponse(200, content=npz_bytes([np.zeros(4)]))))

    synth.render_batch(["x"], lora_mode="on")
    sent = log[-1]["json"]["lora"]
    assert sent == {"lm": settings.siangtts_lora_lm_scale, "dit": settings.siangtts_lora_dit_scale}

    log.clear()
    synth.render_batch(["x"], lora_mode="off")
    assert log[-1]["json"]["lora"] == {"lm": 0.0, "dit": 0.0}

    log.clear()
    synth.render_batch(["x"], lora_mode="legacy")
    assert log[-1]["json"]["lora"] == {"lm": 2.0, "dit": 2.0}


def test_named_speaker_is_resolved_rather_than_uploaded(wired):
    synth, log, answers = wired
    answers.append(("/v2/voices/resolve", FakeResponse(200, payload={"voice_handle": "vh_named"})))
    answers.append(("/v2/jobs/render", FakeResponse(200, content=npz_bytes([np.zeros(4)]))))

    synth.render_batch(["x"], speaker_id="thai_female")

    resolve_call = [c for c in log if "resolve" in c["url"]][0]
    assert resolve_call["json"]["allow_sidecar"] is False, "tone studio must disable sidecar to keep reference mode"
    assert log[-1]["json"]["voice"] == {"handle": "vh_named"}
    assert not any("files" in c for c in log), "no clip should be uploaded for a named voice"


def test_a_handle_beats_a_speaker_name(wired):
    synth, log, answers = wired
    answers.append(("/v2/jobs/render", FakeResponse(200, content=npz_bytes([np.zeros(4)]))))
    synth.render_batch(["x"], prompt_cache="vh_pinned", speaker_id="thai_female")
    assert log[-1]["json"]["voice"] == {"handle": "vh_pinned"}
    assert not any("resolve" in c["url"] for c in log)


def test_unpinned_request_sends_no_voice(wired):
    synth, log, answers = wired
    answers.append(("/v2/jobs/render", FakeResponse(200, content=npz_bytes([np.zeros(4)]))))
    synth.render_batch(["x"])
    assert log[-1]["json"]["voice"] is None


def test_seed_voice_comes_from_the_service(wired):
    synth, log, answers = wired
    answers.append(("/v2/voices/seed", FakeResponse(200, payload={"voice_handle": "_auto_seed"})))
    assert synth.seed_voice() == "_auto_seed"
    assert log[-1]["verb"] == "POST"


def test_seed_voice_failure_is_none_not_a_crash(wired):
    synth, log, answers = wired
    answers.append(("/v2/voices/seed", FakeResponse(503, payload={"error": "nope"})))
    assert synth.seed_voice() is None


def test_a_job_that_outlives_the_wait_says_so(wired):
    synth, log, answers = wired
    answers.append(("/v2/jobs/render", FakeResponse(202, payload={"job_id": "g_slow"})))
    with pytest.raises(RemoteSynthesisError, match="g_slow"):
        synth.render_batch(["x"])


def test_render_failure_surfaces_the_status(wired):
    synth, log, answers = wired
    answers.append(("/v2/jobs/render", FakeResponse(500, payload={"error": "boom"})))
    with pytest.raises(RemoteSynthesisError, match="500"):
        synth.render_batch(["x"])


def test_pronunciation_and_normalisation_are_applied_before_sending(wired, monkeypatch):
    """The studio owns text preparation; the service is a byte-exact passthrough."""
    synth, log, answers = wired
    answers.append(("/v2/jobs/render", FakeResponse(200, content=npz_bytes([np.zeros(4)]))))

    import app.services.siangtts_service as svc

    monkeypatch.setattr(svc, "prepare_text", lambda t: f"<prepared>{t}")
    synth.render_batch(["ดิบ"])
    assert log[-1]["json"]["chunks"] == ["<prepared>ดิบ"]


def test_style_instruction_is_left_alone(wired):
    synth, log, answers = wired
    answers.append(("/v2/jobs/render", FakeResponse(200, content=npz_bytes([np.zeros(4)]))))
    tagged = "(Sad and melancholic voice, slight sighs)วันนี้เหนื่อยจัง"
    synth.render_batch([tagged])
    assert log[-1]["json"]["chunks"][0].startswith("(Sad and melancholic voice, slight sighs)")
