"""The GPU service's contract, exercised against the stub engine.

No model and no GPU: `SIANGTTS_GPU_STUB=1` swaps in src/stub_synth.py, which keeps
the parts these tests care about — that the right voice reached the generator, that
chunks come back in order, that a job's LoRA scale is applied — while making no
sound worth listening to.
"""
import importlib
import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def service(tmp_path, monkeypatch):
    """A fresh service per test, pointed at throwaway directories.

    Reloaded rather than imported once because the module reads its configuration at
    import time — and pointing the cache at tmp_path matters: a stub-built cache
    dropped into the real voices/ would be loaded by the next production run.
    """
    ref = tmp_path / "ref"
    ref.mkdir()
    _write_wav(ref / "alice.wav")
    _write_wav(ref / "bob.wav")

    monkeypatch.setenv("SIANGTTS_GPU_STUB", "1")
    monkeypatch.setenv("SIANGTTS_STUB_DELAY", "0")
    monkeypatch.setenv("SIANGTTS_REF_DIR", str(ref))
    monkeypatch.setenv("SIANGTTS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SIANGTTS_WORK_DIR", str(tmp_path / "work"))

    import src.gpu_service as gpu_service

    importlib.reload(gpu_service)
    with TestClient(gpu_service.app) as client:
        client.ref_dir = ref
        client.work_dir = tmp_path / "work"
        yield client


def _write_wav(path, seconds=1.0, sr=48000):
    import soundfile as sf

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)


def unpack(response):
    with np.load(io.BytesIO(response.content)) as bundle:
        return (
            [bundle[f"chunk_{i:03d}"] for i in range(int(bundle["count"]))],
            int(bundle["sample_rate"]),
        )


def pitch(wav, sr=48000):
    """The stub encodes the voice it was conditioned on as the tone's frequency."""
    return float(np.sum(np.diff(np.signbit(wav)) != 0)) / (len(wav) / sr) / 2


def render(client, **body):
    body.setdefault("chunks", ["สวัสดีค่ะ"])
    return client.post("/v2/jobs/render?wait=30", json=body)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

def test_health_admits_it_is_a_stub(service):
    body = service.get("/health").json()
    assert body["status"] == "ok"
    assert body["stub"] is True, "a client must be able to tell this is not the model"
    assert body["sample_rate"] == 48000


# --------------------------------------------------------------------------- #
# Voices
# --------------------------------------------------------------------------- #

def test_named_voice_resolves_and_is_cached_under_the_legacy_key(service):
    first = service.post("/v2/voices/resolve", json={"speaker_id": "alice"})
    assert first.status_code == 200
    handle = first.json()["voice_handle"]

    # The webhook used `<voice>-<sha1(ref_text)[:8]>.pt` in-process. Keeping the key
    # identical is what stops the split from re-encoding every voice already on disk.
    import hashlib

    assert handle == f"alice-{hashlib.sha1(b'').hexdigest()[:8]}"
    second = service.post("/v2/voices/resolve", json={"speaker_id": "alice"})
    assert second.json()["voice_handle"] == handle


def test_different_ref_text_is_a_different_voice(service):
    """VoxCPM2 binds a prompt cache to the transcript it was built with."""
    a = service.post("/v2/voices/resolve", json={"speaker_id": "alice"}).json()
    b = service.post("/v2/voices/resolve",
                     json={"speaker_id": "alice", "ref_text": "สวัสดี"}).json()
    assert a["voice_handle"] != b["voice_handle"]


def test_sidecar_transcript_is_only_used_when_the_caller_allows_it(service):
    (service.ref_dir / "alice.txt").write_text("บทพูดอ้างอิง", encoding="utf-8")

    with_sidecar = service.post(
        "/v2/voices/resolve", json={"speaker_id": "alice", "allow_sidecar": True}).json()
    without = service.post(
        "/v2/voices/resolve", json={"speaker_id": "alice", "allow_sidecar": False}).json()
    assert with_sidecar["voice_handle"] != without["voice_handle"]


def test_unknown_speaker_is_a_404(service):
    res = service.post("/v2/voices/resolve", json={"speaker_id": "nobody"})
    assert res.status_code == 404
    assert "nobody" in res.json()["error"]


def test_uploaded_clip_becomes_a_throwaway_handle(service, tmp_path):
    clip = tmp_path / "upload.wav"
    _write_wav(clip)
    res = service.post("/v2/voices", files={"clip": ("upload.wav", clip.read_bytes(), "audio/wav")})
    assert res.status_code == 200
    assert res.json()["voice_handle"].startswith("up_")


def test_registering_a_speaker_keeps_the_clip(service, tmp_path):
    clip = tmp_path / "carol.wav"
    _write_wav(clip)
    res = service.post(
        "/v2/voices",
        data={"speaker_id": "carol", "save_as_speaker": "true"},
        files={"clip": ("carol.wav", clip.read_bytes(), "audio/wav")},
    )
    assert res.status_code == 200
    assert (service.ref_dir / "carol.wav").exists()
    assert "carol" in {v["id"] for v in service.get("/v2/voices").json()["voices"]}


def test_deleting_a_speaker_removes_clip_and_cache(service):
    service.post("/v2/voices/resolve", json={"speaker_id": "bob"})
    assert service.delete("/v2/voices/bob").status_code == 200
    assert not (service.ref_dir / "bob.wav").exists()
    assert service.post("/v2/voices/resolve", json={"speaker_id": "bob"}).status_code == 404


def test_seed_voice_is_minted_once_and_shared(service):
    first = service.post("/v2/voices/seed").json()["voice_handle"]
    second = service.post("/v2/voices/seed").json()["voice_handle"]
    assert first == second == "_auto_seed"

    assert service.delete("/v2/voices/seed").json()["cache_removed"] is True
    assert "_auto_seed" not in service.get("/v2/voices").json()["handles"]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def test_chunks_come_back_in_order_and_in_one_bundle(service):
    res = render(service, chunks=["หนึ่ง", "สองสองสอง", "สามสามสามสามสาม"])
    assert res.status_code == 200
    chunks, sr = unpack(res)
    assert len(chunks) == 3 and sr == 48000
    # The stub's duration tracks text length, which is how order is checked here.
    assert [len(c) for c in chunks] == sorted(len(c) for c in chunks)


def test_every_chunk_of_a_job_shares_one_voice(service):
    """The whole reason a run of chunks is one job: a per-chunk voice drifts."""
    handle = service.post("/v2/voices/resolve", json={"speaker_id": "alice"}).json()["voice_handle"]
    chunks, sr = unpack(render(service, chunks=["ก", "ข", "ค"], voice={"handle": handle}))
    assert pitch(chunks[0], sr) == pytest.approx(pitch(chunks[2], sr), abs=1.0)


def test_two_speakers_are_actually_different(service):
    a, _ = unpack(render(service, voice={"speaker_id": "alice"}))
    b, _ = unpack(render(service, voice={"speaker_id": "bob"}))
    assert pitch(a[0]) != pytest.approx(pitch(b[0]), abs=1.0)


def test_a_stale_handle_fails_the_job_rather_than_the_service(service):
    res = render(service, voice={"handle": "vh_long_gone"})
    assert res.status_code == 500
    assert "unknown voice" in res.json()["error"]
    assert service.get("/health").json()["status"] == "ok", "the service kept serving"


def test_files_mode_writes_under_the_work_root_with_the_names_asked_for(service):
    res = render(
        service,
        chunks=["หนึ่ง", "สอง"],
        voice={"speaker_id": "alice"},
        output={"mode": "files", "job_dir": "job42", "names": ["job42_000", "job42_001"]},
    )
    assert res.status_code == 200
    result = res.json()["result"]
    assert [os.path.basename(p) for p in result["files"]] == ["job42_000.wav", "job42_001.wav"]
    assert (service.work_dir / "job42" / "job42_000.wav").exists()


@pytest.mark.parametrize("job_dir", ["../escape", "a/b", "..", "C:/windows/temp"])
def test_job_dir_must_be_a_plain_name(service, job_dir):
    """`job_dir` names a folder; it must never be able to choose a location.

    Rejected rather than trimmed to its last component: quietly writing somewhere
    other than where the caller asked leaves it hunting for files that never arrive.
    """
    res = render(service, voice={"speaker_id": "alice"},
                 output={"mode": "files", "job_dir": job_dir})
    assert res.status_code == 500
    assert "job_dir" in res.json()["error"]


def test_files_mode_without_a_job_dir_uses_the_job_id(service):
    res = render(service, voice={"speaker_id": "alice"}, output={"mode": "files"})
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    assert (service.work_dir / job_id).is_dir()


@pytest.mark.parametrize("body,reason", [
    ({"chunks": []}, "chunks is empty"),
    ({"chunks": ["  "]}, "chunks is empty"),
    ({"chunks": ["x"], "output": {"mode": "flac"}}, "output mode"),
    ({"chunks": ["x"], "lane": "asap"}, "lane"),
])
def test_bad_requests_are_rejected_before_they_are_queued(service, body, reason):
    res = service.post("/v2/jobs/render", json=body)
    assert res.status_code == 400
    assert reason.split()[0] in res.json()["error"]


def test_result_is_delivered_once(service):
    job = service.post("/v2/jobs/render", json={"chunks": ["ทดสอบ"]}).json()
    # No ?wait, so poll to completion.
    for _ in range(200):
        state = service.get(f"/v2/jobs/{job['job_id']}").json()
        if state["status"] == "completed":
            break
    assert service.get(f"/v2/jobs/{job['job_id']}/result").status_code == 200
    assert service.get(f"/v2/jobs/{job['job_id']}/result").status_code == 410


def test_result_before_completion_is_a_409(service):
    job = service.post("/v2/jobs/render", json={"chunks": ["x"], "voice": {"handle": "gone"}}).json()
    for _ in range(200):
        if service.get(f"/v2/jobs/{job['job_id']}").json()["status"] == "failed":
            break
    res = service.get(f"/v2/jobs/{job['job_id']}/result")
    assert res.status_code == 409 and res.json()["status"] == "failed"


def test_unknown_job_is_a_404(service):
    assert service.get("/v2/jobs/g_nope").status_code == 404
    assert service.delete("/v2/jobs/g_nope").status_code == 404


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #

def test_each_job_sets_its_own_lora_scale(service):
    """Scale is global state on a shared model, so it has to travel per job."""
    render(service, lora={"lm": 2.0, "dit": 0.0})
    assert service.get("/health").json()["lora_now"] == [2.0, 0.0]

    render(service, lora="shipped")
    assert service.get("/health").json()["lora_now"] == [2.0, 2.0]

    render(service, lora="off")
    assert service.get("/health").json()["lora_now"] == [0.0, 0.0]


def test_a_job_that_names_no_scale_gets_the_shipped_one(service):
    """What the webhook path has always run at, so it stays the default."""
    render(service, lora="off")
    render(service)
    assert service.get("/health").json()["lora_now"] == [2.0, 2.0]


def test_an_unknown_mode_falls_back_rather_than_failing_the_job(service):
    assert render(service, lora="enthusiastic").status_code == 200
    assert service.get("/health").json()["lora_now"] == [2.0, 2.0]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def test_it_reads_dotenv_from_the_working_directory(tmp_path, monkeypatch):
    """Production keeps the adapter and reference directories in `.env`. Those are
    this service's settings now, so it has to read that file — the webhook used to,
    and nothing else would have."""
    from src.env_file import load_env_file

    (tmp_path / ".env").write_text(
        "# a comment\n"
        "SIANGTTS_REF_DIR=C:\temp\tts_jobs\voices\n"
        "SIANGTTS_ADAPTER='checkpoints/siangtts-v1'\n"
        "\n"
        "MALFORMED\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIANGTTS_REF_DIR", raising=False)
    monkeypatch.delenv("SIANGTTS_ADAPTER", raising=False)

    assert load_env_file() == 2
    assert os.environ["SIANGTTS_REF_DIR"] == "C:\temp\tts_jobs\voices"
    assert os.environ["SIANGTTS_ADAPTER"] == "checkpoints/siangtts-v1", "quotes stripped"


def test_a_real_environment_variable_beats_dotenv(tmp_path, monkeypatch):
    """`SIANGTTS_GPU_STUB=1 uvicorn …` must win over whatever the file says."""
    from src.env_file import load_env_file

    (tmp_path / ".env").write_text("SIANGTTS_ADAPTER=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIANGTTS_ADAPTER", "from-command-line")

    load_env_file()
    assert os.environ["SIANGTTS_ADAPTER"] == "from-command-line"


def test_a_missing_dotenv_is_not_an_error(tmp_path, monkeypatch):
    from src.env_file import load_env_file

    monkeypatch.chdir(tmp_path)
    assert load_env_file() == 0


def test_several_reference_directories_can_be_configured(tmp_path, monkeypatch):
    """The two pipelines arrived with their own ref/ folders and the service answers
    for both — listing only one made the other's voices vanish from every picker."""
    import importlib

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_wav(a / "from_webhook.wav")
    _write_wav(b / "from_studio.wav")

    monkeypatch.setenv("SIANGTTS_GPU_STUB", "1")
    monkeypatch.setenv("SIANGTTS_STUB_DELAY", "0")
    monkeypatch.setenv("SIANGTTS_REF_DIR", os.pathsep.join([str(a), str(b)]))
    monkeypatch.setenv("SIANGTTS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SIANGTTS_WORK_DIR", str(tmp_path / "work"))

    import src.gpu_service as gpu_service

    importlib.reload(gpu_service)
    with TestClient(gpu_service.app) as client:
        ids = [v["id"] for v in client.get("/v2/voices").json()["voices"]]
        assert {"from_webhook", "from_studio"} <= set(ids)
        # Both are usable, not just listed.
        for sid in ("from_webhook", "from_studio"):
            assert client.post("/v2/voices/resolve", json={"speaker_id": sid}).status_code == 200
        # New clips land in the first directory.
        client.post(
            "/v2/voices",
            data={"speaker_id": "fresh", "save_as_speaker": "true"},
            files={"clip": ("fresh.wav", (a / "from_webhook.wav").read_bytes(), "audio/wav")},
        )
        assert (a / "fresh.wav").exists()


def test_a_directory_listed_twice_is_not_listed_twice(tmp_path, monkeypatch):
    import importlib

    ref = tmp_path / "ref"
    ref.mkdir()
    _write_wav(ref / "solo.wav")

    monkeypatch.setenv("SIANGTTS_GPU_STUB", "1")
    monkeypatch.setenv("SIANGTTS_REF_DIR", os.pathsep.join([str(ref), str(ref)]))
    monkeypatch.setenv("SIANGTTS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SIANGTTS_WORK_DIR", str(tmp_path / "work"))

    import src.gpu_service as gpu_service

    importlib.reload(gpu_service)
    with TestClient(gpu_service.app) as client:
        ids = [v["id"] for v in client.get("/v2/voices").json()["voices"]]
        assert ids.count("solo") == 1
        audio_res = client.get("/v2/voices/solo/audio")
        assert audio_res.status_code == 200
        assert "audio/" in audio_res.headers.get("content-type", "")
        assert client.get("/v2/voices/non_existent/audio").status_code == 404
