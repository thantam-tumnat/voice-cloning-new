import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services import siangtts_service as svc


@pytest.fixture
def client():
    return TestClient(app)


def test_model_load_failure_raises_instead_of_beeping():
    """A failed load must surface as an error, not a 440Hz tone.

    This was the original bug: get_synthesizer() swallowed every exception and
    substituted the mock, so a model that never loaded was indistinguishable from
    a model producing artifacts.
    """
    service = svc.SiangTTSService()
    service._synthesizer = None

    def boom(*args, **kwargs):
        raise RuntimeError("The paging file is too small for this operation to complete.")

    svc.settings.siangtts_allow_mock = False
    original = svc._RealSynthesizer.__init__
    svc._RealSynthesizer.__init__ = boom
    try:
        with pytest.raises(svc.SynthesizerUnavailable) as exc:
            service.get_synthesizer()
        assert "paging file" in str(exc.value)
        assert service.status["loaded"] is False
    finally:
        svc._RealSynthesizer.__init__ = original


def test_mock_fallback_only_when_explicitly_enabled():
    service = svc.SiangTTSService()
    service._synthesizer = None

    def boom(*args, **kwargs):
        raise RuntimeError("no GPU")

    svc.settings.siangtts_allow_mock = True
    original = svc._RealSynthesizer.__init__
    svc._RealSynthesizer.__init__ = boom
    try:
        synth = service.get_synthesizer()
        assert isinstance(synth, svc._MockSynthesizer)
        assert service.status["using_mock"] is True
        assert "no GPU" in service.status["load_error"]
    finally:
        svc._RealSynthesizer.__init__ = original


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "speakers_count" in data


def test_speakers_crud(client):
    # 1. List speakers initially
    res = client.get("/speakers")
    assert res.status_code == 200
    initial_count = len(res.json()["speakers"])

    # 2. Register a new speaker via upload
    dummy_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    files = {"file": ("test_speaker_alpha.wav", io.BytesIO(dummy_wav), "audio/wav")}
    data = {"speaker_id": "test_speaker_alpha"}
    
    upload_res = client.post("/speakers", files=files, data=data)
    assert upload_res.status_code == 200
    speaker_data = upload_res.json()
    assert speaker_data["id"] == "test_speaker_alpha"

    # 3. List speakers again
    list_res = client.get("/speakers")
    assert list_res.status_code == 200
    ids = [s["id"] for s in list_res.json()["speakers"]]
    assert "test_speaker_alpha" in ids

    # 4. Stream/get reference audio for speaker
    audio_res = client.get("/speakers/test_speaker_alpha/audio")
    assert audio_res.status_code == 200
    assert "audio/" in audio_res.headers.get("content-type", "")
    assert audio_res.content == dummy_wav

    # 4.1 Non-existent speaker returns 404
    audio_404_res = client.get("/speakers/non_existent_speaker_xyz/audio")
    assert audio_404_res.status_code == 404

    # 5. Delete speaker
    del_res = client.delete("/speakers/test_speaker_alpha")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True


def test_synthesize_endpoint_json(client):
    res = client.post("/synthesize", json={
        "text": "หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ",
        "guidance": "สงบ นุ่มนวล",
        "engine": "voxcpm",
        "cfg_value": 2.5,
        "inference_timesteps": 10,
        "auto_annotate": True,
        "lora_mode": "on",
    })
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert "lora_on.wav" in res.headers.get("content-disposition", "")
    assert len(res.content) > 0


def test_synthesize_endpoint_lora_off(client):
    res = client.post("/synthesize", json={
        "text": "Hello this is a test without lora",
        "engine": "voxcpm",
        "cfg_value": 2.5,
        "inference_timesteps": 10,
        "auto_annotate": False,
        "lora_mode": "off",
    })
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert "lora_off.wav" in res.headers.get("content-disposition", "")
    assert len(res.content) > 0


def test_synthesize_endpoint_with_upload(client):
    dummy_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    files = {"file": ("my_sample.wav", io.BytesIO(dummy_wav), "audio/wav")}
    data = {
        "text": "[calm] หายใจเข้าลึกๆ ผ่อนคลาย",
        "cfg_value": "2.5",
        "inference_timesteps": "10",
        "auto_annotate": "false",
        "lora_mode": "off",
    }
    res = client.post("/synthesize/upload", files=files, data=data)
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert "lora_off.wav" in res.headers.get("content-disposition", "")
    assert len(res.content) > 0


def test_speak_endpoint_voxcpm(client):
    res = client.post("/speak", json={
        "text": "หายใจเข้าลึกๆ ผ่อนคลาย",
        "guidance": "สงบ นุ่มนวล",
        "engine": "voxcpm"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["engine"] == "voxcpm"
    assert len(data["segments"]) > 0
    assert "(Calm" in data["text"] or "หายใจเข้าลึกๆ" in data["text"]


# ---------------------------------------------------------------------------
# Auto voice consistency across chunks
# ---------------------------------------------------------------------------

def _spy_prompt_caches(monkeypatch):
    """Record the prompt_cache handed to each chunk of a synthesize_many run."""
    service = svc.siangtts_service
    synth = service.get_synthesizer()
    seen = []
    original = synth.synth

    def spy(text, **kwargs):
        seen.append(kwargs.get("prompt_cache"))
        return original(text, **kwargs)

    monkeypatch.setattr(synth, "synth", spy)
    return service, seen


def test_auto_consistency_clones_first_chunk_when_no_speaker(monkeypatch):
    """Unpinned voice: chunks after the first reuse the first chunk's timbre.

    Without this, VoxCPM2 resamples the speaker per call and a single utterance
    changes voice mid-sentence.
    """
    monkeypatch.setattr(svc.settings, "siangtts_auto_voice_consistency", True, raising=False)
    service, seen = _spy_prompt_caches(monkeypatch)

    service.synthesize_many(["(happy)หนึ่ง", "(sad)สอง", "(calm)สาม"])

    assert seen[0] is None, "first chunk has nothing to clone from yet"
    assert seen[1] is not None, "second chunk should reuse the cloned voice"
    assert seen[1] == seen[2], "every later chunk shares one voice"


def test_auto_consistency_can_be_disabled(monkeypatch):
    monkeypatch.setattr(svc.settings, "siangtts_auto_voice_consistency", False, raising=False)
    service, seen = _spy_prompt_caches(monkeypatch)

    service.synthesize_many(["(happy)หนึ่ง", "(sad)สอง"])

    assert seen == [None, None]


def _fresh_seed(monkeypatch, service):
    """Force the seed to be minted in this test rather than inherited from another."""
    monkeypatch.setattr(service, "_seed_voice", None, raising=False)
    monkeypatch.setattr(service, "_seed_voice_failed", False, raising=False)


def test_single_chunk_still_gets_the_shared_seed(monkeypatch):
    """A one-chunk request must be the same speaker as every other request.

    This used to be skipped on the grounds that one chunk cannot drift against
    itself and so should not pay for a clone. It drifts against *other requests*
    instead: unpinned, VoxCPM2 samples a fresh speaker per call, so the benchmark
    -- which sends exactly one chunk per take -- rendered every take as a
    different person (median F0 141.6 / 152.9 / 202.1 Hz on identical text).

    The cost argument only ever applied to the first request: the seed is minted
    once and then served from memory and voice_cache/_auto_seed.pt.
    """
    monkeypatch.setattr(svc.settings, "siangtts_auto_voice_consistency", True, raising=False)
    service, seen = _spy_prompt_caches(monkeypatch)
    _fresh_seed(monkeypatch, service)

    service.synthesize_many(["(happy)หนึ่งเดียว"])

    # Minting the seed is itself a generation, and it is the unpinned one; the
    # chunk the caller asked for is the last and must carry the seed.
    assert seen[-1] is not None, "a lone chunk must still be pinned to the seed voice"


def test_seed_is_minted_once_and_reused(monkeypatch):
    """The second single-chunk request must not pay to build the seed again."""
    monkeypatch.setattr(svc.settings, "siangtts_auto_voice_consistency", True, raising=False)
    service, seen = _spy_prompt_caches(monkeypatch)
    _fresh_seed(monkeypatch, service)

    service.synthesize_many(["(happy)หนึ่ง"])
    first_count = len(seen)
    service.synthesize_many(["(sad)สอง"])

    assert len(seen) - first_count == 1, "reusing the cached seed costs one generation"
    assert seen[-1] == seen[first_count - 1], "both requests are the same speaker"


def test_single_chunk_seed_matches_multi_chunk_seed(monkeypatch):
    """One take and a multi-chunk run have to land on the same speaker."""
    monkeypatch.setattr(svc.settings, "siangtts_auto_voice_consistency", True, raising=False)
    service, seen = _spy_prompt_caches(monkeypatch)
    _fresh_seed(monkeypatch, service)

    service.synthesize_many(["(happy)หนึ่งเดียว"])
    lone = seen[-1]
    service.synthesize_many(["(happy)หนึ่ง", "(sad)สอง"])
    multi = seen[-2:]

    assert lone is not None
    assert all(c == lone for c in multi), "every chunk shares one seed voice"


def test_explicit_speaker_is_not_overridden(monkeypatch):
    """A registered speaker stays the reference for every chunk."""
    monkeypatch.setattr(svc.settings, "siangtts_auto_voice_consistency", True, raising=False)
    service, seen = _spy_prompt_caches(monkeypatch)
    service._voices["pinned"] = "pinned_latent"

    try:
        service.synthesize_many(["(happy)หนึ่ง", "(sad)สอง"], speaker_id="pinned")
    finally:
        service._voices.pop("pinned", None)

    assert seen == ["pinned_latent", "pinned_latent"]


def test_multi_chunk_seeds_one_neutral_voice_for_every_chunk():
    """No chunk may be cloned from another, or it inherits that chunk's emotion.

    Cloning chunk 1 collapsed [sad]->[happy] to a measured -4.9Hz median-F0 change.
    Seeding from a neutral line instead keeps each style tag independent.
    """
    service = svc.SiangTTSService()
    service._synthesizer = svc._MockSynthesizer()

    spoken, caches = [], []

    def record(text, *, ref_audio=None, prompt_cache=None, **kw):
        spoken.append(text)
        caches.append(prompt_cache)
        import numpy as np
        return np.zeros(1000, dtype="float32")

    service._synthesizer.synth = record
    service._synthesizer.build_voice = lambda path, prompt_text=None: "seed_voice"

    svc.settings.siangtts_auto_voice_consistency = True
    service.synthesize_many(["(happy)หนึ่ง", "(sad)สอง"])

    # The seed line is generated first and is never part of the output chunks.
    assert spoken[0] == svc.settings.siangtts_voice_seed_text
    assert spoken[1:] == ["(happy)หนึ่ง", "(sad)สอง"]
    # Both real chunks ride the same neutral seed -- including the first one.
    assert caches[1] == "seed_voice"
    assert caches[2] == "seed_voice"


def test_seed_voice_skipped_when_speaker_pinned():
    """A pinned speaker already fixes the timbre; no seed generation needed."""
    service = svc.SiangTTSService()
    service._synthesizer = svc._MockSynthesizer()
    service._voices["pinned"] = "pinned_latent"

    spoken = []

    def record(text, *, ref_audio=None, prompt_cache=None, **kw):
        spoken.append(text)
        import numpy as np
        return np.zeros(1000, dtype="float32")

    service._synthesizer.synth = record
    svc.settings.siangtts_auto_voice_consistency = True
    try:
        service.synthesize_many(["(happy)หนึ่ง", "(sad)สอง"], speaker_id="pinned")
    finally:
        service._voices.pop("pinned", None)

    assert spoken == ["(happy)หนึ่ง", "(sad)สอง"]


def test_seed_voice_is_built_once_and_reused_across_requests():
    """Two requests must come back in the same voice, and pay for one seed.

    Regenerating the seed per request made every call a slightly different speaker
    and spent an extra generation each time to do it.
    """
    service = svc.SiangTTSService()
    service._synthesizer = svc._MockSynthesizer()

    spoken = []
    built = []

    def record(text, **kw):
        spoken.append(text)
        import numpy as np
        return np.zeros(1000, dtype="float32")

    def build(path, prompt_text=None):
        built.append(path)
        return "seed_voice"

    service._synthesizer.synth = record
    service._synthesizer.build_voice = build

    svc.settings.siangtts_auto_voice_consistency = True
    service.synthesize_many(["(happy)หนึ่ง", "(sad)สอง"])
    service.synthesize_many(["(calm)สาม", "(tired)สี่"])

    assert spoken.count(svc.settings.siangtts_voice_seed_text) == 1
    assert len(built) == 1


def test_get_available_models_endpoint(client, monkeypatch):
    monkeypatch.setattr("app.config.settings.openai_api_key", "test-key")
    res = client.get("/models?refresh=true")
    assert res.status_code == 200
    data = res.json()
    assert "providers" in data
    assert "gemini" in data["providers"]
    assert "anthropic" in data["providers"]
    assert "openai" in data["providers"]
    assert len(data["providers"]["gemini"]["models"]) > 0
    assert len(data["providers"]["openai"]["models"]) > 0
    assert data["providers"]["openai"]["available"] is True


def test_synthesize_endpoint_raw_tts_post_process_false(client, monkeypatch):
    """Verify synthesize endpoint accepts post_process=False for raw TTS without crashing."""
    svc.settings.siangtts_allow_mock = True
    res = client.post(
        "/synthesize",
        json={
            "text": "[sad] ทดสอบเสียงสด [angry] แบบปิด post process",
            "cfg_value": 2.5,
            "inference_timesteps": 4,
            "lora_mode": "on",
            "post_process": False,
        },
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert len(res.content) > 100
