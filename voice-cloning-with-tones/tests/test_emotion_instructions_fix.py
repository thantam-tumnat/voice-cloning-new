import sys
from pathlib import Path
import pytest
from app.services.queue_client import QueueSynthesizer

# Ensure voice-cloning is discoverable for cross-repo tests
vc_path = Path(__file__).resolve().parent.parent.parent / "voice-cloning"
if vc_path.exists() and str(vc_path) not in sys.path:
    sys.path.insert(0, str(vc_path))

try:
    from src.voices import VoiceStore
except ImportError:
    VoiceStore = None


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, log, answers):
        self.log = log
        self.answers = answers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, **kw):
        self.log.append({"verb": "POST", "url": url, **kw})
        for pattern, resp in self.answers:
            if pattern in url:
                return resp
        return FakeResponse(404, payload={"error": "not found"})


def test_queue_client_resolve_disables_sidecar(monkeypatch):
    log = []
    answers = [("/v2/voices/resolve", FakeResponse(200, payload={"voice_handle": "vh_ref_only"}))]
    synth = QueueSynthesizer("http://gpu.test")
    monkeypatch.setattr(synth, "_client", lambda timeout=None: FakeClient(log, answers))

    handle = synth.resolve_speaker("thai_female")
    assert handle == "vh_ref_only"
    assert len(log) == 1
    assert log[0]["json"]["allow_sidecar"] is False
    assert log[0]["json"]["speaker_id"] == "thai_female"


def test_voices_resolve_with_allow_sidecar_false_ignores_txt(tmp_path):
    if VoiceStore is None:
        pytest.skip("VoiceStore from voice-cloning not available")
    ref_dir = tmp_path / "ref"
    cache_dir = tmp_path / "voices"
    ref_dir.mkdir()
    cache_dir.mkdir()

    audio = ref_dir / "speaker_a.wav"
    audio.write_bytes(b"RIFFmockWAVE")
    sidecar = ref_dir / "speaker_a.txt"
    sidecar.write_text("ข้อความถอดเสียง", encoding="utf-8")

    class FakeSynth:
        def build_voice(self, ref_audio, prompt_text=None):
            return {"mode": "ref_continuation" if prompt_text else "reference", "prompt_text": prompt_text}

        def save_voice(self, cache, path):
            pass

        def load_voice(self, path):
            return {"mode": "reference"}

    store = VoiceStore(FakeSynth(), cache_dir, [ref_dir])

    # Resolving with allow_sidecar=False must NOT use sidecar transcript
    handle_no_sidecar = store.resolve_speaker("speaker_a", allow_sidecar=False)
    assert handle_no_sidecar == "speaker_a-da39a3ee"  # digest of empty string
    cache = store.get(handle_no_sidecar)
    assert cache["mode"] == "reference"
    assert cache["prompt_text"] is None

    # Resolving with allow_sidecar=True uses sidecar
    handle_with_sidecar = store.resolve_speaker("speaker_a", allow_sidecar=True)
    assert handle_with_sidecar != "speaker_a-da39a3ee"
    cache_sidecar = store.get(handle_with_sidecar)
    assert cache_sidecar["mode"] == "ref_continuation"
    assert cache_sidecar["prompt_text"] == "ข้อความถอดเสียง"
