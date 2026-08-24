import pytest

from app.services import siangtts_service as svc


@pytest.fixture(autouse=True)
def mock_synthesizer(monkeypatch, tmp_path):
    """Force the sine-tone mock for tests, against throwaway cache directories.

    The service refuses to fall back to the mock by default -- a silent fallback is
    what made a failed model load sound like a broken model instead of an error. Tests
    opt in explicitly rather than depending on an 8GB model being loadable.

    The cache directory is redirected because the service persists its auto seed
    voice there. Left pointing at the real voice_cache/, a test run dropped a
    mock-built seed into the project, and the next production run loaded it.
    """
    cache_dir = tmp_path / "voice_cache"
    cache_dir.mkdir()

    # No test may reach for the shared GPU service. Left at its default the suite
    # would depend on whether a machine happens to have one running on :8020, and
    # the tests that exercise the in-process load path would never reach it.
    monkeypatch.setattr(svc.settings, "voxcpm_service_url", "", raising=False)
    monkeypatch.setattr(svc.settings, "siangtts_allow_mock", True, raising=False)
    monkeypatch.setattr(svc.settings, "siangtts_cache_dir", str(cache_dir), raising=False)
    monkeypatch.setattr(svc.siangtts_service, "cache_dir", cache_dir, raising=False)
    monkeypatch.setattr(svc.siangtts_service, "_synthesizer", svc._MockSynthesizer())
    monkeypatch.setattr(svc.siangtts_service, "_using_mock", True)
    yield
    svc.siangtts_service._synthesizer = None
    svc.siangtts_service._using_mock = False
    svc.siangtts_service._voices.clear()
    # Built at most once per process, so it has to be cleared between tests or the
    # next test sees a seed it never generated.
    svc.siangtts_service._seed_voice = None
    svc.siangtts_service._seed_voice_failed = False
