"""Ultimate cloning silently disables style tags; these pin the guards on it."""
import pytest

from app.services import siangtts_service as svc


@pytest.fixture
def ref_with_transcript(tmp_path, monkeypatch):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    wav = ref_dir / "narrator.wav"
    wav.write_bytes(b"")
    (ref_dir / "narrator.txt").write_text("ทดสอบเสียงต้นแบบ", encoding="utf-8")
    monkeypatch.setattr(svc.siangtts_service, "ref_dir", ref_dir, raising=False)
    return wav


def test_transcript_is_ignored_by_default(ref_with_transcript, monkeypatch):
    monkeypatch.setattr(svc.settings, "siangtts_hifi_cloning", False, raising=False)
    assert svc.siangtts_service._transcript_for(ref_with_transcript) is None


def test_transcript_is_used_when_hifi_is_opted_into(ref_with_transcript, monkeypatch):
    monkeypatch.setattr(svc.settings, "siangtts_hifi_cloning", True, raising=False)
    assert svc.siangtts_service._transcript_for(ref_with_transcript) == "ทดสอบเสียงต้นแบบ"


@pytest.mark.parametrize("mode", ["continuation", "ref_continuation"])
def test_a_styled_chunk_against_a_hifi_voice_warns(capsys, mode):
    svc.siangtts_service._hifi_warned = False
    svc.siangtts_service._warn_if_instructions_are_dead(
        {"mode": mode}, [(0, "(Crying voice, trembling)สวัสดี", False)]
    )
    assert "will not be heard" in capsys.readouterr().err


def test_the_warning_is_said_once_not_per_chunk(capsys):
    svc.siangtts_service._hifi_warned = False
    planned = [(0, "(Crying voice, trembling)สวัสดี", False)]
    svc.siangtts_service._warn_if_instructions_are_dead({"mode": "continuation"}, planned)
    capsys.readouterr()
    svc.siangtts_service._warn_if_instructions_are_dead({"mode": "continuation"}, planned)
    assert capsys.readouterr().err == ""


def test_reference_mode_honours_instructions_so_stays_quiet(capsys):
    svc.siangtts_service._hifi_warned = False
    svc.siangtts_service._warn_if_instructions_are_dead(
        {"mode": "reference"}, [(0, "(Crying voice, trembling)สวัสดี", False)]
    )
    assert capsys.readouterr().err == ""


def test_untagged_text_against_a_hifi_voice_stays_quiet(capsys):
    """Nothing was asked for, so nothing is being lost."""
    svc.siangtts_service._hifi_warned = False
    svc.siangtts_service._warn_if_instructions_are_dead(
        {"mode": "continuation"}, [(0, "สวัสดีครับ", False)]
    )
    assert capsys.readouterr().err == ""
