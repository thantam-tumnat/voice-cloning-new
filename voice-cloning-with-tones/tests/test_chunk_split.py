"""Long chunks must be split before VoxCPM2 sees them.

The reported failure: five copies of one 98-character line went in as a single
494-character chunk -- crfcut found no sentence break in it, and every copy carried
the same tone, so the renderer merged them into one -- and the third sentence came
back in a different voice. VoxCPM2 has no internal splitter, and past roughly
140 spoken characters the speaker identity drifts mid-generation.
"""
import pytest

from app.services import siangtts_service as svc

LINE = "ทั้งสองไฟล์มีโครงสร้าง เนื้อหา และผลการตรวจจับทางดิจิทัลที่แทบจะถอดรหัสออกมาเหมือนกันทุกประการครับ"


def _bodies(pieces):
    return [svc._LEADING_STYLE_RE.sub("", p.text) for p in pieces]


# --------------------------------------------------------------------------- #
# split_for_synthesis
# --------------------------------------------------------------------------- #

def test_repeated_lines_split_back_into_one_piece_per_line():
    """The reported case: five identical lines, one generation each."""
    pieces = svc.split_for_synthesis("\n".join([LINE] * 5))

    assert len(pieces) == 5
    assert _bodies(pieces) == [LINE] * 5


def test_line_break_seam_earns_the_paragraph_pause():
    """Pieces broken at a newline get the long gap a hand-written break gets.

    The first piece keeps whatever the caller said about the chunk's own lead-in;
    only the seams this function introduced are marked.
    """
    pieces = svc.split_for_synthesis("\n".join([LINE] * 3))

    assert [p.paragraph_seam for p in pieces] == [False, True, True]


def test_short_text_is_left_alone():
    pieces = svc.split_for_synthesis("สวัสดีครับ")

    assert len(pieces) == 1
    assert pieces[0].text == "สวัสดีครับ"
    assert pieces[0].paragraph_seam is False


def test_blank_text_yields_nothing():
    assert svc.split_for_synthesis("   \n  ") == []


def test_every_piece_stays_within_the_budget():
    long_run = " ".join(["เขาเดินจากไปโดยไม่หันกลับมามองอีกเลยแม้แต่ครั้งเดียว"] * 6)
    pieces = svc.split_for_synthesis(long_run, limit=140)

    assert len(pieces) > 1
    assert all(svc.spoken_len(p.text) <= 140 for p in pieces)
    # Nothing may be dropped on the floor.
    assert "".join(_bodies(pieces)).replace(" ", "") == long_run.replace(" ", "")


def test_style_instruction_rides_every_piece():
    """VoxCPM2 honours a parenthetical only at position 0.

    A piece that lost it would fall back to neutral partway through the emotion,
    which is the same class of bug as speaking the tag aloud.
    """
    instruction = "(Sad and melancholic voice, slight sighs)"
    pieces = svc.split_for_synthesis(instruction + " ".join([LINE] * 4))

    assert len(pieces) > 1
    assert all(p.text.startswith(instruction) for p in pieces)
    assert all(instruction not in svc._LEADING_STYLE_RE.sub("", p.text) for p in pieces)


def test_text_with_no_seam_at_all_still_gets_cut():
    """Thai with no spaces, newlines or punctuation still has word boundaries."""
    pieces = svc.split_for_synthesis("ก" * 400, limit=140)

    assert len(pieces) >= 3
    assert all(svc.spoken_len(p.text) <= 140 for p in pieces)


def test_splitting_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(svc.settings, "siangtts_max_chunk_chars", 0, raising=False)

    pieces = svc.split_for_synthesis("\n".join([LINE] * 5), limit=0)

    assert len(pieces) == 5, "limit=0 falls back to the module default, not to no split"


# --------------------------------------------------------------------------- #
# render_chunks wiring
# --------------------------------------------------------------------------- #

def _spy(monkeypatch):
    """Record (text, prompt_cache) for every generation of a run."""
    service = svc.siangtts_service
    synth = service.get_synthesizer()
    seen = []
    original = synth.synth

    def spy(text, **kwargs):
        seen.append((text, kwargs.get("prompt_cache")))
        return original(text, **kwargs)

    monkeypatch.setattr(synth, "synth", spy)
    return service, seen


def test_long_chunk_becomes_several_generations(monkeypatch):
    monkeypatch.setattr(svc.settings, "siangtts_auto_voice_consistency", False, raising=False)
    service, seen = _spy(monkeypatch)

    rendered, _ = service.render_chunks(["\n".join([LINE] * 5)], tones=["neutral"])

    assert len(rendered) == 5, "one 494-char generation is what drifted"
    assert [t for t, _ in seen] == [LINE] * 5


def test_split_pieces_keep_the_tone_of_their_source_chunk(monkeypatch):
    monkeypatch.setattr(svc.settings, "siangtts_auto_voice_consistency", False, raising=False)
    service, _ = _spy(monkeypatch)

    rendered, _ = service.render_chunks(
        ["\n".join([LINE] * 4), "สั้นๆ"],
        tones=["sad", "happy"],
        breaks=[True, False],
    )

    assert [c.tone for c in rendered] == ["sad"] * 4 + ["happy"]
    # The caller's own break applies to the first piece; the rest carry the
    # paragraph seams the split introduced.
    assert [c.break_before for c in rendered] == [True, True, True, True, False]
    assert all(c.text_len > 0 for c in rendered)


def test_split_pieces_all_share_one_voice(monkeypatch):
    """The whole point: five pieces, one speaker.

    Before, this text was a single chunk, so ``len(chunks) > 1`` was false and the
    seed voice was skipped -- leaving the longest, most drift-prone generation the
    only one with no voice anchor at all.
    """
    monkeypatch.setattr(svc.settings, "siangtts_auto_voice_consistency", True, raising=False)
    service, seen = _spy(monkeypatch)
    service._synthesizer.build_voice = lambda path, prompt_text=None: "seed_voice"

    service.render_chunks(["\n".join([LINE] * 5)])

    spoken = [t for t, _ in seen]
    caches = [c for _, c in seen]

    # The seed line is generated first and never reaches the output.
    assert spoken[0] == svc.settings.siangtts_voice_seed_text
    assert spoken[1:] == [LINE] * 5
    assert caches[1:] == ["seed_voice"] * 5


def test_pinned_speaker_conditions_every_piece(monkeypatch):
    service, seen = _spy(monkeypatch)
    service._voices["pinned"] = "pinned_latent"

    try:
        service.render_chunks(["\n".join([LINE] * 3)], speaker_id="pinned")
    finally:
        service._voices.pop("pinned", None)

    assert [c for _, c in seen] == ["pinned_latent"] * 3


def test_uploaded_reference_is_encoded_once_for_the_whole_run(monkeypatch):
    """Re-encoding the same clip per piece is wasted work, not extra safety."""
    service, seen = _spy(monkeypatch)
    calls = []

    def counting_build(path, prompt_text=None):
        calls.append(path)
        return "ref_voice"

    service._synthesizer.build_voice = counting_build

    service.render_chunks(
        ["\n".join([LINE] * 4)],
        ref_audio_bytes=b"RIFF....WAVE",
        ref_filename="upload.wav",
    )

    assert len(calls) == 1, "one encode, shared by every piece"
    assert [c for _, c in seen] == ["ref_voice"] * 4


def test_empty_input_still_rejected():
    with pytest.raises(ValueError):
        svc.siangtts_service.render_chunks(["", "   "])
