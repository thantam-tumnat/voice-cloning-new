import pytest
from app.models import Tone, Segment
from app.validator import validate_and_build_segments, ValidationError


def test_validator_invalid_tone_becomes_neutral():
    """Tone outside enum defaults to neutral."""
    original = "สวัสดีครับ วันนี้อากาศดี"
    clauses = ["สวัสดีครับ ", "วันนี้อากาศดี"]
    raw_labels = [
        {"i": 0, "tone": "invalid_alien_tone", "intensity": 2},
        {"i": 1, "tone": "happy", "intensity": 2}
    ]
    segments = validate_and_build_segments(original, clauses, raw_labels)
    assert len(segments) == 2
    assert segments[0].tone == Tone.NEUTRAL
    assert segments[1].tone == Tone.HAPPY


@pytest.mark.parametrize("bad_intensity", [0, 5, -1, 100, "invalid_str", None])
def test_validator_invalid_intensity_becomes_2(bad_intensity):
    """Intensity outside {1, 2, 3} defaults to 2."""
    original = "สวัสดีครับ"
    clauses = ["สวัสดีครับ"]
    raw_labels = [{"i": 0, "tone": "happy", "intensity": bad_intensity}]
    segments = validate_and_build_segments(original, clauses, raw_labels)
    assert len(segments) == 1
    assert segments[0].intensity == 2


def test_validator_missing_index_raises_validation_error():
    """Missing clause index must raise ValidationError to trigger escalation."""
    original = "หนึ่ง สอง สาม"
    clauses = ["หนึ่ง ", "สอง ", "สาม"]
    raw_labels = [
        {"i": 0, "tone": "sad", "intensity": 2},
        {"i": 2, "tone": "sad", "intensity": 2}
        # missing i=1
    ]
    with pytest.raises(ValidationError):
        validate_and_build_segments(original, clauses, raw_labels)


def test_validator_extra_or_out_of_bounds_index_raises_validation_error():
    """Out of bound index must raise ValidationError."""
    original = "หนึ่ง สอง"
    clauses = ["หนึ่ง ", "สอง"]
    raw_labels = [
        {"i": 0, "tone": "sad", "intensity": 2},
        {"i": 1, "tone": "sad", "intensity": 2},
        {"i": 5, "tone": "angry", "intensity": 2}
    ]
    with pytest.raises(ValidationError):
        validate_and_build_segments(original, clauses, raw_labels)


def test_validator_merges_identical_consecutive_tones():
    """Consecutive clauses with same tone are merged with max intensity."""
    original = "ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย"
    clauses = ["ขอโทษนะ ", "ฉันไม่ได้ตั้งใจ ", "แต่เธอก็ไม่ฟังฉันเลย"]
    raw_labels = [
        {"i": 0, "tone": "sad", "intensity": 1},
        {"i": 1, "tone": "sad", "intensity": 3},
        {"i": 2, "tone": "angry", "intensity": 2}
    ]
    segments = validate_and_build_segments(original, clauses, raw_labels)
    assert len(segments) == 2
    assert segments[0].text == "ขอโทษนะ ฉันไม่ได้ตั้งใจ "
    assert segments[0].tone == Tone.SAD
    assert segments[0].intensity == 3  # max of (1, 3)
    assert segments[1].text == "แต่เธอก็ไม่ฟังฉันเลย"
    assert segments[1].tone == Tone.ANGRY
    assert segments[1].intensity == 2


def test_validator_max_segments_cap():
    """Ensure segment count does not exceed max_segments."""
    clauses = [f"คำที่ {i} " for i in range(30)]
    original = "".join(clauses)
    raw_labels = [
        {"i": i, "tone": "sad" if i % 2 == 0 else "happy", "intensity": 2}
        for i in range(30)
    ]
    segments = validate_and_build_segments(original, clauses, raw_labels, max_segments=10)
    assert len(segments) <= 10
    assert "".join(s.text for s in segments) == original


def test_is_safe_spoken_text():
    from app.validator import is_safe_spoken_text

    # Safe punctuation additions
    assert is_safe_spoken_text("สวัสดีครับ", "สวัสดีครับ...")
    assert is_safe_spoken_text("สวัสดีครับ ", "สวัสดีครับ!")
    assert is_safe_spoken_text("ทำไมล่ะ", "ทำไมล่ะ?!")
    assert is_safe_spoken_text("ไม่นะ", "ไม่นะ—")
    assert is_safe_spoken_text("จริงเหรอ", "จริง...เหรอ?")

    # Unsafe alterations (words changed, added, or deleted)
    assert not is_safe_spoken_text("สวัสดีครับ", "สวัสดีจ้า!")
    assert not is_safe_spoken_text("สวัสดีครับ", "สวัสดี")
    assert not is_safe_spoken_text("สวัสดีครับ", "สวัสดีครับผม")
    assert not is_safe_spoken_text("สวัสดีครับ", "")
    assert not is_safe_spoken_text("สวัสดีครับ", None)


def test_validator_accepts_safe_spoken_text_and_merges():
    original = "ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย"
    clauses = ["ขอโทษนะ ", "ฉันไม่ได้ตั้งใจ ", "แต่เธอก็ไม่ฟังฉันเลย"]
    raw_labels = [
        {"i": 0, "tone": "sad", "intensity": 2, "spoken_text": "ขอโทษนะ... "},
        {"i": 1, "tone": "sad", "intensity": 2, "spoken_text": "ฉันไม่ได้ตั้งใจ... "},
        {"i": 2, "tone": "angry", "intensity": 2, "spoken_text": "แต่เธอก็ไม่ฟังฉันเลย!"}
    ]
    segments = validate_and_build_segments(original, clauses, raw_labels)
    assert len(segments) == 2
    assert segments[0].text == "ขอโทษนะ ฉันไม่ได้ตั้งใจ "
    assert segments[0].spoken_text == "ขอโทษนะ... ฉันไม่ได้ตั้งใจ... "
    assert segments[1].text == "แต่เธอก็ไม่ฟังฉันเลย"
    assert segments[1].spoken_text == "แต่เธอก็ไม่ฟังฉันเลย!"


def test_validator_rejects_altered_spoken_text_safely():
    original = "สวัสดีครับ วันนี้อากาศดี"
    clauses = ["สวัสดีครับ ", "วันนี้อากาศดี"]
    raw_labels = [
        {"i": 0, "tone": "happy", "intensity": 2, "spoken_text": "สวัสดีจ้าเพื่อนๆ!"},  # Altered!
        {"i": 1, "tone": "happy", "intensity": 2, "spoken_text": "วันนี้อากาศดี!"}       # Safe
    ]
    segments = validate_and_build_segments(original, clauses, raw_labels)
    assert len(segments) == 1  # Merged (both happy)
    assert segments[0].text == "สวัสดีครับ วันนี้อากาศดี"
    # The first clause's spoken_text was rejected and fell back to original clause text "สวัสดีครับ "
    assert segments[0].spoken_text == "สวัสดีครับ วันนี้อากาศดี!"

