import math

import numpy as np
import pytest

from app.services.audio_post import (
    ENERGY_MATCH,
    _match_rate,
    GAP_EMOTION_S,
    GAP_PARAGRAPH_S,
    GAP_SAME_TONE_S,
    MAX_STRETCH,
    OUTPUT_PEAK,
    TONE_ENERGY_DB,
    Chunk,
    apply_gain_db,
    assemble,
    butt_join,
    fade_edges,
    gap_before,
    remove_dc,
    time_stretch,
    trim_silence,
    voiced_rms,
)

SR = 24000


def tone(seconds: float, freq: float = 200.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype="float32")


def dominant_freq(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x.astype("float64") * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / SR)[int(np.argmax(spec))])


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

def test_remove_dc_centres_signal():
    x = tone(0.2) + 0.3
    assert abs(float(np.mean(remove_dc(x)))) < 1e-6


def test_trim_silence_drops_padding_but_keeps_speech():
    padded = np.concatenate([silence(0.5), tone(1.0), silence(0.5)])
    trimmed = trim_silence(padded, SR)
    # 1 s of speech plus the deliberate ~30 ms of padding either side.
    assert 1.0 <= len(trimmed) / SR <= 1.15


def test_trim_silence_leaves_all_silence_alone():
    quiet = silence(0.5)
    assert len(trim_silence(quiet, SR)) == len(quiet)


def test_fade_edges_zeroes_the_endpoints():
    faded = fade_edges(tone(0.5), SR)
    assert abs(float(faded[0])) < 1e-6
    assert abs(float(faded[-1])) < 1e-6
    # The middle is untouched. Check a window, not one sample -- a single sample can
    # land on a zero crossing of the test tone.
    mid = len(faded) // 2
    assert float(np.max(np.abs(faded[mid - 100:mid + 100]))) > 0.4


def test_fade_edges_no_op_on_very_short_audio():
    tiny = tone(0.005)
    assert np.array_equal(fade_edges(tiny, SR), tiny)


def test_apply_gain_db():
    x = tone(0.1)
    assert float(np.max(np.abs(apply_gain_db(x, 6.0)))) == pytest.approx(
        float(np.max(np.abs(x))) * 2, rel=0.01
    )


def test_voiced_rms_ignores_silence():
    speech = tone(1.0)
    padded = np.concatenate([silence(2.0), speech, silence(2.0)])
    # Whole-signal RMS would be dragged down by the padding; voiced_rms must not be.
    assert voiced_rms(padded, SR) == pytest.approx(voiced_rms(speech, SR), rel=0.1)


# --------------------------------------------------------------------------- #
# Time stretch
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ratio", [0.85, 0.95, 1.05, 1.15, 1.30])
def test_time_stretch_hits_the_ratio(ratio):
    x = tone(1.0)
    y = time_stretch(x, SR, ratio)
    assert len(y) / len(x) == pytest.approx(ratio, abs=0.02)


@pytest.mark.parametrize("ratio", [0.85, 1.15, 1.30])
def test_time_stretch_preserves_pitch(ratio):
    """The whole point: rate changes, pitch does not, so cloned identity survives."""
    x = tone(1.0, freq=200.0)
    y = time_stretch(x, SR, ratio)
    assert dominant_freq(y) == pytest.approx(200.0, abs=3.0)


def test_time_stretch_preserves_level():
    x = tone(1.0)
    y = time_stretch(x, SR, 1.15)
    assert float(np.sqrt(np.mean(y.astype("float64") ** 2))) == pytest.approx(
        float(np.sqrt(np.mean(x.astype("float64") ** 2))), rel=0.1
    )


def test_time_stretch_is_a_no_op_near_unity():
    x = tone(0.5)
    assert np.array_equal(time_stretch(x, SR, 1.0), x)


def test_time_stretch_leaves_very_short_audio_alone():
    tiny = tone(0.01)
    assert len(time_stretch(tiny, SR, 1.2)) == len(tiny)


# --------------------------------------------------------------------------- #
# Gap policy
# --------------------------------------------------------------------------- #

def test_first_chunk_gets_no_leading_gap():
    assert gap_before(Chunk(tone(0.1), "sad"), None) == 0.0


def test_gap_policy_by_boundary_kind():
    sad = Chunk(tone(0.1), "sad")
    happy = Chunk(tone(0.1), "happy")
    para = Chunk(tone(0.1), "happy", break_before=True)

    assert gap_before(sad, sad) == GAP_SAME_TONE_S
    assert gap_before(happy, sad) == GAP_EMOTION_S
    # A written line break outranks the tone comparison.
    assert gap_before(para, sad) == GAP_PARAGRAPH_S
    assert gap_before(para, para) == GAP_PARAGRAPH_S


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def test_assemble_empty_input():
    assert assemble([], SR).size == 0
    assert assemble([Chunk(np.zeros(0, dtype="float32"), "sad")], SR).size == 0


def test_assemble_inserts_the_planned_gaps():
    chunks = [
        Chunk(tone(1.0), "sad"),
        Chunk(tone(1.0), "happy"),
        Chunk(tone(1.0), "happy", break_before=True),
    ]
    out = assemble(chunks, SR, match_energy=False, match_rate=False)
    expected = 3.0 + GAP_EMOTION_S + GAP_PARAGRAPH_S
    assert len(out) / SR == pytest.approx(expected, abs=0.05)


def test_assemble_realizes_the_per_tone_energy_offsets():
    """scared should end up audibly louder than sad, by the measured margin."""
    chunks = [Chunk(tone(1.5), "sad"), Chunk(tone(1.5), "scared")]
    out = assemble(chunks, SR, match_rate=False)

    n = int(SR * 1.5)
    sad_rms = float(np.sqrt(np.mean(out[:n].astype("float64") ** 2)))
    scared_rms = float(np.sqrt(np.mean(out[-n:].astype("float64") ** 2)))
    measured = 20 * math.log10(scared_rms / sad_rms)
    wanted = TONE_ENERGY_DB["scared"] - TONE_ENERGY_DB["sad"]
    assert measured == pytest.approx(wanted, abs=1.0)


def test_assemble_pulls_in_chunks_the_model_rendered_at_different_levels():
    """Two same-tone chunks are pulled together, but not flattened onto each other.

    Levelling is partial by design -- see ENERGY_MATCH -- because a chunk's own level
    is now mostly the emotion the model was asked for rather than noise. What still
    has to hold is that a wild difference gets most of the way closed.
    """
    chunks = [Chunk(tone(1.5, amp=0.1), "neutral"), Chunk(tone(1.5, amp=0.6), "neutral")]
    out = assemble(chunks, SR, match_rate=False)

    n = int(SR * 1.5)
    first = float(np.sqrt(np.mean(out[:n].astype("float64") ** 2)))
    last = float(np.sqrt(np.mean(out[-n:].astype("float64") ** 2)))
    rendered_gap = 20 * math.log10(0.6 / 0.1)
    remaining = 20 * math.log10(last / first)
    assert remaining == pytest.approx(rendered_gap * (1 - ENERGY_MATCH), abs=1.0)
    assert remaining < rendered_gap / 2


def test_assemble_paces_chunks_toward_their_duration_ratio():
    """happy is the fastest measured tone and tired the slowest, so happy ends shorter."""
    chunks = [Chunk(tone(2.0), "happy"), Chunk(tone(2.0), "tired")]
    out = assemble(chunks, SR, match_energy=False)
    # Total grows only by the gap; the two halves are now uneven.
    assert len(out) / SR > 4.0 + GAP_EMOTION_S - 0.3


def test_rate_matching_corrects_toward_the_target_not_blindly_toward_the_ratio():
    """A chunk the model already rushed must be slowed down, not sped up further.

    VoxCPM2 rendered "happy" well under its target pace on the reference script.
    Applying the 0.927 target as a raw stretch made it faster still; the correction
    has to start from what was actually rendered.
    """
    # Same text, but happy came back much shorter than tired.
    chunks = [
        Chunk(tone(1.4), "happy", text_len=100),
        Chunk(tone(2.0), "tired", text_len=100),
    ]
    happy, tired = _match_rate(chunks, SR)
    assert len(happy.audio) / SR > 1.4
    # ... without overshooting past the slower tone it is being measured against.
    assert len(happy.audio) < len(tired.audio)


def test_rate_matching_leaves_neutral_where_the_model_put_it():
    """Neutral is the tone every ratio is defined against, so it is the fixed point.

    It used to be normalised by the mean of whatever tones shared the take, which
    rushed it in a slow-toned script and dragged it in a fast-toned one.
    """
    slow = [
        Chunk(tone(2.0), "neutral", text_len=100),
        Chunk(tone(2.0), "calm", text_len=100),
        Chunk(tone(2.0), "tired", text_len=100),
    ]
    fast = [
        Chunk(tone(2.0), "neutral", text_len=100),
        Chunk(tone(2.0), "happy", text_len=100),
        Chunk(tone(2.0), "excited", text_len=100),
    ]
    slow_neutral = len(_match_rate(slow, SR)[0].audio) / SR
    fast_neutral = len(_match_rate(fast, SR)[0].audio) / SR

    assert slow_neutral == pytest.approx(2.0, abs=0.05)
    assert fast_neutral == pytest.approx(2.0, abs=0.05)


def test_rate_matching_does_not_apply_a_single_tone_takes_ratio_twice():
    """An all-calm take is already slow; it must not be slowed 8% again on top."""
    chunks = [Chunk(tone(2.0), "calm", text_len=100) for _ in range(3)]
    for c in _match_rate(chunks, SR):
        assert len(c.audio) / SR == pytest.approx(2.0, abs=0.05)


def test_rate_matching_normalizes_by_text_length():
    """Twice the text in twice the time is the same pace, so nothing should move."""
    chunks = [
        Chunk(tone(2.0), "neutral", text_len=100),
        Chunk(tone(4.0), "neutral", text_len=200),
    ]
    out = assemble(chunks, SR, match_energy=False)
    assert len(out) / SR == pytest.approx(6.0 + GAP_SAME_TONE_S, abs=0.05)


def test_assemble_stretch_stays_within_the_transparency_bound():
    chunks = [Chunk(tone(2.0), "excited"), Chunk(tone(2.0), "calm")]
    out = assemble(chunks, SR, match_energy=False)
    speech = len(out) / SR - GAP_EMOTION_S
    assert speech <= 4.0 * (1 + MAX_STRETCH) + 0.05
    assert speech >= 4.0 * (1 - MAX_STRETCH) - 0.05


def test_assemble_never_clips():
    chunks = [Chunk(tone(1.0, amp=0.95), "scared"), Chunk(tone(1.0, amp=0.95), "angry")]
    out = assemble(chunks, SR)
    assert float(np.max(np.abs(out))) <= OUTPUT_PEAK + 1e-6


def test_assemble_tolerates_unknown_and_missing_tones():
    chunks = [Chunk(tone(0.5), None), Chunk(tone(0.5), "not-a-tone")]
    out = assemble(chunks, SR)
    assert out.size > 0
    assert np.isfinite(out).all()


def test_assemble_single_chunk_is_left_at_its_own_level():
    x = tone(1.0, amp=0.4)
    out = assemble([Chunk(x, "neutral")], SR)
    assert float(np.sqrt(np.mean(out.astype("float64") ** 2))) == pytest.approx(
        float(np.sqrt(np.mean(x.astype("float64") ** 2))), rel=0.05
    )


# --------------------------------------------------------------------------- #
# Legacy join
# --------------------------------------------------------------------------- #

def test_butt_join_matches_the_old_flat_gap_behaviour():
    chunks = [Chunk(tone(1.0), "sad"), Chunk(tone(1.0), "happy")]
    out = butt_join(chunks, SR)
    assert len(out) / SR == pytest.approx(2.060, abs=0.001)


def test_butt_join_empty():
    assert butt_join([], SR).size == 0
