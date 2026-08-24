"""Tests for src/augment.py — especially the §8.6.2 safety gates.

The thaiword_to_num round-trip is the core invariant. These tests use deterministic
RNGs so probability-driven branches are exercised reliably.
"""

from __future__ import annotations

import random
import unicodedata

import pytest

from src.augment import (
    _is_safe_to_digitize,
    case_jitter,
    maybe_digitize_thai,
    pick_libritts_text,
    whitespace_jitter,
)


# ---------------------------------------------------------------------------
# Safety-gate unit tests
# ---------------------------------------------------------------------------

def test_safe_simple_cardinal():
    """'ยี่สิบสาม' = 23 should be safe."""
    assert _is_safe_to_digitize("ยี่สิบสาม", left_word="", right_word="", context="ยี่สิบสาม")


def test_unsafe_after_ordinal_marker():
    """'ที่หนึ่ง' is ordinal — never substitute."""
    assert not _is_safe_to_digitize(
        "หนึ่ง", left_word="ที่", right_word="", context="ลำดับที่หนึ่ง"
    )


def test_unsafe_before_fractional_marker():
    """'สาม จุด' (decimal point boundary) — skip the bare three."""
    assert not _is_safe_to_digitize(
        "สาม", left_word="", right_word="จุด", context="สามจุดหนึ่งสี่"
    )


def test_unsafe_year_with_song_phan():
    """Years starting 'สองพัน...' often read digit-by-digit; skip when ปี/พ.ศ. nearby."""
    assert not _is_safe_to_digitize(
        "สองพันห้าร้อยหกสิบเจ็ด",
        left_word="ปี",
        right_word="",
        context="ในปีสองพันห้าร้อยหกสิบเจ็ด",
    )


def test_unsafe_digit_recital_unspaced():
    """'หนึ่งสองสามสี่' (no spaces, four single-digit words) is phone-recital style.

    This is the case the original code missed because it split on whitespace.
    """
    assert not _is_safe_to_digitize(
        "หนึ่งสองสามสี่",
        left_word="",
        right_word="",
        context="หนึ่งสองสามสี่",
    )


def test_safe_with_unit_modifier_even_no_spaces():
    """A unit modifier (ร้อย/พัน/...) is enough to call it cardinal."""
    assert _is_safe_to_digitize(
        "หนึ่งร้อย",
        left_word="",
        right_word="",
        context="ราคาหนึ่งร้อยบาท",
    )


def test_unsafe_lone_single_digit():
    """Lone 'หนึ่ง' may be 'a/one' colloquially — too risky."""
    assert not _is_safe_to_digitize(
        "หนึ่ง", left_word="ใน", right_word="ของ", context="หนึ่งในของ"
    )


# ---------------------------------------------------------------------------
# Driver tests — maybe_digitize_thai end-to-end
# ---------------------------------------------------------------------------

def test_digitize_full_substitution():
    """p_full=1.0 → every safe span flips to digits."""
    rng = random.Random(0)
    out = maybe_digitize_thai("ราคาหนึ่งร้อยบาท", p_full=1.0, p_partial=0.0, rng=rng)
    assert "100" in out


def test_digitize_zero_probability_passthrough():
    rng = random.Random(0)
    text = "ราคาหนึ่งร้อยบาท"
    assert maybe_digitize_thai(text, p_full=0.0, p_partial=0.0, rng=rng) == text


def test_digitize_skips_empty_string():
    assert maybe_digitize_thai("") == ""


def test_digitize_skips_no_thai_numbers():
    """No number-words in the text → no change."""
    text = "สวัสดีครับ"
    assert maybe_digitize_thai(text, p_full=1.0, p_partial=0.0) == text


def test_digitize_does_not_touch_recital():
    """Even at p_full=1.0, digit-recital should NOT be substituted."""
    rng = random.Random(0)
    out = maybe_digitize_thai("หนึ่งสองสามสี่", p_full=1.0, p_partial=0.0, rng=rng)
    # Either unchanged OR (if PyThaiNLP round-trip fails) at least no '1234'.
    assert out == "หนึ่งสองสามสี่" or "1234" not in out


# ---------------------------------------------------------------------------
# Whitespace jitter — Thai cluster safety
# ---------------------------------------------------------------------------

def test_whitespace_jitter_does_not_split_combining_marks():
    """Insert space, but never between a Thai base and its tone mark."""
    rng = random.Random(7)
    text = "ภาษาไทย"
    for _ in range(200):
        out = whitespace_jitter(text, p=1.0, rng=rng)
        # Walk the output: a space must never directly precede a combining mark.
        for i, ch in enumerate(out):
            if ch == " " and i + 1 < len(out):
                assert not unicodedata.combining(out[i + 1]), \
                    f"space inserted before combining mark at index {i}: {out!r}"


def test_whitespace_jitter_passthrough_below_threshold():
    rng = random.Random(0)
    text = "abc"  # too short to trigger insertion
    out = whitespace_jitter(text, p=1.0, rng=rng)
    assert " " not in out or out.endswith("abc")  # may drop period or no-op


def test_whitespace_jitter_drops_trailing_period():
    """At high p, the period drop branch should fire eventually."""
    rng = random.Random(42)
    saw_drop = False
    for _ in range(50):
        out = whitespace_jitter("hello.", p=1.0, rng=rng)
        if not out.endswith("."):
            saw_drop = True
            break
    assert saw_drop


# ---------------------------------------------------------------------------
# Case jitter
# ---------------------------------------------------------------------------

def test_case_jitter_lowers_latin_only():
    rng = random.Random(0)
    out = case_jitter("Hello ภาษาไทย WORLD", p=1.0, rng=rng)
    assert "hello" in out and "world" in out
    assert "ภาษาไทย" in out


def test_case_jitter_passthrough_at_p_zero():
    text = "Hello WORLD"
    assert case_jitter(text, p=0.0) == text


# ---------------------------------------------------------------------------
# LibriTTS picker
# ---------------------------------------------------------------------------

def test_libritts_picker_prefers_original():
    """Default p_normalized=0.3 → ~70% of samples return text_original."""
    rng = random.Random(123)
    counts = {"o": 0, "n": 0}
    for _ in range(2000):
        chosen = pick_libritts_text("ORIG", "NORM", p_normalized=0.3, rng=rng)
        counts["o" if chosen == "ORIG" else "n"] += 1
    ratio_norm = counts["n"] / 2000
    assert 0.25 <= ratio_norm <= 0.35   # ±0.05 around 0.3


def test_libritts_picker_falls_back():
    """Both empty → both return empty (callers must validate upstream)."""
    assert pick_libritts_text("", "raw", p_normalized=0.0) == ""


@pytest.mark.parametrize("p_normalized", [0.0, 1.0])
def test_libritts_picker_extremes(p_normalized: float):
    rng = random.Random(0)
    chosen = pick_libritts_text("ORIG", "NORM", p_normalized=p_normalized, rng=rng)
    assert chosen == ("NORM" if p_normalized == 1.0 else "ORIG")
