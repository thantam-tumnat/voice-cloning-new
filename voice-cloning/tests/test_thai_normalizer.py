"""Tests for src/thai_normalizer.py.

Goal: confirm encoding hygiene works without altering pronunciation form.
"""

from __future__ import annotations

from src.thai_normalizer import normalize_thai_text


def test_strips_zero_width_chars():
    text = "สวัสดี​ครับ‌"
    assert normalize_thai_text(text) == "สวัสดีครับ"


def test_strips_bom():
    assert normalize_thai_text("﻿สวัสดี") == "สวัสดี"


def test_strips_bidi_marks():
    assert normalize_thai_text("‪hello‬") == "hello"


def test_collapses_3_plus_repeats_outside_thai_marks():
    assert normalize_thai_text("hellooooo") == "helloo"


def test_preserves_thai_tone_marks():
    """Tone marks legitimately repeat in clusters; don't collapse them."""
    text = "ภาษาไทย"
    assert normalize_thai_text(text) == text


def test_collapses_multi_space():
    assert normalize_thai_text("hello   world") == "hello world"


def test_nfc_normalization():
    """Pre-composed (NFC) output regardless of input form."""
    decomposed = "kä"   # 'k' + 'a' + combining diaeresis
    out = normalize_thai_text(decomposed)
    assert out == "kä"


def test_handles_empty_string():
    assert normalize_thai_text("") == ""


def test_does_not_convert_digits_to_words():
    """We deliberately keep raw digits — no number-to-word."""
    assert normalize_thai_text("ราคา 1923 บาท") == "ราคา 1923 บาท"


def test_does_not_strip_english_letters():
    """Code-switching should survive."""
    assert normalize_thai_text("ใช้ AI ทำงาน") == "ใช้ AI ทำงาน"


def test_strips_tab():
    assert normalize_thai_text("hello\tworld") == "hello world"


def test_nbsp_to_space():
    assert normalize_thai_text("hello world") == "hello world"
