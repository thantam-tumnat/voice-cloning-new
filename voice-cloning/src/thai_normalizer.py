"""Encoding-only Thai text hygiene for SiangTTS.

Intentionally minimal compared to IndexTTS2-Thai's normalizer. VoxCPM 2 is
tokenizer-free on the audio side and uses MiniCPM-4 BPE on the text side, so we do
*not* run number-to-word, word segmentation, or phonetic respelling — JaiTTS-style.
See RESEARCH.md §8.4 for the rationale.

Kept:
- Invisible / control unicode stripping (zero-width, BOM, bidi, etc.)
- NBSP / tab → space, multi-space collapse
- Repeated-character collapse (3+ same char → 2) — outside Thai vowel/tone marks.
- PyThaiNLP `thai_normalize` for tone-mark / zero-width fixes.
"""

from __future__ import annotations

import re
import unicodedata

from pythainlp.util import normalize as pythainlp_normalize

# Explicit escapes — easier to audit than hard-to-see literal code points.
INVISIBLE_RE = re.compile(
    "["
    "­"            # SOFT HYPHEN
    "͏"            # COMBINING GRAPHEME JOINER
    "؜"            # ARABIC LETTER MARK
    "ᅟᅠ"      # HANGUL FILLERS
    "឴឵"      # KHMER VOWEL INHERENT AQ/AA
    "᠎"            # MONGOLIAN VOWEL SEPARATOR
    "​‌‍"  # ZWSP / ZWNJ / ZWJ
    "‎‏"      # LRM / RLM
    "  "      # LINE / PARAGRAPH SEPARATOR
    "‪-‮"     # bidi embeddings / overrides
    "⁠"            # WORD JOINER
    "⁦-⁩"     # bidi isolates
    "﻿"            # ZERO WIDTH NO-BREAK SPACE / BOM
    "￹-￻"     # interlinear annotation anchors
    "]"
)
NBSP_TAB_RE = re.compile("[ \t]")
MULTI_SPACE_RE = re.compile(r" {2,}")

# Repeated-char collapse: only collapse outside Thai marks, since Thai uses
# combining tone marks (ิ-ฺ, ็-๎) and these legitimately
# repeat in cluster sequences. We only collapse 3+ identical *base* chars.
_THAI_MARK = re.compile(r"[ะ-ฺเ-๎]")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")


def _collapse_repeats(text: str) -> str:
    """Collapse 3+ runs of identical chars to 2, but skip Thai vowels/tone marks."""

    def repl(m: re.Match[str]) -> str:
        ch = m.group(1)
        if _THAI_MARK.match(ch):
            return m.group(0)
        return ch * 2

    return REPEATED_CHAR_RE.sub(repl, text)


def normalize_thai_text(text: str) -> str:
    """Encoding-only cleanup. Does NOT alter pronunciation form.

    Order: NFC → invisible-strip → NBSP/tab→space → PyThaiNLP fix →
    repeated-char collapse → multi-space collapse → strip.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE_RE.sub("", text)
    text = NBSP_TAB_RE.sub(" ", text)
    text = pythainlp_normalize(text)
    text = _collapse_repeats(text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


__all__ = ["normalize_thai_text"]
