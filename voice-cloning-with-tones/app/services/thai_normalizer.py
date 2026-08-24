from __future__ import annotations

import re
import unicodedata
from pythainlp.util import normalize as pythainlp_normalize

INVISIBLE_CHARS = [
    "\u00ad",  # SOFT HYPHEN
    "\u034f",  # COMBINING GRAPHEME JOINER
    "\u061c",  # ARABIC LETTER MARK
    "\u115f", "\u1160",  # HANGUL FILLERS
    "\u17b4", "\u17b5",  # KHMER VOWEL INHERENT
    "\u180e",  # MONGOLIAN VOWEL SEPARATOR
    "\u200b", "\u200c", "\u200d",  # ZWSP, ZWNJ, ZWJ
    "\u200e", "\u200f",  # LRM, RLM
    "\u2028", "\u2029",  # LINE/PARAGRAPH SEPARATOR
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # Bidi overrides
    "\u2060",  # WORD JOINER
    "\u2066", "\u2067", "\u2068", "\u2069",  # Bidi isolates
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
]
INVISIBLE_PATTERN = re.compile("|".join(re.escape(c) for c in INVISIBLE_CHARS))
MULTI_SPACE_PATTERN = re.compile(r"[ \t\u00a0\u3000]+")
REPEATED_CHARS_PATTERN = re.compile(r"([^\u0e30-\u0e4e\s])\1{2,}")


def normalize_thai_text(text: str) -> str:
    """
    Encoding-only Thai text hygiene:
    - Unicode NFC normalization
    - Stripping zero-width, BOM, bidi, and invisible control codes
    - Normalizing whitespace & collapsing spaces
    - Collapsing 3+ repeated characters to 2
    - PyThaiNLP vowel/tone-mark order hygiene
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE_PATTERN.sub("", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    text = REPEATED_CHARS_PATTERN.sub(r"\1\1", text)
    text = pythainlp_normalize(text)
    return text.strip()
