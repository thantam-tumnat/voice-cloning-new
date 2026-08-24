"""DataLoader-time text augmentations for SiangTTS.

Implements the augmentation menu from RESEARCH.md §8.6:

- Thai number-word → Arabic digit (cardinal-only, gated by §8.6.2 safety rules).
- Whitespace / punctuation jitter.
- Case jitter for Latin tokens.
- LibriTTS text_original ↔ text_normalized sampling helper.

All augmentations are pure-text and applied inside `__getitem__` so the model sees a
fresh spelling of the same audio across epochs (the invariance is what we want).
"""

from __future__ import annotations

import random
import re
import unicodedata

from pythainlp.util import num_to_thaiword, thaiword_to_num

THAI_DIGIT_WORDS = {
    "ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่",
    "ห้า", "หก", "เจ็ด", "แปด", "เก้า",
}
THAI_UNIT_WORDS = {"สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน", "ยี่สิบ", "เอ็ด"}

ORDINAL_PRECEDERS = {"ที่", "ลำดับ", "อันดับ", "ครั้งที่"}
NON_CARDINAL_NEIGHBORS = {"ครึ่ง", "ส่วน", "จุด", "โหล", "กุรุส"}
YEAR_CONTEXT = {"ปี", "พ.ศ.", "ค.ศ."}

NUMWORD_SPAN = re.compile(
    r"(?:หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ|ยี่สิบ|"
    r"ร้อย|พัน|หมื่น|แสน|ล้าน|เอ็ด|ศูนย์)+"
)

# Used by the digit-recital filter (§8.6.2 rule 4). Longest-first so the alternation
# prefers "ศูนย์" over a shorter prefix match. Built once at import.
_DIGIT_WORD_RE = re.compile(
    "|".join(sorted(THAI_DIGIT_WORDS, key=len, reverse=True))
)


def _is_safe_to_digitize(span: str, left_word: str, right_word: str, context: str) -> bool:
    """Apply the §8.6.2 safety gates. All must pass before substituting."""
    # (1) round-trip cardinal check
    try:
        n = thaiword_to_num(span)
        if n is None or num_to_thaiword(n) != span:
            return False
    except Exception:
        return False

    # (2) span must contain a unit OR be ≥ 2 distinct number-words
    has_unit = any(unit in span for unit in THAI_UNIT_WORDS)
    digit_word_hits = sum(1 for w in THAI_DIGIT_WORDS if w in span)
    if not has_unit and digit_word_hits < 2:
        return False

    # (3) ordinal / fraction / quantifier neighbor
    if left_word in ORDINAL_PRECEDERS:
        return False
    if left_word in NON_CARDINAL_NEIGHBORS or right_word in NON_CARDINAL_NEIGHBORS:
        return False

    # (4) digit-by-digit recital — span is all single-digit words with no unit
    # modifier. Thai is unspaced, so tokenize against the digit-word vocabulary
    # rather than splitting on whitespace.
    if not has_unit:
        digits = _DIGIT_WORD_RE.findall(span)
        if digits and "".join(digits) == span and len(digits) >= 4:
            return False

    # (5) year heuristic — Thai years often read digit-by-digit; skip if unsure
    if any(yc in context for yc in YEAR_CONTEXT) and span.startswith("สองพัน"):
        return False

    return True


def _word_left_of(text: str, idx: int) -> str:
    left = text[:idx].strip()
    return left.split()[-1] if left else ""


def _word_right_of(text: str, idx: int) -> str:
    right = text[idx:].strip()
    return right.split()[0] if right else ""


def maybe_digitize_thai(
    text: str,
    p_full: float = 0.4,
    p_partial: float = 0.1,
    rng: random.Random | None = None,
) -> str:
    """Thai number-word → Arabic digit augmentation. Word→digit only (never reverse).

    Args:
        text:      input Thai text (post-normalizer).
        p_full:    probability of substituting *all* safe spans in the sentence.
        p_partial: probability of substituting exactly one randomly chosen safe span.
        rng:       optional Random instance for reproducibility.

    Returns:
        Possibly augmented text. Unchanged with probability 1 - p_full - p_partial,
        or when no safe spans are found.
    """
    rng = rng or random

    spans = [(m.start(), m.end(), m.group()) for m in NUMWORD_SPAN.finditer(text)]
    if not spans:
        return text

    safe = []
    for start, end, span in spans:
        left = _word_left_of(text, start)
        right = _word_right_of(text, end)
        ctx = text[max(0, start - 30) : min(len(text), end + 30)]
        if _is_safe_to_digitize(span, left, right, ctx):
            safe.append((start, end, span))

    if not safe:
        return text

    r = rng.random()
    if r < p_full:
        chosen = safe
    elif r < p_full + p_partial:
        chosen = [rng.choice(safe)]
    else:
        return text

    out, last = [], 0
    for start, end, span in sorted(chosen):
        out.append(text[last:start])
        out.append(str(thaiword_to_num(span)))
        last = end
    out.append(text[last:])
    return "".join(out)


def _safe_split_indices(text: str) -> list[int]:
    """Indices where inserting a space won't break a Thai grapheme cluster.

    A Thai cluster is a base character optionally followed by tone marks / vowel
    signs (Unicode `Mn`/`Mc` / `Me`). Inserting a space between a base and its
    combining marks corrupts the cluster — rendering / pronunciation breaks.
    Safe positions are those where the *next* character is not a combining mark.
    """
    return [i for i in range(1, len(text)) if not unicodedata.combining(text[i])]


def whitespace_jitter(text: str, p: float = 0.15, rng: random.Random | None = None) -> str:
    """Random extra space / drop trailing period — Thai-cluster-safe."""
    rng = rng or random
    if rng.random() >= p:
        return text
    if rng.random() < 0.5 and len(text) > 4:
        candidates = _safe_split_indices(text)
        if candidates:
            i = rng.choice(candidates)
            text = text[:i] + " " + text[i:]
    if rng.random() < 0.3 and text.endswith("."):
        text = text[:-1]
    return text


def case_jitter(text: str, p: float = 0.1, rng: random.Random | None = None) -> str:
    """Lowercase the Latin substring with probability p. Cheap EN robustness."""
    rng = rng or random
    if rng.random() >= p:
        return text

    def lower_match(m: re.Match) -> str:
        return m.group(0).lower()

    return re.sub(r"[A-Za-z]+", lower_match, text)


def pick_libritts_text(
    text_original: str,
    text_normalized: str,
    p_normalized: float = 0.3,
    rng: random.Random | None = None,
) -> str:
    """Sample between LibriTTS's two text columns.

    Bias toward `text_original` (raw digits + abbreviations) since that's the
    capability we actually want — see RESEARCH.md §8.3.
    """
    rng = rng or random
    return text_normalized if rng.random() < p_normalized else text_original


__all__ = [
    "maybe_digitize_thai",
    "whitespace_jitter",
    "case_jitter",
    "pick_libritts_text",
]
