"""Thai text preparation for the webhook pipeline — port of two n8n nodes.

The old flow did this work in n8n: a `pyThai` node (pythainlp normalisation +
breakpoint insertion) feeding a `chunker` node (split into synth-sized pieces).
Both are reproduced here so the pipeline chunks a script the same way the n8n
flow did, minus the JS/Python split that forced BREAK_EVERY and MAX to be kept
in sync by hand across two nodes.

Distinct from `src/thai_normalizer.py`: that one is encoding-only hygiene
applied at synthesis time and deliberately does *not* touch pronunciation.
This module is the upstream pass that turns written Thai into speakable Thai
(numbers, abbreviations) and gives the chunker safe places to cut.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from pythainlp.tokenize import sent_tokenize, word_tokenize
from pythainlp.util import num_to_thaiword

# Longest chunk handed to the model. Was MAX in the n8n chunker node.
MAX_CHARS = 250

# Breakpoints are inserted at least this often, guaranteeing the chunker always
# finds a space to latch onto within MAX_CHARS. Was BREAK_EVERY in pyThai.
BREAK_EVERY = 160

THAI = "฀-๿"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# Abbreviations the model reads as symbols rather than words.
# "&" is deliberately absent — it would turn AT&T into ATและT.
ABBREV = {
    "%": "เปอร์เซ็นต์", "฿": "บาท", "$": "ดอลลาร์",
    "กม.": "กิโลเมตร", "ก.ม.": "กิโลเมตร", "ซม.": "เซนติเมตร", "กก.": "กิโลกรัม",
    "บ.": "บาท", "น.": "นาฬิกา", "ฯลฯ": "และอื่นๆ",
    "พ.ศ.": "พุทธศักราช", "ค.ศ.": "คริสต์ศักราช", "ดร.": "ดอกเตอร์",
    # Month abbreviations: without these "8 ก.ย." comes out as "แปดก.ย." and the
    # model spells the letters out.
    "ม.ค.": "มกราคม", "ก.พ.": "กุมภาพันธ์", "มี.ค.": "มีนาคม", "เม.ย.": "เมษายน",
    "พ.ค.": "พฤษภาคม", "มิ.ย.": "มิถุนายน", "ก.ค.": "กรกฎาคม", "ส.ค.": "สิงหาคม",
    "ก.ย.": "กันยายน", "ต.ค.": "ตุลาคม", "พ.ย.": "พฤศจิกายน", "ธ.ค.": "ธันวาคม",
}

# Spaces around digits are typography, not a pause — "อายุ 23 ปี" is written that
# way because numerals need breathing room. Once the numeral becomes a word that
# space is wrong twice over: the model reads it as a pause, and the chunker may
# cut there. Eat it when both sides are Thai; leave it otherwise ("500 MB" keeps
# its space).
NUMBER = re.compile(
    rf"(?:(?<=[{THAI}]) +)?"
    r"\d+(?:,\d{3})*(?:\.\d+)?"
    rf"(?: +(?=[{THAI}]))?"
)

MAI_YAMOK = re.compile(r"(\S+?)\s*ๆ")


def _read_number(m: re.Match[str]) -> str:
    raw = m.group(0).strip().replace(",", "")
    if "." in raw:
        whole, frac = raw.split(".", 1)
        head = num_to_thaiword(int(whole)) if whole else "ศูนย์"
        return head + "จุด" + "".join(num_to_thaiword(int(d)) for d in frac)
    return num_to_thaiword(int(raw))


def _expand_mai_yamok(text: str) -> str:
    """ๆ repeats the word before it: เร็วๆ -> เร็วเร็ว

    The n8n node used a greedy `(\\S+)\\s*ๆ`. Thai is normally written without
    spaces, so \\S+ swallowed the whole run and "ผิวสวยๆ" became
    "ผิวสวยผิวสวย" instead of "ผิวสวยสวย". Tokenise the run and repeat only its
    final word.
    """

    def repl(m: re.Match[str]) -> str:
        run = m.group(1)
        try:
            words = word_tokenize(run, engine="newmm")
        except Exception:
            return run * 2
        last = words[-1] if words else run
        return run + last

    return MAI_YAMOK.sub(repl, text)


def expand(text: str, is_thai: bool) -> str:
    """Rewrite written Thai as speakable Thai. `is_thai=False` skips the
    number/abbreviation pass so "save 20 dollars" doesn't become
    "save ยี่สิบ dollars"."""
    text = unicodedata.normalize("NFC", text).translate(THAI_DIGITS)

    if is_thai:
        # Longest key first so "ก.ม." beats "น.".
        for a in sorted(ABBREV, key=len, reverse=True):
            text = text.replace(a, ABBREV[a])

    # "เทอ" is the chat spelling of "เธอ" — same pronunciation, so this is a
    # spelling fix. Lookarounds protect "เทอม" / "เทอร์โบ".
    text = re.sub(rf"(?<![{THAI}])เทอ(?![{THAI}])", "เธอ", text)

    if is_thai:
        text = NUMBER.sub(_read_number, text)

    text = _expand_mai_yamok(text)
    return re.sub(r"[ \t ]+", " ", text).strip()


def _pack_words(unit: str, limit: int) -> list[str]:
    """Split on newmm word boundaries so no piece exceeds `limit`."""
    if len(unit) <= limit:
        return [unit]
    try:
        words = word_tokenize(unit, engine="newmm")
    except Exception:
        return [unit]
    out: list[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + len(w) > limit:
            out.append(cur)
            cur = ""
        if len(w) > limit:               # single word over budget — no other option
            out.extend(w[i:i + limit] for i in range(0, len(w), limit))
            continue
        cur += w
    if cur:
        out.append(cur)
    return [p for p in (x.strip() for x in out) if p]


def mark_breakpoints(text: str, limit: int = BREAK_EVERY) -> str:
    """Insert spaces where a cut is safe, giving the chunker somewhere to latch.

    The chunker looks for a space first, but Thai prompts are typically typed
    with no spaces at all, so it falls back to slicing at exactly MAX_CHARS —
    mid-word.

    Two passes, best cut first:
      1. Sentence joins — thaisum reads sentence-final particles (ค่ะ/ครับ/นะคะ)
         so it finds joins in space-free text. crfcut, the default, reads spaces
         and hands text like this back in one lump.
      2. Word boundaries — sentences still over `limit` get split further on
         newmm boundaries. Without this pass the chunker finds no space within
         MAX_CHARS and cuts mid-word anyway (measured on the n8n version: 6% of
         chunks landed mid-word, and one chunk ran to 223 chars).

    Both passes are lossless — nothing is added or removed but the spaces.
    """
    try:
        sents = [s.strip() for s in sent_tokenize(text, engine="thaisum")]
    except Exception:
        sents = [text]
    out: list[str] = []
    for s in sents:
        if not s:
            continue
        out.extend(_pack_words(s, limit) if len(s) > limit else [s])
    return " ".join(out) or text


def prepare_prompt(prompt: str, language: str = "th") -> str:
    """Full pyThai node: expand + breakpoint-mark every line of the script."""
    is_thai = str(language or "th").lower().startswith("th")
    lines: list[str] = []
    for line in str(prompt or "").splitlines():
        norm = expand(line, is_thai)
        if norm:
            # English already carries its own spaces; the chunker can latch on
            # those, so skip the breakpoint pass.
            lines.append(mark_breakpoints(norm) if is_thai else norm)
    return "\n".join(lines)


@dataclass(frozen=True)
class Chunk:
    text: str
    filename: str          # no extension — "<queue_id>_000"
    index: int             # 1-based, matches the n8n chunkIndex
    total: int


def chunk_text(text: str, prefix: str = "chunk", max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Split a prepared script into synth-sized pieces.

    Port of the n8n `chunker` node, including its fallback order: prefer the
    last space at or before `max_chars`; if the line has no space that early,
    take the first space *after* it rather than cutting a word in half; only
    hard-cut at `max_chars` when the line has no spaces at all.
    """
    pieces: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        while len(line) > max_chars:
            brk = line.rfind(" ", 0, max_chars + 1)
            if brk == -1:
                brk = line.find(" ", max_chars)
            if brk == -1:
                brk = max_chars
            pieces.append(line[:brk].strip())
            line = line[brk:].strip()
        if line:
            pieces.append(line)

    total = len(pieces)
    return [
        Chunk(text=p, filename=f"{prefix}_{i:03d}", index=i + 1, total=total)
        for i, p in enumerate(pieces)
    ]


__all__ = [
    "MAX_CHARS",
    "BREAK_EVERY",
    "Chunk",
    "chunk_text",
    "expand",
    "mark_breakpoints",
    "prepare_prompt",
]
