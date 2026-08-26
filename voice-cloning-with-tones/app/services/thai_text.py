"""Turn written Thai into *speakable* Thai before it reaches the model.

Ported from the port-8010 webhook service (``voice-cloning/src/thai_text.py``), which
reproduced the old n8n ``pyThai`` node. Distinct from ``thai_normalizer.py`` (encoding-only
hygiene): this module rewrites content so the model pronounces it correctly —

  * Arabic/Thai digits -> Thai number words ("1,250" -> "หนึ่งพันสองร้อยห้าสิบ")
  * common abbreviations -> full words ("กม." -> "กิโลเมตร", "8 ก.ย." -> "แปดกันยายน")
  * mai-yamok repetition (ๆ) expanded on the last tokenised word ("ผิวสวยๆ" -> "ผิวสวยสวย")
  * "เทอ" chat spelling -> "เธอ"

``mark_breakpoints`` additionally inserts spaces at sentence / word boundaries so a long
prompt can be chunked without cutting mid-word; short single-utterance inputs are unaffected.
"""

from __future__ import annotations

import re
import unicodedata

from pythainlp.tokenize import sent_tokenize, word_tokenize
from pythainlp.util import num_to_thaiword

# Insert a safe break at least this often so a chunker always finds a space in a long line.
BREAK_EVERY = 160

THAI = "฀-๿"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# Abbreviations the model would otherwise read as symbols or spell out letter by letter.
# "&" is deliberately absent — it would turn AT&T into ATและT.
ABBREV = {
    "%": "เปอร์เซ็นต์", "฿": "บาท", "$": "ดอลลาร์",
    "กม.": "กิโลเมตร", "ก.ม.": "กิโลเมตร", "ซม.": "เซนติเมตร", "กก.": "กิโลกรัม",
    "บ.": "บาท", "น.": "นาฬิกา", "ฯลฯ": "และอื่นๆ",
    "พ.ศ.": "พุทธศักราช", "ค.ศ.": "คริสต์ศักราช", "ดร.": "ดอกเตอร์",
    "ม.ค.": "มกราคม", "ก.พ.": "กุมภาพันธ์", "มี.ค.": "มีนาคม", "เม.ย.": "เมษายน",
    "พ.ค.": "พฤษภาคม", "มิ.ย.": "มิถุนายน", "ก.ค.": "กรกฎาคม", "ส.ค.": "สิงหาคม",
    "ก.ย.": "กันยายน", "ต.ค.": "ตุลาคม", "พ.ย.": "พฤศจิกายน", "ธ.ค.": "ธันวาคม",
}

# Eat spaces around a numeral when both sides are Thai — that space is typography, not a
# pause, and once the numeral becomes a word the model would read it as a pause.
NUMBER = re.compile(
    rf"(?:(?<=[{THAI}]) +)?"
    r"\d+(?:,\d{3})*(?:\.\d+)?"
    rf"(?: +(?=[{THAI}]))?"
)

MAI_YAMOK = re.compile(r"(\S+?)\s*ๆ")


def _read_number(m: "re.Match[str]") -> str:
    raw = m.group(0).strip().replace(",", "")
    if "." in raw:
        whole, frac = raw.split(".", 1)
        head = num_to_thaiword(int(whole)) if whole else "ศูนย์"
        return head + "จุด" + "".join(num_to_thaiword(int(d)) for d in frac)
    return num_to_thaiword(int(raw))


def _expand_mai_yamok(text: str) -> str:
    """ๆ repeats the word before it: เร็วๆ -> เร็วเร็ว (repeat only the final tokenised word)."""

    def repl(m: "re.Match[str]") -> str:
        run = m.group(1)
        try:
            words = word_tokenize(run, engine="newmm")
        except Exception:
            return run * 2
        last = words[-1] if words else run
        return run + last

    return MAI_YAMOK.sub(repl, text)


def expand(text: str, is_thai: bool = True) -> str:
    """Rewrite written Thai as speakable Thai.

    ``is_thai=False`` skips the number/abbreviation pass so "save 20 dollars" does not
    become "save ยี่สิบ dollars".
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).translate(THAI_DIGITS)

    if is_thai:
        # Longest key first so "ก.ม." beats "น.".
        for a in sorted(ABBREV, key=len, reverse=True):
            text = text.replace(a, ABBREV[a])

    # "เทอ" is the chat spelling of "เธอ". Lookarounds protect "เทอม" / "เทอร์โบ".
    text = re.sub(rf"(?<![{THAI}])เทอ(?![{THAI}])", "เธอ", text)

    if is_thai:
        text = NUMBER.sub(_read_number, text)

    text = _expand_mai_yamok(text)
    return re.sub(r"[ \t ]+", " ", text).strip()


def _pack_words(unit: str, limit: int) -> list[str]:
    """Split on newmm word boundaries so no piece exceeds ``limit``."""
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
    """Insert spaces at sentence / word boundaries so a long line can be chunked safely.

    Lossless: nothing is added or removed but the spaces. Short inputs come back unchanged.
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
    """Full pyThai pass: expand + breakpoint-mark every line of the script."""
    is_thai = str(language or "th").lower().startswith("th")
    lines: list[str] = []
    for line in str(prompt or "").splitlines():
        norm = expand(line, is_thai)
        if norm:
            lines.append(mark_breakpoints(norm) if is_thai else norm)
    return "\n".join(lines)


__all__ = ["expand", "mark_breakpoints", "prepare_prompt", "BREAK_EVERY"]
