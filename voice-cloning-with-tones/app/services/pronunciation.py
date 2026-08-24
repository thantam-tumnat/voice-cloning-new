"""User-defined pronunciation overrides, applied just before synthesis.

Thai loanwords written with a silent consonant (การันต์) are the usual reason to
want one: "ไฟล์" should read as ฟาย /faː j/, and when the model instead reads the
base word ไฟ /faj/ it has said "fire". Names, brands and acronyms need the same
escape hatch.

Two rules govern the implementation, both learned by measuring:

* **Match on word boundaries, never substrings.** "โปรไฟล์" is genuinely
  /proː faj/ with the *short* vowel, so a blanket find-and-replace of ไฟล์ would
  corrupt it. PyThaiNLP tokenizes โปรไฟล์ as one token distinct from ไฟล์, so
  token-level matching keeps them apart.
* **Never corrupt text we could not tokenize losslessly.** If the tokens do not
  rejoin to exactly the input, the original is returned unchanged.

Overrides change only what reaches the model. The script shown in the studio, and
everything the API returns, keeps the user's original spelling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.config import settings

# Reloaded when the file's mtime changes, so editing the JSON takes effect without
# restarting the server.
_cache: Dict[str, object] = {"path": None, "mtime": None, "mapping": {}, "max_span": 0}


def dictionary_path(path: str | Path | None = None) -> Path:
    return Path(path or settings.pronunciation_path)


def _tokenize(text: str) -> Optional[List[str]]:
    """Tokens that rejoin to exactly ``text``, or None if that cannot be guaranteed."""
    try:
        from pythainlp.tokenize import word_tokenize

        tokens = word_tokenize(text, keep_whitespace=True)
    except Exception:
        return None
    return tokens if "".join(tokens) == text else None


def _index(mapping: Dict[str, str]) -> Tuple[Dict[str, str], int]:
    """Clean the mapping and work out the longest key in tokens."""
    clean: Dict[str, str] = {}
    max_span = 0
    for key, value in mapping.items():
        key = str(key).strip()
        if not key or value is None:
            continue
        clean[key] = str(value)
        tokens = _tokenize(key)
        max_span = max(max_span, len(tokens) if tokens else 1)
    return clean, max_span


def load_dictionary(path: str | Path | None = None) -> Dict[str, str]:
    """Read the override file, caching until its mtime changes."""
    file = dictionary_path(path)
    try:
        mtime = file.stat().st_mtime
    except OSError:
        _cache.update(path=str(file), mtime=None, mapping={}, max_span=0)
        return {}

    if _cache["path"] == str(file) and _cache["mtime"] == mtime:
        return _cache["mapping"]  # type: ignore[return-value]

    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("expected a JSON object of {written: spoken}")
        mapping, max_span = _index(raw)
    except Exception as e:
        # A malformed file must not take synthesis down; say so and carry on.
        print(f"[Pronunciation] WARNING: could not read {file}: {e}", file=sys.stderr)
        mapping, max_span = {}, 0

    _cache.update(path=str(file), mtime=mtime, mapping=mapping, max_span=max_span)
    return mapping


def save_dictionary(mapping: Dict[str, str], path: str | Path | None = None) -> Dict[str, str]:
    """Write the override file and refresh the cache."""
    file = dictionary_path(path)
    clean, _ = _index(mapping)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    load_dictionary(file)
    return clean


def apply_pronunciation(
    text: str, mapping: Optional[Dict[str, str]] = None
) -> str:
    """Rewrite any token run that the dictionary names, longest match first."""
    if mapping is None:
        mapping = load_dictionary()
        max_span = int(_cache["max_span"])  # type: ignore[arg-type]
    else:
        mapping, max_span = _index(mapping)

    if not text or not mapping:
        return text

    tokens = _tokenize(text)
    if tokens is None:
        return text

    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        for span in range(min(max_span, n - i), 0, -1):
            candidate = "".join(tokens[i:i + span])
            if candidate in mapping:
                out.append(mapping[candidate])
                i += span
                break
        else:
            out.append(tokens[i])
            i += 1
    return "".join(out)
