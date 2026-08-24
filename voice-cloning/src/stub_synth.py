"""A synthesizer that makes no sound worth listening to and needs no GPU.

Turned on with `SIANGTTS_GPU_STUB=1`. It exists so the whole service split — queue,
lanes, voice handles, LoRA switching, the webhook's merge/upload/callback path, the
tone studio's chunk assembly — can be exercised end to end while the real model is
busy or absent, on a machine with one GPU that is already committed.

It is *not* a production fallback. The real engine never falls back to this: a
silent fallback sounds exactly like a broken model. `/health` reports `stub: true`
so no client can mistake one for the other, and the service refuses to start in stub
mode unless the environment asked for it explicitly.

What it preserves, so tests mean something:

* audio length tracks text length, so merge and rate-matching see realistic input;
* the tone's pitch is derived from the voice cache, so "was this chunk conditioned
  on the voice I asked for?" is answerable from the audio alone;
* prompt caches round-trip through disk without torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

SAMPLE_RATE = 48000

# Seconds to spend "generating" each chunk. Zero by default — but queueing, lane
# priority and progress reporting are only observable if a job takes measurable
# time, and numpy finishes instantly. Set SIANGTTS_STUB_DELAY to test them.
DELAY_S = float(os.environ.get("SIANGTTS_STUB_DELAY", "0"))

# Roughly Thai reading speed at 1x, so a 250-character chunk lands near 10 s and the
# ffmpeg merge sees plausible durations.
SECONDS_PER_CHAR = 0.04
MIN_SECONDS = 0.35


def _hz(seed: str) -> float:
    """A stable pitch per voice, inside a range that survives MP3 encoding."""
    h = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16)
    return 110.0 + (h % 400)


class StubSynthesizer:
    """Mirrors the parts of `src.inference.Synthesizer` the service actually calls."""

    is_stub = True

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.model = None
        self.tts_model = None
        # Every generation, for assertions. Cheap enough to always keep: a chunk is
        # one small dict, and the service bounds job history anyway.
        self.calls: list[dict] = []

    # -- voices ---------------------------------------------------------- #

    def build_voice(
        self,
        ref_audio: str,
        prompt_text: Optional[str] = None,
        prompt_audio: Optional[str] = None,
    ) -> dict:
        name = Path(ref_audio).stem
        return {
            "stub": True,
            "ref": name,
            "prompt_text": prompt_text or "",
            # The real cache carries this; the tone studio reads it to warn that
            # continuation mode will ignore style instructions.
            "mode": "ref_continuation" if prompt_text else "reference",
            "voice_seed": f"{name}:{prompt_text or ''}",
        }

    def save_voice(self, prompt_cache: dict, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prompt_cache, ensure_ascii=False), encoding="utf-8")
        return path

    def load_voice(self, path: str | Path) -> dict:
        raw = Path(path).read_text(encoding="utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            # A real .pt from an earlier non-stub run. Say so plainly instead of
            # failing somewhere further down with a confusing error.
            raise RuntimeError(
                f"{path} is a real prompt cache; the stub engine cannot read it. "
                f"Point SIANGTTS_CACHE_DIR at a scratch directory in stub mode."
            ) from e

    # -- generation ------------------------------------------------------ #

    def _tone(self, text: str, seed: str):
        import numpy as np

        if DELAY_S:
            time.sleep(DELAY_S)
        duration = max(MIN_SECONDS, len(text) * SECONDS_PER_CHAR)
        n = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        wav = 0.2 * np.sin(2 * np.pi * _hz(seed) * t)
        # Fade the ends so the webhook's silence-trim filter has something to bite on
        # and the merge does not click at every chunk boundary.
        edge = min(n // 8, int(0.05 * self.sample_rate)) or 1
        wav[:edge] *= np.linspace(0, 1, edge)
        wav[-edge:] *= np.linspace(1, 0, edge)
        return wav.astype("float32")

    def synth_cached(
        self,
        text: str,
        prompt_cache: dict,
        *,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
    ):
        seed = (prompt_cache or {}).get("voice_seed", "unconditioned")
        self.calls.append(
            {
                "text": text,
                "voice_seed": seed,
                "cfg_value": cfg_value,
                "timesteps": inference_timesteps,
            }
        )
        return self._tone(text, seed)

    def synth(
        self,
        text: str,
        *,
        ref_audio: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        **kwargs: Any,
    ):
        seed = Path(ref_audio).stem if ref_audio else "unconditioned"
        self.calls.append(
            {
                "text": text,
                "voice_seed": seed,
                "cfg_value": cfg_value,
                "timesteps": inference_timesteps,
            }
        )
        return self._tone(text, seed)


__all__ = ["StubSynthesizer", "SAMPLE_RATE"]
