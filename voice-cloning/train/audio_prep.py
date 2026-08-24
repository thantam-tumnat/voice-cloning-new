"""Shared audio-prep helpers for manifest writers.

Two responsibilities (RESEARCH.md §4.D):

1. **Trim trailing silence to <0.5 s.** Long trailing silence is the #1 cause of
   "model never stops generating" after fine-tuning.
2. **Reject clips outside the 1–30 s usable window.** <1 s clips are unstable;
   >30 s exceeds VoxCPM's recommended training-clip length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Thresholds chosen to match VoxCPM's published guidance.
MIN_DURATION_S = 1.0
MAX_DURATION_S = 30.0
MAX_TRAILING_SILENCE_S = 0.5
SILENCE_DBFS_THRESHOLD = -45.0      # below this is "silence" for trimming purposes


def _is_silent_frame(frame: np.ndarray) -> bool:
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12))
    if rms <= 0:
        return True
    dbfs = 20.0 * np.log10(rms)
    return dbfs < SILENCE_DBFS_THRESHOLD


def trim_trailing_silence(
    wav: np.ndarray,
    sample_rate: int,
    max_trailing_s: float = MAX_TRAILING_SILENCE_S,
    frame_ms: int = 20,
) -> np.ndarray:
    """Trim trailing silence so at most `max_trailing_s` of quiet remains.

    Operates on a 1-D float waveform in [-1, 1]. Frames are scanned from the end
    backward; the last non-silent frame anchors the cut, plus a small tail.
    """
    if wav.ndim != 1:
        raise ValueError("trim_trailing_silence expects a 1-D waveform")
    n = wav.shape[0]
    if n == 0:
        return wav

    frame = max(1, int(sample_rate * frame_ms / 1000))
    last_voiced_end = 0
    for start in range(n - frame, -1, -frame):
        if not _is_silent_frame(wav[start : start + frame]):
            last_voiced_end = start + frame
            break

    # Keep up to `max_trailing_s` of the trailing silence.
    keep = min(n, last_voiced_end + int(sample_rate * max_trailing_s))
    return wav[:keep] if keep > 0 else wav


def resample_trim_save(
    audio_array: np.ndarray,
    src_sr: int,
    dst_path: Path,
    target_sr: int,
    *,
    min_duration_s: float = MIN_DURATION_S,
    max_duration_s: float = MAX_DURATION_S,
) -> float | None:
    """Resample → trim trailing silence → save to `dst_path`. Returns duration in
    seconds, or **None** if the clip is outside the [min, max] window after trim
    (caller should skip those rows)."""
    wav = torch.tensor(audio_array, dtype=torch.float32)
    if wav.dim() > 1:
        wav = wav.mean(dim=0)         # downmix to mono
    if src_sr != target_sr:
        wav = torchaudio.functional.resample(wav, src_sr, target_sr)
    arr = wav.numpy()
    arr = trim_trailing_silence(arr, target_sr)
    duration = arr.shape[0] / target_sr
    if duration < min_duration_s or duration > max_duration_s:
        return None
    sf.write(str(dst_path), arr, target_sr)
    return duration


__all__ = [
    "MIN_DURATION_S",
    "MAX_DURATION_S",
    "MAX_TRAILING_SILENCE_S",
    "trim_trailing_silence",
    "resample_trim_save",
]
