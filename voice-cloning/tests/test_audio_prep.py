"""Tests for train/audio_prep.py — silence trimming + duration filter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from train.audio_prep import resample_trim_save, trim_trailing_silence


def _voiced_then_silence(sr: int, voiced_s: float, silence_s: float) -> np.ndarray:
    voiced = 0.5 * np.sin(2 * np.pi * 220 * np.arange(int(voiced_s * sr)) / sr)
    silence = np.zeros(int(silence_s * sr))
    return np.concatenate([voiced, silence]).astype(np.float32)


def test_trim_keeps_voiced_section():
    sr = 24000
    wav = _voiced_then_silence(sr, voiced_s=2.0, silence_s=3.0)
    out = trim_trailing_silence(wav, sr, max_trailing_s=0.5)
    # Should keep roughly 2.0 + 0.5 = 2.5 s, with frame-quantization slack.
    duration = out.shape[0] / sr
    assert 2.4 <= duration <= 2.7


def test_trim_does_not_extend_beyond_input():
    sr = 16000
    wav = _voiced_then_silence(sr, voiced_s=1.0, silence_s=0.1)
    out = trim_trailing_silence(wav, sr, max_trailing_s=0.5)
    assert out.shape[0] <= wav.shape[0]


def test_trim_handles_all_silence():
    sr = 16000
    wav = np.zeros(int(2.0 * sr), dtype=np.float32)
    out = trim_trailing_silence(wav, sr, max_trailing_s=0.5)
    # All-silent → last_voiced_end stays 0; we keep `max_trailing_s` worth.
    assert out.shape[0] <= int(0.5 * sr) + 1


def test_resample_trim_save_rejects_too_short(tmp_path: Path):
    sr = 24000
    wav = _voiced_then_silence(sr, voiced_s=0.3, silence_s=0.05)   # < 1 s total
    out = resample_trim_save(wav, sr, tmp_path / "out.wav", target_sr=48000)
    assert out is None
    assert not (tmp_path / "out.wav").exists()


def test_resample_trim_save_rejects_too_long(tmp_path: Path):
    sr = 24000
    wav = _voiced_then_silence(sr, voiced_s=31.0, silence_s=0.0)
    out = resample_trim_save(wav, sr, tmp_path / "out.wav", target_sr=48000)
    assert out is None


def test_resample_trim_save_writes_for_valid_clip(tmp_path: Path):
    sr = 24000
    wav = _voiced_then_silence(sr, voiced_s=2.0, silence_s=0.3)
    dst = tmp_path / "out.wav"
    duration = resample_trim_save(wav, sr, dst, target_sr=48000)
    assert duration is not None
    assert dst.exists()
    assert 2.0 <= duration <= 2.6
