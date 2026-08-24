"""Tests for prompt-cache persistence in src/inference.py.

A prompt cache must stay on CPU. voxcpm's `_encode_wav` ends in `.cpu()` and
`_generate_with_prompt_cache` assembles the ref prefix around `text_token` —
always a CPU tensor, since the tokenizer builds it with `torch.LongTensor` —
moving the finished result to the GPU itself. Putting a CUDA tensor back into
that cache makes `torch.cat` fail with a device mismatch, so the round trip
through disk must not "helpfully" relocate anything.

The Synthesizer is built without __init__ so no model is loaded.
"""

from __future__ import annotations

import torch

from src.inference import Synthesizer


def _synth() -> Synthesizer:
    return object.__new__(Synthesizer)


def _cache() -> dict:
    return {
        "ref_audio_feat": torch.randn(3, 2, 4),
        "audio_feat": torch.randn(2, 2, 4),
        "prompt_text": "ข้อความอ้างอิง",
        "mode": "ref_continuation",
    }


def test_round_trip_stays_on_cpu(tmp_path):
    s = _synth()
    s.save_voice(_cache(), tmp_path / "v.pt")
    loaded = s.load_voice(tmp_path / "v.pt")
    tensors = [v for v in loaded.values() if torch.is_tensor(v)]
    assert tensors, "expected tensors in the cache"
    assert all(t.device.type == "cpu" for t in tensors)


def test_round_trip_preserves_values(tmp_path):
    s = _synth()
    original = _cache()
    s.save_voice(original, tmp_path / "v.pt")
    loaded = s.load_voice(tmp_path / "v.pt")

    assert set(loaded) == set(original)
    assert loaded["prompt_text"] == original["prompt_text"]
    assert loaded["mode"] == original["mode"]
    assert torch.equal(loaded["ref_audio_feat"], original["ref_audio_feat"])
    assert torch.equal(loaded["audio_feat"], original["audio_feat"])


def test_load_does_not_need_a_model(tmp_path):
    """load_voice must not reach for self.model — a cache is not model state,
    and reading the model's device is what pulled it onto the GPU before."""
    s = _synth()
    s.save_voice(_cache(), tmp_path / "v.pt")
    assert not hasattr(s, "model")
    s.load_voice(tmp_path / "v.pt")     # would AttributeError if it did


def test_save_creates_parent_dirs(tmp_path):
    s = _synth()
    out = s.save_voice(_cache(), tmp_path / "nested" / "dir" / "v.pt")
    assert out.exists()
