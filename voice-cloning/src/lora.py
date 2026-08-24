"""Live LoRA strength control for the shared VoxCPM2 model.

Ported from voice-cloning-with-tones/app/services/siangtts_service.py, because with
one model serving both pipelines the scale is no longer a process-wide setting: the
webhook path wants the adapter at its shipped strength, the tone studio wants the
DiT side at zero (see `LORA_MODES` below), and the GPU worker has to switch between
them job by job.

Everything here is a no-op on a model without LoRA layers, so the base-model-only
deployment keeps working.
"""

from __future__ import annotations

import sys
from typing import Any

# Named strengths, so a client can ask for a behaviour rather than two floats it
# would have to keep in sync with this repo.
#
#   shipped/legacy — what `from_pretrained` produces: r=64 alpha=128 on both sides.
#                    This is what the webhook path has always run at.
#   tones          — LM at full strength, DiT at zero. Measured in the tone studio
#                    (tools/expr_sweep.py --stage lora, 5 emotions x 3 paired reps):
#                    at the shipped strength "[angry]" comes out 3.1 dB *quieter*
#                    and 3.6 semitones *lower* than a neutral read. Zeroing the DiT
#                    side turns that into +1.2 dB / +1.9 st and roughly triples the
#                    angry-vs-sad contrast, while the LM side still carries the Thai.
#   off            — base model behaviour with the adapter loaded but inert.
LORA_MODES: dict[str, tuple[float, float]] = {
    "shipped": (2.0, 2.0),
    "legacy": (2.0, 2.0),
    "on": (2.0, 2.0),
    "tones": (2.0, 0.0),
    "off": (0.0, 0.0),
}

DEFAULT_MODE = "shipped"


def resolve(spec: Any) -> tuple[float, float]:
    """Turn a request's `lora` field into (lm, dit).

    Accepts a mode name, `{"mode": "..."}`, or explicit `{"lm": .., "dit": ..}`.
    Anything unrecognised falls back to the shipped strength rather than raising —
    a mistyped mode should not fail a job that would otherwise be fine.
    """
    if spec is None:
        return LORA_MODES[DEFAULT_MODE]
    if isinstance(spec, str):
        return LORA_MODES.get(spec.strip().lower(), LORA_MODES[DEFAULT_MODE])
    if isinstance(spec, dict):
        if spec.get("lm") is not None and spec.get("dit") is not None:
            return float(spec["lm"]), float(spec["dit"])
        return LORA_MODES.get(
            str(spec.get("mode", DEFAULT_MODE)).strip().lower(), LORA_MODES[DEFAULT_MODE]
        )
    return LORA_MODES[DEFAULT_MODE]


def set_lora_strength(tts_model: Any, lm: float, dit: float) -> dict:
    """Scale the loaded Thai LoRA independently on the LM and the DiT side.

    VoxCPM2 injects LoRA as ``LoRALinear`` layers that keep their strength in a
    ``scaling`` buffer rather than baked into the weights, so this is a live dial,
    not a reload. Nothing on a layer says which side it belongs to — both sides were
    injected with the same rank and alpha — so the split comes from where the layer
    lives: the LM is base_lm + residual_lm, the DiT is the feature decoder's
    estimator.

    Returns what was actually applied. A model with no LoRA layers returns zero
    counts and is otherwise untouched.
    """
    if tts_model is None:
        return {"lm": lm, "dit": dit, "lm_layers": 0, "dit_layers": 0}

    try:
        from voxcpm.modules.layers.lora import LoRALinear
    except Exception as e:  # pragma: no cover — depends on the installed voxcpm
        print(f"[lora] cannot scale ({e}); leaving it at its shipped strength.", file=sys.stderr)
        return {"lm": lm, "dit": dit, "lm_layers": 0, "dit_layers": 0}

    def apply(root: Any, value: float) -> int:
        n = 0
        for module in root.modules():
            if isinstance(module, LoRALinear):
                module.scaling.fill_(value)
                n += 1
        return n

    n_lm = apply(tts_model.base_lm, lm) + apply(tts_model.residual_lm, lm)
    n_dit = apply(tts_model.feat_decoder.estimator, dit)
    return {"lm": lm, "dit": dit, "lm_layers": n_lm, "dit_layers": n_dit}


__all__ = ["LORA_MODES", "DEFAULT_MODE", "resolve", "set_lora_strength"]
