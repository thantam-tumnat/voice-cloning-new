"""Publish a trained SiangTTS LoRA adapter to the HuggingFace Hub.

Stages a *clean* release (no optimizer/scheduler state), rewrites the LoRA
config's `base_model` to the public HF id so others can load it, writes a model
card (README.md) + LICENSE, optionally bundles sample audio, then pushes.

Usage:
    uv run python train/publish_to_hf.py \
        --repo-id dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA \
        --checkpoint checkpoints/siangtts-lora-v0/latest \
        --samples-dir demo/samples \
        --public
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

BASE_MODEL_ID = "openbmb/VoxCPM2"
LICENSE_ID = "cc-by-sa-4.0"   # most-restrictive training-data license (porjai CC-BY-SA-4.0)

MODEL_CARD = """---
license: {license_id}
base_model: {base_model}
library_name: voxcpm
pipeline_tag: text-to-speech
language:
- th
- en
tags:
- text-to-speech
- voice-cloning
- thai
- voxcpm
- lora
---

# SiangTTS — Thai Voice-Cloning TTS (VoxCPM2 LoRA)

**SiangTTS** (เสียง = *voice*) is a LoRA adapter for
[`{base_model}`]({base_model_url}) that gives it clear, natural **Thai** speech
with zero-shot **voice cloning**, trained on a single RTX 3090 (24 GB).

It's a parameter-efficient (LoRA) fine-tune of VoxCPM2 — small enough to train on
one consumer GPU while keeping the base model's voice-design and cloning abilities.
The Thai-adaptation approach is inspired by **JaiTTS** (a separate, closed-source
Thai VoxCPM model by others); SiangTTS is an independent open reproduction at LoRA
scale and is not affiliated with it.

🔊 **Listen / compare (ref vs ground-truth vs base vs SiangTTS):**
<https://dubbing-ai.github.io/VoxCPM-thai/> · Code: <https://github.com/dubbing-ai/VoxCPM-thai>

## Results

Measured with **Typhoon-Whisper-Large-v3** (Thai ASR; CER) and **WavLM** x-vectors
(speaker SIM) on small eval sets, so numbers are directional. CER is an *upper
bound on error* — the ASR judge itself mis-recognises some rare/archaic Thai
words the model pronounces correctly.

VoxCPM2's base is already a capable Thai speaker (it reads numerals and handles
long-form). **SiangTTS clones at essentially the real same-speaker similarity
ceiling and is as intelligible as the original recordings**, with much lower CER
than the base. The `GT` column is the real recording, included as the reference.

| Voice cloning (80 prompts) | GT (real rec.) | Base | **SiangTTS** |
|---|---|---|---|
| Intelligibility — CER ↓ | 0.97% | 3.26% | **0.84%** |
| Speaker similarity — SIM ↑ | 0.913 | 0.906 | **0.909** |

(CER ≤ GT reflects clean, ASR-friendly synthesis plus the ASR judge's own floor;
SIM ≈ GT means cloning is about as close as two real recordings of one speaker.)
Separately, short-form Thai CER 5.7%→3.8% and long-form 2.7%→1.6% (small 5 / 2
-prompt sets — directional).

Trained 2 epochs over ~205 h: Common Voice Thai (diverse speakers) +
porjai_central (studio-clean) + a LibriTTS-R English slice (retains English &
code-switching). Audio encoded at 16 kHz, generated at 48 kHz (VoxCPM2 design).

## Usage

```python
from voxcpm import VoxCPM
from voxcpm.model.voxcpm2 import LoRAConfig
import json

cfg = json.load(open("lora_config.json"))["lora_config"]
model = VoxCPM.from_pretrained(
    "{base_model}",
    lora_config=LoRAConfig(**cfg),
    lora_weights_path=".",        # dir holding lora_weights.safetensors
)

# Plain TTS
wav = model.generate(text="สวัสดีครับ ยินดีที่ได้รู้จัก", cfg_value=2.5, inference_timesteps=10)

# Voice cloning from a 3-10 s reference clip
wav = model.generate(text="ทดสอบการโคลนเสียง", reference_wav_path="ref.wav",
                     cfg_value=2.5, inference_timesteps=10)
```

Or via the CLI: `voxcpm clone --text "..." --reference-audio ref.wav --lora-path . -o out.wav`

## Limitations

- Rare archaic / liturgical Pali-Sanskrit vocabulary may occasionally be
  mispronounced (no phonetic-respelling dict is used — raw-text approach).
- Eval prompt sets are small; numbers are directional.
- Best for everyday/conversational Thai; not tuned for specific domains.

## License

**{license_id}** - inherited from the most restrictive training-data license
(porjai_central is CC-BY-SA-4.0; Common Voice is CC0; LibriTTS-R is CC-BY-4.0).
Commercial use is permitted under share-alike. The base model
[`{base_model}`]({base_model_url}) is Apache-2.0; SiangTTS code is Apache-2.0.
"""

LICENSE_TEXT = """SiangTTS LoRA adapter weights - Creative Commons Attribution-ShareAlike 4.0
International (CC-BY-SA-4.0).

This license is inherited as the most restrictive term among the training data:
  - porjai_central (via dubbing-ai/vaja-thai) ... CC-BY-SA-4.0  (share-alike)
  - Common Voice Thai ........................... CC0
  - LibriTTS-R .................................. CC-BY-4.0

Full text: https://creativecommons.org/licenses/by-sa/4.0/legalcode

The base model openbmb/VoxCPM2 is licensed Apache-2.0; the SiangTTS training and
inference code is Apache-2.0. Those terms are separate from these adapter weights.
"""


def build_release(checkpoint: Path, staging: Path) -> Path:
    """Assemble a clean release dir from a training checkpoint."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    weights = checkpoint / "lora_weights.safetensors"
    cfg_path = checkpoint / "lora_config.json"
    if not weights.exists() or not cfg_path.exists():
        raise FileNotFoundError(
            f"Expected lora_weights.safetensors + lora_config.json in {checkpoint}"
        )

    shutil.copy2(weights, staging / "lora_weights.safetensors")

    # Rewrite base_model from the local cache path to the public HF id.
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["base_model"] = BASE_MODEL_ID
    (staging / "lora_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    base_url = f"https://huggingface.co/{BASE_MODEL_ID}"
    (staging / "README.md").write_text(
        MODEL_CARD.format(license_id=LICENSE_ID, base_model=BASE_MODEL_ID, base_model_url=base_url),
        encoding="utf-8",
    )
    (staging / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    return staging


def push(
    repo_id: str,
    checkpoint: Path,
    samples_dir: Path | None = None,
    private: bool = True,
) -> None:
    staging = Path("dist/hf_release")
    build_release(checkpoint, staging)

    if samples_dir and samples_dir.exists():
        dst = staging / "samples"
        dst.mkdir(exist_ok=True)
        for wav in sorted(samples_dir.glob("*.wav")):
            shutil.copy2(wav, dst / wav.name)
        for j in samples_dir.glob("manifest.json"):
            shutil.copy2(j, dst / j.name)

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(repo_id=repo_id, folder_path=str(staging), path_in_repo=".")
    vis = "private" if private else "public"
    print(f"pushed {repo_id} ({vis}) - https://huggingface.co/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="e.g. dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/siangtts-lora-v0/latest"))
    p.add_argument("--samples-dir", type=Path, default=None)
    p.add_argument("--public", action="store_true", help="Push as public; default is private")
    args = p.parse_args()

    push(
        repo_id=args.repo_id,
        checkpoint=args.checkpoint,
        samples_dir=args.samples_dir,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
