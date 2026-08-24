"""Measure what the Thai LoRA does to emotion control, with it on and with it off.

Style control lives in VoxCPM2's base weights; a LoRA trained on Thai speech
overwrites part of them, and the suspicion this tool exists to settle is that it
also washes out the model's instruction-following. So: one sentence, one seed
voice, four directions, rendered under each condition and measured.

The takes are joined with a fixed gap and measured at known boundaries rather than
by silence detection -- a flat, unemotional read can trail off quietly enough to be
cut in two, which would silently shift every label.

Raw model output is measured, before audio_post's per-tone gain and stretch, so
what is being judged is the model rather than our own post-processing.

Usage:
    python tools/lora_ab.py --lora on  --out scratch/
    python tools/lora_ab.py --lora off --out scratch/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf

from app.renderers.voxcpm import resolve_style_tag
from app.services.siangtts_service import siangtts_service
from tools.prosody_eval import analyze

SENTENCE = (
    "ทั้งสองไฟล์มีโครงสร้าง เนื้อหา "
    "และผลการตรวจจับทางดิจิทัลที่แทบจะถอดรหัสออกมาเหมือนกันทุกประการครับ"
)

# Neutral leads so it becomes the yardstick the rest are read against. "crying" and
# the free-form mixed direction are the two the studio actually disagreed about.
DIRECTIONS = [
    ("neutral", None),
    ("sad", "sad"),
    ("crying", "crying"),
    ("scared", "scared"),
    ("mixed", "scared and crying, tearful"),
]

GAP_S = 1.0


def build_texts() -> list[str]:
    texts = []
    for _, tag in DIRECTIONS:
        instruction = resolve_style_tag(tag).instruction if tag else None
        texts.append(f"{instruction}{SENTENCE}" if instruction else SENTENCE)
    return texts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", choices=("on", "off"), required=True)
    ap.add_argument("--speaker", default=None, help="pin a cloned voice; default is the auto seed")
    ap.add_argument("--out", default="scratch")
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args(argv)

    if args.lora == "off":
        siangtts_service.adapter_path = ""
        siangtts_service._synthesizer = None

    texts = build_texts()
    for label, _ in DIRECTIONS:
        print(f"  rendering {label} ...", flush=True)

    chunks, sr = siangtts_service.render_chunks(
        texts,
        speaker_id=args.speaker,
        cfg_value=args.cfg,
        inference_timesteps=args.steps,
    )
    if len(chunks) != len(DIRECTIONS):
        # The splitter would break the mapping between take and label.
        raise SystemExit(f"expected {len(DIRECTIONS)} takes, got {len(chunks)}")

    gap = np.zeros(int(sr * GAP_S), dtype="float32")
    pieces, boundaries, cursor = [], [], 0.0
    for i, chunk in enumerate(chunks):
        if i:
            pieces.append(gap)
            cursor += GAP_S
        pieces.append(chunk.audio)
        dur = len(chunk.audio) / sr
        boundaries.append((cursor, cursor + dur))
        cursor += dur

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"lora_{args.lora}.wav"
    sf.write(str(wav_path), np.concatenate(pieces), sr)

    labels = [name for name, _ in DIRECTIONS]
    result = analyze(str(wav_path), labels=labels, boundaries=boundaries)
    json_path = out_dir / f"lora_{args.lora}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nlora={args.lora}  sr={sr}  ->  {wav_path}")
    print(f"{'label':<9}{'dur_ratio':>10}{'energy_db':>11}{'f0_offset_st':>14}{'f0_spread_st':>14}")
    for u in result["utterances"]:
        print(f"{u['label']:<9}{u['dur_ratio']:>10.3f}{u['energy_db']:>11.2f}"
              f"{u['f0_offset_st']:>14.2f}{u['f0_spread_st']:>14.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
