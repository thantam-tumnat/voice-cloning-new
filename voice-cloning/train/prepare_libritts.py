"""Prepare LibriTTS-R → JSONL manifest @ 16 kHz for SiangTTS.

Two key choices (RESEARCH.md §8.3):

1. Carry BOTH `text_original` and `text_normalized` in the manifest. The DataLoader
   samples between them via `src.augment.pick_libritts_text` so the model sees raw
   digits ("1923") most of the time and spelled-out form some of the time.
2. Resample 24 kHz → 16 kHz at manifest write time (VoxCPM2's AudioVAE encoder
   rate) so the train loop doesn't need to.

Usage:
    uv run python train/prepare_libritts.py --output-dir data/libritts --subset train-clean-100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from train.audio_prep import resample_trim_save  # noqa: E402

TARGET_SR = 16000  # AudioVAE encoder input rate (VoxCPM2 decodes at 48 kHz)
REF_AUDIO_PROBABILITY = 0.4
DATASET_ID = "libritts_r"


def prepare(
    output_dir: Path,
    config: str = "clean",
    split: str = "train.clean.100",
    max_samples: int = 0,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> None:
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    # mythicinfinity/libritts_r configs: dev/clean/other/all; clean splits are
    # train.clean.100, train.clean.360, dev.clean, test.clean.
    ds = load_dataset("mythicinfinity/libritts_r", config, split=split, streaming=True)

    rng = random.Random(seed)
    rows: list[dict] = []
    speaker_to_clips: dict[str, list[str]] = defaultdict(list)
    seen = 0

    for sample in tqdm(ds, desc=f"streaming libritts-r/{split}"):
        if max_samples and seen >= max_samples:
            break

        text_original = (sample.get("text_original") or "").strip()
        text_normalized = (sample.get("text_normalized") or "").strip()
        if not text_original and not text_normalized:
            continue
        if not text_original:
            text_original = text_normalized
        if not text_normalized:
            text_normalized = text_original

        audio = sample["audio"]
        wav_id = f"libritts_{seen:08d}"
        wav_rel = f"wavs/{wav_id}.wav"
        duration = resample_trim_save(
            audio["array"], audio["sampling_rate"],
            wav_dir / f"{wav_id}.wav", target_sr=TARGET_SR,
        )
        if duration is None:
            continue   # outside [1, 30] s window after silence trim

        speaker = str(sample.get("speaker_id", "unknown"))
        speaker_to_clips[speaker].append(wav_rel)

        rows.append(
            {
                "audio": wav_rel,
                "text": text_original,            # default text the trainer reads
                "text_original": text_original,
                "text_normalized": text_normalized,
                "duration": round(duration, 3),
                "speaker": speaker,
                "dataset_id": DATASET_ID,
            }
        )
        seen += 1

    for r in rows:
        if rng.random() >= REF_AUDIO_PROBABILITY:
            continue
        candidates = [c for c in speaker_to_clips[r["speaker"]] if c != r["audio"]]
        if candidates:
            r["ref_audio"] = rng.choice(candidates)

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    for r in val_rows:
        r["no_digit_aug"] = True   # not Thai but harmless; signals "no aug"

    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "val.jsonl", val_rows)
    print(f"wrote {len(train_rows)} train / {len(val_rows)} val rows → {output_dir}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("data/libritts"))
    p.add_argument("--config", default="clean", help="HF config: dev/clean/other/all")
    p.add_argument("--split", default="train.clean.100",
                   help="e.g. train.clean.100, train.clean.360")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prepare(
        output_dir=args.output_dir,
        config=args.config,
        split=args.split,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
    # See prepare_vaja_thai.py: hard-exit to dodge a finalization-time crash in
    # datasets/torchaudio background threads after output is written.
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
