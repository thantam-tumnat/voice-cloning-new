"""Download and organize ~1GB of emotion speech dataset separated by male/female.

Downloads parquet shards from airesearch/thai-ser on HuggingFace Hub,
extracts clips, resamples to 24 kHz mono 16-bit PCM WAV,
saves transcript sidecars (.txt), and categorizes into:
  dataset/male/     (~500 MB)
  dataset/female/   (~500 MB)
and populates ref/emotions/male/ and ref/emotions/female/ with donor clips for
{neutral, angry, happy, sad, frustrated}.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download

SUPPORTED_EMOTIONS = ("neutral", "angry", "happy", "sad", "frustrated")


def _trim_silence(y: np.ndarray, sr: int, thresh_db: float = -40.0, pad_s: float = 0.05) -> np.ndarray:
    if y.size == 0:
        return y
    amp = np.abs(y)
    peak = float(amp.max())
    if peak <= 0:
        return y
    gate = peak * (10.0 ** (thresh_db / 20.0))
    voiced = np.where(amp >= gate)[0]
    if voiced.size == 0:
        return y
    pad = int(pad_s * sr)
    start = max(0, int(voiced[0]) - pad)
    end = min(y.size, int(voiced[-1]) + pad)
    return y[start:end]


def _to_mono_resampled(array: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    y = np.asarray(array, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0 if y.shape[0] < y.shape[1] else 1)
    if sr != target_sr:
        y = torchaudio.functional.resample(torch.from_numpy(y), sr, target_sr).numpy()
    return y


def download_dataset(
    target_mb_per_gender: float = 500.0,
    out_dir: Path = Path("dataset"),
    donor_ref_dir: Optional[Path] = Path("ref/emotions"),
    dataset_name: str = "airesearch/thai-ser",
    mic: str = "mic_clip",
    min_agreement: float = 0.5,
    min_seconds: float = 2.0,
    max_seconds: float = 12.0,
    target_sr: int = 24000,
    max_shards: int = 15,
) -> int:
    target_bytes_per_gender = int(target_mb_per_gender * 1024 * 1024)
    male_dir = out_dir / "male"
    female_dir = out_dir / "female"
    male_dir.mkdir(parents=True, exist_ok=True)
    female_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Dataset] Target: ~{target_mb_per_gender:.1f} MB male + ~{target_mb_per_gender:.1f} MB female (~{target_mb_per_gender*2/1024:.2f} GB total)")
    print(f"[Dataset] Downloading parquet shards from {dataset_name} (mic={mic}, target_sr={target_sr}) ...")

    bytes_saved = {"male": 0, "female": 0}
    counts = {"male": 0, "female": 0}
    manifests: Dict[str, List[dict]] = {"male": [], "female": []}
    per_emo_counts: Dict[str, Dict[str, int]] = {
        "male": {e: 0 for e in SUPPORTED_EMOTIONS},
        "female": {e: 0 for e in SUPPORTED_EMOTIONS},
    }

    t0 = time.time()
    total_processed = 0

    for shard_idx in range(max_shards):
        if bytes_saved["male"] >= target_bytes_per_gender and bytes_saved["female"] >= target_bytes_per_gender:
            print("[Dataset] Target size reached for both genders!")
            break

        shard_filename = f"data/train-{shard_idx:05d}-of-00029.parquet"
        print(f"\n[Dataset] Fetching shard {shard_idx + 1}/{max_shards}: {shard_filename} ...")
        try:
            shard_path = hf_hub_download(
                repo_id=dataset_name,
                filename=shard_filename,
                repo_type="dataset",
            )
            df = pd.read_parquet(shard_path)
            print(f"[Dataset] Shard loaded: {len(df)} rows")
        except Exception as e:
            print(f"[Dataset] Failed to download or read shard {shard_filename}: {e}", file=sys.stderr)
            break

        for _, row in df.iterrows():
            total_processed += 1
            if bytes_saved["male"] >= target_bytes_per_gender and bytes_saved["female"] >= target_bytes_per_gender:
                break

            gender_raw = str(row.get("actor_gender") or "").strip().lower()
            if gender_raw.startswith("m"):
                gender = "male"
            elif gender_raw.startswith("f"):
                gender = "female"
            else:
                continue

            if bytes_saved[gender] >= target_bytes_per_gender:
                continue

            emo = str(row.get("assigned_emo") or "").strip().lower()
            if emo not in SUPPORTED_EMOTIONS:
                continue

            try:
                agreement = float(row.get("agreement") or 0.0)
            except Exception:
                agreement = 0.0
            if agreement < min_agreement:
                continue

            text = str(row.get("script_sent") or "").strip()
            if not text or text == "nan" or text == "None":
                continue

            aud_cell = row.get(mic)
            if not isinstance(aud_cell, dict) or "bytes" not in aud_cell:
                continue

            try:
                audio_bytes = aud_cell["bytes"]
                array, in_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                y = _trim_silence(_to_mono_resampled(array, in_sr, target_sr), target_sr)
                sec = y.size / target_sr
            except Exception:
                continue

            if not (min_seconds <= sec <= max_seconds):
                continue

            target_folder = male_dir if gender == "male" else female_dir
            per_emo_counts[gender][emo] += 1
            emo_idx = per_emo_counts[gender][emo]
            stem = f"{emo}_{emo_idx:04d}"
            wav_path = target_folder / f"{stem}.wav"
            txt_path = target_folder / f"{stem}.txt"

            sf.write(str(wav_path), y, target_sr, subtype="PCM_16")
            txt_path.write_text(text, encoding="utf-8")

            file_size = wav_path.stat().st_size
            bytes_saved[gender] += file_size
            counts[gender] += 1

            entry = {
                "file": f"{stem}.wav",
                "audio_id": str(row.get("audio_id") or ""),
                "emotion": emo,
                "text": text,
                "actor_id": str(row.get("actor_id") or ""),
                "gender": gender,
                "agreement": round(agreement, 3),
                "sec": round(sec, 2),
                "bytes": file_size,
            }
            manifests[gender].append(entry)

            if (counts["male"] + counts["female"]) % 50 == 0 or (counts["male"] + counts["female"]) <= 10:
                mb_m = bytes_saved["male"] / (1024 * 1024)
                mb_f = bytes_saved["female"] / (1024 * 1024)
                elapsed = time.time() - t0
                print(f"[Dataset] Processed {total_processed} | Male: {counts['male']} ({mb_m:.1f} MB) | Female: {counts['female']} ({mb_f:.1f} MB) | {elapsed:.0f}s", flush=True)

    # Write manifests
    (male_dir / "manifest.json").write_text(
        json.dumps(manifests["male"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (female_dir / "manifest.json").write_text(
        json.dumps(manifests["female"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[Dataset] Download complete!")
    print(f"  Male:   {counts['male']} clips ({bytes_saved['male']/(1024*1024):.2f} MB) in {male_dir}")
    print(f"  Female: {counts['female']} clips ({bytes_saved['female']/(1024*1024):.2f} MB) in {female_dir}")
    print(f"  Total:  {(bytes_saved['male']+bytes_saved['female'])/(1024*1024):.2f} MB")

    # Set up donor clips in donor_ref_dir (ref/emotions/male and ref/emotions/female)
    if donor_ref_dir:
        for g, g_dir in (("male", male_dir), ("female", female_dir)):
            donor_g_dir = donor_ref_dir / g
            donor_g_dir.mkdir(parents=True, exist_ok=True)
            for emo in SUPPORTED_EMOTIONS:
                matches = sorted(g_dir.glob(f"{emo}_*.wav"))
                if matches:
                    src_wav = matches[0]
                    src_txt = src_wav.with_suffix(".txt")
                    dest_wav = donor_g_dir / f"{emo}_1.wav"
                    dest_txt = donor_g_dir / f"{emo}_1.txt"
                    shutil.copy2(src_wav, dest_wav)
                    if src_txt.exists():
                        shutil.copy2(src_txt, dest_txt)
            print(f"[Dataset] Copied reference donor clips for {g} -> {donor_g_dir}")

    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mb-per-gender", type=float, default=500.0, help="Target MB per gender (default: 500MB -> ~1GB total)")
    p.add_argument("--out-dir", type=Path, default=Path("dataset"), help="Output directory")
    p.add_argument("--donor-dir", type=Path, default=Path("ref/emotions"), help="Donor reference dir")
    args = p.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    return download_dataset(
        target_mb_per_gender=args.mb_per_gender,
        out_dir=args.out_dir,
        donor_ref_dir=args.donor_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
