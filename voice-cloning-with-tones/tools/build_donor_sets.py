"""Build **same-person** emotion donor sets from the thai-ser dataset.

The per-gender donor clips used by the pipeline (``ref/emotions/female/<emotion>_1.wav`` etc.)
were each copied from the *first* clip of that emotion, so the 5 emotions almost certainly come
from *different* actors. That entangles "emotion" with "speaker identity" during F5 prosody
transfer.

This tool streams ``airesearch/thai-ser`` parquet shards, groups clips by ``actor_id``, and for
every actor that provides **all 5 emotions** it picks the best clip per emotion (highest
inter-annotator ``agreement``, duration within bounds). Each qualifying actor becomes one donor
set on disk:

    ref/emotions/<gender>_<actor_id>/<emotion>_1.wav   (24 kHz mono PCM16)
    ref/emotions/<gender>_<actor_id>/<emotion>_1.txt   (transcript sidecar)

plus a rich ``ref/emotions/donors_manifest.json`` describing every set (actor_id, gender, and
per-emotion text / audio_id / agreement / seconds) for future reuse.

Existing donor folders are never deleted; new actor folders are additive.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download

SUPPORTED_EMOTIONS: Tuple[str, ...] = ("neutral", "angry", "happy", "sad", "frustrated")


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


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_") or "unknown"


def build_donor_sets(
    sets_per_gender: int = 5,
    donor_dir: Path = Path("ref/emotions"),
    dataset_name: str = "airesearch/thai-ser",
    mic: str = "mic_clip",
    min_agreement: float = 0.6,
    min_seconds: float = 2.0,
    max_seconds: float = 8.0,
    target_sr: int = 24000,
    max_shards: int = 29,
) -> int:
    donor_dir.mkdir(parents=True, exist_ok=True)

    # best[(gender, actor_id)][emotion] = {"y", "sr", "agreement", "sec", "text", "audio_id"}
    best: Dict[Tuple[str, str], Dict[str, dict]] = defaultdict(dict)

    print(
        f"[Donors] Building same-person donor sets from {dataset_name} "
        f"(mic={mic}, target={sets_per_gender}/gender, min_agreement={min_agreement})"
    )
    t0 = time.time()
    total_processed = 0

    def _full_actor_counts() -> Dict[str, int]:
        counts = {"male": 0, "female": 0}
        for (gender, _actor), emos in best.items():
            if all(e in emos for e in SUPPORTED_EMOTIONS) and gender in counts:
                counts[gender] += 1
        return counts

    for shard_idx in range(max_shards):
        counts = _full_actor_counts()
        if counts["male"] >= sets_per_gender and counts["female"] >= sets_per_gender:
            print("[Donors] Enough complete actors collected for both genders.")
            break

        shard_filename = f"data/train-{shard_idx:05d}-of-00029.parquet"
        print(
            f"\n[Donors] Shard {shard_idx + 1}/{max_shards}: {shard_filename} "
            f"(complete actors so far: male={counts['male']}, female={counts['female']}) ..."
        )
        try:
            shard_path = hf_hub_download(
                repo_id=dataset_name,
                filename=shard_filename,
                repo_type="dataset",
            )
            df = pd.read_parquet(shard_path)
            print(f"[Donors] Shard loaded: {len(df)} rows")
        except Exception as e:
            print(f"[Donors] Failed to download/read shard {shard_filename}: {e}", file=sys.stderr)
            break

        for _, row in df.iterrows():
            total_processed += 1

            gender_raw = str(row.get("actor_gender") or "").strip().lower()
            if gender_raw.startswith("m"):
                gender = "male"
            elif gender_raw.startswith("f"):
                gender = "female"
            else:
                continue

            actor_id = str(row.get("actor_id") or "").strip()
            if not actor_id or actor_id in ("nan", "None"):
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
            if not text or text in ("nan", "None"):
                continue

            key = (gender, actor_id)
            slot = best[key].get(emo)
            # Cheap reject before decoding: if we already have a strictly better take, skip.
            if slot is not None and agreement <= slot["agreement"]:
                continue

            aud_cell = row.get(mic)
            if not isinstance(aud_cell, dict) or "bytes" not in aud_cell:
                continue

            try:
                array, in_sr = sf.read(io.BytesIO(aud_cell["bytes"]), dtype="float32")
                y = _trim_silence(_to_mono_resampled(array, in_sr, target_sr), target_sr)
                sec = y.size / target_sr
            except Exception:
                continue

            if not (min_seconds <= sec <= max_seconds):
                continue

            best[key][emo] = {
                "y": y,
                "sr": target_sr,
                "agreement": round(agreement, 3),
                "sec": round(sec, 2),
                "text": text,
                "audio_id": str(row.get("audio_id") or ""),
            }

        counts = _full_actor_counts()
        print(
            f"[Donors] After shard {shard_idx + 1}: processed={total_processed}, "
            f"complete actors male={counts['male']}, female={counts['female']} "
            f"({time.time() - t0:.0f}s)"
        )

    # Select the best complete actors per gender (rank by mean agreement across the 5 emotions).
    complete: Dict[str, List[Tuple[str, float, Dict[str, dict]]]] = {"male": [], "female": []}
    for (gender, actor_id), emos in best.items():
        if gender not in complete:
            continue
        if not all(e in emos for e in SUPPORTED_EMOTIONS):
            continue
        mean_agree = sum(emos[e]["agreement"] for e in SUPPORTED_EMOTIONS) / len(SUPPORTED_EMOTIONS)
        complete[gender].append((actor_id, mean_agree, emos))

    manifest_sets: List[dict] = []
    written = {"male": 0, "female": 0}
    for gender in ("female", "male"):
        chosen = sorted(complete[gender], key=lambda t: t[1], reverse=True)[:sets_per_gender]
        for actor_id, mean_agree, emos in chosen:
            set_id = f"{gender}_{_slug(actor_id)}"
            set_dir = donor_dir / set_id
            set_dir.mkdir(parents=True, exist_ok=True)

            emo_manifest: Dict[str, dict] = {}
            for emo in SUPPORTED_EMOTIONS:
                info = emos[emo]
                wav_path = set_dir / f"{emo}_1.wav"
                txt_path = set_dir / f"{emo}_1.txt"
                sf.write(str(wav_path), info["y"], target_sr, subtype="PCM_16")
                txt_path.write_text(info["text"], encoding="utf-8")
                emo_manifest[emo] = {
                    "file": f"{emo}_1.wav",
                    "text": info["text"],
                    "audio_id": info["audio_id"],
                    "agreement": info["agreement"],
                    "sec": info["sec"],
                }

            manifest_sets.append({
                "id": set_id,
                "actor_id": actor_id,
                "gender": gender,
                "mean_agreement": round(mean_agree, 3),
                "emotions": emo_manifest,
            })
            written[gender] += 1
            print(f"[Donors] Wrote set {set_id} (mean_agreement={mean_agree:.3f}) -> {set_dir}")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": dataset_name,
        "sample_rate": target_sr,
        "params": {
            "mic": mic,
            "min_agreement": min_agreement,
            "min_seconds": min_seconds,
            "max_seconds": max_seconds,
            "sets_per_gender": sets_per_gender,
        },
        "sets": manifest_sets,
    }
    manifest_path = donor_dir / "donors_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\n[Donors] Done. Wrote {written['female']} female + {written['male']} male same-person "
        f"donor sets to {donor_dir}\n[Donors] Manifest: {manifest_path}"
    )
    if written["female"] < sets_per_gender or written["male"] < sets_per_gender:
        print(
            "[Donors] NOTE: fewer sets than requested were found. "
            "Increase --max-shards or lower --min-agreement for more.",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sets-per-gender", type=int, default=5, help="Same-person sets to build per gender")
    p.add_argument("--donor-dir", type=Path, default=Path("ref/emotions"), help="Donor reference dir")
    p.add_argument("--min-agreement", type=float, default=0.6, help="Min inter-annotator agreement")
    p.add_argument("--min-seconds", type=float, default=2.0)
    p.add_argument("--max-seconds", type=float, default=8.0)
    p.add_argument("--max-shards", type=int, default=29, help="Max thai-ser shards to scan (of 29)")
    args = p.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    return build_donor_sets(
        sets_per_gender=args.sets_per_gender,
        donor_dir=args.donor_dir,
        min_agreement=args.min_agreement,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        max_shards=args.max_shards,
    )


if __name__ == "__main__":
    raise SystemExit(main())
