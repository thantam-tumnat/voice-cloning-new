"""JSONL → VoxCPM-compatible dataset, with DataLoader-time augmentation hooks.

Bridges our manifest format and VoxCPM 2.0.3's training pipeline. Two modes:

- **Plain mode** (default): returns `{"audio_path", "text", ...}` dicts. Used by
  `--dry-run` and unit tests; needs no `voxcpm` install.
- **VoxCPM mode** (after `attach_voxcpm(tokenizer, sample_rate)`): returns the
  sample dict `voxcpm.training.data.HFVoxCPMDataset.collate_fn` expects —
  `text_ids` / `audio_array` / `ref_audio_array` / `dataset_id` (int) /
  `is_prompt`. Text is tokenized *inside* `__getitem__`, after augmentation, so
  each epoch sees fresh spellings — this is why we can't use VoxCPM's own
  `ds.map(tokenize)` flow, which freezes text once.

Augmentations from `src.augment` run inside `__getitem__` so each epoch produces
fresh text spellings of the same audio (RESEARCH.md §8.6.3).

Manifest row schema (Vaja-Thai):
    {"audio": "wavs/x.wav", "text": "...", "duration": 4.31,
     "speaker": "...", "dataset_id": "vaja_thai", "tier": 1,
     "ref_audio": "wavs/y.wav"?, "no_digit_aug": false?}

Manifest row schema (LibriTTS-R adds):
    "text_original": "...", "text_normalized": "..."
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.augment import (
    case_jitter,
    maybe_digitize_thai,
    pick_libritts_text,
    whitespace_jitter,
)

log = logging.getLogger(__name__)


class SiangTTSDataset(Dataset):
    """Multi-source JSONL dataset with per-source weights for the sampler.

    Construct via either:
        - `SiangTTSDataset(manifest_paths=[...])` — equal weighting (back-compat).
        - `SiangTTSDataset.from_sources([{"path": "...", "weight": 1.0}, ...])` —
          per-source weights honored by `make_weighted_sampler()`.

    The weights are *normalized*: `weight: 0.25` for one source out of two with
    `weight: 1.0` means the smaller source contributes 0.25 / 1.25 ≈ 20% of the
    effective batch.
    """

    def __init__(
        self,
        manifest_paths: list[str | Path] | None = None,
        *,
        sources: list[dict[str, Any]] | None = None,
        root_dirs: list[str | Path] | None = None,
        is_train: bool = True,
        augment_cfg: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        if (manifest_paths is None) == (sources is None):
            raise ValueError("Pass exactly one of `manifest_paths` or `sources`.")

        self.is_train = is_train
        self.augment_cfg = augment_cfg or {}
        self._base_seed = seed
        self.rows: list[dict[str, Any]] = []
        self.row_roots: list[Path] = []
        self.row_source_idx: list[int] = []
        self.source_weights: list[float] = []
        self._source_row_counts: list[int] = []

        if sources is not None:
            paths = [s["path"] for s in sources]
            self.source_weights = [float(s.get("weight", 1.0)) for s in sources]
        else:
            paths = list(manifest_paths)  # type: ignore[arg-type]
            self.source_weights = [1.0] * len(paths)

        roots = [Path(r) for r in (root_dirs or [Path(p).parent for p in paths])]
        if len(roots) != len(paths):
            raise ValueError("root_dirs length must match number of manifests")

        for src_idx, (path, root) in enumerate(zip(paths, roots)):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Manifest not found: {path} — did you run the prepare scripts?"
                )
            count = 0
            with open(path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{path}:{line_no} bad JSONL: {e}") from e
                    self.rows.append(row)
                    self.row_roots.append(root)
                    self.row_source_idx.append(src_idx)
                    count += 1
            self._source_row_counts.append(count)
            log.info("loaded %d rows from %s (weight=%.3f)", count, path, self.source_weights[src_idx])

        if not self.rows:
            raise ValueError("No rows loaded from any manifest.")

        # String dataset_id → contiguous int index (VoxCPM's packer indexes
        # tensors by dataset id). Order of first appearance, stable across runs
        # as long as manifest order is stable.
        self.dataset_id_map: dict[str, int] = {}
        for row in self.rows:
            ds_id = str(row.get("dataset_id", ""))
            if ds_id not in self.dataset_id_map:
                self.dataset_id_map[ds_id] = len(self.dataset_id_map)

        # VoxCPM mode (off until attach_voxcpm() is called)
        self._tokenizer = None
        self._sample_rate: int | None = None

    @classmethod
    def from_sources(
        cls,
        sources: list[dict[str, Any]],
        **kwargs: Any,
    ) -> "SiangTTSDataset":
        return cls(sources=sources, **kwargs)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def _rng_for(self, idx: int) -> random.Random:
        # Per-item RNG keeps augmentation deterministic given (seed, idx).
        # `set_epoch` rotates the seed so successive epochs see different spellings.
        return random.Random(self._base_seed * 10_000_019 + idx)

    def _augment_text(self, row: dict[str, Any], rng: random.Random) -> str:
        cfg = self.augment_cfg
        ds_id = row.get("dataset_id", "")
        no_aug = bool(row.get("no_digit_aug", False))

        if ds_id == "libritts_r":
            text = pick_libritts_text(
                row.get("text_original") or row.get("text", ""),
                row.get("text_normalized") or row.get("text", ""),
                p_normalized=cfg.get("en_text_original_vs_normalized", {}).get("p_normalized", 0.3),
                rng=rng,
            )
            text = case_jitter(text, p=cfg.get("case_jitter", {}).get("p", 0.1), rng=rng)
        else:
            text = row.get("text", "")
            if not no_aug:
                text = maybe_digitize_thai(
                    text,
                    p_full=cfg.get("thai_digit", {}).get("p_full", 0.4),
                    p_partial=cfg.get("thai_digit", {}).get("p_partial", 0.1),
                    rng=rng,
                )

        text = whitespace_jitter(
            text,
            p=cfg.get("whitespace_jitter", {}).get("p", 0.15),
            rng=rng,
        )
        return text

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        root = self.row_roots[idx]

        if self.is_train:
            rng = self._rng_for(idx)
            text = self._augment_text(row, rng)
        else:
            text = row.get("text") or row.get("text_normalized") or ""

        if self._tokenizer is not None:
            return self._voxcpm_item(row, root, text)

        item: dict[str, Any] = {
            "audio_path": str(root / row["audio"]),
            "text": text,
            "duration": row.get("duration"),
            "dataset_id": row.get("dataset_id"),
        }
        if row.get("ref_audio"):
            item["ref_audio_path"] = str(root / row["ref_audio"])
        return item

    # ------------------------------------------------------------------
    # VoxCPM mode
    # ------------------------------------------------------------------
    _REF_SENTINEL = [-100.0]  # matches voxcpm HFVoxCPMDataset._SENTINEL

    def attach_voxcpm(self, tokenizer: Any, sample_rate: int) -> None:
        """Switch to VoxCPM-format samples (see module docstring).

        `tokenizer` is the model's wrapped text tokenizer: callable
        `str -> list[int]`. `sample_rate` is the AudioVAE *encoder* rate.
        """
        self._tokenizer = tokenizer
        self._sample_rate = int(sample_rate)

    @property
    def dataset_cnt(self) -> int:
        return len(self.dataset_id_map)

    def _load_audio(self, path: str) -> "Any":
        import numpy as np
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self._sample_rate:
            import torch
            import torchaudio

            wav = (
                torchaudio.functional.resample(
                    torch.from_numpy(wav), sr, self._sample_rate
                )
                .numpy()
            )
        return np.ascontiguousarray(wav, dtype=np.float32)

    def _voxcpm_item(self, row: dict[str, Any], root: Path, text: str) -> dict[str, Any]:
        sample = {
            "text_ids": self._tokenizer(text),
            "audio_array": self._load_audio(str(root / row["audio"])),
            "audio_sampling_rate": self._sample_rate,
            "dataset_id": self.dataset_id_map[str(row.get("dataset_id", ""))],
            "is_prompt": False,
        }
        ref = row.get("ref_audio")
        sample["ref_audio_array"] = (
            self._load_audio(str(root / ref)) if ref else self._REF_SENTINEL
        )
        return sample

    def estimate_packed_lengths(self, audio_vae_fps: float, patch_size: int) -> list[int]:
        """Estimated packed sequence length per row, mirroring
        `voxcpm.training.data.compute_sample_lengths`, for max_batch_tokens
        filtering. Uses the *un-augmented* manifest text; augmentation changes
        token counts only marginally (digits are shorter than number words), so
        a small safety margin is added.
        """
        import math

        if self._tokenizer is None:
            raise RuntimeError("attach_voxcpm() first — token lengths need the tokenizer")

        margin = 8
        lengths: list[int] = []
        for row in self.rows:
            text = row.get("text") or row.get("text_normalized") or ""
            text_len = len(self._tokenizer(text))
            duration = float(row.get("duration") or 0.0)
            t_seq = math.ceil(math.ceil(duration * audio_vae_fps) / patch_size)
            ref_seq = 0
            if row.get("ref_audio"):
                ref_dur = float(row.get("ref_duration") or duration)
                ref_seq = math.ceil(math.ceil(ref_dur * audio_vae_fps) / patch_size)
            overhead = 4 if ref_seq > 0 else 2
            lengths.append(text_len + t_seq + ref_seq + overhead + margin)
        return lengths

    def select(self, indices: list[int]) -> None:
        """Keep only `indices` (in-place), preserving source bookkeeping."""
        keep = set(indices)
        self.rows = [r for i, r in enumerate(self.rows) if i in keep]
        self.row_roots = [r for i, r in enumerate(self.row_roots) if i in keep]
        self.row_source_idx = [s for i, s in enumerate(self.row_source_idx) if i in keep]
        counts = [0] * len(self.source_weights)
        for s in self.row_source_idx:
            counts[s] += 1
        self._source_row_counts = counts
        if not self.rows:
            raise ValueError("select() removed every row — check max_batch_tokens.")

    def set_epoch(self, epoch: int) -> None:
        """Trainer should call this at start of each epoch to rotate augmentation."""
        self._base_seed = epoch

    # ------------------------------------------------------------------
    def per_row_weights(self) -> list[float]:
        """Per-row sampling weights so each *source* contributes its `weight`
        fraction of the effective batch, regardless of source row count.

        For source `s` with `n_s` rows and configured weight `w_s`, every row in
        that source gets weight `w_s / n_s`. Normalization is handled by
        `WeightedRandomSampler` (it accepts unnormalized weights).
        """
        per_source = [
            (w / n if n else 0.0)
            for w, n in zip(self.source_weights, self._source_row_counts)
        ]
        return [per_source[s] for s in self.row_source_idx]

    def make_weighted_sampler(
        self,
        num_samples: int | None = None,
        replacement: bool = True,
        generator=None,
    ):
        """Build a `torch.utils.data.WeightedRandomSampler` honoring source weights."""
        from torch.utils.data import WeightedRandomSampler

        weights = self.per_row_weights()
        if num_samples is None:
            num_samples = len(self.rows)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )


__all__ = ["SiangTTSDataset"]
