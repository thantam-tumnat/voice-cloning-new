"""Entry point: LoRA fine-tune VoxCPM 2 on the SiangTTS mix.

Mirrors the mechanics of VoxCPM's official `scripts/train_voxcpm_finetune.py`
(v2.0.3) with three SiangTTS-specific changes:

1. **Per-source weighted sampling** — `manifests.train[].weight` controls each
   source's share of the effective batch (e.g. ~20% LibriTTS-R EN).
2. **DataLoader-time text augmentation** — text is tokenized inside
   `__getitem__` *after* `src.augment` runs, so each epoch sees fresh spellings
   of the same audio. (The official script pre-tokenizes with `ds.map`, which
   would freeze one spelling forever.)
3. **MonitorBundle** — TensorBoard + in-training audio snapshots + timing JSON.

Checkpoints use VoxCPM's LoRA layout (`lora_weights.safetensors` +
`lora_config.json`) so they load directly via
`voxcpm.VoxCPM(..., lora_weights_path=...)` or `voxcpm clone --lora-path ...`.

`--dry-run` exercises datasets + monitor callbacks with a stub synth and no
voxcpm/GPU dependency.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
from pathlib import Path

import yaml

# Make `python train/train_lora.py` work without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.callbacks import MonitorBundle  # noqa: E402
from train.dataset import SiangTTSDataset  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_datasets(cfg: dict) -> tuple[SiangTTSDataset, SiangTTSDataset]:
    """Train uses per-source weights from YAML. Val ignores weights (uniform)."""
    train_sources = cfg["manifests"]["train"]
    val_sources = [{"path": m["path"], "weight": 1.0} for m in cfg["manifests"]["val"]]
    train_ds = SiangTTSDataset.from_sources(
        train_sources, is_train=True, augment_cfg=cfg.get("augment")
    )
    val_ds = SiangTTSDataset.from_sources(
        val_sources, is_train=False, augment_cfg=cfg.get("augment")
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Checkpointing (VoxCPM-compatible LoRA layout)
# ---------------------------------------------------------------------------

# Pretrained-dir files copied into full-SFT checkpoints so each one is a
# standalone model dir loadable by `VoxCPM(voxcpm_model_path=...)`.
_PRETRAINED_FILES = (
    "config.json",
    "audiovae.pth",
    "audiovae.safetensors",
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    save_dir: Path,
    step: int,
    pretrained_path: str,
    keep_last_n: int = 3,
) -> Path:
    import shutil

    import torch
    from safetensors.torch import save_file

    folder = save_dir / f"step_{step:07d}"
    folder.mkdir(parents=True, exist_ok=True)

    unwrapped = model.module if hasattr(model, "module") else model
    lora_cfg = unwrapped.lora_config

    if lora_cfg is not None:
        lora_state = {k: v for k, v in unwrapped.state_dict().items() if "lora_" in k}
        save_file(lora_state, folder / "lora_weights.safetensors")
        lora_info = {
            "base_model": str(pretrained_path),
            "lora_config": lora_cfg.model_dump() if hasattr(lora_cfg, "model_dump") else vars(lora_cfg),
        }
        with open(folder / "lora_config.json", "w", encoding="utf-8") as f:
            json.dump(lora_info, f, indent=2, ensure_ascii=False)
    else:
        # Full SFT: non-VAE weights + the config/tokenizer/VAE files from the
        # base snapshot, so the folder loads as a complete model.
        state = {
            k: v for k, v in unwrapped.state_dict().items() if not k.startswith("audio_vae.")
        }
        save_file(state, folder / "model.safetensors")
        for fname in _PRETRAINED_FILES:
            src = Path(pretrained_path) / fname
            if src.exists():
                shutil.copy2(src, folder / fname)

    torch.save(optimizer.state_dict(), folder / "optimizer.pth")
    torch.save(scheduler.state_dict(), folder / "scheduler.pth")
    with open(folder / "training_state.json", "w", encoding="utf-8") as f:
        json.dump({"step": int(step)}, f)

    latest = save_dir / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(folder, latest)

    # Prune old numbered checkpoints (keep the newest `keep_last_n`) so a long
    # run can't fill the disk. `latest` is always a separate full copy.
    if keep_last_n and keep_last_n > 0:
        numbered = sorted(
            (d for d in save_dir.glob("step_*") if d.is_dir()),
            key=lambda d: int(d.name.split("_")[1]),
        )
        for old in numbered[:-keep_last_n]:
            shutil.rmtree(old, ignore_errors=True)
    return folder


def load_checkpoint(model, optimizer, scheduler, save_dir: Path) -> int:
    """Resume from `<save_dir>/latest` if present. Returns the resume step."""
    import torch
    from safetensors.torch import load_file

    unwrapped = model.module if hasattr(model, "module") else model
    latest = save_dir / "latest"
    weights = latest / (
        "lora_weights.safetensors" if unwrapped.lora_config is not None else "model.safetensors"
    )
    if not weights.exists():
        return 0

    unwrapped.load_state_dict(load_file(str(weights)), strict=False)
    print(f"Loaded weights from {weights}", file=sys.stderr)

    opt_path = latest / "optimizer.pth"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
    sched_path = latest / "scheduler.pth"
    if sched_path.exists():
        scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

    state_path = latest / "training_state.json"
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            return int(json.load(f).get("step", 0))
    return 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def resolve_pretrained_path(pretrained: str) -> str:
    """Local dir as-is; otherwise treat as an HF repo id and snapshot it."""
    if os.path.isdir(pretrained):
        return pretrained
    from huggingface_hub import snapshot_download

    print(f"Downloading base model snapshot: {pretrained}", file=sys.stderr)
    return snapshot_download(repo_id=pretrained)


def build_optimizer(model, optim_cfg: dict):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    name = (optim_cfg.get("optimizer") or "adamw").lower()
    kwargs = dict(
        lr=float(optim_cfg.get("lr", 1e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
    if name == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, **kwargs)
        except ImportError:
            print("bitsandbytes unavailable — falling back to AdamW", file=sys.stderr)
    return torch.optim.AdamW(params, **kwargs)


def run_training(cfg: dict, monitor: MonitorBundle, train_ds, val_ds) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    from voxcpm.model import VoxCPM2Model
    from voxcpm.model.voxcpm2 import LoRAConfig
    from voxcpm.training import Accelerator, BatchProcessor
    from voxcpm.training.data import HFVoxCPMDataset

    tcfg = cfg["train"]
    sample_rate = int(cfg["sample_rate"])
    batch_size = int(tcfg["batch_size"])
    grad_accum = max(int(tcfg.get("gradient_accumulation_steps", 1)), 1)
    num_epochs = int(tcfg.get("num_epochs", 1))
    lambdas = {str(k): float(v) for k, v in (tcfg.get("lambdas") or {}).items()} or {
        "loss/diff": 1.0,
        "loss/stop": 1.0,
    }
    max_grad_norm = float(tcfg.get("max_grad_norm", 0.0))
    save_dir = Path(cfg["paths"]["checkpoint_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(amp=True)
    if accelerator.world_size > 1:
        raise NotImplementedError(
            "SiangTTS's weighted sampler is single-GPU; use VoxCPM's official "
            "script for multi-GPU runs."
        )

    # `lora:` present → adapter training; absent/null → full SFT of everything
    # except the AudioVAE (from_local freezes the VAE in training mode either
    # way). Full SFT of VoxCPM2-2B needs ~40 GB VRAM — not a 3090 recipe; see
    # conf/voxcpm_sft.yaml.
    lora_section = cfg.get("lora") or None
    lora_config = LoRAConfig(**lora_section) if lora_section else None
    print(f"mode: {'LoRA' if lora_config else 'full SFT'}", file=sys.stderr)

    pretrained_path = resolve_pretrained_path(str(cfg["pretrained_path"]))
    model = VoxCPM2Model.from_local(
        pretrained_path,
        optimize=False,
        training=True,
        lora_config=lora_config,
    )

    expected_sr = model.audio_vae.sample_rate
    assert sample_rate == expected_sr, (
        f"sample_rate mismatch: config says {sample_rate}, AudioVAE encoder "
        f"expects {expected_sr}. Set sample_rate: {expected_sr} in the YAML."
    )

    tokenizer = model.text_tokenizer
    train_ds.attach_voxcpm(tokenizer, sample_rate)
    val_ds.attach_voxcpm(tokenizer, sample_rate)

    # Drop samples whose packed length would overflow max_batch_tokens/batch.
    max_batch_tokens = int(tcfg.get("max_batch_tokens", 0))
    if max_batch_tokens > 0:
        fps = model.audio_vae.sample_rate / model.audio_vae.hop_length
        max_len = max_batch_tokens // batch_size
        for name, ds in (("train", train_ds), ("val", val_ds)):
            lengths = ds.estimate_packed_lengths(fps, model.config.patch_size)
            keep = [i for i, length in enumerate(lengths) if length <= max_len]
            if len(keep) < len(ds):
                print(
                    f"[{name}] dropping {len(ds) - len(keep)}/{len(ds)} samples "
                    f"longer than {max_len} packed tokens",
                    file=sys.stderr,
                )
                ds.select(keep)

    num_workers = int(tcfg.get("num_workers", 2))
    steps_per_epoch = max(len(train_ds) // (batch_size * grad_accum), 1)
    total_steps = steps_per_epoch * num_epochs

    def make_train_loader() -> DataLoader:
        # Rebuilt each epoch: workers re-fork and pick up the rotated
        # augmentation seed from set_epoch().
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_ds.make_weighted_sampler(num_samples=len(train_ds)),
            num_workers=num_workers,
            collate_fn=HFVoxCPMDataset.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=HFVoxCPMDataset.collate_fn,
        pin_memory=True,
    )

    batch_processor = BatchProcessor(
        config=model.config,
        audio_vae=model.audio_vae,
        dataset_cnt=max(train_ds.dataset_cnt, val_ds.dataset_cnt),
        device=accelerator.device,
    )
    # The VAE lives in the BatchProcessor (encode) and is re-attached for
    # generation only; detach so the optimizer/DDP never see it.
    audio_vae_for_gen = model.audio_vae
    del model.audio_vae
    model = accelerator.prepare_model(model)
    unwrapped = accelerator.unwrap(model)
    unwrapped.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable/1e6:.1f}M / {total_params/1e6:.1f}M "
        f"({100*trainable/max(total_params,1):.2f}%)",
        file=sys.stderr,
    )

    optimizer = build_optimizer(model, cfg.get("optim", {}))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.get("optim", {}).get("warmup_steps", 100)),
        num_training_steps=total_steps,
    )

    start_step = load_checkpoint(model, optimizer, scheduler, save_dir)
    if start_step > 0:
        print(f"Resuming from step {start_step}", file=sys.stderr)

    # ---- in-training audio synth (MonitorBundle AudioSampler hook) ---------
    a_cfg = (cfg.get("monitor", {}) or {}).get("audio_sampler", {}) or {}
    synth_cfg = {
        "inference_timesteps": int(a_cfg.get("inference_timesteps", 10)),
        "cfg_value": float(a_cfg.get("cfg_value", 2.0)),
    }

    def synthesize(text: str, ref_audio: str | None = None):
        was_training = unwrapped.training
        try:
            unwrapped.eval()
            unwrapped.audio_vae = audio_vae_for_gen.to(torch.float32)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else contextlib.nullcontext()
            )
            with torch.no_grad(), autocast_ctx:
                wav = unwrapped.generate(
                    target_text=text,
                    reference_wav_path=ref_audio or "",
                    **synth_cfg,
                )
            return wav.cpu().float().numpy().flatten()
        finally:
            unwrapped.audio_vae = None
            unwrapped.train(was_training)

    # ---- validation ---------------------------------------------------------
    def run_validation(step: int, max_batches: int = 10) -> dict[str, float]:
        from collections import defaultdict

        unwrapped.eval()
        sums: dict[str, list] = defaultdict(list)
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if bi >= max_batches:
                    break
                processed = batch_processor(batch)
                with accelerator.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        processed["text_tokens"],
                        processed["text_mask"],
                        processed["audio_feats"],
                        processed["audio_mask"],
                        processed["loss_mask"],
                        processed["position_ids"],
                        processed["labels"],
                        progress=0.0,
                    )
                for key, value in outputs.items():
                    if key.startswith("loss/"):
                        sums[key].append(value.detach())
        unwrapped.train()

        metrics: dict[str, float] = {}
        total = 0.0
        for key, values in sums.items():
            mean = torch.stack(values).mean().item()
            metrics[f"val/{key}"] = mean
            total += lambdas.get(key, 1.0) * mean
        metrics["val/loss_total"] = total
        monitor.tb.add_scalars(metrics, step)
        monitor.timing.on_eval()
        return metrics

    # ---- graceful stop on signal --------------------------------------------
    # Set a flag and let the loop finish the current step, then break to the
    # normal save + DataLoader teardown path. (Saving + os._exit from inside the
    # handler orphans the worker processes, which keep GPU memory reserved.)
    stop_requested = {"flag": False}

    def _signal_handler(signum, frame):
        if not stop_requested["flag"]:
            print(
                f"Signal {signum} — finishing current step, then saving and exiting",
                file=sys.stderr,
            )
        stop_requested["flag"] = True

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # ---- main loop -----------------------------------------------------------
    monitor.timing.on_train_start(config_snapshot=cfg)

    data_epoch = start_step // steps_per_epoch
    train_ds.set_epoch(data_epoch)
    train_iter = iter(make_train_loader())

    def next_batch():
        nonlocal train_iter, data_epoch
        try:
            return next(train_iter)
        except StopIteration:
            data_epoch += 1
            train_ds.set_epoch(data_epoch)  # rotate augmentation seeds
            monitor.timing.on_epoch_end()
            train_iter = iter(make_train_loader())
            return next(train_iter)

    val_metrics: dict[str, float] = {}
    log_every = int(tcfg.get("log_every_steps", 50))
    eval_every = int(tcfg.get("eval_every_steps", 1000))
    save_every = int(tcfg.get("save_every_steps", 1000))

    steps_done = start_step
    for step in range(start_step, total_steps):
        if stop_requested["flag"]:
            print(f"Stopping at step {step} (signal received).", file=sys.stderr)
            break
        optimizer.zero_grad(set_to_none=True)

        loss_log: dict[str, float] = {}
        for _ in range(grad_accum):
            batch = next_batch()
            processed = batch_processor(batch)
            with accelerator.autocast(dtype=torch.bfloat16):
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=step / max(1, total_steps),
                )
            total_loss = 0.0
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    total_loss = total_loss + lambdas.get(key, 1.0) * value / grad_accum
                    loss_log[f"train/{key}"] = float(value.detach())
            accelerator.backward(total_loss)

        scaler = getattr(accelerator, "scaler", None)
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            unwrapped.parameters(), max_norm=max_grad_norm if max_grad_norm > 0 else 1e9
        )
        accelerator.step(optimizer)
        accelerator.update()
        scheduler.step()
        steps_done = step + 1

        scalars = monitor.timing.on_step_end(step) or {}
        if step % log_every == 0 or step == total_steps - 1:
            scalars.update(loss_log)
            scalars["train/lr"] = float(optimizer.param_groups[0]["lr"])
            scalars["train/grad_norm"] = float(grad_norm)
            scalars["train/epoch"] = step / steps_per_epoch
            monitor.tb.add_scalars(scalars, step)

        if step > 0 and step % eval_every == 0 or step == total_steps - 1:
            val_metrics = run_validation(step)

        if monitor.audio.should_run(step):
            monitor.audio.run(step, synthesize_fn=synthesize, tb=monitor.tb)

        if step > 0 and step % save_every == 0:
            ckpt = save_checkpoint(model, optimizer, scheduler, save_dir, step, pretrained_path)
            monitor.timing.on_checkpoint(step, ckpt, val_metrics=val_metrics)

    # Final save covers both normal completion and a graceful stop.
    ckpt = save_checkpoint(model, optimizer, scheduler, save_dir, steps_done, pretrained_path)
    monitor.timing.on_checkpoint(steps_done, ckpt, val_metrics=val_metrics)
    monitor.timing.on_train_end(final_metrics=val_metrics)
    monitor.tb.close()

    # Deterministically reap DataLoader worker processes so they don't orphan
    # and keep GPU/RAM reserved after we return.
    with contextlib.suppress(Exception):
        del train_iter
    print(f"done — final checkpoint at {ckpt} (step {steps_done})", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def _stub_synth(text: str, ref_audio: str | None = None):
    """Sine-wave placeholder synth so the audio sampler's wiring can be validated
    without VoxCPM. Generates ~1s of a 440 Hz tone at 48 kHz."""
    import numpy as np
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    wav = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return wav


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("conf/voxcpm_lora.yaml"))
    p.add_argument("--dry-run", action="store_true",
                   help="Exercise dataset + monitor callbacks without invoking the GPU trainer.")
    args = p.parse_args()

    cfg = load_config(args.config)
    train_ds, val_ds = build_datasets(cfg)
    print(f"train: {len(train_ds)} items   val: {len(val_ds)} items")
    if len(train_ds):
        print("first train sample:", train_ds[0])

    monitor = MonitorBundle.from_config(cfg)

    if args.dry_run:
        monitor.timing.on_train_start(config_snapshot=cfg)
        print("[dry-run] exercising 3 fake steps + 1 eval + 1 audio snapshot + 1 ckpt")
        for step in (1, 2, 1000):
            scalars = {"train/loss": 5.0 - step * 0.001}
            scalars.update(monitor.timing.on_step_end(step) or {})
            monitor.tb.add_scalars(scalars, step)
            if monitor.audio.should_run(step):
                monitor.audio.run(step, synthesize_fn=_stub_synth, tb=monitor.tb)
        monitor.timing.on_eval()
        monitor.tb.add_scalars({"val/loss": 4.5}, 1000)
        monitor.timing.on_checkpoint(1000, "checkpoints/dryrun", val_metrics={"val/loss": 4.5})
        monitor.timing.on_epoch_end()
        monitor.timing.on_train_end(final_metrics={"val/loss": 4.5})
        monitor.tb.close()
        print("[dry-run] OK — see TB log dir + training_summary.json under runs/")
        return

    run_training(cfg, monitor, train_ds, val_ds)


if __name__ == "__main__":
    main()
