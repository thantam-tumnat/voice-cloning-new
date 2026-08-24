"""Training-time callbacks: TensorBoard logging, in-training audio eval, timing.

Three independent components (`TBLogger`, `AudioSampler`, `TimingTracker`) bundled
into one `MonitorBundle` for convenience. The trainer calls hooks at well-defined
points; if a component is disabled in the config it becomes a no-op.

Hooks:
    on_train_start()
    on_step_end(step, scalars: dict)
    on_eval(step, val_metrics: dict)
    on_checkpoint(step, ckpt_dir)
    on_train_end()
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:                                                 # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------

class TBLogger:
    """Thin SummaryWriter wrapper. Scalars + audio + (optional) text/hparams."""

    def __init__(self, log_dir: str | Path, flush_every_steps: int = 100, enabled: bool = True):
        self.enabled = enabled and (SummaryWriter is not None)
        self.flush_every_steps = flush_every_steps
        self._writer = None
        if self.enabled:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))

    def add_scalars(self, scalars: dict[str, float], step: int) -> None:
        if not self._writer:
            return
        for k, v in scalars.items():
            self._writer.add_scalar(k, float(v), step)
        if step % self.flush_every_steps == 0:
            self._writer.flush()

    def add_audio(self, tag: str, waveform, sample_rate: int, step: int) -> None:
        """waveform: 1D numpy array or torch tensor in [-1, 1]."""
        if not self._writer:
            return
        import numpy as np
        import torch
        if isinstance(waveform, np.ndarray):
            wav = torch.from_numpy(waveform).float()
        else:
            wav = waveform.float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        self._writer.add_audio(tag, wav, global_step=step, sample_rate=sample_rate)

    def add_text(self, tag: str, text: str, step: int) -> None:
        if self._writer:
            self._writer.add_text(tag, text, step)

    def add_hparams(self, hparams: dict[str, Any], metrics: dict[str, float]) -> None:
        if self._writer:
            self._writer.add_hparams(hparams, metrics)

    def close(self) -> None:
        if self._writer:
            self._writer.flush()
            self._writer.close()


# ---------------------------------------------------------------------------
# Audio sampler — synthesize a fixed prompt set during training, log to TB
# ---------------------------------------------------------------------------

class AudioSampler:
    """Periodically synthesize a small fixed prompt set, push to TensorBoard Audio tab.

    The trainer supplies `synthesize_fn(text, ref_audio) -> np.ndarray` so this class
    stays decoupled from the specific VoxCPM API surface.
    """

    def __init__(
        self,
        prompts_path: str | Path,
        every_steps: int = 1000,
        sample_rate: int = 48000,
        max_prompts: int = 6,
        out_dir: str | Path | None = None,
        enabled: bool = True,
        strict: bool = False,
    ) -> None:
        self.enabled = enabled
        self.every_steps = every_steps
        self.sample_rate = sample_rate
        self.max_prompts = max_prompts
        self.out_dir = Path(out_dir) if out_dir else None
        self.prompts: list[dict[str, str]] = []
        path = Path(prompts_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    self.prompts.append({k: (v or "").strip() for k, v in row.items()})
            self.prompts = self.prompts[: self.max_prompts]
            if not self.prompts:
                msg = f"[AudioSampler] prompts file is empty: {path}"
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg, stacklevel=2)
                log.warning(msg)
                self.enabled = False
        elif self.enabled:
            msg = (
                f"[AudioSampler] prompts file not found: {path}. "
                "In-training audio sampling will be DISABLED — TensorBoard 'Audio' tab "
                "will be empty. Create the file or set monitor.audio_sampler.enabled=false "
                "in the config to silence this warning."
            )
            if strict:
                raise FileNotFoundError(msg)
            warnings.warn(msg, stacklevel=2)
            log.warning(msg)
            self.enabled = False

    def should_run(self, step: int) -> bool:
        return self.enabled and step > 0 and step % self.every_steps == 0

    def run(
        self,
        step: int,
        synthesize_fn: Callable[..., Any],
        tb: TBLogger | None = None,
    ) -> None:
        """Synthesize each prompt, log to TB, and optionally save WAVs to `out_dir`."""
        if not self.enabled:
            return
        import soundfile as sf

        for row in self.prompts:
            pid = row["id"]
            text = row["text"]
            ref = row.get("ref_audio") or None
            try:
                wav = synthesize_fn(text=text, ref_audio=ref)
            except Exception as e:
                print(f"[AudioSampler] step={step} prompt={pid} synth failed: {e}")
                continue

            if tb is not None:
                tb.add_audio(f"audio/{pid}", wav, self.sample_rate, step)
                tb.add_text(f"audio/{pid}_text", text, step)

            if self.out_dir is not None:
                snap_dir = self.out_dir / f"step_{step:08d}"
                snap_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(snap_dir / f"{pid}.wav"), wav, self.sample_rate)


# ---------------------------------------------------------------------------
# Timing tracker — wall-clock + step throughput + GPU info → JSON for publication
# ---------------------------------------------------------------------------

@dataclass
class TimingSummary:
    started_at: str = ""
    ended_at: str = ""
    total_seconds: float = 0.0
    total_steps: int = 0
    steps_per_second_overall: float = 0.0
    seconds_per_step_overall: float = 0.0
    epochs_completed: int = 0
    eval_runs: int = 0
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)


class TimingTracker:
    """Wall-clock + per-step throughput + GPU info, written out as JSON.

    Every checkpoint records {step, wall_clock_s, vram_peak_gb, val_metrics}.
    Final summary is suitable for paper/README/HF model card use.
    """

    def __init__(
        self,
        summary_path: str | Path,
        log_gpu_stats: bool = True,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.summary_path = Path(summary_path)
        self.log_gpu_stats = log_gpu_stats
        self._t0: float = 0.0
        self._t_last_step: float = 0.0
        self._summary = TimingSummary()
        self._step_window: list[float] = []   # rolling window for live throughput
        self._window_max = 200

    # ---- lifecycle hooks ------------------------------------------------
    def on_train_start(self, config_snapshot: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self._t0 = time.time()
        self._t_last_step = self._t0
        self._summary.started_at = _iso_now()
        self._summary.hardware = _hardware_info(self.log_gpu_stats)
        self._summary.config_snapshot = config_snapshot or {}
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._flush()

    def on_step_end(self, step: int) -> dict[str, float]:
        if not self.enabled:
            return {}
        now = time.time()
        dt = now - self._t_last_step
        self._t_last_step = now
        self._step_window.append(dt)
        if len(self._step_window) > self._window_max:
            self._step_window.pop(0)
        self._summary.total_steps = step
        return {
            "timing/step_s": dt,
            "timing/steps_per_s": (1.0 / dt) if dt > 0 else 0.0,
            "timing/elapsed_s": now - self._t0,
            "timing/rolling_steps_per_s": _safe_div(len(self._step_window), sum(self._step_window)),
        }

    def on_eval(self) -> None:
        if self.enabled:
            self._summary.eval_runs += 1

    def on_epoch_end(self) -> None:
        if self.enabled:
            self._summary.epochs_completed += 1

    def on_checkpoint(
        self,
        step: int,
        ckpt_dir: str | Path,
        val_metrics: dict[str, float] | None = None,
        reset_vram_peak: bool = True,
    ) -> None:
        """Record a checkpoint snapshot. Resets the CUDA peak-VRAM counter by
        default so the *next* entry reflects only work done since this one.
        Disable with `reset_vram_peak=False` if you want a cumulative max."""
        if not self.enabled:
            return
        entry = {
            "step": step,
            "wall_clock_s": round(time.time() - self._t0, 2),
            "ckpt_dir": str(ckpt_dir),
            "vram_peak_gb": _gpu_peak_vram_gb() if self.log_gpu_stats else None,
            "val_metrics": val_metrics or {},
        }
        self._summary.checkpoints.append(entry)
        self._flush()
        if reset_vram_peak:
            _reset_gpu_peak_vram()

    def on_train_end(self, final_metrics: dict[str, float] | None = None) -> None:
        if not self.enabled:
            return
        end = time.time()
        total = end - self._t0
        self._summary.ended_at = _iso_now()
        self._summary.total_seconds = round(total, 2)
        self._summary.steps_per_second_overall = _safe_div(self._summary.total_steps, total)
        self._summary.seconds_per_step_overall = _safe_div(total, self._summary.total_steps)
        self._summary.final_metrics = final_metrics or {}
        self._flush()

    # ---- internals ------------------------------------------------------
    def _flush(self) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self._summary), f, indent=2, ensure_ascii=False)


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _hardware_info(log_gpu: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if log_gpu and torch.cuda.is_available():
            i = torch.cuda.current_device()
            info["gpu_name"] = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            info["gpu_total_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["cuda_version"] = torch.version.cuda
    except Exception as e:
        info["torch_error"] = repr(e)
    return info


def _gpu_peak_vram_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        peak = torch.cuda.max_memory_allocated()
        return round(peak / (1024 ** 3), 3)
    except Exception:
        return None


def _reset_gpu_peak_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class MonitorBundle:
    tb: TBLogger | None = None
    audio: AudioSampler | None = None
    timing: TimingTracker | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MonitorBundle":
        m = cfg.get("monitor", {}) or {}

        tb_cfg = m.get("tensorboard", {}) or {}
        tb = TBLogger(
            log_dir=tb_cfg.get("log_dir", "runs/run0"),
            flush_every_steps=int(tb_cfg.get("flush_every_steps", 100)),
            enabled=bool(tb_cfg.get("enabled", True)),
        )

        a_cfg = m.get("audio_sampler", {}) or {}
        audio = AudioSampler(
            prompts_path=a_cfg.get("prompts", "eval/prompts_listen.tsv"),
            every_steps=int(a_cfg.get("every_steps", 1000)),
            sample_rate=int(a_cfg.get("sample_rate", cfg.get("sample_rate", 48000))),
            max_prompts=int(a_cfg.get("max_prompts", 6)),
            out_dir=Path(tb_cfg.get("log_dir", "runs/run0")) / "audio_snapshots",
            enabled=bool(a_cfg.get("enabled", True)),
        )

        t_cfg = m.get("timing", {}) or {}
        timing = TimingTracker(
            summary_path=t_cfg.get("summary_path", "runs/run0/training_summary.json"),
            log_gpu_stats=bool(t_cfg.get("log_gpu_stats", True)),
            enabled=bool(t_cfg.get("enabled", True)),
        )

        return cls(tb=tb, audio=audio, timing=timing)


__all__ = ["TBLogger", "AudioSampler", "TimingTracker", "MonitorBundle"]
