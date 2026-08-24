import os
import sys
import json
import time
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def find_cached_snapshot(repo_id: str) -> str:
    """Find local cached snapshot for a Hugging Face repo if available."""
    org_repo = repo_id.replace("/", "--")
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org_repo}" / "snapshots"
    if hub_dir.exists():
        for snap in hub_dir.iterdir():
            if snap.is_dir():
                if (snap / "model.safetensors").exists() or (snap / "lora_weights.safetensors").exists():
                    return str(snap)
    return repo_id

def check_lora():
    print("=" * 65)
    print("[CHECK] VERIFYING THAI LORA STATUS IN SIANGTTS / VOXCPM2")
    print("=" * 65)

    # 1. Hardware & Framework Info
    print("\n[1] Environment & Hardware Info:")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  - Total VRAM: {total_mem:.2f} GB")

    # 2. Config & Adapter Resolution
    from app.config import settings
    from app.services.siangtts_service import SiangTTSService, _RealSynthesizer

    base_model_local = find_cached_snapshot(settings.siangtts_base_model)
    adapter_local = find_cached_snapshot(settings.siangtts_adapter)

    print("\n[2] Configuration & Model Paths:")
    print(f"  - siangtts_base_model (config): {settings.siangtts_base_model}")
    print(f"  - Local base model path: {base_model_local}")
    print(f"  - siangtts_adapter (config): {settings.siangtts_adapter}")
    print(f"  - Local adapter path: {adapter_local}")
    print(f"  - siangtts_device: {settings.siangtts_device or 'auto (GPU if available)'}")

    # Check files in adapter directory
    adapter_path = Path(adapter_local)
    files = list(adapter_path.iterdir())
    print(f"\n[3] Files in Thai LoRA Adapter Directory ({len(files)} files):")
    for f in sorted(files, key=lambda x: x.name):
        size_str = f"{f.stat().st_size / (1024*1024):.2f} MB" if f.stat().st_size > 1024*1024 else f"{f.stat().st_size / 1024:.1f} KB"
        print(f"    * {f.name:25s} ({size_str})")

    svc = SiangTTSService(base_model=base_model_local, adapter_path=adapter_local)
    lora_cfg = svc._load_lora_config(adapter_local)
    print(f"\n[4] Parsed LoRA Configuration:")
    print(f"  - Rank (r): {lora_cfg.r}")
    print(f"  - Scaling Alpha (alpha): {lora_cfg.alpha} (scaling factor = {lora_cfg.alpha / lora_cfg.r})")
    print(f"  - Dropout: {lora_cfg.dropout}")
    print(f"  - enable_lm: {lora_cfg.enable_lm}")
    print(f"  - enable_dit: {lora_cfg.enable_dit}")
    print(f"  - enable_proj: {lora_cfg.enable_proj}")
    print(f"  - target_modules_lm: {lora_cfg.target_modules_lm}")
    print(f"  - target_modules_dit: {lora_cfg.target_modules_dit}")

    # 3. Model Loading
    print("\n[5] Initializing Real Synthesizer and injecting LoRA...")
    t0 = time.time()
    synth = _RealSynthesizer(
        base_model=base_model_local,
        adapter_path=adapter_local,
        lora_config=lora_cfg,
        device=device_str,
        load_denoiser=False,
        optimize=False,
    )
    load_time = time.time() - t0
    print(f"  - Model loaded successfully in {load_time:.2f} seconds!")
    print(f"  - Synth sample rate: {synth.sample_rate} Hz")
    print(f"  - Synth lora_loaded flag: {getattr(synth, 'lora_loaded', False)}")

    tts_model = synth.tts_model

    # 4. LoRA Layer Inspection
    print("\n[6] Detailed Inspection of LoRA Layers in Neural Network:")
    from voxcpm.modules.layers.lora import LoRALinear

    lora_modules = []
    for name, module in tts_model.named_modules():
        if isinstance(module, LoRALinear):
            lora_modules.append((name, module))

    print(f"  - Total LoRALinear layers present: {len(lora_modules)}")

    # Inspect weights of LoRA layers
    non_zero_lora_count = 0
    total_lora_params = 0
    lm_lora_count = 0
    dit_lora_count = 0

    for name, mod in lora_modules:
        a_norm = mod.lora_A.norm().item()
        b_norm = mod.lora_B.norm().item()
        total_lora_params += mod.lora_A.numel() + mod.lora_B.numel()
        if a_norm > 1e-6 and b_norm > 1e-6:
            non_zero_lora_count += 1
        if "base_lm" in name or "residual_lm" in name:
            lm_lora_count += 1
        elif "feat_decoder" in name or "estimator" in name:
            dit_lora_count += 1

    print(f"    * Base LM / Residual LM LoRA layers: {lm_lora_count}")
    print(f"    * Local DiT Decoder LoRA layers: {dit_lora_count}")
    print(f"    * Total trainable/adapted LoRA parameters: {total_lora_params:,} ({total_lora_params * 2 / (1024*1024):.2f} MB in bf16)")
    print(f"    * Layers with verified active weights: {non_zero_lora_count} / {len(lora_modules)}")

    print("\n  - Sample Active LoRA Modules:")
    for name, mod in lora_modules[:4]:
        scaling_val = getattr(mod, "scaling", getattr(mod, "scale", mod.alpha / mod.r))
        enabled_val = getattr(mod, "enabled", True)
        print(f"    * Layer: {name}")
        print(f"        r={mod.r}, scaling={scaling_val}, enabled={enabled_val}")
        print(f"        lora_A shape={list(mod.lora_A.shape)}, L2-norm={mod.lora_A.norm().item():.4f}")
        print(f"        lora_B shape={list(mod.lora_B.shape)}, L2-norm={mod.lora_B.norm().item():.4f}")

    # 5. Functional Test: Synthesis with LoRA ON vs LoRA OFF
    test_text = "สวัสดีครับ ทดสอบระบบเสียงภาษาไทย SiangTTS ทำงานได้ถูกต้องสมบูรณ์ครับ"
    print(f"\n[7] Functional Synthesis Test (LoRA ON vs LoRA OFF):")
    print(f"  - Test Thai Text: '{test_text}'")

    # Step A: Synthesize with LoRA ENABLED
    tts_model.set_lora_enabled(True)
    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)

    print("  [A] Generating speech with Thai LoRA ENABLED...")
    t_start = time.time()
    wav_lora_on = synth.synth(test_text, cfg_value=2.5, inference_timesteps=10)
    dur_on = time.time() - t_start
    audio_dur_on = len(wav_lora_on) / synth.sample_rate
    rms_on = float(np.sqrt(np.mean(wav_lora_on**2)))
    print(f"      -> Generation time: {dur_on:.2f}s")
    print(f"      -> Audio length: {audio_dur_on:.2f}s ({len(wav_lora_on)} samples @ {synth.sample_rate}Hz)")
    print(f"      -> Audio RMS energy: {rms_on:.4f}")

    # Step B: Synthesize with LoRA DISABLED (Base VoxCPM2 only)
    print("  [B] Generating speech with Thai LoRA DISABLED (Base model only)...")
    tts_model.set_lora_enabled(False)
    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)

    t_start = time.time()
    wav_lora_off = synth.synth(test_text, cfg_value=2.5, inference_timesteps=10)
    dur_off = time.time() - t_start
    audio_dur_off = len(wav_lora_off) / synth.sample_rate
    rms_off = float(np.sqrt(np.mean(wav_lora_off**2)))
    print(f"      -> Generation time: {dur_off:.2f}s")
    print(f"      -> Audio length: {audio_dur_off:.2f}s ({len(wav_lora_off)} samples @ {synth.sample_rate}Hz)")
    print(f"      -> Audio RMS energy: {rms_off:.4f}")

    # Re-enable LoRA
    tts_model.set_lora_enabled(True)

    # Compare difference
    min_len = min(len(wav_lora_on), len(wav_lora_off))
    diff = np.abs(wav_lora_on[:min_len] - wav_lora_off[:min_len])
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean((wav_lora_on[:min_len] - wav_lora_off[:min_len]) ** 2)))

    print(f"\n[8] Mathematical & Acoustic Comparison:")
    print(f"  - Max waveform amplitude difference: {max_diff:.6f}")
    print(f"  - Mean absolute difference: {mean_diff:.6f}")
    print(f"  - Root Mean Square Error (RMSE): {rmse:.6f}")

    # Save audio files
    import soundfile as sf
    out_dir = PROJECT_ROOT / "scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_on = out_dir / "test_thai_lora_ON.wav"
    out_off = out_dir / "test_thai_lora_OFF.wav"
    sf.write(str(out_on), wav_lora_on, synth.sample_rate)
    sf.write(str(out_off), wav_lora_off, synth.sample_rate)

    print(f"\n[9] Saved Audio Outputs for Listening:")
    print(f"  - Thai LoRA ON  : {out_on}")
    print(f"  - Thai LoRA OFF : {out_off}")

    print("\n" + "=" * 65)
    if non_zero_lora_count == len(lora_modules) and max_diff > 0.05 and rms_on > 0.01:
        print("[SUCCESS] RESULT: Thai LoRA is 100% ACTIVE, LOADED, AND WORKING PROPERLY!")
        print("   (LoRA weights directly altered acoustic features and synthesis output)")
    else:
        print("[WARNING] RESULT: Thai LoRA check completed with warnings.")
    print("=" * 65)

if __name__ == "__main__":
    check_lora()
