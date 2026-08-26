"""Pre-download all SeedVC model weights with progress reporting.

Run this with the seed-vc python environment to ensure all models are cached
before starting the SeedVC worker service:

    C:\\Users\\opendream002\\Desktop\\seed-vc\\seedvc-venv\\Scripts\\python.exe tools/download_seedvc_weights.py --seedvc-repo C:\\Users\\opendream002\\Desktop\\seed-vc
"""

import argparse
import os
import sys
import time
from pathlib import Path

def download_models(seedvc_repo: str):
    repo = Path(seedvc_repo).resolve()
    sys.path.insert(0, str(repo))

    print("=" * 60)
    print("SeedVC Weights Downloader")
    print(f"SeedVC Repo: {repo}")
    print("=" * 60)

    from huggingface_hub import hf_hub_download
    from transformers import WhisperModel
    from hf_utils import load_custom_model_from_hf
    from modules.bigvgan import bigvgan

    steps = [
        ("1/6", "Whisper Small (OpenAI)", lambda: WhisperModel.from_pretrained("openai/whisper-small")),
        ("2/6", "Whisper Base (OpenAI)", lambda: WhisperModel.from_pretrained("openai/whisper-base")),
        ("3/6", "RMVPE Pitch Estimator", lambda: load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)),
        ("4/6", "CAMPPlus Speaker Extractor", lambda: load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", None)),
        ("5/6", "BigVGAN 22kHz Vocoder", lambda: bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)),
        ("6/6", "BigVGAN 44kHz Vocoder", lambda: bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_44khz_128band_512x", use_cuda_kernel=False)),
    ]

    for step_num, desc, fn in steps:
        print(f"\n[{step_num}] Downloading/verifying {desc} ...", flush=True)
        t0 = time.time()
        try:
            fn()
            print(f"[{step_num}] OK in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[{step_num}] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return 1

    print("\n" + "=" * 60)
    print("ALL SEEDVC WEIGHTS VERIFIED & CACHED SUCCESSFULLY!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seedvc-repo", default=r"C:\Users\opendream002\Desktop\seed-vc")
    args = parser.parse_args()
    sys.exit(download_models(args.seedvc_repo))
