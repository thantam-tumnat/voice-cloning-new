"""Phase 0 spike — does the new emotion pipeline actually work?

Two questions this answers, in order, before any of the pipeline is rewritten:

  1. Thonburian (F5) cloning a thai-ser *emotion donor* clip — does the generated
     Thai speech come out audibly in that emotion, for a sentence the donor never
     said? (stage: thonburian)
  2. SeedVC re-timbring that output to the target speaker — does the emotion
     survive the voice swap, and does it sound like the target? (stage: seedvc)

The two models do not share a dependency set, so SeedVC runs out of its own repo
via subprocess (``--seedvc-repo``), never imported here. flowtts is imported from
a checkout on ``--flowtts-src`` (or $FLOWTTS_SRC) because its package has no
top-level __init__.py and does not pip-install cleanly.

Outputs land in ``--out-dir`` (default scratch/spike):
    thon_<emo>.wav   Thonburian output (donor emotion, donor-ish timbre)
    vc_<emo>.wav     after SeedVC to the target voice

Run (from the voice-cloning repo root):
    FLOWTTS_SRC=/path/to/thonburian-tts \
    python tools/spike_thonburian_seedvc.py --stage thonburian
    python tools/spike_thonburian_seedvc.py --stage seedvc \
        --seedvc-repo /path/to/seed-vc --target ../voice-cloning-with-tones/ref/determination.wav
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Content-neutral so any emotion heard is delivery, not words. Deliberately NOT
# the sentence the donor clips say, to test that emotion generalises to new text.
TEST_SENTENCE = "ตอนนี้เป็นเวลาบ่ายสองโมงตรง อากาศข้างนอกกำลังสบายดี"

CHECKPOINT = "hf://biodatlab/ThonburianTTS/megaF5/mega_f5_last.safetensors"
VOCAB = "hf://biodatlab/ThonburianTTS/megaF5/mega_vocab.txt"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["thonburian", "seedvc", "all"], default="thonburian")
    p.add_argument("--refs-dir", type=Path, default=Path("ref/emotions"),
                   help="donor clips from build_emotion_refs.py (default: ref/emotions)")
    p.add_argument("--out-dir", type=Path, default=Path("scratch/spike"))
    p.add_argument("--emotions", nargs="+", default=None,
                   help="subset to run (default: every <emo>_1.wav in refs-dir)")
    p.add_argument("--text", default=TEST_SENTENCE, help="sentence to synthesize")
    p.add_argument("--flowtts-src", default=os.environ.get("FLOWTTS_SRC", ""),
                   help="path to a thonburian-tts checkout (or set $FLOWTTS_SRC)")
    p.add_argument("--device", default="cuda:0" if _cuda() else "cpu")
    p.add_argument("--cfg-strength", type=float, default=2.0)
    p.add_argument("--nfe-step", type=int, default=32)
    # flowtts loads the F5 model in float16 on any CUDA card with compute >= 7.0
    # (utils_infer.load_checkpoint). On GTX 16-series cards (no tensor cores) that
    # yields all-NaN audio, written out as a constant -1.0 rail. Forcing fp32 fixes
    # it at some speed cost. Default on for CUDA; --no-fp32 to compare.
    p.add_argument("--fp32", dest="fp32", action="store_true", default=None)
    p.add_argument("--no-fp32", dest="fp32", action="store_false")
    # seedvc stage
    p.add_argument("--seedvc-repo", type=Path, default=None,
                   help="path to a seed-vc checkout (its inference.py is run via subprocess)")
    p.add_argument("--seedvc-python", default=sys.executable,
                   help="python that has seed-vc's requirements (default: this one)")
    p.add_argument("--target", type=Path, default=None,
                   help="target-voice reference wav for the SeedVC swap")
    p.add_argument("--diffusion-steps", type=int, default=25)
    # Emotion survives the timbre swap only if the source pitch contour is kept.
    # --f0-condition True conditions SeedVC on the source F0; --auto-f0-adjust True
    # then shifts that contour into the target's register (keeps the variation,
    # moves the mean), which is what preserves emotion while still sounding like
    # the target. --semi-tone-shift is a manual register nudge if needed.
    p.add_argument("--f0-condition", default="False", choices=["True", "False"])
    p.add_argument("--auto-f0-adjust", default="False", choices=["True", "False"])
    p.add_argument("--semi-tone-shift", type=int, default=0)
    p.add_argument("--vc-tag", default="vc",
                   help="output prefix so a tuned run (e.g. --vc-tag vcf0) sits beside the first")
    return p.parse_args(argv)


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _emotion_clips(refs_dir: Path, only):
    """(emotion, wav_path, ref_text) for each <emo>_1.wav that has a transcript."""
    out = []
    for wav in sorted(refs_dir.glob("*_1.wav")):
        emo = wav.stem.rsplit("_", 1)[0]
        if only and emo not in only:
            continue
        txt = wav.with_suffix(".txt")
        out.append((emo, wav, txt.read_text(encoding="utf-8").strip() if txt.exists() else None))
    return out


def stage_thonburian(args) -> int:
    src = args.flowtts_src.strip()
    if not src or not Path(src).exists():
        print("[spike] --flowtts-src (or $FLOWTTS_SRC) must point at a thonburian-tts "
              "checkout — its 'flowtts' package has no top-level __init__.py and does "
              "not pip-install cleanly.", file=sys.stderr)
        return 2
    sys.path.insert(0, str(Path(src).resolve()))

    from flowtts.inference import FlowTTSPipeline, ModelConfig, AudioConfig

    clips = _emotion_clips(args.refs_dir, args.emotions)
    if not clips:
        print(f"[spike] no <emo>_1.wav in {args.refs_dir}; run build_emotion_refs.py first",
              file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[spike] loading Thonburian F5 on {args.device} …")
    t0 = time.time()
    pipeline = FlowTTSPipeline(
        model_config=ModelConfig(language="th", model_type="F5", vocoder="vocos",
                                 checkpoint=CHECKPOINT, vocab_file=VOCAB, device=args.device),
        audio_config=AudioConfig(cfg_strength=args.cfg_strength, nfe_step=args.nfe_step),
        temp_dir=str(args.out_dir / "_temp"),
    )
    fp32 = args.fp32 if args.fp32 is not None else ("cuda" in args.device)
    if fp32:
        import torch
        m = pipeline.model
        for attr in ("ema_model", "vocoder"):
            mod = getattr(m, attr, None)
            if mod is not None and hasattr(mod, "float"):
                setattr(m, attr, mod.float())
        torch.set_default_dtype(torch.float32)
        print("[spike] forced model + vocoder to float32 (GTX 16-series fp16 -> NaN guard)")
    print(f"[spike] model ready in {time.time()-t0:.0f}s. text = {args.text!r}")

    rc = 0
    for emo, wav, ref_text in clips:
        if not ref_text:
            print(f"[spike]   {emo}: no transcript sidecar, skipping", file=sys.stderr)
            rc = 1
            continue
        out = args.out_dir / f"thon_{emo}.wav"
        t = time.time()
        try:
            pipeline(text=args.text, ref_voice=str(wav), ref_text=ref_text, output_file=str(out))
            print(f"[spike]   {emo:11s} -> {out.name}  ({time.time()-t:.1f}s)")
        except Exception as e:                                    # noqa: BLE001
            print(f"[spike]   {emo}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            rc = 1
    print(f"[spike] thonburian stage done -> {args.out_dir}")
    return rc


def stage_seedvc(args) -> int:
    if not args.seedvc_repo or not (args.seedvc_repo / "inference.py").exists():
        print("[spike] --seedvc-repo must point at a seed-vc checkout (with inference.py)",
              file=sys.stderr)
        return 2
    if not args.target or not args.target.exists():
        print("[spike] --target <target-voice.wav> is required for the seedvc stage",
              file=sys.stderr)
        return 2

    sources = sorted(args.out_dir.glob("thon_*.wav"))
    if not sources:
        print(f"[spike] no thon_*.wav in {args.out_dir}; run --stage thonburian first",
              file=sys.stderr)
        return 1

    vc_dir = args.out_dir / "_seedvc_raw"
    vc_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for src in sources:
        emo = src.stem.replace("thon_", "")
        print(f"[spike]   seedvc {emo}: {src.name} -> target={args.target.name} "
              f"(f0={args.f0_condition}, autof0={args.auto_f0_adjust}, shift={args.semi_tone_shift})")
        cmd = [
            str(args.seedvc_python), "inference.py",
            "--source", str(src.resolve()),
            "--target", str(args.target.resolve()),
            "--output", str(vc_dir.resolve()),
            "--diffusion-steps", str(args.diffusion_steps),
            "--inference-cfg-rate", "0.7",
            "--f0-condition", args.f0_condition,
            "--auto-f0-adjust", args.auto_f0_adjust,
            "--semi-tone-shift", str(args.semi_tone_shift),
            "--fp16", "True",
        ]
        proc = subprocess.run(cmd, cwd=str(args.seedvc_repo), capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(f"[spike]   seedvc {emo} FAILED:\n{(proc.stderr or '')[-600:]}", file=sys.stderr)
            rc = 1
            continue
        # seed-vc names the output after the source+target; grab the newest wav.
        produced = max(vc_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, default=None)
        if produced is not None:
            dest = args.out_dir / f"{args.vc_tag}_{emo}.wav"
            dest.write_bytes(produced.read_bytes())
            print(f"[spike]   seedvc {emo} -> {dest.name}")
    print(f"[spike] seedvc stage done -> {args.out_dir}")
    return rc


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = _parse_args(argv)
    rc = 0
    if args.stage in ("thonburian", "all"):
        rc |= stage_thonburian(args)
    if args.stage in ("seedvc", "all"):
        rc |= stage_seedvc(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
