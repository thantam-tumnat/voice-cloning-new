"""Trace one utterance from Thonburian F5 through SeedVC, keeping every stage.

Unlike ThonburianService.render_chunks (which deletes intermediates), this script
saves each stage to a timestamped folder under test_runs/ so you can *hear* and
measure how the voice is transformed at each step:

    donor clip      -> the thai-ser emotion reference (what F5 imitates)
    stage A (F5)     -> emotional Thai speech, still in the DONOR's timbre  (24 kHz)
    stage B (SeedVC) -> same speech, timbre swapped to the TARGET speaker   (44.1 kHz)
    target ref       -> the speaker whose voice we are cloning

For each file it prints duration, sample rate, loudness (RMS dBFS) and pitch
(median f0 + range in semitones), so the transformation is visible as numbers,
not just audible.

Usage (from voice-cloning-with-tones/):
    python tools/trace_pipeline.py
    python tools/trace_pipeline.py --emotion happy --gender female --speaker lion \
        --text "สวัสดีครับ วันนี้อากาศดีมากเลย" --speed 0.9
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make "app" importable when run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import soundfile as sf

from app.config import settings
from app.services.thonburian_service import ThonburianService


def analyze(path: Path) -> dict:
    """Duration, sample rate, loudness and pitch of one audio file."""
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    dur = len(y) / sr if sr else 0.0

    rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
    dbfs = 20 * np.log10(rms) if rms > 1e-9 else float("-inf")

    f0_med = f0_range_st = None
    if y.size and dur > 0.1:
        try:
            f0, voiced, _ = librosa.pyin(
                y, sr=sr, fmin=70, fmax=400, frame_length=2048
            )
            f0v = f0[np.isfinite(f0)]
            if f0v.size:
                f0_med = float(np.median(f0v))
                lo, hi = np.percentile(f0v, [5, 95])
                if lo > 0:
                    f0_range_st = float(12 * np.log2(hi / lo))
        except Exception:
            pass

    return {
        "dur_s": dur,
        "sr": sr,
        "dbfs": dbfs,
        "f0_med_hz": f0_med,
        "f0_range_st": f0_range_st,
    }


def fmt(m: dict) -> str:
    f0 = f"{m['f0_med_hz']:.0f}Hz" if m["f0_med_hz"] else "  -  "
    rng = f"{m['f0_range_st']:.1f}st" if m["f0_range_st"] else "  -  "
    dbfs = f"{m['dbfs']:.1f}" if np.isfinite(m["dbfs"]) else " -inf"
    return f"{m['dur_s']:6.2f}s  {m['sr']:6d}Hz  {dbfs:>7}dBFS  f0 {f0:>7}  range {rng:>7}"


ALL_EMOTIONS = ["neutral", "happy", "sad", "angry", "frustrated"]


def run_one(svc, emotion, gender, target_wav, text, speed, out_dir) -> dict:
    """Trace one emotion end-to-end; save both stages into out_dir. Returns metrics."""
    donor_wav, donor_txt = svc._resolve_donor_clip(emotion, gender=gender)
    stage_a = out_dir / f"A_f5_{emotion}.wav"       # F5 raw (donor timbre)
    stage_b = out_dir / f"B_vc_{emotion}.wav"       # after SeedVC (target timbre)

    print("=" * 78)
    print(f"TRACE  emotion={emotion}  gender={gender}  speed={speed}")
    print(f"  donor clip  : {donor_wav}")
    print(f"  donor text  : {donor_txt}")
    print(f"  target ref  : {target_wav}")

    # --- stage A: Thonburian F5 (emotional speech in donor's voice) -------- #
    body = svc._prepare_text(text)
    t0 = time.time()
    pipeline = svc.get_pipeline()
    pipeline(
        text=body,
        ref_voice=str(donor_wav.resolve()),
        ref_text=donor_txt,
        output_file=str(stage_a.resolve()),
        speed=speed,
    )
    print(f"  [A] F5    done in {time.time()-t0:5.1f}s -> {stage_a.name}")

    # Free F5's GPU cache so SeedVC (same GPU) is not starved into a timeout.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    # --- stage B: SeedVC (timbre -> target speaker) ------------------------ #
    t0 = time.time()
    svc._convert_seedvc(source_wav=stage_a, target_wav=Path(target_wav), output_wav=stage_b)
    print(f"  [B] SeedVC done in {time.time()-t0:5.1f}s -> {stage_b.name}")

    return {
        "emotion": emotion,
        "donor": analyze(donor_wav),
        "A": analyze(stage_a),
        "B": analyze(stage_b),
        "stage_a": stage_a,
        "stage_b": stage_b,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emotion", default="all",
                    help="one of neutral|angry|happy|sad|frustrated, or 'all'")
    ap.add_argument("--gender", default="female", help="female|male")
    ap.add_argument("--speaker", default=None,
                    help="target speaker id in ref/ (e.g. lion). Default: resolved fallback")
    ap.add_argument("--text", default="สวัสดีครับ วันนี้เป็นการทดสอบระบบแปลงเสียง",
                    help="Thai text to synthesize")
    ap.add_argument("--speed", type=float, default=None,
                    help="F5 speed multiplier (<1 slower). Default: settings.thonburian_speed")
    ap.add_argument("--outdir", default=None,
                    help="output folder under test_runs/. Default: trace_<timestamp>")
    args = ap.parse_args()

    svc = ThonburianService()
    speed = settings.thonburian_speed if args.speed is None else args.speed
    emotions = ALL_EMOTIONS if args.emotion.lower() == "all" else [args.emotion]

    # --- resolve target speaker (shared across emotions) ------------------- #
    if args.speaker:
        target_wav = svc.get_speaker_audio_path(args.speaker)
        if not target_wav:
            print(f"[error] speaker '{args.speaker}' not found in ref/", file=sys.stderr)
            return 2
    else:
        target_wav = svc._resolve_default_speaker(gender=args.gender)

    folder = args.outdir or f"trace_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path("test_runs") / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"target ref : {target_wav}")
    print(f"gen text   : {args.text}")
    print(f"output dir : {out_dir}\n")

    results = []
    for emo in emotions:
        try:
            results.append(run_one(svc, emo, args.gender, target_wav, args.text, speed, out_dir))
        except Exception as e:
            print(f"  [skip] {emo}: {type(e).__name__}: {e}", file=sys.stderr)

    # --- combined comparison table ---------------------------------------- #
    tgt = analyze(Path(target_wav))
    print("\n" + "=" * 90)
    print(f"{'EMOTION':<12}{'STAGE':<10}{'DUR':>7}{'SR':>8}{'LOUDNESS':>11}{'PITCH':>9}{'RANGE':>9}")
    print("-" * 90)
    for r in results:
        for tag, key in (("donor", "donor"), ("A:F5", "A"), ("B:VC", "B")):
            m = r[key]
            f0 = f"{m['f0_med_hz']:.0f}Hz" if m["f0_med_hz"] else "-"
            rng = f"{m['f0_range_st']:.1f}st" if m["f0_range_st"] else "-"
            dbfs = f"{m['dbfs']:.1f}" if np.isfinite(m["dbfs"]) else "-inf"
            emo_col = r["emotion"] if tag == "donor" else ""
            print(f"{emo_col:<12}{tag:<10}{m['dur_s']:6.2f}s{m['sr']:7d}Hz"
                  f"{dbfs:>9}dB{f0:>9}{rng:>9}")
        print("-" * 90)
    f0t = f"{tgt['f0_med_hz']:.0f}Hz" if tgt["f0_med_hz"] else "-"
    print(f"{'TARGET':<12}{'ref':<10}{tgt['dur_s']:6.2f}s{tgt['sr']:7d}Hz"
          f"{tgt['dbfs']:>8.1f}dB{f0t:>9}")
    print("=" * 90)

    # --- per-emotion interpretation --------------------------------------- #
    print("\nPitch shift A->B (how far SeedVC moved timbre toward target), and match to target:")
    for r in results:
        a, b = r["A"], r["B"]
        if a["f0_med_hz"] and b["f0_med_hz"] and tgt["f0_med_hz"]:
            d_st = 12 * np.log2(b["f0_med_hz"] / a["f0_med_hz"])
            gap = 12 * np.log2(b["f0_med_hz"] / tgt["f0_med_hz"])
            print(f"  {r['emotion']:<11} shift {d_st:+5.1f} st   |  gap-to-target {gap:+5.1f} st "
                  f"(0 = perfect pitch match)")
    print(f"\nAll files in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
