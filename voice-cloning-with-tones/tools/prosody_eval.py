"""Measure the prosody of a synthesized take, and diff it against ElevenLabs.

The pipeline's emotion wording was originally tuned to maximise happy-vs-sad
median-F0 separation. Measuring the ElevenLabs reference showed that is close to
the wrong objective: across four emotions on identical Thai text its median F0
moves only ~1.6 semitones, while duration moves 14% and energy moves 5 dB. This
tool reports the dimensions that actually carry the emotion, in units that stay
comparable across voices, so a change can be judged instead of guessed at.

Usage:
    python tools/prosody_eval.py take.wav --labels sad,happy,scared,tired
    python tools/prosody_eval.py take.wav --json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

HOP_S = 0.010
WIN_S = 0.025

# Measured from ElevenLabs_2026-08-18T04_26_39_Liam ... _v3.mp3: one voice, one
# Thai sentence, four emotion tags. Stored as ratios and semitones so they stay
# meaningful when measured against a different speaker.
ELEVENLABS_REFERENCE: Dict[str, Dict[str, float]] = {
    # Produced by this module's own analyze() on that file, so the reference passes
    # its own targets -- regenerate with tools/refresh_reference.py if the tracker
    # ever changes.
    #          dur_ratio    energy_db    f0_spread_st  f0_slope_st  f0_offset_st
    #          (vs mean)    (vs mean)    (semitones)   (st/sec)     (st vs median)
    "sad":    {"dur_ratio": 1.035, "energy_db": -2.01, "f0_spread_st": 12.82, "f0_slope_st": -1.20, "f0_offset_st": -0.14},
    "happy":  {"dur_ratio": 0.927, "energy_db": -0.52, "f0_spread_st": 10.08, "f0_slope_st": -1.12, "f0_offset_st": +0.14},
    "scared": {"dur_ratio": 0.966, "energy_db": +2.99, "f0_spread_st": 11.77, "f0_slope_st": -0.10, "f0_offset_st": +1.60},
    "tired":  {"dur_ratio": 1.073, "energy_db": -1.37, "f0_spread_st": 11.84, "f0_slope_st": -0.98, "f0_offset_st": -0.15},
}

# Emotion-boundary silences in the reference, in seconds. The pipeline used to
# join chunks with a flat 60 ms, which is why the output sounded spliced.
ELEVENLABS_GAPS_S = (1.31, 0.95, 1.44)

# Largest median-F0 excursion across the reference's four takes. Beyond this the
# listener starts hearing a different person per emotion, which is the failure
# mode chunk-at-a-time synthesis is most prone to.
IDENTITY_MAX_SPREAD_ST = 1.8


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def load_mono(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Read any audio file to a mono float32 array, via ffmpeg if libsndfile cannot."""
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(f"Cannot decode {path}: soundfile failed and ffmpeg is not on PATH")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            cmd = [ffmpeg, "-y", "-v", "error", "-i", path, "-ac", "1"]
            if target_sr:
                cmd += ["-ar", str(target_sr)]
            subprocess.run(cmd + [tmp.name], check=True)
            x, sr = sf.read(tmp.name, dtype="float32", always_2d=False)
        finally:
            os.unlink(tmp.name)

    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype="float32"), sr


# --------------------------------------------------------------------------- #
# Framing & segmentation
# --------------------------------------------------------------------------- #

def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame RMS in dBFS."""
    hop, win = int(sr * HOP_S), int(sr * WIN_S)
    n = max(0, (len(x) - win) // hop)
    if n == 0:
        return np.zeros(0, dtype="float32")
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n]
    rms = np.sqrt(np.mean(frames.astype("float64") ** 2, axis=1))
    return (20 * np.log10(rms + 1e-9)).astype("float32")


def find_utterances(
    db: np.ndarray,
    floor_offset_db: float = 35.0,
    min_gap_s: float = 0.30,
    min_dur_s: float = 0.40,
) -> List[Tuple[int, int]]:
    """Split into utterances on silences at least ``min_gap_s`` long.

    Returns half-open frame index pairs. Used when the caller does not already
    know where the chunk boundaries are; ab_gen passes them in explicitly.
    """
    if db.size == 0:
        return []
    voiced = db > (db.max() - floor_offset_db)
    min_gap = int(min_gap_s / HOP_S)

    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    silence = 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_gap:
                spans.append((start, i - silence + 1))
                start = None
    if start is not None:
        spans.append((start, len(voiced)))

    return [(a, b) for a, b in spans if (b - a) * HOP_S >= min_dur_s]


# --------------------------------------------------------------------------- #
# Pitch
# --------------------------------------------------------------------------- #

def track_f0(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0) -> np.ndarray:
    """Normalized-autocorrelation F0 track in Hz, unvoiced frames dropped.

    Deliberately dependency-free -- the project already pulls numpy and soundfile,
    and adding librosa just for this would make the eval harness harder to install
    than the thing it measures.
    """
    win = int(sr * 0.040)
    hop = int(sr * HOP_S)
    lo, hi = int(sr / fmax), int(sr / fmin)
    if len(x) < win or hi <= lo:
        return np.zeros(0, dtype="float32")

    window = np.hanning(win)
    energy_floor = 0.02 * np.sqrt(np.mean(x.astype("float64") ** 2) + 1e-12)

    out: List[float] = []
    for i in range(0, len(x) - win, hop):
        frame = x[i:i + win].astype("float64")
        if np.sqrt(np.mean(frame ** 2)) < energy_floor:
            continue
        frame = (frame - frame.mean()) * window
        ac = np.correlate(frame, frame, mode="full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        band = ac[lo:hi]
        if band.size == 0:
            continue
        k = int(np.argmax(band))
        if band[k] < 0.35:
            continue
        # Parabolic interpolation around the peak for sub-bin resolution.
        j = lo + k
        shift = 0.0
        if 0 < j < len(ac) - 1:
            a, b, c = ac[j - 1], ac[j], ac[j + 1]
            denom = a - 2 * b + c
            if denom != 0:
                shift = 0.5 * (a - c) / denom
        out.append(sr / (j + shift))

    f0 = np.array(out, dtype="float32")
    if f0.size < 3:
        return f0

    # Median-smooth to kill octave jumps, which otherwise dominate the spread stat.
    k = 5
    padded = np.pad(f0, (k // 2, k // 2), mode="edge")
    return np.array([np.median(padded[i:i + k]) for i in range(len(f0))], dtype="float32")


def hz_to_st(hz: float, ref_hz: float) -> float:
    """Interval from ref_hz to hz, in semitones."""
    if hz <= 0 or ref_hz <= 0:
        return 0.0
    return 12.0 * math.log2(hz / ref_hz)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def segment_metrics(seg: np.ndarray, sr: int) -> Dict[str, float]:
    """Raw, un-normalized measurements for one utterance."""
    dur = len(seg) / sr
    db = frame_db(seg, sr)
    rms = float(np.sqrt(np.mean(seg.astype("float64") ** 2))) if seg.size else 0.0

    f0 = track_f0(seg, sr)
    if f0.size >= 3:
        med = float(np.median(f0))
        p10, p90 = float(np.percentile(f0, 10)), float(np.percentile(f0, 90))
        spread_st = hz_to_st(p90, p10)
        t = np.arange(len(f0)) * HOP_S
        slope_hz = float(np.polyfit(t, f0, 1)[0])
        # Convert Hz/s to semitones/s about the utterance median.
        slope_st = slope_hz * 12.0 / (med * math.log(2)) if med > 0 else 0.0
    else:
        med = p10 = p90 = spread_st = slope_st = 0.0

    # Internal pauses: >=120 ms below the utterance's own speech floor.
    pauses: List[float] = []
    if db.size:
        quiet = db <= (db.max() - 32.0)
        run = 0
        for q in quiet:
            if q:
                run += 1
            else:
                if run >= 12:
                    pauses.append(run * HOP_S)
                run = 0
        if run >= 12:
            pauses.append(run * HOP_S)

    return {
        "dur_s": dur,
        "rms": rms,
        "energy_dbfs": 20 * math.log10(rms + 1e-9),
        "f0_med_hz": med,
        "f0_p10_hz": p10,
        "f0_p90_hz": p90,
        "f0_spread_st": spread_st,
        "f0_slope_st": slope_st,
        "n_pauses": float(len(pauses)),
        "pause_total_s": float(sum(pauses)),
    }


def analyze(
    path: str,
    labels: Optional[Sequence[str]] = None,
    boundaries: Optional[Sequence[Tuple[float, float]]] = None,
) -> Dict:
    """Measure every utterance in a take and normalize against the take's own mean.

    Normalizing within the take is what makes these numbers comparable to the
    ElevenLabs reference despite a different speaker: what we are matching is the
    contrast *between* emotions, not absolute pitch or level.
    """
    x, sr = load_mono(path)

    if boundaries:
        spans = [(int(a * sr), int(b * sr)) for a, b in boundaries]
    else:
        db = frame_db(x, sr)
        hop = int(sr * HOP_S)
        spans = [(a * hop, b * hop) for a, b in find_utterances(db)]

    raw = [segment_metrics(x[a:b], sr) for a, b in spans]
    if not raw:
        return {"path": path, "sample_rate": sr, "utterances": [], "gaps_s": [], "identity": {}}

    mean_dur = float(np.mean([m["dur_s"] for m in raw]))
    mean_rms = float(np.mean([m["rms"] for m in raw]))
    meds = [m["f0_med_hz"] for m in raw if m["f0_med_hz"] > 0]
    ref_f0 = float(np.median(meds)) if meds else 0.0

    utterances = []
    for i, m in enumerate(raw):
        name = labels[i] if labels and i < len(labels) else f"utt{i}"
        utterances.append({
            "label": name,
            "start_s": spans[i][0] / sr,
            "end_s": spans[i][1] / sr,
            **m,
            "dur_ratio": m["dur_s"] / mean_dur if mean_dur else 0.0,
            "energy_db": 20 * math.log10((m["rms"] + 1e-9) / (mean_rms + 1e-9)),
            "f0_offset_st": hz_to_st(m["f0_med_hz"], ref_f0),
        })

    gaps = [(spans[i][0] - spans[i - 1][1]) / sr for i in range(1, len(spans))]

    offsets = [u["f0_offset_st"] for u in utterances if u["f0_med_hz"] > 0]
    identity_spread = (max(offsets) - min(offsets)) if len(offsets) > 1 else 0.0

    return {
        "path": path,
        "sample_rate": sr,
        "ref_f0_hz": ref_f0,
        "utterances": utterances,
        "gaps_s": gaps,
        "identity": {
            "f0_spread_st": identity_spread,
            "budget_st": IDENTITY_MAX_SPREAD_ST,
            "ok": identity_spread <= IDENTITY_MAX_SPREAD_ST,
        },
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _cell(value: float, target: Optional[float], tol: float) -> str:
    if target is None:
        return f"{value:>7.2f}          "
    d = value - target
    mark = "ok" if abs(d) <= tol else "MISS"
    return f"{value:>7.2f} {d:+6.2f} {mark:<4}"


def report(result: Dict, compare: bool = True) -> str:
    lines = [
        f"file        : {result['path']}",
        f"sample rate : {result['sample_rate']} Hz",
        f"reference F0: {result.get('ref_f0_hz', 0):.1f} Hz",
        "",
        f"{'label':<9}{'dur_s':>7}  {'dur_ratio':^19} {'energy_dB':^19} "
        f"{'f0_spread_st':^19} {'f0_offset_st':^19}{'pauses':>8}",
    ]
    for u in result["utterances"]:
        tgt = ELEVENLABS_REFERENCE.get(u["label"], {}) if compare else {}
        lines.append(
            f"{u['label']:<9}{u['dur_s']:>7.2f}  "
            f"{_cell(u['dur_ratio'], tgt.get('dur_ratio'), 0.04)} "
            f"{_cell(u['energy_db'], tgt.get('energy_db'), 1.5)} "
            f"{_cell(u['f0_spread_st'], tgt.get('f0_spread_st'), 2.0)} "
            f"{_cell(u['f0_offset_st'], tgt.get('f0_offset_st'), 0.8)}"
            f"{int(u['n_pauses']):>8}"
        )

    lines.append("")
    if result["gaps_s"]:
        lines.append("inter-utterance gaps : " + ", ".join(f"{g:.2f}s" for g in result["gaps_s"]))
        lines.append(
            "  ElevenLabs reference: "
            + ", ".join(f"{g:.2f}s" for g in ELEVENLABS_GAPS_S)
            + " at emotion boundaries"
        )

    ident = result.get("identity") or {}
    if ident:
        verdict = "OK" if ident["ok"] else "DRIFT"
        lines.append(
            f"voice identity       : median-F0 spread {ident['f0_spread_st']:.2f} st "
            f"(budget {ident['budget_st']:.1f} st) -> {verdict}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("audio", help="WAV/MP3 take to measure")
    ap.add_argument("--labels", help="comma-separated tone labels, in utterance order")
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of a table")
    ap.add_argument("--no-compare", action="store_true", help="skip the ElevenLabs target diff")
    args = ap.parse_args(argv)

    labels = [s.strip() for s in args.labels.split(",")] if args.labels else None
    result = analyze(args.audio, labels=labels)

    if args.json:
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(report(result, compare=not args.no_compare))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
