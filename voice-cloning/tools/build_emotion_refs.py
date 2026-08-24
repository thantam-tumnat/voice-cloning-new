"""Curate emotion reference clips from the airesearch/thai-ser dataset.

These clips are the *emotion donors* of the new pipeline: the Thonburian TTS
stage clones one of them per chunk so the generated Thai speech carries that
emotion's prosody, and SeedVC then re-timbres the result to the target voice.

The dataset is tens of GB of multi-mic studio FLAC, so this streams it
(`streaming=True`) and stops as soon as it has enough clips per emotion — a full
download is never materialised. It prefers *scripted* turns, whose spoken text is
in the dataset's own ``script_sent`` column, so each saved clip gets a matching
transcript with no ASR step. Thonburian is F5-based and needs that reference text.

Output, under ``--out-dir`` (default ``ref/emotions``):

    <emo>_<n>.wav   mono, resampled to --target-sr, PCM_16
    <emo>_<n>.txt   the reference transcript (script_sent)
    manifest.json   {emotion: [{file, text, actor_id, gender, agreement, sec}]}

Only the five thai-ser emotions the studio supports are kept:
``neutral, angry, happy, sad, frustrated``. Everything else ("other") is skipped.

Run (from the voice-cloning repo root):

    uv run python tools/build_emotion_refs.py --per-emotion 5

The one mic column asked for is the only Audio feature decoded; the other three
are dropped before iteration so streaming does not pay to decode them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The five emotions the new pipeline supports, keyed by the lower-cased value of
# thai-ser's `assigned_emo`. "other" is deliberately absent — it is not a target.
SUPPORTED = ("neutral", "angry", "happy", "sad", "frustrated")
MICS = ("mic_clip", "mic_con", "mic_middle", "mic_zoom")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path("ref/emotions"),
                   help="where clips + transcripts are written (default: ref/emotions)")
    p.add_argument("--per-emotion", type=int, default=5,
                   help="how many clips to keep per emotion (default: 5)")
    p.add_argument("--mic", choices=MICS, default="mic_clip",
                   help="which microphone to use; the lapel clip is closest/cleanest (default: mic_clip)")
    p.add_argument("--min-agreement", type=float, default=0.6,
                   help="skip clips whose annotator agreement is below this (default: 0.6)")
    p.add_argument("--min-seconds", type=float, default=3.0, help="shortest clip to keep (default: 3.0)")
    p.add_argument("--max-seconds", type=float, default=10.0, help="longest clip to keep (default: 10.0)")
    p.add_argument("--target-sr", type=int, default=24000,
                   help="resample every clip to this rate; F5/Thonburian expects 24000 (default: 24000)")
    p.add_argument("--gender", choices=["m", "f"], default=None,
                   help="optionally keep only one actor gender for prosody consistency (default: any)")
    p.add_argument("--audio-ids", nargs="+", default=None,
                   help="pick these exact clips by audio_id (copy them from the HF dataset "
                        "viewer) instead of auto-selecting; agreement/duration/gender filters "
                        "are ignored and each clip is saved under its own assigned emotion")
    p.add_argument("--dataset", default="airesearch/thai-ser", help="HF dataset id")
    p.add_argument("--split", default="train", help="dataset split to stream (default: train)")
    p.add_argument("--emotions", nargs="+", default=list(SUPPORTED),
                   help=f"subset of emotions to build (default: all {len(SUPPORTED)})")
    p.add_argument("--max-scanned", type=int, default=200_000,
                   help="give up after streaming this many rows (default: 200000)")
    return p.parse_args(argv)


def _trim_silence(y, sr: int, thresh_db: float = -40.0, pad_s: float = 0.05):
    """Trim near-silent head and tail so the reference is speech, not room tone.

    A crude energy gate rather than a VAD: on a clean single-speaker studio clip
    the first/last voiced sample is unambiguous, and pulling in a dependency for
    this would not earn its keep. Leaves ``pad_s`` of margin so onsets are intact.
    """
    import numpy as np

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


def _extract_audio(aud):
    """A thai-ser Audio cell -> (array, sample_rate), across ``datasets`` versions.

    Older ``datasets`` decodes an Audio feature to ``{"array", "sampling_rate"}``;
    newer versions hand back a ``torchcodec`` ``AudioDecoder`` that is not
    subscriptable and must be pulled with ``get_all_samples()``.
    """
    if isinstance(aud, dict) and "array" in aud:
        return aud["array"], int(aud["sampling_rate"])
    if hasattr(aud, "get_all_samples"):
        s = aud.get_all_samples()
        return s.data, int(s.sample_rate)          # data: (channels, n) tensor
    raise TypeError(f"unrecognised audio cell type: {type(aud)!r}")


def _to_mono_resampled(array, sr: int, target_sr: int):
    """A thai-ser Audio array -> 1-D float32 at ``target_sr``."""
    import numpy as np
    import torch
    import torchaudio

    y = np.asarray(array, dtype=np.float32)
    if y.ndim > 1:                       # (channels, n) or (n, channels) -> mono
        y = y.mean(axis=0 if y.shape[0] < y.shape[1] else 1)
    if sr != target_sr:
        y = torchaudio.functional.resample(torch.from_numpy(y), sr, target_sr).numpy()
    return y


def _gender_ok(row_gender: str | None, want: str | None) -> bool:
    if want is None:
        return True
    g = (row_gender or "").strip().lower()
    return g.startswith(want)  # "male"/"female" or "m"/"f"


def build(args: argparse.Namespace) -> int:
    import soundfile as sf
    from datasets import load_dataset

    targets = [e.lower() for e in args.emotions if e.lower() in SUPPORTED]
    unknown = [e for e in args.emotions if e.lower() not in SUPPORTED]
    if unknown:
        print(f"[refs] ignoring unsupported emotion(s): {unknown}", file=sys.stderr)
    if not targets:
        print(f"[refs] no supported emotions requested; supported = {SUPPORTED}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[refs] streaming {args.dataset}:{args.split} — need {args.per_emotion} "
          f"clip(s) each for {targets}, mic={args.mic}")

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    # Decode only the mic we asked for: dropping the other three Audio columns
    # keeps streaming from decoding audio we throw away.
    cols = getattr(ds, "column_names", None) or list(MICS)
    drop = [m for m in MICS if m != args.mic and m in cols]
    if drop:
        ds = ds.remove_columns(drop)

    # In --audio-ids mode any of the five emotions may be picked, so bucket over
    # all of SUPPORTED rather than just the requested targets.
    want_ids = list(dict.fromkeys(args.audio_ids)) if args.audio_ids else None
    buckets = list(SUPPORTED) if want_ids else targets
    kept: dict[str, list[dict]] = {e: [] for e in buckets}
    manifest: dict[str, list[dict]] = {e: [] for e in buckets}
    remaining = set(want_ids) if want_ids else None
    scanned = 0

    def _write(emo: str, y, sec: float, text: str, row: dict) -> None:
        n = len(kept[emo]) + 1
        stem = f"{emo}_{n}"
        sf.write(str(args.out_dir / f"{stem}.wav"), y, args.target_sr, subtype="PCM_16")
        (args.out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        entry = {
            "file": f"{stem}.wav",
            "audio_id": row.get("audio_id"),
            "text": text,
            "actor_id": row.get("actor_id"),
            "gender": row.get("actor_gender"),
            "agreement": round(float(row.get("agreement") or 0.0), 3),
            "sec": round(sec, 2),
        }
        kept[emo].append(entry)
        manifest[emo].append(entry)
        print(f"[refs] {stem}.wav  ({sec:.1f}s, agree={entry['agreement']}, "
              f"{entry['gender']}, id={entry['audio_id']})  {text[:36]}…")

    def _decode(row: dict):
        aud = row.get(args.mic)
        if aud is None:
            return None, None
        array, in_sr = _extract_audio(aud)
        y = _trim_silence(_to_mono_resampled(array, in_sr, args.target_sr), args.target_sr)
        return y, y.size / args.target_sr

    if want_ids:
        print(f"[refs] picking {len(want_ids)} clip(s) by audio_id (filters ignored)")

    for row in ds:
        scanned += 1
        if scanned > args.max_scanned:
            print(f"[refs] stopped after scanning {args.max_scanned} rows", file=sys.stderr)
            break

        # ---- explicit pick by audio_id ------------------------------------
        if want_ids is not None:
            if not remaining:
                break
            aid = str(row.get("audio_id") or "")
            if aid not in remaining:
                continue
            remaining.discard(aid)
            emo = (row.get("assigned_emo") or "").strip().lower()
            if emo not in SUPPORTED:
                print(f"[refs]   {aid}: assigned_emo {emo!r} not supported, skipping", file=sys.stderr)
                continue
            text = (row.get("script_sent") or "").strip()
            if not text:
                print(f"[refs]   {aid}: no script_sent transcript (improv turn); F5 needs ref "
                      f"text, skipping", file=sys.stderr)
                continue
            y, sec = _decode(row)
            if y is None:
                continue
            _write(emo, y, sec, text, row)
            continue

        # ---- auto-select first-N per emotion ------------------------------
        if all(len(kept[e]) >= args.per_emotion for e in targets):
            break
        emo = (row.get("assigned_emo") or "").strip().lower()
        if emo not in kept or len(kept[emo]) >= args.per_emotion:
            continue
        if float(row.get("agreement") or 0.0) < args.min_agreement:
            continue
        text = (row.get("script_sent") or "").strip()
        if not text:                     # improvised turn: no known transcript, skip
            continue
        if not _gender_ok(row.get("actor_gender"), args.gender):
            continue
        y, sec = _decode(row)
        if y is None or not (args.min_seconds <= sec <= args.max_seconds):
            continue
        _write(emo, y, sec, text, row)

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[refs] scanned {scanned} rows -> wrote to {args.out_dir}")

    if want_ids:
        for e in SUPPORTED:
            if kept[e]:
                print(f"[refs]   {e:11s} {len(kept[e])}")
        if remaining:
            print(f"[refs] WARNING: audio_id(s) not found in first {scanned} rows: "
                  f"{sorted(remaining)} (raise --max-scanned)", file=sys.stderr)
            return 1
        return 0

    missing = []
    for e in targets:
        got = len(kept[e])
        flag = "" if got >= args.per_emotion else "  << SHORT"
        print(f"[refs]   {e:11s} {got}/{args.per_emotion}{flag}")
        if got == 0:
            missing.append(e)
    if missing:
        print(f"[refs] WARNING: found nothing for {missing}. "
              f"Lower --min-agreement or raise --max-scanned.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    # Thai transcripts and the progress dashes are printed; the default Windows
    # console codepage cannot encode them and would raise mid-run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    return build(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
