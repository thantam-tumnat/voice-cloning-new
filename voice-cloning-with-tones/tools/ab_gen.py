"""Render the ElevenLabs reference script through the pipeline and score both joins.

Generates once and assembles the same chunks two ways -- the old flat 60 ms
butt-join and the current assembler -- so the comparison isolates the treatment
instead of also capturing the sampler's run-to-run variation.

The default script is the exact one behind the reference take: one Thai sentence,
read four times as sad / happy / scared / tired. Keeping it identical is what lets
tools/prosody_eval diff our contrast against theirs directly.

Usage:
    python tools/ab_gen.py
    python tools/ab_gen.py --speaker determination --out takes/
    python tools/ab_gen.py --script my_script.txt --labels calm,angry
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf

from app.renderers.voxcpm import split_style_chunk_specs
from app.services.audio_post import assemble_with_spans, butt_join_with_spans
from app.services.siangtts_service import SynthesizerUnavailable, siangtts_service
from tools.prosody_eval import analyze, report

# The sentence and tag layout of the reference take, reproduced exactly.
SENTENCE = (
    "ทั้งสองไฟล์มีโครงสร้าง เนื้อหา "
    "และผลการตรวจจับทางดิจิทัลที่แทบจะถอดรหัสออกมาเหมือนกันทุกประการครับ"
)
DEFAULT_SCRIPT = (
    f"[sad] {SENTENCE}\n"
    f"[happy] {SENTENCE} [scared]{SENTENCE}\n"
    f"[tired]{SENTENCE}"
)


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, format="WAV", subtype="PCM_16")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--script", help="file holding the tagged script (default: the reference script)")
    ap.add_argument("--speaker", help="registered speaker id to clone")
    ap.add_argument("--out", default="takes", help="output directory (default: takes/)")
    ap.add_argument("--cfg", type=float, default=2.5, help="VoxCPM2 cfg_value")
    ap.add_argument("--timesteps", type=int, default=10, help="VoxCPM2 inference timesteps")
    ap.add_argument("--labels", help="override the tone labels used in the report")
    args = ap.parse_args(argv)

    script = Path(args.script).read_text(encoding="utf-8") if args.script else DEFAULT_SCRIPT

    specs = split_style_chunk_specs(script)
    if not specs:
        print("Script carries no style tags; nothing to compare.", file=sys.stderr)
        return 2

    labels: List[str] = (
        [s.strip() for s in args.labels.split(",")] if args.labels else [s.tone for s in specs]
    )

    print(f"chunks: {len(specs)}")
    for spec, label in zip(specs, labels):
        mark = "para" if spec.break_before else "    "
        print(f"  [{mark}] {label:<9} {spec.text[:60]}...")
    print()

    try:
        chunks, sr = siangtts_service.render_chunks(
            [s.text for s in specs],
            speaker_id=args.speaker,
            cfg_value=args.cfg,
            inference_timesteps=args.timesteps,
            tones=[s.tone for s in specs],
            breaks=[s.break_before for s in specs],
        )
    except SynthesizerUnavailable as e:
        print(f"Cannot generate: {e}", file=sys.stderr)
        return 1

    if getattr(siangtts_service, "_using_mock", False):
        print(
            "WARNING: the sine-tone mock produced this audio. The numbers below "
            "describe a test tone, not speech.\n",
            file=sys.stderr,
        )

    out_dir = Path(args.out)
    before_audio, before_spans = butt_join_with_spans(chunks, sr)
    after_audio, after_spans = assemble_with_spans(chunks, sr)

    before_path = out_dir / "before_buttjoin.wav"
    after_path = out_dir / "after_assembled.wav"
    _write(before_path, before_audio, sr)
    _write(after_path, after_audio, sr)

    for title, path, spans in (
        ("BEFORE  (flat 60 ms join, no levelling)", before_path, before_spans),
        ("AFTER   (assembler: trim, level, pace, pause)", after_path, after_spans),
    ):
        print("=" * 78)
        print(title)
        print("=" * 78)
        print(report(analyze(str(path), labels=labels, boundaries=spans)))
        print()

    print(f"wrote {before_path}")
    print(f"wrote {after_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
