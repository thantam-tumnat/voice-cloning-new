"""Check that every tone in the Tone enum actually changes the audio.

expr_sweep.py asks "which knob setting makes emotion land hardest", over the four
tones that were measured off the ElevenLabs reference. This asks the narrower
question the studio keeps running into: for each of the ten tones the annotator can
emit, does the instruction move the audio at all, and does it move the right way?

The bar is deliberately not "did it score well". A tone passes only if its movement
is larger than the model's own rep-to-rep noise, measured the only way that means
anything -- by rendering the *same* neutral text several times and seeing how far
apart those takes land. Every neutral pair becomes a null sample, and a tone's
delta is reported in standard deviations of that null. A tone that moves less than
its own sampling noise is not "a weak emotion", it is no emotion at all, and no
amount of per-tone gain in audio_post can put one back.

Measured on raw model output, before audio_post: the assembler applies TONE_ENERGY_DB
and TONE_DURATION_RATIO itself, so measuring after it would show those constants
being applied and tell us nothing about whether the *model* heard the instruction.

Usage:
    python tools/tone_audit.py --reps 4 --out scratch/tone_audit
    python tools/tone_audit.py --level 3 --reps 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf

from app.models import Tone
from app.renderers.voxcpm import resolve_style_tag
from app.services.siangtts_service import siangtts_service
from tools.expr_sweep import NORM, SENTENCE, deltas, lora_modules, measure, set_lora

TONES: List[str] = [t.value for t in Tone if t is not Tone.NEUTRAL]

# Which way each metric should move relative to neutral, and how much it counts.
#
# angry/sad/happy/scared mirror expr_sweep.EXPECT, which took its signs from the
# ElevenLabs reference. The other five have no reference take, so their signs are
# read off the constants audio_post already assumes for them -- TONE_ENERGY_DB for
# energy, TONE_DURATION_RATIO for pace -- with f0 and spread filled in by family.
# That makes this audit a consistency check between the model and the assembler:
# where they disagree, one of the two is wrong and the tone is flagged either way.
#
# 0 means "no expectation" -- the metric is still reported, just not scored. Energy
# is unscored for happy because the reference measured it at -0.5 dB while every
# intuition (and expr_sweep) says up; that disagreement is not this tool's to settle.
EXPECT: Dict[str, Dict[str, float]] = {
    "angry":     {"energy": +1.0, "f0": +1.0, "spread": +1.0, "pace": -1.0},
    "sad":       {"energy": -1.0, "f0": -1.0, "spread": +0.5, "pace": +1.0},
    "happy":     {"energy":  0.0, "f0": +1.0, "spread": +1.0, "pace": -1.0},
    "scared":    {"energy": +1.0, "f0": +1.0, "spread": +1.0, "pace": -0.5},
    "calm":      {"energy": -1.0, "f0": -1.0, "spread": -0.5, "pace": +1.0},
    "excited":   {"energy": +1.0, "f0": +1.0, "spread": +1.0, "pace": -1.0},
    "nervous":   {"energy": -0.5, "f0": +1.0, "spread": +1.0, "pace":  0.0},
    "sarcastic": {"energy": -0.5, "f0":  0.0, "spread": +1.0, "pace": +1.0},
    "tired":     {"energy": -1.0, "f0": -1.0, "spread": -0.5, "pace": +1.0},
}

# Two different questions, two different denominators, and conflating them is the
# easiest way to misread this table.
#
#   d = mean delta / (1 SD of a single take). "On the render in front of me, is the
#       emotion audible?" This is the studio's question. d < 1 means the tone moves
#       the audio less than re-rolling the same neutral line does, so on any given
#       take the emotion is buried in sampler noise.
#
#   t = mean delta / (SD / sqrt(reps)). "Averaged over many takes, is the effect
#       real at all?" A tone can be real (t > 2) and still inaudible (d < 1) -- a
#       consistent nudge inside a much wider spread. That combination is precisely
#       what "the emotion is only sometimes applied" feels like from the outside.
D_AUDIBLE = 1.0
T_REAL = 2.0


# The Thai probe sentence, and an English one carrying the same content in the same
# register. Isolating "is VoxCPM2 unemotional" from "is our Thai LoRA unemotional"
# needs the model asked in the language it was pretrained in, with the LoRA off.
BODY = {
    "th": SENTENCE,
    "en": "He really did say that to me. I did not just imagine it.",
}


def build_text(tone: Optional[str], level: int, lang: str = "th") -> str:
    body = BODY[lang]
    if tone is None:
        return body
    instruction = resolve_style_tag(tone, str(level)).instruction
    return f"{instruction}{body}" if instruction else body


def null_spread(neutral_m: List[Dict[str, float]]) -> Dict[str, float]:
    """Per-metric SD of neutral-vs-neutral deltas: the noise floor of one take.

    Every unordered pair of neutral reps contributes one sample, so n reps give
    n*(n-1)/2 -- 6 from 4 reps. Signs are symmetric by construction, so the SD is
    taken about zero rather than about the sample mean.
    """
    pairs = [deltas(neutral_m[i], neutral_m[j])
             for i, j in itertools.combinations(range(len(neutral_m)), 2)]
    return {
        k: float(np.sqrt(np.mean([p[k] ** 2 for p in pairs]))) or 1e-9
        for k in NORM
    }


def verdict(d: Dict[str, float], eff: Dict[str, float], t: Dict[str, float],
            want: Dict[str, float]) -> str:
    """How a listener would experience this tone, judged on the metrics in ``want``.

    AUDIBLE     a scored metric clears the single-take noise, in the right direction.
    REAL/BURIED consistent across reps but smaller than that noise -- present in the
                mean, inaudible on any one render.
    BACKWARDS   the only real movement runs opposite to what the tone should do.
    NONE        nothing moved, in the mean or otherwise.
    """
    scored = {k: w for k, w in want.items() if w != 0.0}
    right = lambda ks: {k for k in ks if np.sign(d[k]) == np.sign(scored[k])}
    audible = right({k for k in scored if abs(eff[k]) >= D_AUDIBLE})
    real = {k for k in scored if abs(t[k]) >= T_REAL}

    if audible:
        return "AUDIBLE" if len(audible) >= 2 else "AUDIBLE(1)"
    if right(real):
        return "REAL/BURIED"
    if real:
        return "BACKWARDS"
    return "NONE"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--level", type=int, default=2, choices=(1, 2, 3))
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--out", default="scratch/tone_audit")
    ap.add_argument("--only", help="comma-separated tones to run")
    ap.add_argument("--lang", default="th", choices=("th", "en"),
                    help="language of the spoken body (the direction stays English)")
    ap.add_argument("--lm", type=float, help="LoRA scale, LM side (default: shipped)")
    ap.add_argument("--dit", type=float, help="LoRA scale, DiT side (default: shipped)")
    ap.add_argument("--voice", default="seed", choices=("seed", "none"),
                    help="condition on the neutral seed voice, or nothing at all")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    args = ap.parse_args(argv)

    tones = TONES
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        tones = [t for t in tones if t in wanted]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[audit] loading model ...", flush=True)
    synth = siangtts_service.get_synthesizer()
    sr = synth.sample_rate
    mods = lora_modules(synth.tts_model)

    # Defaults are the shipped configuration, so the numbers describe what users
    # actually hear; --lm/--dit override it to ask what the base model can do.
    from app.config import settings
    lm = settings.siangtts_lora_lm_scale if args.lm is None else args.lm
    dit = settings.siangtts_lora_dit_scale if args.dit is None else args.dit

    # The seed voice is built at the shipped strength whatever the condition: it is
    # the constant speaker, and minting it under the condition would make a LoRA
    # change a voice change too, measuring two things at once.
    set_lora(mods, settings.siangtts_lora_lm_scale, settings.siangtts_lora_dit_scale)
    seed_voice = (siangtts_service._build_seed_voice(synth, sr, args.cfg, 10)
                  if args.voice == "seed" else None)
    set_lora(mods, lm, dit)
    print(f"[audit] lm={lm} dit={dit} cfg={args.cfg} level={args.level} "
          f"lang={args.lang} voice={args.voice} reps={args.reps} "
          f"seed={'ok' if seed_voice is not None else 'none'}", flush=True)

    import torch

    labels = ["neutral"] + tones
    takes: Dict[str, List[np.ndarray]] = {k: [] for k in labels}
    measures: Dict[str, List[Dict[str, float]]] = {k: [] for k in labels}

    for rep in range(args.reps):
        for label in labels:
            text = build_text(None if label == "neutral" else label,
                              args.level, args.lang)
            # Paired: rep r gets the same noise for every tone, so a difference
            # between tones is the instruction and not the draw.
            torch.manual_seed(1000 + rep)
            t0 = time.time()
            wav = np.asarray(
                synth.synth(text=text, prompt_cache=seed_voice,
                            cfg_value=args.cfg, inference_timesteps=10),
                dtype="float32",
            )
            takes[label].append(wav)
            measures[label].append(measure(wav, sr, len(BODY[args.lang])))
            print(f"  rep{rep} {label:<10} {time.time() - t0:5.1f}s "
                  f"{len(wav) / sr:5.2f}s", flush=True)

    null = null_spread(measures["neutral"])
    print("\n[audit] neutral-vs-neutral noise floor (1 SD): "
          + "  ".join(f"{k}={v:.2f}" for k, v in null.items()))

    rows = []
    for tone in tones:
        per_rep = [deltas(measures[tone][r], measures["neutral"][r])
                   for r in range(args.reps)]
        d = {k: float(np.mean([p[k] for p in per_rep])) for k in NORM}
        eff = {k: d[k] / null[k] for k in NORM}          # audible on one take?
        t = {k: eff[k] * float(np.sqrt(args.reps)) for k in NORM}   # real at all?
        rows.append({"tone": tone, "deltas": d, "d": eff, "t": t,
                     "verdict": verdict(d, eff, t, EXPECT[tone])})

    hdr = (f"\n{'tone':<11}{'verdict':<13}{'dE_dB':>8}{'df0_st':>8}{'dspread':>9}"
           f"{'dpace_%':>9}   {'d(E)':>6}{'d(f0)':>7}{'d(sp)':>7}{'d(pc)':>7}"
           f"  {'t(E)':>6}{'t(f0)':>7}{'t(sp)':>7}{'t(pc)':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        dd, e, t = r["deltas"], r["d"], r["t"]
        print(f"{r['tone']:<11}{r['verdict']:<13}{dd['energy']:>8.2f}{dd['f0']:>8.2f}"
              f"{dd['spread']:>9.2f}{dd['pace']:>9.1f}   {e['energy']:>6.1f}"
              f"{e['f0']:>7.1f}{e['spread']:>7.1f}{e['pace']:>7.1f}"
              f"  {t['energy']:>6.1f}{t['f0']:>7.1f}{t['spread']:>7.1f}{t['pace']:>7.1f}")

    audible = [r["tone"] for r in rows if r["verdict"].startswith("AUDIBLE")]
    print(f"\n{len(audible)}/{len(rows)} tones audible on a single take: "
          f"{', '.join(audible) if audible else 'none'}")
    for name in ("REAL/BURIED", "BACKWARDS", "NONE"):
        got = [r["tone"] for r in rows if r["verdict"] == name]
        if got:
            print(f"{name:<12}: {', '.join(got)}")

    # One listenable file: neutral first, then each tone, 1 s apart.
    gap = np.zeros(int(sr * 1.0), dtype="float32")
    pieces: List[np.ndarray] = []
    for label in labels:
        if pieces:
            pieces.append(gap)
        pieces.append(takes[label][0])
    stem = f"lvl{args.level}_{args.lang}{args.tag}"
    sf.write(str(out_dir / f"tones_{stem}.wav"), np.concatenate(pieces), sr)

    (out_dir / f"results_{stem}.json").write_text(
        json.dumps({"level": args.level, "cfg": args.cfg, "reps": args.reps,
                    "lm": lm, "dit": dit, "lang": args.lang,
                    "voice": args.voice, "sentence": BODY[args.lang],
                    "null": null, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
