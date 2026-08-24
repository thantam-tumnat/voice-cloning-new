"""Sweep every knob that could make an audio tag actually land, and measure each.

The studio's complaint is that "[angry]" comes back barely angry in Thai while the
same tag in English is unmistakable. tools/lora_ab.py already showed the Thai LoRA
compresses the dynamic range, but it only tested one knob at one setting. This tool
turns the rest of the suspects into a grid:

  * LoRA strength, LM side and DiT side independently -- LoRALinear keeps its
    scaling in a buffer, so every setting is reachable on one loaded model instead
    of one model load per condition;
  * cfg_value, the classifier-free-guidance weight and therefore the literal
    "how hard do you follow the conditioning" dial;
  * the language the direction is written in, since the LoRA moved the LM toward
    Thai while the instruction stayed English;
  * punctuation in the spoken body, which VoxCPM2 reads as prosody.

Every condition renders the same sentence under the same emotion set, from the same
seed voice, with the same random seeds, so conditions are paired rather than merely
averaged. Measurement is of raw model output -- audio_post's per-tone gain and
stretch would otherwise mask exactly what is being measured.

Usage:
    python tools/expr_sweep.py --stage lora --out scratch/sweep
    python tools/expr_sweep.py --stage cfg  --out scratch/sweep
    python tools/expr_sweep.py --stage text --out scratch/sweep
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf

from app.renderers.voxcpm import resolve_style_tag
from app.services.audio_post import trim_silence
from app.services.siangtts_service import siangtts_service
from tools.prosody_eval import segment_metrics

# Emotion-compatible on purpose. lora_ab.py used a sentence about two files having
# identical digital forensics, which no speaker has ever been angry about; VoxCPM2
# conditions on the text's own semantics, so that sentence was quietly fighting
# every direction it was given.
SENTENCE = "เขาบอกฉันแบบนั้นจริงๆ นะ ฉันไม่ได้คิดไปเองสักหน่อย"

# Thai renderings of the English directions the app ships. Same content, same
# register -- the only variable is the language the engine reads it in.
THAI_INSTRUCTION = {
    "angry": "(น้ำเสียงโกรธ ดุดัน เกรี้ยวกราด)",
    "sad": "(น้ำเสียงเศร้า หม่นหมอง ถอนหายใจเบาๆ)",
    "happy": "(น้ำเสียงสดใส ร่าเริง ยิ้มขณะพูด)",
    "scared": "(น้ำเสียงหวาดกลัว สั่นเครือ หายใจถี่)",
}

# The same directions at intensity 3, mirroring VOXCPM_INSTRUCTION_MAP's level 3.
THAI_INSTRUCTION_L3 = {
    "angry": "(น้ำเสียงโกรธจัด ตะโกนเสียงดัง ดุดันรุนแรง)",
    "sad": "(น้ำเสียงเศร้าสลด ร้องไห้ สั่นเครือ)",
    "happy": "(น้ำเสียงดีใจสุดขีด หัวเราะร่าเริง)",
    "scared": "(น้ำเสียงหวาดผวา ตื่นตระหนก หายใจไม่ทัน)",
}

# Punctuation the body gets in the "punct" text variant. VoxCPM2 reads these as
# prosody marks, which is a channel the style parenthetical does not use.
PUNCT_SUFFIX = {"angry": "!", "sad": "...", "happy": "!", "scared": "!"}

EMOTIONS = ["angry", "sad", "happy", "scared"]

# What each emotion should do relative to the same take's neutral read. Signs come
# from the ElevenLabs reference in prosody_eval; magnitudes are not asserted, only
# direction, so a condition scores for moving the right way and is penalised for
# moving backwards.
EXPECT: Dict[str, Dict[str, float]] = {
    "angry": {"energy": +1.0, "f0": +1.0, "spread": +1.0, "pace": -1.0},
    "sad": {"energy": -1.0, "f0": -1.0, "spread": +0.5, "pace": +1.0},
    "happy": {"energy": +0.5, "f0": +1.0, "spread": +1.0, "pace": -1.0},
    "scared": {"energy": +1.0, "f0": +1.0, "spread": +1.0, "pace": -0.5},
}

# Divisors that put each metric on a roughly comparable "one unit = clearly audible"
# scale, so the score is not dominated by whichever metric has the largest numbers.
NORM = {"energy": 3.0, "f0": 2.0, "spread": 3.0, "pace": 8.0}


class Condition(NamedTuple):
    name: str
    lm: float = 2.0          # LoRA scaling on the language model side (2.0 = shipped)
    dit: float = 2.0         # LoRA scaling on the acoustic DiT side
    cfg: float = 2.5
    steps: int = 10
    lang: str = "en"         # direction language: en | th | both
    punct: bool = False
    level: int = 2           # intensity wording; 2 is what a bare "[angry]" gets
    voice: str = "seed"      # seed = the neutral auto seed voice, none = unconditioned


STAGES: Dict[str, List[Condition]] = {
    # What the LoRA itself costs, split by where it was injected.
    "lora": [
        Condition("base_lora_2.0", lm=2.0, dit=2.0),
        Condition("lora_1.0", lm=1.0, dit=1.0),
        Condition("lora_0.5", lm=0.5, dit=0.5),
        Condition("lora_off", lm=0.0, dit=0.0),
        Condition("lm_only", lm=2.0, dit=0.0),
        Condition("dit_only", lm=0.0, dit=2.0),
    ],
    # Guidance weight, at the shipped LoRA and at a partial one.
    "cfg": [
        Condition("cfg2.5", cfg=2.5),
        Condition("cfg4.0", cfg=4.0),
        Condition("cfg6.0", cfg=6.0),
        Condition("cfg4.0_lora1.0", lm=1.0, dit=1.0, cfg=4.0),
        Condition("cfg6.0_lora1.0", lm=1.0, dit=1.0, cfg=6.0),
    ],
    # How the direction is written, at the shipped model.
    "text": [
        Condition("en", lang="en"),
        Condition("th", lang="th"),
        Condition("both", lang="both"),
        Condition("en_punct", lang="en", punct=True),
        Condition("th_punct", lang="th", punct=True),
    ],
    # Intensity wording. A bare "[angry]" resolves to level 2, so if level 3 lands
    # much harder the cheapest fix in the whole study is a default change.
    "word": [
        Condition("lvl2", level=2),
        Condition("lvl3", level=3),
        Condition("lvl3_th", level=3, lang="th"),
    ],
    # Everything else, anchored on the stage-1 winner (LoRA on the LM only). Stage 1
    # found the DiT side is what flattens the emotion, so these ask what is left to
    # gain once it is off, and how much of it can be given back for Thai acoustics.
    "tune": [
        Condition("lm_only", lm=2.0, dit=0.0),
        Condition("lm_dit0.5", lm=2.0, dit=0.5),
        Condition("lm_dit1.0", lm=2.0, dit=1.0),
        Condition("lm_cfg4.0", lm=2.0, dit=0.0, cfg=4.0),
        Condition("lm_cfg6.0", lm=2.0, dit=0.0, cfg=6.0),
        Condition("lm_lvl3", lm=2.0, dit=0.0, level=3),
        Condition("lm_th", lm=2.0, dit=0.0, lang="th"),
        Condition("lm_both", lm=2.0, dit=0.0, lang="both"),
        Condition("lm_punct", lm=2.0, dit=0.0, punct=True),
        Condition("lm_noprompt", lm=2.0, dit=0.0, voice="none"),
    ],
    # What the neutral seed voice costs. It exists to stop the speaker drifting
    # between chunks, but it conditions every take on an emotionless read.
    "voice": [
        Condition("seed_neutral", voice="seed"),
        Condition("no_prompt", voice="none"),
    ],
}


def instruction_for(emotion: str, lang: str, level: int = 2) -> Optional[str]:
    """The parenthetical that leads the text, in the requested language."""
    if emotion == "neutral":
        return None
    english = resolve_style_tag(emotion, str(level)).instruction
    thai = (THAI_INSTRUCTION_L3 if level >= 3 else THAI_INSTRUCTION)[emotion]
    if lang == "en":
        return english
    if lang == "th":
        return thai
    # Both: English first, since that is the register the base model was tuned in,
    # with the Thai restating it for the LoRA-shifted LM.
    return f"{english}{thai}"


def build_text(emotion: str, cond: Condition) -> str:
    body = SENTENCE
    if cond.punct and emotion != "neutral":
        body = body + PUNCT_SUFFIX[emotion]
    instruction = instruction_for(emotion, cond.lang, cond.level)
    return f"{instruction}{body}" if instruction else body


# --------------------------------------------------------------------------- #
# LoRA strength control
# --------------------------------------------------------------------------- #

def lora_modules(model) -> Dict[str, list]:
    """Split the injected LoRA layers into the LM side and the DiT side.

    Both sides were injected with the same rank and alpha, so nothing in the layer
    itself says which is which -- the split has to come from where they live.
    """
    from voxcpm.modules.layers.lora import LoRALinear

    def collect(root) -> list:
        return [m for m in root.modules() if isinstance(m, LoRALinear)]

    lm = collect(model.base_lm) + collect(model.residual_lm)
    dit = collect(model.feat_decoder.estimator)
    return {"lm": lm, "dit": dit}


def set_lora(mods: Dict[str, list], lm: float, dit: float) -> None:
    """Set LoRA strength per side. 2.0 is what the shipped alpha/r produces."""
    for m in mods["lm"]:
        m.scaling.fill_(lm)
    for m in mods["dit"]:
        m.scaling.fill_(dit)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def measure(wav: np.ndarray, sr: int, n_chars: int) -> Dict[str, float]:
    """Speech-only prosody of one take, paced per character so lengths compare."""
    speech = trim_silence(np.asarray(wav, dtype="float32"), sr)
    if speech.size == 0:
        speech = np.asarray(wav, dtype="float32")
    m = segment_metrics(speech, sr)
    m["pace_s_per_char"] = m["dur_s"] / max(1, n_chars)
    return m


def deltas(emotion_m: Dict[str, float], neutral_m: Dict[str, float]) -> Dict[str, float]:
    """How far an emotion take moved from the neutral read of the same text."""

    def st(a: float, b: float) -> float:
        return 12.0 * math.log2(a / b) if a > 0 and b > 0 else 0.0

    return {
        "energy": 20 * math.log10((emotion_m["rms"] + 1e-9) / (neutral_m["rms"] + 1e-9)),
        "f0": st(emotion_m["f0_med_hz"], neutral_m["f0_med_hz"]),
        "spread": emotion_m["f0_spread_st"] - neutral_m["f0_spread_st"],
        # Percent slower than neutral. Positive = slower, which is what "sad" wants.
        "pace": 100.0 * (emotion_m["pace_s_per_char"] / neutral_m["pace_s_per_char"] - 1.0),
    }


# Past this many normalized units a metric stops counting as "more expressive" and
# starts being a defect. Without the cap the score rewarded pathology: cfg 4.0 made
# the sad take 233% slower than neutral -- the model stalling, not grief -- and that
# one number outranked every condition whose anger actually worked.
SCORE_CAP = 2.0


def score(emotion: str, d: Dict[str, float]) -> float:
    """Signed projection of the measured move onto the direction it should have gone."""
    want = EXPECT[emotion]
    total = sum(abs(w) for w in want.values())
    if total == 0:
        return 0.0
    return sum(
        w * max(-SCORE_CAP, min(SCORE_CAP, d[k] / NORM[k]))
        for k, w in want.items()
    ) / total


def contrast(d_a: Dict[str, float], d_b: Dict[str, float]) -> float:
    """How far apart two emotions sit -- what a listener hears as 'different'."""
    return sum(abs(d_a[k] - d_b[k]) / NORM[k] for k in NORM) / len(NORM)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_condition(synth, cond: Condition, mods, seed_voice, sr: int, reps: int,
                  out_dir: Path) -> Dict:
    import torch

    set_lora(mods, cond.lm, cond.dit)
    labels = ["neutral"] + EMOTIONS
    takes: Dict[str, List[np.ndarray]] = {k: [] for k in labels}
    measures: Dict[str, List[Dict[str, float]]] = {k: [] for k in labels}

    for rep in range(reps):
        for label in labels:
            text = build_text(label, cond)
            # Paired across conditions: the same rep gets the same noise everywhere,
            # so a difference between conditions is the condition, not the draw.
            torch.manual_seed(1000 + rep)
            t0 = time.time()
            wav = np.asarray(
                synth.synth(
                    text=text,
                    prompt_cache=seed_voice if cond.voice == "seed" else None,
                    cfg_value=cond.cfg,
                    inference_timesteps=cond.steps,
                ),
                dtype="float32",
            )
            takes[label].append(wav)
            measures[label].append(measure(wav, sr, len(SENTENCE)))
            print(f"    {cond.name:<16} rep{rep} {label:<8} {time.time() - t0:5.1f}s",
                  flush=True)

    # Average the per-rep deltas rather than the per-rep audio: sampling noise moves
    # each take, and averaging waveforms would just smear them.
    per_emotion: Dict[str, Dict[str, float]] = {}
    for emotion in EMOTIONS:
        reps_d = [deltas(measures[emotion][r], measures["neutral"][r]) for r in range(reps)]
        per_emotion[emotion] = {k: float(np.mean([d[k] for d in reps_d])) for k in NORM}

    scores = {e: score(e, per_emotion[e]) for e in EMOTIONS}
    result = {
        "condition": cond._asdict(),
        "deltas": per_emotion,
        "scores": scores,
        "score_mean": float(np.mean(list(scores.values()))),
        "angry_vs_sad": contrast(per_emotion["angry"], per_emotion["sad"]),
        "raw": measures,
    }

    # One listenable file per condition: neutral first, then each emotion, 1 s apart.
    gap = np.zeros(int(sr * 1.0), dtype="float32")
    pieces: List[np.ndarray] = []
    for label in labels:
        if pieces:
            pieces.append(gap)
        pieces.append(takes[label][0])
    sf.write(str(out_dir / f"{cond.name}.wav"), np.concatenate(pieces), sr)
    return result


def print_table(results: List[Dict]) -> None:
    print(f"\n{'condition':<18}{'score':>7}{'angry':>7}{'sad':>7}{'happy':>7}"
          f"{'scared':>8}{'A-vs-S':>8}")
    for r in sorted(results, key=lambda r: -r["score_mean"]):
        s = r["scores"]
        print(f"{r['condition']['name']:<18}{r['score_mean']:>7.2f}{s['angry']:>7.2f}"
              f"{s['sad']:>7.2f}{s['happy']:>7.2f}{s['scared']:>8.2f}"
              f"{r['angry_vs_sad']:>8.2f}")

    print(f"\n{'condition':<18}{'emotion':<9}{'dE_dB':>8}{'df0_st':>8}"
          f"{'dspread':>9}{'dpace_%':>9}")
    for r in results:
        for emotion in EMOTIONS:
            d = r["deltas"][emotion]
            print(f"{r['condition']['name']:<18}{emotion:<9}{d['energy']:>8.2f}"
                  f"{d['f0']:>8.2f}{d['spread']:>9.2f}{d['pace']:>9.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--out", default="scratch/sweep")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", help="comma-separated condition names to run")
    args = ap.parse_args(argv)

    conditions = STAGES[args.stage]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        conditions = [c for c in conditions if c.name in wanted]

    out_dir = Path(args.out) / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[sweep] loading model ...", flush=True)
    synth = siangtts_service.get_synthesizer()
    sr = synth.sample_rate
    mods = lora_modules(synth.tts_model)
    print(f"[sweep] LoRA layers: lm={len(mods['lm'])} dit={len(mods['dit'])}", flush=True)

    # One seed voice for every condition, built at the shipped LoRA strength. The
    # speaker has to be a constant here or a strength change would also be a voice
    # change, and the scores would be measuring two things at once.
    set_lora(mods, 2.0, 2.0)
    seed_voice = siangtts_service._build_seed_voice(synth, sr, 2.5, 10)
    print(f"[sweep] seed voice: {'ok' if seed_voice is not None else 'FAILED'}", flush=True)

    results = []
    for cond in conditions:
        print(f"[sweep] {cond.name}: lm={cond.lm} dit={cond.dit} cfg={cond.cfg} "
              f"lang={cond.lang} punct={cond.punct}", flush=True)
        results.append(run_condition(synth, cond, mods, seed_voice, sr, args.reps, out_dir))

    # Leave the model as the app expects to find it.
    set_lora(mods, 2.0, 2.0)

    payload = {"stage": args.stage, "sentence": SENTENCE, "reps": args.reps,
               "results": results}
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(results)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
