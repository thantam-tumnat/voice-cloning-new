"""Catch style directions that VoxCPM2 reads aloud instead of obeying.

The leading parenthetical is supposed to be direction, not speech. On some wordings
VoxCPM2 drops out of control mode and simply *voices it* -- the take opens with a
couple of seconds of spoken English before the Thai line starts. It is obvious the
moment anyone listens and invisible to everything else: no error, no warning, and
every prosody metric in tools/prosody_eval.py still reports a plausible number,
because the English is real speech with real pitch. That is how
"(Sad and melancholic voice, slight sighs)" shipped while leaking 6 takes out of 6.

There is no rule to encode. The trigger is the exact phrasing, not any one word:
the other eight tone directions leaked 0/24 against the same speaker, and tired@3
carries "heavy sighs" without leaking at all. So the only way to know is to render
the wording and listen to it -- which is what this does, with Whisper standing in
for the listening. Any Latin script in the transcript of a Thai line is the leak.

Run it after touching VOXCPM_INSTRUCTION_MAP or STYLE_VOCABULARY, and on whatever
the LLM tag converter starts emitting (`--instruction`), which is unbounded text
that reaches the engine exactly the same way.

Needs the GPU service up and a local Whisper snapshot; it never downloads one.
Run it with the *engine* interpreter (../voice-cloning/.venv), not the studio's --
the studio is a thin client of :8020 and has no torch to run ASR with. It reaches
the engine over HTTP for the same reason, so it depends on nothing in app/ beyond
the two instruction tables.

Usage:
    ../voice-cloning/.venv/Scripts/python tools/instruction_leak_audit.py --speaker thai_male2
    ../voice-cloning/.venv/Scripts/python tools/instruction_leak_audit.py --scope all --reps 5
    ../voice-cloning/.venv/Scripts/python tools/instruction_leak_audit.py \
        --instruction "(Sad voice, quiet and downcast)"
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf

from app.models import Tone
from app.renderers.voxcpm import STYLE_VOCABULARY, VOXCPM_INSTRUCTION_MAP

DEFAULT_SERVICE_URL = os.environ.get("VOXCPM_SERVICE_URL", "http://127.0.0.1:8020")

# The same sentence tools/expr_sweep.py sweeps on, for the same reason: VoxCPM2
# conditions on the text's own semantics, so a sentence that resists the direction
# fights the measurement. Pure Thai, so any Latin script in the transcript came
# from the direction rather than from the line.
SENTENCE = "เขาบอกฉันแบบนั้นจริงๆ นะ ฉันไม่ได้คิดไปเองสักหน่อย"

DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"

# Two runs of three or more Latin letters. One is noise -- Whisper will hand back a
# stray "Oh" or "Mm" on a breathy Thai onset -- but a leaked direction always
# arrives as a phrase ("Sad and Melancholic Voice Slight Sighs").
_LATIN_RUN_RE = re.compile(r"[A-Za-z]{3,}")


class Take(NamedTuple):
    label: str
    instruction: str
    rep: int
    duration: float
    transcript: str

    @property
    def leaked(self) -> bool:
        return len(_LATIN_RUN_RE.findall(self.transcript)) >= 2


def collect_instructions(scope: str) -> Dict[str, str]:
    """Label -> instruction, for everything that can lead a chunk."""
    out: Dict[str, str] = {}
    for tone in Tone:
        for intensity in (1, 2, 3):
            instruction = VOXCPM_INSTRUCTION_MAP.get(tone, {}).get(intensity)
            if instruction:
                out[f"{tone.value}@{intensity}"] = instruction
    if scope == "all":
        for word, (instruction, _family) in sorted(STYLE_VOCABULARY.items()):
            # A None entry means "use the family's instruction", already covered.
            if instruction and instruction not in out.values():
                out[word] = instruction
    return out


def load_asr(model_id: str):
    """Whisper, from the local snapshot only -- an audit should not pull 1.5 GB."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch.float32, local_files_only=True
    ).eval()
    return processor, model


def transcribe(processor, model, wav: np.ndarray, sample_rate: int) -> str:
    import torch

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    n = int(round(len(wav) * 16000 / sample_rate))
    wav16 = np.interp(
        np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav
    ).astype("float32")
    feats = processor(wav16, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(feats, task="transcribe", max_new_tokens=200)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def render(service_url: str, instruction: str, speaker_id: Optional[str],
           cfg_value: float, timesteps: int, lora: str):
    """One take, straight from the engine.

    Deliberately raw: no chunk splitting, no audio_post gain or rate matching. Those
    would be measuring the studio, and what is on trial here is the wording.
    """
    body = json.dumps({
        "chunks": [f"{instruction}{SENTENCE}"],
        "voice": ({"speaker_id": speaker_id, "allow_sidecar": False}
                  if speaker_id else {"seed": True}),
        "cfg_value": cfg_value,
        "timesteps": timesteps,
        "lora": lora,
        "output": {"mode": "npz"},
        "lane": "interactive",
        "client": "leak_audit",
    }).encode()
    request = urllib.request.Request(
        f"{service_url.rstrip('/')}/v2/jobs/render?wait=300",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=310) as response:
        raw = response.read()
        sample_rate = int(response.headers.get("X-Sample-Rate") or 0)
    if not sample_rate:
        raise RuntimeError(f"render failed: {raw[:400]!r}")
    return np.load(io.BytesIO(raw))["chunk_000"], sample_rate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speaker", default=None,
                   help="Pinned speaker id from ref/. Unpinned uses the seed voice.")
    p.add_argument("--scope", choices=("tones", "all"), default="tones",
                   help="'tones' audits VOXCPM_INSTRUCTION_MAP; 'all' adds STYLE_VOCABULARY.")
    p.add_argument("--instruction", action="append", default=[],
                   help="Audit this exact parenthetical instead of the tables. Repeatable.")
    p.add_argument("--reps", type=int, default=3,
                   help="Takes per instruction. A leak is usually 3/3, but 1/3 happens.")
    p.add_argument("--cfg-value", type=float, default=2.5)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--lora", default="shipped",
                   help="LoRA preset for the engine: shipped | tones | off.")
    p.add_argument("--service-url", default=DEFAULT_SERVICE_URL)
    p.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    p.add_argument("--out", default="scratch/leak_audit",
                   help="Where takes and index.json land, for listening to the leaks.")
    args = p.parse_args()

    if args.instruction:
        instructions = {f"arg{i}": s for i, s in enumerate(args.instruction)}
    else:
        instructions = collect_instructions(args.scope)
    if not instructions:
        print("Nothing to audit.", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_asr(args.asr_model)

    takes: List[Take] = []
    for label, instruction in instructions.items():
        for rep in range(args.reps):
            wav, sample_rate = render(
                args.service_url, instruction, args.speaker,
                args.cfg_value, args.timesteps, args.lora,
            )
            name = f"{label.replace('@', '_').replace(' ', '_')}_{rep}.wav"
            sf.write(str(out_dir / name), wav, sample_rate)
            take = Take(label, instruction, rep, len(wav) / sample_rate,
                        transcribe(processor, model, wav, sample_rate))
            takes.append(take)
            print(f"  {name:<34} {take.duration:5.2f}s "
                  f"{'LEAK' if take.leaked else '  ok'}  {take.transcript[:70]}",
                  flush=True)

    (out_dir / "index.json").write_text(
        json.dumps([t._asdict() | {"leaked": t.leaked} for t in takes],
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print(f"\n{'instruction':<52} {'leaked':>8}  {'mean dur':>8}")
    leaking = 0
    for label, instruction in instructions.items():
        group = [t for t in takes if t.label == label]
        n = sum(t.leaked for t in group)
        leaking += bool(n)
        flag = "  <-- reword" if n else ""
        print(f"{instruction[:52]:<52} {n:>4}/{len(group):<3} "
              f"{sum(t.duration for t in group) / len(group):8.2f}{flag}")

    print(f"\n{leaking}/{len(instructions)} instructions leaked. Takes in {out_dir}/")
    return 1 if leaking else 0


if __name__ == "__main__":
    raise SystemExit(main())
