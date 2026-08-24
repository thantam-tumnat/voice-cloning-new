"""Evaluation: Thai CER (Typhoon-Whisper-Large-v3) + speaker SIM (WavLM) + digit-eval.

Mirrors JaiTTS's `cal_wer.sh` / `cal_sim.sh` pipeline (RESEARCH.md §8.7 step 8):

- **CER** — transcribe synthesized audio with a Thai-capable Whisper
  (default `typhoon-ai/typhoon-whisper-large-v3`, the ASR JaiTTS evaluated
  with) and compute character error rate vs the prompt text. Whitespace and
  punctuation are stripped from both sides first, the usual Thai CER protocol.
- **SIM** — cosine similarity between WavLM x-vectors
  (`microsoft/wavlm-base-plus-sv`) of the synthesized audio and the reference
  clip, per prompt with a `ref_audio` column.

Requires the `eval` extra: `uv sync --extra eval`.

Prompt TSV format (eval/prompts_*.tsv):
    id<TAB>text<TAB>ref_audio_path
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

DEFAULT_ASR_MODEL = "typhoon-ai/typhoon-whisper-large-v3"
DEFAULT_SV_MODEL = "microsoft/wavlm-base-plus-sv"
SV_SAMPLE_RATE = 16_000

# Strip whitespace + punctuation before CER (Thai has no word spaces; spacing
# and punctuation variation would otherwise dominate the error count).
_CER_STRIP = re.compile(r"[\s\.,!\?;:\"'“”‘’\(\)\[\]…\-—_/\\]+")


def load_prompts(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def synthesize_all(
    prompts: list[dict[str, str]],
    out_dir: Path,
    *,
    base_model: str,
    adapter_path: str | None = None,
    **synth_kwargs,
) -> None:
    from .inference import Synthesizer

    out_dir.mkdir(parents=True, exist_ok=True)
    synth = Synthesizer(base_model=base_model, adapter_path=adapter_path)
    for row in prompts:
        synth.synth_to_file(
            row["text"],
            out_dir / f"{row['id']}.wav",
            ref_audio=row.get("ref_audio") or None,
            **synth_kwargs,
        )


def _cer_normalize(text: str) -> str:
    return _CER_STRIP.sub("", text)


def compute_cer(
    audio_dir: Path,
    prompts: list[dict[str, str]],
    asr_model: str = DEFAULT_ASR_MODEL,
    device: str | None = None,
) -> float:
    """Transcribe synthesized audio and return corpus-level CER vs prompt text.

    Drives Whisper via processor + model directly (not transformers'
    `pipeline`), feeding soundfile-decoded 16 kHz arrays. This avoids the
    pipeline's mandatory torchcodec audio backend, whose binary links against a
    CUDA-13 lib (libnvrtc.so.13) absent in our cu128 build. Each eval prompt is
    <30 s, so a single Whisper window suffices (no chunking).
    """
    import torch
    from jiwer import cer
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if "cuda" in str(device) else torch.float32
    processor = AutoProcessor.from_pretrained(asr_model)
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(asr_model, torch_dtype=dtype)
        .to(device)
        .eval()
    )

    refs: list[str] = []
    hyps: list[str] = []
    for row in prompts:
        wav_path = audio_dir / f"{row['id']}.wav"
        if not wav_path.exists():
            print(f"[cer] missing {wav_path}, skipping")
            continue
        wav = _load_16k(wav_path).numpy()
        feats = processor(
            wav, sampling_rate=SV_SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device, dtype)
        with torch.no_grad():
            ids = model.generate(
                feats, language="th", task="transcribe", max_new_tokens=256
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        refs.append(_cer_normalize(row["text"]))
        hyps.append(_cer_normalize(text))

    if not refs:
        raise ValueError(f"No synthesized audio found in {audio_dir}")
    return float(cer(refs, hyps))


def _load_16k(path: Path):
    import soundfile as sf
    import torch
    import torchaudio

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    t = torch.from_numpy(wav)
    if t.dim() > 1:
        t = t.mean(dim=1)
    if sr != SV_SAMPLE_RATE:
        t = torchaudio.functional.resample(t, sr, SV_SAMPLE_RATE)
    return t


def compute_sim(
    audio_dir: Path,
    prompts: list[dict[str, str]],
    sv_model: str = DEFAULT_SV_MODEL,
    device: str | None = None,
) -> float:
    """Mean speaker cosine similarity (gen vs ref) over prompts with ref_audio."""
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(sv_model)
    model = WavLMForXVector.from_pretrained(sv_model).to(device).eval()

    def embed(path: Path):
        wav = _load_16k(path)
        inputs = extractor(wav.numpy(), sampling_rate=SV_SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            return model(**inputs).embeddings.squeeze(0)

    sims: list[float] = []
    for row in prompts:
        ref = row.get("ref_audio")
        if not ref:
            continue
        gen_path = audio_dir / f"{row['id']}.wav"
        ref_path = Path(ref)
        if not gen_path.exists() or not ref_path.exists():
            print(f"[sim] missing {gen_path if not gen_path.exists() else ref_path}, skipping")
            continue
        sims.append(
            float(torch.nn.functional.cosine_similarity(embed(gen_path), embed(ref_path), dim=0))
        )

    if not sims:
        raise ValueError("No (generated, reference) pairs found — do prompts have ref_audio?")
    return sum(sims) / len(sims)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate SiangTTS checkpoints.")
    p.add_argument("--prompts", required=True, help="TSV file with id/text/ref_audio columns")
    p.add_argument("--out-dir", default="eval/out", help="Where to write synthesized audio")
    p.add_argument("--adapter", default=None, help="LoRA adapter path; omit for base-only")
    p.add_argument("--base-model", default="openbmb/VoxCPM2")
    p.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    p.add_argument("--skip-synth", action="store_true",
                   help="Reuse already-synthesized audio in --out-dir")
    p.add_argument("--cer", action="store_true", help="Compute Thai CER after synthesis")
    p.add_argument("--sim", action="store_true", help="Compute speaker SIM after synthesis")
    args = p.parse_args()

    prompts = load_prompts(args.prompts)
    out_dir = Path(args.out_dir)
    if not args.skip_synth:
        synthesize_all(
            prompts,
            out_dir,
            base_model=args.base_model,
            adapter_path=args.adapter,
        )

    if args.cer:
        print(f"CER: {compute_cer(out_dir, prompts, asr_model=args.asr_model):.4f}")
    if args.sim:
        print(f"SIM: {compute_sim(out_dir, prompts):.4f}")


if __name__ == "__main__":
    main()
