# Thai Voice-Cloning TTS — Research Notes (Round 2: OmniVoice)

Companion to [`RESEARCH.md`](./RESEARCH.md). Same target: build a Thai voice-cloning TTS
on a single RTX 3090 24 GB, training data primarily `dubbing-ai/vaja-thai`, personal
use with possible free (non-commercial) publishing. This document evaluates
**OmniVoice** (k2-fsa, Apr 2026) as an alternative to the VoxCPM/JaiTTS path.

Notes dated **2026-05-10**. The OmniVoice repo was created 2026-03-31 and last pushed
2026-05-09, so everything here is fresh — but also new, and the public ecosystem
around it is still small.

---

## 1. OmniVoice — Overview

Repo: <https://github.com/k2-fsa/OmniVoice> · Model: <https://huggingface.co/k2-fsa/OmniVoice>
Paper: arXiv:2604.00688 · Demo: <https://zhu-han.github.io/omnivoice> · License: **Apache-2.0**
(both code and weights).

### Lineage

OmniVoice comes from **Daniel Povey's k2-fsa / Next-gen Kaldi group** — same team
behind icefall (Zipformer / Pruned-RNNT for ASR) and the Sherpa runtime. Authors:
Han Zhu, Lingxuan Ye, Wei Kang, Zengwei Yao, Liyong Guo, Fangjun Kuang, Zhifeng Han,
Weiji Zhuang, Long Lin, Daniel Povey.

OmniVoice is **not** a Zipformer or FSA-based system. It does *not* reuse icefall's
ASR architecture. It's a fresh TTS effort from the same lab — the connection to the
broader k2-fsa stack is organizational (same authors, same infra), not architectural.
Sherpa (the C++/ONNX runtime) does not yet appear to ship OmniVoice support as of
2026-05-10.

### Architecture

Per the README and paper abstract, OmniVoice is a **discrete non-autoregressive (NAR)
diffusion-language-model TTS**:

```
text + (ref audio tokens) ──► LLM (Qwen3-0.6B) ──► 8-codebook discrete acoustic tokens
                                                          │
                                                          ▼
                                            Higgs-Audio-v2 codec decoder ──► 24 kHz waveform
```

Concretely (sources: paper abstract; `train_config_emilia.json`; `data_preparation.md`):

| Component | Role |
|---|---|
| **Backbone LLM** | `Qwen/Qwen3-0.6B` — initialized from a pretrained text LLM. The paper says LLM init is what "ensures superior intelligibility." |
| **Audio tokenizer** | `eustlb/higgs-audio-v2-tokenizer` (~0.2B params, 24 kHz). Produces `[8, T]` discrete tokens — i.e., **8 codebooks**, vocab 1024 + 1 mask id (`audio_vocab_size: 1025`, `audio_mask_id: 1024`). The repo treats this as a fixed external codec. The exact RVQ/FSQ structure is **not documented** on the tokenizer's HF card; assume RVQ-style based on the 8-codebook layout. |
| **Generation** | Discrete-NAR with **diffusion-language-model masking**: `mask_ratio_range: [0.0, 1.0]`, `drop_cond_ratio: 0.1` (CFG drop), and a "full-codebook random masking strategy" per the paper. Inference is iterative denoising over masked positions; default `num_step=32` (or 16 for faster). |
| **Codebook weighting** | Loss weights `[8, 8, 6, 6, 4, 4, 2, 2]` across the 8 codebooks (residual layers get downweighted). |

Important framing: this is **codec-based** in the sense that audio is reconstructed
from discrete acoustic tokens. That puts OmniVoice in the same architectural family as
IndexTTS / XTTS / CosyVoice, not VoxCPM. The k2-fsa team's bet is that scale (581k h)
+ NAR diffusion + LLM init is enough to overcome the codec ceiling — see §3 for
whether that worked for Thai.

### Key features (per README)

- **646 languages**, **581k hours** total training data ([`docs/languages.md`](https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md)).
- **Zero-shot voice cloning** from a 3–10 s reference clip.
- **Voice design** — generate without a reference using attribute prompts
  ("female, low pitch, british accent"). README explicitly notes: *"Voice design was
  trained on Chinese and English data only. It can generalize to other languages, but
  results can be unstable for some low-resource languages."*
- **Non-verbal tags** (`[laughter]`, `[sigh]`, etc.) and English/Chinese pronunciation
  override (CMU phones / pinyin).
- **Inference**: RTF as low as **0.025** (40× real-time), 24 kHz output. README
  doesn't say which GPU the RTF was measured on. Default `num_step=32`.
- **Apache-2.0** model and code.

### What's NOT documented

- Total OmniVoice parameter count (Qwen3-0.6B is the LLM core, but the discrete-token
  heads + extra layers add unknown overhead — call it ~0.7–0.9B effective).
- VRAM for inference or training (no number anywhere in the README, paper abstract,
  or training docs).
- Per-GPU-type RTF.
- Per-language CER/WER. The paper abstract mentions only "state-of-the-art performance
  across Chinese, English, and diverse multilingual benchmarks." The repo's
  [`docs/evaluation.md`](https://github.com/k2-fsa/OmniVoice/blob/master/docs/evaluation.md)
  lists the test sets (LibriSpeech-PC, Seed-TTS en/zh, FLEURS 102-lang, MiniMax
  multilingual) and the WER backends (HuBERT/Whisper/Paraformer/Omnilingual-ASR), but
  the repo does not publish a results table.
- LoRA / PEFT support — `gh search code` for "lora" / "peft" inside the repo returns
  **zero hits**. Fine-tuning is **full SFT only** as shipped.

---

## 2. OmniVoice vs VoxCPM — head-to-head

| Axis | **VoxCPM 2 / JaiTTS path** (RESEARCH.md) | **OmniVoice** |
|---|---|---|
| Architecture family | Tokenizer-free continuous-latent + flow-matching (TSLM/FSQ/RALM/LocDiT/AudioVAE) | Discrete 8-codebook NAR diffusion-LM over Higgs-Audio-v2 codec |
| Backbone init | MiniCPM-4 (0.5B–2B) | Qwen3-0.6B |
| Total params | 0.5B / 0.6–0.75B / 2B | ~0.6B LLM core (full count not documented) |
| Output sample rate | 16 / 44.1 / **48 kHz** | **24 kHz** (fixed by codec) |
| Codec | None (continuous AudioVAE) | External fixed codec (Higgs-Audio-v2) |
| Generation | AR (TSLM/RALM) + flow-matching local diffusion | NAR masked diffusion over discrete tokens |
| Reported RTF | 0.15 (v1.5) / 0.30 (v2) on 4090; JaiTTS 0.11 | **0.025** (40×, GPU not stated) |
| Languages | CN/EN (0.5B); 30 langs incl. Thai (v2, ~2M h) | **646 langs incl. Thai** (581k h) |
| **Thai training hours** | **Minority slice of ~2M h v2 corpus** (exact unknown) | **10,499.77 h Thai** (per `docs/languages.md`) |
| Best published Thai CER | JaiTTS 1.94% short / 2.55% long (closed checkpoint) | **7.71% on FLEURS — worse than ground truth 6.98%** (paper Appendix C) |
| Voice cloning | Zero-shot via `ref_audio` | Zero-shot via 3–10 s `ref_audio` (+ optional `ref_text`, else auto-Whisper) |
| Voice design / attr-prompt | Not advertised | Yes (CN/EN trained, generalizes weakly) |
| Code-switching | VoxCPM v2 multilingual; JaiTTS handles raw EN inline | Cross-lingual cloning supported but README warns of accent leakage from ref's language |
| **Fine-tune scripts shipped** | Yes (LoRA + full) — `lora_ft_webui.py`, two YAMLs | Yes (full only) — `examples/run_finetune.sh` + JSONs |
| **LoRA / PEFT** | **Yes** (rank 32–64, alpha 64–128) | **No** (no `lora`/`peft` references in repo) |
| Init-from-checkpoint | Yes | Yes (`init_from_checkpoint: "k2-fsa/OmniVoice"`) |
| License (model + code) | Apache-2.0 | Apache-2.0 |
| Released training pipeline | Yes (Emilia recipe) | Yes (Emilia recipe + finetune recipe) |
| Stars / community (2026-05-10) | OpenBMB/VoxCPM well-established | k2-fsa/OmniVoice ~5.6k stars, ~6 weeks old |
| Open weights actually downloadable | Yes | Yes |

---

## 3. Thai support

**This is the headline result.** OmniVoice ships **10,499.77 hours of Thai** in its
training corpus per [`docs/languages.md`](https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md)
(line: `| 563 | Thai | th | tha | 10499.77 |`). That is:

- **~5× more Thai** than JaiTTS's reported ~10k internal corpus is roughly comparable
  to (numbers similar order of magnitude; OmniVoice's is from public/open data).
- **~20× more Thai** than the entire `vaja-thai` dataset you have available.
- Effectively the same scale of Thai exposure as the closed JaiTTS-v1.0 corpus, but
  with **open weights**.

### Thai training data sources (from paper Appendix A)

The paper lists 50 open-source datasets in Appendix A. No per-language breakdown is
given, but the datasets that most plausibly contribute the 10,499.77 Thai hours are:

| Dataset | Notes |
|---|---|
| **GigaSpeech 2** | Large-scale Southeast Asian scrape; explicitly covers Thai |
| **Common Voice** (Mozilla) | Crowd-sourced, has Thai speakers |
| **FLEURS** (Google) | Multilingual few-shot benchmark, Thai included |
| **Meta Omnilingual ASR Corpus (MMS)** | Scales to 1000+ languages, Thai included |
| **Emilia / Emilia-YODAS** | Multilingual web-crawl audio |

All are public, open-source datasets. No curated studio-quality Thai TTS corpus
(comparable to JaiTTS's internal pipeline) appears in the list. The 10k hours is
likely ASR-derived scraped speech rather than clean TTS-studio recordings — which
matters for voice-cloning quality.

### Published Thai CER — now available from paper

**Updated finding (paper read 2026-05-18):** The paper does publish Thai CER in
Appendix C (Table 10, FLEURS-Multilingual-102 benchmark):

| | Ground-truth CER | OmniVoice CER |
|---|---|---|
| **Thai** | **6.98%** | **7.71%** |

**Thai CER is worse than ground truth.** This places Thai among the "filled circle"
languages in Figure 4 — OmniVoice produces Thai speech that is *harder to transcribe
correctly* than the original FLEURS recordings. This is a significant quality signal.

From Table 3 (MiniMax-Multilingual-24 benchmark): Thai WER = 3.978, SIM-o = 0.841.
(WER rather than CER; different ASR backend; Whisper-large-v3.)

Caveats:

- FLEURS is a read-speech benchmark from a controlled source. Real-world Thai
  prosody and voice-cloning quality may differ.
- The CER is measured with Whisper-large-v3, which has known Thai recognition
  limitations. The true intelligibility gap may be smaller — or larger.
- Thai data composition is not broken down by source. The 10k hours is plausibly
  ASR-derived scrape, not curated TTS studio data, explaining the quality ceiling.
- "Voice design" attributes were trained CN/EN-only; for Thai you'd use plain
  voice-cloning with a Thai reference clip.

**Revised bottom line on Thai:** OmniVoice has the largest Thai training data of
any open TTS, but base-model Thai intelligibility is *below ground truth* on
FLEURS. The codec ceiling (§1 of RESEARCH.md) appears to be real here despite
scale. Compare with JaiTTS's 1.94% CER (beats human 1.98%) achieved via VoxCPM's
tokenizer-free architecture — a structural, not just a scale, advantage.

---

## 4. Fine-tuning on RTX 3090 24 GB

### What ships

- Full fine-tune recipe: [`examples/run_finetune.sh`](https://github.com/k2-fsa/OmniVoice/blob/master/examples/run_finetune.sh).
  Two stages: (0) tokenize audio with Higgs-Audio-v2-tokenizer into WebDataset tar
  shards; (1) `accelerate launch -m omnivoice.cli.train`.
- Fine-tune config: [`examples/config/train_config_finetune.json`](https://github.com/k2-fsa/OmniVoice/blob/master/examples/config/train_config_finetune.json):

  ```json
  {
    "llm_name_or_path": "Qwen/Qwen3-0.6B",
    "init_from_checkpoint": "k2-fsa/OmniVoice",
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "steps": 5000,
    "batch_tokens": 8192,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "warmup_ratio": 0.01,
    "drop_cond_ratio": 0.1,
    "language_ratio": 0.8,
    "num_audio_codebook": 8,
    "audio_vocab_size": 1025
  }
  ```

  (The `examples/README.md` says "Fine-tune LR 5e-5"; the actual JSON has `1e-5`.
  Treat the JSON as authoritative.)
- SDPA fallback config [`train_config_finetune_sdpa.json`](https://github.com/k2-fsa/OmniVoice/blob/master/examples/config/train_config_finetune_sdpa.json)
  for GPUs without working `flex_attention`. **The 3090 (Ampere) supports
  flex_attention with PyTorch ≥ 2.5**, so you can stay on the default.
- Default launch line:
  ```bash
  accelerate launch --gpu_ids "0,1" --num_processes 2 \
      -m omnivoice.cli.train ...
  ```
  (Default ships **2 GPUs** — single-GPU works by setting `--num_processes 1`.)

### What does NOT ship

- **No LoRA / PEFT.** `gh search code` over the repo for `lora` and `peft` returns
  zero matches. Fine-tuning is full-parameter SFT only. You could bolt on `peft`
  manually around the Qwen3 backbone, but that's unsupported and would skip the
  diffusion-LM head adapters.
- No DeepSpeed ZeRO offload config beyond a stub `ds_config_zero2.json` in
  `examples/config/`. Not exercised by the provided scripts.
- No officially published memory/throughput numbers per GPU.

### Will it fit on a 3090 24 GB?

**Honest answer: probably yes for full SFT, but tight.** Reasoning:

- Qwen3-0.6B in BF16 ≈ 1.2 GB weights. With AdamW master weights + gradients +
  optimizer state for full SFT, ~1.2 GB × ~6 ≈ **~7 GB just for the LLM**.
- Plus the diffusion-LM heads (8 × 1025 vocab projection + positional/conditioning
  layers): unknown but probably 1–3 GB more parameter-state.
- Activations at `batch_tokens=8192`, single packed sequence with `flex_attention`:
  hard to estimate without measuring, but Qwen3-0.6B at 8k tokens fits comfortably
  on a 24 GB card for inference, and BF16 + gradient checkpointing (not on by
  default in the config — would need to add) would handle training.
- Higgs-Audio-v2 codec is **only used at data-prep time** (Stage 0), not during
  training. The training step sees pre-tokenized integer arrays. That's a real
  VRAM advantage over end-to-end-codec systems.

**Recommended 3090-safe overrides** (none documented; these are extrapolations
from VoxCPM-style guidance and the config schema):

| Knob | Default | 3090-safe | Notes |
|---|---|---|---|
| `batch_tokens` | 8192 | **4096** (try first) or 6144 | Biggest VRAM lever. |
| `gradient_accumulation_steps` | 1 | **2 or 4** | Restore effective batch. |
| `mixed_precision` | bf16 | bf16 | Keep — 3090 supports BF16. |
| Gradient checkpointing | not in default config | enable manually | Trade ~20% wall-clock for ~30% activation VRAM. |
| `attn_implementation` | flex_attention | flex_attention | 3090 + PyTorch 2.5+ works. |

If even `batch_tokens=4096` OOMs, you can fall back to `attn_implementation=sdpa`
with `max_batch_size=32, max_sample_tokens=1500` — slower per step but lighter on
peak memory.

### Wall-clock estimate for a 3090

The fine-tune config uses **5,000 steps** as the default, designed for 2× GPUs at
`batch_tokens=8192/GPU`. Translating to single 3090 at `batch_tokens=4096` with
`grad_accum=4` (same effective tokens-per-update):

- Per-step time at `batch_tokens=4096` BF16 on a 3090: order of 1–3 s
  (Qwen3-0.6B-class models). Not measured on this exact recipe — verify in the first
  500 steps.
- 5,000 steps at 2 s/step → **~3 hours wall-clock for the default recipe**.
- For a Thai-focused fine-tune over the 543 h `vaja-thai` commercial-clean subset,
  you'll likely want more — say **10–30k steps** = ~6–18 h on the 3090. Still very
  tractable.

### Risks specific to single-3090 OmniVoice fine-tuning

1. **No LoRA path.** Every fine-tune touches all ~0.6–0.9B params. There's no cheap
   speaker-adapter run — the smallest unit of work is "another 5k-step full SFT."
2. **`flex_attention` quirks.** The team explicitly ships the SDPA fallback because
   `flex_attention` is fragile across CUDA / PyTorch versions. Plan for one debug
   afternoon.
3. **Tokenization step (Stage 0) is GPU-bound on Higgs-Audio-v2.** ~543 h Vaja
   should tokenize in a couple of hours on a 3090 at `nj_per_gpu=3`. One-time cost.
4. **Memory headroom for eval.** Default `eval_steps=500` — eval batches share VRAM
   with training; if you tune batch up to the limit, eval may OOM. Set lower
   eval batch or skip in-loop eval and run offline.

---

## 5. Vaja-Thai dataset compatibility

Recap of `dubbing-ai/vaja-thai`: 554.6 h total / 543 h commercial-clean
(`porjai_central` + `commonvoice`), 24 kHz WAV, clip length 1–30 s, Whisper-validated
transcripts, tier 1–4 quality column. Full breakdown in §5 of RESEARCH.md.

| Compatibility axis | OmniVoice fit |
|---|---|
| **Sample rate** | **Native match.** OmniVoice operates at 24 kHz (codec is fixed). Vaja is 24 kHz. Zero resampling, zero quality loss. This is OmniVoice's *biggest* advantage over VoxCPM 2 for this dataset. |
| **Manifest format** | JSONL with `id`, `audio_path`, `text`, optional `language_id`. Trivial port from CSV/HF format. Use `language_id: "th"`. |
| **Clip length** | Configurable via `min_length` / `max_length` in `extract_audio_tokens.py`. Vaja's 1–30 s range fits. |
| **Tier filtering / oversampling** | Done at JSONL-construction time. Same approach as RESEARCH.md §8.3 (tier 1 ×2, tier 2/3 ×1, drop tier 4). |
| **`language_id`** | The fine-tune config has `language_ratio: 0.8` — language ID is conditioned 80% of the time during training. Set `language_id: "th"` for all Vaja rows. |
| **Reference audio for cloning** | OmniVoice's training does **not** require explicit `ref_audio` rows in the manifest the way VoxCPM does — `prompt_ratio_range: [0.0, 0.3]` means up to 30% of the *front of each sample* is used as the speech prompt during training. That auto-creates self-prompted training without any per-sample pairing logic. This is simpler than the VoxCPM `ref_audio` 30–50 % pairing scheme. |
| **Text normalization** | OmniVoice README explicitly recommends external normalization for digits ("For better results with Arabic numerals, normalize them to words first … with WeTextProcessing"). Vaja is *already* digit-normalized to Thai words. **Vaja's preprocessing happens to match OmniVoice's input expectations directly** — no need for the §8.6 "digit augmentation" tricks from RESEARCH.md. |
| **Data scale** | 543 h of Thai → fine-tuning, not from-scratch. Well above any reasonable adapter floor. Below the 10k-h Thai already in the base model — i.e., you're nudging style/voice, not teaching the language. |

There is **one mismatch**: Vaja's Whisper-validated transcripts strip non-Thai chars
(no English code-switch present), so a Vaja-only fine-tune narrows the EN-in-Thai
code-switching the base model can do. Same fix as in RESEARCH.md §7.3: mix in
~50–100 h of LibriTTS-R `text_original` to keep EN alive. With `language_ratio=0.8`,
the model already expects multi-language conditioning during fine-tuning.

---

## 6. Recommendation — which path for *your* setup?

**Updated recommendation (2026-05-18, after reading the full paper): VoxCPM-LoRA is
the stronger default. OmniVoice is worth a quick base-eval but the paper's own Thai
numbers argue against it as primary path.**

### The decisive data point

The paper (Appendix C, Table 10) shows OmniVoice Thai CER on FLEURS-Multilingual-102:

> **OmniVoice Thai CER: 7.71% — worse than ground truth (6.98%)**

JaiTTS on the same task reports 1.94% (short-form) / 2.55% (long-form), beating human
ground truth. That is a **4× intelligibility gap** between OmniVoice base and the
VoxCPM-architecture result. The codec ceiling that §1 of RESEARCH.md warned about
appears real for Thai even at 10k-hour scale.

Furthermore, the 10.5k Thai hours are from ASR-derived open scrapes (GigaSpeech 2,
Common Voice, FLEURS, MMS) — not studio-quality TTS data. JaiTTS's internal corpus
was purpose-built for TTS quality. Data quality, not just quantity, matters.

### VoxCPM-LoRA: why it wins

1. **Architecture advantage is structural.** VoxCPM's tokenizer-free AudioVAE
   eliminates the codec reconstruction bottleneck entirely. JaiTTS's 1.94% CER
   is direct evidence this works for Thai.
2. **JaiTTS is a reproducible north star.** You know the architecture, the
   hyperparameters (RESEARCH.md §3 / §8), and the target CER. You can benchmark
   your fine-tune against a known result.
3. **LoRA on a single 3090 is practical.** ~17–20 GB VRAM with the §8.1 overrides;
   3–5 day run for 3 epochs. Fast iteration cycles.
4. **48 kHz output** — higher ceiling than OmniVoice's fixed 24 kHz codec.
5. **Well-documented fine-tuning** — webUI, two YAML configs, the entire RESEARCH.md
   §8 plan is ready to execute.

### OmniVoice: still worth a look for these reasons

1. **Voice cloning UX is simpler** — no `ref_audio` pairing logic; any 3–10 s clip
   works at inference. Good for a downstream demo product.
2. **Faster inference** — RTF 0.022–0.032 on H20 vs VoxCPM 2's ~0.30 on 4090.
   If you ship a real-time service, OmniVoice's NAR speed advantage matters.
3. **Fine-tune is faster per run** — 5k steps ≈ 3 h on 3090 for a quick experiment.
4. **24 kHz = zero Vaja resampling.** Small but real data-prep convenience.

### Revised concrete plan

1. **Default path: VoxCPM 2 LoRA** per RESEARCH.md §8.7. The architecture is proven
   for Thai (via JaiTTS), LoRA fits the 3090, and the plan is fully specified.
2. **Optional OmniVoice sanity check** (half a day): install OmniVoice, generate
   ~20 Thai sentences, listen. If audio sounds surprisingly clean despite the
   FLEURS numbers, it may be worth a 5k-step fine-tune experiment for comparison.
   The FLEURS benchmark uses read speech from one domain; your Vaja-Thai voices may
   behave differently.
3. **If voice-cloning UX or inference speed becomes the constraint** (e.g., building
   a demo service), revisit OmniVoice after your VoxCPM Thai LoRA is working —
   use the VoxCPM adapter as a quality bar and compare directly.

**Bottom line: start with VoxCPM 2 LoRA. OmniVoice's base Thai quality is below
ground truth per its own paper; VoxCPM's architecture structurally avoids the
problem. OmniVoice is worth knowing about for inference speed and cloning UX, but
not the right first bet for Thai intelligibility on a single 3090.**

---

## 7. References

- **OmniVoice paper**: Zhu, Han, Lingxuan Ye, Wei Kang, Zengwei Yao, Liyong Guo,
  Fangjun Kuang, Zhifeng Han, Weiji Zhuang, Long Lin, Daniel Povey. *OmniVoice:
  Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models.*
  arXiv:2604.00688 (v1 2026-04-01, v3 2026-04-21). License CC-BY-4.0.
- **OmniVoice repo**: <https://github.com/k2-fsa/OmniVoice> (Apache-2.0, ~5.6k stars,
  created 2026-03-31, last push 2026-05-09).
- **OmniVoice HF model card**: <https://huggingface.co/k2-fsa/OmniVoice>.
- **Demo page**: <https://zhu-han.github.io/omnivoice>.
- **Languages list (Thai = 10,499.77 h)**:
  <https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md>.
- **Audio tokenizer**: `eustlb/higgs-audio-v2-tokenizer`
  (<https://huggingface.co/eustlb/higgs-audio-v2-tokenizer>) — 24 kHz, 8 codebooks,
  ~0.2B params; further details not documented on the HF card.
- **LLM backbone**: `Qwen/Qwen3-0.6B`.
- **Companion document**: [`RESEARCH.md`](./RESEARCH.md) — VoxCPM/JaiTTS plan, the
  fallback path discussed in §6.
- **Vaja-Thai dataset**: <https://huggingface.co/datasets/dubbing-ai/vaja-thai>.
