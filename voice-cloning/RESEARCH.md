# Thai Voice-Cloning TTS — Research Notes

Research target: build a TTS model that voice-clones and speaks Thai natively.
Anchor reference: **JaiTTS** (arXiv:2604.27607, May 2026) — a Thai adaptation of **VoxCPM**.
Candidate dataset: [`dubbing-ai/vaja-thai`](https://huggingface.co/datasets/dubbing-ai/vaja-thai).

---

## 1. Why not IndexTTS

Per your own experience: IndexTTS isn't natively Thai-aware. Adding a LoRA on top of it
gave unsatisfactory quality. The likely reasons (also called out in the JaiTTS paper §1):

- IndexTTS-class systems lean on a **discrete neural audio codec** (X-codec2 etc.) whose
  training data was multilingual but **does not appear to include Thai**. The codec
  cannot faithfully reconstruct Thai-specific phonetics — lexical tone, consonant
  clusters — so a downstream LoRA inherits a lossy bottleneck it cannot fix.
- Thai is also a **small slice of pretraining data** in most "multilingual" open-source
  TTS, so prosody and pronunciation drift remain even after adapter tuning.

Conclusion: codec-based architectures cap how much a Thai LoRA can recover.
A **tokenizer-free / continuous-latent** architecture removes that ceiling — which is
what VoxCPM (and therefore JaiTTS) provides.

---

## 2. VoxCPM (the upstream / base model)

Repo: <https://github.com/OpenBMB/VoxCPM> · License: **Apache-2.0**.

### Architecture (4-stage hierarchical pipeline)

```
text ──► TSLM ──► FSQ ──► RALM ──► LocDiT ──► continuous latent patches ──► AudioVAE ──► waveform
              (semantic skeleton)   (residual          (flow-matching
                                     acoustic)          local diffusion)
```

| Component | Role |
|---|---|
| **TSLM** (Text-Semantic LM) | Decoder-only transformer **initialized from MiniCPM-4**. Plans semantic + prosodic content from BPE-tokenized text + reference-audio embeddings. |
| **FSQ** (Finite Scalar Quantization) | Compresses TSLM hidden state into a *semi-discrete skeleton* — acts as a regularizer rather than a vocabulary. Uses straight-through estimator so gradients flow end-to-end. |
| **RALM** (Residual Acoustic LM) | Decoder-only transformer that adds the speaker-identity / fine-acoustic detail the FSQ bottleneck cannot carry. |
| **LocDiT** (Local Diffusion Transformer) | Flow-matching denoiser that decodes the next latent patch from the FSQ skeleton + RALM residual + previous patch. |
| **AudioVAE** | Causal continuous VAE — replaces the discrete codec entirely. |

Training is **joint, end-to-end** (one combined loss `L_FM + λ·L_Stop`), unlike codec
pipelines that train tokenizer and decoder separately.

### Variants (current)

| Variant | Params | Sample rate | VRAM (inference) | RTF on RTX 4090 |
|---|---|---|---|---|
| VoxCPM-0.5B | 0.5B (MiniCPM4-0.5B backbone) | 16 kHz | ~5 GB | ~0.17 |
| VoxCPM 1.5 | ~0.6–0.75B | 44.1 kHz | ~6 GB | ~0.15 |
| VoxCPM 2 | 2B | 48 kHz | ~8 GB | ~0.30 |

### Languages

- **VoxCPM-0.5B (HF)**: bilingual Chinese + English only, **~1.8M hours**. Thai output
  is "not guaranteed."
- **VoxCPM 2** (newer release): claims 30 languages including Thai over **~2M hours** —
  but Thai is a minority slice and real-world Thai CER lags behind dedicated Thai work.
  This is exactly the gap JaiTTS targets.

### Built-in fine-tuning support

The repo ships with two configs and a WebUI (`lora_ft_webui.py`):

- `conf/voxcpm_v2/voxcpm_finetune_lora.yaml`
- `conf/voxcpm_v2/voxcpm_finetune_all.yaml`

Manifest is JSONL with `audio`, `text`, optional `ref_audio`, `duration`, `dataset_id`.
30–50% of samples should include `ref_audio` so zero-shot cloning capability is preserved.

---

## 3. JaiTTS — what's actually different from VoxCPM

The headline: **JaiTTS = VoxCPM architecture, continually trained on a Thai-centric
corpus.** No new layers, no architecture changes.

| Axis | VoxCPM (upstream) | JaiTTS-v1.0 |
|---|---|---|
| Architecture | TSLM + FSQ + RALM + LocDiT + AudioVAE | **Identical** ("adapted from VoxCPM") |
| Backbone init | MiniCPM-4 | Inherits VoxCPM weights → continual training |
| Languages | CN/EN (0.5B) or 30 langs (v2) | Thai-centric, with Thai-English code-switch |
| Training corpus | ~1.8–2M hrs multilingual | **~10,000 hrs** Thai-centric |
| Text normalization | External pipeline typical | **None** — model handles raw numerals + EN code-switch directly |
| Short-form Thai CER | ≥2.5% (best baseline 2.34%) | **1.94%** (beats human 1.98%) |
| Long-form (16–30 s) Thai CER | 3.6–6%+ | **2.55%** vs human 2.47% |
| RTF | 1.5+ for AR baselines | **0.1136** (~9× real-time) |
| Code release | Full training + inference | **Eval/benchmark code only** — no training scripts, no checkpoint published as of paper |

### Training corpus composition (JaiTTS, §3.1)

- ~10,000 hours of Thai speech.
- Built from an **internal data pipeline** (curation, cleaning, ASR-aided transcription,
  multi-step post-processing for transcript verification).
- Mix of **broad-domain general speech** (podcasts, conversational + formal) and
  **vertical-specific** content for stylistic + terminological coverage:
  *Finance, Healthcare, Education, Law*.
- Acoustic conditions deliberately mixed: studio-grade **and** crowd-sourced for
  prosodic + speaker diversity.

### Training procedure (JaiTTS, §2 + §3.1)

- **Continual training** of the VoxCPM checkpoint (i.e., language adaptation, not from
  scratch). Joint optimization of all four modules + FSQ end-to-end.
- Loss `L = L_FM + λ·L_Stop` (flow-matching + stop-prediction).
- **Classifier-free guidance**: LM conditioning into LocDiT randomly dropped with
  probability 0.1 during training.
- Inference settings reported in the paper: `cfg_value = 2.5`, `inference_timesteps = 10`.

### What the paper does NOT disclose

- Exact GPU count, GPU type, wall-clock training time.
- Optimizer / learning rate schedule.
- Which VoxCPM variant they continued from (most plausible: the 2B v2, given 48 kHz +
  multilingual coverage).
- Total parameter count.
- Whether the trained checkpoint will be released (repo currently: benchmark only).

---

## 4. Training-setup expectations (what you'd need to reproduce)

These come from the **VoxCPM official fine-tuning guide** (the JaiTTS paper itself
withholds compute details, so this is the credible upper-bound for your own run).

### A. Hardware / VRAM

| Variant | LoRA | Full SFT |
|---|---|---|
| VoxCPM 1.5 (~0.7B) | ~12 GB | ~24 GB |
| VoxCPM 2 (2B) | ~20 GB | ~40 GB |

Numbers assume batch_size=16, max_batch_tokens=8192. So:

- **LoRA on VoxCPM 2**: a single RTX 4090 / A6000 (48 GB) is comfortable.
- **Full SFT on VoxCPM 2**: needs A100-40G+ minimum, or multi-GPU. JaiTTS-grade
  results (10k hrs, full SFT) realistically want **≥4× A100/H100** for a multi-day run.

### B. Hyperparameters (VoxCPM defaults)

| | LoRA | Full SFT |
|---|---|---|
| LR | 1e-4 | 1e-5 |
| LoRA rank | 32 (speaker) / 64 (style or language) | — |
| LoRA alpha | r or 2r | — |
| Batch size | 16 | 16 |
| Warmup steps | 100 | 100 |
| Weight decay | — | 0.01 |
| Epochs | 1–3 | 1–2 |

### C. Data scale heuristics (VoxCPM docs)

| Goal | Data | Approach |
|---|---|---|
| Single speaker clone | 5–50 clips (5–10 min) | LoRA |
| Domain/style adapt | 50–500 clips | LoRA r=32–64 |
| **New language** | **500+ hours** | **Full SFT, with 10–20 % CN/EN mix to prevent forgetting** |
| Large customization | 1000+ clips | Full SFT |

JaiTTS used **~10,000 hours** (well above the 500-hour floor) which is why it can beat
human ground truth — most users targeting Thai will not need that much.

### D. Audio-prep gotchas (from the VoxCPM guide)

- Clip length **3–30 s**; <1 s is unstable.
- **Trim trailing silence to <0.5 s** — long trailing silence is the #1 cause of
  "model never stops generating" after fine-tuning.
- WAV preferred; auto-resamples to model rate.

---

## 5. Vaja-Thai dataset — fit for this job

Source: <https://huggingface.co/datasets/dubbing-ai/vaja-thai>

| | |
|---|---|
| Total | 289,916 samples / **554.6 hours** |
| Sample rate | 24 kHz, WAV 16-bit PCM |
| Clip length | 1.0–30.0 s |
| Validation | Whisper CER ≤ 0.15, RMS > −50 dBFS, <1% clipping |
| Quality tiers | 1–4 (CER + SNR + provenance) |

### Source breakdown & licensing (this matters)

| Source | Hours | License | Commercial use? |
|---|---|---|---|
| `tsync2` | 3.7 | CC-BY-NC-SA-3.0 | **No** |
| `porjai_central` | 412.5 | CC-BY-SA-4.0 | Yes (share-alike) |
| `gigaspeech2` | 8.1 | Non-commercial research only | **No** |
| `commonvoice` | 130.3 | CC0 | Yes |

When combined, **the most restrictive license dominates** — the `all` config is
effectively non-commercial. For a commercially deployable model, restrict training to
`porjai_central` + `commonvoice` ≈ **543 hours**.

### Suitability vs the JaiTTS recipe

**The good:**

- 543 hrs (commercial-clean) ≥ VoxCPM's **500-hour floor** for new-language
  adaptation. Right at threshold.
- Whisper-validated transcripts mean you can feed text/audio pairs straight into the
  manifest with minimal cleaning.
- 24 kHz native — clean upsample to 48 kHz (VoxCPM 2) or downsample to 16 kHz
  (VoxCPM-0.5B). VoxCPM auto-resamples either way.
- Tiered quality column lets you build a curriculum: tier ≤ 2 for pre-flight test
  runs, tiers 3–4 for bulk pretraining.
- Single-speaker subset (`tsync2`) is useful for an initial LoRA voice clone test.

**The gaps vs. JaiTTS's 10k-hour corpus:**

- **~20× less data.** Expect CER above 1.94% — but you can still likely beat
  vanilla VoxCPM and most open Thai TTS.
- **No domain breadth** — Vaja-Thai is general/conversational; JaiTTS deliberately
  added Finance/Healthcare/Education/Law verticals for terminology coverage.
- **Speaker diversity** is mostly proxy IDs, not verified — fine for general Thai
  pronunciation/prosody, weaker for diverse speaker timbre coverage.
- **No code-switch corpus.** JaiTTS handles raw EN words + Arabic numerals because the
  training set contained them. Vaja-Thai's transcript pipeline strips non-Thai
  characters and converts digits to Thai number words — meaning your model will
  *not* learn EN code-switching from this data alone. To match JaiTTS behavior you'd
  need to either (a) preserve raw text on a subset, or (b) augment with a code-switched
  corpus.

---

## 6. Recommended path forward

Two reasonable strategies depending on resources:

### Strategy A — "Thai LoRA on VoxCPM 2" (low-cost, fast)

- Base: `VoxCPM 2` (Thai is *nominally* supported, just weak).
- Adapter: LoRA, r=64, alpha=128, lr=1e-4, 1–3 epochs.
- Data: ~50–200 hours from Vaja-Thai's high-quality tier (≤ 2) + ~20–40 h LibriTTS-R/VCTK.
- Hardware: 1× RTX 4090 / A6000 48 GB.
- Wall-clock: roughly hours-to-a-couple-days.
- Expected: cleaner Thai prosody than vanilla VoxCPM 2, but won't reach JaiTTS CER.

### Strategy B — "Full continual pretrain, JaiTTS-style" (higher cost, near-SOTA)

- Base: `VoxCPM 2` weights.
- Method: full SFT on **all four modules + FSQ + LocDiT** end-to-end.
- Data: full Vaja-Thai (~554 h, NC license is fine for personal/free-publish) +
  ~80–120 h LibriTTS-R + ~40 h VCTK to prevent EN forgetting and re-establish
  code-switching.
- Hyperparams: lr=1e-5, weight decay 0.01, batch 16, 1–2 epochs, CFG drop 0.1.
- Hardware: 4–8× A100/H100, multi-day.
- Add later: code-switch + numeric data if you want JaiTTS's "no normalization" trick.

**Suggested first move:** run Strategy A for a few days on tier-1/2 Vaja-Thai
(~200 h) + a small EN slice (~30 h LibriTTS-R) to confirm the architecture is the
right substrate, then escalate to Strategy B if quality is insufficient.

---

## 7. Open questions — answered (2026-05-08)

### 7.1 Will JaiTTS release the v1.0 checkpoint?

**Likely no, at least not soon.** Direct evidence:

- HuggingFace org `JTS-AI` only hosts:
  - `JTS-AI/JaiTTS-F5TTS` (uploaded 2026-04-21) — explicitly tagged as *"Research
    prototype: one of our experimental variants… released for research and
    benchmarking only,"* finetuned from `SWivid/F5-TTS`. **This is NOT JaiTTS-v1.0.**
    Its CER (4.26% short / 11.57% long) is much worse than v1.0 (1.94% / 2.55%).
  - `JTS-AI/OpenJAI-v1.0-14B` — unrelated text-only LLM.
- GitHub repo (`JTS-AI-Team/JaiTTS`) has **0 releases**, **0 issues**, **2 stars**, and
  contents = eval/benchmark scripts only (`cal_wer.sh`, `cal_sim.sh`, `run_wer.py`,
  `average_wer.py`, etc.). No training code, no checkpoint, no roadmap, no
  "coming soon" notes anywhere.
- Live demo exists (`https://jaitts-demo.jts.co.th/`) → JaiTTS is deployed as a
  **product**, not as open weights. Jasmine Technology Solution is a commercial
  vendor; the F5-TTS variant being labeled "research-only" while the strong v1.0 is
  withheld is the typical pattern for "paper + demo, weights stay closed."

Practical implication: assume you will **not** be fine-tuning from JaiTTS-v1.0.
You'll fine-tune VoxCPM directly.

### 7.2 Which VoxCPM variant did JaiTTS continue from?

**Almost certainly the larger VoxCPM (the v2 / 2B family), not the 0.5B.** The paper
itself doesn't say it explicitly, but the indirect evidence is consistent:

- The original VoxCPM paper (Zhou et al. 2025, arXiv:2509.24650) introduces
  **VoxCPM-0.5B** trained on 1.8M hours of **CN/EN-only** bilingual data. There is
  no Thai in that corpus.
- The JaiTTS authors say they did *continual* training on a Thai-centric corpus.
  Continual training on a base with **zero Thai exposure** would essentially be
  training a new language head from scratch — feasible, but it makes more sense to
  start from a multilingual base.
- The OpenBMB repo's "VoxCPM 2" (2B params, 48 kHz, **30 languages including Thai**,
  ~2M hrs) is the only public VoxCPM variant that already speaks Thai weakly. That
  matches JaiTTS's positioning ("VoxCPM Thai is unverified, we're going to fix it").
- VoxCPM 2's native 48 kHz output also matches the kind of audio quality JaiTTS
  showcases.
- Paper's inference settings (`cfg_value=2.5`, `inference_timesteps=10`, CFG drop 0.1)
  are consistent with VoxCPM 2 defaults.

**Best assumption:** continual SFT of **VoxCPM-2 (2B, 48 kHz)**. If you reproduce, plan
hardware around that (≥40 GB VRAM for full SFT, ≥20 GB for LoRA).

If a smaller / quicker run is preferred, **VoxCPM 1.5 (0.6B, 44.1 kHz)** is the
sensible mid-point: ~12 GB VRAM for LoRA, and still trained with multilingual
coverage.

### 7.3 Code-switching policy — adding an English dataset

**Yes, this is the right move and explicitly recommended by VoxCPM's docs:**

> "New language adaptation: 500+ hours … Full fine-tuning with **10–20% Chinese/English
> mix** to prevent forgetting."

So mixing in a clean EN corpus serves **two purposes at once**:

1. Prevents catastrophic forgetting of the EN capability the base already has.
2. Re-teaches code-switching, since Vaja-Thai's transcripts strip non-Thai characters.

#### Recommended open EN corpora (all CC-friendly):

| Corpus | Hours | Speakers | Sample rate | License | Notes |
|---|---|---|---|---|---|
| **VCTK** | ~44 h | 109 native EN | 48 kHz | CC-BY 4.0 | Studio-quality, multi-accent. Drop-in for VoxCPM 2. Source text is mostly digit-free phonetic-balance prose. |
| **LibriTTS** (or **LibriTTS-R**) | ~585 h | ~2,400 | 24 kHz | CC-BY 4.0 | The de-facto open TTS English corpus. Cleaner than LibriSpeech. **First choice.** Exposes both `text_original` (raw — keeps digits "1923" + abbreviations "Mr.") and `text_normalized` (spelled-out). **Use `text_original` to teach the model raw numerals**, which is one of JaiTTS-v1.0's two no-normalization features. |
| **Common Voice (en)** | thousands | 80k+ | 48 kHz | CC0 | Crowd-sourced, noisy — use a CER/MOS filter first. Good for prosodic diversity. |
| **GigaSpeech (en)** | 10k h | many | 16 kHz | Apache-2.0 (S/M/L/XL splits) | Big, varied, news/podcast — overkill for adapter mixing. |
| **HiFi-TTS** | ~292 h | 10 | 44.1 kHz | CC-BY 4.0 | Studio LibriVox readers, very clean. |

#### Proposed mix (with Vaja-Thai as the Thai backbone)

| Slice | Hours | Why |
|---|---|---|
| Vaja-Thai (all tiers ≤ 3) | ~400–550 h | Primary target language. |
| LibriTTS-R (subset) | ~50–100 h | Maintain EN, give the model clean read-speech reference points. |
| VCTK | ~40 h | Studio quality + accent diversity, helps high-frequency detail at 48 kHz. |
| **Optional code-switch synthetic** | ~10–20 h | Stitch Thai + EN words into mixed sentences from Vaja-Thai + LibriTTS to teach inline switching. |

That's roughly an **80% Thai / 20% EN** split — within VoxCPM's 10–20% guidance.
Resample everything to the variant's native rate (48 kHz for VoxCPM 2, 44.1 for
v1.5) before manifest creation; VoxCPM auto-resamples but pre-normalizing avoids
silent surprises.

> **One thing to redo**: Vaja-Thai's transcript pipeline already converted Arabic
> numerals to Thai number words and stripped non-Thai chars. If you want JaiTTS-style
> "no normalization" behavior, you need access to **raw transcripts** for at least a
> subset. Either re-derive them from the upstream sources (`commonvoice` raw text is
> recoverable; `porjai_central` may not be), or skip this and accept that your model
> will need an external normalizer at inference.

### 7.4 License — personal / free-publish use

You said **personal use, may publish for free**. That widens what you can use:

- Vaja-Thai **`all` config (full 554 h) is fine for non-commercial release** — the
  Non-Commercial slices (`tsync2` 3.7 h, `gigaspeech2` 8.1 h) are now usable.
- If you publish weights, the resulting checkpoint must be released **non-commercial**
  (most restrictive ingredient wins → CC-BY-NC-SA from `tsync2` propagates).
- Add a `LICENSE.md` to your release naming the dataset slices and inheriting their
  most restrictive term (NC-SA), plus VoxCPM's Apache-2.0 for the architecture/code.
- Easy alternative if you want a more permissive checkpoint: drop `tsync2` +
  `gigaspeech2` (you only lose ~12 h out of 554) and you can release under CC-BY-SA-4.0
  (from `porjai_central`). Functionally indistinguishable.

EN add-ons (LibriTTS-R, VCTK, HiFi-TTS) are all CC-BY-4.0 and add no new restriction.

---

## 8. RTX 3090 24 GB plan — LoRA on VoxCPM 2

Confirmed target: single **RTX 3090 24 GB**, LoRA on **VoxCPM 2 (2B, 48 kHz)**, building
on the conventions from the user's prior repo
[`dubbing-ai/indextts2-thai`](https://github.com/dubbing-ai/indextts2-thai).

### 8.1 Will LoRA on VoxCPM 2 actually fit in 24 GB?

VoxCPM's docs cite **~20 GB for VoxCPM 2 LoRA** at `batch_size=16, max_batch_tokens=8192`.
That's already over budget by ~4 GB once activations + temporary buffers are counted on a
24 GB card. The fix is to back off batch settings — same total compute, smaller live
working set:

| Knob | Default | 3090-safe | Effect |
|---|---|---|---|
| `max_batch_tokens` | 8192 | **4096** | Single biggest VRAM win. |
| `batch_size` | 16 | **4** (with `accum=4`) | Effective batch still 16. |
| Optimizer | AdamW | **AdamW8bit** (`bitsandbytes`) | ~30% optimizer-state saving. |
| Mixed precision | BF16 | **BF16** | 3090 supports BF16; keep on. |
| Gradient checkpointing | usually off | **on** | Trades ~20% step time for ~30% activation VRAM. |
| LoRA rank | 32–64 | **r=64, α=128** | "Style/language" tier per VoxCPM docs. |

Expected steady-state VRAM with the above: **~17–20 GB**, leaving headroom for the
DataLoader and occasional spikes.

### 8.2 Recipe

```yaml
# Conceptual — drop into conf/voxcpm_v2/voxcpm_finetune_lora.yaml
base_model: openbmb/VoxCPM2-2B
sample_rate: 48000

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  # VoxCPM applies LoRA across TSLM + RALM transformer layers. Keep LocDiT
  # and AudioVAE frozen for adapter runs.

optim:
  optimizer: adamw_8bit
  lr: 1.0e-4
  weight_decay: 0.0
  warmup_steps: 100
  scheduler: cosine

train:
  precision: bf16
  gradient_checkpointing: true
  batch_size: 4
  gradient_accumulation_steps: 4   # effective batch 16
  max_batch_tokens: 4096
  num_epochs: 3
  save_every_steps: 1000
  eval_every_steps: 1000
  cfg_drop_prob: 0.1               # match the JaiTTS / VoxCPM default
```

### 8.3 Data plan

Mirror your IndexTTS2-Thai pipeline conventions (CSV manifest + tier oversampling +
PyThaiNLP normalization) but emit **VoxCPM's JSONL format**:

```jsonl
{"audio": "wavs/000001.wav", "text": "สวัสดีครับ", "ref_audio": "refs/spk_42.wav", "duration": 4.31, "dataset_id": "vaja_porjai"}
```

| Slice | Hours | Tier weight | Notes |
|---|---|---|---|
| Vaja-Thai tier 1 | ~70 h | **×2** | Studio / best — same boost you used in IndexTTS2-Thai. |
| Vaja-Thai tier 2 | ~50 h | ×1 | Clean. |
| Vaja-Thai tier 3 | ~80 h | ×1 | Bulk. |
| Vaja-Thai tier 4 | — | skip | Too noisy. |
| **LibriTTS-R train-clean-100** with `text_original` | ~55 h | ×1 | **Use `text_original`, not `text_normalized`** — keeps Arabic numerals, abbreviations, raw punctuation. Teaches digit reading + raw-text robustness. Resample 24 kHz → 48 kHz. |
| **VCTK** (optional) | ~44 h | ×1 | Native 48 kHz, accent diversity. Source text is news/phonetic-balance prose — minimal digit content, useful for prosody not numerals. |

That's ~200 h Thai (post-oversample) + ~50–100 h EN — well above VoxCPM's "500-clip"
adapter floor while staying within reasonable wall-clock for a 3090.

> **Why `text_original` matters here:** LibriTTS exposes both `text_original` (raw,
> with digits like "1923", "Mr.", "etc.") and `text_normalized` (spelled out: "nineteen
> twenty three"). The HF dataset `mythicinfinity/libritts` provides both columns.
> Using `text_original` lets the model see Arabic numerals during training, which
> directly addresses one of the two things JaiTTS-v1.0 advertises ("processes numerals
> without explicit text normalization"). VCTK doesn't help here — its source text is
> phonetic-balance newspaper prose with little numeric content. **Don't use both
> `text_original` and `text_normalized` on LibriTTS — pick one (original).**

> **Caveat on Vaja-Thai:** Vaja-Thai's transcripts are **already pre-normalized** by
> the dataset authors — Arabic digits converted to Thai number words, non-Thai
> characters stripped. So your model will see normalized Thai but raw English. This is
> a one-sided code-switch / numeral capability:
> - "Born in 1923" → ✅ (LibriTTS taught raw digits in EN context)
> - "เกิดเมื่อปี 1923" → ⚠️ unverified (Thai context never had digits in training)
>
> Two options if Thai-side digits matter:
> 1. **Accept it**, use a thin numeric-handler at inference for Thai digits only.
> 2. **Re-source raw transcripts** from CommonVoice (recoverable from upstream Mozilla
>    CV) and rebuild the manifest. `porjai_central` raw text is likely not recoverable.
> For personal use, option (1) is the pragmatic call.

**`ref_audio` policy** (from VoxCPM docs): include `ref_audio` on **30–50%** of
samples — for Vaja-Thai use a same-speaker (or proxy-id) clip ≠ the target audio. Your
existing per-speaker grouping from IndexTTS2-Thai's prepare script can be reused for
this.

### 8.4 Reusing your IndexTTS2-Thai pipeline

> **Note on "tokenizer-free":** VoxCPM is tokenizer-free **only on the audio side**
> (no discrete neural codec, latents stay continuous). The text path still uses
> **MiniCPM-4 BPE**. So we still do *encoding hygiene* on text — just not the heavy
> normalization (digit-to-word, segmentation, phonetic respelling) that IndexTTS2-Thai
> needed. JaiTTS's headline trick is exactly this: trained on raw text, no
> normalization pipeline.

Most of `train/prepare_vaja_thai.py` and `src/thai_normalizer.py` ports over with
minimal change:

| Module | Reuse as-is? | Adjust |
|---|---|---|
| `src/thai_normalizer.py` — Unicode hygiene only (invisible-char strip, NBSP/tab fix, repeated-char collapse, PyThaiNLP `thai_normalize` for tone-mark/zero-width fixes) | **Keep** | These are encoding cleanup, not pronunciation rewrites — JaiTTS would do the equivalent. |
| `src/thai_normalizer.py` — number-to-Thai-word, PyThaiNLP `word_tokenize` segmentation, `PHONETIC_DICT` Sanskrit/Pali respelling | **Drop** | These were IndexTTS2 compensations for a weak codec + small Thai BPE vocab. VoxCPM 2 has continuous latents and MiniCPM-4 BPE; respelling teaches the model to pronounce loanwords *only when pre-respelled* and breaks at inference on raw text. JaiTTS deliberately skips all of this. |
| `train/prepare_vaja_thai.py` (HF streaming, tier filter, oversampling, val split) | **Mostly** | Change `TARGET_SR = 22050` → `48000`. Emit JSONL instead of CSV. Add `ref_audio` field via per-speaker grouping. |
| `train/dataset.py` | Replace | VoxCPM ships its own dataset class; point it at the JSONL manifest. |
| `train/train_lora.py` | Replace | Use VoxCPM's `lora_ft_webui.py` or its YAML-driven trainer instead of IndexTTS2's GPT-style trainer. |
| `train/retrain_bpe.py` | **Skip entirely** | VoxCPM is **tokenizer-free** for audio and uses MiniCPM-4's BPE for text — no codec retraining or vocab expansion needed. This is one of VoxCPM's structural advantages over IndexTTS2 for Thai. |
| `src/inference_thai.py`, `src/eval_multilang.py`, `src/upsampler.py` | Adapt | Swap inference backend to `voxcpm.VoxCPM`; AP-BWE upsampler is unnecessary since VoxCPM 2 outputs native 48 kHz. |
| `train/publish_to_hf.py` | Reuse | Same flow, push the LoRA adapter weights + config + sample audio. |

### 8.5 Wall-clock estimate

Your IndexTTS2-Thai run was **~22 h/epoch on the 3090 over ~239k samples**, ~9 days for
10 epochs at BS=1 / accum=8 / r=32. For VoxCPM 2 LoRA, the model is ~2.3× larger but:
- BS=4 instead of 1 → fewer steps per epoch.
- Only TSLM+RALM transformer layers get LoRA; LocDiT and AudioVAE are frozen.
- VoxCPM converges in **1–3 epochs** for adapter runs (per the docs), not 10.

Rough estimate for ~200 h Thai + ~50 h EN, BS=4, accum=4 on a 3090:

| Run | Steps | Wall-clock |
|---|---|---|
| Smoke test (tier-1 only, ~70 h) | ~10–15k | **8–12 h** |
| Full LoRA, 1 epoch | ~30–40k | **24–36 h** |
| Full LoRA, 3 epochs | ~90–120k | **3–5 days** |

These are best estimates — measure throughput in the first 500 steps and recalibrate.

### 8.6 Text-side training augmentation (recover digit reading on the cheap)

The cleanest fix for the §8.3 caveat ("Vaja-Thai pre-normalized digits → Thai words")
is **text-only augmentation at DataLoader time**. The audio target is fixed by the
recording; only the input text is randomized. This teaches the model invariance:
several text spellings → one acoustic realization, mirroring JaiTTS-v1.0's
no-normalization behavior without re-sourcing transcripts.

#### 8.6.1 Augmentation menu

| Aug | Direction | Probability | Where applied |
|---|---|---|---|
| **Thai number-word → Arabic digit** (`"ยี่สิบสาม"` → `"23"`) | word → digit (safe) | **0.4** | Vaja-Thai (all tiers) |
| **Mixed Thai+digit forms** (`"หนึ่งล้าน 200"`) | partial swap | 0.1 | Vaja-Thai |
| **EN raw digit ↔ spelled-out** (sample between LibriTTS `text_original` and `text_normalized`) | bidirectional (both forms validated by LibriTTS) | 0.3 toward `text_normalized` | LibriTTS-R |
| **Whitespace/punctuation jitter** — random extra spaces, optional trailing period removal | — | 0.15 | Both |
| **Case jitter** on EN words | — | 0.1 | LibriTTS-R |
| ~~Audio SpecAugment / noise / pitch-shift~~ | — | **skip** | — |

> **Why no audio augmentation**: VoxCPM's flow-matching LocDiT regresses the *clean
> continuous latent*. SpecAugment / noise / pitch-shift perturb the regression target
> and degrade the model. This is the opposite of ASR practice — keep the audio path
> clean.

#### 8.6.2 When to run `swap_thai_numwords_to_digits` — and when NOT to

The substitution is safe **only when the audio actually says the matched span as a
cardinal number**. Several real-world cases break this assumption:

| Spoken form | Don't substitute when… | Why |
|---|---|---|
| `"สอง สาม สี่ ห้า"` (phone-number recital) | digits are read individually | Pattern would map to `"2345"` but the audio reads each digit separately. The acoustic durations are wrong. |
| `"ปีสองพันห้าร้อยหกสิบเจ็ด"` (Buddhist year 2567) | year context | `"2567"` is fine *if* read as cardinal, but Thai often reads years digit-by-digit with prefix. Detection is hard. |
| `"จุด"` (decimal point) appears mid-number | mixed cardinal + non-cardinal | `"สามจุดหนึ่งสี่"` → `"3.14"` is fine, but boundary detection trips on `"สาม"` alone. |
| `"ที่หนึ่ง" / "ครั้งที่สอง"` (ordinals) | ordinal markers `ที่` precede the number | Audio says ordinal, digit form drops the ordinal cue. |
| `"ครึ่ง"` / `"สี่ส่วนสาม"` (fractions) | fractional words | Not cleanly representable as digits. |
| `"โหล" / "กุรุส"` (dozen / gross — quantifiers) | non-decimal counters | Audio doesn't say "12" or "144." |
| Number embedded in a name/idiom (`"เจ็ดสิ่งมหัศจรรย์"`) | proper noun / idiom | The number is part of a fixed phrase. |

**Decision logic** — only substitute when **all** are true:

1. The matched span is a contiguous cardinal-number Thai-word sequence (use PyThaiNLP
   to detect, then round-trip: `num_to_thaiword(thaiword_to_num(span)) == span`).
2. The span is **≥ 2** Thai number words long (skips lone `"หนึ่ง"`/`"สอง"` which often
   aren't quantitative — `"หนึ่งในนั้น"` ≠ "1").
3. **No ordinal/fraction/quantifier neighbor**: the preceding word is not in
   `{"ที่", "ลำดับ", "อันดับ", "ครั้งที่"}` and the span is not flanked by
   `{"ครึ่ง", "ส่วน", "จุด", "โหล", "กุรุส"}`.
4. **Digit-by-digit recital filter**: skip spans matching `(หนึ่ง|สอง|...|เก้า|ศูนย์)
   (\s*(หนึ่ง|...|ศูนย์)){3,}` — sequences of single digits with no ten/hundred/thousand
   modifier.
5. **Year heuristic** (optional): if the span begins with `"สองพัน"` or `"สองพันห้า"`
   and the surrounding context contains `"ปี"` or `"พ.ศ."`, **skip** — too easy to get
   wrong.

If any check fails → leave the span as Thai words.

#### 8.6.3 Reference implementation sketch

```python
# train/augment.py — runs inside the DataLoader (fresh per epoch)

import random
import re
from pythainlp.util import num_to_thaiword, thaiword_to_num

THAI_DIGIT_WORDS = {"ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่",
                    "ห้า", "หก", "เจ็ด", "แปด", "เก้า"}
ORDINAL_PRECEDERS = {"ที่", "ลำดับ", "อันดับ", "ครั้งที่"}
NON_CARDINAL_NEIGHBORS = {"ครึ่ง", "ส่วน", "จุด", "โหล", "กุรุส"}
YEAR_CONTEXT = {"ปี", "พ.ศ.", "ค.ศ."}

NUMWORD_SPAN = re.compile(
    r"(?:หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ|ยี่สิบ|"
    r"ร้อย|พัน|หมื่น|แสน|ล้าน|เอ็ด|ศูนย์)+"
)

def is_safe_to_digitize(span: str, left_word: str, right_word: str,
                         context_window: str) -> bool:
    # (1) round-trip check
    try:
        n = thaiword_to_num(span)
        if n is None or num_to_thaiword(n) != span:
            return False
    except Exception:
        return False
    # (2) min length
    if sum(1 for w in THAI_DIGIT_WORDS if w in span) < 2 and \
       not any(unit in span for unit in ("สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน")):
        return False
    # (3) ordinal / fraction / quantifier neighbor
    if left_word in ORDINAL_PRECEDERS:
        return False
    if left_word in NON_CARDINAL_NEIGHBORS or right_word in NON_CARDINAL_NEIGHBORS:
        return False
    # (4) digit-by-digit recital — span is all single-digit words, no unit modifier
    tokens = span.split()  # if you've inserted spaces; else use a tokenizer
    if all(t in THAI_DIGIT_WORDS for t in tokens) and len(tokens) >= 4:
        return False
    # (5) year heuristic
    if any(yc in context_window for yc in YEAR_CONTEXT) and span.startswith("สองพัน"):
        return False
    return True

def maybe_digitize_thai(text: str, p_full: float = 0.4, p_partial: float = 0.1) -> str:
    """Apply Thai number-word → Arabic digit augmentation.

    p_full:    full substitution of every safe span
    p_partial: partial — substitute exactly one randomly chosen safe span
    """
    spans = [(m.start(), m.end(), m.group()) for m in NUMWORD_SPAN.finditer(text)]
    if not spans:
        return text

    safe = []
    for start, end, span in spans:
        left = text[:start].strip().split()[-1] if text[:start].strip() else ""
        right = text[end:].strip().split()[0] if text[end:].strip() else ""
        ctx = text[max(0, start - 30): min(len(text), end + 30)]
        if is_safe_to_digitize(span, left, right, ctx):
            safe.append((start, end, span))
    if not safe:
        return text

    r = random.random()
    if r < p_full:
        chosen = safe                                  # all safe spans
    elif r < p_full + p_partial:
        chosen = [random.choice(safe)]                 # exactly one
    else:
        return text                                    # leave alone

    out, last = [], 0
    for start, end, span in sorted(chosen):
        out.append(text[last:start])
        out.append(str(thaiword_to_num(span)))
        last = end
    out.append(text[last:])
    return "".join(out)
```

**Apply at DataLoader-time, inside `__getitem__`.** The whole point is invariance —
the model has to see the *same audio* paired with *different valid text spellings*
across epochs. Manifest-time substitution freezes one spelling per row and wastes most
of the augmentation's value.

Concretely:

- Hook `maybe_digitize_thai(...)` into the VoxCPM dataset class's text-prep callback so
  it runs once per `__getitem__`. Seed the RNG per-worker for reproducibility.
- Add an optional `no_digit_aug: true` field to the JSONL for the rare row whose audio
  you've confirmed is digit-by-digit recital — `__getitem__` checks the flag and skips.
  Default is false.
- Build the **validation manifest with substitution disabled** (either set the flag on
  every val row or short-circuit augmentation when `split == "val"`) so eval CER is
  stable across epochs.

#### 8.6.4 Validation

Hold out a small **digit-eval set** (50–100 prompts: `"เกิดเมื่อปี 1923"`,
`"เบอร์โทร 081-234-5678"`, `"ราคา 1,250 บาท"`) and CER-eval after each epoch.
Vanilla VoxCPM 2 will fail most of these; a well-trained adapter with this
augmentation should handle the cardinal cases (price, year-as-cardinal) cleanly,
while phone-number digit-by-digit will still degrade — that's fine and expected.

### 8.7 Execution plan

1. **Set up the base.** Clone VoxCPM, install, and load `openbmb/VoxCPM2-2B`. Synthesize a few Thai sentences (and digit-eval prompts from §8.6.4) with the unmodified base — this is your "before" baseline.
2. **Port the data prep.** Adapt `train/prepare_vaja_thai.py` to emit JSONL at 48 kHz with the `audio` / `text` / `ref_audio` / `duration` / `dataset_id` / `no_digit_aug` schema. Keep tier oversampling (`{1: 2, 2: 1, 3: 1, 4: 0}`); strip the IndexTTS2-only normalization layers (number-to-word, word segmentation, phonetic dict).
3. **Generate `ref_audio` pairings.** Group Vaja-Thai by speaker / proxy ID, attach a different same-speaker clip as `ref_audio` on 30–50% of rows. Leave the rest without `ref_audio` so zero-shot capability is preserved.
4. **Add the EN slice.** Include LibriTTS-R `train-clean-100` using `text_original` (raw digits + abbreviations); resample to 48 kHz at manifest write time. Optionally add VCTK for accent/prosody diversity.
5. **Wire up DataLoader-time augmentation** (§8.6) — Thai number-word → digit (p=0.4 full / 0.1 partial), EN `text_original` ↔ `text_normalized` sampling, whitespace/punctuation jitter. Validate with the digit-eval set.
6. **Smoke run.** LoRA r=64 on Vaja-Thai tier-1 only (~70 h), 1 epoch, ~10–15k steps. Confirm VRAM stays <22 GB on the 3090, generation actually terminates (the trailing-silence gotcha), and outputs are intelligible.
7. **Full LoRA.** Add tier-2/3 + LibriTTS-R, run 1 epoch end-to-end. Save every 1k steps for ablation.
8. **Evaluate.** Use JaiTTS's `cal_wer.sh` (with Typhoon-Whisper-Large-v3 for Thai CER) and `cal_sim.sh` (with the WavLM speaker-verification checkpoint) on the short/long benchmarks. Add the digit-eval set as a third axis. Compare against the §1 baseline from step 1.
9. **Iterate or extend.** If quality looks good after 1 epoch, extend to 3. If not, the bottleneck is almost always data quality or `ref_audio` coverage rather than model capacity — refine the manifest before scaling steps.
10. **Publish.** Reuse `train/publish_to_hf.py` — push the LoRA adapter weights, the YAML config, sample audio, and a model card noting the dataset license inheritance (CC-BY-NC-SA, non-commercial).

---

## 9. References

- **JaiTTS paper (arXiv:2604.27607v2, 2026-05-07):** Karnjanaekarin et al.,
  *JaiTTS: A Thai Voice Cloning Model.* CC-BY-4.0.
- **JaiTTS code:** <https://github.com/JTS-AI-Team/JaiTTS> (Apache-2.0, eval only).
- **VoxCPM paper:** Zhou et al. 2025, arXiv:2509.24650 — *VoxCPM: Tokenizer-free TTS
  for context-aware speech generation and true-to-life voice cloning.*
- **VoxCPM repo + fine-tuning guide:** <https://github.com/OpenBMB/VoxCPM> ·
  <https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html>
- **Vaja-Thai dataset:** <https://huggingface.co/datasets/dubbing-ai/vaja-thai>
- **Underlying LM backbone:** MiniCPM4 (arXiv:2506.07900).
