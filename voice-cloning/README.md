# SiangTTS

Thai voice-cloning TTS built by LoRA fine-tuning **VoxCPM 2 (2B)** on the
[`dubbing-ai/vaja-thai`](https://huggingface.co/datasets/dubbing-ai/vaja-thai) corpus,
plus a small **LibriTTS-R** slice with raw `text_original` to retain English digit
reading and code-switching.

This repo is the practical implementation of the plan in [`RESEARCH.md`](RESEARCH.md).
**[`PLAN.md`](PLAN.md) is the live execution roadmap** — next steps and the
decision tree for each evaluation outcome.
The reference architecture / training recipe is adapted from the
[JaiTTS paper](https://arxiv.org/abs/2604.27607), but SiangTTS trains within a single
**RTX 3090 24 GB** budget using LoRA rather than full SFT.

## Status

Fully implemented against **voxcpm 2.0.3**. `train/train_lora.py` mirrors the
mechanics of VoxCPM's official `scripts/train_voxcpm_finetune.py` (packer, bf16
autocast, grad accumulation, LoRA-only checkpoints, resume, save-on-signal) and
adds per-source weighted sampling, DataLoader-time text augmentation with
tokenize-after-augment, and the MonitorBundle (TensorBoard + audio snapshots +
timing JSON). Checkpoints use VoxCPM's loadable LoRA layout
(`lora_weights.safetensors` + `lora_config.json`), so they work directly with
`voxcpm clone --lora-path ...` and `src/inference.py --adapter ...`.

Key facts learned from the installed API (these differ from early RESEARCH.md
assumptions): VoxCPM2's AudioVAE **encodes at 16 kHz** and decodes at 48 kHz, so
manifests are stored at 16 kHz; the LoRA config keys are
`enable_lm/enable_dit/enable_proj/r/alpha/dropout`; the base HF id is
`openbmb/VoxCPM2`. Torch is pinned to cu128 wheels (driver 575.x = CUDA 12.9
cannot run cu130).

**Trained & published** (v1): short Thai CER ~3.8%, long-form 1.6%, voice cloning
SIM 0.882 — see [`RESULTS.md`](RESULTS.md). Model:
[dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA](https://huggingface.co/dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA).
45 unit tests pass; `--dry-run` exercises the pipeline CPU-only.

## Layout

```
SiangTTS/
├── conf/
│   ├── voxcpm_lora.yaml       # LoRA training recipe (§8.2 of RESEARCH.md) — 3090-sized
│   └── voxcpm_sft.yaml        # full-SFT recipe — A100-80G sized (escalation path)
├── src/
│   ├── webhook.py             # FastAPI webhook & async job service (primary production server)
│   ├── serve.py               # synchronous inference API (KhongkhunAPI-compatible)
│   ├── app.py                 # Gradio interactive web UI
│   ├── pipeline.py            # audio merge, upload, and callback pipeline
│   ├── thai_text.py           # Thai text segmentation and prompt preparation
│   ├── thai_normalizer.py     # encoding hygiene only (no number-to-word, no segmentation)
│   ├── augment.py             # DataLoader-time text augmentations
│   ├── inference.py           # thin wrapper around voxcpm.VoxCPM
│   └── eval.py                # CER (Typhoon-Whisper) + SIM (WavLM) + digit-eval
├── train/
│   ├── prepare_vaja_thai.py   # vaja-thai → JSONL @ 16 kHz, ref_audio pairing
│   ├── prepare_libritts.py    # LibriTTS-R text_original → JSONL @ 16 kHz
│   ├── dataset.py             # JSONL → VoxCPM dataset; hooks src/augment.py
│   ├── train_lora.py          # invokes VoxCPM trainer with conf/voxcpm_lora.yaml
│   └── publish_to_hf.py       # push LoRA adapter + config + samples to HF
├── eval/
│   ├── prompts_short.tsv      # Thai short-form (1–15 s)
│   ├── prompts_long.tsv       # Thai long-form (16–30 s)
│   ├── prompts_digits.tsv     # digit-eval (years / prices / phones)
│   └── prompts_listen.tsv     # in-training audio snapshots (TB Audio tab)
├── tests/                     # pytest unit tests for normalizer + augment + dataset + audio prep
├── data/                      # gitignored
├── checkpoints/               # gitignored
├── pyproject.toml             # uv-managed
├── conftest.py                # makes `pytest` resolve src/ + train/ from any CWD
├── COMMANDS.md                # operational commands reference
├── DEPLOY.md                  # deployment guide
├── RESEARCH.md
└── README.md
```

## Setup

```bash
# Install uv (one-time): https://docs.astral.sh/uv/
uv sync
```

## Workflow

```bash
# 1. Sanity-check the base model on Thai
uv run python -m src.inference --base-only --text "สวัสดีครับ"

# 2. Prepare manifests
uv run python train/prepare_vaja_thai.py --output-dir data/vaja-main --tiers 1 2 3 --max-samples 80000
uv run python train/prepare_libritts.py  --output-dir data/libritts --config clean --split train.clean.100

# 3. Train LoRA (the dry-run validates the dataset + monitor pipeline without GPU)
uv run python train/train_lora.py --config conf/voxcpm_lora.yaml --dry-run
uv run python train/train_lora.py --config conf/voxcpm_lora.yaml

# 3b. Full SFT (rented A100-80G — does NOT fit the 3090; same script, no `lora:`)
uv run python train/train_lora.py --config conf/voxcpm_sft.yaml

# 4. Evaluate
uv run python -m src.eval --adapter checkpoints/siangtts-lora-v0/latest --prompts eval/prompts_short.tsv
```

## Serving & demos

The model is loaded **once per host**, by the GPU service. Everything else is a client of it:

```
n8n / LiveAI ──► :8010 webhook       ──┐
                                       ├──► :8020 GPU service   VoxCPM2 + Thai LoRA ×1
browser      ──► :8011 tone studio   ──┘                        (../voice-cloning-with-tones)
```

Before the split each pipeline loaded its own copy, which on a single GPU meant they
could not run at the same time.

### 1. GPU Service (FastAPI — Port 8020)

The only process that touches the GPU. Owns the model, the prompt-cache store, the job
queue and its single worker. Knows nothing about scripts, ffmpeg, uploads or emotion tags.

```bash
$env:PYTHONIOENCODING = "utf-8"
$env:SIANGTTS_ADAPTER = "checkpoints/siangtts-v1"

uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8020
```

Localhost only: no authentication, and a render job names the directory it writes into.
`SIANGTTS_GPU_STUB=1` runs the whole service without a model for testing.

**Key endpoints:**
- `POST /v2/jobs/render` — a list of ready-to-speak chunks + a voice → audio. `?wait=N` blocks for the result; otherwise poll `GET /v2/jobs/{id}`.
- `POST /v2/voices` · `POST /v2/voices/resolve` · `POST /v2/voices/seed` — prompt caches, addressed by opaque handle
- `GET /health` — model, adapter, queue depth per lane, current LoRA scale
- `GET /` — queue dashboard

Two lanes: the studio's requests (`interactive`) jump ahead of webhook scripts (`batch`),
capped so production traffic is never starved.

### 2. Production Webhook Service (FastAPI — Port 8010)

Async voice synthesis jobs, chunking, audio merging, upload, and webhook callbacks
(replaces the n8n flow). Live dashboard at `http://localhost:8010/`. Generation goes to
the GPU service; no model is loaded here, so it restarts in about a second.

```bash
$env:PYTHONIOENCODING = "utf-8"
$env:SIANGTTS_GPU_URL = "http://127.0.0.1:8020"
$env:SIANGTTS_UPLOAD_TOKEN = "<bearer-token>"

uv run uvicorn src.webhook:app --host 0.0.0.0 --port 8010
```

`SIANGTTS_WORK_DIR` must resolve to the same directory in both services — that is how the
chunk WAVs get from one to the other.

**Key Endpoints:**
- `POST /webhook/live-ai-create-new` — Submit async TTS job (returns immediate `{"status":"success"}`)
- `GET /jobs` · `GET /jobs/{job_id}` — Query job history and queue status
- `GET /voices` — List available reference voices and cached prompt caches
- `GET /health` — Service health, engine status, and cache metrics
- `GET /` — Real-time web dashboard

---

### Retired: Synchronous Inference API (`src/serve.py`, Port 8000)

Superseded by the GPU service, which does the same job for both pipelines instead of
loading a third copy of the model. Left in the tree for reference; do not start it.

---

### 3. Live Demo (Gradio UI — Port 7860)

Standalone interactive browser UI to type text and compare Base VoxCPM2 vs. SiangTTS LoRA:

```bash
uv run python -m src.app                   # http://localhost:7860 (--share for public link)
```

---

### 4. Static Demo Page & Model Publishing

```bash
# 0. Generate the curated comparison set (GPU): diverse gender / length / numeric
uv run python -m src.demo prep

# 1. Build static demo page → docs/ (viewable on GitHub Pages)
uv run python -m src.demo html

# 2. Publish adapter to Hugging Face Hub:
uv run python train/publish_to_hf.py \
    --repo-id dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA --public
```

**Surfaces summary:**
- **GPU Service** → `src/gpu_service.py` (FastAPI, Port 8020) — the only process that loads the model
- **Webhook Service** → `src/webhook.py` (FastAPI, Port 8010)
- **Tone Studio** → `../voice-cloning-with-tones` (FastAPI, Port 8011)
- **Live Demo** → `src/app.py` (Gradio, Port 7860)
- **Static Demo** → `src/demo.py` (GitHub Pages)
- ~~Inference API~~ → `src/serve.py` (Port 8000) — retired, replaced by the GPU service

Published model: <https://huggingface.co/dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA>
(CC-BY-SA-4.0). Always launch GPU commands with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Tests

```bash
uv run pytest tests/ -q
```

Covers the Thai text normalizer (Unicode hygiene, NFC, tone-mark preservation), the
augmentation safety gates from §8.6.2 (round-trip / ordinal / fraction / year /
digit-recital filters, Thai-cluster-safe whitespace jitter), the manifest weighting
(`WeightedRandomSampler` ratios), and the audio-prep silence trim + duration filter.

## Monitoring

Three things run alongside training, configured under `monitor:` in
`conf/voxcpm_lora.yaml`:

- **TensorBoard** — scalars (loss, throughput, val metrics) flushed every 100 steps.
  Launch with:

    ```bash
    uv run tensorboard --logdir runs/
    ```

- **In-training audio sampler** — every `every_steps` (default 1000) the model
  synthesizes the prompts in `eval/prompts_listen.tsv` (Thai short / long / digits /
  code-switch + EN sanity checks). Output goes to:
  - the **TensorBoard Audio tab** (listen in browser), and
  - `runs/<run>/audio_snapshots/step_<N>/` (raw WAVs, easy to A/B locally).

- **Timing tracker** — wall-clock, step throughput, GPU name + peak VRAM, and val
  metrics at every checkpoint, written incrementally to:

    ```
    runs/<run>/training_summary.json
    ```

  This file is the source of truth for any publication / model-card timing claims.

See `RESEARCH.md` §8 for the full execution plan.

## License

Code: Apache-2.0. Trained checkpoints inherit the most restrictive license of their
training data. The published **v1** used Common Voice Thai (CC0), porjai_central
(CC-BY-SA-4.0), and LibriTTS-R (CC-BY-4.0) — **no non-commercial sources** — so it
is released **CC-BY-SA-4.0** (commercial use OK, share-alike). Adding the `tsync2`
or `gigaspeech2` Vaja-Thai slices would make a future checkpoint non-commercial
(CC-BY-NC-SA).
