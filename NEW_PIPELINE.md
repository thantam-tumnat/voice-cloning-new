# New generation pipeline — Thonburian + SeedVC (handoff)

Replacing the VoxCPM2 voice-generation backend with **Thonburian F5 (emotion) →
SeedVC (re-timbre)**, driven by emotion reference clips from **airesearch/thai-ser**.
The Tone Studio (`:8011`) front-end and its features stay the same — only the way
audio is generated changes.

## Pipeline

```
[angry] + text
   │  pick donor clip for the emotion (ref/emotions/<emo>_1.wav + .txt)
   ▼
Thonburian F5  — clone the donor's emotion onto the target text (Thai speech)
   ▼
SeedVC (f0-condition) — swap timbre to the user's chosen voice, keep the emotion
   ▼
assemble chunks → WAV
```

The target voice = whatever speaker the studio user selects/uploads (SeedVC target).
The emotion tag = which thai-ser donor clip Thonburian clones.

## Status — Phase 0 spike is PROVEN ✅

Ran end-to-end (`voice-cloning/tools/spike_thonburian_seedvc.py`). Findings, all
verified with `librosa` F0/energy measurement:

- **Thonburian transfers emotion** from a thai-ser donor to unseen text. angry came
  out `+99 Hz` median-F0 vs neutral; sad/frustrated shifted duration and F0 range.
- **fp16 → NaN on GTX 16-series** (compute 7.5, no tensor cores): flowtts loads F5
  in fp16 on any compute≥7.0 card and emits an all-`-1.0` waveform. The spike forces
  fp32 as a guard. **On Ampere (RTX 4070, sm_89) fp16 is fine** — drop the fp32
  guard there for speed/VRAM.
- **SeedVC timbre-only flattens pitch emotion** (angry `+99 → -40 Hz`). Running with
  **`--f0-condition True`** restores it (`+97 Hz`); add **`--auto-f0-adjust True`** to
  keep the emotion contour *and* land in the target's register. This is the config to
  use.

A/B comparison page (audio for all stages) was published as a private Artifact.

## Why we stopped here (4 GB box) → continue on the 4070 12 GB

SeedVC alone holds ~2.9 GB of the 4 GB card; Thonburian F5 fp32 needs ~2 GB, so the
two models cannot be GPU-resident together on the dev box. On the **RTX 4070 (12 GB)**
both fit comfortably — run them as two persistent services (below).

## Environment setup on the new machine

1. **Thonburian / flowtts** (main env, for the studio process):
   ```bash
   pip install git+https://github.com/biodatlab/thonburian-tts.git
   ```
   Its `flowtts` package has no top-level `__init__.py` and does not import after a
   plain pip install — clone the repo and put it on `PYTHONPATH` / `FLOWTTS_SRC`:
   ```bash
   git clone https://github.com/biodatlab/thonburian-tts.git
   export FLOWTTS_SRC=/path/to/thonburian-tts
   ```
   Model auto-downloads: `hf://biodatlab/ThonburianTTS/megaF5/mega_f5_last.safetensors`
   (+ `mega_vocab.txt`, vocos vocoder).

2. **SeedVC** (its own venv — pins torch 2.4, conflicts with the studio env):
   ```bash
   git clone https://github.com/Plachtaa/seed-vc.git
   python -m venv seedvc-venv
   seedvc-venv/bin/pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
       --index-url https://download.pytorch.org/whl/cu121
   seedvc-venv/bin/pip install accelerate scipy==1.13.1 librosa==0.10.2 \
       "huggingface-hub>=0.28.1" munch==4.0.0 einops==0.8.0 descript-audio-codec==1.0.0 \
       pydub==0.25.1 resemblyzer jiwer==3.0.3 transformers==4.46.3 soundfile==0.12.1 \
       modelscope==1.18.1 funasr==1.1.5 numpy==1.26.4 hydra-core==1.3.2 pyyaml \
       python-dotenv fastapi uvicorn "pydantic>=2"
   ```

3. **Donor clips** (already committed under `voice-cloning/ref/emotions/`, one per
   emotion). Regenerate / add more:
   ```bash
   cd voice-cloning
   python tools/build_emotion_refs.py --per-emotion 5      # auto-pick
   python tools/build_emotion_refs.py --audio-ids <id> ... # pick exact clips
   ```

## Architecture on the 4070 (two persistent services)

```
Studio :8011 (main env + flowtts)  ── annotate / chunk / assemble + Thonburian gen
        │  HTTP POST /convert
        ▼
SeedVC worker :8022 (seedvc venv)  ── SeedVCWrapper loaded once
```

Start the SeedVC worker (persistent, loads the model once):
```bash
seedvc-venv/bin/python voice-cloning-with-tones/tools/seedvc_server.py \
    --seedvc-repo /path/to/seed-vc --port 8022
# POST /convert {source, target, output, f0_condition:true, auto_f0_adjust:true}
```

## What's built vs what's next

Built and committed:
- `voice-cloning/tools/build_emotion_refs.py` — thai-ser donor builder (`--audio-ids`).
- `voice-cloning/tools/spike_thonburian_seedvc.py` — the proof, with f0/fp32 flags.
- `voice-cloning-with-tones/tools/seedvc_server.py` — persistent SeedVC HTTP worker.

Next (studio wiring — needs the GPU to test, so left for the 4070):
1. `app/services/thonburian_service.py` — mirror `siangtts_service.synthesize_many`:
   per chunk, map tone→emotion (the 5 thai-ser emotions; **raise on unsupported**),
   strip the leading `(...)` instruction, Thonburian-gen with the donor, POST the
   result to the SeedVC worker with the target voice, assemble via `audio_post`.
2. Wire `app/main.py` `/synthesize` and `/synthesize/upload` to call it (replace
   VoxCPM2 — decided).
3. Config: `FLOWTTS_SRC`, `SEEDVC_URL=http://127.0.0.1:8022`, donor dir, seedvc params.
4. Emotion map: neutral / angry / happy / sad supported (+ explicit `frustrated`);
   everything else errors with the supported list.

Ports are unchanged (:8010 webhook, :8011 studio, :8020/:8021 queue). VoxCPM2 / LoRA
code is being removed, not kept.
