# Thai TTS Tone Annotation & Voice Cloning Studio (SiangTTS / VoxCPM2)

FastAPI service for analyzing emotional tones and intensities in Thai text and synthesizing expressive speech with **Zero-Shot & Cached Voice Cloning** via **SiangTTS (VoxCPM2 + Thai LoRA)**, ElevenLabs audio tags, or Gemini prompt instructions.

---

## Architecture Overview

```
Thai Raw Text / Script (e.g. '[calm] หายใจเข้า...') ───► 1. SEGMENT & ANNOTATE (PyThaiNLP + LLM)
                                                                 │
                                                                 ▼
Uploaded Reference Audio / Registered Speaker ────────► 2. VOICE CLONING (SiangTTS Speaker Cache)
                                                                 │
                                                                 ▼
                                                        3. EMOTION RENDERER
                                                           (Tone -> VoxCPM Instruction)
                                                                 │
                                                                 ▼
                                                        4. SYNTHESIS ENGINE
                                                           shared GPU service :8020 -- one job,
                                                           all chunks, one voice
                                                           (VoxCPM2 + dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA)
                                                                 │
                                                                 ▼
                                                        5. AUDIO ASSEMBLER
                                                           (trim / level / pace / pause)
                                                                 │
                                                                 ▼
                                                        6. 48kHz WAV Audio Output
```

---

## Tone Enum & Engine Mapping

| Tone | VoxCPM2 / SiangTTS Instruction | ElevenLabs Tag | Gemini Prompt (Thai) |
|---|---|---|---|
| `neutral` | *(no instruction)* | *(no tag)* | น้ำเสียงปกติ เป็นกลาง |
| `sad` | `(Sad voice, quiet and downcast)` | `[sad]` | เศร้า สะเทือนใจ |
| `happy` | `(Happy and cheerful voice, smiling while speaking)` | `[happily]` | ร่าเริง ยิ้มขณะพูด |
| `angry` | `(Angry, firm and aggressive tone)` | `[angry]` | โกรธ เสียงแข็ง |
| `excited` | `(Excited and energetic tone)` | `[excited]` | ตื่นเต้น กระตือรือร้น |
| `calm` | `(Calm and soothing voice, speaking softly)` | `[calm]` | สงบ นุ่มนวล พูดช้า |
| `nervous` | `(Nervous and trembling voice, hesitant)` | `[nervous]` | ประหม่า ลังเล |
| `sarcastic` | `(Sarcastic and mocking tone)` | `[sarcastic]` | ประชด แดกดัน |

Intensity levels for VoxCPM (1: Mild / 2: Standard / 3: Strong).

Reword these only with `tools/instruction_leak_audit.py` in hand. On some phrasings
VoxCPM2 stops treating the parenthetical as direction and reads it aloud in English
ahead of the line — the tag still "works" as far as every check in this repo is
concerned, it just also speaks the stage direction. The audit renders each wording
and transcribes it, which is the only way that shows up.

---

## Custom Pronunciation Dictionary

`pronunciation.json` respells words for the synthesizer only — the script shown in
the studio and returned by the API keeps the original spelling.

```json
{
  "ไฟล์": "ฟาย"
}
```

The usual reason is a Thai loanword written with a silent consonant (การันต์):
`ไฟล์` should read as ฟาย `/faː j/`, and reading the base word ไฟ `/faj/` instead
says "fire".

**Matching is on word boundaries, never substrings.** This matters: `โปรไฟล์` is
genuinely `/proː faj/` with the *short* vowel, so a find-and-replace of `ไฟล์`
would mispronounce it while fixing `ไฟล์`. PyThaiNLP tokenizes the two apart, and
the longest key wins, so `ไฟล์เอกสาร` can override `ไฟล์`. If the text cannot be
tokenized losslessly it is passed through untouched.

Edits are picked up on the next synthesis — the file is re-read when its mtime
changes, no restart needed. Also editable over the API:

```bash
curl -X PUT localhost:8011/pronunciation -H "Content-Type: application/json" -d "{\"entries\":{\"ไฟล์\":\"ฟาย\"}}"
```

`GET /pronunciation` returns the current entries and the file path.

> **Measured caveat for `ไฟล์` specifically.** Whisper transcription of 4 takes each
> found the model already reads `ไฟล์` correctly 3 of 4 times, and respelling it to
> `ฟาย` produced the same 3-of-4 result — the two are homophones, so the respelling
> cannot change the phonetics. The residual error looks like generation variance, not
> a text problem, and is more likely to respond to `cfg_value`. The dictionary earns
> its keep on words the model genuinely does not know: names, brands, acronyms.

---

## Audio Assembly & Prosody Targets

Each tone run is synthesized as its own chunk, because VoxCPM2 only honours a style
parenthetical at position 0. `app/services/audio_post.py` then joins those chunks
into one take: it trims each chunk's own padding, levels every chunk to the take's
median and applies a per-emotion loudness offset, nudges each chunk toward a
per-emotion speaking pace, fades the joins, and places a pause sized by the
boundary kind.

The targets come from measuring an ElevenLabs take -- one Thai sentence read four
times as `[sad] [happy] [scared] [tired]`:

| tone | duration vs mean | energy vs mean |
|---|---|---|
| `sad` | 1.035x | -2.0 dB |
| `happy` | 0.927x | -0.5 dB |
| `scared` | 0.966x | +3.0 dB |
| `tired` | 1.073x | -1.4 dB |

The headline finding: across all four, its **median F0 moves only 1.75 semitones**.
ElevenLabs does not change the speaker's pitch to change the emotion -- it changes
duration, loudness, pitch *range* and pausing. Emotion wording tuned to maximise
median-F0 separation is therefore optimising the one axis the target barely uses,
and pitch-shifting a cloned voice is what makes it stop sounding like one person.

Pause lengths depend on how the script was written: `GAP_SAME_TONE_S` (0.20s) within
a tone run, `GAP_EMOTION_S` (0.45s) at an inline tone change, and `GAP_PARAGRAPH_S`
(1.20s) when the source put a line break before the tag.

### Measuring a take

```bash
python tools/prosody_eval.py take.wav --labels sad,happy,scared,tired
```

Reports duration ratio, energy, F0 spread and F0 offset per utterance, each diffed
against the ElevenLabs target, plus a voice-identity check (median-F0 spread across
utterances, budget 1.8 semitones).

```bash
python tools/ab_gen.py --speaker determination
```

Renders the reference script once and assembles the same chunks both the old way
(flat 60 ms butt-join) and the current way, then scores both -- so a change is
measured against a fixed target rather than judged by ear.

---

## Setup & Running

### 1. Install Dependencies
```bash
py -m pip install -r requirements.txt
```
*(For GPU inference: install `voxcpm`, `soundfile`, `torch` with CUDA)*

### 2. Configure Environment
Copy `.env.example` to `.env` and configure:
```env
GEMINI_API_KEY=your_gemini_api_key
LLM_PROVIDER=gemini
# The shared GPU service that holds VoxCPM2 for every pipeline on this host.
VOXCPM_SERVICE_URL=http://127.0.0.1:8020
VOXCPM_REMOTE_REQUIRED=true
# Used only when VOXCPM_REMOTE_REQUIRED=false, i.e. loading the model in-process.
SIANGTTS_BASE_MODEL=openbmb/VoxCPM2
SIANGTTS_ADAPTER=dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA
# How hard the Thai LoRA is applied, per side of the model (2.0 = as shipped).
# The DiT side defaults to 0: at full strength it flattens the style tags, and
# "[angry]" came out quieter and lower-pitched than a neutral read. The LM side
# stays at full strength and is what carries the Thai. Measured with
# tools/expr_sweep.py; set SIANGTTS_LORA_DIT_SCALE=2.0 to restore the adapter's
# shipped behaviour if you prefer its pronunciation.
SIANGTTS_LORA_LM_SCALE=2.0
SIANGTTS_LORA_DIT_SCALE=0.0
```

### 3. Run FastAPI Server & Web Studio

The studio does not load VoxCPM2 itself. Start the shared GPU service first — the
one process on the host holding the model, shared with the webhook queue on :8010:

```bash
cd ../voice-cloning && uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8020
```

Then the studio:

```bash
py -m uvicorn app.main:app --reload --port 8011
```
Then open your browser and navigate to:
👉 **`http://localhost:8011/`** to access the interactive **Thai TTS & Voice Cloning Studio**.

Startup logs which engine it connected to. Without the GPU service, `/synthesize`
returns 503 rather than quietly loading a second copy of the model — on a single-GPU
host that competes with the shared one for VRAM. `VOXCPM_REMOTE_REQUIRED=false`
allows the in-process fallback when nothing else is using the GPU.

The shared queue — what the GPU is actually working on, across both pipelines — is at
**`http://localhost:8020/`**.

### 4. Run Test Suite
```bash
py -m pytest -v
```

---

## Web Studio Features

- **Interactive Script & Emotion Editor**: Real-time parsing of bracket tags like `[calm] ...` into audio instructions.
- **Editable script stays in short tag form**: the main textarea holds `[sad] ... [happy] ...` (the `script` field of `/speak`), not the expanded VoxCPM2 instruction. That form is what the box sends back to `/synthesize`, and it is the only one that round-trips — `text` is a single-shot rendering carrying just the *first* instruction, so re-submitting it collapsed a multi-emotion script into one tone.
- **Voice Cloning & Speaker Manager**:
  - Select from pre-cached registered voices in `ref/` & `voice_cache/`.
  - Drag-and-drop / Upload any reference audio clip (`.wav`, `.mp3`) for zero-shot voice cloning.
- **Engine Selector**: Switch seamlessly between **SiangTTS (VoxCPM2 Thai LoRA)**, **ElevenLabs**, and **Gemini**.
- **Live Visual Tag & Instruction Preview**: Dynamic highlighting of tags and emotional instructions.
- **Built-in Audio Player Studio**: 1-click **"🎙️ สร้างเสียงพูด (Synthesize Audio)"** with Play/Pause, Timeline, and WAV download.

---

## API Endpoints

### `POST /synthesize`
Synthesizes speech with registered voice or base model:
```json
{
  "text": "[calm] หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ",
  "speaker_id": "speaker_1",
  "engine": "voxcpm",
  "cfg_value": 2.5,
  "inference_timesteps": 10,
  "auto_annotate": true
}
```
*Returns: `audio/wav` binary stream (48kHz)*

### `POST /synthesize/upload`
Synthesizes speech with a direct one-off uploaded reference audio file (Multipart Form).

### `GET /speakers`
Lists all registered voice profiles and their prompt cache status.

### `POST /speakers`
Uploads a reference audio clip (`file`) to register a new custom voice profile in `ref/` and caches prompt latents in `voice_cache/*.pt`.

### `DELETE /speakers/{speaker_id}`
Deletes a registered voice profile.

### `POST /annotate`
Analyzes raw Thai text into emotional clauses and intensities.

### `POST /render`
Renders annotated segments into engine-specific formats (`voxcpm`, `elevenlabs`, `gemini`).

### `GET /health`
Health check endpoint returning system status and registered speaker count.
