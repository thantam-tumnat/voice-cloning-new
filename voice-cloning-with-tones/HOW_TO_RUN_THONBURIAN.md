# How to Run ThonburianTTS + SeedVC Tone Studio (:8012)

This guide explains how to run the emotional speech generation pipeline using **Thonburian F5 (Emotion Transfer)** + **SeedVC (Timbre Conversion)** on the Tone Studio UI.

---

## Architecture Overview

```
[User Input with [emotion] tags] -> Tone Studio (:8012)
   │
   ├─► 1. Thonburian F5 (fp16 on RTX 4070 SUPER)
   │      - Clones emotion donor from airesearch/thai-ser (neutral, angry, happy, sad, frustrated)
   │      - Respects chosen Gender (👩 Female / 👨 Male)
   │      - Generates 24 kHz emotional Thai speech
   │
   ├─► 2. SeedVC Worker (:8022)
   │      - Persistent server (SeedVCWrapper loaded once)
   │      - f0_condition=True, auto_f0_adjust=True
   │      - Converts timbre onto target speaker voice at 44.1 kHz
   │
   └─► 3. Audio Assembler (audio_post)
          - Assembles chunks, balances loudness, pace & natural pauses
          - Emits final 44.1 kHz WAV
```

---

## Step 1: Start SeedVC Worker (Port 8022)

In a terminal window (or double-click `run_seedvc.bat`):

```powershell
cd voice-cloning-with-tones
.\run_seedvc.bat
```

Or manually using the virtual environment:
```powershell
C:\Users\opendream002\Desktop\seed-vc\seedvc-venv\Scripts\python.exe tools\seedvc_server.py --seedvc-repo C:\Users\opendream002\Desktop\seed-vc --port 8022
```

*When ready, it outputs:*
```
[seedvc] loading SeedVCWrapper (device=auto) …
[seedvc] ready in 3s on cuda:0
```

---

## Step 2: Start Tone Studio (Port 8012)

In a second terminal window (or double-click `run_studio_8012.bat`):

```powershell
cd voice-cloning-with-tones
.\run_studio_8012.bat
```

Or manually:
```powershell
set FLOWTTS_SRC=C:\Users\opendream002\Desktop\thonburian\thonburian-tts
set SERVICE_PORT=8012
set SEEDVC_URL=http://127.0.0.1:8022
python -m uvicorn app.main:app --host 0.0.0.0 --port 8012 --reload
```

---

## Step 3: Open Tone Studio in Browser

1. Open your browser and navigate to:
   **`http://localhost:8012/`**

2. **Select Voice Gender:**
   - 👩 **หญิง (Female)**
   - 👨 **ชาย (Male)**

3. **Select or Upload Target Speaker:**
   - Choose a registered voice (e.g. `determination`, `lion`, `customer`) or upload an audio sample.

4. **Type Thai Text with Emotion Tags:**
   Supported emotions: `[neutral]`, `[happy]`, `[angry]`, `[sad]`, `[frustrated]`.
   
   Example:
   ```
   [happy] วันนี้อากาศสดใสมากเลยครับ [angry] แต่ทำไมรถติดขนาดนี้! [sad] สงสัยจะไปทำงานสายแน่ๆ
   ```

5. Click **"สังเคราะห์เสียง (Synthesize Speech)"** and play the generated audio!

---

## Dataset Separation (~1 GB Male / Female)

To download or refresh the dataset separated into `dataset/male/` (~500 MB) and `dataset/female/` (~500 MB) with transcripts:

```powershell
python tools/download_gender_dataset.py --mb-per-gender 500 --out-dir dataset --donor-dir ref/emotions
```
