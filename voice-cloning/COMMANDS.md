# Command reference

Every command runs from the project root:

```
C:\Users\opendream002\Desktop\SIANGTTS\VoxCPM-thai
```

`DEPLOY.md` explains *why*; this file is just the commands.

---

## Pre-flight

| check | command | want to see |
|---|---|---|
| ffmpeg | `ffmpeg -version` | version 7.x |
| GPU | `nvidia-smi` | ~8 GB free (the 3090 is shared with IndexTTS) |
| LoRA adapter | `dir checkpoints\siangtts-v1\lora_weights.safetensors` | 144,749,240 bytes |
| base model cached | `dir "%USERPROFILE%\.cache\huggingface\hub"` | a `models--openbmb--VoxCPM2` folder |
| reference voices | `dir ref` | one file per `voice_id` callers send |

Download the base model ahead of time (4.7 GB) so the first start isn't a surprise:

```
uv run hf download openbmb/VoxCPM2
```

---

## Install / update dependencies

```
uv sync --extra serve
```

Add `--extra dev` too if you want to run the tests on this machine.

---

## Environment

`cmd` — lasts only for that window:

```
set SIANGTTS_ADAPTER=checkpoints/siangtts-v1
set SIANGTTS_UPLOAD_TOKEN=<bearer>
set PYTHONIOENCODING=utf-8
```

PowerShell:

```
$env:SIANGTTS_ADAPTER = "checkpoints/siangtts-v1"
$env:SIANGTTS_UPLOAD_TOKEN = "<bearer>"
$env:PYTHONIOENCODING = "utf-8"
```

Permanent, for the machine (needs an elevated prompt, and a new window to take effect):

```
setx SIANGTTS_ADAPTER "checkpoints/siangtts-v1" /M
```

### All variables

Which process reads which matters now that there are two. Anything about the model belongs to the **GPU service**; anything about scripts, uploads and callbacks belongs to the **webhook**. `SIANGTTS_WORK_DIR` is the one both must agree on.

| variable | read by | default | what it does |
|---|---|---|---|
| `SIANGTTS_GPU_URL` | webhook | `http://127.0.0.1:8020` | where the GPU service is |
| `SIANGTTS_GPU_STUB` | gpu | — | `1` runs without the model; output is a test tone. Never in production. |
| `SIANGTTS_STUB_DELAY` | gpu | `0` | seconds per chunk in stub mode, so queueing is observable |
| `SIANGTTS_INTERACTIVE_BURST` | gpu | `3` | interactive jobs allowed to jump the batch queue before one batch job goes through regardless |
| `SIANGTTS_DEFAULT_LORA` | gpu | `shipped` | LoRA strength for a job that does not name one |
| `SIANGTTS_GPU_RECONNECT_BUDGET` | webhook | `180` | seconds a running job keeps retrying through an unreachable GPU service before failing |

| variable | default | what it does |
|---|---|---|
| `SIANGTTS_ADAPTER` | `checkpoints/siangtts-v1` | **GPU service.** LoRA directory. **Missing = refuses to start.** `""` runs the base model with no Thai LoRA. |
| `SIANGTTS_BASE_MODEL` | `openbmb/VoxCPM2` | **GPU service.** base HF id |
| `SIANGTTS_DEVICE` | auto | **GPU service.** `cuda` / `cpu` |
| `SIANGTTS_UPLOAD_TOKEN` | — | bearer for the upload endpoint. Unset = every job fails at upload. |
| `SIANGTTS_UPLOAD_URL` | `https://looklike.ai/api/v1/live-gpt/upload` | where the merged mp3 goes |
| `SIANGTTS_DEFAULT_CALLBACK` | `https://test.looklike.ai/.../audio-callback` | used when the caller omits `callback_url` |
| `SIANGTTS_REF_DIR` | `ref` | **GPU service** (the webhook reads it only to list voices while the engine is down). Reference clips, named `<voice_id>.mp3`. Accepts **several directories separated by `;`** — the service answers for both pipelines, so it has to see both their voice folders. `C:\temp\tts_jobs\voices\` is always checked as well. New clips are written to the first entry. |
| `SIANGTTS_CACHE_DIR` | `voices/` if it exists, else `voice_cache` | **GPU service.** Cached encodings (`.pt`), derived from `ref/`. Safe to delete; costs a re-encode. Keys are unchanged by the split, so existing files stay hits. |
| `SIANGTTS_WORK_DIR` | `work` | job scratch, and how chunk WAVs travel between the two services — **they must resolve it to the same directory**. Relative to the working directory; set an absolute path when running as a service. |
| `SIANGTTS_KEEP_WORK` | — | `1` keeps `work/<queue_id>/` for successful jobs too; failed jobs are kept either way |
| `SIANGTTS_NUM_STEP` | `10` | inference steps, sent with each render job (was `num_step` in n8n, where it was `32` — see DEPLOY.md) |
| `SIANGTTS_GUIDANCE` | `2` | CFG scale (was `guidance_scale`) |
| `SIANGTTS_MAX_HISTORY` | `500` | finished jobs kept for `/jobs` |
| `SIANGTTS_HTTP_TIMEOUT` | `120` | seconds for upload + callback |
| `SIANGTTS_FFMPEG` | `ffmpeg` | path to the binary if it isn't on PATH |
| `PYTHONIOENCODING` | — | **set to `utf-8`** or Thai log lines crash the console on Windows |

---

## Run

Two processes now. The model lives in the **GPU service** (`src/gpu_service.py`, port 8020); the **webhook** (`src/webhook.py`, port 8010) keeps the n8n contract and no longer loads anything. Start the GPU service first — the webhook comes up either way, but jobs cannot run until it is there.

### 1. GPU service — the only process that loads the model

```
uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8020
```

Ready when the console prints `[gpu] ready — stub=False sr=48000 …`. Model load takes 30–60 s; the port isn't listening before that.

Bind to **127.0.0.1**, not `0.0.0.0`: it has no authentication and a render job names the folder it writes into.

Do **not** add `--workers N`: each worker loads its own copy of the model into VRAM. Concurrency is handled inside the process by the job queue.

Without a GPU (or while one is busy), `SIANGTTS_GPU_STUB=1` runs the whole service on a test-tone generator — every queue, voice and callback path works, the audio is a sine wave. `/health` reports `"stub": true` and both clients say so on startup.

### 2. Webhook — same port, same contract as before

Testing — localhost only, no one else can reach it:

```
uv run uvicorn src.webhook:app --host 127.0.0.1 --port 8010
```

Reachable from other machines (**the service has no authentication** — make sure the firewall blocks 8010 from the internet):

```
uv run uvicorn src.webhook:app --host 0.0.0.0 --port 8010
```

Ready when the console prints `[webhook] ready — gpu=http://127.0.0.1:8020 work=work`. Starts in about a second — there is no model here any more, which is what makes restarting it cheap.

`SIANGTTS_WORK_DIR` **must match** the GPU service's: that directory is how the chunk WAVs get from one process to the other.

### As Windows services

```
nssm install SiangTTS-GPU "C:\Users\opendream002\.local\bin\uv.exe" "run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8020"
```
```
nssm set SiangTTS-GPU AppEnvironmentExtra SIANGTTS_ADAPTER=checkpoints/siangtts-v1 PYTHONIOENCODING=utf-8
```
```
nssm install SiangTTS "C:\Users\opendream002\.local\bin\uv.exe" "run uvicorn src.webhook:app --host 0.0.0.0 --port 8010"
```
```
nssm set SiangTTS AppEnvironmentExtra SIANGTTS_GPU_URL=http://127.0.0.1:8020 SIANGTTS_UPLOAD_TOKEN=<bearer> PYTHONIOENCODING=utf-8
```
```
nssm set SiangTTS DependOnService SiangTTS-GPU
```

Set `AppDirectory` on **both** to the same folder:

```
nssm set SiangTTS-GPU AppDirectory "C:\Users\opendream002\Desktop\SIANGTTS\VoxCPM-thai"
```
```
nssm set SiangTTS AppDirectory "C:\Users\opendream002\Desktop\SIANGTTS\VoxCPM-thai"
```

`nssm restart SiangTTS` · `nssm stop SiangTTS` · `nssm edit SiangTTS`

`AppDirectory` matters: `work/`, `ref/`, `voices/` and the adapter path are all relative to it — and the two services have to resolve `work/` to the same place.

Restarting the GPU service kills whatever it was generating: its job table is in memory, so any job in flight fails with *"it restarted while the job was queued or running"* and the caller is told through the normal callback. Drain the queue first (`/v2/jobs` should show nothing running) if that matters.

---

## Test

Health:

```
curl http://localhost:8010/health
```

Create audio (cmd — inner quotes must be escaped):

```
curl -X POST http://localhost:8010/webhook/live-ai-create-new -H "Content-Type: application/json" -d "{\"queue_id\":\"smoke1\",\"prompt\":\"สนใจสินค้าตัวไหน กดที่ตะกร้าได้เลยนะคะ\",\"voice_id\":\"demo_female\",\"callback_url\":\"https://webhook.site/xxxx\"}"
```

PowerShell is easier for JSON:

```
Invoke-RestMethod -Method Post http://localhost:8010/webhook/live-ai-create-new -ContentType "application/json" -Body '{"queue_id":"smoke1","prompt":"สนใจสินค้าตัวไหน","voice_id":"demo_female","callback_url":"https://webhook.site/xxxx"}'
```

Or import `SiangTTS.postman_collection.json` into Postman — 7 requests, ready to go.

Browser: `http://localhost:8010/docs` is a full Swagger UI you can fire requests from.

---

## Watch the queue

Open <http://localhost:8010/> in a browser — live table, polls every 2 s, no
build step and no CDN (works on a box with no outbound internet). Everything on
it comes from `/jobs`, so the CLI below shows the same data.

```
curl http://localhost:8010/jobs
```
```
curl http://localhost:8010/jobs/smoke1
```
```
curl "http://localhost:8010/jobs?status=failed"
```

PowerShell, as a table:

```
(Invoke-RestMethod http://localhost:8010/jobs).jobs | Format-Table job_id,status,progress,position,waited_s,elapsed_s
```

Live view, refreshing every 3 s:

```
while ($true) { Clear-Host; Invoke-RestMethod http://localhost:8010/health | Format-List; Start-Sleep 3 }
```

Failures with their reasons:

```
(Invoke-RestMethod "http://localhost:8010/jobs?status=failed").jobs | Format-Table job_id,created,error -Wrap
```

---

## Voices

Register a voice — the filename *is* the `voice_id`:

```
copy C:\temp\tts_jobs\voices\<voice_id>.mp3 ref\
```

No restart needed; it is encoded on first use and cached.

List what's cached:

```
dir voices
```

Force a re-encode (after replacing a reference clip):

```
del voices\<voice_id>-*.pt
```

The cache key is `<voice_id>-<hash of voice_text>`, so the same voice sent with different `voice_text` produces more than one `.pt`. That is expected — a prompt cache is bound to the transcript it was built with.

---

## Output files

```
work\<queue_id>\<queue_id>_000.wav     chunk 1, 48 kHz
work\<queue_id>\<queue_id>_001.wav     chunk 2 …
work\<queue_id>\<queue_id>.mp3         merged, 192 kbps — this is what gets uploaded
```

The merged path is printed to the console as an absolute path the moment it
exists, before the upload is attempted, so it is in the log even when the upload
is the thing that fails.

Deleted after the callback fires, unless `SIANGTTS_KEEP_WORK=1`. **Failed jobs
are always kept** — a failure normally happens at upload or callback, after the
synthesis is done, so the audio survives for inspection or a manual re-upload.
The console says where.

Listen to the newest result:

```
start work\smoke1\smoke1.mp3
```

Sweep leftovers if a run crashed hard:

```
rmdir /s /q work
```

---

## Development

```
uv run --extra dev pytest -q
```
```
uv run --extra dev pytest tests/test_thai_text.py -q
```
```
uvx ruff check src
```

Check text preparation without touching the GPU — how a script gets expanded and split:

```
uv run python -c "from src.thai_text import prepare_prompt, chunk_text; t=prepare_prompt('ราคา 250 บาท ลดเหลือ 199 บาทค่ะ','th'); print(t); [print(c.filename, len(c.text), c.text) for c in chunk_text(t,'demo')]"
```

---

## Update the code

With git:

```
git pull
```
```
uv sync --extra serve
```

Without git — extract `siangtts-patch.zip` over the project root, choosing Replace, then:

```
uv sync --extra serve
```

Restart the service afterwards either way. Confirm the new code is in place:

```
dir src\webhook.py src\thai_text.py src\pipeline.py tests\test_thai_text.py
```

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `RuntimeError: adapter ... not found` | LoRA missing at that path | check `dir checkpoints\siangtts-v1` |
| `voice 'x' has no reference clip in ref/` | no `ref\x.mp3` | copy the clip in, or send a different `voice_id` |
| `SIANGTTS_UPLOAD_TOKEN is not set` | env not set | set it, restart — or accept it while testing and use `SIANGTTS_KEEP_WORK=1` |
| `ffmpeg merge failed - no audio generated` | every chunk failed, or the prompt was empty | look further up the console for the real error |
| `status: "loading"` on /health | model still loading | wait 30–60 s |
| `UnicodeEncodeError` in the console | Windows cp1252 | `set PYTHONIOENCODING=utf-8`, restart |
| CUDA out of memory | IndexTTS is holding the GPU | `nvidia-smi`, free VRAM before starting |
| port 8010 in use | already running | `netstat -ano | findstr :8010` then `taskkill /PID <pid> /F` |
| queue stuck, nothing progressing | a job is wedged | restart — there is no per-job cancel |
