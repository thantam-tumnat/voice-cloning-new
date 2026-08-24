# Deploying the SiangTTS webhook service

`src/webhook.py` replaces the whole n8n "LIveAI_Audio" flow with one process.
Callers keep the same contract: POST a script, get `{"status":"success"}` back
immediately, receive the finished audio URL on `callback_url` when it's done.

| n8n node | now |
|---|---|
| Webhook, set, success/error | `POST /webhook/live-ai-create-new` |
| pyThai, chunker | `src/thai_text.py` (tested, in git) |
| send_prompt_new, get_job_new, Wait, If, Loop, Aggregate | in-process loop — no polling |
| merge_audio_ffmpeg-trimmed (svc :8000) | `pipeline.merge_chunks` |
| upload to, send_link, send_link_error | `pipeline.upload`, `pipeline.post_callback` |
| Cleanup (svc :8000) | `pipeline.cleanup` |

Services `:8000` and `:8001` are no longer used by this path.

## 1. Prerequisites

- NVIDIA GPU, ~8 GB VRAM free. The 3090 is shared with IndexTTS — check
  `nvidia-smi` before starting.
- Driver 575.x / CUDA 12.9 → torch stays on **cu128** wheels (already pinned in
  `pyproject.toml`).
- `ffmpeg` on PATH (7.x verified).
- Python 3.10–3.12, `uv`.

```bash
uv sync --extra serve
```

## 2. Configuration

```bash
SIANGTTS_ADAPTER=checkpoints/siangtts-v1      # REQUIRED — service refuses to start if missing
SIANGTTS_UPLOAD_TOKEN=<bearer>                # was the n8n credential "VR_live Auth"
SIANGTTS_DEFAULT_CALLBACK=https://test.looklike.ai/api/v1/live-gpt/n8n/audio-callback
SIANGTTS_REF_DIR=ref                          # or C:\temp\tts_jobs\voices to reuse the old server's clips
SIANGTTS_CACHE_DIR=voice_cache                # derived .pt files — NOT the reference audio
SIANGTTS_WORK_DIR=work
PYTHONIOENCODING=utf-8                        # Windows consoles are cp1252; Thai log lines crash without this
```

Tuning knobs: `SIANGTTS_NUM_STEP=10`, `SIANGTTS_GUIDANCE=2`.

`GUIDANCE` comes from the old flow's `send_prompt_new` body. `NUM_STEP` does not:
the flow's `32` was an IndexTTS value, and here each LM step re-runs the
flow-matching DiT that many times, doubled again by CFG — 64 DiT forwards per
step, ~1s/step on a build without `torch.compile` (no triton). VoxCPM's own
default is `10`, which is what the paper reports and what every other entry
point in this repo uses. Raise it if you hear artifacts; 16 is the ceiling worth
trying.

Set `SIANGTTS_KEEP_WORK=1` to keep per-job scratch dirs while debugging. Failed
jobs keep theirs regardless — note that this means a run of failures accumulates
audio under `SIANGTTS_WORK_DIR` and nothing prunes it.

> The adapter path is a hard failure on purpose. `src/serve.py` only warned and
> fell back to the base model, which sounds plausible but has no Thai LoRA — an
> easy thing to ship without noticing.

## 3. Voices

Reference clips go in `ref/` named after the `voice_id` the caller sends:

```
ref/2f6d7e8a-8767-4875-95c3-360fb061a090.mp3
ref/thai_female.mp3          # the default when voice_id is absent
```

Copy them over from the old server's `C:\temp\tts_jobs\voices\` — or skip the
copy and set `SIANGTTS_REF_DIR` to that folder directly. (The old system's
`voices\` holds reference *audio*; ours holds derived caches. Same word, two
meanings — hence `SIANGTTS_CACHE_DIR` for the latter.) Encodings are
built on first use and cached in `voice_cache/`; the cache key includes the
`voice_text` so a voice sent with a different reference transcript re-encodes
rather than silently reusing the wrong prompt cache.

## 4. Run

```bash
uv run uvicorn src.webhook:app --host 0.0.0.0 --port 8002
```

**One worker only.** Every uvicorn worker loads its own copy of the model into
VRAM. Concurrency is handled inside the process: jobs go through a queue and run
one at a time, so bursts wait instead of thrashing the GPU.

### Windows service (NSSM)

```bash
nssm install SiangTTS "C:\Users\thantam\.local\bin\uv.exe" "run uvicorn src.webhook:app --host 0.0.0.0 --port 8002"
nssm set SiangTTS AppDirectory "C:\path\to\VoxCPM-thai"
nssm set SiangTTS AppEnvironmentExtra SIANGTTS_ADAPTER=checkpoints/siangtts-v1 SIANGTTS_UPLOAD_TOKEN=... PYTHONIOENCODING=utf-8
nssm start SiangTTS
```

Don't containerise on Windows — GPU passthrough there is more trouble than the
isolation is worth. On Linux, a systemd unit with the same command works.

## 5. Smoke test

```bash
curl -s http://localhost:8002/health
```

Expect `"status":"ok"`, a `sample_rate`, and `"upload_token":true`. The model
takes ~30–60 s to load; until then the port is not listening.

Then replay the payload pinned in the old workflow:

```bash
curl -s -X POST http://localhost:8002/webhook/live-ai-create-new -H "Content-Type: application/json" -d "{\"job_id\":\"smoke-1\",\"queue_id\":\"smoke-1\",\"prompt\":\"สนใจสินค้าตัวไหน กดที่ตะกร้าได้เลยนะคะ\",\"audio_speed\":1,\"voice_id\":\"thai_female\",\"country_code\":\"th\",\"callback_url\":\"https://webhook.site/<your-id>\"}"
```

Returns `{"status":"success","job_id":"smoke-1","chunks":1}`. Follow progress
with `GET /jobs/smoke-1`; the audio URL lands on the callback URL.

## 6. Watching the queue

n8n's execution list is gone; these three endpoints replace it.

| endpoint | answers |
|---|---|
| `GET /` | queue page — live table in the browser, refreshes every 2 s |
| `GET /health` | is the service up, how deep is the queue, how many have failed |
| `GET /jobs` | the last 500 jobs, newest first — `?status=failed`, `?limit=20` |
| `GET /jobs/{job_id}` | one job in detail |

A job reports `status` (`queued` → `running` → `completed` / `failed`),
`progress` as `chunks_done/total`, `position` in line while it waits,
`waited_s` before it started, `elapsed_s` since, and `file_url` or `error` at
the end.

```bash
curl -s http://localhost:8002/jobs?status=failed | python -m json.tool
```

Reading the numbers:

- `waiting` climbing while `running` never changes → a job is wedged. Check the
  console; the chunk loop prints a line per chunk.
- `waited_s` large but `elapsed_s` small → the GPU is the bottleneck, not the
  code. Jobs run one at a time by design.
- `voices_cached` growing past the number of real voices → callers are sending
  varying `voice_text` for the same `voice_id`, and each variant re-encodes.
- `upload_token: false` → `SIANGTTS_UPLOAD_TOKEN` is unset and every job will
  fail at the upload step. Check this before the first real request.

History lives in memory and holds the newest `SIANGTTS_MAX_HISTORY` (500)
finished jobs; queued and running jobs are never evicted. A restart clears it.

## 7. Cutover and rollback

Point the caller at `:8002` instead of the n8n webhook. Keep n8n's workflow
deactivated but saved — reactivating it is the rollback, and it still works
because `:8000` and `:8001` are untouched.

## Known gaps

- **`speed`** is applied as an ffmpeg `atempo` on the merged file. The old
  `:8001` server took `speed` as a model parameter, so timbre under
  non-1.0 speed will not match exactly. Callers currently always send `1`
  (the `Speed Reader` node was disabled), so this is untested in anger.
- **`denoise`** (the flow sent `true`) is not wired up. It loads a separate
  heavy model and only matters for noisy reference clips.
- `t_shift`, `position_temperature`, `class_temperature`, `layer_penalty_factor`,
  `duration`, `preprocess_prompt`, `postprocess_output`, `audio_chunk_*`,
  `file_check` were IndexTTS-only. They are accepted and ignored.
- Job state is in memory. A restart loses the queue; in-flight jobs never call
  back, and `/jobs` starts empty. Fine at current volume — revisit if the queue
  is ever deep.
- No way to cancel a queued job short of restarting the service.
