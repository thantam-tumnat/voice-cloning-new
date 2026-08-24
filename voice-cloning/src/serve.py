"""FastAPI central inference server for SiangTTS (VoxCPM2 + Thai LoRA).

Shared GPU backend for:
- Standalone Queue Service (Port 8010)
- Tone Studio Pipeline (Port 8011)

Interface modelled on KhongkhunAPI / VoxCPM2:
    GET  /health
    POST /tts                      text [+ reference / speaker_id] -> wav
    POST /tts/speaker/{speaker_id} text in a registered voice
    POST /synthesize               JSON payload for programmatic TTS
    POST /speakers                 register a voice (saves clip to ref/ + caches it)
    GET  /speakers                 list registered voices
    DELETE /speakers/{speaker_id}  remove a voice
    POST /speakers/seed/reroll     reset unpinned seed voice
    GET  /                         small HTML test form · /docs for Swagger

Run:
    uv run uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .inference import DEFAULT_BASE_MODEL, Synthesizer

_BASE_MODEL = os.environ.get("SIANGTTS_BASE_MODEL", DEFAULT_BASE_MODEL)
_ADAPTER = os.environ.get(
    "SIANGTTS_ADAPTER",
    "checkpoints/siangtts-v1" if Path("checkpoints/siangtts-v1").exists() else "checkpoints/siangtts-lora-v0/latest",
)
_DEVICE = os.environ.get("SIANGTTS_DEVICE") or None

REF_DIR = Path(os.environ.get("SIANGTTS_REF_DIR", "ref"))
CACHE_DIR = Path(os.environ.get("SIANGTTS_CACHE_DIR", "voice_cache"))
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
DEFAULT_VOICE = os.environ.get("SIANGTTS_DEFAULT_VOICE", "thai_female")
DEFAULT_SEED_TEXT = os.environ.get(
    "SIANGTTS_SEED_TEXT", "วันนี้อากาศปกติ อุณหภูมิยี่สิบห้าองศา"
)

_state: dict = {}
_inference_lock = asyncio.Lock()


class TTSJsonRequest(BaseModel):
    text: str
    speaker_id: Optional[str] = None
    ref_text: Optional[str] = None
    cfg_value: float = Field(default=2.5, ge=0.5, le=5.0)
    inference_timesteps: int = Field(default=10, ge=1, le=50)
    lora_mode: Optional[str] = "on"


def _find_ref_file(speaker_id: str) -> Optional[Path]:
    """Find audio reference file for speaker_id in REF_DIR or fallback directories."""
    dirs_to_check = [REF_DIR, Path("C:/temp/tts_jobs/voices"), Path("voices")]
    for d in dirs_to_check:
        if not d.exists():
            continue
        for ext in AUDIO_EXTS:
            candidate = d / f"{speaker_id}{ext}"
            if candidate.exists():
                return candidate
    return None


def _get_or_build_voice(synth: Synthesizer, speaker_id: str, ref_text: Optional[str] = None) -> dict:
    """Retrieve voice cache for speaker_id, building or loading from disk if needed."""
    voices: dict = _state.setdefault("voices", {})
    key = f"{speaker_id}::{ref_text}" if ref_text else speaker_id

    if key in voices:
        return voices[key]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_name = f"{speaker_id}-{hash(ref_text):x}.pt" if ref_text else f"{speaker_id}.pt"
    cache_path = CACHE_DIR / cache_name

    ref_file = _find_ref_file(speaker_id)
    if not ref_file and not cache_path.exists():
        raise HTTPException(404, f"Speaker '{speaker_id}' reference audio not found.")

    if cache_path.exists() and (not ref_file or cache_path.stat().st_mtime >= ref_file.stat().st_mtime):
        cache = synth.load_voice(cache_path)
    else:
        print(f"[serve] encoding voice '{speaker_id}' from {ref_file} ...")
        cache = synth.build_voice(str(ref_file), prompt_text=ref_text)
        synth.save_voice(cache, cache_path)

    voices[key] = cache
    return cache


def _init_ref_speakers(synth: Synthesizer) -> None:
    """Precompute + cache a prompt cache for every clip in ref/ (skip if cached)."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dirs_to_check = [REF_DIR]
    fallback_dir = Path("C:/temp/tts_jobs/voices")
    if fallback_dir.exists() and fallback_dir.resolve() != REF_DIR.resolve():
        dirs_to_check.append(fallback_dir)

    for d in dirs_to_check:
        for ref_file in sorted(d.iterdir()):
            if ref_file.suffix.lower() not in AUDIO_EXTS:
                continue
            sid = ref_file.stem
            cache_path = CACHE_DIR / f"{sid}.pt"
            if cache_path.exists() and cache_path.stat().st_mtime >= ref_file.stat().st_mtime:
                try:
                    _state["voices"][sid] = synth.load_voice(cache_path)
                    print(f"  [skip] {sid} (cached)")
                    continue
                except Exception as e:
                    print(f"  [re-encode] cache error for {sid}: {e}")
            try:
                print(f"  [init] encoding voice '{sid}' ...")
                cache = synth.build_voice(str(ref_file))
                synth.save_voice(cache, cache_path)
                _state["voices"][sid] = cache
                print(f"  [init] '{sid}' ready")
            except Exception as e:
                print(f"  [error] Failed encoding '{sid}': {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    adapter = _ADAPTER or None
    if adapter and not Path(adapter).exists():
        print(f"[serve] adapter {adapter!r} not found — base only")
        adapter = None
    print(f"[serve] loading base={_BASE_MODEL} adapter={adapter} device={_DEVICE} ...")
    synth = Synthesizer(base_model=_BASE_MODEL, adapter_path=adapter, device=_DEVICE)
    _state["synth"] = synth
    _state["adapter"] = adapter
    _state["voices"] = {}
    _state["seed_voice"] = None
    print("[serve] initialising voices from ref/ ...")
    _init_ref_speakers(synth)
    print(f"[serve] ready — sample_rate={synth.sample_rate} voices={list(_state['voices'])}")
    yield
    _state.clear()


app = FastAPI(
    title="SiangTTS Central Model Service",
    description="Unified VoxCPM2 + Thai LoRA synthesis backend for Queue (8010) & Tone Studio (8011).",
    version="2.0.0",
    lifespan=lifespan,
)


def _wav_response(wav, sample_rate: int) -> Response:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


async def _save_upload(upload: UploadFile, dest: Path | None = None) -> str:
    suffix = Path(upload.filename or "ref.wav").suffix or ".wav"
    if dest is not None:
        dest.write_bytes(await upload.read())
        return str(dest)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await upload.read())
        return tmp.name


# ---------------------------------------------------------------------------
# TTS Endpoints
# ---------------------------------------------------------------------------

@app.post("/tts", summary="TTS (zero-shot cloning / registered speaker / seed)",
          responses={200: {"content": {"audio/wav": {}}}})
async def tts(
    text: Annotated[str, Form(description="Thai text to synthesize")],
    speaker_id: Annotated[Optional[str], Form(description="Optional registered speaker ID")] = None,
    reference: Annotated[Optional[UploadFile], File(description="Optional reference clip to clone")] = None,
    ref_text: Annotated[Optional[str], Form(description="Optional transcript of reference clip")] = None,
    cfg_value: Annotated[float, Form(description="CFG guidance scale (1.0–3.0)")] = 2.5,
    timesteps: Annotated[int, Form(description="Inference steps (4–30)")] = 10,
    lora_mode: Annotated[Optional[str], Form(description="LoRA mode (on/off)")] = "on",
) -> Response:
    synth: Synthesizer = _state["synth"]
    if not text.strip():
        raise HTTPException(400, "text must be non-empty")

    ref_path = None
    async with _inference_lock:
        try:
            if reference is not None:
                ref_path = await _save_upload(reference)
                wav = await asyncio.to_thread(
                    synth.synth,
                    text,
                    ref_audio=ref_path,
                    prompt_text=ref_text,
                    cfg_value=cfg_value,
                    inference_timesteps=timesteps,
                )
            elif speaker_id:
                cache = _get_or_build_voice(synth, speaker_id, ref_text)
                wav = await asyncio.to_thread(
                    synth.synth_cached,
                    text,
                    cache,
                    cfg_value=cfg_value,
                    inference_timesteps=timesteps,
                )
            elif DEFAULT_VOICE in _state.get("voices", {}):
                cache = _state["voices"][DEFAULT_VOICE]
                wav = await asyncio.to_thread(
                    synth.synth_cached,
                    text,
                    cache,
                    cfg_value=cfg_value,
                    inference_timesteps=timesteps,
                )
            else:
                wav = await asyncio.to_thread(
                    synth.synth,
                    text,
                    cfg_value=cfg_value,
                    inference_timesteps=timesteps,
                )
        finally:
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)

    return _wav_response(wav, synth.sample_rate)


@app.post("/tts/speaker/{speaker_id}", summary="TTS with a registered speaker",
          responses={200: {"content": {"audio/wav": {}}}})
async def tts_with_speaker(
    speaker_id: str,
    text: Annotated[str, Form(description="Thai text to synthesize")],
    ref_text: Annotated[Optional[str], Form(description="Optional transcript")] = None,
    cfg_value: Annotated[float, Form(description="CFG guidance scale (1.0–3.0)")] = 2.5,
    timesteps: Annotated[int, Form(description="Inference steps (4–30)")] = 10,
) -> Response:
    synth: Synthesizer = _state["synth"]
    if not text.strip():
        raise HTTPException(400, "text must be non-empty")

    async with _inference_lock:
        cache = _get_or_build_voice(synth, speaker_id, ref_text)
        wav = await asyncio.to_thread(
            synth.synth_cached,
            text,
            cache,
            cfg_value=cfg_value,
            inference_timesteps=timesteps,
        )

    return _wav_response(wav, synth.sample_rate)


@app.post("/synthesize", summary="Synthesize via JSON request",
          responses={200: {"content": {"audio/wav": {}}}})
async def synthesize_json(req: TTSJsonRequest) -> Response:
    synth: Synthesizer = _state["synth"]
    if not req.text.strip():
        raise HTTPException(400, "text must be non-empty")

    spk = req.speaker_id or DEFAULT_VOICE if (DEFAULT_VOICE in _state.get("voices", {}) or _find_ref_file(DEFAULT_VOICE)) else None

    async with _inference_lock:
        if spk:
            cache = _get_or_build_voice(synth, spk, req.ref_text)
            wav = await asyncio.to_thread(
                synth.synth_cached,
                req.text,
                cache,
                cfg_value=req.cfg_value,
                inference_timesteps=req.inference_timesteps,
            )
        else:
            wav = await asyncio.to_thread(
                synth.synth,
                req.text,
                cfg_value=req.cfg_value,
                inference_timesteps=req.inference_timesteps,
            )

    return _wav_response(wav, synth.sample_rate)


# ---------------------------------------------------------------------------
# Speaker management
# ---------------------------------------------------------------------------

@app.post("/speakers", summary="Register a speaker (caches its reference encoding)")
async def register_speaker(
    speaker_id: Annotated[str, Form(description="Unique name/ID for this voice")],
    reference: Annotated[UploadFile, File(description="Reference clip (3–10 s)")],
    ref_text: Annotated[Optional[str], Form(description="Optional reference transcript")] = None,
) -> JSONResponse:
    synth: Synthesizer = _state["synth"]
    REF_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(reference.filename or "ref.wav").suffix or ".wav"
    ref_dest = REF_DIR / f"{speaker_id}{suffix}"
    await _save_upload(reference, dest=ref_dest)

    async with _inference_lock:
        cache = synth.build_voice(str(ref_dest), prompt_text=ref_text)
        synth.save_voice(cache, CACHE_DIR / f"{speaker_id}.pt")
        _state["voices"][speaker_id] = cache

    return JSONResponse({"speaker_id": speaker_id, "status": "registered"})


@app.get("/speakers", summary="List registered speakers")
def list_speakers() -> JSONResponse:
    speakers = set(_state.get("voices", {}).keys())
    if REF_DIR.exists():
        for f in REF_DIR.iterdir():
            if f.suffix.lower() in AUDIO_EXTS:
                speakers.add(f.stem)
    return JSONResponse({"speakers": sorted(speakers)})


@app.delete("/speakers/{speaker_id}", summary="Remove a registered speaker")
def delete_speaker(speaker_id: str) -> JSONResponse:
    voices = _state.get("voices", {})
    keys_to_del = [k for k in voices if k == speaker_id or k.startswith(f"{speaker_id}::")]
    for k in keys_to_del:
        voices.pop(k, None)

    (CACHE_DIR / f"{speaker_id}.pt").unlink(missing_ok=True)
    for ref_file in REF_DIR.glob(f"{speaker_id}.*"):
        ref_file.unlink(missing_ok=True)
    return JSONResponse({"speaker_id": speaker_id, "status": "deleted"})


@app.post("/speakers/seed/reroll", summary="Reset seed voice cache")
def reroll_seed_voice() -> JSONResponse:
    _state["seed_voice"] = None
    seed_cache = CACHE_DIR / "seed_voice.pt"
    removed = False
    if seed_cache.exists():
        seed_cache.unlink(missing_ok=True)
        removed = True
    return JSONResponse({"rerolled": True, "cache_removed": removed})


@app.get("/health", summary="Health check")
def health() -> JSONResponse:
    synth: Synthesizer = _state.get("synth")
    return JSONResponse({
        "status": "ok" if synth else "loading",
        "base_model": _BASE_MODEL,
        "adapter": _state.get("adapter"),
        "sample_rate": getattr(synth, "sample_rate", 48000),
        "speakers": sorted(_state.get("voices", {}).keys()),
        "device": _DEVICE or "auto",
    })


_INDEX_HTML = """<!doctype html><meta charset="utf-8"><title>SiangTTS Model Service</title>
<h2>SiangTTS Central Model Engine (Port 8000)</h2>
<p>Shared VoxCPM2 + Thai LoRA synthesis server for Queue Service (8010) & Tone Studio (8011).</p>
<h3>Text-to-speech Test</h3>
<form action="/tts" method="post" enctype="multipart/form-data" target="_blank">
  <input name="text" size="55" placeholder="พิมพ์ข้อความภาษาไทย" required><br>
  speaker_id: <input name="speaker_id" placeholder="e.g. thai_female"><br>
  reference (optional): <input type="file" name="reference" accept="audio/*"><br>
  <button>Speak</button>
</form>
<p>API docs: <a href="/docs">/docs</a> · Registered speakers: <a href="/speakers">/speakers</a></p>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML
