import os
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.models import (
    AnnotateRequest,
    AnnotateResponse,
    RenderRequest,
    RenderResponse,
    SpeakRequest,
    SpeakResponse,
    SpeakerListResponse,
    SynthesizeRequest,
    PostProcessParams,
    ABSynthesizeRequest,
    ABSynthesizeResponse,
    ABVariantResult,
    ABChunkMetric,
    PronunciationResponse,
    PronunciationUpdateRequest,
    BenchmarkSessionInitRequest,
    BenchmarkSessionInitResponse,
    BenchmarkTakeRequest,
    BenchmarkTakeResult,
    BenchmarkSessionSummary,
)
from app.config import settings
from app.segmenter import segment_text
from app.annotator import annotator
from app.renderers import get_renderer
from app.renderers.voxcpm import (
    split_style_chunk_specs,
    parse_tagged_segments,
    collect_tag_warnings,
)
from app.services.siangtts_service import siangtts_service, SynthesizerUnavailable
from app.services.pronunciation import (
    dictionary_path,
    load_dictionary,
    save_dictionary,
)
from app.services.benchmark_service import benchmark_service



ChunkPlan = tuple[list[str], list[Optional[str]], list[bool]]


def _plan_chunks(
    text: str,
    *,
    auto_annotate: bool,
    guidance: Optional[str],
    model: Optional[str],
    engine: str = "voxcpm",
) -> ChunkPlan:
    """Decide what to synthesize, and how each piece should be placed.

    Returns texts alongside the tone and layout of each, which the assembler needs
    to set per-emotion loudness and the pause in front of each chunk. Hand-written
    style tags win: the user has already said where each emotion starts, so split on
    them rather than re-annotating (and rather than sending one blob, which would
    speak every mid-text tag aloud).
    """
    specs = split_style_chunk_specs(text, use_llm=True, custom_model=model)
    if specs:
        return (
            [s.text for s in specs],
            [s.tone for s in specs],
            [s.break_before for s in specs],
        )

    if not auto_annotate:
        return ([text], [None], [False])

    clauses = segment_text(text)
    annotated = annotator.annotate(
        original_text=text,
        clauses=clauses,
        guidance=guidance,
        custom_model=model,
    )
    renderer = get_renderer(engine if engine in ("voxcpm", "siangtts") else "voxcpm")
    rendered = renderer.render(annotated.segments)
    # One chunk per tone run, each with its instruction leading, so no style
    # parenthetical is ever spoken aloud mid-utterance.
    if rendered.chunks:
        return (
            [c.text for c in rendered.chunks],
            [c.tone for c in rendered.chunks],
            [c.break_before for c in rendered.chunks],
        )
    return ([rendered.text], [None], [False])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to the engine and, when it is local, precompute the speaker caches.
    # A failure here is a warning rather than a crash: the studio's annotation half
    # works without an engine, and saying so on /health beats refusing to boot.
    try:
        siangtts_service.init_speakers()
        status = siangtts_service.status
        print(f"[Startup] synthesizer: {status['mode']} {status.get('remote_url') or ''}".rstrip())
    except Exception as e:
        print(f"[Startup] Warning: synthesizer unavailable: {e}")
    yield


app = FastAPI(
    title="Thai TTS Tone Annotation & Voice Cloning Studio",
    description="LLM-based emotional tone annotation and SiangTTS (VoxCPM2) Voice Cloning pipeline for Thai speech.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder if exists
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def root_ui():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Thai TTS Tone Annotation & Voice Cloning API is running. Visit /docs for API documentation."}


@app.get("/test", include_in_schema=False)
@app.get("/benchmark", include_in_schema=False)
def benchmark_ui():
    test_file = os.path.join(static_dir, "test.html")
    if os.path.exists(test_file):
        return FileResponse(test_file)
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Benchmark UI is loading."}



@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "thai-tts-tone-annotation",
        "version": "2.0.0",
        "provider": settings.llm_provider,
        "default_model": (
            settings.openai_model if settings.llm_provider == "openai"
            else (settings.gemini_model if settings.llm_provider == "gemini" else settings.llm_model)
        ),
        "escalate_model": (
            settings.openai_escalate_model if settings.llm_provider == "openai"
            else (settings.gemini_escalate_model if settings.llm_provider == "gemini" else settings.llm_escalate_model)
        ),
        "speakers_count": len(siangtts_service.list_speakers()),
        "synthesizer": siangtts_service.status,
    }


@app.get("/models")
def get_available_models_endpoint(refresh: bool = False):
    """
    Returns available LLM models from configured API keys (Gemini, Anthropic).
    Pass ?refresh=true to query the providers live.
    """
    return annotator.list_available_models(refresh=refresh)


@app.post("/annotate", response_model=AnnotateResponse)
def annotate_endpoint(req: AnnotateRequest):
    """
    Segment input Thai text into clauses, query LLM for tone & intensity labels,
    validate, merge, and return annotated segments.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Hand-written tags are already an annotation -- re-deriving them with the LLM
    # would both cost a call and leave the raw markers sitting in the spoken text.
    tagged = parse_tagged_segments(text, use_llm=True, custom_model=req.model)
    if tagged:
        return AnnotateResponse(
            original=text,
            segments=tagged,
            model_used=req.model or "manual-tags",
            fallback=False,
            warnings=collect_tag_warnings(text),
        )

    clauses = segment_text(text)
    response = annotator.annotate(
        original_text=text,
        clauses=clauses,
        guidance=req.guidance,
        custom_model=req.model
    )
    return response


@app.post("/render", response_model=RenderResponse)
def render_endpoint(req: RenderRequest):
    """
    Render annotated segments for a specific TTS engine (voxcpm/siangtts, elevenlabs, gemini).
    """
    try:
        renderer = get_renderer(req.engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return renderer.render(req.segments)


@app.post("/speak", response_model=SpeakResponse)
def speak_endpoint(req: SpeakRequest):
    """
    End-to-end preparation pipeline:
    Receives raw text -> annotates tone -> renders engine payload.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    tagged = parse_tagged_segments(text, use_llm=True, custom_model=req.model)
    if tagged:
        annotated = AnnotateResponse(
            original=text,
            segments=tagged,
            model_used=req.model or "manual-tags",
            fallback=False,
            warnings=collect_tag_warnings(text),
        )
    else:
        clauses = segment_text(text)
        annotated = annotator.annotate(
            original_text=text,
            clauses=clauses,
            guidance=req.guidance,
            custom_model=req.model
        )

    renderer = get_renderer(req.engine)
    rendered = renderer.render(annotated.segments)

    return SpeakResponse(
        engine=req.engine,
        text=rendered.text,
        prompt=rendered.prompt,
        segments=annotated.segments,
        model_used=annotated.model_used,
        fallback=annotated.fallback,
        error=annotated.error,
        error_detail=annotated.error_detail,
        attempts=annotated.attempts,
        chunks=rendered.chunks,
        script=rendered.script,
        warnings=annotated.warnings
    )


# ---------------------------------------------------------------------------
# Custom Pronunciation Dictionary
# ---------------------------------------------------------------------------

@app.get("/pronunciation", response_model=PronunciationResponse)
def get_pronunciation_endpoint():
    """Current pronunciation overrides, e.g. {"ไฟล์": "ฟาย"}."""
    return PronunciationResponse(entries=load_dictionary(), path=str(dictionary_path()))


@app.put("/pronunciation", response_model=PronunciationResponse)
def put_pronunciation_endpoint(req: PronunciationUpdateRequest):
    """Replace the pronunciation overrides.

    Takes effect on the next synthesis -- the file is re-read whenever its mtime
    changes, so no restart is needed.
    """
    try:
        saved = save_dictionary(req.entries)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not write dictionary: {e}")
    return PronunciationResponse(entries=saved, path=str(dictionary_path()))


# ---------------------------------------------------------------------------
# Voice Cloning & Speaker Management Endpoints
# ---------------------------------------------------------------------------

@app.get("/speakers", response_model=SpeakerListResponse)
def list_speakers_endpoint():
    """List all registered voice cloning profiles in ref/ directory."""
    speakers = siangtts_service.list_speakers()
    return SpeakerListResponse(speakers=speakers)


@app.post("/speakers/seed/reroll")
def reroll_seed_voice_endpoint():
    """Discard the auto seed voice so the next unpinned request draws a new speaker.

    Only affects requests with no speaker_id: those are conditioned on one cached
    generation, so a bad draw otherwise persists across every retry and restart.
    """
    removed = siangtts_service.reset_seed_voice()
    return {
        "rerolled": True,
        "cache_removed": removed,
        "detail": ("Seed voice cleared; the next synthesis without a pinned speaker "
                   "will mint a new one."),
    }


@app.post("/speakers")
async def register_speaker_endpoint(
    file: UploadFile = File(...),
    speaker_id: Optional[str] = Form(None),
):
    """Upload a reference audio clip (.wav/.mp3) to register a new voice profile."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    sid = speaker_id.strip() if speaker_id and speaker_id.strip() else os.path.splitext(file.filename)[0]
    result = siangtts_service.register_speaker(
        speaker_id=sid,
        audio_bytes=content,
        filename=file.filename or "voice.wav",
    )
    return result


@app.get("/speakers/{speaker_id}/audio")
def get_speaker_audio_endpoint(speaker_id: str):
    """Stream reference audio file for a registered speaker profile."""
    clean_id = speaker_id.strip()

    # 1. Try local file lookup
    local_path = siangtts_service.get_speaker_audio_path(clean_id)
    if local_path and local_path.exists():
        suffix = local_path.suffix.lower()
        media_type = "audio/mpeg" if suffix in (".mp3", ".m4a") else ("audio/ogg" if suffix == ".ogg" else "audio/wav")
        return FileResponse(
            local_path,
            media_type=media_type,
            headers={"Content-Disposition": f'inline; filename="{local_path.name}"'}
        )

    # 2. Try remote GPU service if active
    remote = siangtts_service._remote()
    if remote is not None:
        audio_info = remote.get_speaker_audio_bytes(clean_id)
        if audio_info is not None:
            content, media_type, filename = audio_info
            return Response(
                content=content,
                media_type=media_type,
                headers={"Content-Disposition": f'inline; filename="{filename}"'}
            )

    raise HTTPException(status_code=404, detail=f"Reference audio for speaker '{speaker_id}' not found")


@app.delete("/speakers/{speaker_id}")
def delete_speaker_endpoint(speaker_id: str):
    """Remove a voice cloning profile."""
    deleted = siangtts_service.delete_speaker(speaker_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"deleted": True, "speaker_id": speaker_id}


# ---------------------------------------------------------------------------
# Audio Synthesis Endpoint
# ---------------------------------------------------------------------------

@app.post("/synthesize")
async def synthesize_endpoint(req: SynthesizeRequest):
    """
    Synthesizes speech using SiangTTS (VoxCPM2) or mapped engines.
    If auto_annotate is True, extracts emotions and injects control instructions automatically.
    Returns 48kHz WAV audio stream with timestamped filename.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    parts, tones, breaks = _plan_chunks(
        text,
        auto_annotate=req.auto_annotate,
        guidance=req.guidance,
        model=req.model,
        engine=req.engine,
    )

    lora_mode = req.lora_mode or "on"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_tag = "lora_off" if lora_mode == "off" else "lora_on"
    spk_tag = f"{req.speaker_id}_" if req.speaker_id else ""
    filename = f"{ts}_{spk_tag}{lora_tag}.wav"

    # Perform synthesis via SiangTTSService
    try:
        wav_bytes = siangtts_service.synthesize_many(
            parts,
            speaker_id=req.speaker_id,
            cfg_value=req.cfg_value,
            inference_timesteps=req.inference_timesteps,
            tones=tones,
            breaks=breaks,
            post_process=req.post_process,
            post_process_params=(
                req.post_process_params.model_dump(exclude_none=True)
                if req.post_process_params else None
            ),
            lora_mode=lora_mode,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )
    except SynthesizerUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.post("/synthesize/upload")
async def synthesize_with_upload_endpoint(
    text: str = Form(...),
    file: Optional[UploadFile] = File(None),
    guidance: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    cfg_value: float = Form(2.5),
    inference_timesteps: int = Form(10),
    auto_annotate: bool = Form(True),
    lora_mode: Optional[str] = Form("on"),
    post_process: bool = Form(True),
    post_process_params: Optional[str] = Form(
        None, description="JSON object of audio_post overrides; same shape as PostProcessParams"
    ),
):
    """
    Synthesizes speech with a direct one-off uploaded reference audio file.
    Returns 48kHz WAV audio stream with timestamped filename.
    """
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    parts, tones, breaks = _plan_chunks(
        clean_text,
        auto_annotate=auto_annotate,
        guidance=guidance,
        model=model,
    )

    audio_bytes = await file.read() if file else None
    ref_filename = file.filename if file else None

    # Multipart carries the overrides as a JSON string; a malformed one is a client
    # error rather than something to silently render at defaults.
    dsp_params = None
    if post_process_params:
        try:
            dsp_params = PostProcessParams.model_validate_json(post_process_params).model_dump(
                exclude_none=True
            )
        except (ValidationError, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid post_process_params: {e}")

    active_lora_mode = lora_mode or "on"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_tag = "lora_off" if active_lora_mode == "off" else "lora_on"
    filename = f"{ts}_custom_{lora_tag}.wav"

    try:
        wav_bytes = siangtts_service.synthesize_many(
            parts,
            ref_audio_bytes=audio_bytes,
            ref_filename=ref_filename,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            tones=tones,
            breaks=breaks,
            post_process=post_process,
            post_process_params=dsp_params,
            lora_mode=active_lora_mode,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )
    except SynthesizerUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.post("/synthesize/ab", response_model=ABSynthesizeResponse)
def synthesize_ab_endpoint(req: ABSynthesizeRequest):
    """Render one generation and return it assembled several ways.

    Two ordinary /synthesize calls cannot answer "did the post-processing help",
    because the sampler produces a different take each time and that variation is
    mixed into whatever the processing did. Here the generation happens once and
    every variant is built from the same chunks, so the difference between the
    returned files is only the treatment -- and it costs one generation, not one
    per variant.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    seen = {v.id for v in req.variants}
    if len(seen) != len(req.variants):
        raise HTTPException(status_code=400, detail="Variant ids must be unique")

    parts, tones, breaks = _plan_chunks(
        text,
        auto_annotate=req.auto_annotate,
        guidance=req.guidance,
        model=req.model,
    )

    specs = [
        {
            "id": v.id,
            "post_process": v.post_process,
            "params": v.params.model_dump(exclude_none=True) if v.params else None,
        }
        for v in req.variants
    ]

    run_id = f"ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        takes, sample_rate, chunk_tones = siangtts_service.synthesize_variants(
            parts,
            variants=specs,
            speaker_id=req.speaker_id,
            cfg_value=req.cfg_value,
            inference_timesteps=req.inference_timesteps,
            tones=tones,
            breaks=breaks,
            lora_mode=req.lora_mode or "on",
        )
    except SynthesizerUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"A/B synthesis failed: {str(e)}")

    # Written into the benchmark runs tree so the existing audio route can serve
    # them and the existing history cleanup applies.
    run_dir = benchmark_service._session_dir(run_id)

    results = []
    for take, spec in zip(takes, req.variants):
        filename = f"{spec.id}.wav"
        (run_dir / filename).write_bytes(take["wav"])

        levels = [c["level_db"] for c in take["chunks"] if c["level_db"] is not None]
        paces = [
            c["pace_s_per_char"] for c in take["chunks"]
            if c.get("pace_s_per_char")
        ]
        results.append(ABVariantResult(
            id=spec.id,
            label=spec.label,
            filename=filename,
            audio_url=f"/api/benchmark/audio/{run_id}/{filename}",
            dur_s=take["dur_s"],
            level_spread_db=round(max(levels) - min(levels), 2) if len(levels) > 1 else None,
            pace_spread_pct=(
                round((max(paces) / min(paces) - 1) * 100, 1) if len(paces) > 1 else None
            ),
            chunks=[ABChunkMetric(**c) for c in take["chunks"]],
        ))

    return ABSynthesizeResponse(
        run_id=run_id,
        sample_rate=sample_rate,
        chunk_count=len(chunk_tones),
        tones=chunk_tones,
        variants=results,
    )


# ---------------------------------------------------------------------------
# Benchmark & Testing Suite Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/benchmark/presets")
def get_benchmark_presets_endpoint():
    """Returns preset test sentences, emotion definitions, and available speakers."""
    return benchmark_service.get_presets()


@app.post("/api/benchmark/session/init", response_model=BenchmarkSessionInitResponse)
def init_benchmark_session_endpoint(req: BenchmarkSessionInitRequest):
    """Initialize a new emotion benchmark session and create storage folder."""
    return benchmark_service.init_session(req)


@app.post("/api/benchmark/run-take", response_model=BenchmarkTakeResult)
def run_benchmark_take_endpoint(req: BenchmarkTakeRequest):
    """Synthesize a single take for an emotion and return audio URL + prosody metrics."""
    return benchmark_service.run_take(req)


@app.get("/api/benchmark/sessions", response_model=list[BenchmarkSessionSummary])
def list_benchmark_sessions_endpoint():
    """List all previous benchmark test runs."""
    return benchmark_service.list_sessions()


@app.get("/api/benchmark/sessions/{session_id}")
def get_benchmark_session_endpoint(session_id: str):
    """Retrieve full metadata and results of a specific benchmark session."""
    data = benchmark_service.get_session(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return data


@app.get("/api/benchmark/audio/{session_id}/{filename}")
def get_benchmark_audio_endpoint(session_id: str, filename: str):
    """Stream synthesized audio file from a benchmark session."""
    path = benchmark_service.get_audio_path(session_id, filename)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@app.get("/api/benchmark/export/{session_id}")
def export_benchmark_session_endpoint(session_id: str):
    """Download ZIP package containing all WAV audio files, report.csv, and session.json."""
    zip_bytes = benchmark_service.export_session_zip(session_id)
    if zip_bytes is None:
        raise HTTPException(status_code=404, detail="Session not found or export failed")
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.zip"'},
    )

