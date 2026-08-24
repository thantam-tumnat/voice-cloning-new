from enum import Enum
from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class Tone(str, Enum):
    NEUTRAL = "neutral"
    SAD = "sad"
    HAPPY = "happy"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    NERVOUS = "nervous"
    SARCASTIC = "sarcastic"
    SCARED = "scared"
    TIRED = "tired"


class Segment(BaseModel):
    text: str
    tone: Tone
    intensity: int = Field(default=2, ge=1, le=3)
    # Free-form style word as written, e.g. "appalled". `tone` stays the coarse
    # family used for colour and for the ElevenLabs/Gemini renderers.
    style: Optional[str] = None
    # Text with prosodic punctuation marks added for expressive TTS delivery
    spoken_text: Optional[str] = None
    # The source put a line break before this segment's tag, which earns a longer
    # pause than an inline tone change. Kept on the segment so the short script form
    # can round-trip it.
    break_before: bool = False


class AnnotateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    guidance: Optional[str] = Field(default=None, description="Optional custom emotion/tone guidance")
    model: Optional[str] = Field(default=None, description="Optional specific LLM model to use")


class AnnotateResponse(BaseModel):
    original: str
    segments: list[Segment]
    model_used: str
    fallback: bool  # True = validate failed, fallback to all neutral
    error: Optional[str] = None
    error_detail: Optional[str] = None
    attempts: Optional[list[dict]] = None
    warnings: List[str] = Field(default_factory=list)


class RenderRequest(BaseModel):
    segments: list[Segment]
    engine: Literal["elevenlabs", "gemini", "voxcpm", "siangtts"]


class RenderedChunk(BaseModel):
    """One synthesis unit: a style instruction plus the text it applies to."""
    text: str  # instruction + body, ready to hand to the engine
    instruction: Optional[str] = None
    body: str = ""
    # Carried through to the audio assembler, which sets this chunk's loudness and
    # the pause in front of it from the tone. Without it every chunk lands at
    # whatever level the model happened to pick.
    tone: Optional[str] = None
    # True when the source had a line break before this chunk, which earns a longer
    # pause than an inline tone change does.
    break_before: bool = False


class RenderResponse(BaseModel):
    text: str  # text ready for TTS / instruction prompt
    prompt: Optional[str] = None  # for engines using separate field (Gemini/VoxCPM summary)
    # The short, editable form: "[sad] ... [happy] ...". This is what the studio puts
    # in the editable box, because `text` is a single-shot rendering that carries only
    # the FIRST instruction -- sending that back collapsed a multi-emotion script into
    # one tone.
    script: Optional[str] = None
    # Per-segment units. VoxCPM2 only honours a style parenthetical at the start of
    # the text it is given, so multi-tone input must be synthesized chunk by chunk.
    chunks: List[RenderedChunk] = Field(default_factory=list)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    guidance: Optional[str] = Field(default=None, description="Optional custom emotion/tone guidance")
    engine: Literal["elevenlabs", "gemini", "voxcpm", "siangtts"] = "voxcpm"
    model: Optional[str] = Field(default=None, description="Optional specific LLM model to use")


class SpeakResponse(BaseModel):
    engine: Literal["elevenlabs", "gemini", "voxcpm", "siangtts"]
    text: str
    prompt: Optional[str] = None
    segments: list[Segment]
    model_used: str
    fallback: bool
    error: Optional[str] = None
    error_detail: Optional[str] = None
    attempts: Optional[list[dict]] = None
    chunks: Optional[list[RenderedChunk]] = None
    script: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class PronunciationResponse(BaseModel):
    """The custom pronunciation overrides applied just before synthesis."""
    entries: dict[str, str]
    path: str


class PronunciationUpdateRequest(BaseModel):
    # Replaces the whole dictionary. Keys are matched on word boundaries, so
    # respelling "ไฟล์" leaves "โปรไฟล์" -- a genuinely different vowel -- alone.
    entries: dict[str, str]


class SpeakerInfo(BaseModel):
    id: str
    name: str
    filename: str
    cached: bool


class SpeakerListResponse(BaseModel):
    speakers: List[SpeakerInfo]


class PostProcessParams(BaseModel):
    """Per-request overrides for the audio_post DSP module.

    Every field is optional and defaults to ``None``, which means "leave the measured
    constant in app/services/audio_post.py alone". Only the keys the client actually
    moved are sent, so the reference values stay the source of truth.
    """
    gap_same_tone_s: Optional[float] = Field(default=None, ge=0.0, le=3.0)
    gap_emotion_s: Optional[float] = Field(default=None, ge=0.0, le=3.0)
    gap_paragraph_s: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    match_energy: Optional[bool] = None
    match_rate: Optional[bool] = None
    energy_match: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_stretch: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    trim_floor_db: Optional[float] = Field(default=None, ge=10.0, le=80.0)
    trim_keep_s: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    edge_fade_s: Optional[float] = Field(default=None, ge=0.0, le=0.2)
    output_peak: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    # Per-tone overrides, merged over the module tables. Keys are Tone values.
    tone_energy_db: Optional[dict[str, float]] = None
    tone_duration_ratio: Optional[dict[str, float]] = None


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    speaker_id: Optional[str] = None
    guidance: Optional[str] = None
    engine: Literal["voxcpm", "siangtts", "elevenlabs", "gemini"] = "voxcpm"
    model: Optional[str] = Field(default=None, description="Optional specific LLM model to use")
    cfg_value: float = Field(default=2.5, ge=1.0, le=10.0)
    inference_timesteps: int = Field(default=10, ge=4, le=50)
    auto_annotate: bool = True
    lora_mode: Optional[Literal["on", "off", "legacy"]] = Field(
        default="on", description="LoRA mode: 'on' (Thai optimized), 'off' (Base model), or 'legacy' (shipped 2.0/2.0)"
    )
    post_process: bool = Field(default=True, description="Enable audio_post DSP processing (loudness, pace, pauses)")
    post_process_params: Optional[PostProcessParams] = Field(
        default=None, description="Per-request overrides for the audio_post DSP constants"
    )


class ABVariantSpec(BaseModel):
    """One way of assembling the shared generation."""
    id: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=80)
    post_process: bool = True
    params: Optional[PostProcessParams] = None


class ABChunkMetric(BaseModel):
    """Where one chunk landed in a variant, and how loud it came out."""
    tone: Optional[str] = None
    start_s: float
    end_s: float
    dur_s: float
    level_db: Optional[float] = None
    text_len: int = 0
    pace_s_per_char: Optional[float] = None


class ABVariantResult(BaseModel):
    id: str
    label: str
    filename: str
    audio_url: str
    dur_s: float
    # Loudest chunk minus quietest, in dB: the single number that says how much
    # emotional contrast survived this assembly.
    level_spread_db: Optional[float] = None
    # Slowest chunk's pace over the fastest, as a percentage. Pace is measured in
    # seconds per character so chunks of different lengths stay comparable.
    pace_spread_pct: Optional[float] = None
    chunks: List[ABChunkMetric] = Field(default_factory=list)


class ABSynthesizeRequest(BaseModel):
    """Render one generation, assemble it several ways.

    Sampling is not deterministic, so comparing two ordinary /synthesize calls
    compares two different takes as well as two treatments. This renders once and
    hands the same chunks to every variant, which is the only way the difference
    heard is the treatment.
    """
    text: str = Field(min_length=1, max_length=5000)
    speaker_id: Optional[str] = None
    guidance: Optional[str] = None
    model: Optional[str] = None
    cfg_value: float = Field(default=2.5, ge=1.0, le=10.0)
    inference_timesteps: int = Field(default=10, ge=4, le=50)
    auto_annotate: bool = True
    lora_mode: Optional[Literal["on", "off", "legacy"]] = "on"
    variants: List[ABVariantSpec] = Field(min_length=1, max_length=6)


class ABSynthesizeResponse(BaseModel):
    run_id: str
    sample_rate: int
    chunk_count: int
    tones: List[Optional[str]] = Field(default_factory=list)
    variants: List[ABVariantResult]


class LLMClauseItem(BaseModel):
    i: int
    text: str


class LLMClauseLabel(BaseModel):
    i: int
    tone: Tone
    intensity: int = Field(default=2, ge=1, le=3)
    spoken_text: Optional[str] = None


class LLMAnnotationResult(BaseModel):
    labels: list[LLMClauseLabel]


class LLMTagConversionResult(BaseModel):
    instruction: str
    tone: Tone
    intensity: int = Field(default=2, ge=1, le=3)


# ---------------------------------------------------------------------------
# Benchmark & Testing Suite Models
# ---------------------------------------------------------------------------

class BenchmarkSessionInitRequest(BaseModel):
    name: Optional[str] = None
    speaker_id: Optional[str] = None
    text: str = Field(min_length=1, max_length=5000)
    emotions: List[str] = Field(default_factory=lambda: [t.value for t in Tone])
    repeats: int = Field(default=3, ge=1, le=10)
    intensity: int = Field(default=2, ge=1, le=3)
    cfg_value: float = Field(default=2.5, ge=1.0, le=10.0)
    inference_timesteps: int = Field(default=10, ge=4, le=50)
    lora_mode: Optional[Literal["on", "off", "legacy"]] = "on"
    post_process: bool = True
    post_process_params: Optional[PostProcessParams] = None


class BenchmarkSessionInitResponse(BaseModel):
    session_id: str
    name: str
    created_at: str
    speaker_id: Optional[str]
    text: str
    emotions: List[str]
    repeats: int
    total_takes: int
    params: dict


class BenchmarkTakeRequest(BaseModel):
    session_id: str
    emotion: str
    take_idx: int = Field(default=1, ge=1, le=10)
    text: str = Field(min_length=1, max_length=5000)
    instruction: Optional[str] = None
    intensity: int = Field(default=2, ge=1, le=3)
    speaker_id: Optional[str] = None
    cfg_value: float = Field(default=2.5, ge=1.0, le=10.0)
    inference_timesteps: int = Field(default=10, ge=4, le=50)
    lora_mode: Optional[Literal["on", "off", "legacy"]] = "on"
    post_process: bool = True
    post_process_params: Optional[PostProcessParams] = None
    # When present, the take is generated once and assembled this many ways, which
    # is the only way a DSP comparison is not also a comparison of two samplings.
    # Empty means the single-take behaviour driven by post_process above.
    variants: Optional[List[ABVariantSpec]] = Field(default=None, max_length=6)


class BenchmarkTakeVariant(BaseModel):
    """One DSP treatment of a take, built from the same generation as its siblings."""
    id: str
    label: str
    filename: str
    audio_url: str
    metrics: Optional[dict] = None


class BenchmarkTakeResult(BaseModel):
    session_id: str
    emotion: str
    take_idx: int
    instruction: str
    spoken_text: str
    audio_url: str
    filename: str
    metrics: Optional[dict] = None
    elapsed_s: float = 0.0
    error: Optional[str] = None
    # Every DSP treatment of this take, first entry mirroring the fields above.
    variants: List[BenchmarkTakeVariant] = Field(default_factory=list)


class BenchmarkSessionSummary(BaseModel):
    session_id: str
    name: str
    created_at: str
    speaker_id: Optional[str] = None
    text: str
    emotions: List[str]
    repeats: int
    total_takes: int
    completed_takes: int
    params: dict


