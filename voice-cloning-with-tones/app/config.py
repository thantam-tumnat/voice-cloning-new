import os
from typing import Optional, Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Provider selection: "gemini", "anthropic", or "openai"
    llm_provider: Literal["gemini", "anthropic", "openai"] = "gemini"

    # Gemini (Google AI Studio)
    gemini_api_key: str = ""
    google_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    # Escalation deliberately hops to a different model family: the usual primary
    # failure here is a transient 503 overload, which a sibling model shares.
    gemini_escalate_model: str = "gemini-3.5-flash-lite"

    # Anthropic Claude
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5"
    llm_escalate_model: str = "claude-sonnet-5"

    # 9arm Gateway / OpenAI-Compatible
    openai_api_key: str = ""
    openai_base_url: str = "https://gateway.9arm.co/v1"
    openai_model: str = "qwen3.8-27b-fp8"
    openai_escalate_model: str = "deepseek-v4-flash-0731"

    # Custom pronunciation overrides, applied to the text just before synthesis.
    # See app/services/pronunciation.py -- matching is token-level, so "ไฟล์" can be
    # respelled without touching "โปรไฟล์", which is a genuinely different vowel.
    pronunciation_path: str = "pronunciation.json"

    # Pipeline & Segmenter
    max_segments: int = 20
    reanchor_chars: Optional[int] = None
    segmenter_engine: str = "crfcut"

    # Shared SiangTTS GPU service — the one process that holds VoxCPM2 for every
    # pipeline on the host (voice-cloning/src/gpu_service.py). The studio sends
    # generation there and keeps annotation, chunking and assembly local.
    voxcpm_service_url: str = "http://127.0.0.1:8020"
    # Refuse to run without it. Loading a second copy of the model here would fight
    # the shared one for VRAM on a single-GPU host, which is the exact problem the
    # split exists to solve -- and it would do it silently, several minutes into a
    # request. Set false only when nothing else is using the GPU.
    voxcpm_remote_required: bool = True
    service_port: int = 8011

    # SiangTTS / VoxCPM2 Voice Cloning
    siangtts_base_model: str = "openbmb/VoxCPM2"
    siangtts_adapter: str = "dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA"
    siangtts_device: str = ""
    siangtts_ref_dir: str = "ref"
    siangtts_cache_dir: str = "voice_cache"

    # The denoiser (ZipEnhancer) is only used when generate(denoise=True), which we
    # never do. Loading it costs memory for nothing, so it is off by default.
    siangtts_load_denoiser: bool = False
    # torch.compile warm-up. Faster steady-state, but slow to start and fragile on
    # Windows; enable once the model is confirmed loading.
    siangtts_optimize: bool = False
    # How hard the Thai LoRA is applied, per side of the model. The adapter ships
    # r=64 alpha=128, i.e. strength 2.0 on both the LM (which reads the style
    # parenthetical and the Thai text) and the DiT (which generates the acoustics).
    #
    # Measured with tools/expr_sweep.py --stage lora over 5 emotions x 3 paired
    # reps: at the shipped strength "[angry]" comes out 3.1 dB *quieter* and 3.6
    # semitones *lower* than the same sentence read neutrally -- the opposite of
    # anger on every axis. The DiT side is what does it. Keeping the LM side at
    # full strength and taking the DiT side to zero turns that into +1.2 dB, +1.9
    # st of pitch range and 18% faster, and roughly triples the angry-vs-sad
    # contrast, while the LM side still carries the Thai the adapter was trained
    # for. Nothing else measured helped: cfg 4-6, level-3 wording, Thai-language
    # directions and dropping the seed voice were all neutral or worse.
    #
    # Set siangtts_lora_dit_scale=2.0 to restore the adapter's shipped behaviour.
    siangtts_lora_lm_scale: float = 2.0
    siangtts_lora_dit_scale: float = 0.0
    # When the real model fails to load, fall back to the sine-tone mock instead of
    # raising. Only ever useful for tests -- a silent fallback in production sounds
    # exactly like a broken model.
    siangtts_allow_mock: bool = False
    # With no speaker pinned, VoxCPM2 resamples the timbre on every call, so a
    # multi-chunk utterance changes speaker mid-sentence. Generate a short neutral
    # seed line, clone its timbre, and condition every chunk on that -- including the
    # first, so no chunk inherits another chunk's emotion.
    siangtts_auto_voice_consistency: bool = True
    # VoxCPM2's "ultimate cloning" (prompt audio + its transcript) buys timbre
    # fidelity by reproducing the prompt clip's own rhythm and emotion -- and the
    # docs are explicit that it *ignores the control instruction* while doing so.
    # Every style tag in this studio is a control instruction, so pairing the two
    # silently discards the emotion. Off by default: a sidecar ref/<id>.txt is
    # treated as documentation unless this is deliberately turned on.
    siangtts_hifi_cloning: bool = False
    # Short, emotionally flat, gender-neutral Thai used only to mint that seed voice.
    # Never spoken in the output.
    siangtts_voice_seed_text: str = "วันนี้อากาศปกติ อุณหภูมิยี่สิบห้าองศา"
    # Longest run of spoken characters handed to VoxCPM2 in one generation. It has
    # no internal splitter, and past roughly this length the speaker identity drifts
    # mid-utterance. Anything longer is broken at the best available seam and the
    # pieces are conditioned on one shared voice. Unset or 0 falls back to the
    # module default rather than disabling the split -- there is no safe "off".
    siangtts_max_chunk_chars: int = 140

    @field_validator("reanchor_chars", mode="before")
    @classmethod
    def parse_reanchor_chars(cls, v):
        if v is None or v == "" or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def effective_gemini_api_key(self) -> str:
        return self.gemini_api_key or self.google_api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    @property
    def effective_openai_api_key(self) -> str:
        return self.openai_api_key or os.getenv("OPENAI_API_KEY", "")


settings = Settings()
