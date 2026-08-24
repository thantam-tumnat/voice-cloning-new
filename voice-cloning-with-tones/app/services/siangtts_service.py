from __future__ import annotations

import io
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Any, Sequence, Tuple

from app.config import settings
from app.services.pronunciation import apply_pronunciation
from app.services.thai_normalizer import normalize_thai_text

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

# Mock-only. The real synthesizer reports the model's own rate.
MOCK_SAMPLE_RATE = 48000


class SynthesizerUnavailable(RuntimeError):
    """Raised when the real VoxCPM2 model could not be loaded."""


_LEADING_STYLE_RE = re.compile(r"^\s*\([^)]*\)\s*")


def spoken_len(text: str) -> int:
    """Characters VoxCPM2 will actually voice.

    The leading style parenthetical is direction, not speech, so counting it would
    make a chunk with a long instruction look slower than it is -- and rate matching
    would then stretch it the wrong way.
    """
    return len(_LEADING_STYLE_RE.sub("", text or "").strip())


# VoxCPM2 voices a whole chunk in one autoregressive pass: voxcpm.core does no
# sentence splitting of its own, and it flattens newlines to spaces before
# generating. Past roughly this many spoken characters the speaker identity
# wanders mid-utterance -- five repetitions of one 98-character line reached the
# model as a single 494-character chunk and came back with the third sentence in
# a different voice. For scale, VoxCPM's own text normalizer splits paragraphs at
# 60-80 tokens. Splitting below the drift point and conditioning every piece on
# one prompt cache is what holds the timbre steady.
DEFAULT_MAX_CHUNK_CHARS = 140


class ChunkPiece(NamedTuple):
    """One synthesizable piece of a chunk that was too long to voice in one pass."""
    text: str              # instruction + body, ready for the engine
    paragraph_seam: bool   # this piece began after a line break in the source


def _atoms(text: str, level: int) -> Optional[List[str]]:
    """Break ``text`` at seam ``level``, or None when that seam is not present.

    Best seam first: a newline is the writer's own sentence mark, then terminal
    punctuation, then a space -- which in Thai separates clauses the way a comma
    does in English. Word tokens come last so even a forced break lands between
    words rather than inside a character cluster.
    """
    if level == 0:
        parts = re.split(r"(?<=\n)", text)
    elif level == 1:
        parts = re.split(r"(?<=[.!?…])\s*", text)
    elif level == 2:
        parts = re.split(r"(?<= )", text)
    elif level == 3:
        try:
            from pythainlp.tokenize import word_tokenize

            parts = word_tokenize(text, keep_whitespace=True)
        except Exception:
            return None
    else:
        return None

    parts = [p for p in parts if p]
    return parts if len(parts) > 1 else None


def _split_body(body: str, limit: int, level: int = 0) -> List[str]:
    """Greedily pack ``body`` into runs of at most ``limit`` spoken characters."""
    if len(body.strip()) <= limit:
        return [body] if body.strip() else []

    atoms = _atoms(body, level)
    if atoms is None:
        if level <= 3:
            return _split_body(body, limit, level + 1)
        # No seam left anywhere: cut on the character budget.
        return [body[i:i + limit] for i in range(0, len(body), limit)]

    packed: List[str] = []
    cur = ""
    for atom in atoms:
        if cur and len((cur + atom).strip()) > limit:
            packed.append(cur)
            cur = atom
        else:
            cur += atom
    if cur.strip():
        packed.append(cur)

    # A single atom can still be over budget on its own -- break that one finer.
    if any(len(p.strip()) > limit for p in packed):
        return [q for p in packed for q in _split_body(p, limit, level + 1)]
    return [p for p in packed if p.strip()]


def split_for_synthesis(text: str, limit: Optional[int] = None) -> List[ChunkPiece]:
    """Break one chunk into pieces VoxCPM2 can voice without the speaker drifting.

    The leading style parenthetical is re-attached to every piece: VoxCPM2 honours
    one only at position 0, so a piece that lost it would fall back to neutral
    partway through an emotion.
    """
    limit = limit or settings.siangtts_max_chunk_chars or DEFAULT_MAX_CHUNK_CHARS

    m = _LEADING_STYLE_RE.match(text or "")
    instruction = m.group(0).strip() if m else ""
    body = (text[m.end():] if m else (text or "")).strip()
    if not body:
        return []

    pieces = _split_body(body, max(1, limit))
    out: List[ChunkPiece] = []
    for i, piece in enumerate(pieces):
        clean = piece.strip()
        if not clean:
            continue
        # A piece that starts a new line earns the same long pause a hand-written
        # line break gets; see audio_post.GAP_PARAGRAPH_S.
        out.append(
            ChunkPiece(
                text=f"{instruction}{clean}",
                paragraph_seam=bool(i and pieces[i - 1].endswith("\n")),
            )
        )
    return out


def prepare_text(text: str) -> str:
    """Everything the text needs between the API and the model.

    Encoding hygiene and pronunciation overrides are applied only over the spoken
    body. The leading style parenthetical is direction for the engine, not speech,
    so normalisation and overrides must never alter characters or punctuation inside it.
    """
    m = _LEADING_STYLE_RE.match(text or "")
    if m:
        instr = m.group(0)
        body = text[m.end():]
        return instr + apply_pronunciation(normalize_thai_text(body))
    return apply_pronunciation(normalize_thai_text(text or ""))


def _wav_to_numpy(wav: Any):
    """Normalize a VoxCPM waveform (tensor or ndarray) to a 1-D float32 array."""
    import numpy as np

    if hasattr(wav, "detach"):
        wav = wav.detach()
        if wav.dim() > 1:
            wav = wav.squeeze(0)
        # bfloat16 has no numpy equivalent; float() is a no-op when already float32.
        return wav.float().cpu().numpy()
    return np.asarray(wav, dtype="float32")


def set_lora_strength(tts_model: Any, lm: float, dit: float) -> Dict[str, float]:
    """Scale the loaded Thai LoRA independently on the LM and the DiT side.

    VoxCPM2 injects LoRA as ``LoRALinear`` layers that keep their strength in a
    ``scaling`` buffer rather than baked into the weights, so this is a live dial,
    not a reload. Nothing on a layer says which side it belongs to -- both sides
    were injected with the same rank and alpha -- so the split comes from where the
    layer lives: the LM is base_lm + residual_lm, the DiT is the feature decoder's
    estimator.

    Why this exists at all, and why the DiT side defaults to zero, is in
    app/config.py next to the settings. Returns what was actually applied so the
    caller can report it; a model without LoRA layers returns zero counts and is
    otherwise untouched.
    """
    try:
        from voxcpm.modules.layers.lora import LoRALinear
    except Exception as e:  # pragma: no cover - depends on the installed voxcpm
        print(f"[SiangTTS] Cannot scale LoRA ({e}); leaving it at its shipped strength.",
              file=sys.stderr)
        return {"lm": 0, "dit": 0}

    def apply(root: Any, value: float) -> int:
        n = 0
        for module in root.modules():
            if isinstance(module, LoRALinear):
                module.scaling.fill_(value)
                n += 1
        return n

    n_lm = apply(tts_model.base_lm, lm) + apply(tts_model.residual_lm, lm)
    n_dit = apply(tts_model.feat_decoder.estimator, dit)
    print(
        f"[SiangTTS] LoRA strength: lm={lm} ({n_lm} layers), dit={dit} ({n_dit} layers)",
        file=sys.stderr,
    )
    return {"lm": lm, "dit": dit, "lm_layers": n_lm, "dit_layers": n_dit}


class _RealSynthesizer:
    """Thin adapter over the VoxCPM wrapper.

    Note that prompt-cache construction and cached generation live on
    ``model.tts_model`` (VoxCPM2Model), not on the ``VoxCPM`` wrapper. Calling them
    on the wrapper silently does nothing useful.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: Optional[str],
        lora_config: Any,
        device: Optional[str],
        load_denoiser: bool,
        optimize: bool,
    ):
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(
            base_model,
            load_denoiser=load_denoiser,
            optimize=optimize,
            device=device,
            lora_config=lora_config,
            lora_weights_path=adapter_path,
        )
        self.tts_model = self.model.tts_model
        self.sample_rate = self.tts_model.sample_rate
        self.lora_loaded = adapter_path is not None
        if self.lora_loaded:
            self.lora_scales = set_lora_strength(
                self.tts_model,
                settings.siangtts_lora_lm_scale,
                settings.siangtts_lora_dit_scale,
            )

    def synth(
        self,
        text: str,
        *,
        ref_audio: Optional[str] = None,
        prompt_cache: Any = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        speaker_id: Optional[str] = None,
        lora_mode: Optional[str] = "on",
        **kwargs,
    ):
        text = prepare_text(text)
        if prompt_cache is not None:
            wav, _, _ = self.tts_model.generate_with_prompt_cache(
                target_text=text,
                prompt_cache=prompt_cache,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                retry_badcase=True,
            )
            return _wav_to_numpy(wav)
        return _wav_to_numpy(
            self.model.generate(
                text=text,
                reference_wav_path=ref_audio,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
        )

    def build_voice(self, ref_audio_path: str, prompt_text: Optional[str] = None) -> Any:
        """Encode a reference clip into a reusable prompt cache.

        With a transcript, VoxCPM2's "ultimate cloning" mode (reference + continuation)
        gives noticeably higher timbre fidelity, so use it when one is available.
        """
        if prompt_text:
            return self.tts_model.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=ref_audio_path,
                reference_wav_path=ref_audio_path,
            )
        return self.tts_model.build_prompt_cache(reference_wav_path=ref_audio_path)

    def save_voice(self, cache: Any, dest_path: Path) -> None:
        import torch

        torch.save(cache, dest_path)

    def load_voice(self, src_path: Path) -> Any:
        import torch

        return torch.load(src_path, map_location="cpu", weights_only=False)


class _MockSynthesizer:
    """Emits a 440 Hz tone. Test scaffolding only -- never a production fallback."""

    def __init__(self):
        self.sample_rate = MOCK_SAMPLE_RATE
        self.lora_loaded = False

    def synth(
        self,
        text: str,
        *,
        ref_audio: Optional[str] = None,
        prompt_cache: Any = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        speaker_id: Optional[str] = None,
        lora_mode: Optional[str] = "on",
        **kwargs,
    ):
        import numpy as np

        text = prepare_text(text)
        duration_sec = max(1.0, min(10.0, len(text) * 0.12))
        num_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 440 * t) * np.exp(-t / 3.0)).astype("float32")

    def build_voice(self, ref_audio_path: str, prompt_text: Optional[str] = None) -> Any:
        return f"mock_latent_for_{ref_audio_path}"

    def save_voice(self, cache: Any, dest_path: Path) -> None:
        dest_path.write_text(str(cache), encoding="utf-8")

    def load_voice(self, src_path: Path) -> Any:
        return f"loaded_mock_from_{src_path}"


class SiangTTSService:
    def __init__(
        self,
        ref_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        base_model: str | None = None,
        adapter_path: str | None = None,
        device: str | None = None,
    ):
        self.ref_dir = Path(ref_dir or settings.siangtts_ref_dir)
        self.cache_dir = Path(cache_dir or settings.siangtts_cache_dir)
        self.base_model = base_model or settings.siangtts_base_model
        self.adapter_path = adapter_path or settings.siangtts_adapter
        self.device = device or settings.siangtts_device or None

        self.ref_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._voices: Dict[str, Any] = {}
        self._synthesizer: Any = None
        self._is_loaded: bool = False
        self._load_error: Optional[str] = None
        self._using_mock: bool = False
        # The neutral voice unpinned multi-chunk requests are conditioned on. Built
        # at most once; see _build_seed_voice.
        self._seed_voice: Any = None
        self._seed_voice_failed: bool = False
        # Ultimate-cloning-vs-style-tags is a property of the voice, not the
        # request, so say it once rather than on every chunk.
        self._hifi_warned: bool = False

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _resolve_adapter(self) -> Optional[str]:
        """Resolve the LoRA adapter to a local directory.

        The configured value is normally a Hugging Face repo id, not a path, so a
        bare ``Path.exists()`` check would always miss and silently drop the LoRA.
        """
        spec = (self.adapter_path or "").strip()
        if not spec:
            return None

        local = Path(spec)
        if local.exists():
            return str(local)

        try:
            from huggingface_hub import snapshot_download

            resolved = snapshot_download(repo_id=spec)
            print(f"[SiangTTS] LoRA adapter resolved: {spec} -> {resolved}", file=sys.stderr)
            return resolved
        except Exception as e:
            # Base VoxCPM2 supports Thai natively, so this degrades quality rather
            # than breaking synthesis. Warn loudly and carry on.
            print(
                f"[SiangTTS] WARNING: could not fetch LoRA adapter '{spec}': {e}\n"
                f"[SiangTTS] Continuing with base {self.base_model} (lower Thai quality).",
                file=sys.stderr,
            )
            return None

    def _load_lora_config(self, adapter_dir: Optional[str]) -> Any:
        if not adapter_dir:
            return None
        cfg_file = Path(adapter_dir) / "lora_config.json"
        if not cfg_file.exists():
            # VoxCPM builds a sensible default when weights are given without a config.
            return None
        try:
            import json

            from voxcpm.model.voxcpm2 import LoRAConfig

            with open(cfg_file, encoding="utf-8") as f:
                cfg_data = json.load(f).get("lora_config", {})
            return LoRAConfig(**cfg_data)
        except Exception as e:
            print(f"[SiangTTS] WARNING: bad lora_config.json ({e}); using defaults.", file=sys.stderr)
            return None

    def get_synthesizer(self) -> Any:
        """Get the synthesizer, preferring the shared GPU service.

        The studio's job is annotation and assembly, not model hosting: generation
        goes to the SiangTTS GPU service, which is the single process holding VoxCPM2
        for every pipeline on the box.

        Falling back to an in-process model when that service is unreachable is off by
        default (`voxcpm_remote_required`). It looks like resilience and is the
        opposite: on a one-GPU host the fallback either fights the shared model for
        VRAM or fails to load after a two-minute pause, and either way it silently
        undoes the whole point of sharing. Failing fast says what is actually wrong.
        """
        if self._synthesizer is not None:
            return self._synthesizer

        remote_url = (getattr(settings, "voxcpm_service_url", "") or "").strip()
        if remote_url:
            from app.services.queue_client import QueueSynthesizer

            remote = QueueSynthesizer(remote_url)
            if remote.check_health():
                self._synthesizer = remote
                self._is_loaded = True
                self._using_mock = False
                self._load_error = None
                print(
                    f"[SiangTTS] Using the shared GPU service at {remote_url} "
                    f"(sample_rate={remote.sample_rate}, lora={'yes' if remote.lora_loaded else 'no'})",
                    file=sys.stderr,
                )
                return self._synthesizer

            self._load_error = f"GPU service at {remote_url} is not answering"
            if getattr(settings, "voxcpm_remote_required", True):
                raise SynthesizerUnavailable(
                    f"{self._load_error}. Start it with:\n"
                    f"    uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8020\n"
                    f"Set VOXCPM_REMOTE_REQUIRED=false to load a second copy of the "
                    f"model in this process instead — only do that if no other "
                    f"pipeline is using the GPU."
                )
            print(
                f"[SiangTTS] {self._load_error}; loading the model in-process "
                f"(VOXCPM_REMOTE_REQUIRED=false).",
                file=sys.stderr,
            )

        try:
            try:
                import torch._dynamo

                torch._dynamo.config.suppress_errors = True
            except Exception:
                pass

            adapter = self._resolve_adapter()
            lora_config = self._load_lora_config(adapter)

            self._synthesizer = _RealSynthesizer(
                base_model=self.base_model,
                adapter_path=adapter,
                lora_config=lora_config,
                device=self.device,
                load_denoiser=settings.siangtts_load_denoiser,
                optimize=settings.siangtts_optimize,
            )
            self._is_loaded = True
            self._using_mock = False
            self._load_error = None
            print(
                f"[SiangTTS] Loaded {self.base_model} "
                f"(LoRA={'yes' if adapter else 'no'}, sample_rate={self._synthesizer.sample_rate})",
                file=sys.stderr,
            )
            return self._synthesizer
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            if not settings.siangtts_allow_mock:
                raise SynthesizerUnavailable(
                    f"VoxCPM2 failed to load: {self._load_error}. "
                    f"This is usually GPU memory (VoxCPM2 needs ~8GB VRAM) or host "
                    f"commit charge. Free the GPU, or set SIANGTTS_DEVICE=cpu. "
                    f"Set SIANGTTS_ALLOW_MOCK=true only if you want a test tone."
                ) from e

            print(
                f"[SiangTTS] WARNING: model load failed ({self._load_error}).\n"
                f"[SiangTTS] SIANGTTS_ALLOW_MOCK is on -- output will be a 440Hz TEST TONE, not speech.",
                file=sys.stderr,
            )
            self._synthesizer = _MockSynthesizer()
            self._using_mock = True
            self._is_loaded = False
            return self._synthesizer

    @property
    def status(self) -> Dict[str, Any]:
        if getattr(self._synthesizer, "is_remote", False):
            mode = "remote"
        elif self._using_mock:
            mode = "mock"
        else:
            mode = "loaded" if self._is_loaded else "unloaded"
        return {
            "loaded": self._is_loaded,
            "mode": mode,
            "using_mock": self._using_mock,
            "load_error": self._load_error,
            "base_model": self.base_model,
            "remote_url": getattr(self._synthesizer, "base_url", None),
            "lora_loaded": bool(getattr(self._synthesizer, "lora_loaded", False)),
            "lora_scales": getattr(self._synthesizer, "lora_scales", None),
            "sample_rate": getattr(self._synthesizer, "sample_rate", None),
            # Loud on purpose: a stub engine returns a test tone, and nothing else
            # downstream would ever mention it.
            "stub_engine": bool(getattr(self._synthesizer, "is_stub", False)),
        }

    # ------------------------------------------------------------------ #
    # Speaker registry
    # ------------------------------------------------------------------ #

    def _transcript_for(self, ref_file: Path) -> Optional[str]:
        """Sidecar transcript (``ref/<id>.txt``), only when Hi-Fi cloning is on.

        Handing VoxCPM2 a transcript alongside the clip switches it into ultimate
        cloning, which the model docs say ignores the control instruction entirely
        and reproduces the prompt clip's own emotion instead. Since every style tag
        here *is* a control instruction, honouring a stray sidecar would silently
        flatten every emotion in the studio -- and leave no error to explain it.
        Opt in with SIANGTTS_HIFI_CLONING=true, knowing tags stop working.
        """
        txt = ref_file.with_suffix(".txt")
        if not txt.exists():
            return None
        if not settings.siangtts_hifi_cloning:
            print(
                f"[SiangTTS] Ignoring transcript {txt.name}: Hi-Fi cloning would "
                f"disable style tags. Set SIANGTTS_HIFI_CLONING=true to use it.",
                file=sys.stderr,
            )
            return None
        content = txt.read_text(encoding="utf-8").strip()
        return content or None

    def _ref_files(self) -> List[Path]:
        return [p for p in sorted(self.ref_dir.iterdir()) if p.suffix.lower() in AUDIO_EXTS]

    def init_speakers(self) -> None:
        """Scan ref/, precomputing and caching prompt latents."""
        try:
            synth = self.get_synthesizer()
        except SynthesizerUnavailable as e:
            print(f"[SiangTTS] Skipping speaker init: {e}", file=sys.stderr)
            return

        # A remote engine owns the reference directory and encodes on first use.
        # Pre-encoding from here would upload every clip we happen to have a local
        # copy of, at startup, to build caches the service already keeps.
        if getattr(synth, "is_remote", False):
            return

        for ref_file in self._ref_files():
            sid = ref_file.stem
            cache_path = self.cache_dir / f"{sid}.pt"
            if cache_path.exists() and cache_path.stat().st_mtime >= ref_file.stat().st_mtime:
                try:
                    self._voices[sid] = synth.load_voice(cache_path)
                    continue
                except Exception as e:
                    print(f"[SiangTTS] Stale cache for '{sid}' ({e}); rebuilding.", file=sys.stderr)
            try:
                cache = synth.build_voice(str(ref_file), self._transcript_for(ref_file))
                synth.save_voice(cache, cache_path)
                self._voices[sid] = cache
            except Exception as ex:
                print(f"[SiangTTS] WARNING: could not cache voice '{sid}': {ex}", file=sys.stderr)

    def _remote(self) -> Any:
        """The synthesizer, when it is a remote one and already connected.

        Deliberately does not connect: the speaker endpoints are also how the studio
        UI loads, and they should not be the thing that raises when the GPU service
        is down.
        """
        return self._synthesizer if getattr(self._synthesizer, "is_remote", False) else None

    def get_speaker_audio_path(self, speaker_id: str) -> Optional[Path]:
        """Locate the reference audio file on disk for a given speaker ID."""
        clean_id = speaker_id.strip()
        search_dirs = [
            self.ref_dir,
            Path(__file__).resolve().parent.parent.parent / "ref",
            Path(__file__).resolve().parent.parent.parent.parent / "voice-cloning" / "ref",
            Path("C:/temp/tts_jobs/voices"),
        ]
        for d in search_dirs:
            if not d.exists():
                continue
            for ext in AUDIO_EXTS:
                cand = d / f"{clean_id}{ext}"
                if cand.is_file():
                    return cand
            for cand in d.iterdir():
                if cand.is_file() and cand.stem.lower() == clean_id.lower() and cand.suffix.lower() in AUDIO_EXTS:
                    return cand
        return None

    def list_speakers(self) -> List[Dict[str, Any]]:
        remote = self._remote()
        if remote is not None:
            return [
                {
                    "id": v["id"],
                    "name": v["id"].replace("_", " ").title(),
                    "filename": v.get("file", ""),
                    "cached": bool(v.get("cached")),
                }
                for v in remote.list_speakers()
            ]
        return [
            {
                "id": f.stem,
                "name": f.stem.replace("_", " ").title(),
                "filename": f.name,
                "cached": (self.cache_dir / f"{f.stem}.pt").exists(),
            }
            for f in self._ref_files()
        ]

    def register_speaker(self, speaker_id: str, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        clean_id = "".join(c for c in speaker_id.strip().lower() if c.isalnum() or c in ("-", "_"))
        if not clean_id:
            clean_id = "custom_speaker"

        ext = Path(filename).suffix.lower()
        if ext not in AUDIO_EXTS:
            ext = ".wav"

        # Register with the engine that will actually be asked to speak in this voice.
        # Writing it into the studio's own ref/ instead would produce a speaker this
        # process can list and the GPU service has never heard of.
        remote = self._remote()
        if remote is not None:
            remote.register_speaker(clean_id, audio_bytes, f"{clean_id}{ext}")
            self._voices.pop(clean_id, None)
            return {
                "id": clean_id,
                "name": clean_id.replace("_", " ").title(),
                "filename": f"{clean_id}{ext}",
                "cached": True,
            }

        ref_path = self.ref_dir / f"{clean_id}{ext}"
        ref_path.write_bytes(audio_bytes)

        cache_path = self.cache_dir / f"{clean_id}.pt"
        try:
            synth = self.get_synthesizer()
            cache = synth.build_voice(str(ref_path), self._transcript_for(ref_path))
            synth.save_voice(cache, cache_path)
            self._voices[clean_id] = cache
        except Exception as e:
            print(f"[SiangTTS] WARNING: failed to cache latent for {clean_id}: {e}", file=sys.stderr)

        return {
            "id": clean_id,
            "name": clean_id.replace("_", " ").title(),
            "filename": ref_path.name,
            "cached": cache_path.exists(),
        }

    def delete_speaker(self, speaker_id: str) -> bool:
        remote = self._remote()
        if remote is not None:
            self._voices.pop(speaker_id, None)
            return remote.delete_speaker(speaker_id)

        found = False
        for f in self.ref_dir.glob(f"{speaker_id}.*"):
            if f.is_file():
                f.unlink(missing_ok=True)
                found = True

        cache_path = self.cache_dir / f"{speaker_id}.pt"
        if cache_path.exists():
            cache_path.unlink(missing_ok=True)

        if speaker_id in self._voices:
            del self._voices[speaker_id]
            found = True

        return found

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    def _resolve_voice(
        self,
        speaker_id: Optional[str],
        ref_audio_bytes: Optional[bytes],
        ref_filename: Optional[str],
        synth: Any = None,
    ) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        """Return (prompt_cache, ref_audio_path, temp_path_to_clean_up)."""
        if ref_audio_bytes:
            ext = Path(ref_filename or "upload.wav").suffix.lower()
            if ext not in AUDIO_EXTS:
                ext = ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(ref_audio_bytes)
                temp_path = tf.name
            return None, temp_path, temp_path

        if speaker_id:
            if speaker_id in self._voices:
                return self._voices[speaker_id], None, None

            # A remote engine keeps its own reference directory, and that is the
            # source of truth for named voices. Ask it to resolve the name rather
            # than uploading our copy of the clip on every request — and so a voice
            # registered through the other pipeline works here too.
            resolver = getattr(synth, "resolve_speaker", None)
            if resolver is not None:
                handle = resolver(speaker_id, allow_sidecar=False)
                if handle:
                    self._voices[speaker_id] = handle
                    return handle, None, None

            for cand in self.ref_dir.glob(f"{speaker_id}.*"):
                if cand.suffix.lower() in AUDIO_EXTS:
                    return None, str(cand), None

        return None, None, None

    def _clone_voice_from_audio(self, synth: Any, wav: Any, sample_rate: int) -> Any:
        """Encode already-generated audio into a timbre-only prompt cache.

        Lets a run of chunks keep one voice when the caller pinned no speaker.
        Returns None if cloning fails -- a drifting voice is better than a failed
        request, and the caller simply carries on unconditioned.

        Deliberately clones in *reference* mode (no transcript). Passing one would
        select VoxCPM2's continuation mode, which reproduces every vocal nuance of
        the source -- including its emotion. That made chunk 2 inherit chunk 1's
        mood, collapsing [sad] -> [happy] to a measured -4.9 Hz median-F0 change
        where a pinned speaker gets +59.5 Hz. Timbre is what we want to carry over;
        prosody must stay free for the next chunk's style tag to steer.
        """
        import soundfile as sf

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
            sf.write(tmp_path, wav, sample_rate, format="WAV", subtype="PCM_16")
            return synth.build_voice(tmp_path)
        except Exception as e:
            print(
                f"[SiangTTS] Could not clone first chunk for voice consistency: {e}",
                file=sys.stderr,
            )
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _voice_from_path(self, synth: Any, path: str) -> Any:
        """Encode a reference clip once so every piece shares the same conditioning.

        Returns None on failure, leaving the caller to fall back to per-piece
        reference passing -- slower, but not a failed request.
        """
        try:
            return synth.build_voice(path)
        except Exception as e:
            print(f"[SiangTTS] Could not pre-encode reference '{path}': {e}", file=sys.stderr)
            return None

    # Filename for the auto seed voice. Leading underscore keeps it out of the
    # speaker listing, which enumerates ref/ rather than the cache.
    SEED_VOICE_FILE = "_auto_seed.pt"

    def _warn_if_instructions_are_dead(
        self, prompt_cache: Any, planned: Sequence[Tuple[int, str, bool]]
    ) -> None:
        """Say so when the voice in hand cannot honour the style tags being sent.

        A prompt cache carrying a transcript puts VoxCPM2 in ultimate-cloning mode,
        where the leading parenthetical is ignored and the prompt clip's own delivery
        wins. The audio still arrives, just flat, so nothing else in the pipeline
        would ever mention it.
        """
        if self._hifi_warned or not isinstance(prompt_cache, dict):
            return
        if prompt_cache.get("mode") not in ("continuation", "ref_continuation"):
            return
        if not any(_LEADING_STYLE_RE.match(text) for _, text, _ in planned):
            return
        self._hifi_warned = True
        print(
            "[SiangTTS] WARNING: this voice was built in ultimate-cloning mode, "
            "which ignores style instructions -- the emotion tags in this request "
            "will not be heard. Rebuild the voice without its transcript, or unset "
            "SIANGTTS_HIFI_CLONING.",
            file=sys.stderr,
        )

    def reset_seed_voice(self) -> bool:
        """Throw away the auto seed voice so the next request mints a new speaker.

        The seed voice is one unseeded generation that then becomes permanent -- it
        is cached in memory and in voice_cache/_auto_seed.pt across restarts, which
        is the point (a stable speaker) right up until the draw is a bad one. Then
        every unpinned request inherits it and re-rendering never escapes, because
        only the speech is re-rolled, not the speaker.

        Returns True if a persisted seed was actually removed. Pinned speakers are
        untouched: they come from ref/, not from this.
        """
        self._seed_voice = None
        self._seed_voice_failed = False

        remote_reset = getattr(self._synthesizer, "reset_seed_voice", None)
        if remote_reset is not None:
            return remote_reset()

        cache_path = self.cache_dir / self.SEED_VOICE_FILE
        try:
            existed = cache_path.exists()
            cache_path.unlink(missing_ok=True)
            return existed
        except Exception as e:
            print(f"[SiangTTS] Could not delete seed voice: {e}", file=sys.stderr)
            return False

    def _build_seed_voice(
        self, synth: Any, sample_rate: int, cfg_value: float, inference_timesteps: int
    ) -> Any:
        """Return the neutral seed voice every unpinned chunk is conditioned on.

        Built once and then reused, from memory within a process and from
        ``voice_cache/_auto_seed.pt`` across restarts. Regenerating it per request
        made every request a slightly different speaker -- two /synthesize calls on
        the same text came back in different voices -- and spent an extra generation
        each time to do it.

        Generated at the service defaults rather than the caller's cfg/timesteps, so
        who the speaker is does not depend on per-request tuning knobs.

        Returns None on failure -- an inconsistent voice beats a failed request.
        """
        if self._seed_voice is not None:
            return self._seed_voice
        if self._seed_voice_failed:
            return None

        # A remote engine mints and keeps the seed voice itself. Let it: that makes
        # the seed shared across every client instead of one per process, and there
        # is no local cache file to go stale against a service that restarted.
        remote_seed = getattr(synth, "seed_voice", None)
        if remote_seed is not None:
            self._seed_voice = remote_seed()
            self._seed_voice_failed = self._seed_voice is None
            return self._seed_voice

        seed_text = (settings.siangtts_voice_seed_text or "").strip()
        if not seed_text:
            return None

        cache_path = self.cache_dir / self.SEED_VOICE_FILE
        if cache_path.exists():
            try:
                self._seed_voice = synth.load_voice(cache_path)
                return self._seed_voice
            except Exception as e:
                print(f"[SiangTTS] Stale seed voice ({e}); rebuilding.", file=sys.stderr)

        try:
            seed_wav = synth.synth(text=seed_text)
            voice = self._clone_voice_from_audio(synth, seed_wav, sample_rate)
        except Exception as e:
            print(f"[SiangTTS] Seed voice generation failed: {e}", file=sys.stderr)
            self._seed_voice_failed = True
            return None

        if voice is None:
            self._seed_voice_failed = True
            return None

        try:
            synth.save_voice(voice, cache_path)
        except Exception as e:
            # Worth keeping in memory even if it could not be persisted.
            print(f"[SiangTTS] Could not persist seed voice: {e}", file=sys.stderr)

        self._seed_voice = voice
        return voice

    def synthesize(
        self,
        text: str,
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        lora_mode: Optional[str] = "on",
    ) -> bytes:
        """Synthesize a single utterance. Returns WAV bytes."""
        return self.synthesize_many(
            [text],
            speaker_id=speaker_id,
            ref_audio_bytes=ref_audio_bytes,
            ref_filename=ref_filename,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            lora_mode=lora_mode,
        )

    def render_chunks(
        self,
        texts: Sequence[str],
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        lora_mode: Optional[str] = "on",
    ) -> Tuple[List[Any], int]:
        """Synthesize each chunk against one voice; return raw audio and sample rate.

        Split out from ``synthesize_many`` so a single generation can be assembled
        more than one way. Sampling is not deterministic, so an A/B that generated
        twice would be comparing two different takes rather than two treatments of
        the same one.
        """
        import numpy as np

        from app.services.audio_post import Chunk

        def _tone_at(idx: int) -> Optional[str]:
            return tones[idx] if tones is not None and idx < len(tones) else None

        def _break_at(idx: int) -> bool:
            return bool(breaks[idx]) if breaks is not None and idx < len(breaks) else False

        # Expand in lockstep so tones/breaks stay aligned with the surviving text.
        # A chunk past the drift budget becomes several pieces that keep its tone
        # and its voice; only the first inherits the chunk's own leading pause.
        planned: List[Tuple[int, str, bool]] = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                continue
            for j, piece in enumerate(split_for_synthesis(t)):
                planned.append((i, piece.text, piece.paragraph_seam if j else _break_at(i)))
        if not planned:
            raise ValueError("No text to synthesize")

        synth = self.get_synthesizer()
        sample_rate = getattr(synth, "sample_rate", MOCK_SAMPLE_RATE)

        # Apply LoRA strength based on lora_mode if loaded
        tts_model = getattr(synth, "tts_model", None)
        if tts_model is not None and getattr(synth, "lora_loaded", False):
            if lora_mode == "off":
                set_lora_strength(tts_model, 0.0, 0.0)
            elif lora_mode == "legacy":
                set_lora_strength(tts_model, 2.0, 2.0)
            else:
                set_lora_strength(
                    tts_model,
                    settings.siangtts_lora_lm_scale,
                    settings.siangtts_lora_dit_scale,
                )

        prompt_cache, ref_audio_path, temp_ref_path = self._resolve_voice(
            speaker_id, ref_audio_bytes, ref_filename, synth
        )

        try:
            # Nothing pinned the voice, so each chunk would otherwise come back a
            # different speaker. Mint one neutral seed voice up front and condition
            # every chunk on it. Seeding from a *neutral* line rather than from
            # chunk 1 is what keeps the style tags independent: cloning chunk 1
            # carried its emotion into chunk 2 and flattened the contrast.
            # The seed is not only about drift *within* one request: an unpinned
            # single-chunk request is a fresh speaker every time, so two calls on
            # the same text came back in different voices. That is what the
            # benchmark does -- one chunk per take -- and it made every take a
            # different person. The seed is minted once and cached, so applying it
            # to single-chunk requests too costs nothing after the first.
            if (
                settings.siangtts_auto_voice_consistency
                and prompt_cache is None
                and ref_audio_path is None
            ):
                prompt_cache = self._build_seed_voice(
                    synth, sample_rate, cfg_value, inference_timesteps
                )

            # Encode the reference once and share it, rather than letting generate()
            # re-encode the same clip for every piece. Identical conditioning either
            # way -- both take VoxCPM2's reference (timbre-only) mode -- but shared
            # so no piece can be conditioned on a slightly different encode.
            if ref_audio_path is not None and len(planned) > 1:
                shared = self._voice_from_path(synth, ref_audio_path)
                if shared is not None:
                    prompt_cache, ref_audio_path = shared, None

            self._warn_if_instructions_are_dead(prompt_cache, planned)

            # A remote engine takes the whole run as one job: the chunks have to be
            # generated against one prompt cache or the speaker drifts between them,
            # and sending them together is what lets the service guarantee that — for
            # one round trip and one place in the queue instead of N of each.
            batch = getattr(synth, "render_batch", None)
            if batch is not None:
                audios, sample_rate = batch(
                    [text for _, text, _ in planned],
                    speaker_id=speaker_id,
                    prompt_cache=prompt_cache,
                    ref_audio=ref_audio_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    lora_mode=lora_mode,
                )
                if len(audios) != len(planned):
                    raise RuntimeError(
                        f"engine returned {len(audios)} chunks for {len(planned)} sent"
                    )
                return [
                    Chunk(
                        audio=np.asarray(audio, dtype="float32"),
                        tone=_tone_at(src_idx),
                        break_before=break_before,
                        text_len=spoken_len(chunk),
                    )
                    for (src_idx, chunk, break_before), audio in zip(planned, audios)
                ], sample_rate

            rendered: List[Chunk] = []
            for src_idx, chunk, break_before in planned:
                wav = np.asarray(
                    synth.synth(
                        text=chunk,
                        ref_audio=ref_audio_path,
                        prompt_cache=prompt_cache,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                        speaker_id=speaker_id,
                        lora_mode=lora_mode,
                    ),
                    dtype="float32",
                )
                rendered.append(
                    Chunk(
                        audio=wav,
                        tone=_tone_at(src_idx),
                        break_before=break_before,
                        text_len=spoken_len(chunk),
                    )
                )

            return rendered, sample_rate
        finally:
            if temp_ref_path and os.path.exists(temp_ref_path):
                try:
                    os.remove(temp_ref_path)
                except Exception:
                    pass

    def synthesize_many(
        self,
        texts: Sequence[str],
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        post_process: bool = True,
        post_process_params: Optional[dict] = None,
        lora_mode: Optional[str] = "on",
    ) -> bytes:
        """Synthesize several chunks against one voice and join them into one take.

        Each chunk carries its own leading style instruction, which is how per-segment
        emotion is expressed -- VoxCPM2 only honours a style parenthetical at the very
        start of the text it is given.

        ``tones`` and ``breaks`` run parallel to ``texts`` and tell the assembler how
        loud each chunk should sit and how much silence belongs in front of it. Without
        them the chunks are still trimmed, faded and levelled, just without the
        per-emotion offsets. ``post_process=False`` returns the bare concatenation,
        which is what tools/ab_gen.py renders as the "before" take.

        ``post_process_params`` overrides individual audio_post constants for this
        take only; keys it omits keep their measured defaults.
        """
        import soundfile as sf

        from app.services.audio_post import PostProcessConfig, assemble, butt_join

        rendered, sample_rate = self.render_chunks(
            texts,
            speaker_id=speaker_id,
            ref_audio_bytes=ref_audio_bytes,
            ref_filename=ref_filename,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            tones=tones,
            breaks=breaks,
            lora_mode=lora_mode,
        )

        if post_process:
            config = PostProcessConfig.from_dict(post_process_params)
            audio = assemble(rendered, sample_rate, config=config)
        else:
            audio = butt_join(rendered, sample_rate)

        out_buf = io.BytesIO()
        sf.write(out_buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return out_buf.getvalue()

    def synthesize_variants(
        self,
        texts: Sequence[str],
        *,
        variants: Sequence[dict],
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        lora_mode: Optional[str] = "on",
    ) -> Tuple[List[dict], int, List[Optional[str]]]:
        """Render one generation and assemble it every way ``variants`` asks for.

        This is the honest A/B: ``render_chunks`` runs once, so the only thing that
        differs between the returned takes is the assembly. Each variant is a dict
        of ``{"post_process": bool, "params": dict | None}``.

        Returns the takes (WAV bytes plus per-chunk placement), the sample rate, and
        the tone of each rendered chunk -- which is not the same list as ``tones``,
        because a long chunk is split for synthesis and each piece keeps its tone.
        """
        import io as _io

        import numpy as np
        import soundfile as sf

        from app.services.audio_post import (
            PostProcessConfig, assemble_with_spans, butt_join_with_spans, voiced_rms,
        )

        rendered, sample_rate = self.render_chunks(
            texts,
            speaker_id=speaker_id,
            ref_audio_bytes=ref_audio_bytes,
            ref_filename=ref_filename,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            tones=tones,
            breaks=breaks,
            lora_mode=lora_mode,
        )

        # The assemblers drop empty chunks before measuring, so mirror that filter
        # here to keep tones aligned with the spans they hand back.
        usable = [c for c in rendered if c.audio is not None and np.asarray(c.audio).size]
        chunk_tones: List[Optional[str]] = [c.tone for c in usable]

        takes: List[dict] = []
        for spec in variants:
            if spec.get("post_process", True):
                config = PostProcessConfig.from_dict(spec.get("params"))
                audio, spans = assemble_with_spans(rendered, sample_rate, config=config)
            else:
                audio, spans = butt_join_with_spans(rendered, sample_rate)

            chunk_stats = []
            for i, (start, end) in enumerate(spans):
                seg = audio[int(start * sample_rate):int(end * sample_rate)]
                level = voiced_rms(seg, sample_rate)
                # Pace has to be per character, or a long chunk reads as a slow one.
                text_len = usable[i].text_len if i < len(usable) else 0
                dur = float(end - start)
                chunk_stats.append({
                    "tone": chunk_tones[i] if i < len(chunk_tones) else None,
                    "start_s": round(float(start), 3),
                    "end_s": round(float(end), 3),
                    "dur_s": round(dur, 3),
                    "text_len": int(text_len),
                    "pace_s_per_char": round(dur / text_len, 5) if text_len else None,
                    "level_db": (
                        round(float(20 * math.log10(level)), 2) if level > 1e-6 else None
                    ),
                })

            buf = _io.BytesIO()
            sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
            takes.append({
                "id": spec.get("id"),
                "wav": buf.getvalue(),
                "dur_s": round(len(audio) / sample_rate, 3),
                "chunks": chunk_stats,
            })

        return takes, sample_rate, chunk_tones


siangtts_service = SiangTTSService()
