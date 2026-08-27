"""Thonburian F5 (emotion transfer) + SeedVC (timbre conversion) Voice Generation Service.

Replaces the VoxCPM2 generation backend in the Tone Studio (:8011/:8012).
Pipeline:
  1. Per chunk: map tone -> thai-ser emotion {neutral, angry, happy, sad, frustrated}
     (unsupported emotions raise ValueError with the supported list).
  2. Strip leading style parenthetical; normalize Thai text and apply pronunciation overrides.
  3. Load thai-ser emotion donor clip (<emotion>_1.wav + transcript .txt), respecting gender.
  4. Thonburian F5 generates emotional Thai speech (24 kHz).
  5. SeedVC (f0_condition=True, auto_f0_adjust=True) swaps timbre to target speaker (44.1 kHz).
  6. Assemble chunks with audio_post at 44.1 kHz.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np
import soundfile as sf

from app.config import settings
from app.services.audio_post import (
    Chunk,
    PostProcessConfig,
    assemble,
    assemble_with_spans,
    butt_join,
    butt_join_with_spans,
    fade_edges,
    trim_silence,
    voiced_rms,
)
from app.services.pronunciation import apply_pronunciation
from app.services.thai_normalizer import normalize_thai_text
from app.services.thai_text import expand

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
SUPPORTED_EMOTIONS = {"neutral", "angry", "happy", "sad", "frustrated"}

_LEADING_STYLE_RE = re.compile(r"^\s*\([^)]*\)\s*")


def spoken_len(text: str) -> int:
    """Characters of text that will actually be voiced."""
    return len(_LEADING_STYLE_RE.sub("", text or "").strip())


class ThonburianServiceUnavailable(RuntimeError):
    """Raised when Thonburian F5 or SeedVC is unreachable or fails to initialize."""


class ThonburianService:
    def __init__(
        self,
        ref_dir: str | Path | None = None,
        donor_dir: str | Path | None = None,
        flowtts_src: str | None = None,
        seedvc_url: str | None = None,
        device: str | None = None,
    ):
        self.ref_dir = Path(ref_dir or settings.siangtts_ref_dir)
        self.donor_dir = Path(donor_dir or settings.emotion_donor_dir)
        self.flowtts_src = flowtts_src or settings.flowtts_src
        self.seedvc_url = (seedvc_url or settings.seedvc_url).rstrip("/")
        self.device = device or settings.thonburian_device or None

        self.ref_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None
        self._is_loaded = False
        self._load_error = None
        self.sample_rate = 44100  # SeedVC f0-conditioned output rate

    # ------------------------------------------------------------------ #
    # Pipeline loading
    # ------------------------------------------------------------------ #

    def _ensure_flowtts_path(self) -> None:
        src = (self.flowtts_src or "").strip()
        if src and Path(src).exists() and str(Path(src).resolve()) not in sys.path:
            sys.path.insert(0, str(Path(src).resolve()))

    def get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        self._ensure_flowtts_path()

        try:
            import torch
            from flowtts.inference import AudioConfig, FlowTTSPipeline, ModelConfig

            dev = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = "hf://biodatlab/ThonburianTTS/megaF5/mega_f5_last.safetensors"
            vocab = "hf://biodatlab/ThonburianTTS/megaF5/mega_vocab.txt"

            print(f"[Thonburian] Loading Thonburian F5 on {dev} ...", file=sys.stderr)
            t0 = time.time()

            temp_dir = Path("scratch/thonburian_temp")
            temp_dir.mkdir(parents=True, exist_ok=True)

            self._pipeline = FlowTTSPipeline(
                model_config=ModelConfig(
                    language="th",
                    model_type="F5",
                    vocoder="vocos",
                    checkpoint=checkpoint,
                    vocab_file=vocab,
                    device=dev,
                ),
                audio_config=AudioConfig(
                    cfg_strength=settings.thonburian_cfg_strength,
                    nfe_step=settings.thonburian_nfe_step,
                    sway_sampling_coef=settings.thonburian_sway_sampling_coef,
                    target_rms=settings.thonburian_target_rms,
                    keep_silence=settings.thonburian_keep_silence,
                    min_silence_len=settings.thonburian_min_silence_len,
                ),
                temp_dir=str(temp_dir),
            )
            self._is_loaded = True
            self._load_error = None
            print(f"[Thonburian] Ready in {time.time()-t0:.1f}s on {dev}", file=sys.stderr)
            return self._pipeline
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            print(f"[Thonburian] Failed to load pipeline: {self._load_error}", file=sys.stderr)
            raise ThonburianServiceUnavailable(f"Thonburian F5 failed to load: {self._load_error}") from e

    # ------------------------------------------------------------------ #
    # F5 acoustic tuning (per-generation, no model reload)
    # ------------------------------------------------------------------ #

    # Adjustable F5 knobs and their sane bounds. The pipeline reads
    # ``self._pipeline.audio_config`` on every call, so writing these before a
    # generation takes effect immediately -- no reconstruction, no reload.
    TUNING_SPECS: Dict[str, Dict[str, Any]] = {
        "cfg_strength":       {"min": 1.0,  "max": 6.0,   "step": 0.1},
        "nfe_step":           {"min": 8,    "max": 64,    "step": 1},
        "sway_sampling_coef": {"min": -1.5, "max": 0.0,   "step": 0.1},
        "target_rms":         {"min": 0.0,  "max": 0.5,   "step": 0.01},
        "keep_silence":       {"min": 0,    "max": 1000,  "step": 50},
        "min_silence_len":    {"min": 200,  "max": 1500,  "step": 50},
    }

    def tuning_defaults(self) -> Dict[str, Any]:
        """Server default for every knob, plus its bounds, for the UI table."""
        vals = {
            "cfg_strength": settings.thonburian_cfg_strength,
            "nfe_step": settings.thonburian_nfe_step,
            "sway_sampling_coef": settings.thonburian_sway_sampling_coef,
            "target_rms": settings.thonburian_target_rms,
            "keep_silence": settings.thonburian_keep_silence,
            "min_silence_len": settings.thonburian_min_silence_len,
            "speed": settings.thonburian_speed,
        }
        return {"values": vals, "specs": self.TUNING_SPECS}

    def _apply_audio_tuning(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write the acoustic knobs onto the live AudioConfig for the next call.

        Every field is set explicitly from ``overrides`` (clamped to its bounds)
        or the server default, so one generation never inherits the previous
        request's values. Returns the effective values for logging/inspection.
        """
        if self._pipeline is None:
            return {}
        overrides = overrides or {}
        defaults = self.tuning_defaults()["values"]
        cfg = self._pipeline.audio_config
        effective: Dict[str, Any] = {}
        for key, spec in self.TUNING_SPECS.items():
            raw = overrides.get(key)
            val = defaults[key] if raw is None else raw
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = float(defaults[key])
            val = max(spec["min"], min(spec["max"], val))
            if key in ("nfe_step", "keep_silence", "min_silence_len"):
                val = int(round(val))
            setattr(cfg, key, val)
            effective[key] = val
        return effective

    # ------------------------------------------------------------------ #
    # Speaker & Emotion donor resolution
    # ------------------------------------------------------------------ #

    def _ref_files(self) -> List[Path]:
        return [p for p in sorted(self.ref_dir.iterdir()) if p.suffix.lower() in AUDIO_EXTS]

    def list_speakers(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": f.stem,
                "name": f.stem.replace("_", " ").title(),
                "filename": f.name,
                "cached": True,
            }
            for f in self._ref_files()
        ]

    def get_speaker_audio_path(self, speaker_id: str) -> Optional[Path]:
        clean_id = speaker_id.strip()
        search_dirs = [
            self.ref_dir,
            Path(__file__).resolve().parent.parent.parent / "ref",
            Path(__file__).resolve().parent.parent.parent.parent / "voice-cloning" / "ref",
            Path("dataset/male"),
            Path("dataset/female"),
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

    def _resolve_default_speaker(self, gender: Optional[str] = None) -> Path:
        """Find a fallback speaker reference WAV on disk."""
        g = (gender or settings.default_gender or "female").strip().lower()
        if g.startswith("m"):
            # Check for male reference files
            for cand in [
                self.ref_dir / "male.wav",
                self.ref_dir / "determination.wav",
                Path("dataset/male/neutral_0001.wav"),
                self.donor_dir / "male" / "neutral_1.wav",
            ]:
                if cand.exists():
                    return cand
        else:
            # Check for female reference files
            for cand in [
                self.ref_dir / "female.wav",
                self.ref_dir / "lion.mp3",
                self.ref_dir / "customer.MP3",
                Path("dataset/female/neutral_0001.wav"),
                self.donor_dir / "female" / "neutral_1.wav",
            ]:
                if cand.exists():
                    return cand

        # Any available speaker file in ref/
        files = self._ref_files()
        if files:
            return files[0]

        # Check donor neutral
        for cand in [
            self.donor_dir / "neutral_1.wav",
            Path("../voice-cloning/ref/emotions/neutral_1.wav"),
        ]:
            if cand.exists():
                return cand

        raise FileNotFoundError("No reference speaker audio found in ref/ or donor directories")

    def _resolve_donor_clip(
        self, emotion: str, gender: Optional[str] = None, donor_set: Optional[str] = None
    ) -> Tuple[Path, str]:
        """Resolve the thai-ser emotion donor clip and its sidecar transcript.

        When ``donor_set`` names an existing same-person set folder
        (``ref/emotions/<set>/<emotion>_1.wav``), it is preferred so every emotion comes from
        the same speaker. Otherwise falls back to the per-gender donor folder.
        """
        g = (gender or settings.default_gender or "female").strip().lower()
        g_name = "male" if g.startswith("m") else "female"

        candidates: List[Path] = []
        if donor_set:
            candidates.append(self.donor_dir / donor_set / f"{emotion}_1.wav")
        candidates += [
            self.donor_dir / g_name / f"{emotion}_1.wav",
            Path("dataset") / g_name / f"{emotion}_0001.wav",
            self.donor_dir / f"{emotion}_1.wav",
            Path("../voice-cloning/ref/emotions") / f"{emotion}_1.wav",
            Path("../voice-cloning/ref/emotions") / g_name / f"{emotion}_1.wav",
        ]

        for wav in candidates:
            if wav.exists():
                txt = wav.with_suffix(".txt")
                if txt.exists():
                    transcript = txt.read_text(encoding="utf-8").strip()
                    if transcript:
                        return wav, transcript

        raise FileNotFoundError(
            f"No donor reference audio/transcript found for emotion '{emotion}' (gender={g_name}). "
            f"Expected {emotion}_1.wav + {emotion}_1.txt in {self.donor_dir}"
        )

    # ------------------------------------------------------------------ #
    # Emotion validation and text prep
    # ------------------------------------------------------------------ #

    def _validate_and_map_emotion(self, raw_tone: Optional[str]) -> str:
        if raw_tone is None or not str(raw_tone).strip():
            return "neutral"
        clean = str(raw_tone).strip().lower()
        if clean not in SUPPORTED_EMOTIONS:
            raise ValueError(
                f"Unsupported emotion '{raw_tone}'. Supported emotions are: "
                f"{', '.join(sorted(SUPPORTED_EMOTIONS))}"
            )
        return clean

    def _prepare_text(self, text: str) -> str:
        """Strip style instruction parenthetical, then make the text speakable for F5.

        Pipeline: encoding hygiene (``normalize_thai_text``) -> speakable-Thai rewrite
        (``expand``: numbers/abbreviations/ๆ/เทอ so Thai is read correctly).

        The pronunciation dictionary is deliberately NOT applied here: an override may
        respell a word as spaced syllables ("เทคโนโลยี" -> "เทค โน โล ยี"), and the
        long-text splitter cuts on whitespace, so applying it this early would let a
        cut land between a word's own syllables. ``_render_f5`` applies it to each
        piece after the text has been split, which cannot expose that seam.
        """
        m = _LEADING_STYLE_RE.match(text or "")
        body = text[m.end():] if m else (text or "")
        clean_body = body.strip()
        if not clean_body:
            return ""
        normalized = normalize_thai_text(clean_body)
        return expand(normalized, is_thai=True)

    # ------------------------------------------------------------------ #
    # SeedVC Voice Conversion
    # ------------------------------------------------------------------ #

    def _convert_seedvc(
        self,
        source_wav: Path,
        target_wav: Path,
        output_wav: Path,
        *,
        auto_f0_adjust: Optional[bool] = None,
        semi_tone_shift: int = 0,
    ) -> Path:
        """Send voice conversion request to SeedVC worker (:8022).

        ``auto_f0_adjust`` / ``semi_tone_shift`` default to the server config, but the
        F0-compare experiment overrides them per call so one F5 generation can be
        converted several ways (see ``render_f0_compare``).
        """
        import requests

        af0 = settings.seedvc_auto_f0_adjust if auto_f0_adjust is None else bool(auto_f0_adjust)
        url = f"{self.seedvc_url}/convert"
        payload = {
            "source": str(source_wav.resolve()),
            "target": str(target_wav.resolve()),
            "output": str(output_wav.resolve()),
            "f0_condition": settings.seedvc_f0_condition,
            "auto_f0_adjust": af0,
            "diffusion_steps": settings.seedvc_diffusion_steps,
            "semi_tone_shift": int(semi_tone_shift),
            "inference_cfg_rate": 0.7,
        }
        try:
            res = requests.post(url, json=payload, timeout=settings.seedvc_timeout)
            if not res.ok:
                err_msg = ""
                try:
                    data = res.json()
                    err_msg = data.get("error") or data.get("detail") or str(data)
                except Exception:
                    err_msg = res.text
                raise RuntimeError(f"SeedVC worker returned error ({res.status_code}): {err_msg}")
            return output_wav
        except requests.exceptions.RequestException as e:
            raise ThonburianServiceUnavailable(
                f"SeedVC worker at {self.seedvc_url} is unreachable: {e}. "
                f"Ensure the SeedVC server is running on port 8022."
            ) from e

    @staticmethod
    def _estimate_speech_rate(wav_path: Path) -> Optional[float]:
        """Estimate speaking rate as syllable nuclei per voiced second (transcript-free).

        A crude energy-envelope peak picker (after De Jong & Wempe's syllable-nuclei
        idea): count local RMS-envelope maxima above a floor, over the voiced duration.
        Absolute values are rough, but the SAME method is applied to donor and target,
        so the *ratio* used for speed matching cancels the estimator's systematic bias.
        Returns None when the clip is unreadable, silent, or too short to measure.
        """
        try:
            y, sr = sf.read(str(wav_path), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if y.size == 0 or sr <= 0:
                return None
            frame = max(1, int(0.02 * sr))
            hop = max(1, int(0.01 * sr))
            if len(y) < frame:
                return None
            n = 1 + (len(y) - frame) // hop
            if n <= 2:
                return None
            idx = np.arange(n) * hop
            env = np.sqrt(np.array([
                float(np.mean(y[i:i + frame] ** 2)) for i in idx
            ]))
            peak = float(env.max())
            if peak <= 0:
                return None
            thr = peak * (10.0 ** (-35.0 / 20.0))  # -35 dB below the loudest frame
            voiced_dur = float(np.count_nonzero(env > thr)) * hop / sr
            if voiced_dur < 0.2:
                return None
            min_dist = max(1, int(0.15 * sr / hop))  # nuclei at least ~150 ms apart
            nuclei = 0
            last = -min_dist
            for i in range(1, n - 1):
                if env[i] > thr and env[i] >= env[i - 1] and env[i] > env[i + 1] and (i - last) >= min_dist:
                    nuclei += 1
                    last = i
            if nuclei <= 0:
                return None
            return nuclei / voiced_dur
        except Exception:
            return None

    def _matched_speed(self, target_wav: Path, donor_wav: Path) -> Optional[float]:
        """Speed multiplier that makes F5 speak the donor's emotion at the *target's* pace.

        F5's output rate tracks its reference clip's rate, so output_rate ≈ donor_rate ×
        speed. Setting speed = target_rate / donor_rate makes output_rate ≈ target_rate.
        Returns None when either rate can't be measured (caller falls back to the default).
        """
        target_rate = self._estimate_speech_rate(target_wav)
        donor_rate = self._estimate_speech_rate(donor_wav)
        if not target_rate or not donor_rate:
            return None
        speed = target_rate / donor_rate
        lo = float(settings.thonburian_rate_speed_min)
        hi = float(settings.thonburian_rate_speed_max)
        return max(lo, min(hi, speed))

    def _depeak_f5(self, wav_path: Path) -> None:
        """Peak-limit an F5 output in place so it stops clipping downstream.

        F5's vocos vocoder overshoots [-1, 1] on loud material, and the flowtts
        library saves the result as PCM_16 with no ceiling, so those peaks hard-clip
        on write -- audible distortion on the louder emotions. This scales any file
        whose peak exceeds ``thonburian_f5_peak`` back down to it and rewrites as
        float32, so neither SeedVC nor the pre-VC player is handed a clipped source.
        Samples already destroyed by the library's PCM_16 write cannot be recovered;
        this keeps the next generation off the ceiling rather than un-clipping this one.
        """
        ceiling = float(settings.thonburian_f5_peak)
        if ceiling <= 0:
            return
        try:
            arr, sr = sf.read(str(wav_path), dtype="float32")
            peak = float(np.max(np.abs(arr))) if arr.size else 0.0
            if peak > ceiling:
                arr = (arr * (ceiling / peak)).astype("float32")
                sf.write(str(wav_path), arr, sr, subtype="FLOAT")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # F5 single-batch guard (stop mid-utterance pauses / skipped words)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _f5_max_bytes(donor_wav: Path, donor_txt: str, speed: float = 1.0) -> Optional[int]:
        """UTF-8 byte budget for one F5 call, sized by how long it will *speak*.

        Two limits apply and the tighter one wins:

        * flowtts ``infer_process`` splits gen_text into batches once it exceeds
          ``ref_bytes / ref_dur * (22 - ref_dur)``, then cross-fades the batches back
          together -- which strands a pause at every seam and swallows syllables in the
          overlap. Staying under it keeps every call a single batch.
        * That ceiling is only where F5 *breaks*. Quality falls off well before it:
          the model repeats and invents words as the sequence grows, and at 560 bytes
          off a 3.7 s donor it was being asked for a 17 s stretch in one shot. So the
          real budget is ``thonburian_f5_max_gen_s`` of speech, converted to bytes via
          the donor's own pace (``ref_bytes / ref_dur``) and the requested speed --
          the same relation F5 uses internally to reserve duration.

        Returns None when the donor is unreadable, so the caller leaves text alone.
        """
        try:
            info = sf.info(str(donor_wav))
            dur = info.frames / info.samplerate if info.samplerate else 0.0
        except Exception:
            return None
        if dur <= 0 or dur >= 22:
            return None
        ref_bytes = len((donor_txt or "").encode("utf-8"))
        if ref_bytes <= 0:
            return None

        pace = ref_bytes / dur                      # donor bytes per second of speech
        batch_ceiling = int(pace * (22 - dur))      # flowtts single-batch limit
        spd = float(speed) if speed and speed > 0 else 1.0
        quality_cap = int(pace * float(settings.thonburian_f5_max_gen_s) * spd)
        return max(1, min(batch_ceiling, quality_cap))

    @staticmethod
    def _pack_units(units: Sequence[str], limit: int, joiner: str) -> List[str]:
        """Greedily pack ``units`` into pieces of at most ``limit`` UTF-8 bytes."""
        pieces: List[str] = []
        cur = ""
        for u in units:
            cand = f"{cur}{joiner}{u}" if cur else u
            if cur and len(cand.encode("utf-8")) > limit:
                pieces.append(cur)
                cur = u
            else:
                cur = cand
        if cur:
            pieces.append(cur)
        return pieces

    def _split_for_f5(
        self, body_text: str, donor_wav: Path, donor_txt: str,
        *, speed: float = 1.0, min_bytes: int = 40,
    ) -> List[str]:
        """Split ``body_text`` into pieces F5 can speak well in a single shot.

        Returns ``[body_text]`` unchanged when it already fits the budget (the common
        case -- most tone runs are short), so ordinary utterances are byte-for-byte
        untouched and behave exactly as before.

        Longer text is cut at the most natural boundary that works, in order:
        the writer's own spaces (which in Thai mark clause/sentence ends and already
        carry a pause), and only for a single span still too long, newmm word
        boundaries -- so a word is never cut in half. A too-small trailing piece is
        folded back into its neighbour, both because a stray fragment sounds clipped
        and because under 10 bytes F5 forces ``speed=0.3`` on it.
        """
        text = (body_text or "").strip()
        if not text:
            return []
        limit = self._f5_max_bytes(donor_wav, donor_txt, speed=speed)
        if not limit:
            return [text]
        limit = max(min_bytes, limit)
        if len(text.encode("utf-8")) <= limit:
            return [text]

        # 1. Prefer the writer's own boundaries.
        spans = [s for s in re.split(r"[ \t\r\n]+", text) if s]
        pieces: List[str] = []
        for piece in self._pack_units(spans, limit, " "):
            if len(piece.encode("utf-8")) <= limit:
                pieces.append(piece)
                continue
            # 2. One span alone is over budget -- fall back to word boundaries.
            try:
                from pythainlp.tokenize import word_tokenize
                words = word_tokenize(piece, engine="newmm")
            except Exception:
                pieces.append(piece)
                continue
            pieces.extend(self._pack_units(words, limit, ""))

        cleaned = [p.strip() for p in pieces if p.strip()]
        if not cleaned:
            return [text]
        if len(cleaned) >= 2 and len(cleaned[-1].encode("utf-8")) < min_bytes:
            cleaned[-2] = f"{cleaned[-2]} {cleaned[-1]}".strip()
            cleaned.pop()
        return cleaned

    def _render_f5(
        self,
        pipeline: Any,
        *,
        text: str,
        donor_wav: Path,
        donor_txt: str,
        out_path: Path,
        speed: float,
        scratch_dir: Path,
        tag: str,
    ) -> None:
        """Generate one F5 clip for ``text`` into ``out_path``, single-batch guaranteed.

        Short text is one plain F5 call (unchanged behaviour). Long text is split into
        F5-single-batch pieces (see ``_split_for_f5``), each generated on its own, then
        each piece's F5-padded trailing/leading silence is trimmed *before* joining --
        so the over-allocated tail that used to surface as a mid-sentence pause is gone,
        and the join is a short controlled fade we own instead of F5's blind cross-fade.
        """
        # Split first, then apply pronunciation overrides to each piece -- see
        # _prepare_text for why the order matters.
        pieces = [apply_pronunciation(p) for p in
                  self._split_for_f5(text, donor_wav, donor_txt, speed=speed)]
        if len(pieces) <= 1:
            pipeline(
                text=pieces[0] if pieces else text,
                ref_voice=str(donor_wav.resolve()),
                ref_text=donor_txt,
                output_file=str(out_path.resolve()),
                speed=speed,
            )
            return

        rendered: List[Tuple[np.ndarray, int]] = []
        for pj, piece in enumerate(pieces):
            ptmp = scratch_dir / f"{tag}_p{pj}.wav"
            try:
                pipeline(
                    text=piece,
                    ref_voice=str(donor_wav.resolve()),
                    ref_text=donor_txt,
                    output_file=str(ptmp.resolve()),
                    speed=speed,
                )
                self._depeak_f5(ptmp)
                a, psr = sf.read(str(ptmp), dtype="float32")
                if a.ndim > 1:
                    a = a.mean(axis=1)
                a = trim_silence(a, psr)
                if a.size:
                    a = fade_edges(a, psr)
                    rendered.append((a, int(psr)))
            finally:
                ptmp.unlink(missing_ok=True)

        if not rendered:
            raise ValueError("F5 produced no audio for any sub-piece")

        psr = rendered[0][1]
        # A short inter-piece gap keeps the seam from sounding glued while staying well
        # under a real pause; edges are already faded so it never clicks.
        gap = np.zeros(int(0.04 * psr), dtype=np.float32)
        merged: List[np.ndarray] = []
        for pj, (a, _ps) in enumerate(rendered):
            if pj:
                merged.append(gap)
            merged.append(a)
        out = np.concatenate(merged).astype("float32")
        sf.write(str(out_path), out, psr, subtype="FLOAT")

    @staticmethod
    def _median_f0(wav_path: Path) -> Optional[float]:
        """Median voiced F0 (Hz) of a clip via autocorrelation, transcript-free.

        Used to anchor the F0-compare "B" mode: the constant semitone shift that
        moves the donor's *neutral* register onto the target speaker's register is
        derived from these two medians, so emotion offsets ride on top unchanged.
        Returns None when nothing voiced is measurable.
        """
        try:
            y, sr = sf.read(str(wav_path), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if y.size == 0 or sr <= 0:
                return None
            frame = int(sr * 0.030)
            hop = int(sr * 0.010)
            if len(y) < frame:
                return None
            rms = float(np.sqrt(np.mean(y.astype("float64") ** 2)))
            min_lag = max(1, int(sr / 450))
            max_lag = int(sr / 60)
            cands: List[float] = []
            for i in range(0, len(y) - frame, hop):
                fr = y[i:i + frame]
                if np.sqrt(np.mean(fr ** 2)) < max(rms * 0.2, 1e-4):
                    continue
                w = fr * np.hanning(len(fr))
                corr = np.correlate(w, w, mode="full")[len(w) - 1:]
                if len(corr) <= max_lag or corr[0] <= 0:
                    continue
                region = corr[min_lag:max_lag]
                if region.size == 0:
                    continue
                peak = int(np.argmax(region)) + min_lag
                if corr[peak] > 0.3 * corr[0]:
                    cands.append(float(sr / peak))
            if len(cands) < 3:
                return None
            return float(np.median(cands))
        except Exception:
            return None

    # F0-compare modes. Each converts the SAME F5 output a different way so the
    # emotion-vs-register trade-off can be judged by ear:
    #   baseline = current behaviour (SeedVC re-centres every emotion's pitch to the
    #              target -> flat register, emotion pitch lost).
    #   A        = keep F5's absolute pitch (emotion survives) but it sits in the
    #              donor's register, not the target's.
    #   B        = keep F5's pitch contour, shifted by a constant so the donor's
    #              neutral lands on the target's register -> emotion AND register.
    F0_COMPARE_MODES: List[Dict[str, Any]] = [
        {"id": "baseline", "label": "ปัจจุบัน (auto-f0)", "auto_f0_adjust": True,  "shift": "none"},
        {"id": "A", "label": "A · คงอารมณ์", "auto_f0_adjust": False, "shift": "none"},
        {"id": "B", "label": "B · คงอารมณ์+ตรง target", "auto_f0_adjust": False, "shift": "to_target"},
    ]

    def render_f0_compare(
        self,
        text: str,
        *,
        emotion: str,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        gender: Optional[str] = None,
        donor_set: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate one utterance once with F5, then voice-convert it three ways.

        Returns ``{"modes": [{id,label,wav(bytes),sr,auto_f0_adjust,semi_tone_shift}],
        "pre_vc": {wav,sr}, "diag": {...}}``. The heavy F5 step runs a single time; the
        three modes differ only in the SeedVC F0 handling, so this is the cheapest way
        to A/B the register-vs-emotion trade-off. Single-utterance (benchmark) scope.
        """
        pipeline = self.get_pipeline()
        self._apply_audio_tuning(None)

        emotion = self._validate_and_map_emotion(emotion)
        body_text = self._prepare_text(text)
        if not body_text:
            raise ValueError("No text to synthesize")

        donor_wav, donor_txt = self._resolve_donor_clip(emotion, gender=gender, donor_set=donor_set)

        # Resolve the target speaker (voice to clone).
        temp_target_path = None
        if ref_audio_bytes:
            ext = Path(ref_filename or "upload.wav").suffix.lower()
            if ext not in AUDIO_EXTS:
                ext = ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(ref_audio_bytes)
                temp_target_path = Path(tf.name)
            target_wav = temp_target_path
        elif speaker_id:
            spk_path = self.get_speaker_audio_path(speaker_id)
            if not spk_path or not spk_path.exists():
                raise FileNotFoundError(f"Speaker '{speaker_id}' reference audio not found")
            target_wav = spk_path
        else:
            target_wav = self._resolve_default_speaker(gender=gender)

        # Speed: explicit wins; else rate-match to the target; else default.
        if speed is not None:
            chunk_speed = float(speed)
        elif settings.thonburian_match_target_rate and (ref_audio_bytes or speaker_id):
            rm = self._matched_speed(Path(target_wav), donor_wav)
            chunk_speed = rm if rm is not None else settings.thonburian_speed
        else:
            chunk_speed = settings.thonburian_speed

        # B's constant shift: move the donor's NEUTRAL register onto the target's.
        # Using neutral (not this emotion) as the anchor is what keeps the emotion's
        # own pitch offset intact after the shift.
        semi_shift_b = 0
        target_med = self._median_f0(Path(target_wav))
        donor_neutral_med = None
        try:
            neutral_wav, _ = self._resolve_donor_clip("neutral", gender=gender, donor_set=donor_set)
            donor_neutral_med = self._median_f0(neutral_wav)
        except Exception:
            donor_neutral_med = None
        if target_med and donor_neutral_med and donor_neutral_med > 0:
            semi_shift_b = int(round(12.0 * math.log2(target_med / donor_neutral_med)))
            semi_shift_b = max(-12, min(12, semi_shift_b))

        scratch_dir = Path("scratch/thonburian_gen")
        scratch_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        thon_out = scratch_dir / f"f0cmp_{emotion}_{ts}.wav"

        modes_out: List[Dict[str, Any]] = []
        pre_vc: Dict[str, Any] = {}
        try:
            # 1. F5 once (single-batch guarded so long text keeps its rhythm).
            self._render_f5(
                pipeline,
                text=body_text,
                donor_wav=donor_wav,
                donor_txt=donor_txt,
                out_path=thon_out,
                speed=chunk_speed,
                scratch_dir=scratch_dir,
                tag=f"f0cmp_{emotion}_{ts}",
            )
            self._depeak_f5(thon_out)

            if settings.thonburian_free_cache:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass

            # Keep the pre-VC F5 clip for playback.
            try:
                pre_arr, pre_sr = sf.read(str(thon_out), dtype="float32")
                if pre_arr.ndim > 1:
                    pre_arr = pre_arr.mean(axis=1)
                buf = io.BytesIO()
                sf.write(buf, pre_arr, int(pre_sr), format="WAV", subtype="PCM_16")
                pre_vc = {"wav": buf.getvalue(), "sr": int(pre_sr)}
            except Exception:
                pre_vc = {}

            # 2. SeedVC three ways off the single F5 output.
            for spec in self.F0_COMPARE_MODES:
                shift = semi_shift_b if spec["shift"] == "to_target" else 0
                vc_out = scratch_dir / f"f0cmp_{emotion}_{spec['id']}_{ts}.wav"
                self._convert_seedvc(
                    source_wav=thon_out,
                    target_wav=Path(target_wav),
                    output_wav=vc_out,
                    auto_f0_adjust=spec["auto_f0_adjust"],
                    semi_tone_shift=shift,
                )
                arr, sr = sf.read(str(vc_out), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                obuf = io.BytesIO()
                sf.write(obuf, arr, sr, format="WAV", subtype="PCM_16")
                modes_out.append({
                    "id": spec["id"],
                    "label": spec["label"],
                    "wav": obuf.getvalue(),
                    "sr": int(sr),
                    "auto_f0_adjust": spec["auto_f0_adjust"],
                    "semi_tone_shift": int(shift),
                })
                vc_out.unlink(missing_ok=True)
        finally:
            thon_out.unlink(missing_ok=True)
            if temp_target_path and temp_target_path.exists():
                try:
                    temp_target_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return {
            "modes": modes_out,
            "pre_vc": pre_vc,
            "diag": {
                "emotion": emotion,
                "donor": donor_wav.name,
                "speed": round(float(chunk_speed), 3),
                "target_med_hz": round(target_med, 1) if target_med else None,
                "donor_neutral_med_hz": round(donor_neutral_med, 1) if donor_neutral_med else None,
                "b_semi_tone_shift": semi_shift_b,
            },
        }

    # ------------------------------------------------------------------ #
    # Core Generation & Chunk Rendering
    # ------------------------------------------------------------------ #

    def render_chunks(
        self,
        texts: Sequence[str],
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        gender: Optional[str] = None,
        donor_set: Optional[str] = None,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 32,
        speed: Optional[float] = None,
        lora_mode: Optional[str] = None,
        pre_vc_out: Optional[List[Tuple[np.ndarray, int]]] = None,
        debug_out: Optional[List[dict]] = None,
    ) -> Tuple[List[Chunk], int]:
        pipeline = self.get_pipeline()
        # Apply this batch's F5 knobs (and reset everything else to the server
        # defaults, so it never inherits values a prior Pipeline Explorer run left on
        # the shared AudioConfig). cfg_value / inference_timesteps come straight from
        # the caller (studio sliders, benchmark params) and DO drive F5 -- higher cfg
        # makes it hew harder to the donor's emotional prosody.
        self._apply_audio_tuning({
            "cfg_strength": cfg_value,
            "nfe_step": inference_timesteps,
        })

        # F5 speech speed: <1.0 slower, >1.0 faster.
        #  * explicit ``speed`` (e.g. a studio slider) always wins.
        #  * otherwise, when a real user target voice is chosen, match the target's own
        #    speaking rate so the *user's* pace drives the tempo, not the emotion donor's.
        #  * failing both, fall back to the fixed server default.
        explicit_speed = None if speed is None else float(speed)
        use_rate_match = (
            explicit_speed is None
            and settings.thonburian_match_target_rate
            and bool(ref_audio_bytes or speaker_id)
        )

        temp_target_path = None
        if ref_audio_bytes:
            ext = Path(ref_filename or "upload.wav").suffix.lower()
            if ext not in AUDIO_EXTS:
                ext = ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(ref_audio_bytes)
                temp_target_path = Path(tf.name)
            target_wav = temp_target_path
        elif speaker_id:
            spk_path = self.get_speaker_audio_path(speaker_id)
            if not spk_path or not spk_path.exists():
                raise FileNotFoundError(f"Speaker '{speaker_id}' reference audio not found")
            target_wav = spk_path
        else:
            target_wav = self._resolve_default_speaker(gender=gender)

        scratch_dir = Path("scratch/thonburian_gen")
        scratch_dir.mkdir(parents=True, exist_ok=True)

        rendered_chunks: List[Chunk] = []

        try:
            for i, text in enumerate(texts):
                if not text or not str(text).strip():
                    continue

                raw_tone = tones[i] if tones is not None and i < len(tones) else None
                break_before = bool(breaks[i]) if breaks is not None and i < len(breaks) else False

                # 1. Validate emotion strictly
                emotion = self._validate_and_map_emotion(raw_tone)

                # 2. Clean and prepare text
                body_text = self._prepare_text(text)
                if not body_text:
                    continue

                # 3. Find donor clip and sidecar transcript
                donor_wav, donor_txt = self._resolve_donor_clip(
                    emotion, gender=gender, donor_set=donor_set
                )

                # 3b. Resolve this chunk's F5 speed. Rate-matching is per donor clip
                # because each emotion's donor has its own natural rate.
                rate_matched = None
                if explicit_speed is not None:
                    chunk_speed = explicit_speed
                elif use_rate_match:
                    rate_matched = self._matched_speed(target_wav, donor_wav)
                    chunk_speed = rate_matched if rate_matched is not None else settings.thonburian_speed
                else:
                    chunk_speed = settings.thonburian_speed

                ts = int(time.time() * 1000)
                thon_out = scratch_dir / f"thon_{emotion}_{i}_{ts}.wav"
                vc_out = scratch_dir / f"vc_{emotion}_{i}_{ts}.wav"

                # Record exactly what each model stage is handed, for UI inspection.
                if debug_out is not None:
                    debug_out.append({
                        "chunk_index": i,
                        "emotion": emotion,
                        "raw_text": text,
                        "f5_input": {
                            "text": body_text,
                            "ref_voice": donor_wav.name,
                            "ref_text": donor_txt,
                            "speed": round(float(chunk_speed), 3),
                            "speed_source": (
                                "explicit" if explicit_speed is not None
                                else "matched_to_target" if rate_matched is not None
                                else "default"
                            ),
                            "cfg_value": cfg_value,
                            "inference_timesteps": inference_timesteps,
                            "lora_mode": lora_mode,
                            "donor_set": donor_set,
                        },
                        "seedvc_input": {
                            "target": Path(target_wav).name,
                            "f0_condition": settings.seedvc_f0_condition,
                            "auto_f0_adjust": settings.seedvc_auto_f0_adjust,
                            "diffusion_steps": settings.seedvc_diffusion_steps,
                            "semi_tone_shift": 0,
                            "inference_cfg_rate": 0.7,
                        },
                    })

                # 4. Thonburian F5 generation (24 kHz). Long runs are pre-split into
                # single-batch pieces so F5 never inserts its own seam pause / drops a
                # word at a batch boundary (see _render_f5).
                self._render_f5(
                    pipeline,
                    text=body_text,
                    donor_wav=donor_wav,
                    donor_txt=donor_txt,
                    out_path=thon_out,
                    speed=chunk_speed,
                    scratch_dir=scratch_dir,
                    tag=f"thon_{emotion}_{i}_{ts}",
                )

                # Tame F5's overshoot before anything reads it (SeedVC + pre-VC).
                self._depeak_f5(thon_out)

                # Release F5's GPU cache before handing off to SeedVC. They share
                # one GPU; without this the SeedVC /convert call can stall past its
                # read timeout while F5's freed-but-cached memory is still held.
                if settings.thonburian_free_cache:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except Exception:
                        pass

                # 4b. Optionally keep the F5 output (emotional speech in donor timbre)
                # *before* SeedVC, so the caller can offer a "pre-VC" player.
                if pre_vc_out is not None:
                    try:
                        pre_arr, pre_sr = sf.read(str(thon_out), dtype="float32")
                        if pre_arr.ndim > 1:
                            pre_arr = pre_arr.mean(axis=1)
                        pre_vc_out.append((pre_arr, int(pre_sr)))
                    except Exception:
                        pass

                # 5. SeedVC voice conversion (44.1 kHz)
                self._convert_seedvc(source_wav=thon_out, target_wav=target_wav, output_wav=vc_out)

                # 6. Load resulting audio
                audio_arr, sr = sf.read(str(vc_out), dtype="float32")
                if audio_arr.ndim > 1:
                    audio_arr = audio_arr.mean(axis=1)

                rendered_chunks.append(
                    Chunk(
                        audio=audio_arr,
                        tone=raw_tone,
                        break_before=break_before,
                        text_len=spoken_len(body_text),
                    )
                )

                # Cleanup intermediate per-chunk files
                thon_out.unlink(missing_ok=True)
                vc_out.unlink(missing_ok=True)

            if not rendered_chunks:
                raise ValueError("No valid text or audio was rendered")

            return rendered_chunks, self.sample_rate

        finally:
            if temp_target_path and temp_target_path.exists():
                try:
                    temp_target_path.unlink(missing_ok=True)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Pipeline explorer (donor -> F5 -> SeedVC, every stage kept)
    # ------------------------------------------------------------------ #

    def list_donor_sets(self) -> List[Dict[str, Any]]:
        """Donor sets available under the donor dir, each with its emotions + transcripts.

        A "set" is one subfolder holding ``<emotion>_1.wav`` + ``<emotion>_1.txt`` donor clips.
        All emotions in one set are meant to be the same speaker, so choosing a set fixes the
        person and only emotion varies. Same-person sets built by ``tools/build_donor_sets.py``
        are named ``<gender>_<actor_id>`` and described in ``donors_manifest.json``; the legacy
        ``female`` / ``male`` folders remain as per-gender fallbacks.
        """
        base = self.donor_dir
        if not base.exists():
            return []

        # Optional richer metadata (actor_id, gender, agreement) keyed by set id.
        manifest_meta: Dict[str, Dict[str, Any]] = {}
        manifest_path = base / "donors_manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                for entry in data.get("sets", []):
                    if entry.get("id"):
                        manifest_meta[entry["id"]] = entry
            except Exception:
                manifest_meta = {}

        def _infer_gender(name: str) -> Optional[str]:
            low = name.lower()
            if low == "male" or low.startswith("male_"):
                return "male"
            if low == "female" or low.startswith("female_"):
                return "female"
            return None

        sets: List[Dict[str, Any]] = []
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            emotions = []
            for emo in sorted(SUPPORTED_EMOTIONS):
                wav = d / f"{emo}_1.wav"
                if not wav.exists():
                    continue
                txt = wav.with_suffix(".txt")
                transcript = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
                emotions.append({"id": emo, "transcript": transcript})
            if not emotions:
                continue

            meta = manifest_meta.get(d.name, {})
            gender = meta.get("gender") or _infer_gender(d.name)
            actor_id = meta.get("actor_id")
            # Human-friendly name: same-person sets show the actor; legacy folders keep their name.
            if actor_id:
                name = f"{gender.title()} · Actor {actor_id}" if gender else f"Actor {actor_id}"
            else:
                name = d.name.replace("_", " ").title()

            sets.append({
                "id": d.name,
                "name": name,
                "gender": gender,
                "actor_id": actor_id,
                "same_person": bool(actor_id),
                "emotions": emotions,
            })
        return sets

    def get_donor_clip_path(self, donor_set: str, emotion: str) -> Optional[Path]:
        wav = self.donor_dir / donor_set / f"{emotion}_1.wav"
        return wav if wav.exists() else None

    def render_trace(
        self,
        *,
        donor_set: str,
        emotion: str,
        run_dir: Path,
        text: Optional[str] = None,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        speed: Optional[float] = None,
        tuning: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run one utterance donor -> F5 -> SeedVC, keeping every stage on disk.

        Returns the donor clip, the F5 output (emotional speech in donor timbre) and
        the SeedVC output (timbre swapped to the target), so the caller can offer
        playback of each step.
        """
        emotion = self._validate_and_map_emotion(emotion)
        donor_wav = self.get_donor_clip_path(donor_set, emotion)
        if donor_wav is None:
            raise FileNotFoundError(f"No donor clip for set '{donor_set}', emotion '{emotion}'")
        donor_txt_p = donor_wav.with_suffix(".txt")
        donor_txt = donor_txt_p.read_text(encoding="utf-8").strip() if donor_txt_p.exists() else ""

        # Resolve the target speaker (voice to clone).
        temp_target_path = None
        if ref_audio_bytes:
            ext = Path(ref_filename or "upload.wav").suffix.lower()
            if ext not in AUDIO_EXTS:
                ext = ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(ref_audio_bytes)
                temp_target_path = Path(tf.name)
            target_wav = temp_target_path
        elif speaker_id:
            target_wav = self.get_speaker_audio_path(speaker_id)
            if not target_wav or not target_wav.exists():
                raise FileNotFoundError(f"Speaker '{speaker_id}' reference audio not found")
        else:
            target_wav = self._resolve_default_speaker(gender=donor_set)

        # Text to synthesize: the user's text, else the donor transcript as a default.
        gen_text_raw = (text or "").strip() or donor_txt
        body = self._prepare_text(gen_text_raw)
        if not body:
            raise ValueError("No text to synthesize")

        speed_value = settings.thonburian_speed if speed is None else float(speed)

        run_dir.mkdir(parents=True, exist_ok=True)
        # Copy the donor clip into the run dir so all stages serve from one folder.
        donor_out = run_dir / f"donor_{emotion}{donor_wav.suffix.lower()}"
        stage_a = run_dir / f"A_f5_{emotion}.wav"
        stage_b = run_dir / f"B_vc_{emotion}.wav"

        try:
            import shutil
            shutil.copy2(donor_wav, donor_out)

            pipeline = self.get_pipeline()
            # Apply the F5 knobs for this run (falls back to server defaults for
            # anything the caller left unset), then record what actually took.
            effective_tuning = self._apply_audio_tuning(tuning)
            t0 = time.time()
            self._render_f5(
                pipeline,
                text=body,
                donor_wav=donor_wav,
                donor_txt=donor_txt,
                out_path=stage_a,
                speed=speed_value,
                scratch_dir=run_dir,
                tag=f"A_f5_{emotion}",
            )
            self._depeak_f5(stage_a)
            f5_secs = time.time() - t0

            if settings.thonburian_free_cache:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass

            t0 = time.time()
            self._convert_seedvc(source_wav=stage_a, target_wav=Path(target_wav), output_wav=stage_b)
            vc_secs = time.time() - t0
        finally:
            if temp_target_path and temp_target_path.exists():
                try:
                    temp_target_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return {
            "emotion": emotion,
            "donor_set": donor_set,
            "gen_text": gen_text_raw,
            "donor_transcript": donor_txt,
            "target": (speaker_id or (ref_filename or "upload") if ref_audio_bytes else speaker_id) or Path(target_wav).stem,
            "speed": speed_value,
            "tuning": effective_tuning,
            "f5_secs": round(f5_secs, 1),
            "vc_secs": round(vc_secs, 1),
            "files": {
                "donor": donor_out.name,
                "f5": stage_a.name,
                "vc": stage_b.name,
            },
        }

    # ------------------------------------------------------------------ #
    # Synthesis & Assembly APIs
    # ------------------------------------------------------------------ #

    def synthesize_many(
        self,
        texts: Sequence[str],
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        gender: Optional[str] = None,
        donor_set: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 32,
        speed: Optional[float] = None,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        post_process: bool = True,
        post_process_params: Optional[dict] = None,
        lora_mode: Optional[str] = None,
    ) -> bytes:
        rendered, sr = self.render_chunks(
            texts,
            speaker_id=speaker_id,
            ref_audio_bytes=ref_audio_bytes,
            ref_filename=ref_filename,
            gender=gender,
            donor_set=donor_set,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            speed=speed,
            tones=tones,
            breaks=breaks,
            lora_mode=lora_mode,
        )

        if post_process:
            config = PostProcessConfig.from_dict(post_process_params)
            audio = assemble(rendered, sr, config=config)
        else:
            audio = butt_join(rendered, sr)

        out_buf = io.BytesIO()
        sf.write(out_buf, audio, sr, format="WAV", subtype="PCM_16")
        return out_buf.getvalue()

    def synthesize_variants(
        self,
        texts: Sequence[str],
        *,
        variants: Sequence[dict],
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        gender: Optional[str] = None,
        donor_set: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 32,
        speed: Optional[float] = None,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        lora_mode: Optional[str] = None,
        pre_vc_sink: Optional[dict] = None,
        debug_sink: Optional[dict] = None,
    ) -> Tuple[List[dict], int, List[Optional[str]]]:
        # When a caller wants the pre-SeedVC F5 audio, collect it here and hand back one
        # joined clip via ``pre_vc_sink`` ({"wav": bytes, "sr": int}). It is the same
        # generation for every DSP variant, so it is produced once.
        pre_vc_chunks: Optional[List[Tuple[np.ndarray, int]]] = [] if pre_vc_sink is not None else None
        # Same idea for the raw model inputs (F5 + SeedVC), for UI inspection.
        debug_chunks: Optional[List[dict]] = [] if debug_sink is not None else None

        rendered, sr = self.render_chunks(
            texts,
            speaker_id=speaker_id,
            ref_audio_bytes=ref_audio_bytes,
            ref_filename=ref_filename,
            gender=gender,
            donor_set=donor_set,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            speed=speed,
            tones=tones,
            breaks=breaks,
            lora_mode=lora_mode,
            pre_vc_out=pre_vc_chunks,
            debug_out=debug_chunks,
        )

        if debug_sink is not None:
            debug_sink["chunks"] = debug_chunks or []

        if pre_vc_sink is not None and pre_vc_chunks:
            pre_sr = pre_vc_chunks[0][1]
            gap = np.zeros(int(0.06 * pre_sr), dtype=np.float32)
            pieces: List[np.ndarray] = []
            for idx, (arr, _asr) in enumerate(pre_vc_chunks):
                if idx:
                    pieces.append(gap)
                pieces.append(np.asarray(arr, dtype=np.float32))
            pre_audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
            pre_buf = io.BytesIO()
            sf.write(pre_buf, pre_audio, pre_sr, format="WAV", subtype="PCM_16")
            pre_vc_sink["wav"] = pre_buf.getvalue()
            pre_vc_sink["sr"] = int(pre_sr)
            pre_vc_sink["dur_s"] = round(len(pre_audio) / pre_sr, 3)

        usable = [c for c in rendered if c.audio is not None and np.asarray(c.audio).size]
        chunk_tones: List[Optional[str]] = [c.tone for c in usable]

        takes: List[dict] = []
        for spec in variants:
            if spec.get("post_process", True):
                config = PostProcessConfig.from_dict(spec.get("params"))
                audio, spans = assemble_with_spans(rendered, sr, config=config)
            else:
                audio, spans = butt_join_with_spans(rendered, sr)

            chunk_stats = []
            for i, (start, end) in enumerate(spans):
                seg = audio[int(start * sr):int(end * sr)]
                level = voiced_rms(seg, sr)
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

            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
            takes.append({
                "id": spec.get("id"),
                "wav": buf.getvalue(),
                "dur_s": round(len(audio) / sr, 3),
                "chunks": chunk_stats,
            })

        return takes, sr, chunk_tones

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self._is_loaded,
            "engine": "Thonburian F5 + SeedVC",
            "device": self.device or "cuda:0",
            "seedvc_url": self.seedvc_url,
            "sample_rate": self.sample_rate,
            "load_error": self._load_error,
            "speakers_count": len(self.list_speakers()),
        }


thonburian_service = ThonburianService()
