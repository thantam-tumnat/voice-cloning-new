"""Prompt-cache registry for the shared GPU service.

A prompt cache is a bag of tensors; it cannot travel over HTTP and it must not be
rebuilt per request. So the GPU service owns them and hands clients an opaque
*voice handle* instead — a short string they can pass back on the next render job.

Cache keys are deliberately the same shape the webhook service used in-process
(`<voice_id>-<sha1(ref_text)[:8]>.pt`), so the .pt files already sitting in
`voices/` keep being hits after the split rather than being re-encoded.

Handles survive a restart because the cache is persisted; a handle the store has
never heard of gets a 410 from the API so the client can register it again.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")

# Handles minted from an uploaded clip are not addressable by name, so nothing
# would ever evict them. Drop them once they have gone unused for this long.
UPLOAD_TTL_S = float(os.environ.get("SIANGTTS_UPLOAD_VOICE_TTL", "3600"))

# Filename of the neutral seed voice every unpinned multi-chunk request is
# conditioned on. Leading underscore keeps it out of the speaker listing, which
# enumerates the reference directories rather than the cache.
SEED_VOICE_KEY = "_auto_seed"


class UnknownVoice(KeyError):
    """The handle is not in memory and has no file behind it."""


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


class VoiceStore:
    """Encoded reference clips, keyed by voice + ref text.

    VoxCPM2's prompt cache is bound to the transcript it was built with, so a voice
    used with two different `ref_text` values needs two caches. Built lazily on first
    use and persisted, because voice ids arrive from the caller and are not known at
    startup.
    """

    def __init__(
        self,
        synth: Any,
        cache_dir: Path,
        ref_dirs: list[Path],
        seed_text: str = "",
    ) -> None:
        self.synth = synth
        self.cache_dir = Path(cache_dir)
        self.ref_dirs = [Path(d) for d in ref_dirs]
        self.seed_text = seed_text
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mem: dict[str, Any] = {}
        self.meta: dict[str, dict] = {}
        self._seed_failed = False

    # ------------------------------------------------------------------ #
    # Reference clip lookup
    # ------------------------------------------------------------------ #

    def ref_file(self, voice_id: str) -> Path:
        for d in self.ref_dirs:
            if not d.exists():
                continue
            for ext in AUDIO_EXTS:
                p = d / f"{voice_id}{ext}"
                if p.exists():
                    return p
        listed = ", ".join(str(d) for d in self.ref_dirs)
        raise UnknownVoice(f"voice '{voice_id}' has no reference clip in {listed}")

    def list_voices(self) -> list[dict]:
        """Reference clips on disk, plus whether one is already encoded."""
        cached_ids = {p.stem.split("-")[0] for p in self.cache_dir.glob("*.pt")}
        out: list[dict] = []
        seen: set[str] = set()
        for d in self.ref_dirs:
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in AUDIO_EXTS and f.stem not in seen:
                    seen.add(f.stem)
                    out.append({"id": f.stem, "file": f.name, "cached": f.stem in cached_ids})
        return out

    # ------------------------------------------------------------------ #
    # Handles
    # ------------------------------------------------------------------ #

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    def _remember(self, key: str, cache: Any, **meta) -> str:
        self.mem[key] = cache
        self.meta[key] = {"created": time.time(), "used": time.time(), **meta}
        return key

    def get(self, handle: str) -> Any:
        """The prompt cache behind a handle, loading it from disk if needed."""
        if handle in self.mem:
            self.meta.setdefault(handle, {})["used"] = time.time()
            return self.mem[handle]

        path = self._path_for(handle)
        if not path.exists():
            raise UnknownVoice(handle)
        cache = self.synth.load_voice(path)
        self._remember(handle, cache, source="disk")
        return cache

    def resolve_speaker(
        self,
        voice_id: str,
        ref_text: str = "",
        allow_sidecar: bool = True,
    ) -> str:
        """Handle for a named voice from the reference directories.

        `allow_sidecar` reproduces the webhook's rule: when the caller supplied no
        transcript of its own, a `<clip>.txt` sitting beside the clip is used as one.
        The caller decides, because only it knows whether the text it sent was a real
        transcript or its own stand-in default.
        """
        ref = self.ref_file(voice_id)

        if allow_sidecar:
            txt = ref.with_suffix(".txt")
            if txt.exists():
                try:
                    ref_text = txt.read_text(encoding="utf-8").strip() or ref_text
                except Exception:
                    pass

        key = f"{voice_id}-{_digest(ref_text)}"
        if key in self.mem:
            self.meta.setdefault(key, {})["used"] = time.time()
            return key

        path = self._path_for(key)
        if path.exists() and path.stat().st_mtime >= ref.stat().st_mtime:
            cache = self.synth.load_voice(path)
        else:
            print(f"[voices] encoding '{voice_id}' from {ref.name} …")
            cache = self.synth.build_voice(str(ref), prompt_text=ref_text or None)
            self.synth.save_voice(cache, path)
        return self._remember(key, cache, speaker_id=voice_id, source="ref")

    def register_clip(
        self,
        clip_path: Path,
        speaker_id: Optional[str] = None,
        ref_text: Optional[str] = None,
        persist: bool = True,
    ) -> str:
        """Encode a clip the client just uploaded.

        With `speaker_id` the clip is filed under the same key scheme as a named
        voice, so the next request naming that speaker is a cache hit. Without one it
        gets a throwaway handle that expires (see UPLOAD_TTL_S) — the tone studio's
        one-off "synthesize with this file" path lands here.
        """
        if speaker_id:
            key = f"{speaker_id}-{_digest(ref_text or '')}"
        else:
            key = f"up_{uuid.uuid4().hex[:12]}"

        cache = self.synth.build_voice(str(clip_path), prompt_text=ref_text or None)
        if persist:
            try:
                self.synth.save_voice(cache, self._path_for(key))
            except Exception as e:                                    # noqa: BLE001
                print(f"[voices] could not persist '{key}': {e}")
        return self._remember(key, cache, speaker_id=speaker_id, source="upload")

    def register_audio(self, wav: Any, sample_rate: int, persist: bool = False) -> str:
        """Encode audio already in memory (used to clone a generated take).

        Deliberately reference mode, no transcript: passing one selects VoxCPM2's
        continuation mode, which reproduces the source's emotion as well as its
        timbre. Timbre is what should carry between chunks; prosody must stay free
        for the next chunk's style instruction to steer.
        """
        import tempfile

        import soundfile as sf

        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp = tf.name
            sf.write(tmp, wav, sample_rate, format="WAV", subtype="PCM_16")
            return self.register_clip(Path(tmp), persist=persist)
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Seed voice
    # ------------------------------------------------------------------ #

    def seed(self, generate) -> Optional[str]:
        """Handle for the shared neutral seed voice, minting it on first use.

        Built once and then reused, from memory within a process and from
        `<cache>/_auto_seed.pt` across restarts. Regenerating it per request made
        every request a slightly different speaker.

        `generate` is a callable taking the seed text and returning audio, so this
        module never has to know how the engine wants to be invoked.

        Returns None on failure — an inconsistent voice beats a failed request.
        """
        if SEED_VOICE_KEY in self.mem:
            return SEED_VOICE_KEY
        if self._seed_failed or not self.seed_text.strip():
            return None

        path = self._path_for(SEED_VOICE_KEY)
        if path.exists():
            try:
                return self._remember(SEED_VOICE_KEY, self.synth.load_voice(path), source="seed")
            except Exception as e:                                    # noqa: BLE001
                print(f"[voices] stale seed voice ({e}); rebuilding.")

        try:
            wav = generate(self.seed_text)
            handle = self.register_audio(wav, self.synth.sample_rate, persist=False)
            cache = self.mem.pop(handle)
            self.meta.pop(handle, None)
        except Exception as e:                                        # noqa: BLE001
            print(f"[voices] seed voice generation failed: {e}")
            self._seed_failed = True
            return None

        try:
            self.synth.save_voice(cache, path)
        except Exception as e:                                        # noqa: BLE001
            print(f"[voices] could not persist seed voice: {e}")
        return self._remember(SEED_VOICE_KEY, cache, source="seed")

    def reset_seed(self) -> bool:
        """Throw away the seed voice so the next request mints a new speaker.

        The seed is one unseeded generation that then becomes permanent, which is the
        point right up until the draw is a bad one — after that every unpinned request
        inherits it and re-rendering never escapes. Pinned speakers are untouched.
        """
        self.mem.pop(SEED_VOICE_KEY, None)
        self.meta.pop(SEED_VOICE_KEY, None)
        self._seed_failed = False
        path = self._path_for(SEED_VOICE_KEY)
        try:
            existed = path.exists()
            path.unlink(missing_ok=True)
            return existed
        except Exception as e:                                        # noqa: BLE001
            print(f"[voices] could not delete seed voice: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #

    def evict_expired(self, now: Optional[float] = None) -> int:
        """Drop upload handles nobody has touched for a while."""
        now = now if now is not None else time.time()
        dropped = 0
        for key in [k for k in self.mem if k.startswith("up_")]:
            if now - self.meta.get(key, {}).get("used", now) > UPLOAD_TTL_S:
                self.mem.pop(key, None)
                self.meta.pop(key, None)
                self._path_for(key).unlink(missing_ok=True)
                dropped += 1
        return dropped

    def stats(self) -> dict:
        return {
            "in_memory": len(self.mem),
            "on_disk": len(list(self.cache_dir.glob("*.pt"))),
            "handles": sorted(self.mem),
        }


__all__ = ["AUDIO_EXTS", "SEED_VOICE_KEY", "UnknownVoice", "VoiceStore"]
