"""Client for the shared SiangTTS GPU service (:8020).

Replaces the in-process model. The studio keeps everything that makes it the studio
— segmentation, the style vocabulary, chunk splitting, `audio_post.assemble` — and
sends only the generation across, so there is one copy of VoxCPM2 on the box instead
of one per pipeline.

Two things this has to get right that the previous remote path did not:

* **Voice identity.** A prompt cache is tensors; it cannot cross HTTP. The service
  keeps them and hands back an opaque *handle*, which is what `build_voice` returns
  here. The old client returned the string ``"remote_latent_for_<path>"`` and the
  remote endpoint ignored it, so every chunk was generated unconditioned and the
  speaker wandered — exactly what the seed voice exists to prevent.

* **LoRA strength.** The studio runs the adapter with the DiT side at zero (see
  ``siangtts_lora_dit_scale`` in app/config.py); at the shipped strength the emotion
  tags measure backwards. The scale is global state on a shared model, so it travels
  with each request and the service applies it per job.

A whole run of chunks goes over as **one job** with one voice, which is what keeps
the timbre steady across chunks and costs one round trip instead of N.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

DEFAULT_TIMEOUT = float(os.environ.get("VOXCPM_CLIENT_TIMEOUT", "600"))


class RemoteSynthesisError(RuntimeError):
    pass


class QueueSynthesizer:
    """Duck-compatible with `_RealSynthesizer`, but the model is in another process."""

    is_remote = True

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sample_rate = 48000
        self.lora_loaded = False
        self.is_stub = False
        # Nothing local to scale — the service does it per job. Present because
        # SiangTTSService checks for it before calling set_lora_strength.
        self.tts_model = None

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #

    def _client(self, timeout: Optional[float] = None):
        import httpx

        return httpx.Client(base_url=self.base_url, timeout=timeout or self.timeout)

    def check_health(self) -> bool:
        try:
            with self._client(timeout=2.0) as c:
                res = c.get("/health")
                if res.status_code != 200:
                    return False
                data = res.json()
        except Exception:
            return False

        self.sample_rate = data.get("sample_rate", 48000)
        self.lora_loaded = bool(data.get("adapter"))
        self.is_stub = bool(data.get("stub"))
        if self.is_stub:
            print(
                "[SiangTTS] WARNING: the GPU service is in STUB MODE — audio will be "
                "a test tone, not speech.",
                file=sys.stderr,
            )
        return True

    def _lora_spec(self, lora_mode: Optional[str]) -> dict:
        """Translate the studio's mode names into explicit scales.

        Sent as numbers rather than a mode name so app/config.py stays the authority
        on what "on" means here, without the GPU service having to track it.
        """
        from app.config import settings

        if lora_mode == "off":
            return {"lm": 0.0, "dit": 0.0}
        if lora_mode == "legacy":
            return {"lm": 2.0, "dit": 2.0}
        return {
            "lm": float(settings.siangtts_lora_lm_scale),
            "dit": float(settings.siangtts_lora_dit_scale),
        }

    # ------------------------------------------------------------------ #
    # Voices
    # ------------------------------------------------------------------ #

    def resolve_speaker(
        self,
        speaker_id: str,
        ref_text: str = "",
        allow_sidecar: bool = False,
    ) -> Optional[str]:
        """Handle for a voice the *service* has in its reference directory.

        Returns None when it does not know the speaker, so the caller can fall back
        to uploading a clip it has locally.
        """
        try:
            with self._client(timeout=60.0) as c:
                res = c.post(
                    "/v2/voices/resolve",
                    json={
                        "speaker_id": speaker_id,
                        "ref_text": ref_text,
                        "allow_sidecar": allow_sidecar,
                    },
                )
            if res.status_code == 200:
                return res.json()["voice_handle"]
        except Exception as e:                                        # noqa: BLE001
            print(f"[SiangTTS] speaker resolve failed for '{speaker_id}': {e}", file=sys.stderr)
        return None

    def build_voice(self, ref_audio_path: str, prompt_text: Optional[str] = None) -> str:
        """Upload a clip and get back a handle.

        Note the transcript is only sent when the caller asked for one: passing one
        selects VoxCPM2's continuation mode, which reproduces the prompt clip's own
        emotion and ignores style instructions.
        """
        path = Path(ref_audio_path)
        data = {"ref_text": prompt_text} if prompt_text else {}
        with self._client(timeout=120.0) as c:
            with open(path, "rb") as fh:
                res = c.post(
                    "/v2/voices",
                    data=data,
                    files={"clip": (path.name, fh, "application/octet-stream")},
                )
        if res.status_code != 200:
            raise RemoteSynthesisError(f"voice upload failed ({res.status_code}): {res.text}")
        return res.json()["voice_handle"]

    def seed_voice(self) -> Optional[str]:
        """Handle for the shared neutral seed voice, minted service-side on first use.

        Doing it there rather than here is what makes it *shared*: every client that
        leaves the voice unpinned gets the same speaker, and it survives a studio
        restart without a local cache file to go stale.
        """
        try:
            with self._client(timeout=300.0) as c:
                res = c.post("/v2/voices/seed")
            if res.status_code == 200:
                return res.json()["voice_handle"]
            print(f"[SiangTTS] seed voice unavailable ({res.status_code}): {res.text}",
                  file=sys.stderr)
        except Exception as e:                                        # noqa: BLE001
            print(f"[SiangTTS] seed voice request failed: {e}", file=sys.stderr)
        return None

    def reset_seed_voice(self) -> bool:
        try:
            with self._client(timeout=30.0) as c:
                res = c.delete("/v2/voices/seed")
            return bool(res.status_code == 200 and res.json().get("cache_removed"))
        except Exception as e:                                        # noqa: BLE001
            print(f"[SiangTTS] seed reroll failed: {e}", file=sys.stderr)
            return False

    def list_speakers(self) -> List[dict]:
        try:
            with self._client(timeout=15.0) as c:
                res = c.get("/v2/voices")
            if res.status_code == 200:
                return res.json().get("voices", [])
        except Exception as e:                                        # noqa: BLE001
            print(f"[SiangTTS] speaker listing failed: {e}", file=sys.stderr)
        return []

    def get_speaker_audio_bytes(self, speaker_id: str) -> Optional[Tuple[bytes, str, str]]:
        """Retrieve reference audio clip from the shared GPU service.
        Returns (audio_bytes, media_type, filename) or None.
        """
        try:
            with self._client(timeout=15.0) as c:
                res = c.get(f"/v2/voices/{speaker_id}/audio")
                if res.status_code == 200:
                    media_type = res.headers.get("content-type", "audio/wav")
                    cd = res.headers.get("content-disposition", "")
                    filename = f"{speaker_id}.wav"
                    if "filename=" in cd:
                        filename = cd.split("filename=")[-1].strip('"')
                    return res.content, media_type, filename
        except Exception as e:                                        # noqa: BLE001
            print(f"[SiangTTS] speaker audio fetch failed for '{speaker_id}': {e}", file=sys.stderr)
        return None

    def register_speaker(self, speaker_id: str, audio_bytes: bytes, filename: str) -> str:
        with self._client(timeout=120.0) as c:
            res = c.post(
                "/v2/voices",
                data={"speaker_id": speaker_id, "save_as_speaker": "true"},
                files={"clip": (filename, audio_bytes, "application/octet-stream")},
            )
        if res.status_code != 200:
            raise RemoteSynthesisError(f"speaker register failed ({res.status_code}): {res.text}")
        return res.json()["voice_handle"]

    def delete_speaker(self, speaker_id: str) -> bool:
        try:
            with self._client(timeout=30.0) as c:
                return c.delete(f"/v2/voices/{speaker_id}").status_code == 200
        except Exception:
            return False

    # A handle is a short string, so "persisting" one is writing it down. Kept for
    # interface compatibility; the studio no longer needs a local voice cache,
    # because the service is the cache.
    def save_voice(self, cache: Any, dest_path: Path) -> None:
        Path(dest_path).write_text(str(cache), encoding="utf-8")

    def load_voice(self, src_path: Path) -> Any:
        return Path(src_path).read_text(encoding="utf-8").strip()

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _voice_spec(
        self,
        prompt_cache: Any,
        speaker_id: Optional[str],
        ref_audio: Optional[str],
    ) -> Optional[dict]:
        """Precedence: an explicit handle, then a name the service can resolve
        itself, then a clip we have to upload. A named speaker beats a local file
        because the service's reference directory is the source of truth — and it
        saves re-uploading the same clip on every request."""
        if isinstance(prompt_cache, str) and prompt_cache:
            return {"handle": prompt_cache}
        if speaker_id:
            handle = self.resolve_speaker(speaker_id, allow_sidecar=False)
            if handle:
                return {"handle": handle}
        if ref_audio and os.path.exists(ref_audio):
            return {"handle": self.build_voice(ref_audio)}
        return None

    def render_batch(
        self,
        texts: Sequence[str],
        *,
        speaker_id: Optional[str] = None,
        prompt_cache: Any = None,
        ref_audio: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        lora_mode: Optional[str] = "on",
        lane: str = "interactive",
    ) -> Tuple[List[Any], int]:
        """Generate every chunk in one job against one voice.

        One job, not one request per chunk: the chunks have to share a prompt cache
        or the speaker drifts between them, and the service can only guarantee that
        if it sees them together.
        """
        import numpy as np

        from app.services.siangtts_service import prepare_text

        prepared_chunks = [prepare_text(t) for t in texts]
        spk_label = speaker_id or (prompt_cache if isinstance(prompt_cache, str) else "auto-seed")
        print(
            f"\n[ToneStudio -> GPU] >>> Submitting {len(prepared_chunks)} chunk(s) "
            f"(speaker={spk_label}, lora={lora_mode}, cfg={cfg_value}):",
            file=sys.stderr,
        )
        for idx, c in enumerate(prepared_chunks):
            print(f"[ToneStudio -> GPU]     [{idx+1}/{len(prepared_chunks)}] {c!r}", file=sys.stderr)

        payload = {
            "chunks": prepared_chunks,
            "voice": self._voice_spec(prompt_cache, speaker_id, ref_audio),
            "cfg_value": cfg_value,
            "timesteps": inference_timesteps,
            "lora": self._lora_spec(lora_mode),
            "output": {"mode": "npz"},
            "lane": lane,
            "client": "tone-studio",
        }

        with self._client() as c:
            res = c.post(f"/v2/jobs/render?wait={self.timeout - 5}", json=payload)

        if res.status_code == 202:
            # The job outlived the wait. It is still running service-side; the studio
            # has no async surface to hand it to, so say so plainly.
            raise RemoteSynthesisError(
                f"render did not finish within {self.timeout:.0f}s "
                f"(job {res.json().get('job_id')} is still running on the GPU service)"
            )
        if res.status_code != 200:
            raise RemoteSynthesisError(f"render failed ({res.status_code}): {res.text[:400]}")

        with np.load(io.BytesIO(res.content)) as bundle:
            self.sample_rate = int(bundle["sample_rate"])
            count = int(bundle["count"])
            chunks = [np.asarray(bundle[f"chunk_{i:03d}"], dtype="float32") for i in range(count)]
        return chunks, self.sample_rate

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
        **kwargs: Any,
    ):
        chunks, _ = self.render_batch(
            [text],
            speaker_id=speaker_id,
            prompt_cache=prompt_cache,
            ref_audio=ref_audio,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            lora_mode=lora_mode,
        )
        return chunks[0]


__all__ = ["QueueSynthesizer", "RemoteSynthesisError"]
