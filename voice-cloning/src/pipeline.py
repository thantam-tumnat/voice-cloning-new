"""Post-synthesis stages of the webhook pipeline: merge, upload, callback, cleanup.

These replace four n8n nodes plus the external helper service the flow called at
`host.docker.internal:8000`:

    merge_audio_ffmpeg-trimmed  ->  merge_chunks()      (was POST /merge-tts-trimmed)
    upload to                   ->  upload()
    send_link / send_link_error ->  post_callback()
    Cleanup                     ->  cleanup()           (was POST /cleanup-tts)

Defaults are the literal values the old flow sent, so output should match.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx


from .env_file import load_env_file


def _load_dotenv() -> None:
    """Kept as a name callers already use; the implementation moved to
    src/env_file.py so the GPU service can share it without importing this
    module, which is all about ffmpeg and uploads."""
    load_env_file()


_load_dotenv()

WORK_ROOT = Path(os.environ.get("SIANGTTS_WORK_DIR", "work"))
UPLOAD_URL = os.environ.get(
    "SIANGTTS_UPLOAD_URL", "https://looklike.ai/api/v1/live-gpt/upload"
).strip()
UPLOAD_TOKEN = os.environ.get("SIANGTTS_UPLOAD_TOKEN", "").strip()
FFMPEG = os.environ.get("SIANGTTS_FFMPEG", "ffmpeg")

HTTP_TIMEOUT = float(os.environ.get("SIANGTTS_HTTP_TIMEOUT", "120"))


@dataclass(frozen=True)
class MergeOptions:
    """Mirrors the JSON body the flow posted to /merge-tts-trimmed."""

    trim_mode: str = "start_only"       # "start_only" | "both" | "none"
    start_silence: float = 0.02
    stop_silence: float = 0.05
    start_threshold: str = "-65dB"
    stop_threshold: str = "-65dB"
    bitrate: str = "192k"
    speed: float = 1.0


def _silence_filter(opts: MergeOptions) -> str | None:
    """Per-chunk silence trim. VoxCPM tends to leave a beat of room tone at the
    head of each generation; concatenating untrimmed chunks stacks those into
    audible gaps mid-sentence."""
    if opts.trim_mode == "none":
        return None
    parts = [
        "silenceremove=start_periods=1",
        f"start_duration={opts.start_silence}",
        f"start_threshold={opts.start_threshold}",
    ]
    if opts.trim_mode == "both":
        parts += [
            "stop_periods=-1",
            f"stop_duration={opts.stop_silence}",
            f"stop_threshold={opts.stop_threshold}",
        ]
    return ":".join(parts)


def merge_chunks(wav_paths: list[Path], out_path: Path, opts: MergeOptions) -> Path:
    """Trim + concatenate chunk WAVs into one MP3.

    One ffmpeg process for the whole job: trimming has to happen per input
    (before the concat) or it would only touch the head of the finished file.
    """
    if not wav_paths:
        raise RuntimeError("ffmpeg merge failed - no audio generated")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trim = _silence_filter(opts)

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    for p in wav_paths:
        cmd += ["-i", str(p)]

    graph = []
    labels = []
    for i in range(len(wav_paths)):
        if trim:
            graph.append(f"[{i}:a]{trim}[t{i}]")
            labels.append(f"[t{i}]")
        else:
            labels.append(f"[{i}:a]")
    graph.append(f"{''.join(labels)}concat=n={len(wav_paths)}:v=0:a=1[m]")

    tail = "m"
    # VoxCPM has no speed control of its own, so the flow's `speed` becomes a
    # tempo change here. atempo only accepts 0.5-2.0 per instance.
    if abs(opts.speed - 1.0) > 1e-3:
        rate = min(2.0, max(0.5, opts.speed))
        graph.append(f"[m]atempo={rate}[s]")
        tail = "s"

    cmd += [
        "-filter_complex", ";".join(graph),
        "-map", f"[{tail}]",
        "-b:a", opts.bitrate,
        str(out_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg merge failed - {(proc.stderr or '').strip()[:400]}")
    return out_path


async def upload(path: Path, *, url: str | None = None, token: str | None = None) -> str:
    """POST the merged audio as multipart `file`, return the `file_url` field.

    The bearer token lived in n8n's credential store as "VR_live Auth"; it now
    comes from SIANGTTS_UPLOAD_TOKEN.
    """
    _load_dotenv()
    target_url = (url or os.environ.get("SIANGTTS_UPLOAD_URL") or UPLOAD_URL).strip()
    bearer_token = (
        token or os.environ.get("SIANGTTS_UPLOAD_TOKEN", "").strip() or UPLOAD_TOKEN
    ).strip()

    if not bearer_token:
        raise RuntimeError("SIANGTTS_UPLOAD_TOKEN is not set")

    headers = {"Authorization": f"Bearer {bearer_token}"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        with open(path, "rb") as fh:
            resp = await client.post(
                target_url, headers=headers, files={"file": (path.name, fh, "audio/mpeg")}
            )
    if resp.status_code >= 400:
        raise RuntimeError(f"upload failed {resp.status_code} - {resp.text[:300]}")

    data = resp.json()
    file_url = data.get("file_url") or data.get("url")
    if not file_url:
        raise RuntimeError(f"upload response has no file_url - {resp.text[:300]}")
    return file_url


async def post_callback(
    url: str, *, job_id: str, queue_id: str, file_url: str | None, error: str | None
) -> None:
    """Report the result. Body shape matches the flow's send_link node.

    The old flow's early-failure path (error_callback) posted a different shape,
    `{status, error}`, with no ids — a receiver could not tell which job it
    referred to. Every failure uses the send_link_error shape here instead.
    """
    payload = {
        "job_id": job_id,
        "queue_id": queue_id,
        "file_url": file_url if file_url else "none",
        "error": error,
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        await client.post(url, json=payload)


def cleanup(work_dir: Path, *, delete_merged: bool = True) -> None:
    """Drop the job's scratch directory. Never raises — a cleanup failure must
    not turn a delivered job into a failed one."""
    try:
        if not work_dir.exists():
            return
        if delete_merged:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            for f in work_dir.glob("*.wav"):
                f.unlink(missing_ok=True)
    except Exception as exc:                                    # noqa: BLE001
        print(f"[cleanup] ignored: {exc}")


__all__ = ["MergeOptions", "WORK_ROOT", "cleanup", "merge_chunks", "post_callback", "upload"]
