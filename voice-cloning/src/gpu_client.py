"""Async client for the GPU service (:8020), used by the webhook service.

Kept deliberately thin: submit, poll, and a couple of voice calls. The retry budget
exists because the GPU service is a separate process now — restarting it to pick up
a model change should stall the webhook's jobs, not fail them.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Optional

import httpx

DEFAULT_URL = os.environ.get("SIANGTTS_GPU_URL", "http://127.0.0.1:8020")

# How long a job may keep retrying through a dead GPU service before giving up. Long
# enough to ride out a restart (model load is 30–60 s), short enough that a service
# that is really gone still reports failure through the normal callback path.
RECONNECT_BUDGET_S = float(os.environ.get("SIANGTTS_GPU_RECONNECT_BUDGET", "180"))
POLL_INTERVAL_S = float(os.environ.get("SIANGTTS_GPU_POLL_INTERVAL", "1.0"))


class GPUServiceError(RuntimeError):
    pass


class GPUUnavailable(GPUServiceError):
    """Could not reach the service at all — distinct from a job that failed."""


class JobNotFound(GPUServiceError):
    """The service is up and has never heard of this job.

    Its job table is in memory, so this is what a restart looks like from here: the
    connection succeeds and the work is simply gone. Distinct from GPUUnavailable
    because there is nothing to wait for — retrying can only burn the whole
    reconnect budget before failing anyway.
    """


class GPUClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _client(self, timeout: Optional[float] = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=timeout or self.timeout)

    # -- status ---------------------------------------------------------- #

    async def health(self) -> Optional[dict]:
        """None when unreachable, so a caller can report status without try/except."""
        try:
            async with self._client(timeout=5.0) as c:
                r = await c.get("/health")
                return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    async def list_voices(self) -> Optional[dict]:
        try:
            async with self._client(timeout=10.0) as c:
                r = await c.get("/v2/voices")
                return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    # -- voices ---------------------------------------------------------- #

    async def resolve_voice(
        self, speaker_id: str, ref_text: str = "", allow_sidecar: bool = True
    ) -> str:
        async with self._client() as c:
            r = await c.post(
                "/v2/voices/resolve",
                json={
                    "speaker_id": speaker_id,
                    "ref_text": ref_text,
                    "allow_sidecar": allow_sidecar,
                },
            )
        if r.status_code != 200:
            raise GPUServiceError(f"voice resolve failed ({r.status_code}): {r.text}")
        return r.json()["voice_handle"]

    # -- jobs ------------------------------------------------------------ #

    async def submit_render(self, body: dict) -> dict:
        async with self._client() as c:
            r = await c.post("/v2/jobs/render", json=body)
        if r.status_code not in (200, 202):
            raise GPUServiceError(f"render submit failed ({r.status_code}): {r.text}")
        return r.json()

    async def get_job(self, job_id: str) -> dict:
        async with self._client(timeout=15.0) as c:
            r = await c.get(f"/v2/jobs/{job_id}")
        if r.status_code == 404:
            raise JobNotFound(job_id)
        if r.status_code != 200:
            raise GPUServiceError(f"job lookup failed ({r.status_code}): {r.text}")
        return r.json()

    async def await_job(
        self,
        job_id: str,
        *,
        on_progress: Optional[Callable[[dict], Any]] = None,
        poll_interval: float = POLL_INTERVAL_S,
        reconnect_budget: float = RECONNECT_BUDGET_S,
    ) -> dict:
        """Poll until the job reaches a terminal state.

        Connection errors do not end the wait: the job is queued *on the service*, so
        a client that gave up on the first refused connection would abandon work that
        is still going to run. Only a sustained outage fails.
        """
        unreachable_since: Optional[float] = None
        while True:
            try:
                job = await self.get_job(job_id)
                unreachable_since = None
            except JobNotFound:
                # The service restarted and lost the queue. Fail now with something
                # the callback can actually explain, instead of polling a job that
                # will never exist for the rest of the reconnect budget.
                raise GPUUnavailable(
                    f"GPU service no longer knows job {job_id} — it restarted while "
                    f"the job was queued or running"
                ) from None
            except Exception:
                now = asyncio.get_event_loop().time()
                unreachable_since = unreachable_since or now
                if now - unreachable_since > reconnect_budget:
                    raise GPUUnavailable(
                        f"GPU service unreachable for {reconnect_budget:.0f}s while "
                        f"waiting on {job_id}"
                    )
                await asyncio.sleep(poll_interval)
                continue

            if on_progress is not None:
                on_progress(job)
            if job["status"] in ("completed", "failed", "cancelled"):
                return job
            await asyncio.sleep(poll_interval)


__all__ = ["GPUClient", "GPUServiceError", "GPUUnavailable", "DEFAULT_URL"]
