"""Persistent SeedVC voice-conversion worker (runs in the seed-vc venv).

The studio (:8011) generates emotional Thai speech with Thonburian, then posts each
chunk here to be re-timbred onto the target speaker's voice. SeedVC's own
``inference.py`` reloads the whole model on every call; ``SeedVCWrapper`` loads it
once, so this wraps that in a tiny HTTP server the studio can call per chunk.

Runs in seed-vc's own virtualenv (torch 2.4), which cannot coexist with the
studio's env — hence a separate process, not an import. Start it with that venv's
python and the seed-vc checkout on the path:

    <seedvc-venv>/python tools/seedvc_server.py \
        --seedvc-repo <path-to-seed-vc> --port 8022

POST /convert  {source, target, output, f0_condition, auto_f0_adjust,
                diffusion_steps, semi_tone_shift}  -> writes `output`, returns
                {"output":..., "sample_rate":...}. Paths are absolute and local;
this binds to localhost and is trusted, like the sibling GPU service.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pydantic import BaseModel


class ConvertRequest(BaseModel):
    source: str
    target: str
    output: str
    f0_condition: bool = True
    auto_f0_adjust: bool = True
    diffusion_steps: int = 25
    semi_tone_shift: int = 0
    inference_cfg_rate: float = 0.7


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seedvc-repo", required=True, help="path to a seed-vc checkout")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8022)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    repo = Path(args.seedvc_repo).resolve()
    if not (repo / "seed_vc_wrapper.py").exists():
        print(f"[seedvc] {repo} is not a seed-vc checkout", file=sys.stderr)
        return 2
    # seed_vc_wrapper imports `modules.commons` etc. by relative name.
    sys.path.insert(0, str(repo))

    import numpy as np
    import soundfile as sf
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from seed_vc_wrapper import SeedVCWrapper

    print(f"[seedvc] loading SeedVCWrapper (device={args.device or 'auto'}) …", flush=True)
    t0 = time.time()
    wrapper = SeedVCWrapper(device=args.device)
    print(f"[seedvc] ready in {time.time()-t0:.0f}s on {wrapper.device}", flush=True)

    app = FastAPI(title="SeedVC worker", version="1.0.0")

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "device": str(wrapper.device)})

    @app.post("/convert")
    def convert(req: ConvertRequest) -> JSONResponse:
        for p in (req.source, req.target):
            if not Path(p).exists():
                return JSONResponse({"error": f"missing file: {p}"}, status_code=400)
        try:
            # stream_output=False returns one numpy array (the whole take).
            audio = wrapper.convert_voice(
                source=req.source,
                target=req.target,
                diffusion_steps=req.diffusion_steps,
                inference_cfg_rate=req.inference_cfg_rate,
                f0_condition=req.f0_condition,
                auto_f0_adjust=req.auto_f0_adjust,
                pitch_shift=req.semi_tone_shift,
                stream_output=False,
            )

            # f0-conditioned models run at 44.1k, the base models at 22.05k.
            sr = 44100 if req.f0_condition else 22050
            audio = np.asarray(audio, dtype="float32").squeeze()
            Path(req.output).parent.mkdir(parents=True, exist_ok=True)
            sf.write(req.output, audio, sr, subtype="PCM_16")
            return JSONResponse({"output": req.output, "sample_rate": sr, "frames": int(audio.size)})
        except Exception as e:                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
