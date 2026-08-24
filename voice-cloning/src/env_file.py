"""Read `.env` from the working directory into the environment.

Deliberately not python-dotenv: this is a handful of lines, and adding a dependency
to the serving path for it was not worth it.

Real environment variables win — a value already in `os.environ` is never
overwritten, so `SIANGTTS_GPU_STUB=1 uvicorn …` still does what it says on a host
whose `.env` says otherwise.

It lives in its own module because both halves of the split need it: the webhook
reads upload credentials from `.env`, and the GPU service reads the adapter and
reference directories from the same file. It used to be private to `pipeline`, which
the GPU service has no other reason to import — and without it that service came up
pointing at the default `ref/` instead of the reference directory production
actually uses, with every voice missing and nothing to say why.

The file is resolved against the **current working directory**, not this package, so
a service started from the wrong folder silently gets no configuration. That is why
the deployment notes insist on `AppDirectory`.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> int:
    """Load `KEY=value` lines. Returns how many variables were set."""
    env_path = Path(path)
    if not env_path.exists():
        return 0

    loaded = 0
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            if k and k not in os.environ:
                os.environ[k] = v
                loaded += 1
    return loaded


__all__ = ["load_env_file"]
