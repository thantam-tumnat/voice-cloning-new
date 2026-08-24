from typing import Literal
from app.renderers.base import BaseRenderer


def get_renderer(engine: Literal["elevenlabs", "gemini", "voxcpm", "siangtts"]) -> BaseRenderer:
    """Factory to get the appropriate renderer instance.

    Imports the concrete renderer here rather than at module scope. Pulling all
    three in eagerly meant that reaching *any* of them -- including the VoxCPM
    instruction tables, which are plain data -- first ran app.config, so a consumer
    with no ElevenLabs or Gemini settings could not import the package at all. That
    is every offline consumer of the tables: tools/ and the engine-side venv.
    """
    if engine == "elevenlabs":
        from app.renderers.elevenlabs import ElevenLabsRenderer
        return ElevenLabsRenderer()
    elif engine == "gemini":
        from app.renderers.gemini import GeminiRenderer
        return GeminiRenderer()
    elif engine in ("voxcpm", "siangtts"):
        from app.renderers.voxcpm import VoxCPMRenderer
        return VoxCPMRenderer()
    else:
        raise ValueError(f"Unknown engine: {engine}")


__all__ = ["BaseRenderer", "get_renderer"]
