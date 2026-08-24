from abc import ABC, abstractmethod
from typing import List
from app.models import Segment, RenderResponse


class BaseRenderer(ABC):
    @abstractmethod
    def render(self, segments: List[Segment]) -> RenderResponse:
        """Render annotated segments into TTS payload format."""
        pass
