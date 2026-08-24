from typing import List, Optional
from app.models import Segment, Tone, RenderResponse
from app.renderers.base import BaseRenderer
from app.config import settings

ELEVENLABS_TAG_MAP = {
    Tone.NEUTRAL: None,
    Tone.SAD: "sad",
    Tone.HAPPY: "happily",
    Tone.ANGRY: "angry",
    Tone.EXCITED: "excited",
    Tone.CALM: "calm",
    Tone.NERVOUS: "nervous",
    Tone.SARCASTIC: "sarcastic",
    Tone.SCARED: "scared",
    Tone.TIRED: "tired",
}


def format_tag(tone: Tone, intensity: int) -> Optional[str]:
    """Format ElevenLabs audio tag with intensity modifier."""
    base_tag = ELEVENLABS_TAG_MAP.get(tone)
    if not base_tag or tone == Tone.NEUTRAL:
        return None

    if intensity == 1:
        return f"[slightly {base_tag}] "
    elif intensity == 3:
        return f"[very {base_tag}] "
    else:  # intensity == 2
        return f"[{base_tag}] "


def _split_for_reanchor(text: str, limit: int) -> List[str]:
    """Break text into <=limit-char pieces at Thai word boundaries.

    Thai does not space its words, so slicing on a raw character count dropped the
    tag inside a word -- and, when the cut landed between a consonant and its
    combining vowel or tone mark, split a single grapheme in half. Packing whole
    tokens avoids both. A token longer than the limit is emitted on its own rather
    than being cut.
    """
    from pythainlp.tokenize import word_tokenize

    try:
        tokens = word_tokenize(text, keep_whitespace=True)
    except Exception:
        # Never fail a render over tokenization; one un-anchored piece is fine.
        return [text]

    parts: List[str] = []
    current = ""
    for tok in tokens:
        if current and len(current) + len(tok) > limit:
            parts.append(current)
            current = tok
        else:
            current += tok
    if current:
        parts.append(current)
    return parts or [text]


class ElevenLabsRenderer(BaseRenderer):
    def __init__(self, reanchor_chars: Optional[int] = None):
        self.reanchor_chars = reanchor_chars if reanchor_chars is not None else settings.reanchor_chars

    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None)

        out = []
        prev_tone: Optional[Tone] = None

        for seg in segments:
            # Tag only inserted when tone changes and tone is not neutral
            if seg.tone != prev_tone and seg.tone != Tone.NEUTRAL:
                tag_str = format_tag(seg.tone, seg.intensity)
                if tag_str:
                    out.append(tag_str)
            
            # Optional re-anchoring for very long segments if configured
            if self.reanchor_chars and len(seg.text) > self.reanchor_chars and seg.tone != Tone.NEUTRAL:
                tag_str = format_tag(seg.tone, seg.intensity)
                out.append(f"{tag_str}".join(_split_for_reanchor(seg.text, self.reanchor_chars)))
            else:
                out.append(seg.text)

            prev_tone = seg.tone

        return RenderResponse(
            text="".join(out),
            prompt=None
        )
