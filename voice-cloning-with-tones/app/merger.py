from typing import List, Optional
from app.models import Segment


def _key(seg: Segment) -> tuple:
    """What makes two segments the same voice direction.

    Tone alone is too coarse: two hand-written styles can share a family
    ("appalled" and "annoyed" are both angry) while resolving to different VoxCPM2
    instructions, and merging them would silently discard one of them.
    """
    return (seg.tone, seg.style)


def _collapse(segments: List[Segment]) -> List[Segment]:
    """Merge runs of adjacent segments that carry the same direction."""
    if not segments:
        return []

    out: List[Segment] = []
    text = segments[0].text
    has_spoken = segments[0].spoken_text is not None
    spoken_text = segments[0].spoken_text or segments[0].text
    tone = segments[0].tone
    style = segments[0].style
    intensity = segments[0].intensity
    break_before = segments[0].break_before

    for seg in segments[1:]:
        if _key(seg) == (tone, style):
            text += seg.text
            if seg.spoken_text is not None:
                has_spoken = True
            spoken_text += (seg.spoken_text or seg.text)
            intensity = max(intensity, seg.intensity)
        else:
            out.append(
                Segment(
                    text=text,
                    spoken_text=spoken_text if has_spoken else None,
                    tone=tone,
                    intensity=intensity,
                    style=style,
                    break_before=break_before,
                )
            )
            text = seg.text
            has_spoken = seg.spoken_text is not None
            spoken_text = seg.spoken_text or seg.text
            tone, style, intensity, break_before = seg.tone, seg.style, seg.intensity, seg.break_before

    out.append(
        Segment(
            text=text,
            spoken_text=spoken_text if has_spoken else None,
            tone=tone,
            intensity=intensity,
            style=style,
            break_before=break_before,
        )
    )
    return out


def merge_segments(segments: List[Segment], max_segments: int = 20) -> List[Segment]:
    """
    Merge consecutive segments that have the same tone (and style).
    The resulting intensity is the maximum intensity of the group.

    If the number of merged segments exceeds max_segments,
    further merge adjacent segments until <= max_segments.
    """
    if not segments:
        return []

    merged = _collapse(segments)

    # Step 2: Enforce max_segments limit if needed
    if max_segments > 0 and len(merged) > max_segments:
        while len(merged) > max_segments:
            # Find the shortest segment pair to merge
            best_idx = 0
            min_combined_len = float("inf")
            for i in range(len(merged) - 1):
                combined_len = len(merged[i].text) + len(merged[i + 1].text)
                if combined_len < min_combined_len:
                    min_combined_len = combined_len
                    best_idx = i

            first = merged[best_idx]
            second = merged[best_idx + 1]

            # Combine them (keep the tone of the more intense or first, max intensity)
            chosen_tone = first.tone if first.intensity >= second.intensity else second.tone
            # A forced merge across two different styles has no single style left to
            # name; falling back to the tone family is the honest answer.
            chosen_style: Optional[str] = first.style if first.style == second.style else None

            first_spoken = first.spoken_text or first.text
            second_spoken = second.spoken_text or second.text
            has_spoken_combined = first.spoken_text is not None or second.spoken_text is not None

            combined_segment = Segment(
                text=first.text + second.text,
                spoken_text=(first_spoken + second_spoken) if has_spoken_combined else None,
                tone=chosen_tone,
                intensity=max(first.intensity, second.intensity),
                style=chosen_style,
                break_before=first.break_before,
            )

            # Replace the two with the merged one
            merged = merged[:best_idx] + [combined_segment] + merged[best_idx + 2:]

            # Re-merge identical consecutive directions if the swap created any
            merged = _collapse(merged)

    return merged
