from typing import List
from app.models import Segment, Tone, RenderResponse
from app.renderers.base import BaseRenderer

GEMINI_TONE_DESCRIPTIONS = {
    Tone.NEUTRAL: "น้ำเสียงปกติ เป็นกลาง",
    Tone.SAD: "เศร้า สะเทือนใจ",
    Tone.HAPPY: "ร่าเริง ยิ้มขณะพูด",
    Tone.ANGRY: "โกรธ เสียงแข็ง",
    Tone.EXCITED: "ตื่นเต้น กระตือรือร้น",
    Tone.CALM: "สงบ นุ่มนวล พูดช้า",
    Tone.NERVOUS: "ประหม่า ลังเล",
    Tone.SARCASTIC: "ประชด แดกดัน",
    Tone.SCARED: "หวาดกลัว ตื่นตระหนก",
    Tone.TIRED: "เหนื่อยล้า อ่อนเพลีย",
}

INTENSITY_MODIFIERS_THAI = {
    1: "เล็กน้อย",
    2: "",
    3: "อย่างมาก/เข้มข้น",
}


class GeminiRenderer(BaseRenderer):
    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None)

        clean_text = "".join(seg.text for seg in segments)

        # Distinct non-neutral tones
        non_neutral_tones = {seg.tone for seg in segments if seg.tone != Tone.NEUTRAL}

        if not non_neutral_tones:
            prompt = "อ่านด้วยน้ำเสียงปกติ เป็นกลาง"
            return RenderResponse(text=clean_text, prompt=prompt)

        # If only 1 segment or all segments share the same tone
        tones = {seg.tone for seg in segments}
        if len(tones) == 1:
            tone = list(tones)[0]
            desc = GEMINI_TONE_DESCRIPTIONS.get(tone, "น้ำเสียงปกติ เป็นกลาง")
            intensity_mod = INTENSITY_MODIFIERS_THAI.get(segments[0].intensity, "")
            if intensity_mod:
                prompt = f"อ่านด้วยน้ำเสียง{desc} {intensity_mod}".strip()
            else:
                prompt = f"อ่านด้วยน้ำเสียง{desc}"
            return RenderResponse(text=clean_text, prompt=prompt)

        # Multi-segment prompt construction
        parts = []
        for idx, seg in enumerate(segments, 1):
            desc = GEMINI_TONE_DESCRIPTIONS.get(seg.tone, "น้ำเสียงปกติ เป็นกลาง")
            mod = INTENSITY_MODIFIERS_THAI.get(seg.intensity, "")
            mod_str = f" ({mod})" if mod else ""
            cleaned_clause = seg.text.strip()
            parts.append(f"ส่วนที่ {idx} \"{cleaned_clause}\" ให้อ่านด้วยน้ำเสียง{desc}{mod_str}")

        prompt = "อ่านข้อความโดยปรับอารมณ์ตามแต่ละส่วนดังนี้:\n" + "\n".join(parts)

        return RenderResponse(
            text=clean_text,
            prompt=prompt
        )
