import pytest
from app.models import Segment, Tone
from app.renderers.elevenlabs import ElevenLabsRenderer
from app.renderers.gemini import GeminiRenderer


def test_renderer_elevenlabs_all_neutral_has_no_tags():
    """All neutral segments produce zero audio tags."""
    renderer = ElevenLabsRenderer()
    segments = [
        Segment(text="ประโยคที่หนึ่ง ", tone=Tone.NEUTRAL, intensity=2),
        Segment(text="ประโยคที่สอง", tone=Tone.NEUTRAL, intensity=2)
    ]
    res = renderer.render(segments)
    assert "[" not in res.text
    assert "]" not in res.text
    assert res.text == "ประโยคที่หนึ่ง ประโยคที่สอง"
    assert res.prompt is None


def test_renderer_elevenlabs_single_tone_one_tag_at_start():
    """Single tone across segments has exactly one tag at the very start."""
    renderer = ElevenLabsRenderer()
    segments = [
        Segment(text="ขอโทษนะ ", tone=Tone.SAD, intensity=2),
        Segment(text="ฉันเสียใจจริงๆ", tone=Tone.SAD, intensity=2)
    ]
    res = renderer.render(segments)
    assert res.text == "[sad] ขอโทษนะ ฉันเสียใจจริงๆ"
    assert res.text.count("[sad]") == 1


def test_renderer_elevenlabs_tone_transition():
    """sad -> sad -> angry produces exactly 2 tags, not 3."""
    renderer = ElevenLabsRenderer()
    segments = [
        Segment(text="ขอโทษนะ ", tone=Tone.SAD, intensity=2),
        Segment(text="ฉันไม่ได้ตั้งใจ ", tone=Tone.SAD, intensity=2),
        Segment(text="แต่เธอก็ไม่ฟังฉันเลย", tone=Tone.ANGRY, intensity=2)
    ]
    res = renderer.render(segments)
    expected = "[sad] ขอโทษนะ ฉันไม่ได้ตั้งใจ [angry] แต่เธอก็ไม่ฟังฉันเลย"
    assert res.text == expected


@pytest.mark.parametrize("intensity,expected_prefix", [
    (1, "[slightly sad] "),
    (2, "[sad] "),
    (3, "[very sad] ")
])
def test_renderer_elevenlabs_intensity_modifiers(intensity, expected_prefix):
    """Intensity 1/2/3 yields slightly / standard / very."""
    renderer = ElevenLabsRenderer()
    segments = [Segment(text="ข้อความ", tone=Tone.SAD, intensity=intensity)]
    res = renderer.render(segments)
    assert res.text.startswith(expected_prefix)
    assert res.text == f"{expected_prefix}ข้อความ"


def test_renderer_elevenlabs_space_after_bracket():
    """Ensure there is always a space after ] tag before Thai text."""
    renderer = ElevenLabsRenderer()
    for tone in [Tone.HAPPY, Tone.ANGRY, Tone.CALM, Tone.NERVOUS, Tone.SARCASTIC, Tone.EXCITED]:
        segments = [Segment(text="ข้อความทดสอบ", tone=tone, intensity=2)]
        res = renderer.render(segments)
        assert "] " in res.text
        assert not res.text.startswith("]ข้อความ")


def test_renderer_gemini_neutral():
    """Gemini renderer returns clean text and neutral prompt."""
    renderer = GeminiRenderer()
    segments = [
        Segment(text="สวัสดีครับ ", tone=Tone.NEUTRAL, intensity=2),
        Segment(text="วันนี้มีข่าวสารมาแจ้งครับ", tone=Tone.NEUTRAL, intensity=2)
    ]
    res = renderer.render(segments)
    assert res.text == "สวัสดีครับ วันนี้มีข่าวสารมาแจ้งครับ"
    assert "เป็นกลาง" in res.prompt


def test_renderer_gemini_multi_tone_prompt():
    """Gemini renderer builds Thai descriptive prompt for multiple tones."""
    renderer = GeminiRenderer()
    segments = [
        Segment(text="ขอโทษนะ ฉันไม่ได้ตั้งใจ ", tone=Tone.SAD, intensity=2),
        Segment(text="แต่เธอก็ไม่ฟังฉันเลย", tone=Tone.ANGRY, intensity=2)
    ]
    res = renderer.render(segments)
    assert res.text == "ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย"
    assert "เศร้า สะเทือนใจ" in res.prompt
    assert "โกรธ เสียงแข็ง" in res.prompt
