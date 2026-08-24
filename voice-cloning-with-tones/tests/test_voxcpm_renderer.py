import pytest
from app.models import Segment, Tone
from app.renderers.voxcpm import (
    VoxCPMRenderer,
    format_voxcpm_instruction,
    split_style_chunks,
    parse_tagged_segments,
    collect_tag_warnings,
    resolve_style_tag,
    split_style_chunk_specs,
    STYLE_VOCABULARY,
    VOXCPM_INSTRUCTION_MAP,
)
from app.renderers import get_renderer


def test_voxcpm_renderer_all_neutral():
    renderer = VoxCPMRenderer()
    segments = [
        Segment(text="สวัสดีครับ ", tone=Tone.NEUTRAL, intensity=2),
        Segment(text="ยินดีต้อนรับครับ", tone=Tone.NEUTRAL, intensity=2)
    ]
    res = renderer.render(segments)
    assert res.text == "สวัสดีครับ ยินดีต้อนรับครับ"
    assert "(" not in res.text
    assert ")" not in res.text


def test_voxcpm_renderer_calm_prompt():
    renderer = VoxCPMRenderer()
    segments = [
        Segment(text="หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ", tone=Tone.CALM, intensity=2)
    ]
    res = renderer.render(segments)
    # VoxCPM2's documented format puts no space between ')' and the content.
    instr = format_voxcpm_instruction(Tone.CALM, 2)
    assert res.text.startswith(f"{instr}หายใจเข้าลึกๆ")


@pytest.mark.parametrize("tone,intensity,expected_substr", [
    (Tone.CALM, 1, "Slightly calm"),
    (Tone.CALM, 2, "Calm and soothing"),
    (Tone.CALM, 3, "Deeply calm"),
    (Tone.SAD, 2, "Sad voice, quiet and downcast"),
    (Tone.ANGRY, 2, "Angry, firm and aggressive"),
    (Tone.HAPPY, 2, "Happy and cheerful"),
    (Tone.EXCITED, 2, "Excited and energetic"),
    (Tone.NERVOUS, 2, "Nervous and trembling"),
    (Tone.SARCASTIC, 2, "Sarcastic and mocking"),
])
def test_voxcpm_renderer_tones_and_intensities(tone, intensity, expected_substr):
    instr = format_voxcpm_instruction(tone, intensity)
    assert instr is not None
    assert expected_substr.lower() in instr.lower()


# Wordings measured to make VoxCPM2 drop out of control mode and *speak* the
# direction ahead of the line, in English, instead of obeying it. Nothing detects
# that downstream: the audio arrives, no error is raised, and the prosody metrics
# still report plausible numbers because the leaked English is real speech. The
# trigger is the exact phrasing rather than any one word -- tired@3 carries "heavy
# sighs" and never leaked -- so this is a list of known-bad strings, not a rule.
# Add to it from tools/instruction_leak_audit.py rather than by intuition.
KNOWN_LEAKING_INSTRUCTIONS = {
    # Leaked 6 takes out of 6 against a pinned speaker; Whisper transcribed
    # "Sad and Melancholic Voice Slight Sighs" ahead of the Thai, +2.5 s of audio.
    "(Sad and melancholic voice, slight sighs)",
    # Leaked 4 takes in 30 -- "Heavy, Sarcastic and Cynical" -- sometimes garbling
    # the Thai behind it. Intermittent is the harder case: one run of 6 came back
    # clean, so only a multi-rep audit finds it. The replacement went 0 in 21.
    "(Heavy sarcastic and cynical tone)",
}


def test_no_known_leaking_instructions():
    """No shipped direction may be one VoxCPM2 is known to read aloud."""
    shipped = {
        instruction
        for by_intensity in VOXCPM_INSTRUCTION_MAP.values()
        for instruction in by_intensity.values()
        if instruction
    } | {
        instruction for instruction, _family in STYLE_VOCABULARY.values() if instruction
    }
    assert not (shipped & KNOWN_LEAKING_INSTRUCTIONS)


def test_voxcpm_renderer_factory():
    renderer = get_renderer("voxcpm")
    assert isinstance(renderer, VoxCPMRenderer)
    renderer_siangtts = get_renderer("siangtts")
    assert isinstance(renderer_siangtts, VoxCPMRenderer)


def test_voxcpm_renderer_multi_segment_transitions():
    """Each tone run becomes its own chunk with the instruction leading it.

    VoxCPM2 only honours a style parenthetical at position 0; one appearing mid-text
    gets spoken aloud, so res.text must carry only the opening instruction.
    """
    renderer = VoxCPMRenderer()
    segments = [
        Segment(text="ขอโทษนะ ฉันไม่ได้ตั้งใจ ", tone=Tone.SAD, intensity=2),
        Segment(text="แต่เธอก็ไม่ฟังฉันเลย", tone=Tone.ANGRY, intensity=2)
    ]
    res = renderer.render(segments)

    sad_instr = format_voxcpm_instruction(Tone.SAD, 2)
    angry_instr = format_voxcpm_instruction(Tone.ANGRY, 2)

    assert len(res.chunks) == 2
    assert res.chunks[0].instruction == sad_instr
    assert res.chunks[1].instruction == angry_instr
    for chunk in res.chunks:
        assert chunk.text.startswith(chunk.instruction)
        assert "(" not in chunk.body

    # Single-shot rendering: leading instruction only, nothing mid-utterance.
    assert res.text.startswith(sad_instr)
    assert angry_instr not in res.text
    assert res.text.count("(") == 1


def test_voxcpm_renderer_merges_consecutive_same_tone():
    renderer = VoxCPMRenderer()
    segments = [
        Segment(text="ดีใจมากเลย ", tone=Tone.HAPPY, intensity=2),
        Segment(text="ขอบคุณนะ", tone=Tone.HAPPY, intensity=2),
    ]
    res = renderer.render(segments)
    assert len(res.chunks) == 1
    assert res.chunks[0].body == "ดีใจมากเลย ขอบคุณนะ"


# ---------------------------------------------------------------------------
# split_style_chunks -- hand-written inline style tags
# ---------------------------------------------------------------------------

def test_split_style_chunks_splits_on_mid_text_tag():
    """A tag typed mid-text becomes its own chunk instead of being spoken aloud."""
    text = "(Excited and energetic tone)ของดีมากครับ\n(sad)แต่ของหมดแล้ว"
    chunks = split_style_chunks(text)
    assert len(chunks) == 2
    assert chunks[0] == "(Excited and energetic tone)ของดีมากครับ"
    # A bare tone name expands to the canonical wording, which measured far better
    # than passing "(sad)" through raw.
    assert chunks[1] == f"{format_voxcpm_instruction(Tone.SAD, 2)}แต่ของหมดแล้ว"


def test_split_style_chunks_returns_empty_without_tags():
    """No tag means the caller should fall back to LLM annotation."""
    assert split_style_chunks("สวัสดีครับ วันนี้อากาศดีมาก") == []


def test_split_style_chunks_ignores_thai_parentheses():
    """Thai inside brackets is spoken content, not a style instruction."""
    assert split_style_chunks("ราคา (พิเศษ) วันนี้เท่านั้น") == []


def test_split_style_chunks_keeps_untagged_lead_text():
    chunks = split_style_chunks("เริ่มก่อน(Angry)แล้วโกรธ")
    assert chunks == ["เริ่มก่อน", f"{format_voxcpm_instruction(Tone.ANGRY, 2)}แล้วโกรธ"]


def test_split_style_chunks_single_leading_tag_is_one_chunk():
    text = "(Calm and soothing voice)หายใจเข้าลึกๆ"
    assert split_style_chunks(text) == [text]


def test_split_style_chunks_drops_tag_with_empty_body():
    assert split_style_chunks("(Angry)   ") == []


# ---------------------------------------------------------------------------
# Square-bracket tags -- the form the UI actually documents
# ---------------------------------------------------------------------------

def test_square_bracket_tags_are_recognised():
    """The UI advertises [emotion]; only (emotion) used to match, so tags were spoken."""
    text = "[sad]ประโยคแรก\n[happy]ประโยคที่สอง"
    chunks = split_style_chunks(text)
    assert chunks == [
        f"{format_voxcpm_instruction(Tone.SAD, 2)}ประโยคแรก",
        f"{format_voxcpm_instruction(Tone.HAPPY, 2)}ประโยคที่สอง",
    ]
    for chunk in chunks:
        assert "[" not in chunk and "]" not in chunk


def test_bracket_tag_intensity_suffix():
    assert split_style_chunks("[sad:3]ทดสอบ") == [
        f"{format_voxcpm_instruction(Tone.SAD, 3)}ทดสอบ"
    ]


def test_parse_tagged_segments_reports_tone_and_intensity():
    segs = parse_tagged_segments("[sad]ประโยคแรก\n[happy:1]ประโยคที่สอง")
    assert [(s.tone, s.intensity, s.text) for s in segs] == [
        (Tone.SAD, 2, "ประโยคแรก"),
        (Tone.HAPPY, 1, "ประโยคที่สอง"),
    ]


def test_parse_tagged_segments_empty_without_tags():
    assert parse_tagged_segments("สวัสดีครับ วันนี้อากาศดี") == []


def test_thai_in_square_brackets_is_content_not_a_tag():
    assert split_style_chunks("ราคา [พิเศษ] วันนี้") == []


# ---------------------------------------------------------------------------
# Vocabulary beyond the original eight tones
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag,expected", [
    ("scared", Tone.SCARED),
    ("tired", Tone.TIRED),
    ("afraid", Tone.SCARED),
    ("sleepy", Tone.TIRED),
    ("anxious", Tone.NERVOUS),
    ("furious", Tone.ANGRY),
])
def test_extended_vocabulary_resolves_to_a_real_tone(tag, expected):
    """These used to fall through the enum lookup and display as NEUTRAL."""
    segs = parse_tagged_segments(f"[{tag}]ข้อความ")
    assert len(segs) == 1
    assert segs[0].tone is expected


@pytest.mark.parametrize("tone", [Tone.SCARED, Tone.TIRED])
def test_new_tones_have_instructions_at_every_intensity(tone):
    for level in (1, 2, 3):
        instr = format_voxcpm_instruction(tone, level)
        assert instr and instr.startswith("(") and instr.endswith(")")


def test_scared_tag_expands_to_full_instruction_not_bare_word():
    chunks = split_style_chunks("[scared]ข้อความ")
    assert chunks == [f"{format_voxcpm_instruction(Tone.SCARED, 2)}ข้อความ"]
    assert chunks[0] != "(scared)ข้อความ"


def test_literal_backslash_n_is_not_spoken():
    """Pasted text often carries "\n" as two characters rather than a line break."""
    chunks = split_style_chunks("[sad]ข้อความ\n\n")
    assert chunks == [f"{format_voxcpm_instruction(Tone.SAD, 2)}ข้อความ"]
    assert "\n" not in chunks[0]


def test_unknown_word_still_reaches_the_model_as_direction():
    """An unrecognised word is passed through verbatim rather than dropped."""
    chunks = split_style_chunks("[bewildered]ข้อความ")
    assert chunks == ["(bewildered)ข้อความ"]
    assert parse_tagged_segments("[bewildered]ข้อความ")[0].tone is Tone.NEUTRAL


# ---------------------------------------------------------------------------
# Open style vocabulary
# ---------------------------------------------------------------------------

def test_every_vocabulary_entry_yields_an_instruction():
    """No entry may silently resolve to nothing.

    NEUTRAL is the one exception: it carries no instruction on purpose, because
    "speak neutrally" is just plain text with no leading parenthetical.
    """
    for word in STYLE_VOCABULARY:
        tag = resolve_style_tag(word)
        assert tag.warning is None, word
        assert tag.label == word
        if tag.tone is Tone.NEUTRAL and tag.instruction is None:
            continue
        assert tag.instruction, word
        assert tag.instruction.startswith("(") and tag.instruction.endswith(")"), word


def test_neutral_synonyms_carry_no_instruction():
    for word in ("normal", "plain", "flat"):
        tag = resolve_style_tag(word)
        assert tag.tone is Tone.NEUTRAL
        assert tag.instruction is None
    assert split_style_chunks("[normal]ข้อความ") == ["ข้อความ"]


def test_style_label_survives_to_the_segment():
    """The UI shows the word typed; tone stays the coarse colour family."""
    seg = parse_tagged_segments("[appalled]ข้อความ")[0]
    assert seg.style == "appalled"
    assert seg.tone is Tone.ANGRY


def test_distinct_styles_get_distinct_instructions():
    a = resolve_style_tag("thoughtful").instruction
    b = resolve_style_tag("curious").instruction
    c = resolve_style_tag("whispering").instruction
    assert len({a, b, c}) == 3


@pytest.mark.parametrize("tag", ["laughs", "sighs", "applause", "gunshot", "sings"])
def test_unsupported_tags_warn_and_send_no_instruction(tag):
    """VoxCPM2 swallows these silently, so say so rather than implying they worked."""
    resolved = resolve_style_tag(tag)
    assert resolved.warning is not None
    assert resolved.instruction is None

    text = f"[{tag}]ข้อความ"
    # Body is spoken plainly -- no dead tag riding along.
    assert split_style_chunks(text) == ["ข้อความ"]
    warnings = collect_tag_warnings(text)
    assert len(warnings) == 1 and tag in warnings[0]


def test_supported_tags_produce_no_warnings():
    assert collect_tag_warnings("[sad]ก[happy]ข[appalled]ค") == []


def test_warnings_are_deduplicated():
    assert len(collect_tag_warnings("[laughs]ก[laughs]ข")) == 1


# ---------------------------------------------------------------------------
# Short script form (what the studio's editable box holds)
# ---------------------------------------------------------------------------

def test_script_uses_short_tags_not_instructions():
    """The editable box shows '[sad] ...', not '(Sad and melancholic voice...)...'."""
    res = VoxCPMRenderer().render(parse_tagged_segments("[sad] หนึ่ง [happy] สอง"))
    assert res.script == "[sad] หนึ่ง [happy] สอง"
    assert "(" not in res.script


def test_script_round_trips_every_emotion():
    """The regression this exists for.

    data.text carries only the FIRST instruction followed by every body, so feeding
    it back to /synthesize collapsed a four-emotion script into one tone.
    """
    source = "[sad] หนึ่ง\n[happy] สอง [scared]สาม\n[tired]สี่"
    res = VoxCPMRenderer().render(parse_tagged_segments(source))

    assert [s.tone for s in split_style_chunk_specs(res.script)] == [
        "sad", "happy", "scared", "tired",
    ]
    # data.text, by contrast, cannot round-trip -- it is a single-shot rendering.
    assert len(split_style_chunk_specs(res.text)) == 1


def test_script_preserves_line_breaks_that_earn_the_longer_pause():
    source = "[sad] หนึ่ง\n[happy] สอง [scared]สาม"
    res = VoxCPMRenderer().render(parse_tagged_segments(source))
    assert [s.break_before for s in split_style_chunk_specs(res.script)] == [
        False, True, False,
    ]


def test_script_leaves_an_untagged_opening_untagged():
    res = VoxCPMRenderer().render(parse_tagged_segments("นำเรื่อง [sad] เศร้า"))
    assert res.script == "นำเรื่อง [sad] เศร้า"


def test_script_writes_intensity_only_when_it_is_not_the_default():
    segs = [
        Segment(text="ก", tone=Tone.SAD, intensity=2, style="sad"),
        Segment(text="ข", tone=Tone.SAD, intensity=3, style="sad"),
    ]
    assert VoxCPMRenderer().render(segs).script == "[sad] ก [sad:3] ข"


def test_script_keeps_the_specific_style_word():
    segs = [Segment(text="ก", tone=Tone.ANGRY, intensity=2, style="appalled")]
    assert VoxCPMRenderer().render(segs).script == "[appalled] ก"


def test_script_from_llm_segments_has_no_style_words():
    """The LLM path sets tone but never style, so the tone name is the tag."""
    segs = [
        Segment(text="ก", tone=Tone.NEUTRAL, intensity=2),
        Segment(text="ข", tone=Tone.EXCITED, intensity=2),
    ]
    assert VoxCPMRenderer().render(segs).script == "ก [excited] ข"


def test_happily_button_tag_resolves_to_the_full_happy_instruction():
    """The studio inserts the ElevenLabs spelling; a bare '(happily)' measured weak."""
    assert resolve_style_tag("happily").instruction == format_voxcpm_instruction(Tone.HAPPY, 2)


def test_mixed_direction_takes_the_family_with_the_most_words():
    """'scared and crying, tearful' is a tearful read, not a +3 dB scared one."""
    assert resolve_style_tag("scared and crying, tearful", use_llm=False).tone == Tone.SAD


def test_unknown_tag_falls_back_to_canonical_tone_when_llm_unavailable():
    """When LLM is unavailable, unknown multi-word phrases keep verbatim and single word gets canonical instruction."""
    resolved = resolve_style_tag("scared and crying, tearful", use_llm=False)
    assert resolved.tone == Tone.SAD
    assert resolved.instruction == "(scared and crying, tearful)"



def test_unknown_tag_converts_via_llm(monkeypatch):
    """When LLM succeeds, unknown tags get the structured VoxCPM2 instruction from LLM."""
    from app.annotator import annotator
    from app.renderers.voxcpm import clear_dynamic_style_cache

    clear_dynamic_style_cache()
    monkeypatch.setattr(
        annotator,
        "convert_style_tag",
        lambda tag, intensity=2, custom_model=None: {
            "instruction": "(Custom weeping and desperate voice, shaking)",
            "tone": Tone.SAD,
            "intensity": 3,
        }
    )

    resolved = resolve_style_tag("very desperate sobbing", level="3", use_llm=True)
    assert resolved.instruction == "(Custom weeping and desperate voice, shaking)"
    assert resolved.tone == Tone.SAD
    assert resolved.intensity == 3

    # Check that it is cached
    monkeypatch.setattr(annotator, "convert_style_tag", lambda *args, **kwargs: None)
    cached = resolve_style_tag("very desperate sobbing", level="3", use_llm=True)
    assert cached.instruction == "(Custom weeping and desperate voice, shaking)"


def test_crying_and_tearful_and_sad_and_cry_resolve_to_proper_instruction():
    assert resolve_style_tag("crying and tearful").instruction == "(Crying voice, broken and tearful, trembling)"
    assert resolve_style_tag("sad and cry").instruction == "(Deeply sorrowful and crying voice, trembling)"


def test_a_tie_keeps_the_earliest_word():
    assert resolve_style_tag("happy and excited").tone == Tone.HAPPY
    assert resolve_style_tag("excited and happy").tone == Tone.EXCITED


def test_single_emotion_direction_is_unchanged_by_voting():
    for word, expected in (("scared", Tone.SCARED), ("crying", Tone.SAD),
                           ("appalled", Tone.ANGRY), ("bored", Tone.TIRED),
                           ("somber", Tone.SAD), ("gloomy", Tone.SAD)):
        assert resolve_style_tag(word).tone == expected


def test_somber_and_gloomy_resolve_to_proper_instruction():
    assert "Somber and melancholic" in resolve_style_tag("somber").instruction
    assert "Gloomy and despondent" in resolve_style_tag("gloomy").instruction
    segs = parse_tagged_segments("[somber] วันนี้อากาศมืดมน")
    assert len(segs) == 1
    assert segs[0].tone == Tone.SAD


def test_voxcpm_renderer_uses_spoken_text():
    renderer = VoxCPMRenderer()
    segments = [
        Segment(text="ฉันบอกแล้ว ", spoken_text="ฉันบอกแล้ว! ", tone=Tone.ANGRY, intensity=2),
        Segment(text="ทำไมไม่ฟัง", spoken_text="ทำไมไม่ฟัง?!", tone=Tone.ANGRY, intensity=2),
    ]
    res = renderer.render(segments)
    assert len(res.chunks) == 1
    assert res.chunks[0].body == "ฉันบอกแล้ว! ทำไมไม่ฟัง?!"
    assert "ฉันบอกแล้ว! ทำไมไม่ฟัง?!" in res.text
    assert res.script == "[angry] ฉันบอกแล้ว! ทำไมไม่ฟัง?!"


def test_thai_emotion_tags_resolve_properly():
    chunks = split_style_chunks("[โกรธ]ทำไมทำแบบนี้\n[ดีใจ:1]ขอบคุณมากนะ\n[เศร้า:3]เสียใจจัง\n[กระซิบ]เบาๆ นะ")
    assert len(chunks) == 4
    assert chunks[0] == f"{format_voxcpm_instruction(Tone.ANGRY, 2)}ทำไมทำแบบนี้"
    assert chunks[1] == f"{format_voxcpm_instruction(Tone.HAPPY, 1)}ขอบคุณมากนะ"
    assert chunks[2] == f"{format_voxcpm_instruction(Tone.SAD, 3)}เสียใจจัง"
    assert chunks[3] == "(Whispering voice, very soft and breathy)เบาๆ นะ"


def test_prepare_text_preserves_parenthetical_instruction():
    from app.services.siangtts_service import prepare_text

    text = "(Happy and cheerful voice, smiling while speaking)สวัสดีครับ... วันนี้อากาศดีมาก"
    res = prepare_text(text)
    assert res.startswith("(Happy and cheerful voice, smiling while speaking)")
    assert "สวัสดีครับ" in res



