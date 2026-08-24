import re
from typing import List, NamedTuple, Optional
from app.models import Segment, Tone, RenderResponse, RenderedChunk
from app.renderers.base import BaseRenderer

# A style tag may use either bracket style -- the UI documents "[sad]" while
# VoxCPM2's own format is "(sad)". Supports English and Thai emotion names.
# An optional ":N" sets intensity.
STYLE_TAG_RE = re.compile(
    r"[\[(]\s*([A-Za-z\u0e00-\u0e7f][A-Za-z\u0e00-\u0e7f\s,.\-]*?)\s*(?::\s*([123]))?\s*[\])]"
)

_TONE_BY_NAME = {t.value: t for t in Tone}

# Users paste text where newlines arrived as the two characters "\" + "n" rather
# than as real line breaks. Left alone they end up inside the spoken body.
_LITERAL_ESCAPE_RE = re.compile(r"\\[nrt]")

# A line break immediately before a tag -- real, or arrived as the two characters
# "\" + "n" the same way _LITERAL_ESCAPE_RE handles.
_BREAK_RE = re.compile(r"(?:\r?\n|\\n)[ \t]*$")


class ResolvedTag(NamedTuple):
    instruction: Optional[str]   # what actually leads the text sent to VoxCPM2
    tone: Tone                   # family, for colour and the other renderers
    intensity: int
    label: str                   # what the user typed, for display
    warning: Optional[str]       # set when the tag cannot do anything


def _clean_body(text: str) -> str:
    return _LITERAL_ESCAPE_RE.sub(" ", text).strip()


def _family_from_body(body: str) -> Optional[Tone]:
    """Best-effort family for free-form direction, e.g. 'Sad and...' -> SAD.

    Every recognised word votes, rather than the first one winning outright. The
    family only drives the per-emotion level and pause in audio_post, and reading
    just the opening word got those backwards on a mixed direction: "scared and
    crying, tearful" landed on SCARED, which is +3 dB and *faster*, when what the
    writer asked for is a tearful read. A tie keeps the earliest word, so a plain
    single-emotion direction resolves exactly as before.
    """
    votes: dict[Tone, int] = {}
    order: dict[Tone, int] = {}
    for pos, word in enumerate(re.findall(r"[a-z\u0e00-\u0e7f]+", body.lower())):
        family = _TONE_BY_NAME.get(word)
        if family is None:
            entry = STYLE_VOCABULARY.get(word)
            family = entry[1] if entry is not None else None
        if family is None:
            continue
        votes[family] = votes.get(family, 0) + 1
        order.setdefault(family, pos)
    if not votes:
        return None
    return max(votes, key=lambda f: (votes[f], -order[f]))


_DYNAMIC_STYLE_CACHE: dict[tuple[str, int], ResolvedTag] = {}


def clear_dynamic_style_cache():
    _DYNAMIC_STYLE_CACHE.clear()


def resolve_style_tag(
    body: str,
    level: Optional[str] = None,
    use_llm: bool = False,
    custom_model: Optional[str] = None
) -> ResolvedTag:
    """Turn a raw tag word into everything the pipeline needs from it.

    A bare tone name expands to this module's canonical instruction rather than
    being passed through: measured against a pinned speaker, "(sad)"/"(happy)" gave
    dF0 -15.0 where the full phrasing gave +28.6.

    If the tag is not found in the static dictionary, queries the LLM to convert it into
    a structured VoxCPM2 style instruction when use_llm=True. Falls back to canonical
    tone instruction or verbatim tag.
    """
    raw = body.strip()
    key = raw.lower()
    intensity = int(level) if level else 2

    tone = _TONE_BY_NAME.get(key)
    if tone is not None:
        return ResolvedTag(format_voxcpm_instruction(tone, intensity), tone, intensity, key, None)

    entry = STYLE_VOCABULARY.get(key)
    if entry is not None:
        instruction, family = entry
        return ResolvedTag(
            instruction or format_voxcpm_instruction(family, intensity),
            family, intensity, key, None,
        )

    if key in UNSUPPORTED_TAGS:
        # Send no instruction at all. VoxCPM2 swallows an unknown tag silently --
        # verified by transcribing its output -- so passing it through would just
        # produce plain speech while implying the tag did something.
        return ResolvedTag(
            None, Tone.NEUTRAL, intensity, key,
            f"[{raw}] is a {UNSUPPORTED_TAGS[key]}; VoxCPM2 cannot produce it and will ignore the tag.",
        )

    cached = _DYNAMIC_STYLE_CACHE.get((key, intensity))
    if cached is not None:
        return cached

    if use_llm:
        try:
            from app.annotator import annotator
            llm_res = annotator.convert_style_tag(raw, intensity, custom_model=custom_model)
            if llm_res and llm_res.get("instruction"):
                resolved = ResolvedTag(
                    instruction=llm_res["instruction"],
                    tone=llm_res.get("tone", _family_from_body(raw) or Tone.NEUTRAL),
                    intensity=llm_res.get("intensity", intensity),
                    label=key,
                    warning=None,
                )
                _DYNAMIC_STYLE_CACHE[(key, intensity)] = resolved
                return resolved
        except Exception:
            pass

    # If the tag is a composite expression or already a full parenthetical instruction
    family = _family_from_body(raw)
    if family is not None:
        if len(raw.split()) > 1:
            instruction = raw if (raw.startswith("(") and raw.endswith(")")) else f"({raw})"
        else:
            instruction = format_voxcpm_instruction(family, intensity)
        resolved = ResolvedTag(instruction, family, intensity, key, None)
        _DYNAMIC_STYLE_CACHE[(key, intensity)] = resolved
        return resolved

    resolved = ResolvedTag(f"({raw})", Tone.NEUTRAL, intensity, key, None)
    _DYNAMIC_STYLE_CACHE[(key, intensity)] = resolved
    return resolved



class Span(NamedTuple):

    tag: ResolvedTag
    body: str
    break_before: bool   # the source put a line break ahead of this tag


class ChunkSpec(NamedTuple):
    """A chunk plus what the audio assembler needs to place it."""
    text: str            # instruction + body, ready for the engine
    tone: str            # Tone value, for the per-emotion level and pause
    intensity: int
    break_before: bool


def _is_valid_tag_name(name: str) -> bool:
    clean = name.strip().lower()
    if not clean:
        return False
    # If it contains Thai characters, it must be in the known Thai emotion vocabulary.
    # Regular Thai in brackets like "(พิเศษ)" or "[กรุงเทพ]" is spoken content, not direction.
    if re.search(r"[\u0e00-\u0e7f]", clean):
        return clean in STYLE_VOCABULARY or clean in _TONE_BY_NAME
    return True


def _tagged_spans(text: str, use_llm: bool = False, custom_model: Optional[str] = None) -> List[Span]:
    """Split text into (resolved tag, body) runs at each style tag."""
    matches = [m for m in STYLE_TAG_RE.finditer(text) if _is_valid_tag_name(m.group(1))]
    if not matches:
        return []

    plain = ResolvedTag(None, Tone.NEUTRAL, 2, "neutral", None)
    spans: List[Span] = []

    lead = _clean_body(text[:matches[0].start()])
    if lead:
        spans.append(Span(plain, lead, False))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = _clean_body(text[m.end():end])
        if not body:
            continue
        # Whether the writer put this tag on its own line decides how long a pause
        # it earns: the ElevenLabs reference leaves ~1.2 s between separately-written
        # blocks, but that is far too much for a tone change mid-sentence.
        raw_before = text[matches[i - 1].end():m.start()] if i else text[:m.start()]
        spans.append(Span(resolve_style_tag(m.group(1), m.group(2), use_llm=use_llm, custom_model=custom_model), body,
                          bool(_BREAK_RE.search(raw_before))))
    return spans


def split_style_chunks(text: str, use_llm: bool = False, custom_model: Optional[str] = None) -> List[str]:
    """Split hand-written text into chunks that each *lead* with a style instruction.

    VoxCPM2 only honours a parenthetical at position 0, so a tag typed mid-text would
    otherwise be read aloud. Splitting at every tag turns "[a]one[b]two" into two
    separately-synthesized chunks, which is what the user meant by writing it.

    Returns [] when the text carries no style tag, so callers can fall back to the
    LLM annotation path.
    """
    return [spec.text for spec in split_style_chunk_specs(text, use_llm=use_llm, custom_model=custom_model)]


def split_style_chunk_specs(text: str, use_llm: bool = False, custom_model: Optional[str] = None) -> List[ChunkSpec]:
    """As ``split_style_chunks``, but keeping the tone and layout of each chunk."""
    return [
        ChunkSpec(
            text=f"{span.tag.instruction}{span.body}" if span.tag.instruction else span.body,
            tone=span.tag.tone.value,
            intensity=span.tag.intensity,
            break_before=span.break_before,
        )
        for span in _tagged_spans(text, use_llm=use_llm, custom_model=custom_model)
    ]


def parse_tagged_segments(text: str, use_llm: bool = False, custom_model: Optional[str] = None) -> List[Segment]:
    """Read hand-written tags as annotated segments, expanding free-form tags with LLM if use_llm=True."""
    return [
        Segment(text=span.body, tone=span.tag.tone, intensity=span.tag.intensity,
                style=span.tag.label, break_before=span.break_before)
        for span in _tagged_spans(text, use_llm=use_llm, custom_model=custom_model)
    ]



def build_script(segments: List[Segment]) -> str:
    """The short, editable form of an annotated script: "[sad] ... [happy] ...".

    This is what the studio shows in its editable box and what gets sent back to
    /synthesize, so it has to round-trip. The single-shot ``text`` rendering cannot:
    it carries only the *first* instruction followed by every body concatenated, so
    feeding it back collapsed a four-emotion script into one tone.

    Intensity is written only when it is not the default, and a segment that started
    a new line keeps its line break, because that is what earns the longer pause.
    """
    parts: List[str] = []
    prev_key = None

    for seg in segments:
        body = (seg.spoken_text or seg.text).strip()
        if not body:
            continue

        label = seg.style or seg.tone.value
        key = (label, seg.intensity, seg.tone)

        if key == prev_key and parts:
            parts.append(f" {body}")
            continue

        plain = seg.tone == Tone.NEUTRAL and label.lower() in _PLAIN_LABELS
        if plain:
            tag = ""
        elif seg.intensity == 2:
            tag = f"[{label}] "
        else:
            tag = f"[{label}:{seg.intensity}] "

        if not parts:
            sep = ""
        elif seg.break_before:
            sep = "\n"
        else:
            sep = " "

        parts.append(f"{sep}{tag}{body}")
        prev_key = key

    return "".join(parts).strip()


def collect_tag_warnings(text: str) -> List[str]:
    """Tags that parsed cleanly but cannot affect the audio."""
    seen, out = set(), []
    for span in _tagged_spans(text):
        if span.tag.warning and span.tag.warning not in seen:
            seen.add(span.tag.warning)
            out.append(span.tag.warning)
    return out


# Wording is load-bearing and was chosen by measurement, not taste. Against a pinned
# speaker over 4 reps, these gave happy-vs-sad dF0 +28.6Hz / pitch-spread +42.4.
# Appending explicit prosody ("bright high pitch, lively quick pace") DILUTED it to
# +4.4 / -2.2, and bare "(happy)" / "(sad)" tags were worse still at -15.0 / -6.1.
# Re-measure before rewording.
#
# Wording also decides whether the direction is *obeyed at all*. VoxCPM2 can drop
# out of control mode on a particular phrasing and read the parenthetical aloud in
# English ahead of the line -- audible, and invisible to every prosody metric.
# Sad@2 used to be "(Sad and melancholic voice, slight sighs)", which leaked in 6 of
# 6 takes (Whisper heard "Sad and Melancholic Voice Slight Sighs", +2.5 s of audio);
# the wording below leaked 0 of 3, at lower F0 and slower pace than neutral.
#
# The trigger is the exact phrasing, not any one word: tired@3 carries "heavy sighs"
# with no leak at all. So there is no rule to encode here -- re-run
# tools/instruction_leak_audit.py after any edit, at enough reps to matter.
# Sarcastic@3 is why "enough reps": "(Heavy sarcastic and cynical tone)" leaked 4
# takes in 30, and one run of 6 came back clean. A three-rep spot check passes it
# roughly two times in three; a listener hears it within the afternoon.
VOXCPM_INSTRUCTION_MAP = {
    Tone.NEUTRAL: {
        1: None,
        2: None,
        3: None,
    },
    Tone.CALM: {
        1: "(Slightly calm and gentle tone)",
        2: "(Calm and soothing voice, speaking softly)",
        3: "(Deeply calm and relaxing tone, very slow pace)",
    },
    Tone.SAD: {
        1: "(Slightly sad tone)",
        2: "(Sad voice, quiet and downcast)",
        3: "(Deeply sorrowful and crying voice, trembling)",
    },
    Tone.HAPPY: {
        1: "(Pleasant tone, slight smile)",
        2: "(Happy and cheerful voice, smiling while speaking)",
        3: "(Extremely joyful and laughing voice)",
    },
    Tone.ANGRY: {
        1: "(Annoyed and sharp voice)",
        2: "(Angry, firm and aggressive tone)",
        3: "(Furious and yelling tone, very loud and harsh)",
    },
    Tone.EXCITED: {
        1: "(Eager voice)",
        2: "(Excited and energetic tone)",
        3: "(Thrilled and loud energetic voice)",
    },
    Tone.NERVOUS: {
        1: "(Slightly hesitant voice)",
        2: "(Nervous and trembling voice, hesitant)",
        3: "(Extremely anxious and panicking voice)",
    },
    Tone.SARCASTIC: {
        1: "(Slightly sarcastic tone)",
        2: "(Sarcastic and mocking tone)",
        3: "(Deeply sarcastic and cynical tone)",
    },
    Tone.SCARED: {
        1: "(Slightly uneasy and wary voice)",
        2: "(Scared and fearful voice, trembling breath)",
        3: "(Terrified and panicked voice, gasping and shaking)",
    },
    Tone.TIRED: {
        1: "(Slightly weary voice)",
        2: "(Tired and weary voice, low energy, slow pace)",
        3: "(Exhausted and drained voice, heavy sighs, very slow)",
    },
}

# ---------------------------------------------------------------------------
# Open style vocabulary
#
# The Tone enum stays small on purpose: it is the LLM's classification label
# space, and every member must be mirrored across six files. Hand-written tags
# do not need that constraint -- VoxCPM2 takes free-form natural language -- so
# extra styles live here alone, each carrying a full instruction plus a "family"
# Tone used only for colour and for the ElevenLabs/Gemini renderers.
#
# Instructions stay in the short descriptive register that measured best
# (dF0 +28.6 / spread +42.4); appending explicit prosody diluted it to +4.4.
# ---------------------------------------------------------------------------

STYLE_VOCABULARY = {
    # -- direct synonyms of an existing tone -------------------------------
    "afraid": (None, Tone.SCARED), "fearful": (None, Tone.SCARED),
    "frightened": (None, Tone.SCARED), "terrified": (None, Tone.SCARED),
    "panicked": (None, Tone.SCARED),
    "sleepy": (None, Tone.TIRED), "exhausted": (None, Tone.TIRED),
    "weary": (None, Tone.TIRED), "drained": (None, Tone.TIRED),
    "anxious": (None, Tone.NERVOUS), "worried": (None, Tone.NERVOUS),
    "hesitant": (None, Tone.NERVOUS),
    "joyful": (None, Tone.HAPPY), "cheerful": (None, Tone.HAPPY),
    "glad": (None, Tone.HAPPY),
    # The studio's tag button inserts the ElevenLabs spelling.
    "happily": (None, Tone.HAPPY),
    "furious": (None, Tone.ANGRY), "mad": (None, Tone.ANGRY),
    "sorrowful": (None, Tone.SAD), "depressed": (None, Tone.SAD),
    "melancholic": (None, Tone.SAD),
    "relaxed": (None, Tone.CALM), "gentle": (None, Tone.CALM),
    "soothing": (None, Tone.CALM),
    "thrilled": (None, Tone.EXCITED), "eager": (None, Tone.EXCITED),
    "energetic": (None, Tone.EXCITED),
    "mocking": (None, Tone.SARCASTIC), "cynical": (None, Tone.SARCASTIC),
    "normal": (None, Tone.NEUTRAL), "plain": (None, Tone.NEUTRAL),
    "flat": (None, Tone.NEUTRAL),

    # -- distinct styles with their own direction --------------------------
    "annoyed": ("(Annoyed and irritated voice, clipped delivery)", Tone.ANGRY),
    "appalled": ("(Appalled and shocked voice, sharp disbelief)", Tone.ANGRY),
    "disappointed": ("(Disappointed voice, quiet and let down)", Tone.SAD),
    "crying": ("(Crying voice, broken and tearful, trembling)", Tone.SAD),
    "crying and tearful": ("(Crying voice, broken and tearful, trembling)", Tone.SAD),
    "tearful and crying": ("(Crying voice, broken and tearful, trembling)", Tone.SAD),
    "sad and cry": ("(Deeply sorrowful and crying voice, trembling)", Tone.SAD),
    "sad and crying": ("(Deeply sorrowful and crying voice, trembling)", Tone.SAD),
    "tearful": (None, Tone.SAD), "sobbing": (None, Tone.SAD),
    "weeping": (None, Tone.SAD), "teary": (None, Tone.SAD),
    "heartbroken": (None, Tone.SAD),
    "somber": ("(Somber and melancholic voice, quiet and grave)", Tone.SAD),
    "gloomy": ("(Gloomy and despondent voice, heavy and subdued)", Tone.SAD),
    "surprised": ("(Surprised voice, sudden rising pitch)", Tone.EXCITED),
    "curious": ("(Curious and inquisitive voice, questioning tone)", Tone.EXCITED),
    "thoughtful": ("(Thoughtful voice, measured and reflective, unhurried)", Tone.CALM),
    "mischievous": ("(Mischievous voice, playful and teasing)", Tone.SARCASTIC),
    "playful": ("(Playful and light voice, teasing lilt)", Tone.HAPPY),
    "whispering": ("(Whispering voice, very soft and breathy)", Tone.CALM),
    "shouting": ("(Shouting voice, very loud and projected)", Tone.ANGRY),
    "confident": ("(Confident and assured voice, steady and firm)", Tone.NEUTRAL),
    "serious": ("(Serious and grave voice, deliberate delivery)", Tone.NEUTRAL),
    "warm": ("(Warm and friendly voice, gentle smile)", Tone.HAPPY),
    "romantic": ("(Warm intimate voice, soft and affectionate)", Tone.CALM),
    "proud": ("(Proud voice, bright and self-assured)", Tone.HAPPY),
    "apologetic": ("(Apologetic voice, soft and regretful)", Tone.SAD),
    "urgent": ("(Urgent voice, fast and pressing)", Tone.EXCITED),
    "bored": ("(Bored voice, flat and disinterested, slow)", Tone.TIRED),
    "disgusted": ("(Disgusted voice, recoiling distaste)", Tone.ANGRY),

    # -- Thai emotion vocabulary ------------------------------------------
    "ปกติ": (None, Tone.NEUTRAL), "เป็นกลาง": (None, Tone.NEUTRAL),
    "เศร้า": (None, Tone.SAD), "เสียใจ": (None, Tone.SAD), "หม่นหมอง": (None, Tone.SAD),
    "ดีใจ": (None, Tone.HAPPY), "ร่าเริง": (None, Tone.HAPPY), "มีความสุข": (None, Tone.HAPPY), "สดใส": (None, Tone.HAPPY),
    "โกรธ": (None, Tone.ANGRY), "โมโห": (None, Tone.ANGRY), "ไม่พอใจ": (None, Tone.ANGRY), "หงุดหงิด": ("(Annoyed and irritated voice, clipped delivery)", Tone.ANGRY),
    "ตื่นเต้น": (None, Tone.EXCITED), "กระตือรือร้น": (None, Tone.EXCITED),
    "สงบ": (None, Tone.CALM), "ผ่อนคลาย": (None, Tone.CALM), "นุ่มนวล": (None, Tone.CALM), "อ่อนโยน": (None, Tone.CALM),
    "ประหม่า": (None, Tone.NERVOUS), "กังวล": (None, Tone.NERVOUS), "ลังเล": (None, Tone.NERVOUS), "ระแวง": (None, Tone.NERVOUS),
    "ประชด": (None, Tone.SARCASTIC), "แดกดัน": (None, Tone.SARCASTIC), "ประชดประชัน": (None, Tone.SARCASTIC),
    "กลัว": (None, Tone.SCARED), "หวาดกลัว": (None, Tone.SCARED), "ตกใจ": ("(Surprised voice, sudden rising pitch)", Tone.EXCITED), "ตื่นตระหนก": (None, Tone.SCARED),
    "เหนื่อย": (None, Tone.TIRED), "อ่อนเพลีย": (None, Tone.TIRED), "หมดแรง": (None, Tone.TIRED), "ง่วง": (None, Tone.TIRED),
    "กระซิบ": ("(Whispering voice, very soft and breathy)", Tone.CALM),
    "ตะโกน": ("(Shouting voice, very loud and projected)", Tone.ANGRY),
    "ร้องไห้": ("(Crying voice, broken and tearful, trembling)", Tone.SAD),
    "ผิดหวัง": ("(Disappointed voice, quiet and let down)", Tone.SAD),
    "ขี้เล่น": ("(Playful and light voice, teasing lilt)", Tone.HAPPY),
    "จริงจัง": ("(Serious and grave voice, deliberate delivery)", Tone.NEUTRAL),
    "มั่นใจ": ("(Confident and assured voice, steady and firm)", Tone.NEUTRAL),
}
STYLE_VOCABULARY["whispers"] = STYLE_VOCABULARY["whispering"]
STYLE_VOCABULARY["shouts"] = STYLE_VOCABULARY["shouting"]
# Words that mean "no direction". A segment carrying one of these gets no tag in the
# short script form, so an untagged opening line stays untagged.
_PLAIN_LABELS = {"neutral"} | {
    word for word, (instruction, family) in STYLE_VOCABULARY.items()
    if instruction is None and family is Tone.NEUTRAL
}

# VoxCPM2 has no mechanism for these. Verified by transcribing its output: the
# tag is silently absorbed -- never spoken, but never realised either. Naming
# them lets the caller say so instead of returning audio that quietly ignored it.
UNSUPPORTED_TAGS = {
    "laughs": "non-verbal sound", "laugh": "non-verbal sound",
    "giggles": "non-verbal sound", "chuckles": "non-verbal sound",
    "wheezing": "non-verbal sound", "sighs": "non-verbal sound",
    "exhales": "non-verbal sound", "inhales": "non-verbal sound",
    "snorts": "non-verbal sound", "swallows": "non-verbal sound",
    "gulps": "non-verbal sound", "clears throat": "non-verbal sound",
    "applause": "sound effect", "clapping": "sound effect",
    "gunshot": "sound effect", "explosion": "sound effect",
    "woo": "sound effect", "sings": "singing",
    "short pause": "timing", "long pause": "timing",
}


def format_voxcpm_instruction(tone: Tone, intensity: int = 2) -> Optional[str]:
    """Format VoxCPM emotion control instruction."""
    intensity = max(1, min(3, intensity))
    tone_map = VOXCPM_INSTRUCTION_MAP.get(tone, {})
    return tone_map.get(intensity)


def instruction_for_segment(seg: Segment) -> Optional[str]:
    """The style parenthetical a segment should actually lead with.

    A hand-written tag carries its exact word in ``style``, and STYLE_VOCABULARY
    often has a sharper instruction for it than its coarse family does --
    "appalled" -> "(Appalled and shocked voice, sharp disbelief)" rather than the
    generic angry line. Falling back to the family alone made /speak preview an
    instruction that /synthesize never used.
    """
    if seg.style:
        return resolve_style_tag(seg.style, str(seg.intensity)).instruction
    if seg.tone == Tone.NEUTRAL:
        return None
    return format_voxcpm_instruction(seg.tone, seg.intensity)


class VoxCPMRenderer(BaseRenderer):
    """
    Renders segments for VoxCPM2 / SiangTTS with natural-language control instructions.
    Example: '(Calm and soothing voice, speaking softly)หายใจเข้าลึกๆ ผ่อนคลาย'

    VoxCPM2 reads a style parenthetical only when it leads the text it is given; one
    appearing mid-text is spoken aloud instead. So each run of same-tone segments
    becomes its own chunk with the instruction at position 0, and the caller
    synthesizes the chunks separately. ``text`` remains a single-shot rendering that
    only carries the opening instruction.
    """

    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None, chunks=[])

        chunks: List[RenderedChunk] = []
        instructions_used: List[str] = []
        prev_key: Optional[tuple] = None

        for seg in segments:
            instruction = instruction_for_segment(seg)
            spoken = seg.spoken_text if seg.spoken_text is not None else seg.text
            # Key on what actually reaches the model. Keying on `tone` alone merged
            # "[sad:1] a [sad:3] b" into one chunk at intensity 1, and merged two
            # different STYLE_VOCABULARY styles that happen to share a family.
            key = (instruction, seg.tone)

            if key == prev_key and chunks:
                chunks[-1].body += spoken
                chunks[-1].text += spoken
                continue

            if instruction:
                label = seg.style or seg.tone.value
                instructions_used.append(f"{label} (lvl {seg.intensity})")

            # No space after ')' -- that is the documented VoxCPM2 format.
            chunks.append(
                RenderedChunk(
                    text=f"{instruction}{spoken}" if instruction else spoken,
                    instruction=instruction,
                    body=spoken,
                    tone=seg.tone.value,
                    break_before=seg.break_before,
                )
            )
            prev_key = key

        for c in chunks:
            c.body = c.body.strip()
            c.text = c.text.strip()
        chunks = [c for c in chunks if c.body]

        # Single-shot form: the leading instruction applies, the rest is plain body
        # text so no parenthetical ever lands mid-utterance.
        lead = chunks[0].instruction if chunks else None
        joined = "".join(c.body for c in chunks)
        rendered_text = f"{lead}{joined}" if lead else joined

        return RenderResponse(
            text=rendered_text.strip(),
            prompt=", ".join(instructions_used) if instructions_used else "neutral",
            script=build_script(segments),
            chunks=chunks,
        )
