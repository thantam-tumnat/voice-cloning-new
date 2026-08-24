"""Assemble per-tone chunks into one take the way ElevenLabs does.

VoxCPM2 is asked for one chunk per tone run, and those chunks used to be joined
with a flat 60 ms of silence and whatever level the model happened to pick. That
is the single largest audible difference from the ElevenLabs reference, which

  * varies loudness by emotion deliberately (scared +3.0 dB, sad -2.0 dB), where
    an unmanaged concatenation varies it by accident;
  * varies duration by emotion (happy 0.93x, tired 1.07x of the take mean);
  * leaves a real pause at every emotion boundary, not 60 ms.

Every constant below traces back to tools/prosody_eval.ELEVENLABS_REFERENCE, which
was measured off the reference take. Four tones were measured directly; the rest
are placed by family and marked as such, so re-measuring can replace them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# Energy per tone, dB relative to the take's own neutral level. The four measured
# values came out near-symmetric about zero, so they double as absolute offsets.
TONE_ENERGY_DB = {
    "neutral": 0.0,
    "sad": -2.0,      # measured
    "happy": -0.5,    # measured
    "scared": 3.0,    # measured
    "tired": -1.4,    # measured
    # Placed by family from the four above; re-measure before trusting them.
    "calm": -2.0,
    "angry": 3.0,
    "excited": 2.5,
    "nervous": -0.5,
    "sarcastic": -0.5,
}

# Duration relative to a neutral read of the same text. Same provenance.
TONE_DURATION_RATIO = {
    "neutral": 1.00,
    "sad": 1.035,     # measured
    "happy": 0.927,   # measured
    "scared": 0.966,  # measured
    "tired": 1.073,   # measured
    # Placed by family.
    "calm": 1.08,
    "angry": 0.95,
    "excited": 0.90,
    "nervous": 1.00,
    "sarcastic": 1.05,
}

# The reference's four takes sit 0.95-1.44 s apart, but they are four independent
# renditions of one sentence, not a flowing script -- copying 1.2 s into every
# emotion change mid-paragraph would drag. So the long gap is reserved for a real
# line break in the source, which is how the reference script was written, and an
# inline tone change gets an ordinary sentence pause instead.
GAP_SAME_TONE_S = 0.20
GAP_EMOTION_S = 0.45
GAP_PARAGRAPH_S = 1.20

# Long enough to kill a boundary click, short enough not to eat a plosive.
EDGE_FADE_S = 0.015

# Leave a little room either side of a trim so speech never starts abruptly.
TRIM_KEEP_S = 0.03
TRIM_FLOOR_DB = 45.0

# Time-stretch beyond this stops being transparent and starts sounding processed.
MAX_STRETCH = 0.15

OUTPUT_PEAK = 0.95

# How much of each chunk's own level is corrected away before the per-tone target is
# applied. At 1.0 -- what this did originally -- every chunk is flattened to the
# take's median and the only surviving loudness difference is TONE_ENERGY_DB, which
# caps the whole take's dynamic range at the 5 dB between "angry" and "sad" no matter
# what the model delivered. That was the right call while the model's own level
# differences were accidental; with the Thai LoRA off the DiT side it renders anger
# 1.2 dB up and sadness 3.2 dB down on its own, and those are worth keeping.
#
# Measured on a real five-emotion take, correction vs the resulting angry-to-sad
# spread: 1.0 -> 4.84 dB, 0.7 -> 5.65 dB, 0.5 -> 6.19 dB. The cost is that the
# model's *weak* emotions pass through too -- it renders "happy" about a dB quieter
# than neutral, and at 0.5 that lands 3.3 dB down against a -0.5 dB reference. 0.7
# holds sad and scared within half a dB of the ElevenLabs reference while widening
# the contrast that was the complaint. Fixing happy properly means fixing it at the
# model, not here.
ENERGY_MATCH = 0.7


class Chunk(NamedTuple):
    """One synthesized tone run, plus what the assembler needs to place it."""
    audio: np.ndarray
    tone: Optional[str] = None
    break_before: bool = False
    # Characters of spoken body text. Rate matching needs it to tell "this chunk is
    # slow" from "this chunk is long"; without it rate matching runs open-loop.
    text_len: int = 0


@dataclass(frozen=True)
class PostProcessConfig:
    """Every knob the assembler exposes, defaulted to the measured constants above.

    The constants are the reference values and stay the defaults; this exists so a
    caller -- the UI, a sweep, an A/B -- can move one of them for a single take
    without editing the module. ``from_dict`` ignores unknown and ``None`` entries so
    a partial payload from the client only overrides what it actually set.
    """
    gap_same_tone_s: float = GAP_SAME_TONE_S
    gap_emotion_s: float = GAP_EMOTION_S
    gap_paragraph_s: float = GAP_PARAGRAPH_S
    match_energy: bool = True
    match_rate: bool = True
    energy_match: float = ENERGY_MATCH
    max_stretch: float = MAX_STRETCH
    trim_floor_db: float = TRIM_FLOOR_DB
    trim_keep_s: float = TRIM_KEEP_S
    edge_fade_s: float = EDGE_FADE_S
    output_peak: float = OUTPUT_PEAK
    # Per-tone overrides, merged over the module tables rather than replacing them,
    # so a client that only cares about "sad" does not have to resend all ten.
    tone_energy_db: Dict[str, float] = field(default_factory=dict)
    tone_duration_ratio: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "PostProcessConfig":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known and v is not None}
        return cls(**clean)

    def energy_for(self, tone: Optional[str]) -> float:
        key = tone or "neutral"
        if key in self.tone_energy_db:
            return float(self.tone_energy_db[key])
        return TONE_ENERGY_DB.get(key, 0.0)

    def duration_for(self, tone: Optional[str]) -> float:
        key = tone or "neutral"
        if key in self.tone_duration_ratio:
            return float(self.tone_duration_ratio[key])
        return TONE_DURATION_RATIO.get(key, 1.0)


DEFAULT_POST_PROCESS = PostProcessConfig()


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

def remove_dc(x: np.ndarray) -> np.ndarray:
    """Strip any DC offset, which otherwise makes every join click."""
    x = np.asarray(x, dtype="float32")
    return x if x.size == 0 else (x - float(np.mean(x))).astype("float32")


def _frame_db(x: np.ndarray, sr: int, hop_s: float = 0.010, win_s: float = 0.025) -> np.ndarray:
    hop, win = max(1, int(sr * hop_s)), max(2, int(sr * win_s))
    n = max(0, (len(x) - win) // hop)
    if n == 0:
        return np.zeros(0, dtype="float32")
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n]
    rms = np.sqrt(np.mean(frames.astype("float64") ** 2, axis=1))
    return (20 * np.log10(rms + 1e-9)).astype("float32")


def trim_silence(
    x: np.ndarray, sr: int, floor_db: float = TRIM_FLOOR_DB, keep_s: float = TRIM_KEEP_S
) -> np.ndarray:
    """Drop leading/trailing silence, keeping ``keep_s`` of padding.

    VoxCPM2 pads each generation by an arbitrary amount, so joining raw chunks
    produced gaps that varied unpredictably regardless of what gap we asked for.
    Trimming first is what makes the gap policy below mean anything.
    """
    x = np.asarray(x, dtype="float32")
    db = _frame_db(x, sr)
    if db.size == 0:
        return x

    voiced = np.flatnonzero(db > (db.max() - floor_db))
    if voiced.size == 0:
        return x

    hop = max(1, int(sr * 0.010))
    keep = int(sr * keep_s)
    start = max(0, int(voiced[0]) * hop - keep)
    end = min(len(x), int(voiced[-1]) * hop + int(sr * 0.025) + keep)
    return x[start:end]


def fade_edges(x: np.ndarray, sr: int, fade_s: float = EDGE_FADE_S) -> np.ndarray:
    """Raised-cosine fade in and out, so concatenation never steps the waveform."""
    x = np.array(x, dtype="float32", copy=True)
    n = int(sr * fade_s)
    if n < 2 or x.size < 2 * n:
        return x
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, n))).astype("float32")
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def voiced_rms(x: np.ndarray, sr: int, floor_db: float = 35.0) -> float:
    """RMS over speech frames only.

    Whole-chunk RMS would let a chunk that happens to carry more silence read as
    quieter and get boosted for it, which is the opposite of what we want.
    """
    db = _frame_db(x, sr)
    if db.size == 0:
        return float(np.sqrt(np.mean(np.asarray(x, dtype="float64") ** 2))) if x.size else 0.0
    speech = db[db > (db.max() - floor_db)]
    if speech.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((10 ** (speech.astype("float64") / 20)) ** 2)))


def apply_gain_db(x: np.ndarray, db: float) -> np.ndarray:
    return (np.asarray(x, dtype="float32") * float(10 ** (db / 20))).astype("float32")


def time_stretch(x: np.ndarray, sr: int, ratio: float, frame_s: float = 0.040,
                 seek_s: float = 0.010) -> np.ndarray:
    """WSOLA time-stretch. ``ratio`` > 1 makes it longer (slower).

    Pitch is untouched, which is the whole point: the reference varies rate by 14%
    across emotions while holding median F0 within 1.75 semitones, and any
    rate-matching that moved pitch would break the cloned identity we are trying
    to preserve.
    """
    x = np.asarray(x, dtype="float64")
    frame = int(sr * frame_s)
    if frame < 8 or x.size < 2 * frame or abs(ratio - 1.0) < 0.005:
        return np.asarray(x, dtype="float32")

    syn_hop = frame // 2
    ana_hop = max(1, int(round(syn_hop / ratio)))
    seek = max(1, int(sr * seek_s))
    win = np.hanning(frame)

    n_out = int(math.ceil(x.size * ratio)) + 2 * frame
    out = np.zeros(n_out, dtype="float64")
    wsum = np.zeros(n_out, dtype="float64")

    template = x[:frame].copy()
    a = 0
    s = 0
    while a + frame <= x.size and s + frame <= n_out:
        lo = max(0, a - seek)
        hi = min(x.size - frame, a + seek)
        if hi <= lo:
            best = min(max(a, 0), x.size - frame)
        else:
            # One correlate call scores every candidate offset at once.
            scores = np.correlate(x[lo:hi + frame], template, mode="valid")
            best = lo + int(np.argmax(scores))

        seg = x[best:best + frame]
        out[s:s + frame] += seg * win
        wsum[s:s + frame] += win

        nxt = best + syn_hop
        template = x[nxt:nxt + frame] if nxt + frame <= x.size else np.zeros(frame)
        s += syn_hop
        a += ana_hop

    keep = min(n_out, s + frame)
    out, wsum = out[:keep], wsum[:keep]
    out = np.divide(out, wsum, out=np.zeros_like(out), where=wsum > 1e-6)
    return out.astype("float32")


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def _match_rate(prepared: List[Chunk], sr: int, cfg: PostProcessConfig = DEFAULT_POST_PROCESS) -> List[Chunk]:
    """Nudge each chunk toward its measured pace target, correcting from where it is.

    This has to be closed-loop. Applying the target ratio directly assumes the model
    delivered a neutral pace and only needs the emotion added, but it does not: on
    the reference script VoxCPM2 rendered "happy" at 0.80x of the take mean when the
    target was 0.927x, so an open-loop 0.927x stretch drove it to 0.73x -- further
    from ElevenLabs than doing nothing.

    Pace is seconds per character, not duration, so chunks of different lengths stay
    comparable. Chunks that did not report their text length fall back to a plain
    duration comparison, which is right only when the chunks are similarly long.

    The baseline is this take's *neutral* pace, and neutral chunks anchor it: every
    ratio in TONE_DURATION_RATIO is defined against a neutral read, so the neutral
    chunks are the ones that should come through exactly as rendered while the
    emotional ones move around them. Normalising the targets by their own mean
    instead -- what this did originally -- gave neutral no fixed point and let it
    drift with whatever else shared the take: against calm (1.08) and tired (1.073)
    it was driven to 0.95x of the take median, i.e. audibly rushed, and against
    happy and excited it was dragged the other way.

    A take with no neutral in it has to estimate one, which it does by dividing each
    chunk's measured pace by the ratio that chunk was asked for. That also stops a
    single-tone take from having its ratio applied a second time on top of whatever
    the model already did.
    """
    wanted = [cfg.duration_for(c.tone) for c in prepared]

    lengths = [max(c.text_len, 1) if c.text_len else 1 for c in prepared]
    pace = [len(c.audio) / sr / n for c, n in zip(prepared, lengths)]
    anchor = [p for p, w in zip(pace, wanted) if p > 0 and w == 1.0]
    if not anchor:
        anchor = [p / w for p, w in zip(pace, wanted) if p > 0 and w > 0]
    baseline = float(np.median(anchor)) if anchor else 0.0
    if baseline <= 0:
        return prepared

    retimed: List[Chunk] = []
    for c, w, p in zip(prepared, wanted, pace):
        if p <= 0:
            retimed.append(c)
            continue
        target_pace = baseline * w
        ratio = float(np.clip(target_pace / p, 1 - cfg.max_stretch, 1 + cfg.max_stretch))
        retimed.append(c._replace(audio=time_stretch(c.audio, sr, ratio)))
    return retimed


def gap_before(
    chunk: Chunk, prev: Optional[Chunk], cfg: PostProcessConfig = DEFAULT_POST_PROCESS
) -> float:
    """Seconds of silence to place ahead of ``chunk``."""
    if prev is None:
        return 0.0
    if chunk.break_before:
        return cfg.gap_paragraph_s
    if chunk.tone != prev.tone:
        return cfg.gap_emotion_s
    return cfg.gap_same_tone_s


Span = Tuple[float, float]


def butt_join_with_spans(
    chunks: Sequence[Chunk], sr: int, gap_s: float = 0.060
) -> Tuple[np.ndarray, List[Span]]:
    """The pre-assembler join: raw chunks, flat gap, no levelling.

    Retained so tools/ab_gen.py can render a genuine "before" from the same
    generation rather than from a second, differently-sampled one.
    """
    usable = [np.asarray(c.audio, dtype="float32") for c in chunks
              if c.audio is not None and np.asarray(c.audio).size]
    if not usable:
        return np.zeros(0, dtype="float32"), []

    gap = np.zeros(int(sr * gap_s), dtype="float32")
    parts: List[np.ndarray] = []
    spans: List[Span] = []
    cursor = 0
    for i, a in enumerate(usable):
        if i:
            parts.append(gap)
            cursor += gap.size
        spans.append((cursor / sr, (cursor + a.size) / sr))
        parts.append(a)
        cursor += a.size

    audio = np.concatenate(parts).astype("float32") if len(parts) > 1 else parts[0]
    return audio, spans


def butt_join(chunks: Sequence[Chunk], sr: int, gap_s: float = 0.060) -> np.ndarray:
    return butt_join_with_spans(chunks, sr, gap_s)[0]


def assemble(
    chunks: Sequence[Chunk],
    sr: int,
    *,
    match_energy: Optional[bool] = None,
    match_rate: Optional[bool] = None,
    config: Optional[PostProcessConfig] = None,
) -> np.ndarray:
    """Trim, level, pace and join tone chunks into one take."""
    return assemble_with_spans(
        chunks, sr, match_energy=match_energy, match_rate=match_rate, config=config
    )[0]


def assemble_with_spans(
    chunks: Sequence[Chunk],
    sr: int,
    *,
    match_energy: Optional[bool] = None,
    match_rate: Optional[bool] = None,
    config: Optional[PostProcessConfig] = None,
) -> Tuple[np.ndarray, List[Span]]:
    """Trim, level, pace and join tone chunks, reporting where each landed.

    Energy matching pulls each chunk part-way toward the take's own median and then
    imposes the measured per-tone offset, so the loudness differences a listener
    hears are mostly intended ones without discarding the model's own; see
    ENERGY_MATCH for why that correction is partial rather than total. Rate matching corrects each chunk
    from the pace it was actually rendered at toward its target, bounded to +/-15%
    so it stays transparent. See _match_rate for why that correction is closed-loop.
    """
    cfg = config or DEFAULT_POST_PROCESS
    # The two legacy keyword flags still win when passed, so existing callers keep
    # their behaviour; otherwise the config carries them.
    if match_energy is not None or match_rate is not None:
        cfg = _dc_replace(
            cfg,
            match_energy=cfg.match_energy if match_energy is None else match_energy,
            match_rate=cfg.match_rate if match_rate is None else match_rate,
        )

    usable = [c for c in chunks if c.audio is not None and np.asarray(c.audio).size]
    if not usable:
        return np.zeros(0, dtype="float32"), []

    prepared: List[Chunk] = [
        c._replace(audio=trim_silence(
            remove_dc(np.asarray(c.audio, dtype="float32")),
            sr,
            floor_db=cfg.trim_floor_db,
            keep_s=cfg.trim_keep_s,
        ))
        for c in usable
    ]
    prepared = [c for c in prepared if c.audio.size]
    if not prepared:
        return np.zeros(0, dtype="float32"), []

    if cfg.match_rate:
        prepared = _match_rate(prepared, sr, cfg)

    if cfg.match_energy:
        levels = [voiced_rms(c.audio, sr) for c in prepared]
        usable_levels = [l for l in levels if l > 1e-6]
        baseline = float(np.median(usable_levels)) if usable_levels else 0.0
        if baseline > 1e-6:
            levelled = []
            for c, level in zip(prepared, levels):
                if level <= 1e-6:
                    levelled.append(c)
                    continue
                correction = 20 * math.log10(baseline / level) * cfg.energy_match
                target = cfg.energy_for(c.tone)
                levelled.append(c._replace(audio=apply_gain_db(c.audio, correction + target)))
            prepared = levelled

    pieces: List[np.ndarray] = []
    spans: List[Span] = []
    prev: Optional[Chunk] = None
    cursor = 0
    for c in prepared:
        gap = gap_before(c, prev, cfg)
        if gap > 0:
            pad = np.zeros(int(sr * gap), dtype="float32")
            pieces.append(pad)
            cursor += pad.size
        body = fade_edges(c.audio, sr, cfg.edge_fade_s)
        spans.append((cursor / sr, (cursor + body.size) / sr))
        pieces.append(body)
        cursor += body.size
        prev = c

    audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > cfg.output_peak:
        audio = audio * (cfg.output_peak / peak)
    return audio.astype("float32"), spans
