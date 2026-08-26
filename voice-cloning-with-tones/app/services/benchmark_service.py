from __future__ import annotations

import io
import json
import math
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import soundfile as sf

from app.models import (
    Tone,
    BenchmarkSessionInitRequest,
    BenchmarkSessionInitResponse,
    BenchmarkTakeRequest,
    BenchmarkTakeResult,
    BenchmarkSessionSummary,
    BenchmarkTakeVariant,
    BenchmarkF0Variant,
    ABVariantSpec,
)
from app.services.thonburian_service import thonburian_service

TEST_RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "test_runs"
TEST_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Preset Sentences for standardized emotion benchmark comparison
PRESET_SENTENCES = [
    {
        "id": "balanced",
        "title": "ประโยคมาตรฐาน (Balanced)",
        "text": "เขาบอกฉันแบบนั้นจริงๆ นะ ฉันไม่ได้คิดไปเองสักหน่อย",
        "desc": "ประโยคกลางๆ ที่สามารถสื่อได้ทุกอารมณ์ ชัดเจนในการวัดความแตกต่างของโทนเสียง",
    },
    {
        "id": "conversational",
        "title": "บทสนทนาประจำวัน (Conversational)",
        "text": "วันนี้มีเรื่องอยากจะเล่าให้ฟัง ตั้งใจฟังดีๆ นะ",
        "desc": "ประโยคเปิดการสนทนา สังเกตการลงท้ายเสียงและลีลาการพูด",
    },
    {
        "id": "dialogue",
        "title": "ประโยคตัดพ้อ / ตั้งคำถาม (Dialogue)",
        "text": "ทำไมเรื่องมันต้องกลายเป็นแบบนี้ด้วย ไม่เข้าใจเลยจริงๆ",
        "desc": "เหมาะมากสำหรับวัดอารมณ์ Sad, Angry, Frustrated",
    },
    {
        "id": "cheer",
        "title": "ประโยคข่าวสาร / ตื่นเต้น (Cheer & Joy)",
        "text": "ในที่สุดโปรเจกต์นี้ก็สำเร็จลุล่วงไปได้ด้วยดี ขอแสดงความยินดีด้วยครับ",
        "desc": "เหมาะสำหรับวัดอารมณ์ Happy, Neutral",
    },
]

# Emotion metadata dictionary for Thonburian F5 + SeedVC (5 thai-ser emotions)
EMOTION_META: Dict[str, Dict[str, Any]] = {
    "neutral": {
        "name_th": "ปกติ / เป็นกลาง",
        "icon": "📰",
        "color_class": "tone-neutral",
        "description": "น้ำเสียงปกติ บรรยาย ข้อมูลทั่วไป ไม่มีการกระแทกหรือลากเสียง",
    },
    "happy": {
        "name_th": "ร่าเริง / มีความสุข",
        "icon": "🎉",
        "color_class": "tone-happy",
        "description": "ดีใจ ร่าเริง มีความสุข ยิ้มแย้มขณะพูด ยกปลายเสียงสูงเล็กน้อย",
    },
    "sad": {
        "name_th": "เศร้า / เสียใจ",
        "icon": "😢",
        "color_class": "tone-sad",
        "description": "เศร้า เสียใจ ผิดหวัง ตัดพ้อ เสียงแผ่วเบา ทอดถอนใจ",
    },
    "angry": {
        "name_th": "โกรธ / ดุดัน",
        "icon": "😡",
        "color_class": "tone-angry",
        "description": "โกรธ ไม่พอใจ เสียงแข็ง ดุดัน กระแทกเสียงและพลังเสียงสูง",
    },
    "frustrated": {
        "name_th": "หงุดหงิด / อึดอัดใจ",
        "icon": "😤",
        "color_class": "tone-frustrated",
        "description": "หงุดหงิด อึดอัดใจ ถอนหายใจ อารมณ์คุกรุ่น",
    },
}


# ---------------------------------------------------------------------------
# Prosody / Acoustic Metric Extraction
# ---------------------------------------------------------------------------

def hz_to_st(hz: float, ref_hz: float = 100.0) -> float:
    """Convert Hz to semitones relative to reference frequency."""
    if hz <= 0 or ref_hz <= 0:
        return 0.0
    return 12.0 * math.log2(hz / ref_hz)


def extract_audio_metrics(wav_data: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute acoustic prosody metrics from audio array."""
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=1)

    dur_s = float(len(wav_data) / sr) if sr > 0 else 0.0
    if len(wav_data) == 0:
        return {
            "dur_s": 0.0,
            "rms": 0.0,
            "energy_dbfs": -100.0,
            "f0_med_hz": 0.0,
            "f0_p10_hz": 0.0,
            "f0_p90_hz": 0.0,
            "f0_spread_st": 0.0,
        }

    rms = float(np.sqrt(np.mean(wav_data.astype("float64") ** 2)))
    energy_dbfs = 20.0 * math.log10(rms + 1e-9)

    # Pitch tracking using Normalized Autocorrelation over speech frames
    hop_size = int(sr * 0.010)  # 10 ms hop
    frame_size = int(sr * 0.030)  # 30 ms window
    f0_candidates: List[float] = []

    # Frequency bounds for human speech: 60 Hz to 450 Hz
    min_lag = int(sr / 450)
    max_lag = int(sr / 60)

    for i in range(0, len(wav_data) - frame_size, hop_size):
        frame = wav_data[i : i + frame_size]
        # Energy threshold check to ignore silent frames
        frame_rms = np.sqrt(np.mean(frame**2))
        if frame_rms < (rms * 0.2) or frame_rms < 1e-4:
            continue

        # Autocorrelation
        windowed = frame * np.hanning(len(frame))
        corr = np.correlate(windowed, windowed, mode="full")
        corr = corr[len(corr) // 2 :]

        if len(corr) > max_lag:
            search_region = corr[min_lag:max_lag]
            if len(search_region) > 0 and np.max(search_region) > 0:
                peak_idx = int(np.argmax(search_region)) + min_lag
                # Basic peak salience test
                if corr[peak_idx] > 0.3 * corr[0]:
                    freq = float(sr / peak_idx)
                    f0_candidates.append(freq)

    if len(f0_candidates) >= 3:
        f0_arr = np.array(f0_candidates)
        f0_med = float(np.median(f0_arr))
        f0_p10 = float(np.percentile(f0_arr, 10))
        f0_p90 = float(np.percentile(f0_arr, 90))
        f0_spread = hz_to_st(f0_p90, f0_p10) if f0_p10 > 0 else 0.0
    else:
        f0_med = f0_p10 = f0_p90 = f0_spread = 0.0

    return {
        "dur_s": round(dur_s, 2),
        "rms": round(rms, 4),
        "energy_dbfs": round(energy_dbfs, 1),
        "f0_med_hz": round(f0_med, 1),
        "f0_p10_hz": round(f0_p10, 1),
        "f0_p90_hz": round(f0_p90, 1),
        "f0_spread_st": round(f0_spread, 2),
    }


class BenchmarkService:
    def __init__(self):
        self.runs_dir = TEST_RUNS_DIR

    def _session_dir(self, session_id: str) -> Path:
        # Sanitise session id
        clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
        s_dir = self.runs_dir / clean_id
        s_dir.mkdir(parents=True, exist_ok=True)
        return s_dir

    def get_presets(self) -> Dict[str, Any]:
        """Return preset sentences, available speakers, and emotion metadata."""
        speakers = thonburian_service.list_speakers()
        return {
            "preset_sentences": PRESET_SENTENCES,
            "emotions": [
                {
                    "id": k,
                    "name_en": k,
                    "name_th": v["name_th"],
                    "icon": v["icon"],
                    "color_class": v["color_class"],
                    "description": v["description"],
                    "default_instruction": f"[{k}]",
                }
                for k, v in EMOTION_META.items()
            ],
            "speakers": speakers,
            "donor_sets": thonburian_service.list_donor_sets(),
            "default_params": {
                "text": PRESET_SENTENCES[0]["text"],
                "repeats": 3,
                "gender": "female",
                "donor_set": None,
                "intensity": 2,
                "cfg_value": 2.0,
                "inference_timesteps": 32,
                "lora_mode": "on",
            },
        }

    def init_session(self, req: BenchmarkSessionInitRequest) -> BenchmarkSessionInitResponse:
        """Create a new benchmark session folder and initialize session.json."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        spk_tag = f"_{req.speaker_id}" if req.speaker_id else "_base"
        session_id = f"bench_{ts}{spk_tag}"
        s_dir = self._session_dir(session_id)

        emotions = req.emotions if req.emotions else list(EMOTION_META.keys())
        total_takes = len(emotions) * req.repeats
        gender = req.gender or "female"

        session_data = {
            "session_id": session_id,
            "name": req.name or f"Emotion Benchmark ({ts})",
            "created_at": datetime.now().isoformat(),
            "speaker_id": req.speaker_id,
            "gender": gender,
            "donor_set": req.donor_set,
            "text": req.text,
            "emotions": emotions,
            "repeats": req.repeats,
            "total_takes": total_takes,
            "completed_takes": 0,
            "params": {
                "gender": gender,
                "donor_set": req.donor_set,
                "intensity": req.intensity,
                "cfg_value": req.cfg_value,
                "inference_timesteps": req.inference_timesteps,
                "lora_mode": req.lora_mode or "on",
            },
            "takes": {},
        }

        meta_file = s_dir / "session.json"
        meta_file.write_text(json.dumps(session_data, ensure_ascii=False, indent=2), encoding="utf-8")

        return BenchmarkSessionInitResponse(
            session_id=session_id,
            name=session_data["name"],
            created_at=session_data["created_at"],
            speaker_id=req.speaker_id,
            gender=gender,
            text=req.text,
            emotions=emotions,
            repeats=req.repeats,
            total_takes=total_takes,
            params=session_data["params"],
        )

    def run_take(self, req: BenchmarkTakeRequest) -> BenchmarkTakeResult:
        """Synthesize a single take, compute prosody metrics, save audio, and update session."""
        start_t = time.time()
        s_dir = self._session_dir(req.session_id)
        meta_file = s_dir / "session.json"

        # Load session metadata
        if meta_file.exists():
            try:
                session_data = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                session_data = {"takes": {}, "completed_takes": 0}
        else:
            session_data = {"takes": {}, "completed_takes": 0}

        # Resolve emotion tone instruction
        tone_val = req.emotion.lower()
        instruction = f"[{tone_val}]"
        clean_text = req.text.strip()
        gender = req.gender or session_data.get("gender") or "female"
        donor_set = req.donor_set or session_data.get("donor_set")

        # One entry means the classic single-file take; more means the generation is
        # shared and only the assembly differs between the files written.
        specs = req.variants or [
            ABVariantSpec(
                id="default",
                label="Post-Processed" if req.post_process else "Raw",
                post_process=req.post_process,
                params=req.post_process_params,
            )
        ]
        single = len(specs) == 1 and not req.variants

        def _name(spec: ABVariantSpec) -> str:
            base = f"{req.emotion}_take_{req.take_idx}"
            return f"{base}.wav" if single else f"{base}__{spec.id}.wav"

        filename = _name(specs[0])
        out_wav_path = s_dir / filename

        try:
            pre_vc_sink: dict = {}
            debug_sink: dict = {}
            takes, sr_out, _tones = thonburian_service.synthesize_variants(
                [clean_text],
                variants=[
                    {
                        "id": v.id,
                        "post_process": v.post_process,
                        "params": v.params.model_dump(exclude_none=True) if v.params else None,
                    }
                    for v in specs
                ],
                speaker_id=req.speaker_id,
                gender=gender,
                donor_set=donor_set,
                cfg_value=req.cfg_value,
                inference_timesteps=req.inference_timesteps,
                tones=[tone_val],
                breaks=[False],
                lora_mode=req.lora_mode or "on",
                pre_vc_sink=pre_vc_sink,
                debug_sink=debug_sink,
            )

            # Save the pre-SeedVC (Thonburian F5) clip so the UI can play it too.
            pre_vc_filename = None
            pre_vc_url = None
            if pre_vc_sink.get("wav"):
                pre_vc_filename = f"{req.emotion}_take_{req.take_idx}__thon_preVC.wav"
                (s_dir / pre_vc_filename).write_bytes(pre_vc_sink["wav"])
                pre_vc_url = f"/api/benchmark/audio/{req.session_id}/{pre_vc_filename}"

            variant_records = []
            for take, spec in zip(takes, specs):
                v_name = _name(spec)
                (s_dir / v_name).write_bytes(take["wav"])
                audio_arr, sr = sf.read(io.BytesIO(take["wav"]), dtype="float32")
                variant_records.append({
                    "id": spec.id,
                    "label": spec.label,
                    "filename": v_name,
                    "audio_url": f"/api/benchmark/audio/{req.session_id}/{v_name}",
                    "metrics": extract_audio_metrics(audio_arr, sr),
                })

            # Optional F0-compare trio (baseline / A / B): one F5 generation, three
            # SeedVC treatments, so the emotion-vs-register trade-off is audible.
            f0_records: List[dict] = []
            f0_diag = None
            if getattr(req, "f0_compare", False):
                cmp = thonburian_service.render_f0_compare(
                    clean_text,
                    emotion=tone_val,
                    speaker_id=req.speaker_id,
                    gender=gender,
                    donor_set=donor_set,
                )
                f0_diag = cmp.get("diag")
                for m in cmp.get("modes", []):
                    f_name = f"{req.emotion}_take_{req.take_idx}__f0_{m['id']}.wav"
                    (s_dir / f_name).write_bytes(m["wav"])
                    f_arr, f_sr = sf.read(io.BytesIO(m["wav"]), dtype="float32")
                    f0_records.append({
                        "id": m["id"],
                        "label": m["label"],
                        "filename": f_name,
                        "audio_url": f"/api/benchmark/audio/{req.session_id}/{f_name}",
                        "metrics": extract_audio_metrics(f_arr, f_sr),
                        "auto_f0_adjust": bool(m["auto_f0_adjust"]),
                        "semi_tone_shift": int(m["semi_tone_shift"]),
                    })

            # The first variant stands in for the take at the top level, so every
            # existing reader -- the results matrix, the ZIP export, old session
            # files -- keeps working without knowing about variants at all.
            metrics = variant_records[0]["metrics"]
            elapsed_s = round(time.time() - start_t, 2)

            take_record = {
                "emotion": req.emotion,
                "take_idx": req.take_idx,
                "instruction": instruction,
                "spoken_text": clean_text,
                "filename": filename,
                "audio_url": f"/api/benchmark/audio/{req.session_id}/{filename}",
                "metrics": metrics,
                "pre_vc_url": pre_vc_url,
                "pre_vc_filename": pre_vc_filename,
                "model_input": debug_sink or None,
                "variants": variant_records,
                "f0_variants": f0_records,
                "f0_diag": f0_diag,
                "elapsed_s": elapsed_s,
                "error": None,
                "timestamp": datetime.now().isoformat(),
            }

            # Update session JSON
            take_key = f"{req.emotion}_{req.take_idx}"
            session_data.setdefault("takes", {})[take_key] = take_record
            session_data["completed_takes"] = len([t for t in session_data["takes"].values() if not t.get("error")])
            meta_file.write_text(json.dumps(session_data, ensure_ascii=False, indent=2), encoding="utf-8")

            return BenchmarkTakeResult(
                session_id=req.session_id,
                emotion=req.emotion,
                take_idx=req.take_idx,
                instruction=instruction,
                spoken_text=clean_text,
                audio_url=take_record["audio_url"],
                filename=filename,
                metrics=metrics,
                pre_vc_url=pre_vc_url,
                pre_vc_filename=pre_vc_filename,
                model_input=debug_sink or None,
                variants=[BenchmarkTakeVariant(**v) for v in variant_records],
                f0_variants=[BenchmarkF0Variant(**v) for v in f0_records],
                f0_diag=f0_diag,
                elapsed_s=elapsed_s,
                error=None,
            )

        except Exception as e:
            elapsed_s = round(time.time() - start_t, 2)
            err_msg = str(e)
            take_record = {
                "emotion": req.emotion,
                "take_idx": req.take_idx,
                "instruction": instruction,
                "spoken_text": clean_text,
                "filename": "",
                "audio_url": "",
                "metrics": None,
                "elapsed_s": elapsed_s,
                "error": err_msg,
                "timestamp": datetime.now().isoformat(),
            }
            take_key = f"{req.emotion}_{req.take_idx}"
            session_data.setdefault("takes", {})[take_key] = take_record
            meta_file.write_text(json.dumps(session_data, ensure_ascii=False, indent=2), encoding="utf-8")

            return BenchmarkTakeResult(
                session_id=req.session_id,
                emotion=req.emotion,
                take_idx=req.take_idx,
                instruction=instruction,
                spoken_text=clean_text,
                audio_url="",
                filename="",
                metrics=None,
                elapsed_s=elapsed_s,
                error=err_msg,
            )

    def get_audio_path(self, session_id: str, filename: str) -> Optional[Path]:
        """Resolve and validate WAV audio path."""
        s_dir = self._session_dir(session_id)
        clean_file = Path(filename).name
        audio_path = s_dir / clean_file
        if audio_path.is_file() and audio_path.suffix.lower() == ".wav":
            return audio_path
        return None

    def list_sessions(self) -> List[BenchmarkSessionSummary]:
        """List all previous benchmark runs sorted by creation date."""
        sessions: List[BenchmarkSessionSummary] = []
        if not self.runs_dir.exists():
            return sessions

        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            meta_file = d / "session.json"
            if not meta_file.exists():
                continue
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                sessions.append(
                    BenchmarkSessionSummary(
                        session_id=data.get("session_id", d.name),
                        name=data.get("name", d.name),
                        created_at=data.get("created_at", ""),
                        speaker_id=data.get("speaker_id"),
                        text=data.get("text", ""),
                        emotions=data.get("emotions", []),
                        repeats=data.get("repeats", 3),
                        total_takes=data.get("total_takes", 0),
                        completed_takes=data.get("completed_takes", len(data.get("takes", {}))),
                        params=data.get("params", {}),
                    )
                )
            except Exception:
                continue

        return sessions

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full details of a benchmark session."""
        s_dir = self._session_dir(session_id)
        meta_file = s_dir / "session.json"
        if not meta_file.exists():
            return None
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def export_session_zip(self, session_id: str) -> Optional[bytes]:
        """Package session audio WAVs and CSV/JSON reports into a downloadable ZIP."""
        s_dir = self._session_dir(session_id)
        meta_file = s_dir / "session.json"
        if not meta_file.exists():
            return None

        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            return None

        # Build CSV Summary
        csv_lines = [
            "session_id,speaker_id,emotion,take_idx,duration_s,rms,energy_dbfs,f0_med_hz,f0_spread_st,instruction,filename"
        ]
        takes = data.get("takes", {})
        for k, v in sorted(takes.items()):
            if v.get("error"):
                continue
            m = v.get("metrics") or {}
            instr = f'"{v.get("instruction", "")}"'
            csv_lines.append(
                f"{session_id},{data.get('speaker_id', 'base')},{v.get('emotion')},{v.get('take_idx')},"
                f"{m.get('dur_s', '')},{m.get('rms', '')},{m.get('energy_dbfs', '')},{m.get('f0_med_hz', '')},"
                f"{m.get('f0_spread_st', '')},{instr},{v.get('filename')}"
            )
        csv_content = "\n".join(csv_lines)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("session.json", json.dumps(data, ensure_ascii=False, indent=2))
            zf.writestr("report.csv", csv_content)
            for wav_file in s_dir.glob("*.wav"):
                zf.write(wav_file, arcname=f"audio/{wav_file.name}")

        return buf.getvalue()


benchmark_service = BenchmarkService()
