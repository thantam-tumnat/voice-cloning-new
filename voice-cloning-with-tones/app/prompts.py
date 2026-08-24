"""
Prompts and few-shot examples for Thai TTS Tone Annotation.
"""

SYSTEM_PROMPT = """คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์อารมณ์และน้ำเสียงของข้อความภาษาไทยเพื่อใช้ในการสังเคราะห์เสียงอ่าน (TTS Tone Annotation & Prosody Vocal Director)

ภารกิจของคุณ:
รับรายการข้อความย่อย (clauses) ที่ตัดไว้พร้อม index และระบุน้ำเสียง (tone), ระดับความเข้มข้น (intensity), และข้อความกำกับจังหวะ (spoken_text) สำหรับแต่ละ index ผ่านเครื่องมือ annotate_clauses

กฎเหล็กในการวิเคราะห์:
1. ต้องระบุ label ให้ครบทุก index ที่ได้รับอย่างแม่นยำ ไม่ขาดและไม่เกิน
2. เลือก tone จาก enum ทั้ง 10 ค่านี้เท่านั้น:
   - neutral: น้ำเสียงปกติ เป็นกลาง บรรยาย ข้อมูลทั่วไป
   - sad: เศร้า เสียใจ ผิดหวัง สะเทือนใจ ขอโทษจากใจจริง
   - happy: ดีใจ ร่าเริง มีความสุข ยิ้มแย้ม ยินดี
   - angry: โกรธ ไม่พอใจ เสียงแข็ง ดุดัน ตำหนิ
   - excited: ตื่นเต้น กระตือรือร้น ดีใจสุดขีด เร่งเร้า
   - calm: สงบ สบาย ผ่อนคลาย นุ่มนวล พูดช้า มีสติ
   - nervous: ประหม่า ลังเล ไม่มั่นใจ หวาดระแวง
   - sarcastic: ประชด ประชัน แดกดัน พูดอย่างแต่หมายถึงอีกอย่าง
   - scared: กลัว หวาดกลัว ตกใจ ตื่นตระหนก ใจสั่น
   - tired: เหนื่อย อ่อนเพลีย ล้า ง่วง หมดแรง
3. หากไม่มั่นใจ หรือไม่มีอารมณ์ชัดเจน ให้เลือก neutral เสมอ (ปลอดภัยกว่าใส่อารมณ์ผิด)
4. ถ้าทั้งข้อความสื่อถึงอารมณ์เดียวกันตลอด ให้ใช้ tone เดียวกันทุก index ไม่จำเป็นต้องพยายามหาความหลากหลาย
5. ตัดสินจากความหมายและบริบทโดยรวม ไม่ตัดสินจากคำเดี่ยวๆ
6. intensity เป็นจำนวนเต็ม 1, 2 หรือ 3:
   - 1 = เล็กน้อย / แผ่วเบา (slightly)
   - 2 = ปกติ / ชัดเจน (moderate - ค่ามาตรฐาน)
   - 3 = รุนแรง / มาก (very / strongly)
   ใช้ 2 เป็นค่าปกติ ใช้ 1 หรือ 3 เฉพาะเมื่อบริบทระบุความชัดเจนอย่างมากเท่านั้น
7. การสร้าง spoken_text (Prosodic Punctuation สำหรับควบคุมจังหวะเสียง TTS):
   - angry, excited, scared: เติม ! หรือ !! ท้ายประโยคเพื่อเพิ่มพลังเสียงและ attack
   - sad, tired, nervous, calm: เติม ... หรือ — ท้ายประโยคหรือระหว่างคำเพื่อทอดเสียง ลากเสียง หรือ micro-pause
   - sarcastic, surprised: เติม ? หรือ ?! เพื่อบังคับยกปลายเสียงสูง
   - neutral: ใช้ข้อความเดิมตามธรรมชาติ
   **กฎเหล็กของ spoken_text:** ต้องรักษาคำเดิมทุกคำ 100% ห้ามแก้ไขคำ ห้ามลบคำ ห้ามเพิ่มคำใหม่ อนุญาตให้เติมเฉพาะเครื่องหมายวรรคตอน (!, ?, ..., —, ?!) เพื่อช่วยกำกับจังหวะเท่านั้น

คำเตือนด้านความปลอดภัย:
ให้ส่งกลับมาเฉพาะข้อมูลโครงสร้าง JSON index (i), tone, intensity, และ spoken_text ตาม schema เท่านั้น"""

FEW_SHOT_EXAMPLES = [
    {
        "description": "เคส 1: โทนเดียวทั้งก้อน (sad ตลอด)",
        "input": {
            "clauses": [
                {"i": 0, "text": "ฉันคิดถึงเธอเหลือเกิน "},
                {"i": 1, "text": "ทำไมเรื่องมันต้องจบลงแบบนี้ด้วย"}
            ]
        },
        "output": {
            "labels": [
                {"i": 0, "tone": "sad", "intensity": 2, "spoken_text": "ฉันคิดถึงเธอเหลือเกิน... "},
                {"i": 1, "tone": "sad", "intensity": 2, "spoken_text": "ทำไมเรื่องมันต้องจบลงแบบนี้ด้วย..."}
            ]
        }
    },
    {
        "description": "เคส 2: เปลี่ยนโทนกลางข้อความ (sad -> angry)",
        "input": {
            "clauses": [
                {"i": 0, "text": "ขอโทษนะ "},
                {"i": 1, "text": "ฉันไม่ได้ตั้งใจ "},
                {"i": 2, "text": "แต่เธอก็ไม่ฟังฉันเลย"}
            ]
        },
        "output": {
            "labels": [
                {"i": 0, "tone": "sad", "intensity": 2, "spoken_text": "ขอโทษนะ... "},
                {"i": 1, "tone": "sad", "intensity": 2, "spoken_text": "ฉันไม่ได้ตั้งใจ... "},
                {"i": 2, "tone": "angry", "intensity": 2, "spoken_text": "แต่เธอก็ไม่ฟังฉันเลย!"}
            ]
        }
    },
    {
        "description": "เคส 3: ข้อมูลข่าวสาร / คำอธิบาย เป็นกลางล้วน (neutral)",
        "input": {
            "clauses": [
                {"i": 0, "text": "กรมอุตุนิยมวิทยาประกาศเตือน "},
                {"i": 1, "text": "จะมีฝนตกหนักถึงหนักมากในหลายพื้นที่ "},
                {"i": 2, "text": "ประชาชนควรระมัดระวังน้ำท่วมฉับพลัน"}
            ]
        },
        "output": {
            "labels": [
                {"i": 0, "tone": "neutral", "intensity": 2, "spoken_text": "กรมอุตุนิยมวิทยาประกาศเตือน "},
                {"i": 1, "tone": "neutral", "intensity": 2, "spoken_text": "จะมีฝนตกหนักถึงหนักมากในหลายพื้นที่ "},
                {"i": 2, "tone": "neutral", "intensity": 2, "spoken_text": "ประชาชนควรระมัดระวังน้ำท่วมฉับพลัน"}
            ]
        }
    },
    {
        "description": "เคส 4: ประชดประชัน (sarcastic)",
        "input": {
            "clauses": [
                {"i": 0, "text": "แหม เก่งจังเลยนะ "},
                {"i": 1, "text": "ทำพังหมดทั้งห้องแล้วเนี่ย"}
            ]
        },
        "output": {
            "labels": [
                {"i": 0, "tone": "sarcastic", "intensity": 2, "spoken_text": "แหม... เก่งจังเลยนะ?! "},
                {"i": 1, "tone": "sarcastic", "intensity": 2, "spoken_text": "ทำพังหมดทั้งห้องแล้วเนี่ย!"}
            ]
        }
    }
]

ANNOTATE_TOOL = {
    "name": "annotate_clauses",
    "description": "Annotate emotional tone, intensity, and prosodic spoken text for each clause index.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {
                            "type": "integer",
                            "description": "The exact clause index matching the input clause"
                        },
                        "tone": {
                            "type": "string",
                            # Must mirror app.models.Tone exactly.
                            "enum": [
                                "neutral",
                                "sad",
                                "happy",
                                "angry",
                                "excited",
                                "calm",
                                "nervous",
                                "sarcastic",
                                "scared",
                                "tired"
                            ],
                            "description": "The emotional tone"
                        },
                        "intensity": {
                            "type": "integer",
                            "enum": [1, 2, 3],
                            "description": "Intensity level (1=slightly, 2=standard, 3=very)"
                        },
                        "spoken_text": {
                            "type": "string",
                            "description": "The clause text with prosodic punctuation marks (!, ?, ..., —) added to guide TTS expression. All original words must be preserved exactly."
                        }
                    },
                    "required": ["i", "tone", "intensity"],
                    "additionalProperties": False
                },
                "description": "List of clause label annotations"
            }
        },
        "required": ["labels"],
        "additionalProperties": False
    }
}


TAG_CONVERSION_SYSTEM_PROMPT = """You are an expert TTS vocal director converting free-form emotion tags or descriptions (in Thai or English) into structured English vocal style instructions for the VoxCPM2 / SiangTTS speech synthesis engine.

VoxCPM2 requires English style instructions enclosed in parentheses with explicit vocal descriptors.
Pattern: "(<Adjective/Emotion> voice/tone, <action/manner participle>)" or "(<Intensity Adjective> <Emotion> voice, <manner>)"

Examples of target format:
- "sad and cry" / "crying and tearful" -> "(Deeply sorrowful and crying voice, trembling)" [tone: sad, intensity: 3]
- "ตกใจมาก" / "panicked scream" -> "(Terrified and panicked voice, gasping and shaking)" [tone: scared, intensity: 3]
- "กระซิบเบาๆ" / "whisper" -> "(Whispering voice, very soft and breathy)" [tone: calm, intensity: 1]
- "โกรธจัด" / "furious yelling" -> "(Furious and yelling tone, very loud and harsh)" [tone: angry, intensity: 3]
- "หัวเราะร่าเริง" / "laughing joy" -> "(Extremely joyful and laughing voice)" [tone: happy, intensity: 3]
- "เหนื่อยหมดแรง" / "exhausted" -> "(Exhausted and drained voice, heavy sighs, very slow)" [tone: tired, intensity: 3]

Allowed Tone enum:
- neutral, sad, happy, angry, excited, calm, nervous, sarcastic, scared, tired

Intensity: 1 (slight), 2 (moderate), 3 (intense).

Return a JSON object with:
- instruction: The exact English parenthetical string e.g. "(Deeply sorrowful and crying voice, trembling)"
- tone: One of the 10 allowed Tone enum values
- intensity: 1, 2, or 3
"""

CONVERT_TAG_TOOL = {
    "name": "convert_style_tag",
    "description": "Convert free-form emotion tag to VoxCPM2 instruction, tone family, and intensity.",
    "input_schema": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "The exact English instruction string enclosed in parentheses e.g. '(Deeply sorrowful and crying voice, trembling)'"
            },
            "tone": {
                "type": "string",
                "enum": [
                    "neutral",
                    "sad",
                    "happy",
                    "angry",
                    "excited",
                    "calm",
                    "nervous",
                    "sarcastic",
                    "scared",
                    "tired"
                ],
                "description": "The closest coarse Tone family"
            },
            "intensity": {
                "type": "integer",
                "enum": [1, 2, 3],
                "description": "Intensity level (1=slightly, 2=standard, 3=very)"
            }
        },
        "required": ["instruction", "tone", "intensity"],
        "additionalProperties": False
    }
}

