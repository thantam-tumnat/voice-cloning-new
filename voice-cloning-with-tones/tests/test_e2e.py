import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)


def make_mock_gemini_response(labels):
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"labels": labels})
    return mock_resp


def make_mock_anthropic_response(labels):
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "annotate_clauses"
    mock_block.input = {"labels": labels}

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    return mock_resp


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "thai-tts-tone-annotation"


def test_root_ui_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Thai TTS Tone Annotation" in response.text


@patch("app.main.segment_text")
@patch("app.annotator.Annotator.get_gemini_client")
def test_annotate_endpoint_gemini_success(mock_get_gemini, mock_segment_text):
    """Test default Gemini provider annotation success."""
    mock_client = MagicMock()
    mock_get_gemini.return_value = mock_client

    text = "ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย"
    mock_segment_text.return_value = [
        "ขอโทษนะ ",
        "ฉันไม่ได้ตั้งใจ ",
        "แต่เธอก็ไม่ฟังฉันเลย"
    ]
    mock_client.models.generate_content.return_value = make_mock_gemini_response([
        {"i": 0, "tone": "sad", "intensity": 2},
        {"i": 1, "tone": "sad", "intensity": 2},
        {"i": 2, "tone": "angry", "intensity": 2},
    ])

    response = client.post("/annotate", json={"text": text, "model": "gemini-3.6-flash"})
    assert response.status_code == 200
    data = response.json()
    assert data["original"] == text
    assert data["fallback"] is False
    assert data["model_used"] == settings.gemini_model
    
    # 2 merged segments: [sad, "ขอโทษนะ ฉันไม่ได้ตั้งใจ "] and [angry, "แต่เธอก็ไม่ฟังฉันเลย"]
    assert len(data["segments"]) == 2
    assert data["segments"][0]["tone"] == "sad"
    assert data["segments"][0]["text"] == "ขอโทษนะ ฉันไม่ได้ตั้งใจ "
    assert data["segments"][1]["tone"] == "angry"
    assert data["segments"][1]["text"] == "แต่เธอก็ไม่ฟังฉันเลย"
    
    # Reconstructed text invariant
    reconstructed = "".join(s["text"] for s in data["segments"])
    assert reconstructed == text


@patch("app.main.segment_text")
@patch("app.annotator.Annotator.get_gemini_client")
def test_annotate_endpoint_escalation_success(mock_get_gemini, mock_segment_text):
    """Primary fails, escalation to the escalate model succeeds."""
    mock_client = MagicMock()
    mock_get_gemini.return_value = mock_client

    text = "สวัสดีครับ วันนี้อากาศดีมาก"
    mock_segment_text.return_value = ["สวัสดีครับ ", "วันนี้อากาศดีมาก"]
    
    mock_client.models.generate_content.side_effect = [
        Exception("Gemini Flash rate limit"),
        make_mock_gemini_response([
            {"i": 0, "tone": "happy", "intensity": 2},
            {"i": 1, "tone": "happy", "intensity": 2},
        ])
    ]

    response = client.post("/annotate", json={"text": text, "model": "gemini-3.6-flash"})
    assert response.status_code == 200
    data = response.json()
    assert data["fallback"] is False
    assert data["model_used"] == settings.gemini_escalate_model
    assert len(data["segments"]) == 1
    assert data["segments"][0]["tone"] == "happy"
    assert data["segments"][0]["text"] == text


@patch("app.main.segment_text")
@patch("app.annotator.Annotator.get_gemini_client")
def test_annotate_endpoint_fallback_on_all_failures(mock_get_gemini, mock_segment_text):
    """Both primary and escalation fail -> fallback to neutral without throwing."""
    mock_client = MagicMock()
    mock_get_gemini.return_value = mock_client

    text = "ขอโทษนะ ฉันไม่ได้ตั้งใจ"
    mock_segment_text.return_value = ["ขอโทษนะ ", "ฉันไม่ได้ตั้งใจ"]
    mock_client.models.generate_content.side_effect = [
        Exception("Flash failed"),
        Exception("Pro failed")
    ]

    response = client.post("/annotate", json={"text": text, "model": "gemini-3.6-flash"})
    assert response.status_code == 200
    data = response.json()
    assert data["fallback"] is True
    assert data["model_used"] == "fallback-neutral"
    assert len(data["segments"]) == 1
    assert data["segments"][0]["tone"] == "neutral"
    assert data["segments"][0]["text"] == text


def test_render_endpoint_elevenlabs():
    payload = {
        "segments": [
            {"text": "ขอโทษนะ ฉันไม่ได้ตั้งใจ ", "tone": "sad", "intensity": 2},
            {"text": "แต่เธอก็ไม่ฟังฉันเลย", "tone": "angry", "intensity": 2}
        ],
        "engine": "elevenlabs"
    }
    response = client.post("/render", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "[sad] ขอโทษนะ ฉันไม่ได้ตั้งใจ [angry] แต่เธอก็ไม่ฟังฉันเลย"
    assert data["prompt"] is None


def test_render_endpoint_gemini():
    payload = {
        "segments": [
            {"text": "ขอโทษนะ ฉันไม่ได้ตั้งใจ ", "tone": "sad", "intensity": 2},
            {"text": "แต่เธอก็ไม่ฟังฉันเลย", "tone": "angry", "intensity": 2}
        ],
        "engine": "gemini"
    }
    response = client.post("/render", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย"
    assert "เศร้า สะเทือนใจ" in data["prompt"]
    assert "โกรธ เสียงแข็ง" in data["prompt"]


@patch("app.main.segment_text")
@patch("app.annotator.Annotator.get_gemini_client")
def test_speak_endpoint(mock_get_gemini, mock_segment_text):
    mock_client = MagicMock()
    mock_get_gemini.return_value = mock_client

    text = "ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย"
    mock_segment_text.return_value = [
        "ขอโทษนะ ",
        "ฉันไม่ได้ตั้งใจ ",
        "แต่เธอก็ไม่ฟังฉันเลย"
    ]
    mock_client.models.generate_content.return_value = make_mock_gemini_response([
        {"i": 0, "tone": "sad", "intensity": 2},
        {"i": 1, "tone": "sad", "intensity": 2},
        {"i": 2, "tone": "angry", "intensity": 2},
    ])

    response = client.post("/speak", json={"text": text, "engine": "elevenlabs", "model": "gemini-3.6-flash"})
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "elevenlabs"
    assert data["text"] == "[sad] ขอโทษนะ ฉันไม่ได้ตั้งใจ [angry] แต่เธอก็ไม่ฟังฉันเลย"
    assert data["fallback"] is False


@patch("app.main.segment_text")
@patch("app.annotator.Annotator.get_gemini_client")
def test_annotate_endpoint_custom_model(mock_get_gemini, mock_segment_text):
    mock_client = MagicMock()
    mock_get_gemini.return_value = mock_client

    text = "สวัสดีครับ วันนี้อากาศดีมาก"
    mock_segment_text.return_value = ["สวัสดีครับ ", "วันนี้อากาศดีมาก"]
    mock_client.models.generate_content.return_value = make_mock_gemini_response([
        {"i": 0, "tone": "happy", "intensity": 2},
        {"i": 1, "tone": "happy", "intensity": 2},
    ])

    response = client.post("/annotate", json={"text": text, "model": "gemini-3.6-flash"})
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "gemini-3.6-flash"
    assert data["fallback"] is False
    assert data["attempts"][0]["model"] == "gemini-3.6-flash"
    assert data["attempts"][0]["status"] == "success"


@patch("app.main.segment_text")
@patch("app.annotator.Annotator.get_gemini_client")
def test_annotate_all_failed_diagnostic_json(mock_get_gemini, mock_segment_text):
    mock_client = MagicMock()
    mock_get_gemini.return_value = mock_client

    text = "สวัสดีครับ วันนี้อากาศดีมาก"
    mock_segment_text.return_value = ["สวัสดีครับ ", "วันนี้อากาศดีมาก"]
    mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded")

    response = client.post("/annotate", json={"text": text, "model": "gemini-3.5-flash-lite"})
    assert response.status_code == 200
    data = response.json()
    assert data["fallback"] is True
    assert data["model_used"] == "fallback-neutral"
    assert data["error"] is not None
    assert "429 RESOURCE_EXHAUSTED" in data["error_detail"]
    assert len(data["attempts"]) >= 1
    assert data["attempts"][0]["status"] == "failed"


@patch("app.main.segment_text")
@patch("app.annotator.Annotator.get_openai_client")
def test_annotate_endpoint_openai_qwen(mock_get_openai, mock_segment_text):
    mock_client = MagicMock()
    mock_get_openai.return_value = mock_client

    text = "สวัสดีครับ วันนี้อากาศดีมาก"
    mock_segment_text.return_value = ["สวัสดีครับ ", "วันนี้อากาศดีมาก"]

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "labels": [
            {"i": 0, "tone": "happy", "intensity": 2},
            {"i": 1, "tone": "happy", "intensity": 2},
        ]
    })
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp

    response = client.post("/annotate", json={"text": text, "model": "qwen3.8-27b-fp8"})
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "qwen3.8-27b-fp8"
    assert data["fallback"] is False
    assert len(data["segments"]) == 1
    assert data["segments"][0]["tone"] == "happy"

