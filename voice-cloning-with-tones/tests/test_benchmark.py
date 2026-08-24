from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_benchmark_presets():
    res = client.get("/api/benchmark/presets")
    assert res.status_code == 200
    data = res.json()
    assert "emotions" in data
    assert len(data["emotions"]) == 10
    assert "preset_sentences" in data
    assert len(data["preset_sentences"]) >= 1


def test_benchmark_ui_routes():
    res_test = client.get("/test")
    assert res_test.status_code == 200
    assert "Emotion TTS Benchmark" in res_test.text

    res_bench = client.get("/benchmark")
    assert res_bench.status_code == 200


def test_benchmark_session_lifecycle():
    # Init
    init_res = client.post(
        "/api/benchmark/session/init",
        json={
            "name": "Unit Test Benchmark",
            "text": "ข้อความทดสอบเสียง",
            "emotions": ["neutral", "happy"],
            "repeats": 2,
            "intensity": 2,
        },
    )
    assert init_res.status_code == 200
    init_data = init_res.json()
    session_id = init_data["session_id"]
    assert session_id.startswith("bench_")
    assert init_data["total_takes"] == 4

    # List
    list_res = client.get("/api/benchmark/sessions")
    assert list_res.status_code == 200
    sessions = list_res.json()
    assert any(s["session_id"] == session_id for s in sessions)

    # Get Details
    get_res = client.get(f"/api/benchmark/sessions/{session_id}")
    assert get_res.status_code == 200
    details = get_res.json()
    assert details["name"] == "Unit Test Benchmark"
