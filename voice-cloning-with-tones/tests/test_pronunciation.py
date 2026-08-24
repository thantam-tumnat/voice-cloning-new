import json

import pytest

from app.services import pronunciation as pron
from app.services.pronunciation import apply_pronunciation, load_dictionary, save_dictionary


@pytest.fixture
def dict_file(tmp_path, monkeypatch):
    """Point the service at a throwaway dictionary."""
    path = tmp_path / "pronunciation.json"
    monkeypatch.setattr(pron.settings, "pronunciation_path", str(path), raising=False)
    pron._cache.update(path=None, mtime=None, mapping={}, max_span=0)
    yield path
    pron._cache.update(path=None, mtime=None, mapping={}, max_span=0)


def write(path, mapping):
    path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def test_replaces_the_word_it_was_given():
    assert apply_pronunciation("ทั้งสองไฟล์มีโครงสร้าง", {"ไฟล์": "ฟาย"}) == "ทั้งสองฟายมีโครงสร้าง"


def test_does_not_touch_a_longer_word_that_contains_the_key():
    """The trap this design exists for.

    โปรไฟล์ is genuinely /proː faj/ with the SHORT vowel, so a substring replace
    would mispronounce it in the course of fixing ไฟล์.
    """
    assert apply_pronunciation("โปรไฟล์ของฉัน", {"ไฟล์": "ฟาย"}) == "โปรไฟล์ของฉัน"


def test_handles_both_words_in_one_sentence():
    out = apply_pronunciation("โปรไฟล์และไฟล์", {"ไฟล์": "ฟาย"})
    assert out == "โปรไฟล์และฟาย"


@pytest.mark.parametrize("text", ["ไฟฟ้าดับ", "ไฟแดง", "ไฟไหม้"])
def test_leaves_unrelated_words_sharing_a_prefix_alone(text):
    assert apply_pronunciation(text, {"ไฟล์": "ฟาย"}) == text


def test_replaces_every_occurrence():
    out = apply_pronunciation("เปิดไฟล์ แล้วปิดไฟล์", {"ไฟล์": "ฟาย"})
    assert out.count("ฟาย") == 2
    assert "ไฟล์" not in out


def test_longest_key_wins():
    mapping = {"ไฟล์": "ฟาย", "ไฟล์เอกสาร": "ฟายเอกะสาน"}
    assert apply_pronunciation("เปิดไฟล์เอกสารนี้", mapping) == "เปิดฟายเอกะสานนี้"


def test_empty_dictionary_and_empty_text_are_no_ops():
    assert apply_pronunciation("ทั้งสองไฟล์", {}) == "ทั้งสองไฟล์"
    assert apply_pronunciation("", {"ไฟล์": "ฟาย"}) == ""


def test_untokenizable_text_is_returned_unchanged(monkeypatch):
    """Never corrupt text we could not split losslessly."""
    monkeypatch.setattr(pron, "_tokenize", lambda text: None)
    assert apply_pronunciation("ทั้งสองไฟล์", {"ไฟล์": "ฟาย"}) == "ทั้งสองไฟล์"


# --------------------------------------------------------------------------- #
# Loading & persistence
# --------------------------------------------------------------------------- #

def test_missing_file_yields_an_empty_dictionary(dict_file):
    assert load_dictionary() == {}


def test_loads_entries_from_disk(dict_file):
    write(dict_file, {"ไฟล์": "ฟาย"})
    assert load_dictionary() == {"ไฟล์": "ฟาย"}


def test_malformed_file_does_not_break_synthesis(dict_file, capsys):
    dict_file.write_text("{not json", encoding="utf-8")
    assert load_dictionary() == {}
    assert "WARNING" in capsys.readouterr().err


def test_non_object_json_is_rejected(dict_file):
    write(dict_file, ["ไฟล์", "ฟาย"])
    assert load_dictionary() == {}


def test_edits_are_picked_up_without_a_restart(dict_file):
    write(dict_file, {"ไฟล์": "ฟาย"})
    assert load_dictionary() == {"ไฟล์": "ฟาย"}

    # Bump mtime so the change is visible even on a coarse clock.
    write(dict_file, {"ไฟล์": "ฟาย", "เมล์": "เมว"})
    import os
    st = dict_file.stat()
    os.utime(dict_file, (st.st_atime, st.st_mtime + 10))

    assert load_dictionary() == {"ไฟล์": "ฟาย", "เมล์": "เมว"}


def test_save_round_trips_and_drops_blank_keys(dict_file):
    saved = save_dictionary({"ไฟล์": "ฟาย", "  ": "x", "เมล์": "เมว"})
    assert saved == {"ไฟล์": "ฟาย", "เมล์": "เมว"}
    assert json.loads(dict_file.read_text(encoding="utf-8")) == saved


def test_saved_file_is_readable_thai_not_escaped(dict_file):
    save_dictionary({"ไฟล์": "ฟาย"})
    assert "ไฟล์" in dict_file.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Synthesis path
# --------------------------------------------------------------------------- #

def test_prepare_text_applies_overrides(dict_file):
    from app.services.siangtts_service import prepare_text

    write(dict_file, {"ไฟล์": "ฟาย"})
    assert prepare_text("ทั้งสองไฟล์มีโครงสร้าง") == "ทั้งสองฟายมีโครงสร้าง"


def test_prepare_text_never_rewrites_the_style_instruction(dict_file):
    """The parenthetical is direction for the engine, not speech."""
    from app.services.siangtts_service import prepare_text

    write(dict_file, {"ไฟล์": "ฟาย", "voice": "VOICE"})
    out = prepare_text("(Sad and melancholic voice, slight sighs)ทั้งสองไฟล์มี")
    assert out.startswith("(Sad and melancholic voice, slight sighs)")
    assert "VOICE" not in out
    assert out.endswith("ทั้งสองฟายมี")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

def test_get_and_put_endpoints(dict_file):
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    assert c.get("/pronunciation").json()["entries"] == {}

    r = c.put("/pronunciation", json={"entries": {"ไฟล์": "ฟาย"}})
    assert r.status_code == 200
    assert r.json()["entries"] == {"ไฟล์": "ฟาย"}

    assert c.get("/pronunciation").json()["entries"] == {"ไฟล์": "ฟาย"}
