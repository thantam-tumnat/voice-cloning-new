"""Tests for src/thai_text.py — the ported n8n pyThai + chunker nodes.

The invariants that matter downstream: chunks never exceed MAX_CHARS, the
breakpoint pass adds nothing but spaces, and Thai number/abbreviation expansion
only fires for Thai.
"""

from __future__ import annotations

from src.thai_text import (
    MAX_CHARS,
    chunk_text,
    expand,
    mark_breakpoints,
    prepare_prompt,
)

# A space-free Thai script, the shape real prompts arrive in.
LONG_THAI = (
    "ผู้หญิงหลายคนดูแลคนอื่นเก่งมากอย่าลืมดูแลตัวเองจนวันนึงรู้สึกว่าทำไมหน้าโทรม"
    "เหนื่อยง่ายไม่มั่นใจเหมือนเมื่อก่อนเราเองก็เคยเป็นค่ะทั้งทำงานหนักพักผ่อนน้อย"
    "ดูแลตัวเองไม่สม่ำเสมอจนได้ลองตัวนี้แล้วรู้สึกว่าเป็นอีกตัวนึงที่ช่วยทำให้เรากลับมา"
    "ดูแลตัวเองได้ง่ายขึ้นสิ่งที่ชอบก็คือใช้ง่ายไม่ยุ่งยากเหมาะสำหรับผู้หญิงที่ใช้ชีวิต"
    "เร่งรีบและที่สำคัญคือช่วยให้เรารู้สึกมั่นใจขึ้นเพราะว่าการดูแลตัวเองไม่ใช่เรื่อง"
    "ฟุ่มเฟือยแต่เป็นการรักตัวเอง"
)


# --- expand ---------------------------------------------------------------

def test_expands_thai_numbers():
    assert expand("ราคา 250 บาท", True) == "ราคาสองร้อยห้าสิบบาท"


def test_expands_decimal():
    assert expand("2.5 ลิตร", True) == "สองจุดห้าลิตร"


def test_expands_abbreviations():
    assert "กิโลเมตร" in expand("ระยะ 5 กม.", True)
    assert "กันยายน" in expand("วันที่ 8 ก.ย.", True)


def test_skips_number_expansion_for_non_thai():
    assert expand("save 20 dollars", False) == "save 20 dollars"


def test_thai_digits_normalised_even_when_not_expanding():
    assert expand("๑๒๓", False) == "123"


def test_keeps_space_between_number_and_latin():
    # "500 MB" must keep its space; only Thai-on-both-sides spaces are eaten.
    assert expand("500 MB", False) == "500 MB"


def test_chat_spelling_fixed():
    assert expand("เทอ", True) == "เธอ"


def test_chat_spelling_fix_does_not_touch_real_words():
    assert expand("เทอม", True) == "เทอม"


def test_mai_yamok_repeats_last_word_only():
    # The n8n regex was greedy and produced "ผิวสวยผิวสวย" here.
    assert expand("ผิวสวยๆ", True) == "ผิวสวยสวย"


# --- mark_breakpoints -----------------------------------------------------

def test_breakpoints_are_lossless():
    marked = mark_breakpoints(LONG_THAI)
    assert marked.replace(" ", "") == LONG_THAI.replace(" ", "")


def test_breakpoints_keep_runs_under_chunk_size():
    for run in mark_breakpoints(LONG_THAI).split(" "):
        assert len(run) <= MAX_CHARS


# --- chunk_text -----------------------------------------------------------

def test_short_text_is_one_chunk():
    chunks = chunk_text("สวัสดีค่ะ", prefix="q1")
    assert len(chunks) == 1
    assert chunks[0].filename == "q1_000"
    assert chunks[0].total == 1


def test_chunks_respect_max_chars():
    chunks = chunk_text(mark_breakpoints(LONG_THAI), prefix="q1")
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHARS for c in chunks)


def test_chunk_filenames_are_zero_padded_and_ordered():
    chunks = chunk_text(mark_breakpoints(LONG_THAI), prefix="abc")
    assert [c.filename for c in chunks] == [f"abc_{i:03d}" for i in range(len(chunks))]
    assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))


def test_chunking_is_lossless():
    marked = mark_breakpoints(LONG_THAI)
    joined = "".join(c.text for c in chunk_text(marked, prefix="q"))
    assert joined.replace(" ", "") == LONG_THAI.replace(" ", "")


def test_hard_cut_when_no_spaces_at_all():
    # Worst case: a single space-free run longer than MAX_CHARS reaching the
    # chunker unmarked. It must still terminate and stay within budget.
    chunks = chunk_text("ก" * (MAX_CHARS * 2 + 7), prefix="q")
    assert len(chunks) == 3
    assert all(len(c.text) <= MAX_CHARS for c in chunks)


def test_blank_lines_dropped():
    assert len(chunk_text("สวัสดีค่ะ\n\n\nขอบคุณค่ะ", prefix="q")) == 2


# --- prepare_prompt -------------------------------------------------------

def test_prepare_prompt_marks_and_expands():
    out = prepare_prompt("ราคา 250 บาท", "th")
    assert "สองร้อยห้าสิบ" in out


def test_prepare_prompt_leaves_english_alone():
    assert prepare_prompt("save 20 dollars now", "en") == "save 20 dollars now"


def test_prepare_prompt_empty_input():
    assert prepare_prompt("   \n  ", "th") == ""


def test_prepare_prompt_output_chunks_cleanly():
    chunks = chunk_text(prepare_prompt(LONG_THAI, "th"), prefix="q")
    assert all(len(c.text) <= MAX_CHARS for c in chunks)
