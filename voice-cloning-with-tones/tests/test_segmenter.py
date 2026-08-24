import pytest
from app.segmenter import segment_text

THAI_TEST_TEXTS = [
    # 1. Plain Thai without spaces
    "ขอโทษนะฉันไม่ได้ตั้งใจแต่เธอก็ไม่ฟังฉันเลย",
    
    # 2. Plain Thai with standard spaces
    "ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย",
    
    # 3. Thai with leading/trailing spaces
    "  สวัสดีครับ ยินดีต้อนรับทุกท่านครับ  ",
    
    # 4. Thai with multiple internal spaces and newlines
    "ข้อความบรรทัดที่ 1\nข้อความบรรทัดที่ 2   และมีช่องว่างหลายช่อง",
    
    # 5. Thai with numbers and decimals
    "ราคาสินค้าชิ้นนี้คือ 1,250.50 บาท และลดอีก 15% ในวันนี้",
    
    # 6. Thai with English words mixed in
    "วันนี้เราจะมาเรียนเรื่อง Machine Learning และ AI เบื้องต้นกันครับ",
    
    # 7. Thai with emojis
    "ยินดีด้วยนะ! 🎉 ขอให้มีความสุขมากๆ ❤️ ยิ้มเยอะๆ นะ 😊",
    
    # 8. Thai with punctuation marks (quotes, parentheses, question marks)
    "เขาพูดว่า \"อย่าทำแบบนั้นนะ!\" แล้วคุณคิดอย่างไร? (โปรดระบุ)",
    
    # 9. Complex compound Thai sentence
    "กรมอุตุนิยมวิทยาประกาศว่า จะมีพายุฤดูร้อนพัดผ่านภาคเหนือและภาคตะวันออกเฉียงเหนือ ขอให้ประชาชนระวังอันตราย",
    
    # 10. Repeated Thai vowels / tonal marks / symbols
    "โหหหหห จริงดิ ไม่น่าเชื่อเลยนะเนี่ย 555555+"
]


@pytest.mark.parametrize("text", THAI_TEST_TEXTS)
def test_segmenter_invariants(text):
    """Invariant: Reconstructed text from segments must be 100% identical to original text."""
    clauses = segment_text(text)
    assert isinstance(clauses, list)
    assert "".join(clauses) == text


def test_segmenter_single_clause():
    """Single short clause returns list of length 1."""
    text = "สวัสดี"
    clauses = segment_text(text)
    assert len(clauses) == 1
    assert clauses[0] == text


def test_segmenter_empty_string():
    """Empty string returns empty list."""
    assert segment_text("") == []
