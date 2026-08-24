from typing import List
from pythainlp.tokenize import sent_tokenize
from app.config import settings


def segment_text(text: str, engine: str | None = None) -> List[str]:
    """
    Segment Thai text into sentence/clause units while strictly preserving
    all original whitespaces, punctuation, digits, and emojis.
    
    Guarantees: "".join(result) == text
    """
    if not text:
        return []
    
    engine_name = engine or settings.segmenter_engine

    try:
        raw_sentences = sent_tokenize(text, engine=engine_name)
    except Exception:
        # Fallback to whitespace/newline or whole text
        try:
            raw_sentences = sent_tokenize(text, engine="whitespace+newline")
        except Exception:
            raw_sentences = [text]

    # Filter out empty strings from raw_sentences if any
    raw_tokens = [s for s in raw_sentences if s]

    if len(raw_tokens) <= 1:
        return [text]

    clauses: List[str] = []
    current_idx = 0
    total_len = len(text)

    for i in range(len(raw_tokens)):
        token = raw_tokens[i]
        
        # If this is the last token, take everything remaining
        if i == len(raw_tokens) - 1:
            clause = text[current_idx:]
            if clause:
                clauses.append(clause)
            break

        # Find the next token start position in text after current_idx
        next_token = raw_tokens[i + 1]
        next_pos = text.find(next_token, current_idx + len(token))

        if next_pos == -1:
            # If next token not found directly (rare edge case with tokenizers),
            # try finding next_token anywhere from current_idx + 1
            next_pos = text.find(next_token, current_idx + 1)

        if next_pos == -1:
            # If still not found, take the rest and stop
            clause = text[current_idx:]
            if clause:
                clauses.append(clause)
            break
        else:
            clause = text[current_idx:next_pos]
            if clause:
                clauses.append(clause)
            current_idx = next_pos

    # If clauses was empty or didn't cover everything, fallback
    if not clauses:
        return [text]

    # Invariant safety check
    if "".join(clauses) != text:
        # If somehow reconstruction does not equal original text, fallback to [text]
        return [text]

    return clauses
