from __future__ import annotations

import re

from app.data_processing.chunking import split_into_sentences


SECTION_PATTERNS = [
    ("history", r"\bhistory:\s*"),
    ("symptoms", r"\bsymptoms:\s*"),
    ("medications", r"\bmedications:\s*"),
    ("procedures", r"\bprocedures:\s*"),
    ("plan", r"\bplan:\s*"),
    ("assessment", r"\bassessment:\s*"),
]


def segment_document(text: str) -> list[dict[str, object]]:
    """Return lightweight sections with names and text spans."""

    lowered = text.lower()
    sections: list[dict[str, object]] = []

    matches: list[tuple[str, int]] = []
    for name, pattern in SECTION_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            matches.append((name, match.start()))

    if matches:
        matches.sort(key=lambda item: item[1])
        for index, (name, start_pos) in enumerate(matches):
            end_pos = matches[index + 1][1] if index + 1 < len(matches) else len(text)
            section_text = text[start_pos:end_pos].strip()
            sections.append(
                {
                    "section_name": name,
                    "start_char": start_pos,
                    "end_char": end_pos,
                    "text": section_text,
                }
            )
        return sections

    sentences = split_into_sentences(text)
    cursor = 0
    for index, sentence in enumerate(sentences, start=1):
        start_pos = text.find(sentence, cursor)
        if start_pos < 0:
            start_pos = cursor
        end_pos = start_pos + len(sentence)
        cursor = end_pos
        sections.append(
            {
                "section_name": f"sentence_block_{index}",
                "start_char": start_pos,
                "end_char": end_pos,
                "text": sentence,
            }
        )
    return sections
