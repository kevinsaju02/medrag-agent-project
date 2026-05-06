from __future__ import annotations

import re
from typing import Iterable


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    """Split clinical-style text into rough sentence units."""

    text = text.strip()
    if not text:
        return []

    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    return sentences or [text]


def chunk_text(
    text: str,
    document_id: str,
    target_words: int = 120,
    overlap_sentences: int = 1,
) -> list[dict[str, object]]:
    """Create sentence-aware overlapping chunks for retrieval."""

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[dict[str, object]] = []
    current_sentences: list[str] = []
    current_word_count = 0

    def flush_chunk(chunk_sentences: Iterable[str]) -> None:
        chunk_text_value = " ".join(chunk_sentences).strip()
        if not chunk_text_value:
            return

        chunk_id = f"{document_id}_chunk_{len(chunks) + 1:02d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text_value,
                "word_count": len(chunk_text_value.split()),
            }
        )

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_sentences and current_word_count + sentence_words > target_words:
            flush_chunk(current_sentences)

            overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
            current_sentences = list(overlap)
            current_word_count = sum(len(item.split()) for item in current_sentences)

        current_sentences.append(sentence)
        current_word_count += sentence_words

    if current_sentences:
        flush_chunk(current_sentences)

    return chunks
