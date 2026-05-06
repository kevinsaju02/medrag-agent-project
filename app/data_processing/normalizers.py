from __future__ import annotations

import re
import unicodedata


MULTISPACE_PATTERN = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.;:])")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving readable sentence spacing."""

    return MULTISPACE_PATTERN.sub(" ", text).strip()


def normalize_clinical_text(text: str) -> str:
    """Normalize text for downstream indexing and ML without changing meaning."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", " ")
    normalized = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", normalized)
    normalized = normalize_whitespace(normalized)
    return normalized
