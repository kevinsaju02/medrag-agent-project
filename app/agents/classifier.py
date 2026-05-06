from __future__ import annotations

from app.schemas.enums import DocumentType


CLINICAL_KEYWORDS = {
    "patient",
    "history",
    "symptoms",
    "medications",
    "assessment",
    "plan",
    "follow-up",
    "follow up",
}


def classify_document(text: str) -> dict[str, object]:
    lowered = text.lower()
    keyword_hits = sum(keyword in lowered for keyword in CLINICAL_KEYWORDS)

    if keyword_hits >= 2:
        return {
            "document_type": DocumentType.CLINICAL_NOTE,
            "confidence": 0.9,
            "rationale": "Detected multiple clinical-note keywords and note-style phrasing.",
        }

    return {
        "document_type": DocumentType.OTHER_MEDICAL_TEXT,
        "confidence": 0.55,
        "rationale": "Limited clinical-note structure detected; using broader medical-text fallback.",
    }
