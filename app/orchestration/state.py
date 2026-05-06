from __future__ import annotations

from typing import TypedDict

from app.schemas.output import ExtractedFields, PredictionOutput, ValidationOutput


class MedRAGState(TypedDict, total=False):
    document_id: str
    raw_text: str
    normalized_text: str
    document_type: str
    classifier_confidence: float
    classifier_rationale: str
    sections: list[dict]
    retrieval_results: dict[str, list[dict]]
    extracted_fields: ExtractedFields
    citations: dict[str, list[object]]
    prediction: PredictionOutput
    validation: ValidationOutput
    retry_count: int
    next_step: str
    debug_trace: list[dict]
    metadata: dict[str, object]
