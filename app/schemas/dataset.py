from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.enums import DocumentType, RiskLevel
from app.schemas.output import ExtractedFields


class DatasetRecord(BaseModel):
    """Synthetic dataset record used for training and evaluation."""

    document_id: str = Field(..., min_length=1)
    raw_text: str = Field(..., min_length=1)
    document_type: DocumentType = Field(default=DocumentType.CLINICAL_NOTE)
    ground_truth: ExtractedFields = Field(default_factory=ExtractedFields)
    risk_label: RiskLevel
