from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.enums import ConfidenceLevel, DocumentType, RiskLevel


class ExtractedFields(BaseModel):
    diagnosis: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    follow_up_actions: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    value: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    evidence_text: str = Field(..., min_length=1)
    retrieval_score: float | None = Field(default=None)


class PredictionOutput(BaseModel):
    risk_level: RiskLevel
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    model_name: str = Field(..., min_length=1)


class ValidationOutput(BaseModel):
    unsupported_fields: list[str] = Field(default_factory=list)
    missing_expected_fields: list[str] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)
    contradiction_flags: list[str] = Field(default_factory=list)
    overall_confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)


class AnalysisMetadata(BaseModel):
    llm_model_name: str | None = None
    embedding_model_name: str | None = None
    risk_model_name: str | None = None
    retrieval_top_k: int | None = None
    processing_time_ms: int | None = None


class AnalysisResponse(BaseModel):
    document_id: str = Field(..., min_length=1)
    document_type: DocumentType
    extracted_fields: ExtractedFields
    citations: dict[str, list[Citation]] = Field(default_factory=dict)
    prediction: PredictionOutput
    validation: ValidationOutput
    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata)
