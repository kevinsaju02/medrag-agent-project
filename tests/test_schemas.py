from __future__ import annotations

from app.schemas.api import AnalyzeRequest
from app.schemas.enums import ConfidenceLevel, DocumentType, RiskLevel
from app.schemas.output import AnalysisResponse, ExtractedFields, PredictionOutput, ValidationOutput


def test_analyze_request_accepts_basic_payload() -> None:
    request = AnalyzeRequest(text="Patient reports chest discomfort.")
    assert request.document_id is None
    assert request.text == "Patient reports chest discomfort."


def test_analysis_response_builds_with_expected_defaults() -> None:
    response = AnalysisResponse(
        document_id="doc_001",
        document_type=DocumentType.CLINICAL_NOTE,
        extracted_fields=ExtractedFields(symptoms=["chest discomfort"]),
        citations={},
        prediction=PredictionOutput(
            risk_level=RiskLevel.MEDIUM,
            risk_probability=0.72,
            model_name="tfidf_logistic_regression",
        ),
        validation=ValidationOutput(overall_confidence=ConfidenceLevel.MEDIUM),
    )

    assert response.document_id == "doc_001"
    assert response.extracted_fields.symptoms == ["chest discomfort"]
    assert response.prediction.risk_level == RiskLevel.MEDIUM
