from __future__ import annotations

from app.schemas.api import AnalyzeRequest
from app.services.analyze_service import analyze_document


def test_analyze_document_runs_end_to_end() -> None:
    response = analyze_document(
        AnalyzeRequest(
            text=(
                "Patient is a 67-year-old male with history of hypertension and type 2 diabetes "
                "presenting with shortness of breath and chest discomfort. Started on metoprolol. "
                "ECG ordered. Follow-up recommended."
            )
        )
    )

    assert response.document_id == "ad_hoc_001"
    assert response.document_type.value in {"clinical_note", "other_medical_text"}
    assert "metoprolol" in response.extracted_fields.medications
    assert response.prediction.model_name == "tfidf_logistic_regression"
    assert response.validation.overall_confidence.value in {"low", "medium", "high"}
