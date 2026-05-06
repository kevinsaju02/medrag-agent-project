from __future__ import annotations

from app.schemas.api import AnalyzeRequest
from app.services.analyze_service import run_analysis_pipeline


def test_run_analysis_pipeline_exposes_debug_trace() -> None:
    state = run_analysis_pipeline(
        AnalyzeRequest(
            text=(
                "Patient is a 67-year-old male with history of hypertension and type 2 diabetes "
                "presenting with shortness of breath and chest discomfort. Started on metoprolol. "
                "ECG ordered. Follow-up recommended."
            )
        )
    )

    assert state["debug_trace"]
    assert any(item["step"] == "validate" for item in state["debug_trace"])
    assert state["metadata"]["processing_time_ms"] >= 0
