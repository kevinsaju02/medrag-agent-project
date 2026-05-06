from __future__ import annotations

import time
from pathlib import Path

from app.core.config import ROOT_DIR, get_settings
from app.orchestration.graph import build_medrag_graph
from app.schemas.api import AnalyzeRequest
from app.schemas.enums import DocumentType
from app.schemas.output import (
    AnalysisMetadata,
    AnalysisResponse,
)


def run_analysis_pipeline(request: AnalyzeRequest) -> dict[str, object]:
    """Run the LangGraph workflow and return the full internal state for debugging."""
    settings = get_settings()
    document_id = request.document_id or "ad_hoc_001"
    graph = build_medrag_graph(
        synthetic_dataset_path=ROOT_DIR / "data" / "synthetic" / "synthetic_clinical_notes.jsonl",
        ml_model_path=ROOT_DIR / "models" / "ml" / "risk_model.joblib",
    )
    start_time = time.perf_counter()
    final_state = graph.invoke(
        {
            "document_id": document_id,
            "raw_text": request.text,
            "metadata": {
                "embedding_model_name": settings.default_embedding_model,
                "risk_model_name": settings.default_risk_model_name,
                "llm_model_name": settings.llm_model_name,
                "retrieval_top_k": 3,
            },
        }
    )
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    final_state["metadata"] = {
        **final_state.get("metadata", {}),
        "processing_time_ms": elapsed_ms,
    }
    return final_state


def analyze_document(request: AnalyzeRequest) -> AnalysisResponse:
    """Run the Day 4 orchestration pipeline and return a validated response."""

    settings = get_settings()
    document_id = request.document_id or "ad_hoc_001"
    final_state = run_analysis_pipeline(request)

    return AnalysisResponse(
        document_id=document_id,
        document_type=DocumentType(final_state["document_type"]),
        extracted_fields=final_state["extracted_fields"],
        citations=final_state["citations"],
        prediction=final_state["prediction"],
        validation=final_state["validation"],
        metadata=AnalysisMetadata(
            llm_model_name=str(final_state.get("metadata", {}).get("extractor_provider", settings.llm_model_name)),
            embedding_model_name=settings.default_embedding_model,
            risk_model_name=settings.default_risk_model_name,
            retrieval_top_k=3,
            processing_time_ms=final_state.get("metadata", {}).get("processing_time_ms"),
        ),
    )
