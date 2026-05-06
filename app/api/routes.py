from __future__ import annotations

from fastapi import APIRouter

from app.schemas.api import AnalyzeRequest
from app.schemas.output import AnalysisResponse
from app.services.analyze_service import analyze_document


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalyzeRequest) -> AnalysisResponse:
    return analyze_document(request)
