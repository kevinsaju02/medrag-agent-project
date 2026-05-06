from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request payload for ad hoc clinical text analysis."""

    document_id: str | None = Field(
        default=None,
        description="Optional external document ID. A generated one can be used if omitted.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Clinical-style text to analyze.",
    )
