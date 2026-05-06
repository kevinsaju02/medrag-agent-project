from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging_config import configure_logging


configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="MedRAG Agent API for clinical-note extraction, retrieval, prediction, and validation.",
)
app.include_router(router)
