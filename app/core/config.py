from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    app_name: str = Field(default="MedRAG Agent")
    app_env: str = Field(default="development")
    app_version: str = Field(default="0.1.0")
    log_level: str = Field(default="INFO")

    data_dir: Path = Field(default=ROOT_DIR / "data")
    models_dir: Path = Field(default=ROOT_DIR / "models")
    default_document_type: str = Field(default="clinical_note")
    default_vector_store: str = Field(default="faiss")
    default_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    default_risk_model_name: str = Field(default="tfidf_logistic_regression")

    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model_name: str = Field(default="mistral")
    ollama_timeout_seconds: float = Field(default=1.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object for consistent app-wide configuration."""

    if "DATA_DIR" in os.environ:
        os.environ["DATA_DIR"] = str(Path(os.environ["DATA_DIR"]).expanduser())

    return Settings()
