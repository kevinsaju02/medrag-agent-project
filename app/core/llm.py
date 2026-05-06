from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.core.config import get_settings


@dataclass
class LLMResult:
    content: str
    used_fallback: bool
    provider: str


class OllamaClient:
    """Small Ollama client with graceful fallback support."""

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model_name = settings.llm_model_name
        self.timeout_seconds = settings.ollama_timeout_seconds

    def generate(self, prompt: str) -> LLMResult:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)
                return LLMResult(
                    content=parsed.get("response", ""),
                    used_fallback=False,
                    provider=f"ollama:{self.model_name}",
                )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return LLMResult(
                content="",
                used_fallback=True,
                provider="fallback",
            )
