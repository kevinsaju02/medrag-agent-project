from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


@dataclass
class EmbeddingConfig:
    model_name: str = "fallback-hashing-embeddings"
    dimensions: int = 256


class EmbeddingBackend:
    """Embedding abstraction with a deterministic local fallback backend."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.vectorizer = HashingVectorizer(
            n_features=self.config.dimensions,
            alternate_sign=False,
            norm="l2",
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype("float32")

    @property
    def model_name(self) -> str:
        return self.config.model_name
