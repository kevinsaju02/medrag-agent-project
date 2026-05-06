from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.retrieval.embeddings import EmbeddingBackend


class FaissVectorStore:
    """Minimal FAISS-backed store for retrieval over document chunks."""

    def __init__(self, embedding_backend: EmbeddingBackend) -> None:
        self.embedding_backend = embedding_backend
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []

    def build(self, chunk_records: list[dict]) -> None:
        texts = [record["text"] for record in chunk_records]
        embeddings = self.embedding_backend.encode(texts)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = chunk_records

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index is None:
            raise ValueError("Vector store has not been built.")

        query_embedding = self.embedding_backend.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        results: list[dict] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            record = self.metadata[int(index)]
            results.append(
                {
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "score": float(score),
                    "document_id": record["document_id"],
                    "field": record.get("field"),
                }
            )
        return results

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        if self.index is None:
            raise ValueError("Vector store has not been built.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        index_path = output_path / "faiss.index"
        metadata_path = output_path / "index_metadata.json"

        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

        return {"index": index_path, "metadata": metadata_path}

    def load(self, index_path: str | Path, metadata_path: str | Path) -> None:
        self.index = faiss.read_index(str(index_path))
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
