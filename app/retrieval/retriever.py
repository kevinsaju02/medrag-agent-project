from __future__ import annotations

from pathlib import Path

from app.data_processing.loaders import load_jsonl_records
from app.retrieval.embeddings import EmbeddingBackend
from app.retrieval.query_templates import FIELD_QUERY_TEMPLATES
from app.retrieval.vector_store import FaissVectorStore


def _flatten_processed_chunks(processed_records: list[dict]) -> list[dict]:
    chunk_records: list[dict] = []
    for record in processed_records:
        for chunk in record["chunks"]:
            chunk_records.append(
                {
                    "document_id": record["document_id"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                }
            )
    return chunk_records


def build_retrieval_index(
    processed_data_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    processed_records = load_jsonl_records(processed_data_path)
    chunk_records = _flatten_processed_chunks(processed_records)

    embedding_backend = EmbeddingBackend()
    vector_store = FaissVectorStore(embedding_backend)
    vector_store.build(chunk_records)
    saved_paths = vector_store.save(output_dir)

    return {
        "vector_store": vector_store,
        "embedding_model_name": embedding_backend.model_name,
        "saved_paths": saved_paths,
        "chunk_count": len(chunk_records),
    }


def retrieve_field_evidence(
    processed_data_path: str | Path,
    index_dir: str | Path,
    document_id: str,
    top_k: int = 3,
) -> dict[str, list[dict]]:
    processed_records = load_jsonl_records(processed_data_path)
    target_record = next(record for record in processed_records if record["document_id"] == document_id)
    embedding_backend = EmbeddingBackend()
    vector_store = FaissVectorStore(embedding_backend)
    document_chunk_records = [
        {
            "document_id": document_id,
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        }
        for chunk in target_record["chunks"]
    ]
    vector_store.build(document_chunk_records)

    results: dict[str, list[dict]] = {}
    for field_name, query in FIELD_QUERY_TEMPLATES.items():
        results[field_name] = vector_store.search(query=query, top_k=top_k)

    return results
