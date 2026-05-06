from __future__ import annotations

from app.data_processing.chunking import chunk_text
from app.retrieval.embeddings import EmbeddingBackend
from app.retrieval.query_templates import FIELD_QUERY_TEMPLATES
from app.retrieval.vector_store import FaissVectorStore


def run_retrieval_agent(document_id: str, text: str, top_k: int = 3) -> dict[str, list[dict]]:
    chunk_records = [
        {
            "document_id": document_id,
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        }
        for chunk in chunk_text(text=text, document_id=document_id, target_words=120, overlap_sentences=1)
    ]

    vector_store = FaissVectorStore(EmbeddingBackend())
    vector_store.build(chunk_records)

    return {
        field_name: vector_store.search(query=query, top_k=top_k)
        for field_name, query in FIELD_QUERY_TEMPLATES.items()
    }
