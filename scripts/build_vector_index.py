from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.retrieval.retriever import build_retrieval_index


def main() -> None:
    processed_path = ROOT_DIR / "data" / "processed" / "processed_clinical_notes.jsonl"
    output_dir = ROOT_DIR / "models" / "vector_index"

    result = build_retrieval_index(processed_data_path=processed_path, output_dir=output_dir)
    payload = {
        "embedding_model_name": result["embedding_model_name"],
        "chunk_count": result["chunk_count"],
        "saved_paths": {name: str(path) for name, path in result["saved_paths"].items()},
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
