from __future__ import annotations

import json
from pathlib import Path

from app.schemas.dataset import DatasetRecord


def load_jsonl_records(path: str | Path) -> list[dict]:
    """Load raw JSONL records from disk."""

    file_path = Path(path)
    records: list[dict] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def load_dataset_records(path: str | Path) -> list[DatasetRecord]:
    """Load and validate dataset records from a JSONL file."""

    return [DatasetRecord.model_validate(item) for item in load_jsonl_records(path)]
