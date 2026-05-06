from __future__ import annotations

import json
from pathlib import Path

from app.data_processing.chunking import chunk_text, split_into_sentences
from app.data_processing.dataset_builder import generate_synthetic_dataset, save_dataset_artifacts
from app.data_processing.loaders import load_dataset_records, load_jsonl_records
from app.data_processing.normalizers import normalize_clinical_text


def test_normalize_clinical_text_collapses_whitespace() -> None:
    raw_text = "Patient\tis here.  Follow-up recommended.\n\nStarted on aspirin."
    assert normalize_clinical_text(raw_text) == "Patient is here. Follow-up recommended. Started on aspirin."


def test_split_into_sentences_returns_units() -> None:
    text = "First sentence. Second sentence? Third sentence!"
    assert split_into_sentences(text) == [
        "First sentence.",
        "Second sentence?",
        "Third sentence!",
    ]


def test_chunk_text_creates_chunk_ids() -> None:
    text = " ".join(f"Sentence {index}." for index in range(1, 9))
    chunks = chunk_text(text=text, document_id="doc_001", target_words=6, overlap_sentences=1)

    assert len(chunks) >= 2
    assert chunks[0]["chunk_id"] == "doc_001_chunk_01"
    assert all("text" in chunk for chunk in chunks)


def test_generate_synthetic_dataset_returns_expected_size() -> None:
    records = generate_synthetic_dataset(record_count=25, seed=7)
    assert len(records) == 25
    assert all(record.document_id.startswith("doc_") for record in records)


def test_save_dataset_artifacts_and_load_records(tmp_path: Path) -> None:
    artifact_paths = save_dataset_artifacts(output_dir=tmp_path, record_count=12, seed=11)

    dataset_records = load_dataset_records(artifact_paths["synthetic"])
    raw_processed_records = load_jsonl_records(artifact_paths["processed"])
    split_payload = json.loads(artifact_paths["splits"].read_text(encoding="utf-8"))

    assert len(dataset_records) == 12
    assert len(raw_processed_records) == 12
    assert set(split_payload) == {"train", "validation", "test"}
    assert raw_processed_records[0]["chunks"]
