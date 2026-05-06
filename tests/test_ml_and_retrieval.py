from __future__ import annotations

from pathlib import Path

from app.ml_model.evaluate import evaluate_risk_model
from app.ml_model.predict import RiskPredictor
from app.ml_model.train import train_risk_model
from app.retrieval.retriever import build_retrieval_index, retrieve_field_evidence


def test_train_evaluate_and_predict_pipeline() -> None:
    base_dir = Path.cwd()
    processed_path = base_dir / "data" / "processed" / "processed_clinical_notes.jsonl"
    splits_path = base_dir / "data" / "splits" / "dataset_splits.json"
    model_dir = base_dir / "models" / "ml"

    artifacts = train_risk_model(processed_path, splits_path, model_dir)
    evaluation = evaluate_risk_model(processed_path, splits_path, artifacts["model"], model_dir)
    predictor = RiskPredictor(artifacts["model"])
    prediction = predictor.predict("Patient with severe shortness of breath and orthopnea.")

    assert artifacts["model"].exists()
    assert evaluation["metrics"]["accuracy"] >= 0.0
    assert prediction["risk_level"] in {"low", "medium", "high"}


def test_build_and_query_retrieval_index() -> None:
    base_dir = Path.cwd()
    processed_path = base_dir / "data" / "processed" / "processed_clinical_notes.jsonl"
    index_dir = base_dir / "models" / "vector_index"

    result = build_retrieval_index(processed_path, index_dir)
    evidence = retrieve_field_evidence(processed_path, index_dir, document_id="doc_001", top_k=2)

    assert result["chunk_count"] > 0
    assert (index_dir / "faiss.index").exists()
    assert "diagnosis" in evidence
    assert all(len(matches) <= 2 for matches in evidence.values())
