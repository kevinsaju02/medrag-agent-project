from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from app.data_processing.loaders import load_jsonl_records


def evaluate_risk_model(
    processed_data_path: str | Path,
    splits_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Evaluate the baseline risk model on the held-out test split."""

    processed_records = load_jsonl_records(processed_data_path)
    split_ids = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    pipeline = joblib.load(model_path)

    test_ids = set(split_ids["test"])
    test_records = [record for record in processed_records if record["document_id"] in test_ids]

    x_test = [record["normalized_text"] for record in test_records]
    y_true = [record["risk_label"] for record in test_records]
    y_pred = pipeline.predict(x_test).tolist()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "labels": sorted(set(y_true)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=sorted(set(y_true))).tolist(),
        "test_size": len(test_records),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
    }
