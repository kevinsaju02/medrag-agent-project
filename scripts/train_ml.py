from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ml_model.evaluate import evaluate_risk_model
from app.ml_model.train import train_risk_model


def main() -> None:
    processed_path = ROOT_DIR / "data" / "processed" / "processed_clinical_notes.jsonl"
    splits_path = ROOT_DIR / "data" / "splits" / "dataset_splits.json"
    output_dir = ROOT_DIR / "models" / "ml"

    artifact_paths = train_risk_model(
        processed_data_path=processed_path,
        splits_path=splits_path,
        output_dir=output_dir,
    )
    evaluation = evaluate_risk_model(
        processed_data_path=processed_path,
        splits_path=splits_path,
        model_path=artifact_paths["model"],
        output_dir=output_dir,
    )

    print(json.dumps(
        {
            "model": str(artifact_paths["model"]),
            "metadata": str(artifact_paths["metadata"]),
            "metrics": str(evaluation["metrics_path"]),
            "accuracy": evaluation["metrics"]["accuracy"],
            "f1_macro": evaluation["metrics"]["f1_macro"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
