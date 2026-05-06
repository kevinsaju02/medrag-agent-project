from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.data_processing.loaders import load_jsonl_records


def _load_split_ids(splits_path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(splits_path).read_text(encoding="utf-8"))


def _load_processed_records(path: str | Path) -> list[dict]:
    return load_jsonl_records(path)


def train_risk_model(
    processed_data_path: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Train and persist the baseline TF-IDF + Logistic Regression risk model."""

    processed_records = _load_processed_records(processed_data_path)
    split_ids = _load_split_ids(splits_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_ids = set(split_ids["train"])
    train_records = [record for record in processed_records if record["document_id"] in train_ids]

    x_train = [record["normalized_text"] for record in train_records]
    y_train = [record["risk_label"] for record in train_records]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=3000)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    model_path = output_path / "risk_model.joblib"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": "tfidf_logistic_regression",
        "train_size": len(train_records),
        "labels": sorted(set(y_train)),
        "vectorizer": "TfidfVectorizer(ngram_range=(1,2), max_features=3000)",
        "classifier": "LogisticRegression(max_iter=1000)",
    }
    metadata_path = output_path / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": model_path,
        "metadata": metadata_path,
    }
