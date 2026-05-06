from __future__ import annotations

from pathlib import Path

import joblib


class RiskPredictor:
    """Thin inference wrapper around the persisted sklearn pipeline."""

    def __init__(self, model_path: str | Path) -> None:
        self.model = joblib.load(model_path)
        self._patch_sklearn_compatibility()

    def _patch_sklearn_compatibility(self) -> None:
        """Backfill attributes expected by different sklearn versions."""

        classifier = None

        if hasattr(self.model, "named_steps"):
            classifier = self.model.named_steps.get("classifier")
        elif hasattr(self.model, "steps") and self.model.steps:
            classifier = self.model.steps[-1][1]

        if classifier is not None and not hasattr(classifier, "multi_class"):
            classifier.multi_class = "auto"

    def predict(self, text: str) -> dict[str, object]:
        probabilities = self.model.predict_proba([text])[0]
        labels = list(self.model.classes_)
        top_index = int(probabilities.argmax())
        return {
            "risk_level": str(labels[top_index]),
            "risk_probability": float(probabilities[top_index]),
            "model_name": "tfidf_logistic_regression",
        }
