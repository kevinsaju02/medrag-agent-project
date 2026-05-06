from __future__ import annotations

from pathlib import Path

from app.ml_model.predict import RiskPredictor
from app.schemas.enums import RiskLevel
from app.schemas.output import PredictionOutput


def run_prediction_agent(text: str, model_path: str | Path) -> PredictionOutput:
    predictor = RiskPredictor(model_path)
    result = predictor.predict(text)
    return PredictionOutput(
        risk_level=RiskLevel(result["risk_level"]),
        risk_probability=result["risk_probability"],
        model_name=result["model_name"],
    )
