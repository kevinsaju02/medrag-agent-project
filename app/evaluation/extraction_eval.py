from __future__ import annotations

from collections import defaultdict

from app.schemas.output import ExtractedFields


FIELD_NAMES = ("diagnosis", "symptoms", "medications", "procedures", "follow_up_actions")


def _normalize_values(values: list[str]) -> set[str]:
    return {value.strip().lower() for value in values if value.strip()}


def score_extractions(predicted: ExtractedFields, expected: ExtractedFields) -> dict[str, dict[str, float]]:
    """Compute simple field-level precision/recall/F1 metrics."""

    results: dict[str, dict[str, float]] = {}
    for field_name in FIELD_NAMES:
        pred_values = _normalize_values(getattr(predicted, field_name))
        exp_values = _normalize_values(getattr(expected, field_name))

        tp = len(pred_values & exp_values)
        fp = len(pred_values - exp_values)
        fn = len(exp_values - pred_values)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        results[field_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": 1.0 if pred_values == exp_values else 0.0,
            "missing_field": 1.0 if not pred_values and exp_values else 0.0,
            "hallucination_rate": fp / len(pred_values) if pred_values else 0.0,
        }
    return results


def aggregate_extraction_scores(scores: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, float]] = defaultdict(dict)
    if not scores:
        return {field: {} for field in FIELD_NAMES}

    for field_name in FIELD_NAMES:
        metrics = ("precision", "recall", "f1", "exact_match", "missing_field", "hallucination_rate")
        aggregated[field_name] = {
            metric: sum(item[field_name][metric] for item in scores) / len(scores) for metric in metrics
        }
    return dict(aggregated)
