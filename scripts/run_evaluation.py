from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.data_processing.loaders import load_dataset_records
from app.evaluation.extraction_eval import aggregate_extraction_scores, score_extractions
from app.evaluation.ml_eval import load_ml_metrics
from app.evaluation.retrieval_eval import compute_retrieval_metrics
from app.schemas.api import AnalyzeRequest
from app.services.analyze_service import analyze_document


def main() -> None:
    dataset_path = ROOT_DIR / "data" / "synthetic" / "synthetic_clinical_notes.jsonl"
    split_path = ROOT_DIR / "data" / "splits" / "dataset_splits.json"
    ml_metrics_path = ROOT_DIR / "models" / "ml" / "evaluation_metrics.json"
    output_path = ROOT_DIR / "models" / "ml" / "end_to_end_evaluation.json"

    records = load_dataset_records(dataset_path)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = set(split_payload["test"])
    test_records = [record for record in records if record.document_id in test_ids]

    extraction_scores: list[dict[str, dict[str, float]]] = []
    retrieval_scores: list[dict[str, float]] = []

    for record in test_records:
        response = analyze_document(
            AnalyzeRequest(
                document_id=record.document_id,
                text=record.raw_text,
            )
        )
        extraction_scores.append(score_extractions(response.extracted_fields, record.ground_truth))
        retrieval_scores.append(compute_retrieval_metrics(response.citations))

    aggregated_extraction = aggregate_extraction_scores(extraction_scores)
    aggregated_retrieval = {
        "retrieval_hit_rate": sum(item["retrieval_hit_rate"] for item in retrieval_scores) / len(retrieval_scores),
        "citation_coverage": sum(item["citation_coverage"] for item in retrieval_scores) / len(retrieval_scores),
        "unsupported_citation_rate": sum(item["unsupported_citation_rate"] for item in retrieval_scores) / len(retrieval_scores),
    }
    ml_metrics = load_ml_metrics(ml_metrics_path)

    payload = {
        "test_record_count": len(test_records),
        "extraction": aggregated_extraction,
        "retrieval": aggregated_retrieval,
        "ml": {
            "accuracy": ml_metrics["accuracy"],
            "precision_macro": ml_metrics["precision_macro"],
            "recall_macro": ml_metrics["recall_macro"],
            "f1_macro": ml_metrics["f1_macro"],
            "confusion_matrix": ml_metrics["confusion_matrix"],
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
