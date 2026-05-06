from __future__ import annotations


def compute_retrieval_metrics(retrieval_results: dict[str, list[dict]]) -> dict[str, float]:
    """Simple retrieval evaluation stub for Day 3 integration."""

    total_fields = len(retrieval_results)
    covered_fields = sum(1 for matches in retrieval_results.values() if matches)

    return {
        "retrieval_hit_rate": covered_fields / total_fields if total_fields else 0.0,
        "citation_coverage": covered_fields / total_fields if total_fields else 0.0,
        "unsupported_citation_rate": 0.0,
    }
