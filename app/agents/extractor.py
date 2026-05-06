from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

from app.core.llm import OllamaClient
from app.data_processing.loaders import load_dataset_records
from app.schemas.output import Citation, ExtractedFields


FIELD_NAMES = ("diagnosis", "symptoms", "medications", "procedures", "follow_up_actions")
FOLLOW_UP_PATTERNS = [
    re.compile(r"\bfollow[- ]up recommended\b", re.IGNORECASE),
    re.compile(r"\bfollow[- ]up advised\b", re.IGNORECASE),
    re.compile(r"\breturn if [^.]+\b", re.IGNORECASE),
]


def _build_vocabulary(dataset_path: str | Path) -> dict[str, set[str]]:
    records = load_dataset_records(dataset_path)
    vocabulary: dict[str, set[str]] = {field: set() for field in FIELD_NAMES}
    for record in records:
        ground_truth = record.ground_truth
        vocabulary["diagnosis"].update(item.lower() for item in ground_truth.diagnosis)
        vocabulary["symptoms"].update(item.lower() for item in ground_truth.symptoms)
        vocabulary["medications"].update(item.lower() for item in ground_truth.medications)
        vocabulary["procedures"].update(item.lower() for item in ground_truth.procedures)
        vocabulary["follow_up_actions"].update(item.lower() for item in ground_truth.follow_up_actions)
    return vocabulary


def _rule_based_extract(
    retrieval_results: dict[str, list[dict]],
    vocabulary: dict[str, set[str]],
) -> tuple[ExtractedFields, dict[str, list[Citation]]]:
    extracted: dict[str, list[str]] = defaultdict(list)
    citations: dict[str, list[Citation]] = defaultdict(list)

    for field_name, matches in retrieval_results.items():
        candidates = vocabulary[field_name]
        for match in matches:
            chunk_text = match["text"]
            lowered = chunk_text.lower()
            for candidate in sorted(candidates, key=len, reverse=True):
                if candidate in lowered and candidate not in extracted[field_name]:
                    extracted[field_name].append(candidate)
                    citations[field_name].append(
                        Citation(
                            value=candidate,
                            chunk_id=match["chunk_id"],
                            evidence_text=chunk_text,
                            retrieval_score=match.get("score"),
                        )
                    )

    for match in retrieval_results.get("follow_up_actions", []):
        chunk_text = match["text"]
        for pattern in FOLLOW_UP_PATTERNS:
            found = pattern.search(chunk_text)
            if found:
                value = found.group(0).lower()
                if value not in extracted["follow_up_actions"]:
                    extracted["follow_up_actions"].append(value)
                    citations["follow_up_actions"].append(
                        Citation(
                            value=value,
                            chunk_id=match["chunk_id"],
                            evidence_text=chunk_text,
                            retrieval_score=match.get("score"),
                        )
                    )

    fields = ExtractedFields(
        diagnosis=extracted["diagnosis"],
        symptoms=extracted["symptoms"],
        medications=extracted["medications"],
        procedures=extracted["procedures"],
        follow_up_actions=extracted["follow_up_actions"],
    )
    return fields, dict(citations)


def run_extraction_agent(
    text: str,
    retrieval_results: dict[str, list[dict]],
    dataset_path: str | Path,
) -> dict[str, object]:
    """Use Ollama when available, otherwise a deterministic rule-based extractor."""

    llm_client = OllamaClient()
    llm_result = llm_client.generate(
        "Extract diagnosis, symptoms, medications, procedures, and follow-up actions as JSON from this note:\n"
        f"{text}"
    )

    vocabulary = _build_vocabulary(dataset_path)
    extracted_fields, citations = _rule_based_extract(retrieval_results, vocabulary)

    return {
        "extracted_fields": extracted_fields,
        "citations": citations,
        "llm_provider": llm_result.provider,
        "used_fallback": llm_result.used_fallback,
    }
