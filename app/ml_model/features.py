from __future__ import annotations

import re


AGE_PATTERN = re.compile(r"(\d{2})-year-old")


HIGH_ACUITY_TERMS = {
    "severe",
    "crushing",
    "hypotension",
    "confusion",
    "stroke",
    "sepsis",
    "embolism",
    "orthopnea",
}


def extract_tabular_features(text: str) -> dict[str, float]:
    """Return lightweight engineered features for optional model comparisons."""

    lowered = text.lower()
    age_match = AGE_PATTERN.search(lowered)
    age_value = float(age_match.group(1)) if age_match else 0.0

    return {
        "age": age_value,
        "symptom_count_hint": float(
            sum(term in lowered for term in ["pain", "cough", "fever", "fatigue", "weakness", "shortness of breath"])
        ),
        "medication_count_hint": float(sum(term in lowered for term in ["started on", "current treatment includes", "given"])),
        "high_acuity_term_count": float(sum(term in lowered for term in HIGH_ACUITY_TERMS)),
    }
