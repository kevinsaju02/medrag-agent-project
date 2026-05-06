from __future__ import annotations

from app.core.llm import OllamaClient
from app.schemas.enums import ConfidenceLevel
from app.schemas.output import ExtractedFields, PredictionOutput, ValidationOutput


def _flatten_extracted_fields(extracted_fields: ExtractedFields) -> dict[str, list[str]]:
    return {
        "diagnosis": extracted_fields.diagnosis,
        "symptoms": extracted_fields.symptoms,
        "medications": extracted_fields.medications,
        "procedures": extracted_fields.procedures,
        "follow_up_actions": extracted_fields.follow_up_actions,
    }


def run_validation_agent(
    text: str,
    extracted_fields: ExtractedFields,
    citations: dict[str, list[object]],
    prediction: PredictionOutput,
) -> dict[str, object]:
    llm_client = OllamaClient()
    llm_result = llm_client.generate(
        "Check whether the extracted clinical values are supported by the note. "
        f"Prediction: {prediction.model_dump(mode='json')}. Note: {text}"
    )

    unsupported_fields: list[str] = []
    hallucination_flags: list[str] = []
    missing_expected_fields: list[str] = []
    contradiction_flags: list[str] = []

    flattened = _flatten_extracted_fields(extracted_fields)
    lowered_note = text.lower()

    for field_name, values in flattened.items():
        if values and not citations.get(field_name):
            unsupported_fields.append(field_name)

        for value in values:
            if value.lower() not in lowered_note:
                hallucination_flags.append(f"{field_name}:{value}")

        if not values:
            missing_expected_fields.append(field_name)

    if prediction.risk_level.value == "high" and "severe" not in lowered_note and "acute" not in lowered_note:
        contradiction_flags.append("high_risk_prediction_without_explicit_severity_signal")

    total_flags = len(unsupported_fields) + len(hallucination_flags) + len(contradiction_flags)
    if total_flags == 0:
        confidence = ConfidenceLevel.HIGH
    elif total_flags <= 2:
        confidence = ConfidenceLevel.MEDIUM
    else:
        confidence = ConfidenceLevel.LOW

    return {
        "validation": ValidationOutput(
            unsupported_fields=unsupported_fields,
            missing_expected_fields=missing_expected_fields,
            hallucination_flags=hallucination_flags,
            contradiction_flags=contradiction_flags,
            overall_confidence=confidence,
        ),
        "llm_provider": llm_result.provider,
        "used_fallback": llm_result.used_fallback,
    }
