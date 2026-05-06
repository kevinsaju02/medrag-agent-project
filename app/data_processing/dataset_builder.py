from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from app.data_processing.chunking import chunk_text
from app.data_processing.normalizers import normalize_clinical_text
from app.schemas.dataset import DatasetRecord
from app.schemas.enums import DocumentType, RiskLevel
from app.schemas.output import ExtractedFields


@dataclass(frozen=True)
class Scenario:
    risk_label: RiskLevel
    diagnoses: tuple[str, ...]
    symptoms: tuple[str, ...]
    medications: tuple[str, ...]
    procedures: tuple[str, ...]
    follow_up_actions: tuple[str, ...]


LOW_RISK_SCENARIOS = [
    Scenario(
        risk_label=RiskLevel.LOW,
        diagnoses=("seasonal allergic rhinitis",),
        symptoms=("nasal congestion", "sneezing"),
        medications=("loratadine",),
        procedures=("allergy review",),
        follow_up_actions=("follow up with primary care if symptoms persist",),
    ),
    Scenario(
        risk_label=RiskLevel.LOW,
        diagnoses=("viral upper respiratory infection",),
        symptoms=("sore throat", "mild cough"),
        medications=("acetaminophen",),
        procedures=("rapid strep test",),
        follow_up_actions=("supportive care at home",),
    ),
    Scenario(
        risk_label=RiskLevel.LOW,
        diagnoses=("tension headache",),
        symptoms=("headache", "neck tightness"),
        medications=("ibuprofen",),
        procedures=("neurologic exam",),
        follow_up_actions=("hydration and rest",),
    ),
    Scenario(
        risk_label=RiskLevel.LOW,
        diagnoses=("gastroesophageal reflux disease",),
        symptoms=("heartburn", "epigastric discomfort"),
        medications=("omeprazole",),
        procedures=("dietary counseling",),
        follow_up_actions=("avoid late meals",),
    ),
    Scenario(
        risk_label=RiskLevel.LOW,
        diagnoses=("mild dehydration",),
        symptoms=("lightheadedness", "dry mouth"),
        medications=("oral rehydration solution",),
        procedures=("vital signs check",),
        follow_up_actions=("increase fluid intake",),
    ),
]


MEDIUM_RISK_SCENARIOS = [
    Scenario(
        risk_label=RiskLevel.MEDIUM,
        diagnoses=("hypertension", "type 2 diabetes"),
        symptoms=("shortness of breath", "chest discomfort"),
        medications=("metoprolol",),
        procedures=("ECG",),
        follow_up_actions=("cardiology follow up recommended",),
    ),
    Scenario(
        risk_label=RiskLevel.MEDIUM,
        diagnoses=("chronic obstructive pulmonary disease",),
        symptoms=("wheezing", "increased cough"),
        medications=("albuterol", "prednisone"),
        procedures=("chest x-ray",),
        follow_up_actions=("return if breathing worsens",),
    ),
    Scenario(
        risk_label=RiskLevel.MEDIUM,
        diagnoses=("urinary tract infection",),
        symptoms=("dysuria", "urinary frequency", "fever"),
        medications=("nitrofurantoin",),
        procedures=("urinalysis",),
        follow_up_actions=("follow up urine culture",),
    ),
    Scenario(
        risk_label=RiskLevel.MEDIUM,
        diagnoses=("atrial fibrillation",),
        symptoms=("palpitations", "fatigue"),
        medications=("diltiazem",),
        procedures=("telemetry monitoring",),
        follow_up_actions=("outpatient cardiology review",),
    ),
    Scenario(
        risk_label=RiskLevel.MEDIUM,
        diagnoses=("community acquired pneumonia",),
        symptoms=("fever", "productive cough", "shortness of breath"),
        medications=("azithromycin",),
        procedures=("chest x-ray",),
        follow_up_actions=("follow up in 48 hours",),
    ),
]


HIGH_RISK_SCENARIOS = [
    Scenario(
        risk_label=RiskLevel.HIGH,
        diagnoses=("congestive heart failure", "coronary artery disease"),
        symptoms=("severe shortness of breath", "orthopnea", "leg swelling"),
        medications=("furosemide", "aspirin"),
        procedures=("BNP test", "echocardiogram"),
        follow_up_actions=("urgent cardiology evaluation",),
    ),
    Scenario(
        risk_label=RiskLevel.HIGH,
        diagnoses=("sepsis",),
        symptoms=("high fever", "hypotension", "confusion"),
        medications=("broad spectrum antibiotics", "intravenous fluids"),
        procedures=("blood cultures", "lactate test"),
        follow_up_actions=("hospital admission",),
    ),
    Scenario(
        risk_label=RiskLevel.HIGH,
        diagnoses=("acute coronary syndrome",),
        symptoms=("crushing chest pain", "diaphoresis", "nausea"),
        medications=("nitroglycerin", "heparin"),
        procedures=("serial troponin test", "ECG"),
        follow_up_actions=("emergency cardiology consultation",),
    ),
    Scenario(
        risk_label=RiskLevel.HIGH,
        diagnoses=("stroke",),
        symptoms=("slurred speech", "right arm weakness", "facial droop"),
        medications=("aspirin",),
        procedures=("CT head",),
        follow_up_actions=("stroke team activation",),
    ),
    Scenario(
        risk_label=RiskLevel.HIGH,
        diagnoses=("pulmonary embolism",),
        symptoms=("pleuritic chest pain", "tachycardia", "shortness of breath"),
        medications=("enoxaparin",),
        procedures=("CT pulmonary angiography",),
        follow_up_actions=("immediate inpatient management",),
    ),
]


AGE_OPTIONS = [24, 31, 38, 45, 52, 59, 67, 74, 81]
SEX_OPTIONS = ["female", "male"]
STYLE_NAMES = (
    "narrative_note",
    "sectioned_note",
    "ed_brief",
    "follow_up_note",
    "assessment_plan",
)


def _join_items(items: tuple[str, ...]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _render_note(style_name: str, age: int, sex: str, scenario: Scenario) -> str:
    diagnoses = _join_items(scenario.diagnoses)
    symptoms = _join_items(scenario.symptoms)
    medications = _join_items(scenario.medications)
    procedures = _join_items(scenario.procedures)
    follow_up = _join_items(scenario.follow_up_actions)

    if style_name == "narrative_note":
        return (
            f"Patient is a {age}-year-old {sex} with history of {diagnoses} presenting with "
            f"{symptoms}. Started on {medications}. {procedures} ordered. {follow_up}."
        )

    if style_name == "sectioned_note":
        return (
            f"History: {age}-year-old {sex} with {diagnoses}. "
            f"Symptoms: {symptoms}. "
            f"Medications: {medications}. "
            f"Procedures: {procedures}. "
            f"Plan: {follow_up}."
        )

    if style_name == "ed_brief":
        return (
            f"ED note for {age} year old {sex}. Complaints include {symptoms}. "
            f"Assessment suggests {diagnoses}. "
            f"Given or started {medications}. "
            f"Workup includes {procedures}. "
            f"Disposition: {follow_up}."
        )

    if style_name == "follow_up_note":
        return (
            f"Follow-up visit: patient is a {age}-year-old {sex}. "
            f"Known problems include {diagnoses}. "
            f"Reports {symptoms}. "
            f"Current treatment includes {medications}. "
            f"Testing discussed: {procedures}. "
            f"Next step: {follow_up}."
        )

    return (
        f"Assessment: {diagnoses}. "
        f"Subjective findings include {symptoms}. "
        f"Plan is to continue or start {medications}, obtain {procedures}, and arrange {follow_up}. "
        f"Patient is a {age}-year-old {sex}."
    )


def _build_record(index: int, scenario: Scenario, style_name: str, rng: random.Random) -> DatasetRecord:
    age = rng.choice(AGE_OPTIONS)
    sex = rng.choice(SEX_OPTIONS)
    raw_text = _render_note(style_name=style_name, age=age, sex=sex, scenario=scenario)

    return DatasetRecord(
        document_id=f"doc_{index:03d}",
        raw_text=raw_text,
        document_type=DocumentType.CLINICAL_NOTE,
        ground_truth=ExtractedFields(
            diagnosis=list(scenario.diagnoses),
            symptoms=list(scenario.symptoms),
            medications=list(scenario.medications),
            procedures=list(scenario.procedures),
            follow_up_actions=list(scenario.follow_up_actions),
        ),
        risk_label=scenario.risk_label,
    )


def generate_synthetic_dataset(record_count: int = 200, seed: int = 42) -> list[DatasetRecord]:
    """Generate a diverse synthetic dataset for MedRAG training and evaluation."""

    rng = random.Random(seed)
    scenarios = LOW_RISK_SCENARIOS + MEDIUM_RISK_SCENARIOS + HIGH_RISK_SCENARIOS

    records: list[DatasetRecord] = []
    for index in range(1, record_count + 1):
        scenario = scenarios[(index - 1) % len(scenarios)]
        style_name = STYLE_NAMES[(index - 1) % len(STYLE_NAMES)]
        records.append(_build_record(index=index, scenario=scenario, style_name=style_name, rng=rng))

    rng.shuffle(records)
    for new_index, record in enumerate(records, start=1):
        record.document_id = f"doc_{new_index:03d}"  # type: ignore[misc]
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _build_split_ids(records: list[DatasetRecord]) -> dict[str, list[str]]:
    total = len(records)
    train_end = int(total * 0.7)
    validation_end = train_end + int(total * 0.15)

    return {
        "train": [record.document_id for record in records[:train_end]],
        "validation": [record.document_id for record in records[train_end:validation_end]],
        "test": [record.document_id for record in records[validation_end:]],
    }


def save_dataset_artifacts(
    output_dir: str | Path,
    record_count: int = 200,
    seed: int = 42,
    target_chunk_words: int = 120,
) -> dict[str, Path]:
    """Generate and save synthetic dataset plus processed retrieval artifacts."""

    base_dir = Path(output_dir)
    synthetic_dir = base_dir / "synthetic"
    processed_dir = base_dir / "processed"
    splits_dir = base_dir / "splits"

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    records = generate_synthetic_dataset(record_count=record_count, seed=seed)
    raw_records = [record.model_dump(mode="json") for record in records]

    processed_records: list[dict] = []
    for record in records:
        normalized_text = normalize_clinical_text(record.raw_text)
        chunks = chunk_text(
            text=normalized_text,
            document_id=record.document_id,
            target_words=target_chunk_words,
        )
        processed_records.append(
            {
                "document_id": record.document_id,
                "document_type": record.document_type.value,
                "normalized_text": normalized_text,
                "risk_label": record.risk_label.value,
                "ground_truth": record.ground_truth.model_dump(mode="json"),
                "chunks": chunks,
            }
        )

    split_payload = _build_split_ids(records)

    synthetic_path = synthetic_dir / "synthetic_clinical_notes.jsonl"
    processed_path = processed_dir / "processed_clinical_notes.jsonl"
    splits_path = splits_dir / "dataset_splits.json"

    _write_jsonl(synthetic_path, raw_records)
    _write_jsonl(processed_path, processed_records)
    splits_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    return {
        "synthetic": synthetic_path,
        "processed": processed_path,
        "splits": splits_path,
    }
