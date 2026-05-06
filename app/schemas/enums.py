from enum import Enum


class DocumentType(str, Enum):
    CLINICAL_NOTE = "clinical_note"
    OTHER_MEDICAL_TEXT = "other_medical_text"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
