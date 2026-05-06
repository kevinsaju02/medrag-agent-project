from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint() -> None:
    response = client.post(
        "/analyze",
        json={
            "text": (
                "Patient is a 67-year-old male with history of hypertension and type 2 diabetes "
                "presenting with shortness of breath and chest discomfort. Started on metoprolol. "
                "ECG ordered. Follow-up recommended."
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["document_type"] == "clinical_note"
    assert "metoprolol" in payload["extracted_fields"]["medications"]
