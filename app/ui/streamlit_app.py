from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.schemas.api import AnalyzeRequest
from app.services.analyze_service import analyze_document


st.set_page_config(page_title="MedRAG Agent", page_icon=":hospital:", layout="wide")


def _read_uploaded_text() -> str:
    uploaded_file = st.file_uploader("Optional .txt upload", type=["txt"])
    if uploaded_file is None:
        return ""
    return uploaded_file.read().decode("utf-8")


st.title("MedRAG Agent")
st.caption("Multi-agent clinical note extraction, citation grounding, risk prediction, and validation.")

with st.sidebar:
    st.subheader("How To Use")
    st.write("Paste clinical-style text or upload a `.txt` note, then click Analyze.")
    st.write("The app returns structured JSON, evidence citations, risk prediction, and validation flags.")

uploaded_text = _read_uploaded_text()
default_text = (
    "Patient is a 67-year-old male with history of hypertension and type 2 diabetes "
    "presenting with shortness of breath and chest discomfort. Started on metoprolol. "
    "ECG ordered. Follow-up recommended."
)
text_input = st.text_area(
    "Clinical Note Input",
    value=uploaded_text or default_text,
    height=220,
    placeholder="Paste an unseen clinical-style note here...",
)
document_id = st.text_input("Optional Document ID", value="")

if st.button("Analyze", type="primary"):
    with st.spinner("Running MedRAG analysis pipeline..."):
        response = analyze_document(
            AnalyzeRequest(
                document_id=document_id or None,
                text=text_input,
            )
        )

    json_payload = response.model_dump(mode="json")
    st.success("Analysis complete.")

    tab_json, tab_citations, tab_prediction, tab_validation = st.tabs(
        ["Structured JSON", "Citations", "Risk Prediction", "Validation"]
    )

    with tab_json:
        st.json(json_payload)
        st.download_button(
            label="Download JSON",
            data=json.dumps(json_payload, indent=2),
            file_name=f"{response.document_id}_analysis.json",
            mime="application/json",
        )

    with tab_citations:
        for field_name, citations in response.citations.items():
            st.markdown(f"**{field_name}**")
            if not citations:
                st.write("No citations found.")
                continue
            for citation in citations:
                st.write(
                    {
                        "value": citation.value,
                        "chunk_id": citation.chunk_id,
                        "retrieval_score": citation.retrieval_score,
                        "evidence_text": citation.evidence_text,
                    }
                )

    with tab_prediction:
        st.metric("Risk Level", response.prediction.risk_level.value.title())
        st.metric("Risk Probability", f"{response.prediction.risk_probability:.2f}")
        st.write({"model_name": response.prediction.model_name})

    with tab_validation:
        st.write(response.validation.model_dump(mode="json"))
        st.write({"metadata": response.metadata.model_dump(mode="json")})
