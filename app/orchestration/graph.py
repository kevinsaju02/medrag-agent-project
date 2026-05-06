from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from app.agents.classifier import classify_document
from app.agents.extractor import run_extraction_agent
from app.agents.predictor import run_prediction_agent
from app.agents.retrieval_agent import run_retrieval_agent
from app.agents.segmenter import segment_document
from app.agents.validator import run_validation_agent
from app.data_processing.normalizers import normalize_clinical_text
from app.orchestration.state import MedRAGState


def _append_trace(state: MedRAGState, step: str, payload: dict[str, object]) -> list[dict]:
    trace = list(state.get("debug_trace", []))
    trace.append({"step": step, "payload": payload})
    return trace


def build_medrag_graph(
    synthetic_dataset_path: str | Path,
    ml_model_path: str | Path,
):
    synthetic_dataset_path = Path(synthetic_dataset_path)
    ml_model_path = Path(ml_model_path)

    def ingest_node(state: MedRAGState) -> MedRAGState:
        normalized_text = normalize_clinical_text(state["raw_text"])
        return {
            "normalized_text": normalized_text,
            "retry_count": state.get("retry_count", 0),
            "debug_trace": _append_trace(state, "ingest", {"normalized_text": normalized_text}),
        }

    def classify_node(state: MedRAGState) -> MedRAGState:
        result = classify_document(state["normalized_text"])
        return {
            "document_type": result["document_type"].value,
            "classifier_confidence": result["confidence"],
            "classifier_rationale": result["rationale"],
            "debug_trace": _append_trace(state, "classify", result),
        }

    def segment_node(state: MedRAGState) -> MedRAGState:
        sections = segment_document(state["normalized_text"])
        return {
            "sections": sections,
            "debug_trace": _append_trace(state, "segment", {"section_count": len(sections)}),
        }

    def retrieve_node(state: MedRAGState) -> MedRAGState:
        retrieval_results = run_retrieval_agent(
            document_id=state["document_id"],
            text=state["normalized_text"],
            top_k=3,
        )
        return {
            "retrieval_results": retrieval_results,
            "debug_trace": _append_trace(
                state,
                "retrieve",
                {field: len(matches) for field, matches in retrieval_results.items()},
            ),
        }

    def extract_node(state: MedRAGState) -> MedRAGState:
        result = run_extraction_agent(
            text=state["normalized_text"],
            retrieval_results=state["retrieval_results"],
            dataset_path=synthetic_dataset_path,
        )
        return {
            "extracted_fields": result["extracted_fields"],
            "citations": result["citations"],
            "metadata": {
                **state.get("metadata", {}),
                "extractor_provider": result["llm_provider"],
                "extractor_fallback": result["used_fallback"],
            },
            "debug_trace": _append_trace(
                state,
                "extract",
                {
                    "diagnosis_count": len(result["extracted_fields"].diagnosis),
                    "symptom_count": len(result["extracted_fields"].symptoms),
                },
            ),
        }

    def predict_node(state: MedRAGState) -> MedRAGState:
        prediction = run_prediction_agent(state["normalized_text"], ml_model_path)
        return {
            "prediction": prediction,
            "debug_trace": _append_trace(
                state,
                "predict",
                prediction.model_dump(mode="json"),
            ),
        }

    def validate_node(state: MedRAGState) -> MedRAGState:
        result = run_validation_agent(
            text=state["normalized_text"],
            extracted_fields=state["extracted_fields"],
            citations=state["citations"],
            prediction=state["prediction"],
        )
        validation = result["validation"]
        retry_count = state.get("retry_count", 0)
        should_retry = bool(validation.unsupported_fields and retry_count < 1)
        return {
            "validation": validation,
            "retry_count": retry_count + 1 if should_retry else retry_count,
            "next_step": "retry_extract" if should_retry else "complete",
            "metadata": {
                **state.get("metadata", {}),
                "validator_provider": result["llm_provider"],
                "validator_fallback": result["used_fallback"],
            },
            "debug_trace": _append_trace(
                state,
                "validate",
                {
                    "unsupported_fields": validation.unsupported_fields,
                    "hallucination_flags": validation.hallucination_flags,
                    "next_step": "retry_extract" if should_retry else "complete",
                },
            ),
        }

    def route_after_validation(state: MedRAGState) -> str:
        return state.get("next_step", "complete")

    graph = StateGraph(MedRAGState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("classify", classify_node)
    graph.add_node("segment", segment_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("extract", extract_node)
    graph.add_node("predict", predict_node)
    graph.add_node("validate", validate_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "classify")
    graph.add_edge("classify", "segment")
    graph.add_edge("segment", "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "predict")
    graph.add_edge("predict", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "retry_extract": "extract",
            "complete": END,
        },
    )
    return graph.compile()
