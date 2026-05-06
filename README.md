# MedRAG Agent

## Overview
MedRAG Agent is a Python project for analyzing unseen clinical-style text and returning structured JSON with:
- extracted medical fields
- citation-backed evidence
- traditional ML risk prediction
- validation and critic output

The system is designed as a production-style demo that showcases:
- multi-agent orchestration
- retrieval-augmented generation with citations
- schema-enforced extraction
- traditional machine learning
- FastAPI backend design
- Streamlit-based Python UI

## Demo Impact
MedRAG Agent is designed to show how unstructured clinical notes can be turned into something operationally useful: structured fields, evidence-backed citations, risk scoring, and explicit validation flags. Instead of returning an opaque chatbot answer, the system makes the reasoning chain more auditable, which is the kind of pattern that matters in high-trust workflows like healthcare operations and clinical documentation review.

## Core Features
- Ad hoc inference on unseen clinical-style text
- Structured JSON output with Pydantic validation
- Retrieval layer that cites evidence from document chunks
- Risk classification using a scikit-learn baseline model
- Validation/critic stage to flag unsupported outputs
- Downloadable JSON result

## ML Strategy
- Primary baseline: TF-IDF + Logistic Regression
- Optional comparison model: XGBoost on engineered clinical features if time allows
- Reasoning: Logistic Regression stays the main model because it is simpler, faster to explain, and well suited to a first text classification baseline on this dataset

## Retrieval Strategy
- Vector store: FAISS
- Current embedding backend: local deterministic hashing-based embeddings behind a swappable interface
- Planned upgrade path: replace the fallback backend with `sentence-transformers` when the local environment supports full installation
- Retrieval scope for v1: document-grounded chunk retrieval for citation-backed extraction

## Agent and Orchestration Strategy
- Orchestration framework: LangGraph
- Agent flow: classify -> segment -> retrieve -> extract -> predict -> validate
- LLM integration: Ollama client interface with graceful fallback behavior when no local model is available
- Debugging support: full orchestration state and per-node debug trace via `run_analysis_pipeline()`

## Tech Stack
- Python 3.10+
- FastAPI
- Streamlit
- Pydantic
- LangGraph
- scikit-learn
- sentence-transformers
- FAISS or Chroma
- pytest

## Project Structure
```text
medrag-agent/
  app/
    agents/
    api/
    core/
    orchestration/
    retrieval/
    ml_model/
    schemas/
    evaluation/
    data_processing/
    services/
    ui/
  data/
    raw/
    synthetic/
    processed/
    ground_truth/
    splits/
  models/
    ml/
    embeddings/
    vector_index/
  scripts/
  tests/
  README.md
  requirements.txt
  .env.example
```

## Artifacts
- ML model: `models/ml/risk_model.joblib`
- ML metrics: `models/ml/evaluation_metrics.json`
- End-to-end evaluation: `models/ml/end_to_end_evaluation.json`
- Vector index: `models/vector_index/faiss.index`
- Vector metadata: `models/vector_index/index_metadata.json`

## How To Use
### Recruiter Quick Start
If you just want to see the project work:

1. Clone the repo.
2. Open a terminal in the project folder.
3. Install the dependencies.
4. Run the Streamlit app.
5. Paste a clinical note and click `Analyze`.

Exact commands:

```bash
git clone <repo-url>
cd medrag-agent
python -m pip install -r requirements.txt
streamlit run app/ui/streamlit_app.py
```

If `streamlit` is not recognized, use:

```bash
python -m streamlit run app/ui/streamlit_app.py
```

### Local Run
Run the UI locally:

```bash
streamlit run app/ui/streamlit_app.py
```

The UI supports:
- pasted clinical note text
- optional `.txt` upload
- structured JSON display
- evidence/citation display
- risk prediction display
- validation display
- JSON download

For the best demo flow:
1. Launch the Streamlit app.
2. Paste a clinical-note-style input or use the default sample.
3. Click `Analyze`.
4. Review the structured JSON, citations, risk prediction, and validation tabs.
5. Download the result as JSON if needed.

### Sample Input
Use this if you want a guaranteed in-scope example:

```text
Patient is a 67-year-old male with history of hypertension and type 2 diabetes presenting with shortness of breath and chest discomfort. Started on metoprolol. ECG ordered. Follow-up recommended.
```

### What You Should See
After clicking `Analyze`, the app should show:
- a structured JSON result
- extracted diagnoses, symptoms, medications, procedures, and follow-up actions
- citation-backed evidence from the note
- a risk level prediction with probability
- validation fields such as unsupported fields, missing fields, and overall confidence

### If Something Goes Wrong
Try these in order:

1. Make sure you are running the command from the project root folder.
2. Reinstall dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the test suite:

```bash
python -m pytest tests -q
```

4. If `streamlit` still fails, use:

```bash
python -m streamlit run app/ui/streamlit_app.py
```

## Best Input Style
MedRAG Agent works best on clinical-note-style text that is reasonably readable and sentence-based. The strongest results come from notes that explicitly mention diagnoses, symptoms, medications, procedures, and follow-up plans in standard clinical language.

Good examples:
- short outpatient or ED-style clinical notes
- sectioned notes with headings like `History`, `Assessment`, `Medications`, or `Plan`
- narrative notes with full medical terms such as `shortness of breath`, `hypertension`, or `echocardiogram`

Harder examples:
- abbreviation-heavy shorthand such as `Pt`, `w hx of`, `CHF`, `CAD`, `SOB`
- typo-heavy notes such as `ordred` or `reccomended`
- very sparse follow-up notes with little explicit clinical detail

In practical terms, the current version is better at:
- `Patient is a 67-year-old male with hypertension...`

than at:
- `Pt is 58 y/o male w hx of CHF + CAD, here for severe SOB...`

That shorthand/noisy style is still useful for testing, but readers should treat it as a current limitation rather than the main supported input format.

## Evaluation Summary
End-to-end evaluation on the 30-record test split produced:
- Diagnosis F1: `1.0`
- Symptoms F1: `0.9857`
- Medications F1: `1.0`
- Procedures F1: `1.0`
- Follow-up actions F1: `0.9667`
- Retrieval hit rate: `1.0`
- Citation coverage: `1.0`
- ML accuracy: `1.0`
- ML macro F1: `1.0`

## Testing
Run the test suite locally:

```bash
python -m pytest tests -q
```

Current status:
- `13` passing tests

## Architecture Walkthrough
The current production-style flow is:

1. The user submits unseen clinical-style text.
2. The analysis service normalizes the note and starts the LangGraph workflow.
3. The classifier labels the document type.
4. The segmenter identifies sections or sentence blocks.
5. The retrieval agent chunks the input note, builds a document-local FAISS index, and retrieves evidence for each target field.
6. The extractor produces structured fields plus citations.
7. The prediction agent runs the trained TF-IDF + Logistic Regression risk model.
8. The validation agent checks support, missing fields, and contradictions.
9. The system returns a validated JSON response through the API or UI.

## Setup
Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional environment setup:

```bash
copy .env.example .env
```

If you have Ollama running locally, you can point the project at your local model through `.env`. If not, the system will fall back gracefully to deterministic local logic for extraction and validation.

## Limitations
- The dataset is synthetic and somewhat templated, so metrics are likely optimistic relative to noisier real-world notes.
- The current project scope is clinical-note-style documents rather than every medical document format.
- Performance is strongest on clearer clinical-note phrasing and weaker on shorthand, abbreviation-heavy, or typo-heavy notes.
- For example, notes written like `Pt is 58 y/o male w hx of CHF + CAD...` are more likely to miss diagnoses or follow-up extraction than cleaner sentence-based notes.
- The `sentence-transformers` backend is planned, but the current retrieval path uses a local fallback embedding backend due local environment constraints during development.
- Ollama integration is implemented through a swappable client interface, but live local-model behavior depends on the user environment.

## Notes
- This project uses synthetic clinical-style notes to avoid PHI/privacy issues.
- This is a portfolio/demo system and not a real clinical decision-support product.
