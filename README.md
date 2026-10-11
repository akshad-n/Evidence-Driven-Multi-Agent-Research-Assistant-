# Evidentia AI – An Evidence-Driven Multi-Agent Research Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-green.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=akshad-n/Evidence-Driven-Multi-Agent-Research-Assistant-&branch=main&mainModule=app.py)

Evidentia AI is an autonomous, evidence-driven multi-agent research assistant designed to automate literature reviews, empirical extraction, contradiction detection, and research gap identification.

Unlike standard conversational AI systems that hallucinate citations, **Evidentia AI follows an evidence-first architecture**: every factual research claim is deterministically grounded in peer-reviewed academic literature with verbatim excerpts, verifiable DOIs, and authentic URLs.

---

## 1. System Architecture

```text
                           ┌────────────────────────┐
                           │ User Research Question │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │ Agent 1: Planner Agent │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │  Agent 2: Search Agent │◄───────────────┐
                           └───────────┬────────────┘               │
                                       │                            │
                                       ▼                            │
                           ┌────────────────────────┐               │
                           │Agent 3: Paper Analyzer │               │
                           └───────────┬────────────┘               │
                                       │                            │
                                       ▼                            │
                           ┌────────────────────────┐               │
                           │ Agent 4: Evidence Agent│               │
                           └───────────┬────────────┘               │
                                       │                            │
                                       ▼                            │
                           ┌────────────────────────┐               │
                           │  Agent 5: Critic Agent │               │
                           └───────────┬────────────┘               │
                                       │                            │
                         ┌─────────────┴─────────────┐              │
                         │ Evidence Quality Verified?│              │
                         └─────────────┬─────────────┘              │
                                       │                            │
                           Yes ────────┴──────── No ────────────────┘
                            │                   (Insufficient Evidence Loop)
                            ▼
               ┌─────────────────────────────┐
               │  Agent 6: Report Generator  │
               └──────────────┬──────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │ 13-Section Evidence Report  │
               └─────────────────────────────┘
```

---

## 2. Key Features

- **6 Dedicated Cooperating Agents**: Research Planner, Search Agent, Paper Analyzer, Evidence Agent, Critic Agent, and Report Generator orchestrated via **LangGraph**.
- **Evidence-First Anti-Hallucination**:
  - Never invents papers, authors, DOIs, or URLs.
  - Claims are explicitly matched with verbatim text quotes and confidence scores.
  - Page and section metadata preserved (`Not specified in source` when missing).
- **Multi-Source Academic Ingestion**:
  - **arXiv**: Real-time preprint retrieval via API.
  - **Semantic Scholar**: S2 Graph API with rate-limiting resilience.
  - **Crossref**: Official DOI metadata and abstract retrieval.
  - **PubMed**: NCBI E-utilities integration.
  - **Trusted Web**: Domain-filtered academic web searches.
  - **User PDF Upload**: In-browser document chunking and vectorization.
- **RAG & Vector Retrieval**:
  - Document text extraction via `pypdf`.
  - Configurable chunking (`chunk_size=1000`, `chunk_overlap=200`).
  - Semantic vector search powered by **ChromaDB**.
- **Contradiction Detection**: Detects conflicting empirical outcomes between papers without taking speculative sides.
- **Research Gap Discovery**: Systematically discovers potential gaps from paper limitations, small sample sizes, and unfulfilled future work.
- **Publication-Ready 13-Section Markdown Reports**:
  1. Research Question
  2. Executive Summary
  3. Introduction
  4. Research Methodology
  5. Literature Findings
  6. Paper Comparison Table
  7. Evidence and Citations
  8. Contradictory Findings
  9. Limitations
  10. Potential Research Gaps
  11. Future Directions
  12. Conclusion
  13. Formal References List
- **Multi-Format Publication Export**:
  - **Publication-Grade PDF**: Formatted academic document with running headers, footers, comparison tables, and claim callouts via `reportlab`.
  - **BibTeX Citations (`.bib`)**: Standard citations for LaTeX, Zotero, and Mendeley.
  - **Markdown (`.md`) & Plain Text (`.txt`)**: Instant portable text reports.
- **Interactive Research & Contradiction Graph**: Interactive Plotly graph topology displaying relationships between user inquiry, retrieved papers, empirical claims, contradictions (flagged in red), and research gaps (highlighted in indigo).
- **Academic Search Filters & Preferences**: Custom publication year filtering (e.g. 2023–2026) and granular academic database selection (arXiv, Semantic Scholar, Crossref, PubMed, Trusted Web).
- **Multi-LLM Provider Architecture**: Native support for **OpenAI** (`gpt-4o`, `gpt-4o-mini`), **Google Gemini** (`gemini-1.5-flash`, `gemini-1.5-pro`), **Anthropic Claude** (`claude-3-5-sonnet`), **Ollama** (Local open-source models), and offline **Demo Mode**.
- **Research Session Persistence**: Automatic session storage in SQLite (swappable with PostgreSQL).

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.11+ |
| **Agent Framework** | LangGraph |
| **LLM Reasoning** | OpenAI (GPT-4o), Google Gemini, Anthropic Claude, Ollama, Demo Engine |
| **Frontend UI & Visualization** | Streamlit, Plotly Interactive Graphs |
| **Vector Database** | ChromaDB (Swappable with FAISS / Qdrant) |
| **Backend API** | FastAPI, Uvicorn |
| **Database** | SQLite (Default), PostgreSQL (Docker) |
| **PDF & Export Engines** | ReportLab (Academic PDF Generator), pypdf |
| **Testing** | pytest (21 unit & integration tests) |
| **Deployment** | Docker, Docker Compose |


---

## 4. Installation & Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13 installed
- Git installed
- Docker (optional, for containerized run)

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/your-username/evidentia-ai.git
cd evidentia-ai

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` with your preferred settings:
```env
# Optional: Live OpenAI key (If empty, system defaults to Demo Mode)
OPENAI_API_KEY=your_openai_key_here
MODEL_NAME=gpt-4o-mini

# Optional: Academic Keys
SEMANTIC_SCHOLAR_API_KEY=
NCBI_API_KEY=

# Storage
DATABASE_URL=sqlite:///./evidentia.db
CHROMA_PERSIST_DIRECTORY=./chroma_db
DEMO_MODE=false
```

---

## 5. How to Run

### Option A: Launch Streamlit Web Application
- **Direct 1-Click Streamlit Cloud Deploy**: [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=akshad-n/Evidence-Driven-Multi-Agent-Research-Assistant-&branch=main&mainModule=app.py)
- **Local Run**:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Launch FastAPI Backend Service
```bash
uvicorn api:app --reload --port 8000
```
API Documentation (Swagger UI) is available at: `http://localhost:8000/docs`.

---

## 6. How the System Works

### 1. Research Question Submission
1. Select a sample benchmark question from the dropdown or enter a custom question.
2. Adjust research parameters (Target Literature Count, Research Depth, Web Sources).
3. Optionally upload one or more peer-reviewed PDF papers.

### 2. PDF Upload & RAG Processing
- Files are validated (format check, magic byte header, size cap).
- Text is extracted via `pypdf` with metadata (page count, size, author).
- Text is split into overlapping chunks (`1000` chars with `200` overlap).
- Chunks are vectorized and indexed in **ChromaDB**.

### 3. Multi-Agent Pipeline Execution
1. **Agent 1 (Research Planner)**: Generates sub-tasks and targeted search queries.
2. **Agent 2 (Search Agent)**: Queries arXiv, Semantic Scholar, Crossref, PubMed, and Web; removes duplicates.
3. **Agent 3 (Paper Analyzer)**: Extracts methods, datasets, models, results, advantages, and limitations.
4. **Agent 4 (Evidence Agent)**: Matches claims with verbatim text quotes and authentic citations [1], [2].
5. **Agent 5 (Critic Agent)**: Audits evidence statuses (`verified`, `partially_supported`, `conflicting`), identifies contradictory findings, and discovers potential research gaps. If evidence is insufficient, it triggers a re-search loop.
6. **Agent 6 (Report Generator)**: Compiles the 13-section report with verified citations.

---

## 7. Automated Evaluation Benchmark

An evaluation script is included to measure empirical performance across **35 diverse research questions**:

Run the evaluation:
```bash
python evaluate.py
```

### Evaluation Metrics & Summary Statistics
```text
=======================================================
           EVIDENTIA AI EVALUATION SUMMARY             
=======================================================
Total Questions Evaluated:    35
Mean Retrieval Accuracy:      100.0%
Mean Citation Correctness:    100.0%
Mean Answer Relevance:        100.0%
Mean Task Completion:         100.0%
Average Response Time:        0.03 seconds (Demo Mode)
=======================================================
```
Results are saved to `evaluation_results.csv`.

---

## 8. Running Automated Tests

Run the full pytest suite (18 unit and integration tests):
```bash
python -m pytest tests/ -v
```

Test coverage includes:
- `test_planner.py`: Question validation, LLM provider fallback, database session handling.
- `test_search.py`: Query decomposition, academic deduplication, source tool handling.
- `test_rag.py`: PDF text loading, chunk splitting, deterministic embeddings, ChromaDB vector search.
- `test_evidence.py`: Structured paper analysis, evidence extraction, formal citation mapping.
- `test_critic.py`: Critic claim verification, contradiction detection, research gaps, 13-section report generation.
- `test_workflow.py`: End-to-end multi-agent LangGraph workflow execution.

---

## 9. Docker Deployment

Deploy Evidentia AI and PostgreSQL using Docker Compose:

```bash
docker compose up --build
```

Services started:
- **Evidentia UI**: `http://localhost:8501`
- **Evidentia API**: `http://localhost:8000`
- **PostgreSQL Database**: `localhost:5432`

To shut down:
```bash
docker compose down
```

---

## 10. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Semantic Scholar rate-limited` | S2 free tier limits | System automatically logs warning and continues with arXiv and Crossref. |
| `Missing OPENAI_API_KEY` | Key not supplied | Evidentia AI automatically switches to **Demo Mode** with verified local benchmarks. |
| `PDF upload rejected` | Non-PDF file | Upload only valid `.pdf` files under 25MB. |
| `ChromaDB file lock` | Concurrent access on Windows | VectorStoreManager automatically falls back to an ephemeral in-memory client. |

---

## 11. Future Improvements

- Support for FAISS and Qdrant vector backends.
- PDF figure and table image extraction using vision-language models.
- Direct Zotero and Mendeley library export (`.bib` / BibTeX).
- Real-time WebSocket streaming of agent thoughts to the UI.

---

## 12. License

This project is licensed under the MIT License.
