# 🔍 Hybrid RAG Engine

A production-quality Retrieval-Augmented Generation system combining **dense** (vector) and **sparse** (BM25) retrieval with **Reciprocal Rank Fusion**, **cross-encoder reranking**, and **quantitative evaluation** via Ragas.

> **Resume Line:** Built a hybrid RAG system combining dense and sparse retrieval with cross-encoder reranking, improving retrieval recall@5 from X% to Y% (measured via Ragas), with full cost/latency tracing via Phoenix.

## Architecture

```
User Query
     │
     ▼
FastAPI (async, Pydantic schemas)
     │
     ├──► Dense Search (Qdrant, bge-small-en-v1.5 embeddings)
     ├──► Sparse Search (BM25 via rank_bm25)
     │
     ▼
Reciprocal Rank Fusion (merge both result sets)
     │
     ▼
Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
     │
     ▼
Top-K chunks ──► LLM (Groq) ──► Answer + Citations
     │
     ▼
Traced: Arize Phoenix │ Evaluated: Ragas
```

## Why Each Component

| Component | Problem It Solves |
|-----------|-------------------|
| **BM25 (sparse)** | Vector search misses exact keyword matches |
| **RRF Fusion** | Merges two ranked lists without needing score calibration |
| **Cross-encoder** | Bi-encoder embeddings compress detail; cross-encoder reads query+chunk together for precision |
| **Ragas Evals** | Proves improvements with numbers, not vibes |
| **Phoenix Tracing** | Visibility into latency, cost, and failure modes per step |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- `GROQ_API_KEY` (free at [console.groq.com](https://console.groq.com))

### 1. Setup

```bash
# Clone
git clone https://github.com/yourusername/hybrid-rag-engine.git
cd hybrid-rag-engine

# Configure
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Run with Docker Compose

```bash
docker compose up --build -d
```

This starts:
- **Qdrant** (vector DB) — `localhost:6333`
- **Phoenix** (observability) — `localhost:6006`
- **FastAPI** (API) — `localhost:8000`
- **Streamlit** (frontend) — `localhost:8501`

### 3. Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest
```

### 4. Query

```bash
# JSON response
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I define path parameters in FastAPI?", "mode": "hybrid_reranked"}'

# Or open the Streamlit UI
open http://localhost:8501
```

### 5. Run Evaluations

```bash
python scripts/evaluate.py --output evaluation_results.json
```

## Local Development (without Docker)

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Start Qdrant (still needs Docker)
docker run -d -p 6333:6333 qdrant/qdrant

# Run the API
uvicorn src.api.main:app --reload

# Run the frontend
streamlit run frontend/app.py
```

## Search Modes

| Mode | Pipeline | Use Case |
|------|----------|----------|
| `naive` | Vector search only | Baseline measurement |
| `hybrid` | Vector + BM25 + RRF | Better recall for exact terms |
| `hybrid_reranked` | Above + cross-encoder | Best precision for final answers |

## Evaluation Results

| Configuration | Context Precision | Context Recall | Faithfulness |
|---|---|---|---|
| Naive (vector only) | _TBD_ | _TBD_ | _TBD_ |
| Hybrid (+ BM25/RRF) | _TBD_ | _TBD_ | _TBD_ |
| Hybrid + Reranker | _TBD_ | _TBD_ | _TBD_ |

_Run `python scripts/evaluate.py` to fill in real numbers._

## Tech Stack

| Layer | Tool |
|---|---|
| API | FastAPI, Pydantic, async/await |
| Vector DB | Qdrant (local via Docker) |
| Sparse retrieval | `rank_bm25` |
| Embeddings | `bge-small-en-v1.5` (free, local) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API (free tier, fast) |
| Evals | Ragas (faithfulness, context precision/recall) |
| Observability | Arize Phoenix |
| Containerization | Docker + Docker Compose |
| Frontend | Streamlit |

## Project Structure

```
├── src/
│   ├── config.py              # Settings (pydantic-settings)
│   ├── models.py              # Pydantic schemas
│   ├── ingestion/             # Load → Chunk → Index pipeline
│   ├── retrieval/             # Dense, Sparse, Fusion, Reranker
│   ├── generation/            # Groq LLM (sync + streaming)
│   ├── observability/         # Phoenix OTEL tracing
│   ├── evaluation/            # Ragas metrics + comparison
│   └── api/                   # FastAPI routes
├── frontend/                  # Streamlit chat UI
├── scripts/                   # CLI tools (ingest, evaluate)
├── data/                      # Corpus documents
├── tests/                     # pytest suite
├── docker-compose.yml
└── Dockerfile
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT
