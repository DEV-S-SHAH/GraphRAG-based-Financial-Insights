# Financial GraphRAG: Multi-Year Corporate Intelligence

A production-ready **Financial GraphRAG system** for extracting verifiable, multi-year financial insights from corporate annual reports and SEC 10-K filings.

Combines **Docling** structured PDF parsing, **semantic chunking**, **PostgreSQL + pgvector** (1024-dim BGE-M3, HNSW), **Neo4j 5 Knowledge Graph**, **deterministic Python financial math**, and **LangGraph** orchestration feeding a configurable **LLM** (Local Ollama, OpenAI, or Anthropic) with an interactive **Streamlit dashboard**.

---

## 4-Way Retrieval Comparison

| Paradigm | Context Source | Strengths in Finance | Failure Modes |
|---|---|---|---|
| **1. LLM without RAG** | Parametric weights | Baseline test; instant response | Hallucinates figures; outdated training cutoff |
| **2. Traditional Vector RAG** | PostgreSQL + pgvector (HNSW) | Narrative prose & MD&A text retrieval | Cross-year trends fail; tables split across chunks |
| **3. GraphRAG** | Neo4j Cypher & multi-hop traversal | Audited multi-year facts; causal chains | Lacks long-form qualitative context |
| **4. Hybrid Vector + GraphRAG** | Fused Graph + Vector + Python Math | **Best-in-Class**: 100% audited numbers + 0-hallucination math + narrative context | Requires query routing and fusion |

---

## Quickstart Guide

### Prerequisites
- **Docker Desktop** (running, WSL2 backend on Windows)
- **Python 3.10 - 3.12**
- *(Optional Local LLM & Embeddings)*: [Ollama](https://ollama.com/) with:
  ```bash
  ollama pull qwen2.5:3b
  ollama pull bge-m3
  ```

---

### Option A: One-Command Startup (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/DEV-S-SHAH/GraphRAG-based-Financial-Insights.git
cd GraphRAG-based-Financial-Insights

# 2. Run automated startup:
# ── Windows (PowerShell):
.\run.ps1
# ── Windows (CMD / Double-click):
run.bat
# ── macOS / Linux:
chmod +x run.sh && ./run.sh
```

> **Windows PowerShell Execution Policy:** If script execution is blocked on Windows, run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

---

### Option B: Manual Setup

```bash
# 1. Start Database Containers
docker compose up -d financial-postgres financial-neo4j

# 2. Create & activate virtual environment
# Windows:
python -m venv .venv && .\.venv\Scripts\Activate.ps1
# macOS / Linux:
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize Database Schemas & Knowledge Graph
python scripts/init_db.py

# 5. Launch Dashboard
streamlit run ui/app.py
```

---

## Access Points

| Service | URL / Access | Default Credentials |
|---|---|---|
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | — |
| **Neo4j Browser** | [http://localhost:7474](http://localhost:7474) | `neo4j` / `password123` |
| **PostgreSQL CLI** | `docker exec -it financial-postgres psql -U financial_user -d financial_db` | `financial_user` / `password123` |

---

## Core CLI Commands

```bash
# Ingest pre-chunked annual reports (Instant load)
python scripts/ingest.py --source chunks

# Ingest local raw PDFs via Docling
python scripts/ingest.py --source local

# Ingest from SEC EDGAR API
python scripts/ingest.py --source sec --ticker AAPL

# Build & enrich Neo4j Knowledge Graph
python scripts/build_graph.py

# Run retrieval test query
python scripts/test_retrieval.py "Compare FY23 and FY24 revenue and margin performance"

# Run 4-way paradigm benchmark
python scripts/run_eval.py --max-questions 3

# Run complete integration test suite (14 tests)
pytest tests/test_end_to_end.py -v
```

---

## Empirical Benchmark Results

| Retrieval Approach | Answer Accuracy | Faithfulness | Citation Accuracy | Retrieval Recall | Avg Latency |
|---|---|---|---|---|---|
| **LLM without RAG** | 0.15 | 0.25 | 0.00 | 0.00 | **1.44s** |
| **Traditional Vector RAG** | 0.30 | 0.86 | **1.00** | 0.44 | 3.88s |
| **GraphRAG** | **0.82** | **1.00** | **1.00** | **0.75** | 4.74s |
| **Hybrid Vector + GraphRAG** | **0.82** | **1.00** | **1.00** | **0.75** | 8.90s |

---

## Configuration (`.env`)

```ini
# PostgreSQL + pgvector (127.0.0.1 prevents Windows IPv6 ::1 loopback collisions)
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=financial_db
POSTGRES_USER=financial_user
POSTGRES_PASSWORD=password123
DATABASE_URL=postgresql://financial_user:password123@127.0.0.1:5432/financial_db

# Neo4j Graph Database
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# LLM & Embeddings (Local Ollama default, or OpenAI / Anthropic)
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024
```

---

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| **Windows: PostgreSQL password authentication failed (`financial_user`)** | Native Windows PostgreSQL service running on port 5432, or `localhost` resolved to `::1`. | Ensure `POSTGRES_HOST=127.0.0.1` in `.env`. If native PostgreSQL is running, stop it via `services.msc` or change port to `5433:5432` in `docker-compose.yml` and set `POSTGRES_PORT=5433`. |
| **PowerShell script execution blocked** | Windows execution policy restrictions. | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` or use `run.bat`. |
| **Docker Desktop not connecting** | WSL2 backend disabled on Windows. | Open Docker Desktop Settings -> General -> Enable "Use the WSL 2 based engine". |
| **Ollama connection refused** | Local Ollama daemon not running. | Run `ollama serve` and pull models: `ollama pull qwen2.5:3b && ollama pull bge-m3`. |
| **Database Reset** | Corrupted or outdated container volume. | Run `docker compose down -v`, then `docker compose up -d` and `python scripts/init_db.py`. |

---

## Project Structure

```
.
├── docker-compose.yml          # PostgreSQL + pgvector (5432), Neo4j 5 (7474, 7687)
├── run.sh / run.bat / run.ps1  # 1-command startup scripts (Windows, macOS, Linux)
├── database/                   # PostgreSQL pgvector client & Neo4j Knowledge Graph
├── ingestion/                  # Docling PDF parser, semantic chunker, and sources
├── rag/                        # LangGraph workflow, hybrid router, financial math, LLMs
├── ui/                         # 8-tab Streamlit financial intelligence dashboard
├── evaluation/                 # 4-way evaluation benchmark suite and results
├── scripts/                    # Ingestion, graph builder, retrieval, eval CLIs
└── tests/                      # Automated end-to-end integration test suite
```
