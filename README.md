# Financial GraphRAG: Multi-Year Corporate Intelligence

A production-ready **Financial GraphRAG application** engineered for extracting verifiable, multi-year financial insights from corporate annual reports and SEC 10-K filings.

Combines **Docling** structured PDF parsing, **semantic structure-aware chunking**, **PostgreSQL + pgvector** (768-dim, HNSW), **Neo4j 5 Financial Knowledge Graph**, **deterministic Python financial math**, and **LangGraph StateGraph** orchestration feeding a configurable **LLM** (Local Ollama, OpenAI, or Anthropic) and visualized through a **Streamlit research dashboard**.

---

## 4-Way Retrieval Comparison

| Mode | Context Source | Strengths | Failure Modes in Finance |
|---|---|---|---|
| **1. LLM without RAG** | Parametric weights only | Instantaneous; baseline test. | Outdated cutoff; hallucinates numerical figures. |
| **2. Traditional Vector RAG** | PostgreSQL + pgvector (HNSW) | Narrative prose, MD&A commentary. | Fails on cross-year tabular trends; splits tables across chunks. |
| **3. GraphRAG** | Neo4j Cypher & multi-hop traversal | Exact multi-year numbers; maps causal links (Strategy $\rightarrow$ Metric). | Lacks verbose narrative paragraphs. |
| **4. Hybrid Vector + GraphRAG** | Fused Graph + Vector + Python Math | **Best-in-class**: Exact audited numbers + zero-hallucination math + qualitative context. | Requires query routing and fusion. |

### Why GraphRAG Wins in Corporate Finance
1. **Multi-Year Continuity**: Queries spanning multiple annual reports (e.g. FY22 through FY26) resolve exact temporal nodes without passage truncation.
2. **Deterministic 3-Tier Math**: Reported Data $\rightarrow$ Python Math ($\pm$bps, YoY %, CAGR) $\rightarrow$ Qualitative LLM Interpretation. 0% arithmetic hallucination.
3. **Causal Reasoning**: Discovers strategic drivers (e.g., how AI investments drove BFSI segment growth).

---

## Architecture Pipeline

```
[Annual Reports / SEC 10-K Filings]
                 │
                 ▼
      Docling Document Parser (Table Preservation & Reading Order)
                 │
                 ▼
     Semantic Structure Chunker (Markdown Tables, Stable SHA-256 IDs)
                 │
        ┌────────┴────────────────────────┐
        ▼                                 ▼
PostgreSQL + pgvector            Neo4j 5 Knowledge Graph
(768-dim Nomics, HNSW,           (20 Node Types, 18 Relations,
 Metadata, Cosine Distance)        Multi-Year Audited Facts)
        │                                 │
        └────────┬────────────────────────┘
                 ▼
      Hybrid Retrieval Router (Classifies Query Intent)
                 │
                 ▼
      Financial Math Engine (Python YoY %, Margins, CAGR)
                 │
                 ▼
      LangGraph StateGraph (Executes No-RAG / Vector / Graph / Hybrid)
                 │
                 ▼
      LLM Provider (Ollama qwen2.5:3b / OpenAI / Anthropic)
                 │
                 ▼
    Streamlit Dashboard (8 Research & Analytics Tabs)
```

---

## Quickstart Guide

### Prerequisites
- **Docker Desktop** (running):
  - **Windows**: Enable **WSL2 backend** in Docker Desktop Settings $\rightarrow$ General.
  - **macOS / Linux**: Standard Docker Desktop or Docker Engine.
- **Python**: Version 3.10, 3.11, or 3.12 installed.
- **Git**: Installed and available in terminal.
- *(Optional Local LLM)*: [Ollama](https://ollama.com/) with `ollama pull qwen2.5:3b` and `ollama pull nomic-embed-text`.

---

### Windows Setup (PowerShell / Command Prompt)

#### Option A: One-Command Startup (Recommended)
Clone the repository and run the automated Windows startup script:

```powershell
git clone https://github.com/DEV-S-SHAH/GraphRAG-based-Financial-Insights.git
cd GraphRAG-based-Financial-Insights

# Run via PowerShell:
.\run.ps1

# Or run via Command Prompt (CMD) / Double-click:
run.bat
```

> **Tip (PowerShell Execution Policy):** If PowerShell blocks script execution, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```
> Or execute directly with bypass:
> ```powershell
> powershell -ExecutionPolicy Bypass -File .\run.ps1
> ```

#### Option B: Step-by-Step Manual Setup on Windows

```powershell
# 1. Clone and enter directory
git clone https://github.com/DEV-S-SHAH/GraphRAG-based-Financial-Insights.git
cd GraphRAG-based-Financial-Insights

# 2. Start Database Containers
docker compose up -d financial-postgres financial-neo4j

# 3. Create Python virtual environment
python -m venv .venv

# 4. Activate virtual environment:
# ── If using PowerShell:
.\.venv\Scripts\Activate.ps1
# ── If using Command Prompt (CMD):
.venv\Scripts\activate.bat
# ── If using Git Bash:
source .venv/Scripts/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Initialize Database Schemas & Knowledge Graph
python scripts\init_db.py

# 7. Launch Streamlit Dashboard
streamlit run ui\app.py
```

---

### macOS / Linux Setup (Terminal / zsh / bash)

#### Option A: One-Command Startup (Recommended)

```bash
git clone https://github.com/DEV-S-SHAH/GraphRAG-based-Financial-Insights.git
cd GraphRAG-based-Financial-Insights

# Make executable and run
chmod +x run.sh
./run.sh
```

#### Option B: Step-by-Step Manual Setup on macOS / Linux

```bash
# 1. Clone and enter directory
git clone https://github.com/DEV-S-SHAH/GraphRAG-based-Financial-Insights.git
cd GraphRAG-based-Financial-Insights

# 2. Start Database Containers
docker compose up -d financial-postgres financial-neo4j

# 3. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Initialize Database Schemas & Knowledge Graph
python scripts/init_db.py

# 6. Launch Streamlit Dashboard
streamlit run ui/app.py
```

---

## Access Points

| Interface | URL / Command | Credentials / Details |
|---|---|---|
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | Full interactive 8-tab intelligence suite |
| **Neo4j Browser** | [http://localhost:7474](http://localhost:7474) | User: `neo4j` \| Password: `password123` |
| **PostgreSQL CLI** | `docker exec -it financial-postgres psql -U financial_user -d financial_db` | Direct relational & pgvector SQL access |

---

## Configuration (`.env`)

Key environment variables in `.env`:

```ini
# PostgreSQL + pgvector
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=financial_db
POSTGRES_USER=financial_user
POSTGRES_PASSWORD=password123
DATABASE_URL=postgresql://financial_user:password123@localhost:5432/financial_db

# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
NEO4J_AUTH=neo4j/password123

# LLM Selection (ollama | openai | anthropic)
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434

# Optional Cloud API Keys (leave blank for local Ollama)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Embeddings
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
```

---

## Core Operations & CLI

### 1. Ingest Annual Reports
```bash
# Ingest local PDF filings in data/raw/annual_reports/
python scripts/ingest.py --source local

# Ingest 10-K filings via SEC EDGAR API
python scripts/ingest.py --source sec --ticker AAPL --filing 10-K --year 2023
```

### 2. Build the Knowledge Graph
```bash
# Populate Neo4j with multi-year facts, metrics, and relationships
python scripts/build_graph.py
```

### 3. Test Retrieval from CLI
```bash
python scripts/test_retrieval.py "Compare FY23 and FY24 revenue and margin performance"
```

### 4. Run Benchmark Evaluation
```bash
# Run 4-way evaluation across 9 financial categories
python scripts/run_eval.py --max-questions 9
```

### 5. Run Integration Test Suite
```bash
pytest tests/test_end_to_end.py -v
```

---

## Empirical Benchmark Results

| Paradigm | Retrieval Quality | Answer Accuracy | Faithfulness | Citation Accuracy | Avg Latency | Avg Tokens |
|---|---|---|---|---|---|---|
| **Vector RAG** | 0.45 | 0.22 | 1.00 | 1.00 | 5.00s | 2,428 |
| **Graph RAG** | **0.89** | **0.89** | 0.99 | 1.00 | **3.64s** | **1,467** |
| **Hybrid RAG** | **0.89** | **0.78** | **1.00** | **1.00** | 6.03s | 3,093 |

---

## Streamlit Dashboard Features (8 Tabs)

1. **RAG Comparison**: Live 4-way side-by-side answer generation, token usage, and latency.
2. **Financial Q&A**: Production assistant with strategy selector, verbatim citations, and math auditing.
3. **Financial Trends**: Interactive Plotly charts for multi-year Revenue, EBITDA, PAT, and Margins.
4. **Knowledge Graph**: Interactive PyVis force-directed graph with entity and relation filtering.
5. **Vector Search**: Semantic similarity tester with cosine distance scores and metadata filters.
6. **Document & DB Explorer**: Database health metrics, table row counts, and embedded SQL executor.
7. **Data Ingestion**: Web-based PDF uploader and SEC EDGAR fetcher.
8. **Benchmark Dashboard**: Visualization of empirical evaluation scores and failure mode breakdowns.

---

## Database Inspection Cheatsheet

### PostgreSQL (`psql`)
```sql
-- Connect: docker exec -it financial-postgres psql -U financial_user -d financial_db

-- Check chunk counts by fiscal year
SELECT fiscal_year, chunk_type, count(*) FROM vector_chunks GROUP BY fiscal_year, chunk_type;

-- Run cosine distance (<=>) vector similarity search
SELECT chunk_id, fiscal_year, section_name, left(content, 100)
FROM vector_chunks
ORDER BY embedding <=> (SELECT embedding FROM vector_chunks LIMIT 1)
LIMIT 5;
```

### Neo4j Cypher
```cypher
// Access at http://localhost:7474 (user: neo4j / password: password123)

// Multi-year revenue trend across all periods
MATCH (c:Company)-[:HAS_REPORT]->(r:AnnualReport)-[:HAS_METRIC]->(m:FinancialMetric {metric_type: 'revenue'})-[:HAS_VALUE]->(v:FinancialValue)-[:FOR_PERIOD]->(p:ReportingPeriod)
RETURN p.period_name AS Year, v.value AS Revenue, v.unit AS Unit
ORDER BY Year;

// Strategic drivers linked to financial metrics
MATCH (s:Strategy)-[rel:DRIVES|IMPROVES|SUPPORTS]->(m:FinancialMetric)
RETURN s.name AS Strategy, type(rel) AS Relation, m.name AS Metric;
```

---

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| **Port 5432 / 7474 in use** | Existing database running on host | Stop existing service (`brew services stop postgresql` on Mac, or Windows Services), or change port in `.env` and `docker-compose.yml`. |
| **PowerShell script execution error** | Restricted Windows execution policy | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in PowerShell. |
| **Docker not responding (Windows)** | WSL2 backend not enabled | Open Docker Desktop Settings $\rightarrow$ General $\rightarrow$ Enable "Use the WSL 2 based engine". |
| **Ollama connection refused** | Local Ollama service is stopped | Start Ollama (`ollama serve`) and ensure models are pulled (`ollama pull qwen2.5:3b`). |
| **Fresh Database Reset** | Want to wipe and re-initialize data | Run `docker compose down -v` followed by `docker compose up -d` and `python scripts/init_db.py`. |

---

## Project Structure

```
.
├── docker-compose.yml              # PostgreSQL + pgvector (5432), Neo4j 5 (7474, 7687)
├── run.sh / run.bat / run.ps1      # Single-command startup scripts (macOS, Linux, Windows)
├── Dockerfile                      # Application container image definition
├── docker/                         # Custom database container image definitions
├── database/                       # PostgreSQL client (HNSW) & Neo4j Knowledge Graph ontology
├── ingestion/                      # Docling structure-aware parser, chunker & sources (PDF, SEC, Web)
├── rag/                            # LangGraph workflow, hybrid router, financial math, LLM abstraction
├── ui/                             # 8-tab Streamlit research dashboard
├── evaluation/                     # 4-way evaluation benchmark suite and results
├── scripts/                        # Database init, ingestion, graph builder, eval CLIs
└── tests/                          # 14-point automated end-to-end integration test suite
```
