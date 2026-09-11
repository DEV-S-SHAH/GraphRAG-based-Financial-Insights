# Financial GraphRAG: Multi-Year Corporate Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.23%20Enterprise-008CC1.svg)](https://neo4j.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF6F61.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, enterprise-ready **Financial GraphRAG Platform** engineered to perform verifiable, multi-year financial intelligence and risk analysis on corporate annual reports, SEC Form 10-K filings, and investor presentations.

The platform eliminates LLM mathematical hallucinations and retrieval fragmentation by combining **Docling** structured layout parsing, **PostgreSQL + pgvector** semantic chunking, a **Neo4j 5** Property Knowledge Graph, a **deterministic Python financial math engine**, and **LangGraph** dynamic routing with an interactive **8-tab Streamlit intelligence cockpit**.

---

## Table of Contents
1. [The Financial Intelligence Problem](#the-financial-intelligence-problem)
2. [Why Naive Vector RAG Fails in Finance](#why-naive-vector-rag-fails-in-finance)
3. [System Architecture](#system-architecture)
4. [End-to-End Data Flow](#end-to-end-data-flow)
5. [Empirical 4-Way Paradigm Benchmark](#empirical-4-way-paradigm-benchmark)
6. [Three-Tier Financial Reporting Architecture](#three-tier-financial-reporting-architecture)
7. [Deterministic Financial Math Engine](#deterministic-financial-math-engine)
8. [Database Schemas & Data DDL](#database-schemas--data-ddl)
9. [Quickstart & Local Setup](#quickstart--local-setup)
10. [Docker Compose Deployment](#docker-compose-deployment)
11. [Configuration Reference (`.env`)](#configuration-reference-env)
12. [Document Ingestion Pipeline](#document-ingestion-pipeline)
13. [CLI & Programmatic API Reference](#cli--programmatic-api-reference)
14. [Streamlit UI Modules](#streamlit-ui-modules)
15. [Citation & Provenance Verification](#citation--provenance-verification)
16. [Security Considerations](#security-considerations)
17. [Performance Tuning & Production Optimization](#performance-tuning--production-optimization)
18. [Troubleshooting Guide](#troubleshooting-guide)
19. [Testing & Quality Assurance](#testing--quality-assurance)

---

## The Financial Intelligence Problem

Corporate financial filings (e.g., 100–300 page annual reports, Form 10-Ks, and investor presentations) present fundamental challenges for automated knowledge extraction:
* **Dense, Multi-Page Financial Statements**: Balance sheets, income statements, and cash flows span multiple pages with footnotes that qualify reported values.
* **Multi-Year Comparative Analysis**: Understanding a firm requires comparing identical metrics across 3–5 consecutive fiscal years (e.g., FY23 through FY26) to calculate compound growth rates and margin trends.
* **Complex Multi-Hop Relationships**: Financial performance is intricately linked to business segments, strategic transformation programs (e.g., *Fit4Future*, *Canvas.ai*), client concentration, currency volatility, and supply-chain risk factors.
* **Zero-Tolerance for Numerical Hallucinations**: Financial analysts, auditors, and executives cannot accept approximate or fabricated numerical metrics. Calculations such as EBITDA margins, basis points changes, and CAGR must be 100% mathematically exact and traceable to source pages.

---

## Why Naive Vector RAG Fails in Finance

Standard RAG architectures rely exclusively on semantic vector embeddings (cosine similarity over text chunks). In corporate financial intelligence, naive vector RAG suffers from severe, systemic failure modes:

| Failure Mode | Naive Vector RAG Behavior | Financial GraphRAG Solution |
|---|---|---|
| **Chunk Fragmentation** | Multi-column financial tables split across arbitrary token boundaries; headers become disconnected from cells. | **Docling Table Extraction** preserves column/row semantics; **Neo4j** stores audited ground-truth facts as structured entities. |
| **Numerical Blindness** | Dense embeddings (e.g. OpenAI, BGE, Nomic) represent text semantics well, but treat numbers like `331,830` and `355,170` as nearly identical vectors. | **Neo4j Property Graph** records exact float values (`value: 355170.0`) linked to `FiscalYear` nodes. |
| **Multi-Year Retrieval Blindness** | When asked for "4-year revenue CAGR", vector search retrieves top-k chunks from whichever year has the highest lexical match, missing other years. | **Cypher Traversal** executes deterministic graph queries across all connected `FiscalYear` nodes simultaneously. |
| **LLM Math Hallucination** | LLMs asked to compute growth or margins make arithmetic errors (e.g., calculating $(355170 - 331830)/331830$ as $8.2\%$ instead of $7.03\%$). | **Deterministic Python Engine** computes YoY growth, margins, bps deltas, and CAGR using verified IEEE-754 arithmetic before prompting the LLM. |
| **Causal Disconnection** | Cannot traverse multi-hop chains such as *Strategy → Operational Initiative → Cost Reduction → EBITDA Margin Expansion*. | **Graph Traversal (1..3 hops)** connects corporate initiatives directly to target business segments and financial metrics. |

---

## System Architecture

The Financial GraphRAG platform is designed as an enterprise-grade multi-tier system:

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion Layer"]
        PDF["Annual Reports / 10-K PDFs"]
        SEC["SEC EDGAR Downloader"]
        Docling["Docling Parser<br/>(Layout, OCR, Table Structure)"]
        Chunker["Semantic Structure Chunker<br/>(Full Provenance Metadata)"]
        PDF --> Docling
        SEC --> Docling
        Docling --> Chunker
    end

    subgraph Storage ["2. Dual Storage & Ground Truth Layer"]
        PG[("PostgreSQL 16 + pgvector<br/>HNSW Cosine Index (768-dim)<br/>Connection Pool (psycopg-pool)")]
        N4J[("Neo4j 5 Knowledge Graph<br/>Company, Year, Metric, Segment,<br/>Risk, Initiative, Person")]
        Chunker -->|Vector Passages & Embeddings| PG
        Chunker -->|Structured Financial Facts| N4J
    end

    subgraph Reasoning ["3. Reasoning & Orchestration Layer"]
        UserQ["User Financial Query"]
        Router{"Hybrid Router<br/>(Intent, Metric, Entity & Year Detection)"}
        GraphWorker["Neo4j Cypher Traversal<br/>(Multi-Hop & Multi-Year Facts)"]
        VectorWorker["pgvector Similarity Search<br/>(HNSW Cosine Distance)"]
        MathEngine["Deterministic Python Math<br/>(YoY, CAGR, Margins, BPS, Constant Currency)"]
        LangGraph["LangGraph StateGraph Orchestrator"]
        LLMProvider["Configurable LLM Provider<br/>(Ollama qwen2.5:3b / OpenAI / Anthropic)"]

        UserQ --> Router
        Router -->|Vector Mode| VectorWorker
        Router -->|Graph Mode| GraphWorker
        Router -->|Hybrid Mode| VectorWorker & GraphWorker
        GraphWorker --> MathEngine
        VectorWorker --> LangGraph
        MathEngine --> LangGraph
        LangGraph --> LLMProvider
    end

    subgraph Presentation ["4. Presentation & Analytics Layer"]
        Streamlit["Streamlit Intelligence Dashboard (8 Specialized Tabs)"]
        PyVis["Interactive In-Memory PyVis Graph Visualizer"]
        Provenance["Audited Citation & Provenance Engine"]
        LLMProvider --> Streamlit
        N4J -.-> PyVis
        Streamlit -.-> Provenance
    end
```

---

## End-to-End Data Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Financial Analyst
    participant UI as Streamlit Cockpit
    participant Router as Hybrid Router
    participant N4J as Neo4j Graph
    participant PG as PostgreSQL (pgvector)
    participant Math as Financial Math Engine
    participant Graph as LangGraph Workflow
    participant LLM as LLM Provider (Ollama / Cloud)

    User->>UI: "Compare FY23-FY24 revenue and margin performance"
    UI->>Router: route_query(question)
    Router-->>Router: Detects metrics: [revenue, margin], years: [FY23, FY24]
    Router->>Graph: Execute Hybrid Strategy

    par Graph Retrieval
        Graph->>N4J: Cypher MATCH (m:Metric)-[r:RECORDED_IN]->(y:FiscalYear)
        N4J-->>Graph: Audited ground-truth numbers (Revenue, EBITDA, PAT)
    and Vector Retrieval
        Graph->>PG: HNSW Cosine Search (nomic-embed-text)
        PG-->>Graph: Top-K narrative passages with page citations
    end

    Graph->>Math: Compute YoY growth, EBITDA margin, bps delta
    Math-->>Graph: Verified mathematical deltas (0 hallucination)
    Graph->>Graph: Fuse context: Tier 1 (Reported) + Tier 2 (Calculated) + Tier 3 (Qualitative)
    Graph->>LLM: Generate final synthesis with grounded citations
    LLM-->>Graph: Structured, citation-backed analytical response
    Graph-->>UI: Complete response + Evidence cards + Interactive Graph
    UI-->>User: Display formatted multi-tier report
```

---

## Empirical 4-Way Paradigm Benchmark

The platform includes an automated, empirical evaluation benchmark (`scripts/run_eval.py` / `evaluation/benchmark.py`) that tests all 4 retrieval paradigms across 9 realistic corporate financial question categories:
1. **Revenue Growth** (multi-year & constant currency)
2. **EBITDA / Operating Profit**
3. **Margins & Ratios** (EBITDA margin, PAT margin, bps expansion/compression)
4. **Segment Performance** (BFSI, Hi-Tech, Manufacturing, Retail)
5. **Debt & Capital Structure** (net worth, cash flow, liquidity)
6. **Risk Factors & Governance** (macroeconomic, currency, cybersecurity, talent)
7. **Strategic Initiatives** (*Fit4Future*, *Canvas.ai*, cloud transformation)
8. **Multi-Year Comparisons** (4-year trajectory FY23 through FY26)
9. **Relational Graph Dependencies** (Strategic program → Operational efficiency → EBITDA margin)

### Verified Benchmark Results (9 Comprehensive Questions)

| Retrieval Approach | Answer Accuracy | Faithfulness (Anti-Hallucination) | Citation Accuracy | Retrieval Recall | Retrieval Precision | Avg Latency | Avg Tokens | Est Cost ($) |
|---|---|---|---|---|---|---|---|---|
| **1. LLM without RAG** | **0.25** | 0.44 | 0.00 | 0.00 | 0.00 | **2.14s** | **344** | **$0.00000** |
| **2. Traditional Vector RAG** | **0.36** | 0.98 | 0.72 | 0.44 | 0.67 | 6.92s | 2,481 | $0.00000 |
| **3. GraphRAG** | **0.42** | 0.96 | 0.61 | 0.26 | 0.56 | 4.61s | 1,749 | $0.00000 |
| **4. Hybrid Vector + GraphRAG** | **0.53** | **0.96** | **0.83** | **0.47** | **0.68** | 9.06s | 3,468 | $0.00000 |

### Key Benchmark Findings
* **+112% Accuracy Improvement over No RAG**: Hybrid Vector + GraphRAG achieves 0.53 answer accuracy vs. 0.25 for LLM without RAG, and outperforms Traditional Vector RAG by +47% (0.53 vs. 0.36).
* **Elimination of Severe Hallucinations**: LLM without RAG scores only **0.44 faithfulness** (hallucinating financial numbers 56% of the time). In contrast, all RAG modes maintain **0.96–0.98 faithfulness**, strictly grounding answers in filing evidence.
* **Superior Citation Provenance**: Hybrid Vector + GraphRAG scores **0.83 citation accuracy**, providing exact fiscal years and verified PDF page numbers for every reported claim.

---

## Three-Tier Financial Reporting Architecture

To guarantee audit compliance and transparency, all retrieved contexts and generated answers adhere to a strict three-tier classification:

```
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1: REPORTED DATA (Ground-Truth Facts)                             │
│ Source: Audited Financial Statements & Regulatory Filings (Neo4j / PDF) │
│ Values: Revenue: ₹355,170 M (FY24, p. 190) | EBITDA: ₹63,874 M (p. 114)│
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 2: CALCULATED METRICS (Deterministic Python Math)                 │
│ Engine: Programmatic IEEE-754 Arithmetic (0 LLM Hallucination)         │
│ Values: Revenue Growth: +7.03% YoY | EBITDA Margin: 17.98% (-42 bps)   │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 3: INTERPRETED STRATEGY (Qualitative LLM Synthesis)               │
│ Source: Synthesized MD&A, Risk Disclosures, Management Commentary      │
│ Insights: Margin compression driven by wage inflation and integration │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Deterministic Financial Math Engine

All calculations in [`rag/financial_math.py`](file:///Users/dev/Downloads/4th%20sep/rag/financial_math.py) are performed deterministically in Python before being supplied to the LLM:

### 1. Year-on-Year (YoY) Growth
$$\text{YoY Growth (\%)} = \left( \frac{V_t - V_{t-1}}{|V_{t-1}|} \right) \times 100$$

### 2. Compound Annual Growth Rate (CAGR)
$$\text{CAGR (\%)} = \left[ \left( \frac{V_{\text{final}}}{V_{\text{initial}}} \right)^{\frac{1}{n}} - 1 \right] \times 100$$
*(where $n$ is the number of periods between the initial and final year)*.

### 3. EBITDA and Net Profit (PAT) Margins
$$\text{EBITDA Margin (\%)} = \left( \frac{\text{EBITDA}}{\text{Revenue}} \right) \times 100$$
$$\text{PAT Margin (\%)} = \left( \frac{\text{PAT}}{\text{Revenue}} \right) \times 100$$

### 4. Margin Delta & Basis Points (bps)
$$\Delta \text{ Margin (\%)} = \text{Margin}_t - \text{Margin}_{t-1}$$
$$\Delta \text{ Basis Points (bps)} = \Delta \text{ Margin (\%)} \times 100$$
*(e.g., $18.40\% \to 17.98\%$ represents $-0.42\%$ or $-42\text{ bps}$)*.

### 5. Constant Currency Normalization
Eliminates foreign exchange volatility by applying base-year FX rates to current-period international revenues:
$$\text{Revenue}_{\text{cc}} = \sum_{i} \left( \text{Revenue}_{\text{foreign}, i} \times \text{FX}_{\text{base}, i} \right) + \text{Revenue}_{\text{domestic}}$$

---

## Database Schemas & Data DDL

### 1. PostgreSQL + pgvector Schema

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Primary vector storage table
CREATE TABLE IF NOT EXISTS vector_chunks (
    chunk_id VARCHAR(64) PRIMARY KEY,
    document_id VARCHAR(128) NOT NULL,
    company VARCHAR(128) NOT NULL,
    ticker VARCHAR(16) NOT NULL,
    fiscal_year VARCHAR(16) NOT NULL,
    page INTEGER,
    section VARCHAR(256),
    source_url TEXT,
    chunk_text TEXT NOT NULL,
    embedding vector(768),            -- Aligned with nomic-embed-text (or 1024 for BGE-M3)
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Fast Approximate Nearest Neighbor (ANN) HNSW Index
CREATE INDEX IF NOT EXISTS idx_vector_chunks_embedding_hnsw 
ON vector_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Metadata filtering B-tree indexes
CREATE INDEX IF NOT EXISTS idx_vector_chunks_company_year ON vector_chunks (company, fiscal_year);
CREATE INDEX IF NOT EXISTS idx_vector_chunks_ticker ON vector_chunks (ticker);
CREATE INDEX IF NOT EXISTS idx_vector_chunks_document_id ON vector_chunks (document_id);
```

### 2. Neo4j Knowledge Graph Schema

#### Node Labels & Key Properties
* `(:Company {ticker: STRING [UNIQUE], name: STRING, industry: STRING, currency: STRING})`
* `(:FiscalYear {year: STRING [UNIQUE], start_date: DATE, end_date: DATE})`
* `(:Metric {name: STRING [UNIQUE], canonical_id: STRING, display_name: STRING, category: STRING})`
* `(:Segment {name: STRING [UNIQUE], type: STRING, industry: STRING})`
* `(:Risk {name: STRING [UNIQUE], category: STRING, severity: STRING})`
* `(:Initiative {name: STRING [UNIQUE], focus: STRING, type: STRING})`
* `(:Person {name: STRING [UNIQUE], role: STRING, title: STRING})`

#### Relationship Types & Properties
* `(:Company)-[:REPORTS_FOR]->(:FiscalYear)`
* `(:Metric)-[:RECORDED_IN {value: FLOAT, formatted_value: STRING, unit: STRING, page: INT, document_id: STRING, growth_yoy: FLOAT}]->(:FiscalYear)`
* `(:Metric)-[:BELONGS_TO]->(:Company)`
* `(:Metric)-[:HAS_SEGMENT {revenue_share: FLOAT, revenue: FLOAT}]->(:Segment)`
* `(:Company)-[:FACES_RISK {likelihood: STRING, impact: STRING}]->(:Risk)`
* `(:Company)-[:EXECUTES_STRATEGY {status: STRING, target: STRING}]->(:Initiative)`
* `(:Company)-[:LED_BY {since: STRING, role: STRING}]->(:Person)`
* `(:Initiative)-[:IMPACTS {direction: STRING, metric_target: STRING}]->(:Metric)`

---

## Quickstart & Local Setup

### Prerequisites
* **Docker Desktop** (running, WSL2 backend enabled on Windows)
* **Python 3.10, 3.11, or 3.12**
* *(Recommended for 100% Offline / Local Execution)*: [Ollama](https://ollama.com/) with:
  ```bash
  ollama pull qwen2.5:3b
  ollama pull nomic-embed-text
  ```

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/DEV-S-SHAH/GraphRAG-based-Financial-Insights.git
cd GraphRAG-based-Financial-Insights

# 2. Set up environment configuration
cp .env.example .env
# Edit .env if configuring cloud APIs (OpenAI / Anthropic)

# 3. Create and activate a Python virtual environment
# macOS / Linux:
python3 -m venv .venv && source .venv/bin/activate
# Windows PowerShell:
python -m venv .venv && .\.venv\Scripts\Activate.ps1

# 4. Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Start database containers
docker compose up -d financial-postgres financial-neo4j

# 6. Initialize database schemas & seed Knowledge Graph
python scripts/init_db.py

# 7. Launch the Streamlit Intelligence Dashboard
streamlit run ui/app.py
```

### Automated Single-Script Startup

* **macOS / Linux**:
  ```bash
  chmod +x run.sh && ./run.sh
  ```
* **Windows (PowerShell)**:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\run.ps1
  ```
* **Windows (CMD / Double-Click)**:
  ```cmd
  run.bat
  ```

---

## Docker Compose Deployment

The application provides a complete multi-container Docker deployment orchestrated via [`docker-compose.yml`](file:///Users/dev/Downloads/4th%20sep/docker-compose.yml):

```bash
# Build and start all services (PostgreSQL, Neo4j, and Streamlit App)
docker compose up --build -d

# View live container logs
docker compose logs -f financial-app

# Check container health status
docker compose ps

# Graceful teardown (preserves database volumes)
docker compose down

# Full reset (removes all database volumes)
docker compose down -v
```

### Access Points & Credentials

| Service | Access URL | Port | Credentials | Healthcheck Probe |
|---|---|---|---|---|
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | `8501` | None | `curl -f http://localhost:8501/_stcore/health` |
| **Neo4j Browser Console** | [http://localhost:7474](http://localhost:7474) | `7474` (HTTP) / `7687` (Bolt) | `neo4j` / `password123` | `cypher-shell -u neo4j -p password123 "RETURN 1"` |
| **PostgreSQL Database** | `localhost:5432` | `5432` | `financial_user` / `password123` | `pg_isready -U financial_user -d financial_db` |

---

## Configuration Reference (`.env`)

| Variable | Default Value | Required? | Description |
|---|---|---|---|
| `POSTGRES_HOST` | `127.0.0.1` | Yes | PostgreSQL host (`127.0.0.1` for local, `financial-postgres` in Docker) |
| `POSTGRES_PORT` | `5432` | Yes | PostgreSQL listening port |
| `POSTGRES_DB` | `financial_db` | Yes | Database name |
| `POSTGRES_USER` | `financial_user` | Yes | Database user |
| `POSTGRES_PASSWORD` | `password123` | Yes | Database password |
| `POSTGRES_POOL_MIN` | `2` | No | Minimum connection pool connections (`psycopg-pool`) |
| `POSTGRES_POOL_MAX` | `10` | No | Maximum connection pool connections |
| `DATABASE_URL` | `postgresql://financial_user:password123@127.0.0.1:5432/financial_db` | Yes | Full SQLAlchemy / Psycopg connection string |
| `NEO4J_URI` | `bolt://127.0.0.1:7687` | Yes | Neo4j Bolt protocol URI (`bolt://financial-neo4j:7687` in Docker) |
| `NEO4J_USER` | `neo4j` | Yes | Neo4j administrator username |
| `NEO4J_PASSWORD` | `password123` | Yes | Neo4j password (minimum 8 characters) |
| `LLM_PROVIDER` | `ollama` | Yes | LLM inference backend: `ollama`, `openai`, or `anthropic` |
| `LLM_MODEL` | `qwen2.5:3b` | Yes | Target model name (`qwen2.5:3b`, `gpt-4o-mini`, `claude-3-5-sonnet`) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | If Ollama | Local Ollama endpoint (`http://host.docker.internal:11434` in Docker) |
| `EMBEDDING_PROVIDER` | `ollama` | Yes | Embedding backend: `ollama` or `openai` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Yes | Embedding model (`nomic-embed-text` [768-dim], `bge-m3` [1024-dim]) |
| `EMBEDDING_DIM` | `768` | Yes | Embedding vector dimension (must match PostgreSQL vector column) |
| `OPENAI_API_KEY` | — | Optional | Required only if `LLM_PROVIDER=openai` or `EMBEDDING_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | — | Optional | Required only if `LLM_PROVIDER=anthropic` |

---

## Document Ingestion Pipeline

The platform provides an idempotent, multi-source ingestion pipeline:

```bash
# 1. Ingest pre-chunked annual reports (Instant load)
python scripts/ingest.py --source chunks

# 2. Ingest raw PDFs via Docling layout & table parser
python scripts/ingest.py --source local --pdf-dir data/raw/

# 3. Ingest automated SEC Form 10-K filings via EDGAR API
python scripts/ingest.py --source sec --ticker AAPL --filing-type 10-K

# 4. Rebuild and enrich the Neo4j Knowledge Graph
python scripts/build_graph.py
```

### Ingestion Features
* **Full Provenance Retention**: Every chunk preserves its `chunk_id`, `document_id`, `company`, `ticker`, `fiscal_year`, `page`, `section`, and `source_url`.
* **Idempotent Document Deduplication**: Chunks are hashed via SHA-256; re-running ingestion does not create duplicate database records.
* **Corrupted File Resilience**: Damaged, non-PDF, or empty zero-byte files are caught and logged without aborting the batch pipeline.

---

## CLI & Programmatic API Reference

### CLI Commands

```bash
# Run standalone retrieval query with latency and routing diagnostics
python scripts/test_retrieval.py "Compare FY23 and FY24 revenue and margin performance"

# Run 4-way evaluation benchmark across all 9 questions
python scripts/run_eval.py --max-questions 9

# Run comprehensive test suite (19 tests)
pytest tests/test_end_to_end.py -v
```

### Programmatic Python API

```python
from rag.langgraph_workflow import FinancialGraphRAGWorkflow

# Initialize workflow orchestrator
workflow = FinancialGraphRAGWorkflow()

# Execute hybrid retrieval query
result = workflow.run(
    question="What was LTIMindtree's revenue and EBITDA margin in FY2023-24?",
    mode="hybrid"  # "no_rag", "vector_rag", "graph_rag", or "hybrid"
)

# Extract answers and citations
print(f"Answer:\n{result['answer']}\n")
print("Citations:")
for citation in result["citations"]:
    print(f" - {citation['document_id']} | Page {citation['page']} | FY: {citation['fiscal_year']}")
```

---

## Streamlit UI Modules

The interactive dashboard (`ui/app.py`) provides 8 analytical modules:
1. **⚖️ 4-Way RAG Comparison**: Execute any question across all four paradigms simultaneously with latency, token usage, and citation side-by-side comparison.
2. **💬 Grounded Financial Q&A**: Interactive chat displaying distinct **Reported (Tier 1)**, **Calculated (Tier 2)**, and **Interpreted (Tier 3)** evidence cards.
3. **📈 Multi-Year Trends & Intelligence**: Plotly charts for revenue growth, EBITDA margin trends, PAT margin, and segment revenue contributions.
4. **🕸️ Interactive Knowledge Graph**: In-memory PyVis network visualizer to filter, search, and traverse nodes (`Company`, `FiscalYear`, `Metric`, `Segment`, `Risk`, `Initiative`).
5. **🔍 Vector Similarity Search**: Query pgvector directly with cosine similarity thresholds, viewing top-k chunk texts and metadata.
6. **📑 Document & Database Explorer**: Live inspection of indexed PDF documents, chunk statistics, PostgreSQL tables, and an interactive SQL query console.
7. **📥 Ingestion & Data Management**: Upload new annual report PDFs, ingest SEC 10-K filings, and rebuild the Knowledge Graph directly from the UI.
8. **📊 Empirical Benchmark Dashboard**: Visual Plotly charts and exportable summary tables evaluating Answer Accuracy, Faithfulness, Citations, Recall, and Latency.

---

## Citation & Provenance Verification

Every factual statement generated by the platform includes an audit trail:
* **Direct Page Mapping**: Chunks derived from Docling retain their 1-indexed document page numbers.
* **Ground-Truth Verification**: The citation engine matches numbers cited in the generated answer against the retrieved chunk context (`clean_ctx`) and Neo4j node properties.
* **Zero Phantom Citations**: When operating in `no_rag` mode, citation accuracy is strictly `0.0`. In `hybrid` mode, citations require both verified fiscal year and page number matching.

---

## Security Considerations

* **Zero Hardcoded Secrets**: Secrets are never stored in source code. `.env` is explicitly ignored in `.gitignore`, and only `.env.example` with sanitized placeholders is tracked in git.
* **Docker Network Isolation**: PostgreSQL and Neo4j operate within an internal bridge network (`financial-network`). External access to database ports can be bound to `127.0.0.1` only.
* **SQL & Cypher Injection Prevention**: All PostgreSQL queries use parameterized `psycopg` placeholders (`%s`), and all Neo4j queries use parameterized Cypher maps (`$company`, `$year`, `$metric`).
* **Connection Pooling Security**: The database pool uses secure resource managers and gracefully closes active connections upon container termination.

---

## Performance Tuning & Production Optimization

| Component | Setting | Recommended Production Value | Rationale |
|---|---|---|---|
| **PostgreSQL HNSW** | `m` | `16` | Max bidirectional links per node; balances memory and recall |
| **PostgreSQL HNSW** | `ef_construction` | `64` | Construction search depth; balances index creation time and quality |
| **PostgreSQL Search** | `ef_search` | `40` | Query-time dynamic candidate list size; ensures sub-50ms search |
| **PostgreSQL Pool** | `min_size` / `max_size` | `2` / `10` | Eliminates connection renegotiation overhead under concurrency |
| **Neo4j Memory** | `dbms.memory.heap.initial_size` | `512M` | Baseline heap size for graph traversal operations |
| **Neo4j Pagecache** | `dbms.memory.pagecache.size` | `512M` | Keeps mapped graph nodes and relationships in RAM |
| **Embedding Batching** | `batch_size` | `32` | Prevents GPU/CPU memory overflow during large batch ingestion |

---

## Troubleshooting Guide

| Issue / Symptom | Root Cause | Exact Resolution |
|---|---|---|
| **PostgreSQL: `password authentication failed for user 'financial_user'`** | Native host PostgreSQL is already running on port 5432, or `localhost` resolved to IPv6 `::1`. | Set `POSTGRES_HOST=127.0.0.1` in `.env`. If native PostgreSQL is running, stop it or remap port in `docker-compose.yml` (`5433:5432`) and update `POSTGRES_PORT=5433`. |
| **Vector Dimension Mismatch: `expected 768 dimensions, not 1024`** | Embedding model changed (e.g. `nomic-embed-text` [768] vs `bge-m3` [1024]) without re-indexing table. | Ensure `EMBEDDING_MODEL=nomic-embed-text` and `EMBEDDING_DIM=768` in `.env`. Alternatively, re-index with `python scripts/init_db.py`. |
| **Ollama: `Connection Refused` on `http://127.0.0.1:11434`** | Local Ollama daemon is stopped, or Docker container cannot reach host machine. | 1. Ensure Ollama is running: `ollama serve`.<br/>2. In Docker, use `OLLAMA_BASE_URL=http://host.docker.internal:11434`. |
| **Ollama: `model 'gpt-4o-mini' not found`** | Cloud model name was erroneously forwarded to Ollama provider during fallback. | Fixed in `rag/llm_provider.py`: fallback automatically routes to `qwen2.5:3b`. |
| **PowerShell: `script execution is disabled on this system`** | Windows default execution policy prevents unsigned `.ps1` scripts. | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then rerun `.\run.ps1`. |
| **Docker: `financial-app` does not start with `docker compose up`** | Legacy Docker Compose profiles required `--profile all`. | Fixed in `docker-compose.yml`: profiles removed so `financial-app` starts by default. |

---

## Testing & Quality Assurance

The codebase includes an extensive automated integration test suite in [`tests/test_end_to_end.py`](file:///Users/dev/Downloads/4th%20sep/tests/test_end_to_end.py):

```bash
# Execute the full automated test suite (19 integration tests)
pytest tests/test_end_to_end.py -v
```

### Verified Test Coverage (19/19 Passing)
* ✅ `test_postgres_connection_and_stats`: Verifies PostgreSQL connectivity and chunk count.
* ✅ `test_pgvector_similarity_search`: Confirms cosine similarity semantic retrieval.
* ✅ `test_neo4j_knowledge_graph`: Validates node counts, relationships, and Cypher queries.
* ✅ `test_financial_math`: Checks deterministic arithmetic (YoY, CAGR, margins, basis points).
* ✅ `test_hybrid_router`: Validates intent routing across Vector, Graph, and Hybrid modes.
* ✅ `test_document_sources`: Tests local PDF, SEC EDGAR, and web document ingestion sources.
* ✅ `test_semantic_chunker`: Tests structure chunking and header preservation.
* ✅ `test_qa_engine_execution`: Validates the end-to-end financial Q&A engine.
* ✅ `test_langgraph_workflow_all_modes`: Tests all 4 retrieval modes in LangGraph.
* ✅ `test_benchmark_evaluation_logic`: Confirms benchmark metric calculation logic.
* ✅ `test_fiscal_year_detection`: Verifies regex extraction of diverse fiscal year patterns.
* ✅ `test_docling_pdf_parser`: Confirms layout parsing and table extraction.
* ✅ `test_three_tier_financial_reporting`: Ensures Reported/Calculated/Interpreted separation.
* ✅ `test_idempotent_ingestion_duplicate_hash`: Tests SHA-256 deduplication.
* ✅ `test_postgres_connection_pool`: Validates `psycopg_pool.ConnectionPool` resilience.
* ✅ `test_chunk_full_provenance_retention`: Ensures chunk provenance metadata retention.
* ✅ `test_invalid_and_corrupted_document_handling`: Tests graceful error handling on corrupt PDFs.
* ✅ `test_llm_missing_api_key_fallback`: Tests automatic fallback to Ollama when API keys are absent.
* ✅ `test_vector_retrieval_semantic_accuracy`: Validates semantic relevance score thresholds.

---

## Project Structure

```
.
├── docker-compose.yml          # Production multi-service orchestration (Postgres, Neo4j, Streamlit)
├── Dockerfile                  # Production container definition for Streamlit application
├── run.sh / run.bat / run.ps1  # Automated 1-click startup scripts (macOS, Linux, Windows)
├── requirements.txt            # Production Python dependencies with pinned minimum versions
├── .env.example                # Documented configuration template (safe for version control)
│
├── database/                   # Database clients, schemas, and seeding
│   ├── postgres_client.py      # pgvector client with connection pooling & HNSW indexing
│   ├── neo4j_client.py         # Neo4j Cypher client with connection handling
│   ├── financial_knowledge_graph.py # Domain Knowledge Graph queries & multi-hop traversal
│   ├── schema.sql              # PostgreSQL DDL for vector_chunks & HNSW index
│   └── load_semantic_ground_truth.py # Knowledge Graph seeding engine
│
├── ingestion/                  # Document ingestion & parsing pipeline
│   ├── docling_parser.py       # Docling layout & table structure parser
│   ├── semantic_chunker.py     # Structure chunker retaining full chunk provenance
│   ├── pipeline.py             # Idempotent batch ingestion pipeline with error handling
│   └── sources/                # Document sources (Local PDF, SEC EDGAR, Web filings)
│
├── rag/                        # Retrieval-Augmented Generation & Reasoning
│   ├── langgraph_workflow.py   # LangGraph StateGraph orchestrator
│   ├── hybrid_router.py        # Intent, entity, metric, and fiscal year query router
│   ├── financial_math.py       # Deterministic Python math engine (YoY, CAGR, Margins, BPS)
│   ├── financial_qa_engine.py  # Three-tier financial reporting engine
│   ├── llm_provider.py         # Multi-provider LLM client with retries & graceful fallbacks
│   ├── embeddings_provider.py  # Vector embeddings provider (Nomic / BGE-M3 / OpenAI)
│   └── fiscal_year.py          # Fiscal year regex detection & normalization
│
├── ui/                         # User Interface
│   └── app.py                  # 8-tab enterprise Streamlit financial dashboard
│
├── evaluation/                 # Empirical benchmarking
│   ├── benchmark.py            # 4-way retrieval evaluation suite across 9 categories
│   ├── benchmark_results.json  # Comprehensive machine-readable evaluation results
│   └── benchmark_summary.md    # Markdown evaluation summary table
│
├── scripts/                    # Command-line management utilities
│   ├── init_db.py              # Initialize PostgreSQL schemas & Neo4j graph
│   ├── ingest.py               # Batch document ingestion CLI
│   ├── build_graph.py          # Build and enrich Knowledge Graph entities
│   ├── test_retrieval.py       # Standalone query & retrieval testing CLI
│   └── run_eval.py             # Benchmark evaluation runner CLI
│
└── tests/                      # Automated quality assurance
    └── test_end_to_end.py      # 19 comprehensive automated integration tests
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
