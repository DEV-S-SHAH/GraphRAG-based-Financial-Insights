# Financial GraphRAG: Corporate Financial Intelligence

An AI-powered financial intelligence platform for analyzing corporate annual reports, SEC Form 10-K filings, and investor presentations with verified accuracy and zero mathematical hallucinations.

---

## What It Gives You

* **Audited Financial Q&A**: Get exact numbers, multi-year comparisons, and page-cited answers directly from corporate filings.
* **Multi-Year Trends & Ratios**: Automatic calculation of YoY growth, CAGR, EBITDA margins, and basis-point changes with zero math errors.
* **Interactive Knowledge Graph**: Visualize connections between company metrics, business segments, strategic initiatives, and risk factors.
* **4-Way Retrieval Comparison**: Compare answers side-by-side across Standard LLM, Vector RAG, GraphRAG, and Hybrid RAG.
* **Full Citation Provenance**: Every metric links back to its verified document name, fiscal year, and source page number.

---

## Quickstart (Run in 1 Command)

### Prerequisites
* [Docker Desktop](https://www.docker.com/) (installed and running)
* Python 3.10 - 3.12 (if running locally without Docker)

### Option 1: Docker (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/DEV-S-SHAH/GraphRAG-based-Financial-Insights.git
cd GraphRAG-based-Financial-Insights

# 2. Build and start services
docker compose up --build
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

### Option 2: 1-Click Startup Script

* **macOS / Linux**:
  ```bash
  chmod +x run.sh && ./run.sh
  ```
* **Windows (PowerShell)**:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\run.ps1
  ```
* **Windows (Command Prompt / Double-click)**:
  ```cmd
  run.bat
  ```

---

## Application Access Points

| Service | URL | Credentials |
|---|---|---|
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | None |
| **Neo4j Graph Browser** | [http://localhost:7474](http://localhost:7474) | `neo4j` / `password123` |
| **PostgreSQL Database** | `localhost:5432` | `financial_user` / `password123` |

---

## How to Use

### 1. Web Dashboard (Streamlit)
Navigate to **[http://localhost:8501](http://localhost:8501)** to use the 8 built-in tools:
1. **⚖️ 4-Way Comparison**: Compare responses from No RAG, Vector RAG, GraphRAG, and Hybrid.
2. **💬 Financial Q&A**: Ask any question and view reported facts, calculated math, and strategic commentary.
3. **📈 Financial Trends**: Explore interactive multi-year charts for revenue, margins, and profit.
4. **🕸️ Knowledge Graph**: Interactively explore entities, segments, and risks in network views.
5. **🔍 Vector Search**: Search raw text passages and inspect similarity scores.
6. **📑 Document Explorer**: Browse indexed chunks, documents, and database records.
7. **📥 Ingest Data**: Upload new annual report PDFs or pull SEC 10-K filings by ticker.
8. **📊 Benchmark**: View empirical accuracy, latency, and hallucination scores.

### 2. Command-Line Tools

```bash
# Ask a financial question in terminal
python scripts/test_retrieval.py "What was the revenue and EBITDA margin in FY2023-24?"

# Ingest new PDF filings from a folder
python scripts/ingest.py --source local --pdf-dir data/raw/

# Download and ingest SEC 10-K filings by ticker
python scripts/ingest.py --source sec --ticker AAPL

# Run automated tests
pytest tests/test_end_to_end.py -v

# Run the 4-way benchmark suite
python scripts/run_eval.py --max-questions 3
```

---

## Configuration (`.env`)

Copy `.env.example` to `.env` to configure your environment:

```bash
cp .env.example .env
```

Default settings run 100% locally with free, offline Ollama models:

```ini
# Database Settings
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=financial_db
POSTGRES_USER=financial_user
POSTGRES_PASSWORD=password123

# Neo4j Settings
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# AI Models (Default: Free local Ollama)
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768

# Optional: Cloud APIs (leave empty to use local Ollama)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

---

## License

This project is licensed under the MIT License.
