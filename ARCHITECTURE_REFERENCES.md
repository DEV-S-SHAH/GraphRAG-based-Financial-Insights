# Financial GraphRAG: Architecture & Reference Foundations

This document details the architectural decisions, design patterns, and open-source references adopted in this Financial GraphRAG system, fulfilling Section 7 of the specification.

---

## 1. Technical References & Adopted Patterns

### A. Neo4j GraphRAG & HybridCypherRetriever
- **Reference**: Neo4j GraphRAG Python (`neo4j-graphrag`), Neo4j Cypher templates.
- **Pattern Adopted**:
  - **Structured Graph Traversal with Cypher**: Rather than relying only on text-to-Cypher (which can hallucinate Cypher syntax or schema relations), we employ deterministic Cypher query templates parameterized by extracted entities, metrics, and fiscal periods.
  - **Multi-Hop Path Tracing (`*1..3`)**: We traverse entity paths (e.g., `Program -[:DRIVES]-> Capability -[:IMPROVES]-> FinancialMetric`) to answer indirect causal and relational questions.
  - **Strict Provenance on Graph Facts**: Every `FinancialValue` node is connected to a `ReportingPeriod`, `SourceDocument`, and `SourcePage`, ensuring every graph fact is attributable.

### B. Docling (IBM Granite / Docling Core)
- **Reference**: IBM Docling (`docling` 2.x, `docling-core`), `DocumentConverter` and `HybridChunker`.
- **Pattern Adopted**:
  - **Structure-Aware Document Hierarchy**: Annual reports are deeply structured into sections (Front Matter, MD&A, Statutory Reports, Financial Statements). Docling's reading-order block parser extracts headings, paragraphs, and tables.
  - **Table Preservation**: Financial tables are extracted as cohesive units (rendered into markdown pipes) so columns and rows are never split across arbitrary token boundaries.
  - **Provenance Preservation**: Chunks retain `document_id`, `fiscal_year`, `page_number`, `section_title`, and `chunk_type` ('table' vs 'prose').

### C. PostgreSQL + pgvector
- **Reference**: pgvector (`pgvector/pgvector:pg16`), LangChain / LlamaIndex Postgres vector store implementations.
- **Pattern Adopted**:
  - **Cosine Similarity (`<=>`)**: Chunks and embeddings are stored alongside rich metadata (`document_id`, `fiscal_year`, `section`, `page`, `chunk_type`).
  - **Metadata-Filtered Vector Search**: Hybrid queries can constrain semantic search to specific fiscal years or sections (e.g., filtering to "Financial Statements" or "Risk Management").
  - **HNSW / Exact Search**: Enables scalable nearest neighbor retrieval with vector similarity scores normalized to `[0, 1]`.

### D. Financial & SEC GraphRAG Projects
- **Reference**: SEC EDGAR Filings API, XBRL / 10-K extraction frameworks, Financial Ontologies (FIBO principles).
- **Pattern Adopted**:
  - **Pluggable `DocumentSource` Architecture**: Separate sources for local PDFs, SEC EDGAR public REST API, and investor relations websites.
  - **Deterministic Financial Math Layer**: The LLM is never permitted to calculate percentages, YoY growth, or margin changes in its head. All math (YoY growth, EBITDA margin, PAT margin, CAGR) is computed programmatically in Python before context generation.
  - **Three-Tier Separation**:
    1. **Reported Data**: Verbatim from annual reports & 10-K filings.
    2. **Calculated Metrics**: Deterministic mathematical formulas.
    3. **LLM Interpretation**: Narrative synthesis explaining the drivers and strategic context.

### E. LLM & Embeddings Abstraction
- **Pattern Adopted**:
  - Unified `LLMProvider` abstract base class supporting **Local Ollama** (`qwen2.5:3b`, fitting in < 4GB VRAM/RAM), **OpenAI** (`gpt-4o-mini`, `gpt-4o`), and **Anthropic** (`claude-3-5-haiku`, `claude-3-5-sonnet`).
  - Seamless fallback: If no API keys are provided in `.env`, the system defaults cleanly to local Ollama and local embeddings without errors.

---

## 2. Summary of Architecture Pipeline

```
[Local PDF / SEC EDGAR / Web API]
              │
              ▼
    Docling Document Parser
  (Headings, Tables, Metadata)
              │
              ▼
   Semantic Structure Chunker
(Unique IDs, Provenance, Page No)
              │
       ┌──────┴──────────────────────────┐
       ▼                                 ▼
PostgreSQL + pgvector           Entity & Fact Extractor
(Chunks, Embeddings, Meta)               │
                                         ▼
                               Neo4j Knowledge Graph
                             (Company, Metrics, Values,
                              Segments, Risks, Strategies)
                                         │
                                         ▼
                             Hybrid Retrieval Router
                     ┌───────────────────┼───────────────────┐
                     ▼                   ▼                   ▼
             pgvector Search      Neo4j Traversal       Cypher Query
                     └───────────────────┬───────────────────┘
                                         ▼
                            Financial Math Engine
                        (YoY Growth, Margins, Ratios)
                                         │
                                         ▼
                              Context Fusion Layer
                       (Reported + Calculated + Provenance)
                                         │
                                         ▼
                                  LLM Provider
                         (Ollama / OpenAI / Anthropic)
                                         │
                                         ▼
                           Streamlit Research Dashboard
                (Q&A, Analysis, Graph Viz, Debugger, Evaluation)
```
