"""
End-to-End Test Suite for Financial GraphRAG Application.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.postgres_client import PostgresVectorClient
from database.financial_knowledge_graph import FinancialKnowledgeGraph
from rag.financial_math import (
    FinancialMathEngine,
    calculate_yoy_growth,
    calculate_margin,
    calculate_cagr,
)
from rag.hybrid_router import HybridRouter, RetrievalStrategy, HybridRetriever
from rag.financial_qa_engine import FinancialQAEngine
from ingestion.sources.local_pdf import LocalPDFSource
from ingestion.sources.sec_edgar import SECEDGARSource
from ingestion.semantic_chunker import SemanticChunker, SemanticChunk


def test_postgres_connection_and_stats():
    pg = PostgresVectorClient()
    stats = pg.get_stats()
    assert "vector_chunks" in stats
    assert stats["vector_chunks"] > 0
    assert stats["documents"] >= 4


def test_pgvector_similarity_search():
    pg = PostgresVectorClient()
    # Query with dummy 768-dim vector
    dummy_vec = [0.01] * 768
    results = pg.search_similarity(dummy_vec, top_k=3, fiscal_year="FY2023-24")
    assert len(results) > 0
    assert results[0]["fiscal_year"] == "FY2023-24"
    assert "score" in results[0]


def test_neo4j_knowledge_graph():
    kg = FinancialKnowledgeGraph()
    summary = kg.get_graph_summary()
    assert summary["total_nodes"] > 100
    assert summary["total_relationships"] > 100

    facts = kg.get_financial_facts_for_metric("revenue")
    assert len(facts) > 0
    assert any(f.get("period") == "FY2023-24" for f in facts)
    kg.close()


def test_financial_math():
    # 1. YoY growth: 355170 vs 331830 -> +7.03%
    yoy = calculate_yoy_growth(355170, 331830)
    assert yoy == 7.03

    # 2. Margin: 63874 / 355170 -> 17.98%
    margin = calculate_margin(63874, 355170)
    assert margin == 17.98

    # 3. CAGR: 100 to 200 over 3 periods -> 25.99%
    cagr = calculate_cagr(100, 200, 3)
    assert cagr == 25.99


def test_hybrid_router():
    # Pure KPI query
    d1 = HybridRouter.route_query("What was LTIMindtree revenue in FY2023-24?")
    assert d1.detected_metric == "revenue"

    # Relational query
    d2 = HybridRouter.route_query("How does cost optimization affect EBITDA margin?")
    assert d2.strategy in [RetrievalStrategy.NEO4J_GRAPH, RetrievalStrategy.HYBRID_FUSION]

    # Narrative query
    d3 = HybridRouter.route_query("What risks does management highlight in the annual report?")
    assert d3.strategy in [RetrievalStrategy.VECTOR_SEARCH, RetrievalStrategy.METADATA_FILTERED, RetrievalStrategy.HYBRID_FUSION]


def test_document_sources():
    raw_dir = PROJECT_ROOT / "data" / "raw" / "annual_reports"
    local_source = LocalPDFSource(directory_path=raw_dir)
    docs = local_source.discover_documents()
    assert len(docs) >= 4

    sec_source = SECEDGARSource(ticker="AAPL", max_filings=1)
    assert sec_source.ticker == "AAPL"


def test_semantic_chunker():
    chunker = SemanticChunker(target_chars=500)
    mock_doc = {
        "document_id": "TEST_DOC_001",
        "company": "Test Corp",
        "ticker": "TST",
        "fiscal_year": "FY2024",
        "pages": [
            {
                "page": 1,
                "section": "Overview",
                "blocks": [{"text": "Paragraph 1 describing performance."}, {"text": "Paragraph 2 details."}],
                "tables": ["| Year | Revenue |\n|---|---|\n| 2024 | $100M |"],
            }
        ],
    }
    chunks = chunker.chunk_document(mock_doc)
    assert len(chunks) == 2
    types = [c.chunk_type for c in chunks]
    assert "table" in types
    assert "prose" in types


def test_qa_engine_execution():
    qa = FinancialQAEngine()
    resp = qa.answer_question("What was revenue in FY2023-24?")
    assert resp.answer is not None
    assert len(resp.citations) > 0
    assert resp.llm_response.latency_seconds > 0


def test_langgraph_workflow_all_modes():
    from rag.langgraph_workflow import FinancialGraphRAGWorkflow

    wf = FinancialGraphRAGWorkflow()
    # Test individual modes without long generation
    state_no_rag = wf.run("What was revenue in FY2023-24?", mode="no_rag")
    assert state_no_rag["mode"] == "no_rag"
    assert len(state_no_rag["vector_chunks"]) == 0
    assert len(state_no_rag["graph_facts"]) == 0

    state_vec = wf.run("What was revenue in FY2023-24?", mode="vector_rag")
    assert len(state_vec["vector_chunks"]) > 0

    state_graph = wf.run("What was revenue in FY2023-24?", mode="graph_rag")
    assert len(state_graph["graph_facts"]) > 0

    state_hyb = wf.run("What was revenue in FY2023-24?", mode="hybrid")
    assert len(state_hyb["vector_chunks"]) > 0
    assert len(state_hyb["graph_facts"]) > 0


def test_benchmark_evaluation_logic():
    from evaluation.benchmark import FinancialBenchmark

    bm = FinancialBenchmark()
    sample_q = {
        "id": "test_q",
        "category": "Revenue",
        "question": "What was revenue?",
        "ground_truth_numbers": ["355,170"],
        "ground_truth_entities": ["revenue"],
        "expected_year": "FY2023-24",
        "expected_page": 114,
    }
    # Evaluated on accurate grounded response
    res = bm.evaluate_response(
        question_meta=sample_q,
        mode="hybrid",
        answer_text="LTIMindtree revenue was 355,170 INR million in FY2023-24.",
        context_text="The reported revenue was 355,170 INR million.",
        citations=[{"fiscal_year": "FY2023-24", "page": 114}],
    )
    assert res["answer_accuracy"] == 1.0
    assert res["faithfulness"] == 1.0
    assert res["citation_accuracy"] == 1.0


def test_fiscal_year_detection():
    from rag.fiscal_year import detect_fiscal_years

    years = detect_fiscal_years("Compare revenue growth between FY22-23 and FY2023-24.")
    assert "FY2022-23" in years
    assert "FY2023-24" in years


def test_docling_pdf_parser():
    from ingestion.docling_parser import DoclingParser

    parser = DoclingParser(use_docling_converter=False)
    pdf_path = PROJECT_ROOT / "data" / "raw" / "annual_reports" / "LTM_FY2023-24_Annual_Report.pdf"
    assert pdf_path.exists()

    parsed = parser.parse_pdf(
        pdf_path=pdf_path,
        document_id="TEST_DOC_LTIM",
        fiscal_year="FY2023-24",
        max_pages=3,
    )
    assert parsed.num_pages == 3
    assert len(parsed.blocks) > 0
    assert len(parsed.pages) == 3


def test_three_tier_financial_reporting():
    from database.financial_knowledge_graph import FinancialKnowledgeGraph

    kg = FinancialKnowledgeGraph()
    rev_facts = kg.get_financial_facts_for_metric("revenue")
    ebitda_facts = kg.get_financial_facts_for_metric("ebitda")
    pat_facts = kg.get_financial_facts_for_metric("profit_after_tax")
    kg.close()

    intel = FinancialMathEngine.build_financial_intelligence_report(rev_facts, ebitda_facts, pat_facts)
    # Verify tiers:
    # 1. Reported
    assert len(intel["revenue_analysis"]["yearly_data"]) > 0
    # 2. Calculated
    assert "ebitda_margins" in intel
    assert "pat_margins" in intel
    assert intel["revenue_analysis"]["cagr"] is not None
    # 3. Formatted table representation
    table_md = FinancialMathEngine.format_as_markdown_table(intel)
    assert "Revenue (INR M)" in table_md
    assert "EBITDA Margin" in table_md



def test_idempotent_ingestion_duplicate_hash():
    from ingestion.sources.base import DocumentSource

    test_file = PROJECT_ROOT / "data" / "raw" / "annual_reports" / "LTM_FY2023-24_Annual_Report.pdf"
    hash1 = DocumentSource.compute_sha256(test_file)
    hash2 = DocumentSource.compute_sha256(test_file)
    assert hash1 == hash2
    assert len(hash1) == 64


