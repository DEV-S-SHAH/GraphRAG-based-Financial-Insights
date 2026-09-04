"""
Phase 15: Full pytest suite for the multi-year GraphRAG pipeline.

Covers:
  - Data validation (manifest, parsed, chunks, facts, entities, relationships)
  - Fiscal year detection
  - Graph retrieval (multi-year facts, entities, semantic paths)
  - Vector retrieval (FAISS multi-year index)
  - Hybrid retrieval (classification, fiscal-year scoping)
  - Context fusion (multi-year metadata)
  - 8 acceptance tests (full pipeline)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest():
    p = PROJECT_ROOT / "data" / "metadata" / "documents_manifest.json"
    return json.load(p.open("r", encoding="utf-8"))


@pytest.fixture(scope="module")
def facts_data():
    p = PROJECT_ROOT / "data" / "processed" / "financial_facts" / "LTIM_canonical_multiyear.json"
    return json.load(p.open("r", encoding="utf-8"))


@pytest.fixture(scope="module")
def entities_data():
    p = PROJECT_ROOT / "data" / "processed" / "entities" / "LTIM_entities_multiyear.json"
    return json.load(p.open("r", encoding="utf-8"))


@pytest.fixture(scope="module")
def relationships_data():
    p = PROJECT_ROOT / "data" / "processed" / "relationships" / "LTIM_relationships_multiyear.json"
    return json.load(p.open("r", encoding="utf-8"))


@pytest.fixture(scope="module")
def parsed_docs():
    from ingestion.financial_extractor_multi import DOCS, load_parsed
    return {fy: load_parsed(fy) for fy in DOCS}


@pytest.fixture(scope="module")
def chunk_files():
    """Load all multi-year chunk files."""
    chunks_dir = PROJECT_ROOT / "data" / "chunks"
    files = {}
    for fy in ["FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"]:
        doc_id = f"LTIM_{fy}_Annual_Report"
        p = chunks_dir / f"{doc_id}_chunks.json"
        files[fy] = json.load(p.open("r", encoding="utf-8"))
    return files


# ---------------------------------------------------------------------------
# DATA VALIDATION TESTS (Phase 8)
# ---------------------------------------------------------------------------

class TestDataValidation:
    """Validate all pipeline artifacts."""

    def test_manifest_has_4_documents(self, manifest):
        docs = manifest.get("documents", [])
        assert len(docs) == 4, f"Expected 4 docs, got {len(docs)}"

    def test_manifest_fiscal_years(self, manifest):
        years = {d["fiscal_year"] for d in manifest["documents"]}
        assert years == {"FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"}

    def test_manifest_files_exist(self, manifest):
        for doc in manifest["documents"]:
            p = PROJECT_ROOT / doc["path"]
            assert p.exists(), f"Missing: {p}"

    def test_parsed_files_exist(self):
        for fy in ["FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"]:
            _, fn = __import__("ingestion.financial_extractor_multi", fromlist=["DOCS"]).DOCS[fy]
            p = PROJECT_ROOT / "data" / "processed" / "pdf" / fn
            assert p.exists(), f"Missing parsed: {p}"

    def test_facts_count(self, facts_data):
        assert len(facts_data["facts"]) == 68

    def test_facts_all_years(self, facts_data):
        years = {f["period"]["label"] for f in facts_data["facts"]}
        assert years == {"FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"}

    def test_facts_metrics_per_year(self, facts_data):
        by_year = {}
        for f in facts_data["facts"]:
            by_year.setdefault(f["period"]["label"], set()).add(f["metric"])
        for fy, metrics in by_year.items():
            if fy == "FY2025-26":
                assert len(metrics) == 20, f"{fy} has {len(metrics)} metrics, expected 20"
            else:
                assert len(metrics) == 16, f"{fy} has {len(metrics)} metrics, expected 16"

    def test_entities_count(self, entities_data):
        assert len(entities_data["entities"]) >= 25

    def test_entities_multi_year(self, entities_data):
        multi = [e for e in entities_data["entities"] if len(e.get("fiscal_years", [])) >= 2]
        assert len(multi) >= 20, f"Expected >=20 multi-year entities, got {len(multi)}"

    def test_relationships_count(self, relationships_data):
        assert len(relationships_data["relationships"]) >= 10

    def test_relationships_controlled_vocab(self, relationships_data):
        controlled = {"SUPPORTS", "DRIVES", "ENABLES", "IMPROVES", "IMPACTS", "INCREASES",
                      "DECREASES", "FUNDS", "DEPENDS_ON", "RELATED_TO", "TARGETS", "MITIGATES",
                      "CREATES", "CONTRIBUTES_TO", "FOLLOWS"}
        for r in relationships_data["relationships"]:
            assert r["relation"].upper() in controlled, f"Unknown relation: {r['relation']}"

    def test_relationships_explicit_inferred(self, relationships_data):
        for r in relationships_data["relationships"]:
            assert r["relationship_type"] in ("explicit", "inferred")


# ---------------------------------------------------------------------------
# CHUNK PROVENANCE TESTS
# ---------------------------------------------------------------------------

class TestChunks:
    """Validate chunk provenance fields."""

    def test_chunk_counts(self, chunk_files):
        expected = {"FY2022-23": 1560, "FY2023-24": 1508, "FY2024-25": 1502, "FY2025-26": 1636}
        for fy, count in expected.items():
            actual = len(chunk_files[fy]["chunks"])
            assert actual == count, f"{fy}: expected {count} chunks, got {actual}"

    def test_chunks_provenance(self, chunk_files):
        for fy, data in chunk_files.items():
            for c in data["chunks"]:
                assert c.get("chunk_id"), "missing chunk_id"
                assert c.get("text"), "missing text"
                assert c.get("fiscal_year") == fy, f"fiscal_year mismatch: {c.get('fiscal_year')} != {fy}"
                assert isinstance(c.get("page"), int), "page not int"
                assert c.get("document_id"), "missing document_id"

    def test_chunks_prose_only(self, chunk_files):
        for fy, data in chunk_files.items():
            for c in data["chunks"]:
                assert c.get("chunk_type") == "prose", f"Non-prose chunk: {c.get('chunk_id')}"


# ---------------------------------------------------------------------------
# FISCAL YEAR DETECTION TESTS (Phase 10)
# ---------------------------------------------------------------------------

class TestFiscalYearDetection:
    """Test rag.fiscal_year module."""

    def test_detect_single_year(self):
        from rag.fiscal_year import detect_fiscal_years
        assert detect_fiscal_years("What was revenue in FY2024-25?") == ["FY2024-25"]

    def test_detect_fy23(self):
        from rag.fiscal_year import detect_fiscal_years
        assert detect_fiscal_years("EBITDA margin for FY23") == ["FY2022-23"]

    def test_detect_two_years(self):
        from rag.fiscal_year import detect_fiscal_years
        years = detect_fiscal_years("Compare revenue FY24 vs FY25")
        assert years == ["FY2023-24", "FY2024-25"]

    def test_detect_range(self):
        from rag.fiscal_year import detect_fiscal_years
        years = detect_fiscal_years("How did revenue change from FY2022-23 to FY2024-25?")
        assert years == ["FY2022-23", "FY2024-25"]

    def test_no_year(self):
        from rag.fiscal_year import detect_fiscal_years
        assert detect_fiscal_years("What was revenue?") == []

    def test_comparison_detection(self):
        from rag.fiscal_year import detect_comparison
        assert detect_comparison("Compare revenue across fiscal years") is True
        assert detect_comparison("Revenue growth year-on-year") is True
        assert detect_comparison("What was revenue?") is False

    def test_fiscal_year_context(self):
        from rag.fiscal_year import fiscal_year_context
        ctx = fiscal_year_context("Revenue in FY2024-25")
        assert ctx["detected_years"] == ["FY2024-25"]
        assert ctx["comparison"] is False
        assert "FY2024-25" in ctx["scope_note"]

    def test_comparison_context(self):
        from rag.fiscal_year import fiscal_year_context
        ctx = fiscal_year_context("Compare revenue across all fiscal years")
        assert ctx["comparison"] is True


# ---------------------------------------------------------------------------
# GRAPH RETRIEVAL TESTS
# ---------------------------------------------------------------------------

class TestGraphRetrieval:
    """Test GraphQueryEngine with multi-year graph."""

    @pytest.fixture(scope="class")
    def engine(self):
        from rag.query_engine import GraphQueryEngine
        e = GraphQueryEngine()
        yield e
        e.close()

    def test_revenue_returns_3_years(self, engine):
        facts = engine.get_financial_metric("revenue")
        periods = {f["period"] for f in facts}
        assert {"FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"} <= periods

    def test_ebitda_margin_returns_3_years(self, engine):
        facts = engine.get_financial_metric("ebitda_margin")
        periods = {f["period"] for f in facts}
        assert {"FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"} <= periods

    def test_detect_metric_revenue(self, engine):
        assert engine.detect_metric("What was revenue?") == "revenue"

    def test_detect_metric_ebitda_margin(self, engine):
        assert engine.detect_metric("EBITDA margin") == "ebitda_margin"

    def test_entity_detection(self, engine):
        entities = engine.detect_entities("How does cost optimization affect EBITDA margin?")
        assert "cost optimization" in entities

    def test_semantic_path_cost_opt_to_ebitda(self, engine):
        paths = engine.get_semantic_paths("cost optimization", "ebitda_margin")
        assert len(paths) >= 1
        rels = paths[0]["relationships"]
        assert "DRIVES" in rels or "SUPPORTS" in rels

    def test_entity_connections(self, engine):
        ent = engine.get_entity("AI")
        assert ent.get("entity") is not None


# ---------------------------------------------------------------------------
# VECTOR RETRIEVAL TESTS
# ---------------------------------------------------------------------------

class TestVectorRetrieval:
    """Test FAISS multi-year vector store."""

    @pytest.fixture(scope="class")
    def retriever(self):
        from rag.vector_retriever import VectorRetriever
        return VectorRetriever()

    def test_index_size(self, retriever):
        assert retriever.index.ntotal >= 4500

    def test_retrieve_has_fiscal_year(self, retriever):
        results = retriever.retrieve("AI productivity", top_k=3)
        for r in results:
            assert r.get("fiscal_year"), f"Missing fiscal_year in chunk {r.get('chunk_id')}"

    def test_retrieve_has_provenance(self, retriever):
        results = retriever.retrieve("EBITDA margin cost optimization", top_k=3)
        for r in results:
            assert r.get("page"), "Missing page"
            assert r.get("section"), "Missing section"


# ---------------------------------------------------------------------------
# HYBRID RETRIEVAL TESTS
# ---------------------------------------------------------------------------

class TestHybridRetrieval:
    """Test HybridRetriever with fiscal-year awareness."""

    @pytest.fixture(scope="class")
    def retriever(self):
        from rag.hybrid_retriever import HybridRetriever
        r = HybridRetriever()
        yield r
        r.close()

    def test_fiscal_year_in_result(self, retriever):
        result = retriever.retrieve("What was revenue in FY2024-25?")
        assert "fiscal_year_context" in result
        assert result["fiscal_year_context"]["detected_years"] == ["FY2024-25"]

    def test_single_year_filters_facts(self, retriever):
        result = retriever.retrieve("What was revenue in FY2024-25?")
        facts = result["graph_context"]["financial_facts"]
        for f in facts:
            assert f.get("period") == "FY2024-25", f"Unexpected period: {f.get('period')}"

    def test_comparison_keeps_all_years(self, retriever):
        result = retriever.retrieve("Compare revenue across all fiscal years")
        periods = {f["period"] for f in result["graph_context"]["financial_facts"]}
        assert len(periods) >= 2, f"Expected multiple periods, got {periods}"

    def test_classification_financial(self, retriever):
        result = retriever.retrieve("What was revenue?")
        assert result["question_type"] == "financial"

    def test_classification_graph(self, retriever):
        result = retriever.retrieve("How does cost optimization affect EBITDA margin?")
        assert result["question_type"] == "graph"

    def test_classification_hybrid(self, retriever):
        result = retriever.retrieve("What risks could affect revenue?")
        assert result["question_type"] == "hybrid"

    def test_semantic_path_found(self, retriever):
        result = retriever.retrieve("How does cost optimization affect EBITDA margin?")
        sems = result["graph_context"]["semantic_paths"]
        assert len(sems) >= 1, "Expected at least one semantic path"

    def test_vector_context_populated(self, retriever):
        result = retriever.retrieve("What strategic priorities did LTIMindtree discuss regarding AI?")
        assert len(result["vector_context"]) >= 1


# ---------------------------------------------------------------------------
# CONTEXT FUSION TESTS
# ---------------------------------------------------------------------------

class TestContextFusion:
    """Test context fusion includes fiscal-year context."""

    def test_fiscal_year_section(self):
        from rag.context_fusion import build_context
        mock_result = {
            "question": "Revenue in FY2024-25",
            "graph_context": {
                "financial_facts": [{"metric": "revenue", "value": 380081, "unit": "INR million", "period": "FY2024-25", "page": 111, "section": "Financial Results"}],
                "entities": [],
                "paths": [],
                "semantic_paths": [],
            },
            "vector_context": [
                {"text": "revenue grew to 380,081 million", "page": 111, "score": 0.8, "fiscal_year": "FY2024-25", "section": "Financial Results", "document_id": "LTIM_FY2024-25_Annual_Report"}
            ],
            "fiscal_year_context": {
                "detected_years": ["FY2024-25"],
                "comparison": False,
                "scope_note": "The question mentions fiscal year(s): FY2024-25.",
                "available_fiscal_years": ["FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"],
            },
        }
        ctx = build_context(mock_result)
        assert "FISCAL-YEAR CONTEXT" in ctx
        assert "FY2024-25" in ctx
        assert "LTIMindtree" in ctx or "LTIM" in ctx


# ---------------------------------------------------------------------------
# ACCEPTANCE TESTS (8 questions, full pipeline)
# ---------------------------------------------------------------------------

class TestAcceptance:
    """
    8 acceptance questions covering the full pipeline:
    question → retrieval → answer generation.
    Requires Ollama running locally with qwen2.5:3b.
    """

    @pytest.fixture(scope="class")
    def generator(self):
        from rag.answer_generator import AnswerGenerator
        return AnswerGenerator()

    @pytest.fixture(scope="class")
    def retriever(self):
        from rag.hybrid_retriever import HybridRetriever
        r = HybridRetriever()
        yield r
        r.close()

    def _run_pipeline(self, question, generator, retriever):
        from rag.context_fusion import build_context
        result = retriever.retrieve(question, top_k=5)
        context = build_context(result)
        answer = generator.generate(context)
        return answer, result

    def test_acceptance_revenue_fy24(self, generator, retriever):
        answer, result = self._run_pipeline(
            "What was LTIMindtree's revenue in FY2024-25?",
            generator, retriever,
        )
        assert "380" in answer or "380,081" in answer, f"Expected revenue 380,081 in answer: {answer[:200]}"
        assert "FY2024-25" in answer or "2024-25" in answer

    def test_acceptance_ebitda_margin_comparison(self, generator, retriever):
        answer, result = self._run_pipeline(
            "Compare EBITDA margin across the three fiscal years.",
            generator, retriever,
        )
        assert "18" in answer, f"Expected EBITDA margin values in answer: {answer[:300]}"

    def test_acceptance_cost_opt_to_ebitda(self, generator, retriever):
        answer, result = self._run_pipeline(
            "How does cost optimization affect EBITDA margin?",
            generator, retriever,
        )
        assert "cost optimization" in answer.lower()
        assert "EBITDA" in answer or "ebitda" in answer.lower()
        # Should mention relationship or acknowledge evidence limitation
        has_graph_ref = any(
            term in answer.lower()
            for term in ["indirect", "connected", "graph", "relationship", "drives"]
        )
        has_margin_value = any(
            v in answer for v in ["18.4", "18.0", "17.1", "17.9"]
        )
        assert has_graph_ref or has_margin_value

    def test_acceptance_ai_strategic(self, generator, retriever):
        answer, result = self._run_pipeline(
            "What strategic priorities did LTIMindtree discuss regarding AI?",
            generator, retriever,
        )
        assert "AI" in answer or "ai" in answer.lower()

    def test_acceptance_risks(self, generator, retriever):
        answer, result = self._run_pipeline(
            "What risks could affect LTIMindtree's revenue and profitability?",
            generator, retriever,
        )
        assert "risk" in answer.lower()

    def test_acceptance_genai_productivity(self, generator, retriever):
        answer, result = self._run_pipeline(
            "What did management say about GenAI productivity in FY2024-25?",
            generator, retriever,
        )
        answer_lower = answer.lower()
        assert ("genai" in answer_lower
                or "generative" in answer_lower
                or "productivity" in answer_lower
                or "ai" in answer_lower), f"Expected GenAI/AI related content in answer: {answer[:300]}"

    def test_acceptance_unsupported_share_price(self, generator, retriever):
        answer, result = self._run_pipeline(
            "What was LTIMindtree's share price on January 1, 2024?",
            generator, retriever,
        )
        not_found = ("not enough evidence" in answer.lower()
                     or "not available" in answer.lower()
                     or "not in" in answer.lower()
                     or "not provided" in answer.lower()
                     or "does not contain" in answer.lower()
                     or "no information" in answer.lower())
        assert not_found, f"Expected unsupported indicator in answer: {answer[:300]}"

    def test_acceptance_revenue_trend(self, generator, retriever):
        answer, result = self._run_pipeline(
            "How did revenue change from FY2022-23 to FY2024-25?",
            generator, retriever,
        )
        assert "331" in answer or "355" in answer or "380" in answer, \
            f"Expected multi-year revenue values in answer: {answer[:300]}"
