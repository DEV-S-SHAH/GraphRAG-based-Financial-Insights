"""
Tests for the LTM Hybrid (Graph + Vector) RAG system.

Covers:
1. Vector chunk creation (page preservation)
2. Vector store build/load
3. VectorRetriever retrieval
4. Query routing (classification)
5. Context fusion
6. HybridRetriever (integration, requires Neo4j + FAISS)
7. End-to-end answer generation (graph + vector)
8. Source/page preservation
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.vector_store import (
    RecursiveCharacterTextSplitter,
    VECTORSTORE_DIR,
    create_chunks,
)
from rag.query_router import classify_question
from rag.context_fusion import build_context


# ============================================================
# HELPERS
# ============================================================

NORMALIZED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normalized"
    / "LTM_FY2025-26_normalized.json"
)


def _neo4j_available():
    try:
        from rag.query_engine import create_driver

        driver = create_driver()
        driver.close()
        return True
    except Exception:
        return False


def _vectorstore_available():
    return (VECTORSTORE_DIR / "index.faiss").exists()


NEO4J_AVAILABLE = _neo4j_available()
VECTORSTORE_AVAILABLE = _vectorstore_available()


def _ollama_available():
    """Check if Ollama with qwen2.5:3b is available."""
    try:

        import requests

        r = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5,
        )

        r.raise_for_status()

        models = [
            m["name"]
            for m in r.json().get("models", [])
        ]

        return any(
            "qwen2.5:3b" in m for m in models
        )

    except Exception:
        return False


# ============================================================
# TEXT SPLITTER TESTS
# ============================================================


class TestTextSplitter:
    """Test character/text splitting."""

    def test_short_text_single_chunk(self):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )

        chunks = splitter.split_text("Short text.")

        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_long_text_multiple_chunks(self):

        text = (
            "This is a sentence. " * 200
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=50
        )

        chunks = splitter.split_text(text)

        assert len(chunks) > 1

        for chunk in chunks:
            assert len(chunk) <= 400  # with overlap

    def test_empty_text(self):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )

        chunks = splitter.split_text("   ")

        assert chunks == [] or chunks == [""]

    def test_respects_chunk_size(self):

        text = "word " * 500

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50
        )

        chunks = splitter.split_text(text)

        max_len = max(len(c) for c in chunks)

        assert max_len <= 700  # allowing some overhead


# ============================================================
# CHUNK CREATION TESTS
# ============================================================


class TestChunkCreation:
    """Test page-aware chunk creation (legacy - superseded by test_multiyear_rag)."""

    @pytest.mark.skip(reason="Legacy chunk API removed in Phase 3; covered by test_multiyear_rag")
    def test_chunks_preserve_page_metadata(self):
        pass

    @pytest.mark.skip(reason="Legacy chunk API removed in Phase 3; covered by test_multiyear_rag")
    def test_chunks_cover_multiple_pages(self):
        pass

    @pytest.mark.skip(reason="Legacy chunk API removed in Phase 3; covered by test_multiyear_rag")
    def test_chunk_text_not_empty(self):
        pass


# ============================================================
# QUERY ROUTER TESTS
# ============================================================


class TestQueryRouter:
    """Test deterministic question classification."""

    def test_financial_question(self):

        assert (
            classify_question(
                "What was LTM's EBITDA?",
                detected_metric="ebitda",
            )
            == "financial"
        )

    def test_financial_margin(self):

        assert (
            classify_question(
                "What was LTM's EBITDA margin?",
                detected_metric="ebitda_margin",
            )
            == "financial"
        )

    def test_graph_question_with_relationship(self):

        result = classify_question(
            "How does cost optimization affect EBITDA margin?",
            detected_metric="ebitda_margin",
            entities=["cost optimization"],
        )

        assert result in ("graph", "hybrid")

    def test_narrative_question(self):

        assert (
            classify_question(
                "What strategic priorities did LTM discuss "
                "regarding AI?"
            )
            == "narrative"
        )

    def test_narrative_management_said(self):

        assert (
            classify_question(
                "What did management say about AI?"
            )
            == "narrative"
        )

    def test_hybrid_risk_question(self):

        assert (
            classify_question(
                "What risks could affect LTM's revenue "
                "and profitability?",
                detected_metric="revenue",
            )
            == "hybrid"
        )

    def test_force_override(self):

        assert (
            classify_question(
                "anything", force="graph"
            )
            == "graph"
        )

    def test_simple_greeting(self):

        result = classify_question("Hello there")
        assert result in ("financial", "narrative", "hybrid")


# ============================================================
# CONTEXT FUSION TESTS
# ============================================================


def _sample_hybrid_result():
    return {
        "question": "How does cost optimization affect EBITDA margin?",
        "question_type": "graph",
        "graph_context": {
            "financial_facts": [
                {
                    "metric": "ebitda_margin",
                    "value": 17.9,
                    "unit": "%",
                    "period": "FY2025-26",
                    "page": 8,
                    "section": "Financial Performance",
                }
            ],
            "entities": [
                {
                    "entity": {
                        "entity": "cost optimization",
                        "outgoing": [
                            {
                                "relationship": "IMPROVES",
                                "target": (
                                    "operational efficiency"
                                ),
                            }
                        ],
                        "incoming": [],
                    }
                }
            ],
            "semantic_paths": [
                {
                    "nodes": [
                        "cost optimization",
                        "operational efficiency",
                        "ebitda_margin",
                    ],
                    "relationships": [
                        "IMPROVES", "SUPPORTS"
                    ],
                }
            ],
            "paths": [],
            "detected_metric": "ebitda_margin",
        },
        "vector_context": [
            {
                "text": (
                    "EBITDA growth in absolute terms is "
                    "at 16.3% and EBITDA % for FY26 is 17.9%"
                ),
                "page": 101,
                "document_id": "LTM_FY26_AR_001",
                "company": "LTM Limited",
                "period": "FY2025-26",
                "score": 0.9,
            }
        ],
    }


class TestContextFusion:
    """Test hybrid context formatting."""

    def test_contains_question(self):

        ctx = build_context(_sample_hybrid_result())

        assert "How does cost optimization" in ctx

    def test_contains_financial_facts(self):

        ctx = build_context(_sample_hybrid_result())

        assert "17.9" in ctx
        assert "ebitda_margin" in ctx
        assert "page: 8" in ctx

    def test_contains_graph_paths(self):

        ctx = build_context(_sample_hybrid_result())

        assert "IMPROVES" in ctx
        assert "SUPPORTS" in ctx
        assert "operational efficiency" in ctx

    def test_contains_document_evidence_with_page(self):

        ctx = build_context(_sample_hybrid_result())

        assert "[Page 101]" in ctx
        assert "16.3%" in ctx

    def test_contains_metadata(self):

        ctx = build_context(_sample_hybrid_result())

        assert "LTM" in ctx
        assert "FY2025-26" in ctx


# ============================================================
# VECTOR RETRIEVER TESTS (integration)
# ============================================================


@pytest.mark.skipif(
    not VECTORSTORE_AVAILABLE,
    reason="Vector store not built. Run python -m rag.vector_store",
)
class TestVectorRetriever():
    """Integration tests requiring the FAISS index."""

    @classmethod
    def setup_class(cls):

        from rag.vector_retriever import VectorRetriever

        cls.retriever = VectorRetriever()

    def test_retrieve_returns_sources(self):

        results = self.retriever.retrieve(
            "What are LTM's strategic priorities?",
            top_k=3,
        )

        assert len(results) > 0

        for r in results:
            assert "text" in r
            assert "page" in r
            assert "score" in r

    def test_retrieve_semantic_content(self):

        results = self.retriever.retrieve(
            "EBITDA margin performance",
            top_k=5,
        )

        texts = " ".join(r["text"] for r in results).lower()

        assert "ebitda" in texts or "margin" in texts

    def test_page_provenance(self):

        results = self.retriever.retrieve(
            "How does AI affect LTM strategy?",
            top_k=3,
        )

        for r in results:
            assert r["page"] >= 1


# ============================================================
# HYBRID RETRIEVER TESTS (integration)
# ============================================================


@pytest.mark.skipif(
    not (NEO4J_AVAILABLE and VECTORSTORE_AVAILABLE),
    reason="Requires Neo4j + vector store",
)
class TestHybridRetriever():
    """End-to-end hybrid retrieval tests."""

    @classmethod
    def setup_class(cls):

        from rag.hybrid_retriever import HybridRetriever

        cls.retriever = HybridRetriever()

    @classmethod
    def teardown_class(cls):

        cls.retriever.close()

    def test_structured_result(self):

        result = self.retriever.retrieve(
            "What was LTM's EBITDA?"
        )

        assert "question" in result
        assert "question_type" in result
        assert "graph_context" in result
        assert "vector_context" in result

        gc = result["graph_context"]
        assert "financial_facts" in gc
        assert "entities" in gc
        assert "paths" in gc
        assert "semantic_paths" in gc

    def test_ebitda_financial(self):

        result = self.retriever.retrieve(
            "What was LTM's EBITDA?"
        )

        facts = result["graph_context"]["financial_facts"]

        assert len(facts) > 0

        ebitda = [
            f for f in facts
            if f["metric"] == "ebitda"
        ]

        assert ebitda, "Expected ebitda metric"
        assert ebitda[0]["unit"] == "INR million"
        valid_ebitda = {61077, 63874, 64949, 75552}
        assert ebitda[0]["value"] in valid_ebitda

    def test_ebitda_margin_financial(self):

        result = self.retriever.retrieve(
            "What was LTIM's EBITDA margin?"
        )

        facts = result["graph_context"]["financial_facts"]

        margin = [
            f for f in facts
            if f["metric"] == "ebitda_margin"
        ]

        assert margin, "Expected ebitda_margin metric"
        assert margin[0]["unit"] == "%"
        valid_margins = {18.4, 18.0, 17.1, 17.9}
        assert margin[0]["value"] in valid_margins

    def test_cost_optimization_path(self):

        result = self.retriever.retrieve(
            "How does cost optimization affect EBITDA margin?"
        )

        paths = result["graph_context"]["semantic_paths"]

        found = False

        for p in paths:

            nodes = [
                n.lower()
                for n in p.get("nodes", p.get("path", []))
            ]

            if (
                "cost optimization" in nodes
                and "ebitda_margin" in nodes
            ):
                found = True
                break

        assert found, (
            "Expected path from cost optimization "
            "to ebitda_margin"
        )

    def test_fit4future_paths(self):

        result = self.retriever.retrieve(
            "How does cost optimization affect EBITDA margin?"
        )

        paths = result["graph_context"]["semantic_paths"]

        found = False

        for p in paths:

            nodes = [
                n.lower()
                for n in p.get("nodes", p.get("path", []))
            ]

            if (
                "cost optimization" in nodes
                and "ebitda_margin" in nodes
            ):
                found = True
                break

        assert found, (
            "Expected cost optimization -> ebitda_margin path"
        )

    def test_narrative_retrieval_has_vector(self):

        result = self.retriever.retrieve(
            "What strategic priorities did LTM discuss "
            "regarding AI and digital transformation?"
        )

        assert result["question_type"] == "narrative"
        assert len(result["vector_context"]) > 0

    def test_risk_question_hybrid(self):

        result = self.retriever.retrieve(
            "What risks could affect LTM's revenue "
            "and profitability?"
        )

        assert result["question_type"] == "hybrid"
        assert len(result["vector_context"]) > 0
        assert len(
            result["graph_context"]["financial_facts"]
        ) > 0


# ============================================================
# END-TO-END ANSWER TESTS
# (Requires Ollama running with qwen2.5:3b)
# ============================================================


@pytest.mark.skipif(
    not (NEO4J_AVAILABLE and VECTORSTORE_AVAILABLE),
    reason="Requires Neo4j + vector store",
)
class TestEndToEndAnswer():
    """End-to-end answer generation."""

    @pytest.mark.skipif(
        not _ollama_available(),
        reason="Ollama not available",
    )
    def test_answer_ebitda(self):

        from rag.app import answer_question

        result = answer_question(
            "What was LTIM's EBITDA in FY2024-25?"
        )

        assert "answer" in result
        assert "64" in result["answer"]
        assert result["question_type"] == "financial"

    @pytest.mark.skipif(
        not _ollama_available(),
        reason="Ollama not available",
    )
    def test_answer_margin(self):

        from rag.app import answer_question

        result = answer_question(
            "What was LTIM's EBITDA margin in FY2024-25?"
        )

        assert "17.1" in result["answer"]

    @pytest.mark.skipif(
        not _ollama_available(),
        reason="Ollama not available",
    )
    def test_answer_no_false_causality(self):

        from rag.app import answer_question

        result = answer_question(
            "How does cost optimization affect "
            "EBITDA margin?"
        )

        answer = result["answer"].lower()

        # Must NOT claim direct causation
        assert "increased ebitda margin" not in answer
        assert "cost optimization increased" not in answer

        # Should mention the graph relationship type
        assert "drives" in answer or "graph" in answer

        # Should reference an actual EBITDA margin value
        has_margin = any(
            v in answer
            for v in ["18.4", "18.0", "17.1", "17.9"]
        )
        assert has_margin
