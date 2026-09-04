"""
Tests for the LTM Graph RAG system.

Tests cover:
1. Metric detection
2. Entity detection
3. Path filtering and deduplication
4. Context formatting
5. Answer generation
6. Financial fact data files
7. Entity graph data files
8. Semantic relationships
9. Integration tests (require Neo4j)
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _neo4j_available():
    """Check if Neo4j is reachable."""
    try:
        from rag.query_engine import create_driver

        driver = create_driver()
        driver.close()
        return True
    except Exception:
        return False


from rag.query_engine import (
    GraphQueryEngine,
    EXCLUDED_RELATIONSHIPS,
)
from rag.answer_generator import AnswerGenerator
from rag.app import format_context


# ============================================================
# METRIC DETECTION TESTS
# ============================================================


class TestMetricDetection:
    """Test metric detection from natural language questions."""

    def setup_method(self):
        """
        Create a mocked GraphQueryEngine
        for metric detection tests.
        """

        self.engine = MagicMock(
            spec=GraphQueryEngine
        )

        # Bind instance methods to the mock
        self.engine.detect_metric = (
            GraphQueryEngine.detect_metric.__get__(
                self.engine,
                GraphQueryEngine,
            )
        )

        self.engine.normalize = (
            GraphQueryEngine.normalize
        )

    def test_detect_ebitda(self):

        metric = self.engine.detect_metric(
            "What was LTM's EBITDA?"
        )

        assert metric == "ebitda"

    def test_detect_ebitda_margin(self):

        metric = self.engine.detect_metric(
            "What was LTM's EBITDA margin?"
        )

        assert metric == "ebitda_margin"

    def test_ebitda_margin_priority_over_ebitda(self):

        metric = self.engine.detect_metric(
            "What is the EBITDA margin percentage?"
        )

        assert metric == "ebitda_margin"

    def test_detect_revenue(self):

        metric = self.engine.detect_metric(
            "What was LTM's revenue?"
        )

        assert metric == "revenue"

    def test_detect_revenue_synonym(self):

        metric = self.engine.detect_metric(
            "What were LTM's sales?"
        )

        assert metric == "revenue"

    def test_detect_profit_after_tax(self):

        metric = self.engine.detect_metric(
            "What was the profit after tax?"
        )

        assert metric == "profit_after_tax"

    def test_detect_pat_shorthand(self):

        metric = self.engine.detect_metric(
            "What was LTM's PAT?"
        )

        assert metric == "profit_after_tax"

    def test_detect_eps_diluted(self):

        metric = self.engine.detect_metric(
            "What was the diluted EPS?"
        )

        assert metric == "eps_diluted"

    def test_detect_no_metric(self):

        metric = self.engine.detect_metric(
            "Tell me about cost optimization"
        )

        assert metric is None

    def test_detect_order_inflow(self):

        metric = self.engine.detect_metric(
            "What was the order inflow?"
        )

        assert metric == "order_inflow"

    def test_detect_market_cap(self):

        metric = self.engine.detect_metric(
            "What is the market cap?"
        )

        assert metric == "market_capitalization"

    def test_case_insensitive_detection(self):

        metric = self.engine.detect_metric(
            "What was the EBITDA MARGIN?"
        )

        assert metric == "ebitda_margin"


# ============================================================
# ENTITY NORMALIZATION TESTS
# ============================================================


class TestEntityNormalization:
    """Test entity name normalization and matching."""

    def test_normalize_lowercases(self):

        result = GraphQueryEngine.normalize(
            "What is Cost Optimization?"
        )

        assert result == "what is cost optimization"

    def test_normalize_strips_special_chars(self):

        result = GraphQueryEngine.normalize(
            "How does Fit4Future affect EBITDA?"
        )

        assert "fit4future" in result

    def test_entity_case_insensitive_matching(self):

        engine = MagicMock(
            spec=GraphQueryEngine
        )

        engine.detect_entities = (
            GraphQueryEngine.detect_entities.__get__(
                engine,
                GraphQueryEngine,
            )
        )

        engine.normalize = (
            GraphQueryEngine.normalize
        )

        # Mock the driver session
        mock_session = MagicMock()
        mock_driver = MagicMock()
        engine.driver = mock_driver

        mock_session.run.return_value.data.return_value = [
            {"name": "cost optimization"},
            {"name": "Fit4Future"},
            {"name": "operational efficiency"},
        ]

        mock_driver.session.return_value.__enter__ = (
            MagicMock(return_value=mock_session)
        )

        mock_driver.session.return_value.__exit__ = (
            MagicMock(return_value=False)
        )

        entities = engine.detect_entities(
            "How does Cost Optimization affect EBITDA margin?"
        )

        assert "cost optimization" in entities


# ============================================================
# PATH FILTERING TESTS
# ============================================================


class TestPathFiltering:
    """Test path filtering and deduplication."""

    def test_filter_excludes_has_fact(self):

        paths = [
            {
                "path": ["A", "B"],
                "relationships": ["HAS_FACT"],
            },
            {
                "path": ["A", "C"],
                "relationships": ["SUPPORTS"],
            },
        ]

        result = (
            GraphQueryEngine._filter_and_deduplicate_paths(
                paths
            )
        )

        assert len(result) == 1
        assert result[0]["relationships"] == [
            "SUPPORTS"
        ]

    def test_filter_excludes_self_loop(self):

        paths = [
            {
                "path": [
                    "ebitda_margin",
                    "ebitda_margin",
                ],
                "relationships": ["HAS_FACT"],
            },
        ]

        result = (
            GraphQueryEngine._filter_and_deduplicate_paths(
                paths
            )
        )

        assert len(result) == 0

    def test_filter_excludes_none_nodes(self):

        paths = [
            {
                "path": ["A", None, "C"],
                "relationships": [
                    "SUPPORTS",
                    "DRIVES",
                ],
            },
        ]

        result = (
            GraphQueryEngine._filter_and_deduplicate_paths(
                paths
            )
        )

        assert len(result) == 0

    def test_filter_excludes_single_node(self):

        paths = [
            {
                "path": ["A"],
                "relationships": [],
            },
        ]

        result = (
            GraphQueryEngine._filter_and_deduplicate_paths(
                paths
            )
        )

        assert len(result) == 0

    def test_deduplication(self):

        paths = [
            {
                "path": ["A", "B", "C"],
                "relationships": [
                    "SUPPORTS",
                    "DRIVES",
                ],
            },
            {
                "path": ["A", "B", "C"],
                "relationships": [
                    "SUPPORTS",
                    "DRIVES",
                ],
            },
            {
                "path": ["A", "B", "C"],
                "relationships": [
                    "DRIVES",
                    "SUPPORTS",
                ],
            },
        ]

        result = (
            GraphQueryEngine._filter_and_deduplicate_paths(
                paths
            )
        )

        assert len(result) == 2

    def test_filter_allows_multi_hop_self_loop(self):

        paths = [
            {
                "path": ["A", "B", "A"],
                "relationships": [
                    "SUPPORTS",
                    "ENABLES",
                ],
            },
        ]

        result = (
            GraphQueryEngine._filter_and_deduplicate_paths(
                paths
            )
        )

        assert len(result) == 1

    def test_excluded_relationships_set(self):

        assert "HAS_FACT" in EXCLUDED_RELATIONSHIPS
        assert "HAS_VALUE" in EXCLUDED_RELATIONSHIPS
        assert (
            "RELATED_FINANCIAL_FACT"
            in EXCLUDED_RELATIONSHIPS
        )

    def test_filter_excludes_has_value(self):

        paths = [
            {
                "path": ["A", "B"],
                "relationships": ["HAS_VALUE"],
            },
        ]

        result = (
            GraphQueryEngine._filter_and_deduplicate_paths(
                paths
            )
        )

        assert len(result) == 0

    def test_filter_excludes_empty_string_node(self):

        paths = [
            {
                "path": ["A", "", "C"],
                "relationships": [
                    "SUPPORTS",
                    "DRIVES",
                ],
            },
        ]

        result = (
            GraphQueryEngine._filter_and_deduplicate_paths(
                paths
            )
        )

        assert len(result) == 0


# ============================================================
# CONTEXT FORMATTING TESTS
# ============================================================


class TestContextFormatting:
    """Test context formatting for Ollama."""

    def test_format_context_with_financial_facts(self):

        result = {
            "question": "What was LTM's EBITDA?",
            "detected_metric": "ebitda",
            "financial_facts": [
                {
                    "metric": "ebitda",
                    "value": 75552,
                    "unit": "INR million",
                    "period": "FY2025-26",
                    "page": 23,
                    "section": "Key Performance Indicators",
                    "confidence": 1.0,
                }
            ],
            "entities": [],
            "paths": [],
            "semantic_paths": [],
        }

        context = format_context(result)

        assert "Question:" in context
        assert "ebitda" in context
        assert "75552" in context
        assert "INR million" in context
        assert "FY2025-26" in context
        assert "23" in context

    def test_format_context_with_entities(self):

        result = {
            "question": "What is cost optimization?",
            "detected_metric": None,
            "financial_facts": [],
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
                    },
                    "paths": [],
                    "semantic_paths": [],
                }
            ],
            "paths": [],
            "semantic_paths": [],
        }

        context = format_context(result)

        assert "cost optimization" in context
        assert "IMPROVES" in context
        assert "operational efficiency" in context

    def test_format_context_with_semantic_paths(self):

        result = {
            "question": (
                "How does Fit4Future affect EBITDA?"
            ),
            "detected_metric": "ebitda_margin",
            "financial_facts": [
                {
                    "metric": "ebitda_margin",
                    "value": 17.9,
                    "unit": "%",
                    "period": "FY2025-26",
                    "page": 8,
                    "section": "Financial Performance",
                    "confidence": 1.0,
                }
            ],
            "entities": [],
            "paths": [],
            "semantic_paths": [
                {
                    "nodes": [
                        "Fit4Future",
                        "cost optimization",
                        "operational efficiency",
                        "ebitda_margin",
                    ],
                    "relationships": [
                        "DRIVES",
                        "IMPROVES",
                        "SUPPORTS",
                    ],
                }
            ],
        }

        context = format_context(result)

        assert "Fit4Future" in context
        assert "DRIVES" in context
        assert "IMPROVES" in context
        assert "SUPPORTS" in context
        assert "ebitda_margin" in context

    def test_format_context_empty(self):

        result = {
            "question": "Hello",
            "detected_metric": None,
            "financial_facts": [],
            "entities": [],
            "paths": [],
            "semantic_paths": [],
        }

        context = format_context(result)

        assert "Question:" in context
        assert len(context) < 500

    def test_format_context_debug_metric_key(
        self,
    ):
        """
        Ensure we read detected_metric, not metric.
        """

        result = {
            "question": "test",
            "detected_metric": "ebitda",
            "financial_facts": [],
            "entities": [],
            "paths": [],
            "semantic_paths": [],
        }

        context = format_context(result)

        assert "ebitda" in context


# ============================================================
# ANSWER GENERATOR TESTS
# ============================================================


class TestAnswerGenerator:
    """Test the AnswerGenerator class."""

    def test_default_model(self):

        gen = AnswerGenerator()

        assert gen.model == "qwen2.5:3b"

    def test_default_url(self):

        gen = AnswerGenerator()

        assert gen.ollama_url == (
            "http://localhost:11434/api/generate"
        )

    @patch("rag.answer_generator.requests.post")
    def test_generate_calls_ollama(self, mock_post):

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": (
                "LTM reported an EBITDA of "
                "75,552 INR million."
            )
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        gen = AnswerGenerator()

        answer = gen.generate("context here")

        assert (
            "75,552" in answer or "EBITDA" in answer
        )

        mock_post.assert_called_once()

        call_args = mock_post.call_args

        assert (
            call_args[1]["json"]["model"]
            == "qwen2.5:3b"
        )
        assert (
            call_args[1]["json"]["stream"] is False
        )

    @patch("rag.answer_generator.requests.post")
    def test_generate_single_argument(self, mock_post):

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Test answer"
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        gen = AnswerGenerator()

        # Must work with single argument
        answer = gen.generate("test context")

        assert answer == "Test answer"

    @patch("rag.answer_generator.requests.post")
    def test_generate_strips_whitespace(
        self, mock_post
    ):

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "  answer  "
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        gen = AnswerGenerator()

        answer = gen.generate("context")

        assert answer == "answer"


# ============================================================
# FINANCIAL FACT DATA FILES TESTS
# ============================================================


class TestFinancialFactDataFiles:
    """Test that financial fact data files are correct."""

    def test_kpi_financial_facts_file(self):

        facts_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "financial_facts"
            / "LTIM_canonical_multiyear.json"
        )

        assert facts_file.exists()

        with open(facts_file, "r") as f:
            data = json.load(f)

        metrics = {
            fact["metric"] for fact in data["facts"]
        }

        assert "ebitda" in metrics
        assert "revenue" in metrics

    def test_ebitda_value_in_kpi_facts(self):

        facts_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "financial_facts"
            / "LTIM_canonical_multiyear.json"
        )

        with open(facts_file, "r") as f:
            data = json.load(f)

        ebitdas = [
            fact for fact in data["facts"]
            if fact["metric"] == "ebitda"
        ]

        assert len(ebitdas) == 4
        values = {f["period"]["label"]: f["value"] for f in ebitdas}
        assert values["FY2022-23"] == 61077
        assert values["FY2023-24"] == 63874
        assert values["FY2024-25"] == 64949
        assert values["FY2025-26"] == 75552

    def test_narrative_facts_file(self):

        facts_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "financial_facts"
            / "LTIM_canonical_multiyear.json"
        )

        assert facts_file.exists()

        with open(facts_file, "r") as f:
            data = json.load(f)

        metrics = {
            fact["metric"] for fact in data["facts"]
        }

        assert "ebitda_margin" in metrics

    def test_ebitda_margin_in_narrative_facts(self):

        facts_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "financial_facts"
            / "LTIM_canonical_multiyear.json"
        )

        with open(facts_file, "r") as f:
            data = json.load(f)

        margins = [
            fact for fact in data["facts"]
            if fact["metric"] == "ebitda_margin"
        ]

        assert len(margins) == 4
        values = {f["period"]["label"]: f["value"] for f in margins}
        assert values["FY2022-23"] == 18.4
        assert values["FY2023-24"] == 18.0
        assert values["FY2024-25"] == 17.1
        assert values["FY2025-26"] == 17.9

    def test_canonical_facts_file(self):

        facts_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "financial_facts"
            / "LTIM_canonical_multiyear.json"
        )

        assert facts_file.exists()

        with open(facts_file, "r") as f:
            data = json.load(f)

        assert len(data["facts"]) == 68

        metrics = {
            fact["metric"] for fact in data["facts"]
        }

        assert "ebitda" in metrics
        assert "ebitda_margin" in metrics

    def test_all_facts_have_provenance(self):

        facts_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "financial_facts"
            / "LTIM_canonical_multiyear.json"
        )

        with open(facts_file, "r") as f:
            data = json.load(f)

        for fact in data["facts"]:

            assert "source" in fact
            assert fact["source"].get("document_id")
            assert fact["source"].get("page")


# ============================================================
# ENTITY GRAPH DATA FILES TESTS
# ============================================================


class TestEntityGraphDataFiles:
    """Test entity graph data files."""

    def test_entities_file_exists(self):

        entities_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "entities"
            / "LTIM_entities_multiyear.json"
        )

        assert entities_file.exists()

    def test_entities_have_required_fields(self):

        entities_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "entities"
            / "LTIM_entities_multiyear.json"
        )

        with open(entities_file, "r") as f:
            data = json.load(f)

        assert "entities" in data
        assert len(data["entities"]) >= 25

        for entity in data["entities"]:

            assert "name" in entity
            assert "fiscal_years" in entity
            assert "entity_type" in entity

    def test_key_entities_present(self):

        entities_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "entities"
            / "LTIM_entities_multiyear.json"
        )

        with open(entities_file, "r") as f:
            data = json.load(f)

        names = {
            e["name"].lower()
            for e in data["entities"]
        }

        assert "cost optimization" in names
        assert "operational efficiency" in names
        assert "large deals program" in names

    def test_no_duplicate_canonical_entities(self):

        entities_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "entities"
            / "LTIM_entities_multiyear.json"
        )

        with open(entities_file, "r") as f:
            data = json.load(f)

        entity_names = [
            e["name"]
            for e in data["entities"]
        ]

        assert len(entity_names) == len(
            set(entity_names)
        )


# ============================================================
# SEMANTIC RELATIONSHIPS TESTS
# ============================================================


class TestSemanticRelationships:
    """Test semantic relationship definitions."""

    def test_relationships_include_key_paths(self):

        relationships_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "relationships"
            / "LTIM_relationships_multiyear.json"
        )

        with open(relationships_file, "r") as f:
            data = json.load(f)

        rel_set = set()
        for r in data["relationships"]:
            src = r["source"]
            tgt = r["target"]
            if isinstance(src, dict):
                src_name = src.get("name") or src.get("entity") or src.get("target_metric", "")
            else:
                src_name = src
            if isinstance(tgt, dict):
                tgt_name = tgt.get("name") or tgt.get("entity") or tgt.get("target_metric", "")
            else:
                tgt_name = tgt
            rel_set.add((src_name, tgt_name, r["relation"].upper()))

        assert (
            "cost optimization",
            "ebitda_margin",
            "DRIVES",
        ) in rel_set

        assert (
            "AI",
            "cost optimization",
            "SUPPORTS",
        ) in rel_set

    def test_relationship_count(self):

        relationships_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "relationships"
            / "LTIM_relationships_multiyear.json"
        )

        with open(relationships_file, "r") as f:
            data = json.load(f)

        assert len(data["relationships"]) >= 10


# ============================================================
# INTEGRATION: GRAPH QUERY ENGINE
# (These tests require a running Neo4j instance)
# ============================================================


NEO4J_AVAILABLE = _neo4j_available()


@pytest.mark.skipif(
    not NEO4J_AVAILABLE,
    reason="Neo4j not available",
)
class TestGraphQueryEngineIntegration:
    """Integration tests requiring Neo4j."""

    @classmethod
    def setup_class(cls):

        try:
            cls.engine = GraphQueryEngine()
        except Exception:
            pytest.skip("Neo4j not available")

    @classmethod
    def teardown_class(cls):

        if hasattr(cls, "engine"):
            cls.engine.close()

    def test_search_ebitda(self):

        result = self.engine.search(
            "What was LTIM's EBITDA?"
        )

        assert (
            result["detected_metric"] == "ebitda"
        )
        assert len(result["financial_facts"]) >= 3

        periods = {
            f["period"] for f in result["financial_facts"]
            if f.get("metric") == "ebitda"
        }
        assert {"FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"} <= periods

    def test_search_ebitda_margin(self):

        result = self.engine.search(
            "What is LTIM's EBITDA margin?"
        )

        assert (
            result["detected_metric"]
            == "ebitda_margin"
        )

        assert len(result["financial_facts"]) >= 3

        values = {
            f["period"]: f["value"]
            for f in result["financial_facts"]
            if f.get("metric") == "ebitda_margin"
        }
        assert values.get("FY2022-23") == 18.4
        assert values.get("FY2024-25") == 17.1

    def test_search_entity_cost_optimization(self):

        result = self.engine.search(
            "What is cost optimization?"
        )

        assert result["detected_metric"] is None
        assert len(result["entities"]) > 0

    def test_search_multi_hop(self):

        result = self.engine.search(
            "How does cost optimization affect EBITDA margin?"
        )

        assert (
            result["detected_metric"]
            == "ebitda_margin"
        )

        has_path = False

        for entity_result in result["entities"]:

            for path in entity_result.get(
                "semantic_paths", []
            ):

                nodes = [
                    n.lower() for n in path["nodes"]
                ]

                if (
                    "cost optimization" in nodes
                    and "ebitda_margin" in nodes
                ):

                    has_path = True
                    break

        assert has_path, (
            "Expected a path from cost optimization "
            "to ebitda_margin"
        )

    def test_search_cost_optimization_to_ebitda(self):

        result = self.engine.search(
            "How does cost optimization affect EBITDA margin?"
        )

        all_semantic = result.get(
            "semantic_paths", []
        )

        assert len(all_semantic) > 0

        found = False

        for path_data in all_semantic:

            nodes = [
                n.lower()
                for n in path_data["nodes"]
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

    def test_no_hallucinated_paths(self):

        result = self.engine.search(
            "What was LTM's EBITDA?"
        )

        for path in result.get("paths", []):

            for node in path["path"]:

                assert node is not None
                assert str(node).strip() != ""

    def test_no_useless_traversal_in_semantic_paths(
        self,
    ):

        result = self.engine.search(
            "Starting from Fit4Future, "
            "trace to EBITDA margin."
        )

        for path_data in result.get(
            "semantic_paths", []
        ):

            for rel in path_data.get(
                "relationships", []
            ):

                assert (
                    rel not in EXCLUDED_RELATIONSHIPS
                ), (
                    f"Semantic path contains "
                    f"excluded relationship: {rel}"
                )

    def test_search_structure_keys(self):

        result = self.engine.search(
            "What was LTM's revenue?"
        )

        assert "question" in result
        assert "detected_metric" in result
        assert "financial_facts" in result
        assert "entities" in result
        assert "paths" in result
        assert "semantic_paths" in result
