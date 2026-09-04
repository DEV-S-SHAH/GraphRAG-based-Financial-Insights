"""
Tests for the evaluation framework itself.

Tests cover:
1. Dataset loading
2. Metric detection comparison
3. Numeric comparison with tolerance
4. Entity precision/recall/F1
5. Path matching
6. Relationship matching
7. Unsupported question handling
8. Report generation
9. Answer metric calculation
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from evaluation.evaluate import (
    _normalize,
    _value_within_tolerance,
    calc_metric_detection,
    calc_financial_fact_accuracy,
    calc_entity_precision_recall,
    calc_relationship_accuracy,
    calc_path_match,
    calc_hop_accuracy,
    calc_answer_metrics,
    compute_overall_metrics,
    generate_report,
    load_dataset,
)


# ============================================================
# DATASET LOADING TESTS
# ============================================================


class TestDatasetLoading:
    """Test that the dataset loads correctly."""

    def test_dataset_file_exists(self):
        path = PROJECT_ROOT / "evaluation" / "dataset.json"
        assert path.exists()

    def test_dataset_loads_valid_json(self):
        dataset = load_dataset()
        assert "metadata" in dataset
        assert "questions" in dataset

    def test_dataset_has_at_least_25_questions(self):
        dataset = load_dataset()
        assert len(dataset["questions"]) >= 25

    def test_dataset_has_required_fields(self):
        dataset = load_dataset()
        for q in dataset["questions"]:
            assert "id" in q
            assert "question" in q
            assert "category" in q
            assert "expected" in q

    def test_dataset_has_negative_questions(self):
        dataset = load_dataset()
        negatives = [
            q for q in dataset["questions"]
            if q["category"] == "negative"
        ]
        assert len(negatives) >= 3

    def test_dataset_has_multi_hop_questions(self):
        dataset = load_dataset()
        multi_hop = [
            q for q in dataset["questions"]
            if q["category"] == "multi_hop_graph"
        ]
        assert len(multi_hop) >= 5

    def test_dataset_has_financial_questions(self):
        dataset = load_dataset()
        financial = [
            q for q in dataset["questions"]
            if q["category"] == "financial_fact"
        ]
        assert len(financial) >= 8

    def test_all_question_ids_unique(self):
        dataset = load_dataset()
        ids = [q["id"] for q in dataset["questions"]]
        assert len(ids) == len(set(ids))


# ============================================================
# NORMALIZE TESTS
# ============================================================


class TestNormalize:
    """Test text normalization."""

    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_strips_special_chars(self):
        assert _normalize("EBITDA margin!") == "ebitda margin"

    def test_preserves_numbers(self):
        assert _normalize("FY2024-25") == "fy2024 25"

    def test_empty_string(self):
        assert _normalize("") == ""


# ============================================================
# NUMERIC COMPARISON TESTS
# ============================================================


class TestNumericComparison:
    """Test numeric tolerance comparison."""

    def test_exact_match(self):
        assert _value_within_tolerance(64949, 64949)

    def test_within_1_percent(self):
        assert _value_within_tolerance(64900, 64949)

    def test_outside_1_percent(self):
        assert not _value_within_tolerance(64000, 64949)

    def test_zero_expected(self):
        assert _value_within_tolerance(0, 0)

    def test_zero_expected_nonzero_actual(self):
        assert not _value_within_tolerance(5, 0)

    def test_percentage_value(self):
        assert _value_within_tolerance(17.1, 17.1)

    def test_percentage_within_tolerance(self):
        assert _value_within_tolerance(17.08, 17.1)

    def test_percentage_outside_tolerance(self):
        assert not _value_within_tolerance(16.5, 17.1)


# ============================================================
# METRIC DETECTION TESTS
# ============================================================


class TestMetricDetection:
    """Test metric detection evaluation."""

    def test_correct_detection(self):
        result = {"detected_metric": "ebitda"}
        expected = {"metric": "ebitda"}
        calc = calc_metric_detection(result, expected)
        assert calc["correct"] is True

    def test_incorrect_detection(self):
        result = {"detected_metric": "revenue"}
        expected = {"metric": "ebitda"}
        calc = calc_metric_detection(result, expected)
        assert calc["correct"] is False

    def test_none_expected_none_detected(self):
        result = {"detected_metric": None}
        expected = {"metric": None}
        calc = calc_metric_detection(result, expected)
        assert calc["correct"] is True

    def test_none_expected_some_detected(self):
        result = {"detected_metric": "ebitda"}
        expected = {"metric": None}
        calc = calc_metric_detection(result, expected)
        assert calc["correct"] is False

    def test_case_insensitive(self):
        result = {"detected_metric": "EBITDA"}
        expected = {"metric": "ebitda"}
        calc = calc_metric_detection(result, expected)
        assert calc["correct"] is True

    def test_ebitda_margin_vs_ebitda(self):
        result = {"detected_metric": "ebitda_margin"}
        expected = {"metric": "ebitda"}
        calc = calc_metric_detection(result, expected)
        assert calc["correct"] is False


# ============================================================
# FINANCIAL FACT ACCURACY TESTS
# ============================================================


class TestFinancialFactAccuracy:
    """Test financial fact accuracy evaluation."""

    def test_exact_match(self):
        result = {
            "financial_facts": [
                {
                    "metric": "ebitda",
                    "value": 64949,
                    "unit": "INR million",
                    "period": "FY2024-25",
                }
            ]
        }
        expected = {
            "metric": "ebitda",
            "value": 64949,
            "unit": "INR million",
            "period": "FY2024-25",
        }
        calc = calc_financial_fact_accuracy(result, expected)
        assert calc["exact_match"] is True
        assert calc["accuracy"] == 1.0

    def test_wrong_value(self):
        result = {
            "financial_facts": [
                {
                    "metric": "ebitda",
                    "value": 50000,
                    "unit": "INR million",
                    "period": "FY2024-25",
                }
            ]
        }
        expected = {
            "metric": "ebitda",
            "value": 64949,
            "unit": "INR million",
            "period": "FY2024-25",
        }
        calc = calc_financial_fact_accuracy(result, expected)
        assert calc["exact_match"] is False

    def test_no_facts_retrieved(self):
        result = {"financial_facts": []}
        expected = {
            "metric": "ebitda",
            "value": 64949,
            "unit": "INR million",
            "period": "FY2024-25",
        }
        calc = calc_financial_fact_accuracy(result, expected)
        assert calc["exact_match"] is False

    def test_multiple_facts_match(self):
        result = {
            "financial_facts": [
                {"metric": "ebitda", "value": 61077, "unit": "INR million", "period": "FY2022-23"},
                {"metric": "ebitda", "value": 63874, "unit": "INR million", "period": "FY2023-24"},
                {"metric": "ebitda", "value": 64949, "unit": "INR million", "period": "FY2024-25"},
            ]
        }
        expected = {
            "metric": "ebitda",
            "facts": [
                {"value": 61077, "unit": "INR million", "period": "FY2022-23"},
                {"value": 63874, "unit": "INR million", "period": "FY2023-24"},
                {"value": 64949, "unit": "INR million", "period": "FY2024-25"},
            ]
        }
        calc = calc_financial_fact_accuracy(result, expected)
        assert calc["accuracy"] == 1.0
        assert calc["matched_count"] == 3

    def test_partial_facts_match(self):
        result = {
            "financial_facts": [
                {"metric": "ebitda", "value": 61077, "unit": "INR million", "period": "FY2022-23"},
                {"metric": "ebitda", "value": 63874, "unit": "INR million", "period": "FY2023-24"},
            ]
        }
        expected = {
            "metric": "ebitda",
            "facts": [
                {"value": 61077, "unit": "INR million", "period": "FY2022-23"},
                {"value": 63874, "unit": "INR million", "period": "FY2023-24"},
                {"value": 64949, "unit": "INR million", "period": "FY2024-25"},
            ]
        }
        calc = calc_financial_fact_accuracy(result, expected)
        assert calc["accuracy"] == pytest.approx(2 / 3, abs=0.01)
        assert calc["matched_count"] == 2

    def test_percentage_value(self):
        result = {
            "financial_facts": [
                {"metric": "ebitda_margin", "value": 17.1, "unit": "%", "period": "FY2024-25"}
            ]
        }
        expected = {
            "metric": "ebitda_margin",
            "value": 17.1,
            "unit": "%",
            "period": "FY2024-25",
        }
        calc = calc_financial_fact_accuracy(result, expected)
        assert calc["exact_match"] is True


# ============================================================
# ENTITY F1 TESTS
# ============================================================


class TestEntityPrecisionRecall:
    """Test entity precision, recall, and F1."""

    def test_perfect_match(self):
        result = {
            "entities": [
                {
                    "entity": {
                        "entity": "cost optimization",
                        "outgoing": [
                            {"relationship": "IMPROVES", "target": "operational efficiency"}
                        ],
                        "incoming": [],
                    }
                }
            ],
            "semantic_paths": [],
            "paths": [],
        }
        expected = {
            "expected_entities": ["cost optimization", "operational efficiency"],
        }
        calc = calc_entity_precision_recall(result, expected)
        assert calc["recall"] == 1.0
        assert calc["f1"] > 0

    def test_partial_entity_match(self):
        result = {
            "entities": [
                {
                    "entity": {
                        "entity": "cost optimization",
                        "outgoing": [],
                        "incoming": [],
                    }
                }
            ],
            "semantic_paths": [],
            "paths": [],
        }
        expected = {
            "expected_entities": ["cost optimization", "operational efficiency"],
        }
        calc = calc_entity_precision_recall(result, expected)
        assert calc["recall"] == pytest.approx(0.5, abs=0.01)

    def test_no_expected_entities(self):
        result = {"entities": [], "semantic_paths": [], "paths": []}
        expected = {"expected_entities": []}
        calc = calc_entity_precision_recall(result, expected)
        assert calc["f1"] == 1.0

    def test_semantic_path_entities_count(self):
        result = {
            "entities": [],
            "semantic_paths": [
                {
                    "nodes": ["Fit4Future", "cost optimization", "operational efficiency", "ebitda_margin"],
                    "relationships": ["DRIVES", "IMPROVES", "SUPPORTS"],
                }
            ],
            "paths": [],
        }
        expected = {
            "expected_entities": ["Fit4Future", "operational efficiency", "ebitda_margin"],
        }
        calc = calc_entity_precision_recall(result, expected)
        assert calc["recall"] == 1.0

    def test_no_entities_retrieved(self):
        result = {"entities": [], "semantic_paths": [], "paths": []}
        expected = {
            "expected_entities": ["cost optimization"],
        }
        calc = calc_entity_precision_recall(result, expected)
        assert calc["recall"] == 0.0
        assert calc["precision"] == 0.0


# ============================================================
# RELATIONSHIP MATCHING TESTS
# ============================================================


class TestRelationshipMatching:
    """Test relationship accuracy evaluation."""

    def test_perfect_relationship_match(self):
        result = {
            "entities": [
                {
                    "entity": {
                        "entity": "cost optimization",
                        "outgoing": [{"relationship": "IMPROVES", "target": "operational efficiency"}],
                        "incoming": [],
                    }
                }
            ],
            "semantic_paths": [],
            "paths": [],
        }
        expected = {"expected_relationships": ["IMPROVES"]}
        calc = calc_relationship_accuracy(result, expected)
        assert calc["accuracy"] == 1.0

    def test_missing_relationship(self):
        result = {
            "entities": [],
            "semantic_paths": [],
            "paths": [],
        }
        expected = {"expected_relationships": ["DRIVES", "SUPPORTS"]}
        calc = calc_relationship_accuracy(result, expected)
        assert calc["accuracy"] == 0.0

    def test_semantic_path_relationships(self):
        result = {
            "entities": [],
            "semantic_paths": [
                {
                    "nodes": ["A", "B", "C"],
                    "relationships": ["IMPROVES", "SUPPORTS"],
                }
            ],
            "paths": [],
        }
        expected = {"expected_relationships": ["IMPROVES", "SUPPORTS"]}
        calc = calc_relationship_accuracy(result, expected)
        assert calc["accuracy"] == 1.0

    def test_partial_relationship_match(self):
        result = {
            "entities": [],
            "semantic_paths": [
                {
                    "nodes": ["A", "B"],
                    "relationships": ["DRIVES"],
                }
            ],
            "paths": [],
        }
        expected = {"expected_relationships": ["DRIVES", "SUPPORTS"]}
        calc = calc_relationship_accuracy(result, expected)
        assert calc["accuracy"] == pytest.approx(0.5, abs=0.01)

    def test_no_expected_relationships(self):
        result = {"entities": [], "semantic_paths": [], "paths": []}
        expected = {"expected_relationships": []}
        calc = calc_relationship_accuracy(result, expected)
        assert calc["accuracy"] == 1.0


# ============================================================
# PATH MATCHING TESTS
# ============================================================


class TestPathMatching:
    """Test graph path matching."""

    def test_exact_path_match(self):
        result = {
            "entities": [],
            "semantic_paths": [
                {
                    "nodes": ["cost optimization", "operational efficiency", "ebitda_margin"],
                    "relationships": ["IMPROVES", "SUPPORTS"],
                }
            ],
            "paths": [],
        }
        expected = {
            "expected_path": [
                "cost optimization",
                "operational efficiency",
                "ebitda_margin",
            ]
        }
        calc = calc_path_match(result, expected)
        assert calc["exact_match_count"] >= 1

    def test_longer_path_contains_expected(self):
        result = {
            "entities": [],
            "semantic_paths": [
                {
                    "nodes": ["Fit4Future", "cost optimization", "operational efficiency", "ebitda_margin"],
                    "relationships": ["DRIVES", "IMPROVES", "SUPPORTS"],
                }
            ],
            "paths": [],
        }
        expected = {
            "expected_path": [
                "cost optimization",
                "operational efficiency",
                "ebitda_margin",
            ]
        }
        calc = calc_path_match(result, expected)
        assert calc["exact_match_count"] >= 1

    def test_no_matching_path(self):
        result = {
            "entities": [],
            "semantic_paths": [
                {
                    "nodes": ["AI", "digital transformation"],
                    "relationships": ["SUPPORTS"],
                }
            ],
            "paths": [],
        }
        expected = {
            "expected_path": [
                "cost optimization",
                "operational efficiency",
                "ebitda_margin",
            ]
        }
        calc = calc_path_match(result, expected)
        assert calc["path_recall"] == 0.0

    def test_no_paths_retrieved(self):
        result = {
            "entities": [],
            "semantic_paths": [],
            "paths": [],
        }
        expected = {
            "expected_path": ["A", "B", "C"],
        }
        calc = calc_path_match(result, expected)
        assert calc["path_recall"] == 0.0

    def test_no_expected_paths(self):
        result = {
            "entities": [],
            "semantic_paths": [{"nodes": ["A", "B"], "relationships": ["X"]}],
            "paths": [],
        }
        expected = {}
        calc = calc_path_match(result, expected)
        assert calc["path_recall"] == 1.0

    def test_multiple_expected_paths(self):
        result = {
            "entities": [],
            "semantic_paths": [
                {
                    "nodes": ["Fit4Future", "cost optimization", "operational efficiency", "ebitda_margin"],
                    "relationships": ["DRIVES", "IMPROVES", "SUPPORTS"],
                },
                {
                    "nodes": ["Fit4Future", "operational efficiency", "ebitda_margin"],
                    "relationships": ["IMPROVES", "SUPPORTS"],
                },
            ],
            "paths": [],
        }
        expected = {
            "expected_paths": [
                ["Fit4Future", "cost optimization", "operational efficiency", "ebitda_margin"],
                ["Fit4Future", "operational efficiency", "ebitda_margin"],
            ]
        }
        calc = calc_path_match(result, expected)
        assert calc["path_recall"] >= 0.5


# ============================================================
# HOP ACCURACY TESTS
# ============================================================


class TestHopAccuracy:
    """Test hop-level accuracy calculation."""

    def test_2_hop_correct(self):
        result = {
            "entities": [],
            "semantic_paths": [
                {
                    "nodes": ["cost optimization", "operational efficiency", "ebitda_margin"],
                    "relationships": ["IMPROVES", "SUPPORTS"],
                }
            ],
            "paths": [],
        }
        expected = {
            "expected_path": ["cost optimization", "operational efficiency", "ebitda_margin"],
            "path_length_hops": 2,
        }
        calc = calc_hop_accuracy(result, expected)
        assert calc["hop_level"] == 2
        assert calc["correct"] is True

    def test_3_hop_correct(self):
        result = {
            "entities": [],
            "semantic_paths": [
                {
                    "nodes": ["Fit4Future", "cost optimization", "operational efficiency", "ebitda_margin"],
                    "relationships": ["DRIVES", "IMPROVES", "SUPPORTS"],
                }
            ],
            "paths": [],
        }
        expected = {
            "expected_path": ["Fit4Future", "cost optimization", "operational efficiency", "ebitda_margin"],
            "path_length_hops": 3,
        }
        calc = calc_hop_accuracy(result, expected)
        assert calc["hop_level"] == 3
        assert calc["correct"] is True

    def test_3_hop_incorrect(self):
        result = {
            "entities": [],
            "semantic_paths": [],
            "paths": [],
        }
        expected = {
            "expected_path": ["Fit4Future", "cost optimization", "operational efficiency", "ebitda_margin"],
            "path_length_hops": 3,
        }
        calc = calc_hop_accuracy(result, expected)
        assert calc["hop_level"] == 3
        assert calc["correct"] is False

    def test_no_hop_level(self):
        result = {"entities": [], "semantic_paths": [], "paths": []}
        expected = {"expected_entities": ["A"]}
        calc = calc_hop_accuracy(result, expected)
        assert calc["hop_level"] is None


# ============================================================
# ANSWER METRICS TESTS
# ============================================================


class TestAnswerMetrics:
    """Test answer quality metric calculation."""

    def test_financial_answer_correct(self):
        answer = "LTIM reported an EBITDA of 64,949 INR million in FY2024-25."
        expected = {"metric": "ebitda", "value": 64949, "unit": "INR million", "period": "FY2024-25"}
        calc = calc_answer_metrics(answer, expected, "financial_fact", "What was EBITDA?")
        assert calc["score"] > 50

    def test_financial_answer_missing_value(self):
        answer = "LTIM reported revenue."
        expected = {"metric": "ebitda", "value": 64949, "unit": "INR million", "period": "FY2024-25"}
        calc = calc_answer_metrics(answer, expected, "financial_fact", "What was EBITDA?")
        assert calc["score"] < 50

    def test_negative_answer_correct(self):
        answer = "There is not enough evidence to answer this question. The data is not available."
        expected = {"answerable": False}
        calc = calc_answer_metrics(answer, expected, "negative", "What was revenue in FY2020?")
        assert calc["score"] > 50

    def test_negative_answer_hallucination(self):
        answer = "The revenue was 500 billion INR in FY2020."
        expected = {"answerable": False}
        calc = calc_answer_metrics(answer, expected, "negative", "What was revenue in FY2020?")
        assert calc["score"] < 50

    def test_multi_hop_answer_correct(self):
        answer = "Fit4Future drives cost optimization, which improves operational efficiency, which supports EBITDA margin through the graph."
        expected = {
            "expected_entities": ["Fit4Future", "cost optimization", "operational efficiency"],
            "expected_relationships": ["DRIVES", "IMPROVES", "SUPPORTS"],
        }
        calc = calc_answer_metrics(answer, expected, "multi_hop_graph", "Trace Fit4Future to EBITDA margin")
        assert calc["score"] >= 50

    def test_multi_hop_answer_no_causation(self):
        answer = "Fit4Future caused EBITDA margin to increase."
        expected = {
            "expected_entities": ["Fit4Future"],
            "expected_relationships": ["DRIVES"],
        }
        calc = calc_answer_metrics(answer, expected, "multi_hop_graph", "Trace Fit4Future to EBITDA margin")
        checks_dict = dict(calc["checks"])
        assert checks_dict.get("no_false_causation") is False


# ============================================================
# OVERALL METRICS TESTS
# ============================================================


class TestOverallMetrics:
    """Test overall metric computation."""

    def test_perfect_results(self):
        results = [
            {
                "id": "q1",
                "category": "financial_fact",
                "retrieval_metrics": {
                    "metric_detection": {"correct": True},
                    "financial_fact_accuracy": {"accuracy": 1.0},
                    "entity_f1": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                },
                "answer_metrics": {"score": 100},
                "generated_answer": "EBITDA was 64,949 INR million.",
            }
        ]
        overall = compute_overall_metrics(results)
        assert overall["metric_detection_accuracy"] == 1.0
        assert overall["financial_fact_accuracy"] == 1.0
        assert overall["entity_f1"] == 1.0

    def test_mixed_results(self):
        results = [
            {
                "id": "q1",
                "category": "financial_fact",
                "retrieval_metrics": {
                    "metric_detection": {"correct": True},
                    "financial_fact_accuracy": {"accuracy": 1.0},
                    "entity_f1": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                },
                "answer_metrics": {"score": 100},
                "generated_answer": "Answer 1",
            },
            {
                "id": "q2",
                "category": "financial_fact",
                "retrieval_metrics": {
                    "metric_detection": {"correct": False},
                    "financial_fact_accuracy": {"accuracy": 0.0},
                    "entity_f1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                },
                "answer_metrics": {"score": 0},
                "generated_answer": "Answer 2",
            },
        ]
        overall = compute_overall_metrics(results)
        assert overall["metric_detection_accuracy"] == 0.5
        assert overall["financial_fact_accuracy"] == 0.5

    def test_negative_questions(self):
        results = [
            {
                "id": "q1",
                "category": "negative",
                "retrieval_metrics": {},
                "answer_metrics": {"score": 100},
                "generated_answer": "Not enough evidence available.",
            },
            {
                "id": "q2",
                "category": "negative",
                "retrieval_metrics": {},
                "answer_metrics": {"score": 0},
                "generated_answer": "The value was 500 billion.",
            },
        ]
        overall = compute_overall_metrics(results)
        assert overall["negative_question_accuracy"] == 0.5

    def test_empty_results(self):
        overall = compute_overall_metrics([])
        assert overall["total_questions"] == 0


# ============================================================
# REPORT GENERATION TESTS
# ============================================================


class TestReportGeneration:
    """Test that the report generates valid markdown."""

    def test_report_contains_header(self):
        results = []
        overall = {
            "total_questions": 0,
            "metric_detection_accuracy": 0,
            "financial_fact_accuracy": 0,
            "entity_precision": 0,
            "entity_recall": 0,
            "entity_f1": 0,
            "relationship_accuracy": 0,
            "graph_path_accuracy": 0,
            "path_recall": 0,
            "2_hop_accuracy": 0,
            "3_hop_accuracy": 0,
            "4_hop_accuracy": 0,
            "answer_correctness": 0,
            "negative_question_accuracy": None,
            "category_stats": {},
        }
        report = generate_report(results, overall)
        assert "GRAPH RAG EVALUATION REPORT" in report
        assert "Overall Metrics" in report

    def test_report_contains_metrics_table(self):
        overall = {
            "total_questions": 1,
            "metric_detection_accuracy": 1.0,
            "financial_fact_accuracy": 0.5,
            "entity_precision": 0.8,
            "entity_recall": 0.9,
            "entity_f1": 0.85,
            "relationship_accuracy": 1.0,
            "graph_path_accuracy": 0.5,
            "path_recall": 0.5,
            "2_hop_accuracy": 1.0,
            "3_hop_accuracy": None,
            "4_hop_accuracy": None,
            "answer_correctness": 0.75,
            "negative_question_accuracy": 1.0,
            "category_stats": {"financial_fact": {"total": 1, "correct": 1}},
        }
        report = generate_report([], overall)
        assert "metric_detection_accuracy" in report
        assert "100.0%" in report
        assert "50.0%" in report

    def test_report_with_failures(self):
        results = [
            {
                "id": "q1",
                "question": "Test question?",
                "category": "financial_fact",
                "generated_answer": "Wrong answer",
                "retrieval_metrics": {
                    "metric_detection": {"correct": False, "detected": "revenue", "expected": "ebitda"},
                    "financial_fact_accuracy": {"accuracy": 0.0, "matched_count": 0, "expected_count": 1},
                    "entity_f1": {"f1": 0.0, "correct_entities": []},
                    "relationship_accuracy": {"missing_rels": ["DRIVES"]},
                    "graph_path_match": {"expected_path_count": 1, "exact_match_count": 0, "path_recall": 0},
                },
                "answer_metrics": {"score": 0},
            }
        ]
        overall = {
            "total_questions": 1,
            "metric_detection_accuracy": 0,
            "financial_fact_accuracy": 0,
            "entity_precision": 0,
            "entity_recall": 0,
            "entity_f1": 0,
            "relationship_accuracy": 0,
            "graph_path_accuracy": 0,
            "path_recall": 0,
            "2_hop_accuracy": None,
            "3_hop_accuracy": None,
            "4_hop_accuracy": None,
            "answer_correctness": 0,
            "negative_question_accuracy": None,
            "category_stats": {"financial_fact": {"total": 1, "correct": 0}},
        }
        report = generate_report(results, overall)
        assert "Failure Analysis" in report
        assert "q1" in report
