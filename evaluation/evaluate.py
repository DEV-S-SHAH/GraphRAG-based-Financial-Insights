"""
Evaluation Framework for LTIMindtree GraphRAG.

Runs the golden evaluation dataset through the existing pipeline
and calculates retrieval, graph reasoning, and answer quality metrics.

Usage:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --category financial
    python -m evaluation.evaluate --category multi_hop
    python -m evaluation.evaluate --limit 10
    python -m evaluation.evaluate --skip-generation
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the golden evaluation dataset."""
    if path is None:
        path = PROJECT_ROOT / "evaluation" / "dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# NORMALIZE HELPER
# ============================================================

def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric chars for comparison."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _value_within_tolerance(
    actual: float,
    expected: float,
    tolerance_pct: float = 1.0,
) -> bool:
    """Check if actual is within tolerance_pct of expected."""
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) * 100 <= tolerance_pct


# ============================================================
# RETRIEVAL METRIC CALCULATORS
# ============================================================

def calc_metric_detection(
    result: Dict[str, Any],
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Metric 1: Did we detect the correct financial metric?
    """
    detected = result.get("detected_metric")
    expected_metric = expected.get("metric")

    if expected_metric is None:
        return {
            "metric": "metric_detection",
            "correct": detected is None,
            "expected": None,
            "detected": detected,
        }

    correct = (
        detected is not None
        and detected.lower() == expected_metric.lower()
    )

    return {
        "metric": "metric_detection",
        "correct": correct,
        "expected": expected_metric,
        "detected": detected,
    }


def calc_financial_fact_accuracy(
    result: Dict[str, Any],
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Metric 2: Are the retrieved financial facts correct?
    Checks metric, value, unit, period.
    """
    facts = result.get("financial_facts", [])

    if "facts" in expected:
        expected_metric = expected["metric"]
        expected_facts = expected["facts"]

        matched = 0
        for ef in expected_facts:
            for f in facts:
                if (
                    f.get("metric", "").lower() == expected_metric.lower()
                    and f.get("period") == ef["period"]
                    and f.get("unit", "").lower() == ef["unit"].lower()
                    and _value_within_tolerance(
                        float(f.get("value", 0)),
                        float(ef["value"]),
                    )
                ):
                    matched += 1
                    break

        return {
            "metric": "financial_fact_accuracy",
            "expected_count": len(expected_facts),
            "matched_count": matched,
            "accuracy": matched / len(expected_facts) if expected_facts else 0,
            "exact_match": matched == len(expected_facts),
        }
    else:
        expected_metric = expected.get("metric")
        expected_value = expected.get("value")
        expected_unit = expected.get("unit")
        expected_period = expected.get("period")

        best_match = None
        for f in facts:
            if f.get("metric", "").lower() != (expected_metric or "").lower():
                continue
            if expected_period and f.get("period") != expected_period:
                continue
            if expected_unit and f.get("unit", "").lower() != expected_unit.lower():
                continue
            if expected_value is not None:
                if not _value_within_tolerance(
                    float(f.get("value", 0)),
                    float(expected_value),
                ):
                    continue
            best_match = f
            break

        return {
            "metric": "financial_fact_accuracy",
            "expected_count": 1,
            "matched_count": 1 if best_match else 0,
            "accuracy": 1.0 if best_match else 0.0,
            "exact_match": best_match is not None,
        }


def calc_entity_precision_recall(
    result: Dict[str, Any],
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Metric 3-5: Entity precision, recall, F1.
    """
    expected_entities = [
        e.lower() for e in expected.get("expected_entities", [])
    ]
    if not expected_entities:
        return {
            "metric": "entity_f1",
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "retrieved_count": 0,
            "expected_count": 0,
        }

    retrieved_entity_names = set()
    for ent_result in result.get("entities", []):
        entity = ent_result.get("entity", {})
        if isinstance(entity, dict):
            name = entity.get("entity", "")
            if name:
                retrieved_entity_names.add(name.lower())
            for rel in entity.get("outgoing", []):
                t = rel.get("target", "")
                if t:
                    retrieved_entity_names.add(t.lower())
            for rel in entity.get("incoming", []):
                s = rel.get("source", "")
                if s:
                    retrieved_entity_names.add(s.lower())

    for path_data in result.get("semantic_paths", []):
        nodes = path_data.get("nodes", path_data.get("path", []))
        for n in nodes:
            if n:
                retrieved_entity_names.add(str(n).lower())

    for path_data in result.get("paths", []):
        nodes = path_data.get("path", [])
        for n in nodes:
            if n:
                retrieved_entity_names.add(str(n).lower())

    expected_set = set(expected_entities)
    correct = expected_set & retrieved_entity_names

    precision = (
        len(correct) / len(retrieved_entity_names)
        if retrieved_entity_names
        else 0.0
    )
    recall = (
        len(correct) / len(expected_set)
        if expected_set
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "metric": "entity_f1",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "correct_entities": sorted(correct),
        "retrieved_count": len(retrieved_entity_names),
        "expected_count": len(expected_set),
    }


def calc_relationship_accuracy(
    result: Dict[str, Any],
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Metric 7: Were the correct relationship types retrieved?
    """
    expected_rels = [
        r.upper() for r in expected.get("expected_relationships", [])
    ]
    if not expected_rels:
        return {
            "metric": "relationship_accuracy",
            "accuracy": 1.0,
            "correct_rels": [],
        }

    retrieved_rels = set()
    for ent_result in result.get("entities", []):
        entity = ent_result.get("entity", {})
        if isinstance(entity, dict):
            for rel in entity.get("outgoing", []):
                r = rel.get("relationship", "")
                if r:
                    retrieved_rels.add(r.upper())
            for rel in entity.get("incoming", []):
                r = rel.get("relationship", "")
                if r:
                    retrieved_rels.add(r.upper())

    for path_data in result.get("semantic_paths", []):
        for r in path_data.get("relationships", []):
            retrieved_rels.add(r.upper())
    for path_data in result.get("paths", []):
        for r in path_data.get("relationships", []):
            retrieved_rels.add(r.upper())

    correct = [
        r for r in expected_rels if r in retrieved_rels
    ]

    return {
        "metric": "relationship_accuracy",
        "accuracy": len(correct) / len(expected_rels) if expected_rels else 1.0,
        "expected_count": len(expected_rels),
        "correct_count": len(correct),
        "correct_rels": correct,
        "missing_rels": [r for r in expected_rels if r not in retrieved_rels],
        "retrieved_rels": sorted(retrieved_rels),
    }


def calc_path_match(
    result: Dict[str, Any],
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Metric 6 & 8: Graph path exact match and path recall.
    """
    expected_paths = expected.get("expected_paths", [])
    expected_single = expected.get("expected_path")

    if expected_single and not expected_paths:
        expected_paths = [expected_single]

    if not expected_paths:
        has_semantic = len(result.get("semantic_paths", [])) > 0
        has_paths = len(result.get("paths", [])) > 0
        return {
            "metric": "graph_path_match",
            "any_path_found": has_semantic or has_paths,
            "exact_match_count": 0,
            "expected_path_count": 0,
            "path_recall": 1.0,
        }

    all_retrieved_paths = []
    for path_data in result.get("semantic_paths", []):
        nodes = [
            str(n).lower()
            for n in path_data.get("nodes", path_data.get("path", []))
        ]
        rels = path_data.get("relationships", [])
        all_retrieved_paths.append({"nodes": nodes, "rels": rels})

    for path_data in result.get("paths", []):
        nodes = [
            str(n).lower()
            for n in path_data.get("path", [])
        ]
        rels = path_data.get("relationships", [])
        all_retrieved_paths.append({"nodes": nodes, "rels": rels})

    matched_count = 0
    for ep in expected_paths:
        ep_lower = [n.lower() for n in ep]
        for rp in all_retrieved_paths:
            rp_nodes = rp["nodes"]
            if ep_lower[0] in rp_nodes and ep_lower[-1] in rp_nodes:
                start_idx = rp_nodes.index(ep_lower[0])
                end_idx = rp_nodes.index(ep_lower[-1])
                if start_idx < end_idx:
                    segment = rp_nodes[start_idx:end_idx + 1]
                    if all(n in segment for n in ep_lower):
                        matched_count += 1
                        break

    path_recall = (
        matched_count / len(expected_paths)
        if expected_paths
        else 1.0
    )

    return {
        "metric": "graph_path_match",
        "any_path_found": len(all_retrieved_paths) > 0,
        "exact_match_count": matched_count,
        "expected_path_count": len(expected_paths),
        "path_recall": round(path_recall, 4),
        "all_retrieved_paths": [
            rp["nodes"] for rp in all_retrieved_paths
        ],
    }


def calc_hop_accuracy(
    result: Dict[str, Any],
    expected: Dict[str, Any],
) -> Dict[str, Optional[int]]:
    """
    Calculate which hop level this question belongs to.
    """
    path_len = expected.get("path_length_hops")

    if path_len is None:
        return {"hop_level": None, "correct": None}

    pm = calc_path_match(result, expected)
    correct = pm["path_recall"] > 0

    return {
        "hop_level": path_len,
        "correct": correct,
    }


# ============================================================
# ANSWER EVALUATION
# ============================================================

def calc_answer_metrics(
    answer: str,
    expected: Dict[str, Any],
    category: str,
    question: str,
) -> Dict[str, Any]:
    """
    Deterministic answer quality metrics.
    """
    a = answer.lower() if answer else ""
    checks = []

    if category == "financial_fact":
        ef = expected.get("facts", [expected])
        if not isinstance(ef, list):
            ef = [expected]

        for fact in ef:
            val = fact.get("value")
            if val is not None:
                val_str = str(val).replace(",", "")
                val_with_comma = f"{val:,.0f}" if isinstance(val, int) else f"{val:,.1f}"
                checks.append((
                    f"value_{val}",
                    val_str in a or val_with_comma in a,
                ))
            period = fact.get("period")
            if period:
                checks.append((
                    f"period_{period}",
                    period in a or period.replace("FY", "") in a,
                ))

    elif category == "negative":
        unsupported_terms = [
            "not enough", "not available", "not provided",
            "not in the", "cannot determine", "not specified",
            "no information", "not mentioned", "not disclosed",
            "insufficient", "not found", "unavailable",
            "do not have", "don't have", "not covered",
            "no data", "not applicable", "outside the scope",
        ]
        checks.append((
            "says_insufficient",
            any(term in a for term in unsupported_terms),
        ))

    elif category == "multi_hop_graph":
        checks.append(("mentions_entities",
                        any(e.lower() in a for e in expected.get("expected_entities", []))))
        checks.append(("mentions_relationships",
                        any(r.lower() in a for r in expected.get("expected_relationships", []))))
        checks.append(("no_false_causation",
                        "caused" not in a and "directly increased" not in a))
        checks.append(("describes_path",
                        any(w in a for w in ["path", "chain", "connect", "hop",
                                             "indirect", "graph", "relationship", "->", "→"])))

    elif category in ("entity_graph", "single_hop_graph"):
        checks.append(("mentions_entities",
                        any(e.lower() in a for e in expected.get("expected_entities", []))))
        checks.append(("mentions_relationships",
                        any(r.lower() in a for r in expected.get("expected_relationships", []))))

    passed = sum(1 for _, v in checks if v)
    total = len(checks) if checks else 1
    score = round(passed / total * 100)

    return {
        "score": score,
        "checks": checks,
        "passed": passed,
        "total": total,
    }


# ============================================================
# ANSWER GENERATION (optional)
# ============================================================

def try_generate_answer(
    context: str,
    answer_generator=None,
) -> Optional[str]:
    """Try to generate an answer using Ollama. Returns None if unavailable."""
    if answer_generator is None:
        try:
            from rag.answer_generator import AnswerGenerator
            answer_generator = AnswerGenerator()
        except Exception:
            return None

    try:
        return answer_generator.generate(context)
    except Exception:
        return None


# ============================================================
# MAIN EVALUATOR
# ============================================================

def evaluate_question(
    q: Dict[str, Any],
    engine=None,
    answer_generator=None,
    skip_generation: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a single question through the full pipeline.
    """
    question_text = q["question"]
    category = q["category"]
    expected = q["expected"]

    result_entry = {
        "id": q["id"],
        "question": question_text,
        "category": category,
    }

    t0 = time.time()

    try:
        from rag.query_engine import GraphQueryEngine
        if engine is None:
            engine = GraphQueryEngine()
            close_engine = True
        else:
            close_engine = False

        search_result = engine.search(question_text)

        result_entry["detected_metric"] = search_result.get("detected_metric")
        result_entry["financial_facts"] = search_result.get("financial_facts", [])
        result_entry["entities"] = search_result.get("entities", [])
        result_entry["paths"] = search_result.get("paths", [])
        result_entry["semantic_paths"] = search_result.get("semantic_paths", [])

        retrieval_time = time.time() - t0
        result_entry["retrieval_time_s"] = round(retrieval_time, 3)

        metrics = {}

        if "metric" in expected and expected["metric"] is not None:
            metrics["metric_detection"] = calc_metric_detection(
                search_result, expected
            )

        if "metric" in expected and expected["metric"] is not None:
            metrics["financial_fact_accuracy"] = calc_financial_fact_accuracy(
                search_result, expected
            )

        if "expected_entities" in expected:
            metrics["entity_f1"] = calc_entity_precision_recall(
                search_result, expected
            )

        if "expected_relationships" in expected:
            metrics["relationship_accuracy"] = calc_relationship_accuracy(
                search_result, expected
            )

        if "expected_path" in expected or "expected_paths" in expected:
            metrics["graph_path_match"] = calc_path_match(
                search_result, expected
            )

        if "path_length_hops" in expected:
            metrics["hop_accuracy"] = calc_hop_accuracy(
                search_result, expected
            )

        result_entry["retrieval_metrics"] = metrics

        answer = None
        if not skip_generation and category != "negative":
            from rag.context_fusion import build_context
            from rag.hybrid_retriever import HybridRetriever

            try:
                retriever = HybridRetriever(
                    graph_engine=engine
                )
                hybrid_result = retriever.retrieve(
                    question_text, top_k=5
                )
                context = build_context(hybrid_result)
                answer = try_generate_answer(context, answer_generator)
                retriever.close()
            except Exception:
                answer = None

        result_entry["generated_answer"] = answer or ""

        if answer:
            result_entry["answer_metrics"] = calc_answer_metrics(
                answer, expected, category, question_text
            )

        if close_engine:
            engine.close()

    except Exception as exc:
        result_entry["error"] = str(exc)
        result_entry["retrieval_metrics"] = {}
        result_entry["generated_answer"] = ""

    return result_entry


def compute_overall_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate per-question metrics into overall scores.
    """
    metrics: Dict[str, List[float]] = {
        "metric_detection": [],
        "financial_fact_accuracy": [],
        "entity_precision": [],
        "entity_recall": [],
        "entity_f1": [],
        "relationship_accuracy": [],
        "graph_path_accuracy": [],
        "path_recall": [],
        "2_hop_accuracy": [],
        "3_hop_accuracy": [],
        "4_hop_accuracy": [],
        "answer_correctness": [],
    }

    negative_correct = 0
    negative_total = 0
    category_stats: Dict[str, Dict[str, int]] = {}

    for r in results:
        cat = r.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1

        rm = r.get("retrieval_metrics", {})

        md = rm.get("metric_detection", {})
        if "correct" in md:
            metrics["metric_detection"].append(
                1.0 if md["correct"] else 0.0
            )

        ffa = rm.get("financial_fact_accuracy", {})
        if "accuracy" in ffa:
            metrics["financial_fact_accuracy"].append(ffa["accuracy"])

        ef1 = rm.get("entity_f1", {})
        if "precision" in ef1:
            metrics["entity_precision"].append(ef1["precision"])
        if "recall" in ef1:
            metrics["entity_recall"].append(ef1["recall"])
        if "f1" in ef1:
            metrics["entity_f1"].append(ef1["f1"])

        ra = rm.get("relationship_accuracy", {})
        if "accuracy" in ra:
            metrics["relationship_accuracy"].append(ra["accuracy"])

        gpm = rm.get("graph_path_match", {})
        if "path_recall" in gpm and gpm.get("expected_path_count", 0) > 0:
            metrics["graph_path_accuracy"].append(
                1.0 if gpm["path_recall"] > 0 else 0.0
            )
            metrics["path_recall"].append(gpm["path_recall"])

        ha = rm.get("hop_accuracy", {})
        hop = ha.get("hop_level")
        correct = ha.get("correct")
        if hop is not None and correct is not None:
            key = f"{hop}_hop_accuracy"
            if key in metrics:
                metrics[key].append(1.0 if correct else 0.0)

        am = r.get("answer_metrics", {})
        if "score" in am:
            metrics["answer_correctness"].append(am["score"] / 100.0)

        if r["category"] == "negative":
            negative_total += 1
            answer = r.get("generated_answer", "").lower()
            unsupported_terms = [
                "not enough", "not available", "not provided",
                "not in the", "cannot determine", "not specified",
                "no information", "not mentioned", "not disclosed",
                "insufficient", "not found", "unavailable",
                "do not have", "don't have", "not covered",
                "no data", "not applicable", "outside the scope",
            ]
            answer_stripped = answer.strip()
            if (
                any(term in answer for term in unsupported_terms)
                or len(answer_stripped) == 0
                or answer_stripped in ("none", "n/a")
            ):
                negative_correct += 1
                category_stats[cat]["correct"] += 1
        elif cat == "financial_fact":
            md = rm.get("metric_detection", {})
            ffa = rm.get("financial_fact_accuracy", {})
            am = r.get("answer_metrics", {})
            if (
                md.get("correct") is True
                and ffa.get("accuracy", 0) >= 0.5
                and am.get("score", 0) >= 50
            ):
                category_stats[cat]["correct"] += 1
        elif cat == "entity_graph":
            ef1 = rm.get("entity_f1", {})
            ra = rm.get("relationship_accuracy", {})
            if (
                ef1.get("recall", 0) >= 0.5
                and ra.get("accuracy", 0) >= 0.5
            ):
                category_stats[cat]["correct"] += 1
        elif cat == "single_hop_graph":
            ef1 = rm.get("entity_f1", {})
            ra = rm.get("relationship_accuracy", {})
            if (
                ef1.get("recall", 0) >= 0.5
                and ra.get("accuracy", 0) >= 0.5
            ):
                category_stats[cat]["correct"] += 1
        elif cat == "multi_hop_graph":
            gpm = rm.get("graph_path_match", {})
            if gpm.get("path_recall", 0) >= 0.5:
                category_stats[cat]["correct"] += 1

    def _avg(vals: List[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "metric_detection_accuracy": _avg(metrics["metric_detection"]),
        "financial_fact_accuracy": _avg(metrics["financial_fact_accuracy"]),
        "entity_precision": _avg(metrics["entity_precision"]),
        "entity_recall": _avg(metrics["entity_recall"]),
        "entity_f1": _avg(metrics["entity_f1"]),
        "relationship_accuracy": _avg(metrics["relationship_accuracy"]),
        "graph_path_accuracy": _avg(metrics["graph_path_accuracy"]),
        "path_recall": _avg(metrics["path_recall"]),
        "2_hop_accuracy": _avg(metrics["2_hop_accuracy"]),
        "3_hop_accuracy": _avg(metrics["3_hop_accuracy"]),
        "4_hop_accuracy": _avg(metrics["4_hop_accuracy"]),
        "answer_correctness": _avg(metrics["answer_correctness"]),
        "negative_question_accuracy": (
            round(negative_correct / negative_total, 4)
            if negative_total > 0
            else None
        ),
        "category_stats": category_stats,
        "total_questions": len(results),
    }


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(
    results: List[Dict[str, Any]],
    overall: Dict[str, Any],
) -> str:
    """Generate a markdown evaluation report."""
    lines = []
    lines.append("# GRAPH RAG EVALUATION REPORT")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Total questions: {overall['total_questions']}")
    lines.append("")

    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key in [
        "metric_detection_accuracy",
        "financial_fact_accuracy",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "relationship_accuracy",
        "graph_path_accuracy",
        "path_recall",
        "2_hop_accuracy",
        "3_hop_accuracy",
        "4_hop_accuracy",
        "answer_correctness",
        "negative_question_accuracy",
    ]:
        val = overall.get(key)
        if val is None:
            display = "N/A"
        elif isinstance(val, float):
            display = f"{val * 100:.1f}%"
        else:
            display = str(val)
        lines.append(f"| {key} | {display} |")

    lines.append("")
    lines.append("## Category Results")
    lines.append("")
    lines.append("| Category | Questions | Correct | Incorrect | Accuracy |")
    lines.append("|----------|-----------|---------|-----------|----------|")
    for cat, stats in overall.get("category_stats", {}).items():
        t = stats["total"]
        c = stats["correct"]
        incorrect = t - c
        acc = f"{c / t * 100:.1f}%" if t > 0 else "N/A"
        lines.append(f"| {cat} | {t} | {c} | {incorrect} | {acc} |")

    lines.append("")
    lines.append("## Failure Analysis")
    lines.append("")

    failed_count = 0
    for r in results:
        cat = r.get("category", "")
        stats = overall.get("category_stats", {}).get(cat, {})
        am = r.get("answer_metrics", {})
        score = am.get("score", -1)

        is_failure = False
        if cat == "negative":
            answer = r.get("generated_answer", "").lower()
            unsupported_terms = [
                "not enough", "not available", "not provided",
                "not in the", "cannot determine", "insufficient",
                "not specified", "no information", "not mentioned",
                "not disclosed", "not found", "unavailable",
                "no data", "not applicable", "outside the scope",
            ]
            answer_stripped = answer.strip()
            passed = (
                any(t in answer for t in unsupported_terms)
                or len(answer_stripped) == 0
                or answer_stripped in ("none", "n/a")
            )
            is_failure = not passed
        elif score >= 0:
            is_failure = score < 50
        else:
            rm = r.get("retrieval_metrics", {})
            md = rm.get("metric_detection", {})
            if "correct" in md:
                is_failure = not md["correct"]
            else:
                gpm = rm.get("graph_path_match", {})
                if gpm.get("expected_path_count", 0) > 0:
                    is_failure = gpm.get("path_recall", 0) <= 0
                else:
                    ef1 = rm.get("entity_f1", {})
                    if "recall" in ef1:
                        is_failure = ef1.get("recall", 0) <= 0.4

        if is_failure:
            failed_count += 1
            lines.append(f"### {r['id']}: {r['question']}")
            lines.append("")
            lines.append(f"- **Category:** {cat}")
            lines.append(f"- **Generated Answer:** {r.get('generated_answer', '(none)')[:200]}")

            rm = r.get("retrieval_metrics", {})
            md = rm.get("metric_detection", {})
            if "detected" in md:
                lines.append(f"- **Detected Metric:** {md['detected']} (expected: {md['expected']})")

            ffa = rm.get("financial_fact_accuracy", {})
            if "matched_count" in ffa:
                lines.append(f"- **Facts Matched:** {ffa['matched_count']}/{ffa['expected_count']}")

            ef1 = rm.get("entity_f1", {})
            if "f1" in ef1:
                lines.append(f"- **Entity F1:** {ef1['f1']} (correct: {ef1.get('correct_entities', [])})")

            ra = rm.get("relationship_accuracy", {})
            if "missing_rels" in ra:
                lines.append(f"- **Missing Relationships:** {ra['missing_rels']}")

            gpm = rm.get("graph_path_match", {})
            if "expected_path_count" in gpm and gpm["expected_path_count"] > 0:
                lines.append(f"- **Paths Matched:** {gpm['exact_match_count']}/{gpm['expected_path_count']}")

            lines.append("")

    if failed_count == 0:
        lines.append("No failures detected.")
    else:
        lines.append(f"**Total failures: {failed_count}**")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by evaluation/evaluate.py*")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LTIMindtree GraphRAG"
    )
    parser.add_argument(
        "--category",
        choices=[
            "financial_fact", "entity_graph",
            "single_hop_graph", "multi_hop_graph",
            "negative", "all",
        ],
        default="all",
        help="Filter by question category",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip Ollama answer generation (retrieval-only eval)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom dataset.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "results"),
        help="Output directory for results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GRAPH RAG EVALUATION")
    print("=" * 80)

    dataset = load_dataset(args.dataset)
    questions = dataset["questions"]

    if args.category != "all":
        questions = [
            q for q in questions if q["category"] == args.category
        ]

    if args.limit:
        questions = questions[: args.limit]

    print(f"\nQuestions to evaluate: {len(questions)}")
    print(f"Category filter: {args.category}")
    print(f"Skip generation: {args.skip_generation}")
    print()

    engine = None
    answer_generator = None

    try:
        from rag.query_engine import GraphQueryEngine
        engine = GraphQueryEngine()
    except Exception as exc:
        print(f"WARNING: Could not connect to Neo4j: {exc}")
        print("Running dataset-only evaluation (no live retrieval)")

    if not args.skip_generation:
        try:
            from rag.answer_generator import AnswerGenerator
            answer_generator = AnswerGenerator()
        except Exception:
            print("WARNING: Ollama not available. Skipping answer generation.")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i:2d}/{len(questions)}] {q['id']}: {q['question'][:70]}...")

        result = evaluate_question(
            q,
            engine=engine,
            answer_generator=answer_generator,
            skip_generation=args.skip_generation,
        )
        results.append(result)

        rm = result.get("retrieval_metrics", {})
        md = rm.get("metric_detection", {})
        ef1 = rm.get("entity_f1", {})
        gpm = rm.get("graph_path_match", {})

        parts = []
        if "correct" in md:
            parts.append(f"metric={'OK' if md['correct'] else 'FAIL'}")
        if "f1" in ef1:
            parts.append(f"entity_f1={ef1['f1']:.2f}")
        if gpm.get("expected_path_count", 0) > 0:
            parts.append(f"path_recall={gpm['path_recall']:.2f}")
        if result.get("error"):
            parts.append(f"ERROR={result['error'][:50]}")

        print(f"    {' | '.join(parts)}")

    if engine:
        try:
            engine.close()
        except Exception:
            pass

    print()
    print("=" * 80)
    print("COMPUTING OVERALL METRICS")
    print("=" * 80)

    overall = compute_overall_metrics(results)

    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 47)
    for key, val in overall.items():
        if key == "category_stats":
            continue
        if val is None:
            display = "N/A"
        elif isinstance(val, float):
            display = f"{val * 100:.1f}%"
        else:
            display = str(val)
        print(f"{key:<35} {display:>10}")

    print()
    print("Category Results:")
    for cat, stats in overall.get("category_stats", {}).items():
        t = stats["total"]
        c = stats["correct"]
        print(f"  {cat}: {c}/{t} correct ({c / t * 100:.0f}%)" if t else f"  {cat}: 0/0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"overall": overall, "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to: {results_path}")

    report = generate_report(results, overall)
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
