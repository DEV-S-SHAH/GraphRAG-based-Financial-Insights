#!/usr/bin/env python3
"""
Comprehensive question test suite for the LTIMindtree Multi-Year GraphRAG.

Runs all suggested questions through the full pipeline and captures answers.
Outputs a structured report for quality review.
"""

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.app import answer_question
from rag.context_fusion import build_context
from rag.hybrid_retriever import HybridRetriever


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    answer: str = ""
    question_type: str = ""
    fiscal_year_context: Dict = field(default_factory=dict)
    graph_facts: int = 0
    graph_entities: int = 0
    graph_paths: int = 0
    vector_chunks: int = 0
    time_s: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# All questions organized by category
# ---------------------------------------------------------------------------

QUESTIONS = [
    # --- Cross-year synthesis ---
    ("S01", "synthesis",
     "Which fiscal year had the best EBITDA margin and why might that be?"),
    ("S02", "synthesis",
     "Is LTIMindtree's profitability improving or declining over the three years?"),
    ("S03", "synthesis",
     "How does revenue growth compare to employee headcount growth?"),

    # --- Graph-path reasoning ---
    ("G01", "graph_reasoning",
     "How does inflation indirectly reach EBITDA margin through the entity graph?"),
    ("G02", "graph_reasoning",
     "What is the longest chain from AI to a financial metric?"),
    ("G03", "graph_reasoning",
     "If cost optimization stops, what entities downstream are affected?"),

    # --- Narrative + graph hybrid ---
    ("H01", "hybrid_narrative",
     "What management priorities mentioned in FY2024-25 actually show up as entity nodes in the graph?"),
    ("H02", "hybrid_narrative",
     "Which risks from FY2023-24 were resolved vs still present in FY2024-25?"),
    ("H03", "hybrid_narrative",
     "What programs did LTIMindtree scale up between FY2022-23 and FY2024-25?"),

    # --- Unanswerable / boundary ---
    ("U01", "unanswerable",
     "What was LTIMindtree's market cap during FY2024-25?"),
    ("U02", "unanswerable",
     "How much did LTIMindtree spend on acquisitions?"),
    ("U03", "unanswerable",
     "What is LTIMindtree's client concentration ratio?"),
    ("U04", "unanswerable",
     "What was the attrition rate in FY2023-24?"),

    # --- Opinion-elicitation ---
    ("O01", "opinion",
     "Is LTIMindtree a good investment?"),
    ("O02", "opinion",
     "Will LTIMindtree's revenue grow next year?"),
    ("O03", "opinion",
     "Should I buy LTIMindtree stock?"),
]


def assess_answer(qid: str, category: str, question: str, answer: str,
                   result: Dict) -> Dict[str, Any]:
    """
    Rule-based quality assessment for each answer.
    Returns a dict with checks and an overall score.
    """
    checks = []
    a = answer.lower()

    if category == "synthesis":
        checks.append(("mentions_financial_metric",
                        any(m in a for m in ["ebitda", "margin", "revenue",
                                              "profit", "headcount", "employee"])))
        checks.append(("mentions_years",
                        any(y in a for y in ["fy2022-23", "fy2023-24", "fy2024-25",
                                              "2022-23", "2023-24", "2024-25"])))
        checks.append(("has_some_values",
                        any(v in a for v in ["18.4", "18.0", "17.1",
                                              "331", "355", "380",
                                              "84,", "43,", "6.98", "91.74"])))
        checks.append(("has_reasoning",
                        any(w in a for w in ["because", "due to", "likely",
                                              "may have", "suggests", "could",
                                              "growth", "compare", "decline",
                                              "improving", "declining"])))
        checks.append(("no_hallucination",
                        "75,552" not in a and "fit4future" not in a))

    elif category == "graph_reasoning":
        checks.append(("mentions_entities",
                        any(e in a for e in ["inflation", "cost optimization", "ebitda",
                                              "margin", "ai", "entity"])))
        checks.append(("describes_path",
                        any(r in a for r in ["→", "->", "drives", "supports",
                                              "increases", "chain", "path",
                                              "relationship", "connection"])))
        checks.append(("distinguishes_evidence",
                        any(d in a for d in ["graph", "relationship", "indirect",
                                              "inferred", "connection"])))
        checks.append(("no_causation_claim",
                        not any(c in a for c in ["caused", "directly increased",
                                                  "directly caused"])))

    elif category == "hybrid_narrative":
        checks.append(("mentions_priorities_or_risks",
                        any(w in a for w in ["priority", "priorities", "risk",
                                              "risks", "program", "initiative",
                                              "management", "strategic"])))
        checks.append(("has_page_reference",
                        "page" in a or "p." in a or "p8" in a or "p9" in a or
                        "p12" in a or "p111" in a or "p72" in a))
        checks.append(("mentions_fiscal_year",
                        any(y in a for y in ["fy2022-23", "fy2023-24", "fy2024-25"])))
        checks.append(("no_hallucination",
                        "fit4future" not in a))

    elif category == "unanswerable":
        checks.append(("says_insufficient",
                        any(w in a for w in ["not enough", "not available",
                                              "not provided", "insufficient",
                                              "not in the reports", "cannot determine",
                                              "not specified", "no information",
                                              "not mentioned", "not disclosed"])))
        checks.append(("does_not_hallucinate",
                        not any(w in a for w in ["3.5 trillion", "$2.1 billion",
                                                  "25%", "34,000",
                                                  "market cap was"])))

    elif category == "opinion":
        checks.append(("refuses_to_advise",
                        any(w in a for w in ["not provide", "not give",
                                              "not suitable", "cannot advise",
                                              "should not", "not recommend",
                                              "not enough", "financial advisor",
                                              "do your own", "not a recommendation"])))
        checks.append(("no_price_prediction",
                        "will" not in a or "grow" not in a or "decline" not in a
                        or "prediction" not in a))

    passed = sum(1 for _, v in checks if v)
    total = len(checks) if checks else 1
    score = round(passed / total * 100)

    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "score": score,
    }


def run_all_questions():
    print("=" * 80)
    print("LTIMindtree GraphRAG — Comprehensive Question Test")
    print("=" * 80)
    print(f"Running {len(QUESTIONS)} questions through the full pipeline...")
    print()

    retriever = HybridRetriever()
    results = []

    for i, (qid, category, question) in enumerate(QUESTIONS, 1):
        print(f"[{i:2d}/{len(QUESTIONS)}] {qid} ({category})")
        print(f"  Q: {question}")
        t0 = time.time()

        try:
            result = answer_question(question, top_k=5)
            elapsed = time.time() - t0

            qr = QuestionResult(
                id=qid,
                category=category,
                question=question,
                answer=result.get("answer", ""),
                question_type=result.get("question_type", ""),
                fiscal_year_context=result.get("fiscal_year_context", {}),
                graph_facts=len(result.get("graph_context", {}).get("financial_facts", [])),
                graph_entities=len(result.get("graph_context", {}).get("entities", [])),
                graph_paths=len(result.get("graph_context", {}).get("semantic_paths", [])),
                vector_chunks=len(result.get("document_chunks", [])),
                time_s=round(elapsed, 1),
            )
        except Exception as exc:
            elapsed = time.time() - t0
            qr = QuestionResult(
                id=qid,
                category=category,
                question=question,
                error=str(exc),
                time_s=round(elapsed, 1),
            )

        # Assessment
        assessment = assess_answer(
            qid, category, question, qr.answer,
            {"question_type": qr.question_type}
        )
        qr_dict = asdict(qr)
        qr_dict["assessment"] = assessment
        results.append(qr_dict)

        status = "✓" if assessment["score"] >= 50 else "✗" if assessment["score"] > 0 else "?"
        print(f"  {status} [{assessment['score']}] {assessment['passed']}/{assessment['total']} checks | "
              f"{qr.time_s}s | facts={qr.graph_facts} paths={qr.graph_paths} chunks={qr.vector_chunks}")
        if qr.error:
            print(f"  ERROR: {qr.error}")
        print()

    retriever.close()

    # --- Summary ---
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    overall_pass = 0
    overall_total = 0
    for cat, cat_results in by_category.items():
        scores = [r["assessment"]["score"] for r in cat_results]
        avg = sum(scores) / len(scores) if scores else 0
        passed = sum(1 for s in scores if s >= 50)
        print(f"\n  {cat.upper()} ({passed}/{len(cat_results)} passed, avg={avg:.0f}%)")
        for r in cat_results:
            a = r["assessment"]
            mark = "✓" if a["score"] >= 50 else "✗"
            print(f"    {mark} [{a['score']:3d}%] {r['id']}: {r['question'][:70]}...")
            if r["error"]:
                print(f"        ERROR: {r['error']}")
            elif a["score"] < 50:
                failed = [name for name, ok in a["checks"] if not ok]
                print(f"        Failed: {', '.join(failed)}")
            overall_pass += (1 if a["score"] >= 50 else 0)
            overall_total += 1

    print()
    print(f"  OVERALL: {overall_pass}/{overall_total} questions passed "
          f"({overall_pass/overall_total*100:.0f}%)")
    print()

    # Save full report
    report_path = PROJECT_ROOT / "evaluation" / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Full report saved to: {report_path}")

    # Save human-readable answers
    txt_path = PROJECT_ROOT / "evaluation" / "test_answers.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            a = r["assessment"]
            f.write(f"{'='*80}\n")
            f.write(f"[{r['id']}] ({r['category']}) Score: {a['score']}%\n")
            f.write(f"Q: {r['question']}\n")
            f.write(f"Q-Type: {r['question_type']}\n")
            f.write(f"Fiscal Year Context: {r.get('fiscal_year_context', {})}\n")
            f.write(f"Graph: {r['graph_facts']} facts, {r['graph_entities']} entities, {r['graph_paths']} paths\n")
            f.write(f"Vector: {r['vector_chunks']} chunks | Time: {r['time_s']}s\n")
            f.write(f"{'-'*80}\n")
            if r["error"]:
                f.write(f"ERROR: {r['error']}\n")
            else:
                f.write(f"A: {r['answer']}\n")
            f.write(f"\nAssessment: {a['passed']}/{a['total']} checks passed\n")
            for name, ok in a["checks"]:
                f.write(f"  {'✓' if ok else '✗'} {name}\n")
            f.write("\n")
    print(f"  Human-readable answers saved to: {txt_path}")

    return results


if __name__ == "__main__":
    run_all_questions()
