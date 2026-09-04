"""
Query Router: deterministic question classifier.

Classifies questions into:
- financial   (structured financial facts)
- graph       (entity/relationship queries)
- narrative   (document/narrative queries)
- hybrid      (complex / multi-hop, use both)

Uses keyword + metric + entity matching. No LLM needed.
"""

import re
import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# METRIC KEYWORDS
# ============================================================

FINANCIAL_METRIC_WORDS = [
    "ebitda",
    "earnings",
    "revenue",
    "turnover",
    "sales",
    "profit",
    "pat",
    "margin",
    "eps",
    "earnings per share",
    "market cap",
    "market capitalization",
    "net worth",
    "dividend",
    "employee",
    "roe",
    "return on equity",
    "cash and investments",
    "current ratio",
    "order inflow",
    "order book",
    "cash flow",
    "net income",
    "net profit",
    "gross margin",
    "working capital",
    "debt",
    "assets",
    "liabilities",
    "roce",
    "balance sheet",
    "cost of goods",
    "opex",
    "capex",
    "capital expenditure",
]


# ============================================================
# GRAPH / RELATIONSHIP KEYWORDS
# ============================================================

GRAPH_KEYWORDS = [
    "relationship between",
    "trace",
    "path",
    "chain",
    "connect",
    "connection",
    "linked",
    "links",
    "influences",
    "impact",
    "affect",
    "improves",
    "drives",
    "supports",
    "enables",
    "from",
    "to",
    "how does",
    "role of",
    "what is the role",
    "relation",
]


# ============================================================
# NARRATIVE / DOCUMENT KEYWORDS
# ============================================================

NARRATIVE_KEYWORDS = [
    "said",
    "say",
    "discuss",
    "discussed",
    "strategic priorities",
    "priorities",
    "strategy",
    "management believes",
    "management said",
    "according to the report",
    "according to management",
    "what does the report",
    "what did",
    "major challenges",
    "risks",
    "outlook",
    "guidance",
    "highlight",
    "emphasized",
    "narrative",
    "paragraph",
    "text",
    "document says",
    "annual report says",
    "company's approach",
    "initiatives",
    "programs",
    "programmes",
    "plans",
    "digital transformation",
    "artificial intelligence",
    "ai",
    "expansion plans",
    "capital allocation",
    "investor presentation",
]


# ============================================================
# ENTITY-PATH DETECTION (deterministic)
# ============================================================

MULTI_HOP_KEYWORDS = [
    "trace every",
    "trace",
    "reach",
    "eventually",
    "multi-hop",
    "step by step",
    "hop by hop",
    "chain of",
    "path from",
    "from ... to",
    "how does ... affect",
    "how did ... contribute",
]


def classify_question(
    question: str,
    detected_metric: Optional[str] = None,
    entities: Optional[List[str]] = None,
    force: Optional[Literal[
        "financial", "graph", "narrative", "hybrid"
    ]] = None,
) -> str:
    """
    Deterministically classify a question into:
    financial, graph, narrative, or hybrid.

    Priority: financial > graph > narrative > hybrid
    """

    if force:
        return force

    q = question.lower().strip()

    # ---------- FINANCIAL ----------
    # Direct metric mention, no relationship language
    has_metric = detected_metric is not None

    has_relationship_lang = any(
        kw in q for kw in [
            "how does", "how did", "relationship",
            "affect", "impact", "trace", "connect",
            "chain", "path",
        ]
    )

    has_narrative_lang = any(
        kw in q
        for kw in [
            "said", "discuss", "strategic",
            "priorities", "according to",
            "what did", "outlook",
        ]
    )

    if not has_relationship_lang and not has_narrative_lang:

        metric_words = [
            kw
            for kw in FINANCIAL_METRIC_WORDS
            if kw in q
        ]

        if has_metric or metric_words:
            return "financial"

    # ---------- HYBRID (risks/impact on financial metrics) ----------
    # e.g. "What risks could affect revenue and profitability?"
    mentions_risk = any(
        kw in q for kw in ["risk", "challenge", "threat",
                           "exposure", "vulnerab"]
    )

    if has_metric and mentions_risk:
        return "hybrid"

    # ---------- GRAPH ----------
    if has_relationship_lang and entities:
        return "graph"

    if any(kw in q for kw in GRAPH_KEYWORDS) and entities:
        return "graph"

    # Check known entity names
    if entities and (
        "trace" in q
        or "path" in q
        or "relationship" in q
        or "affect" in q
        or "impact" in q
    ):
        return "graph"

    # ---------- NARRATIVE ----------
    narrative_hits = [
        kw for kw in NARRATIVE_KEYWORDS if kw in q
    ]

    if narrative_hits and not has_metric:
        return "narrative"

    if "strategic priorities" in q.lower():
        return "narrative"

    if "risks" in q.lower() and "revenue" not in q.lower():
        return "narrative"

    # ---------- MULTI-HOP / COMPLEX ----------
    complex_flags = [
        "contribute",
        "evidence",
        "both",
        "as well as",
        "and what",
        "evaluate",
        "assess",
        "what evidence",
    ]

    if any(kw in q for kw in complex_flags):
        return "hybrid"

    if has_metric and has_relationship_lang:
        return "graph"

    # ---------- DEFAULT ----------
    # If entity-like language present but no clear class,
    # default to hybrid to be safe.
    return "hybrid"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_questions = [
        "What was LTM's EBITDA?",
        "What was LTM's EBITDA margin?",
        "What was LTM's revenue?",
        "How does cost optimization affect EBITDA margin?",
        "What is the relationship between AI platforms and digital transformation?",
        "Trace the path from Fit4Future to EBITDA margin.",
        "What were LTM's major strategic priorities?",
        "What did management say about AI?",
        "What risks could affect LTM's revenue and profitability?",
        "How did Fit4Future contribute to operational efficiency and what evidence exists regarding EBITDA margin?",
    ]

    for q in test_questions:
        result = classify_question(q)
        print(f"{result:<10} | {q}")
