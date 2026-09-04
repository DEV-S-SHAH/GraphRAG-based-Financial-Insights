"""
Fiscal-year awareness helper for the multi-year pipeline.

Detects which fiscal year(s) a user's question refers to and whether the
question is comparative (YoY / trend / across years). Used by the hybrid
retriever and context fusion so that multi-year answers are scoped correctly.

Supported fiscal years:
    FY2022-23, FY2023-24, FY2024-25, FY2025-26

Year tokens are matched leniently, e.g.:
    "FY2024-25", "FY24", "FY2025", "FY 2024-25", "2024-25",
    "fiscal year 2023-24", "FY23", "FY23-24" (normalised), etc.
"""

import re
from typing import Dict, List, Optional, Tuple

FISCAL_YEARS = ["FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"]

# Binding end year for each fiscal year: end -> label
_END_YEAR_TO_LABEL = {
    2023: "FY2022-23",
    2024: "FY2023-24",
    2025: "FY2024-25",
    2026: "FY2025-26",
    "23": "FY2022-23",
    "24": "FY2023-24",
    "25": "FY2024-25",
    "26": "FY2025-26",
}

# fuzzy end-year tokens -> label (e.g. "FY24" -> FY2023-24)
_SHORT_YEAR_TO_LABEL = {
    "23": "FY2022-23",
    "24": "FY2023-24",
    "25": "FY2024-25",
    "26": "FY2025-26",
}

COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bvs\.\b",
    r"\byear[ -]?on[ -]?year\b",
    r"\byoy\b",
    r"\bcompared to\b",
    r"\bcompared with\b",
    r"\bdifference\b",
    r"\btrend\b",
    r"\bgrowth\b",
    r"\bchanged?\b",
    r"\bincreased\b",
    r"\bdecreased\b",
    r"\bas well as\b",
    r"\bacross (the )?(fiscal )?years\b",
    r"\bover the (last|past) (three|3|several)\b",
    r"\bfrom .* to\b",
    r"\band\b.*\bvs\b",
]

def _normalise_year_token(token: str) -> Optional[str]:
    t = re.sub(r"\s+", "", str(token)).lower().replace("fy", "")
    if not t:
        return None
    # e.g. "24" -> FY2023-24, "2024" -> FY2023-24
    if t in _SHORT_YEAR_TO_LABEL:
        return _SHORT_YEAR_TO_LABEL[t]
    if t.isdigit():
        y = int(t)
        if 2022 <= y <= 2026:
            return _END_YEAR_TO_LABEL.get(y)
    return None


# Match an explicit fiscal-year range as ONE token, e.g. "FY2024-25",
# "fiscal year 2023-24", "FY24-25", "2024-25".
_RANGE_RE = re.compile(
    r"""
    (?:
        FY\s*
        | fiscal\s+year\s*
    )?
    (?P<s>20\d{2}|2[2-6])
    \s*[-–/\s]\s*
    (?P<e>20\d{2}|2[2-6])
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Match a single fiscal-year token NOT preceded by a range start, e.g.
# "FY24", "FY2024", "2023", "FY25". Uses a negative lookbehind/awareness by
# operating on text with ranges already removed.
_SINGLE_RE = re.compile(
    r"\bFY\s*(?P<tok>20\d{2}|2[2-6])\b"
    r"|\b(?P<tok2>20\d{2}|2[2-6])\b",
    re.IGNORECASE,
)


def detect_fiscal_years(question: str) -> List[str]:
    """Return the ordered set of canonical FY labels a question mentions."""
    q = question or ""
    labels: List[str] = []
    seen = set()

    def add(label: Optional[str]) -> None:
        if label and label not in seen:
            seen.add(label)
            labels.append(label)

    # 1. Explicit ranges -> map end year to label; remove matched spans.
    remaining = _RANGE_RE.sub(lambda m: "", q)

    for m in _RANGE_RE.finditer(q):
        end = m.group("e")
        if end.isdigit():
            if len(end) == 2:
                add(_SHORT_YEAR_TO_LABEL.get(end))
            elif len(end) == 4:
                y = int(end)
                add(_END_YEAR_TO_LABEL.get(y))

    # 2. Single tokens in the leftover text.
    for m in _SINGLE_RE.finditer(remaining):
        token = m.group("tok") or m.group("tok2")
        if token:
            add(_normalise_year_token(token))

    return labels


def detect_comparison(question: str) -> bool:
    q = question or ""
    ql = q.lower()
    return any(re.search(p, ql) for p in COMPARISON_PATTERNS)


def fiscal_year_context(question: str) -> Dict[str, object]:
    """
    Return a small context dict used by retrieval/fusion:

        {
          "detected_years": [...],
          "comparison": bool,
          "scope_note": "..."
        }
    """
    years = detect_fiscal_years(question)
    comparison = detect_comparison(question)

    if years and comparison:
        note = (
            f"The question mentions {', '.join(years)} and asks for a "
            "comparison / trend across fiscal years."
        )
    elif years:
        note = (
            f"The question mentions fiscal year(s): {', '.join(years)}. "
            "Focus the answer on this year's data."
        )
    elif comparison:
        note = (
            "The question asks for a comparison or trend across fiscal "
            "years; present values for the available years (FY2022-23, "
            "FY2023-24, FY2024-25, FY2025-26)."
        )
    else:
        note = (
            "No specific fiscal year was mentioned; report values for the "
            "available fiscal years (FY2022-23, FY2023-24, FY2024-25, "
            "FY2025-26) and note which year each value is from."
        )

    return {
        "detected_years": years,
        "comparison": comparison,
        "scope_note": note,
        "available_fiscal_years": FISCAL_YEARS,
    }


if __name__ == "__main__":
    tests = [
        "What was revenue in FY2024-25?",
        "EBITDA margin for FY23",
        "Compare revenue FY24 vs FY25",
        "What was revenue?",
        "Revenue growth over the last three years",
        "How did revenue change from FY2022-23 to FY2024-25?",
        "What was EBITDA margin in fiscal year 2023-24?",
        "Show EPS for FY2025",
        "Which year had the highest PAT?",
    ]
    for t in tests:
        fc = fiscal_year_context(t)
        print(f"{fc['detected_years']!r:<15} comp={fc['comparison']!s:<5} | {t}")
