"""
Deterministic Financial Intelligence & Math Engine.

Ensures numerical calculations are performed programmatically in Python
rather than by LLM generation:
- YoY Growth: ((V_t - V_{t-1}) / V_{t-1}) * 100
- EBITDA Margin: (EBITDA / Revenue) * 100
- PAT / Net Margin: (PAT / Revenue) * 100
- Margin Delta: (Margin_t - Margin_{t-1}) in percentage points and basis points (bps)
- CAGR: ((V_final / V_initial) ** (1 / n) - 1) * 100

Enforces three-tier separation:
1. Reported Data (ground-truth facts from filings)
2. Calculated Metrics (Python math calculations)
3. LLM Interpretation (qualitative strategic explanation)
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple


def calculate_yoy_growth(current: float, prior: float) -> Optional[float]:
    """Calculate Year-on-Year growth rate as percentage."""
    if prior == 0 or prior is None or current is None:
        return None
    return round(((current - prior) / abs(prior)) * 100, 2)


def calculate_margin(numerator: float, denominator: float) -> Optional[float]:
    """Calculate margin as percentage (e.g., EBITDA / Revenue)."""
    if denominator == 0 or denominator is None or numerator is None:
        return None
    return round((numerator / denominator) * 100, 2)


def calculate_basis_points(margin_diff: float) -> int:
    """Convert margin difference in % to basis points (bps)."""
    return round(margin_diff * 100)


def calculate_cagr(start_val: float, end_val: float, periods: int) -> Optional[float]:
    """Calculate Compound Annual Growth Rate."""
    if start_val <= 0 or end_val <= 0 or periods <= 0:
        return None
    return round(((end_val / start_val) ** (1.0 / periods) - 1.0) * 100, 2)


class FinancialMathEngine:
    """Performs deterministic financial analysis on reported facts."""

    @staticmethod
    def analyze_multiyear_metric(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a metric across fiscal years, calculating YoY growth and trends.
        """
        if not facts:
            return {"facts": [], "yoy_growth": {}, "trend": "No data"}

        # Sort facts by period
        sorted_facts = sorted(facts, key=lambda f: f.get("period", ""))
        analysis = {
            "metric": sorted_facts[0].get("metric", ""),
            "unit": sorted_facts[0].get("unit", ""),
            "yearly_data": [],
            "cagr": None,
            "overall_change_pct": None,
        }

        prev_val = None
        first_val = None
        last_val = None

        for idx, f in enumerate(sorted_facts):
            val = float(f.get("value", 0))
            period = f.get("period", "")
            page = f.get("page")
            doc = f.get("document_id", "")

            yoy = calculate_yoy_growth(val, prev_val) if prev_val is not None else None
            
            entry = {
                "period": period,
                "value": val,
                "unit": f.get("unit", ""),
                "page": page,
                "document_id": doc,
                "yoy_growth_pct": yoy,
                "yoy_display": f"{yoy:+.2f}%" if yoy is not None else "N/A (base year)",
            }
            analysis["yearly_data"].append(entry)

            if first_val is None:
                first_val = val
            last_val = val
            prev_val = val

        num_years = len(sorted_facts) - 1
        if num_years >= 1 and first_val and last_val:
            analysis["cagr"] = calculate_cagr(first_val, last_val, num_years)
            analysis["overall_change_pct"] = calculate_yoy_growth(last_val, first_val)

        return analysis

    @staticmethod
    def compute_margin_trends(
        revenue_facts: List[Dict[str, Any]],
        profit_facts: List[Dict[str, Any]],
        metric_name: str = "EBITDA Margin",
    ) -> List[Dict[str, Any]]:
        """
        Compute margins across periods given revenue and profit facts.
        """
        rev_by_period = {f.get("period"): float(f.get("value", 0)) for f in revenue_facts}
        margin_results = []
        prev_margin = None

        sorted_profit = sorted(profit_facts, key=lambda f: f.get("period", ""))
        for f in sorted_profit:
            period = f.get("period")
            profit_val = float(f.get("value", 0))
            rev_val = rev_by_period.get(period)

            if rev_val and rev_val > 0:
                margin = calculate_margin(profit_val, rev_val)
                margin_delta = round(margin - prev_margin, 2) if prev_margin is not None else None
                bps = calculate_basis_points(margin_delta) if margin_delta is not None else None

                margin_results.append({
                    "period": period,
                    "metric": metric_name,
                    "numerator_value": profit_val,
                    "revenue_value": rev_val,
                    "margin_pct": margin,
                    "margin_delta_pct": margin_delta,
                    "basis_points_change": bps,
                    "display": f"{margin:.2f}%",
                    "delta_display": f"{margin_delta:+.2f}% ({bps:+d} bps)" if margin_delta is not None else "N/A",
                })
                prev_margin = margin

        return margin_results

    @staticmethod
    def build_financial_intelligence_report(
        revenue_facts: List[Dict[str, Any]],
        ebitda_facts: List[Dict[str, Any]],
        pat_facts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Comprehensive multi-year financial health analysis.
        """
        rev_analysis = FinancialMathEngine.analyze_multiyear_metric(revenue_facts)
        ebitda_analysis = FinancialMathEngine.analyze_multiyear_metric(ebitda_facts)
        pat_analysis = FinancialMathEngine.analyze_multiyear_metric(pat_facts)
        ebitda_margins = FinancialMathEngine.compute_margin_trends(revenue_facts, ebitda_facts, "EBITDA Margin")
        pat_margins = FinancialMathEngine.compute_margin_trends(revenue_facts, pat_facts, "PAT Margin")

        return {
            "revenue_analysis": rev_analysis,
            "ebitda_analysis": ebitda_analysis,
            "pat_analysis": pat_analysis,
            "ebitda_margins": ebitda_margins,
            "pat_margins": pat_margins,
        }

    @staticmethod
    def format_as_markdown_table(intelligence: Dict[str, Any]) -> str:
        """Format calculations into a crisp Markdown table for UI / prompts."""
        rev_data = {e["period"]: e for e in intelligence["revenue_analysis"].get("yearly_data", [])}
        ebitda_data = {e["period"]: e for e in intelligence["ebitda_analysis"].get("yearly_data", [])}
        pat_data = {e["period"]: e for e in intelligence["pat_analysis"].get("yearly_data", [])}
        ebitda_m = {e["period"]: e for e in intelligence.get("ebitda_margins", [])}
        pat_m = {e["period"]: e for e in intelligence.get("pat_margins", [])}

        periods = sorted(list(set(list(rev_data.keys()) + list(ebitda_data.keys()) + list(pat_data.keys()))))
        if not periods:
            return "No multi-year financial metrics available."

        lines = [
            "| Fiscal Year | Revenue (INR M) | Revenue YoY | EBITDA (INR M) | EBITDA Margin | PAT (INR M) | PAT Margin |",
            "|---|---|---|---|---|---|---|",
        ]

        for p in periods:
            r = rev_data.get(p, {})
            r_val = f"{r.get('value', 0):,.0f}" if "value" in r else "-"
            r_yoy = r.get("yoy_display", "-")

            e = ebitda_data.get(p, {})
            e_val = f"{e.get('value', 0):,.0f}" if "value" in e else "-"
            em = ebitda_m.get(p, {}).get("display", "-")

            pt = pat_data.get(p, {})
            pt_val = f"{pt.get('value', 0):,.0f}" if "value" in pt else "-"
            pm = pat_m.get(p, {}).get("display", "-")

            lines.append(f"| **{p}** | {r_val} | {r_yoy} | {e_val} | {em} | {pt_val} | {pm} |")

        return "\n".join(lines)
