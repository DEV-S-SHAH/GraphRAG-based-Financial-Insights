"""
Phase 5: Hybrid, source-grounded entity extraction (multi-year).

Extracts a controlled, grounded entity vocabulary across the three annual
reports. Unlike the legacy single-year extractor (which used a hand-crafted
pattern list that included entities NOT present in the reports, e.g. Fit4Future),
this extractor:

  - defines a curated vocabulary of entities that are actually material to the
    reports (strategy/theme, program/initiative, risk, capability/technology,
    vertical, geography)
  - verifies each entity name actually appears in a report (word-boundary
    matching, case-insensitive) before recording it
  - records the fiscal year(s) in which it appears plus representative source
    pages, so every entity is grounded
  - normalizes spelling/case variants

Output
------
Per-report entity files: data/processed/entities/{document_id}_entities.json
Merged vocabulary:        data/processed/entities/LTIM_entities_multiyear.json

Each entity record:
  entity_id, name, canonical_name, entity_type, fiscal_years, pages,
  source (representative page), confidence

Usage:
    python -m ingestion.entity_extractor_multi
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from ingestion.financial_extractor_multi import DOCS, load_parsed, page_text

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed" / "pdf"
OUTPUT_DIR = ROOT / "data" / "processed" / "entities"

COMPANY = "LTIMindtree Limited"
TICKER = "LTIM"

MAX_PAGES_RECORDED = 12  # cap distinct pages per report per entity

# ---------------------------------------------------------------------------
# GROUNDED ENTITY VOCABULARY
# canonical_name is the display name; aliases are matched in text.
# These were verified to appear in the real FY23/FY24/FY25 reports.
# ---------------------------------------------------------------------------
ENTITY_VOCAB: List[Dict[str, Any]] = [
    # ---- strategies / business themes ----
    {"canonical_name": "digital transformation", "entity_type": "business_theme",
     "aliases": ["digital transformation"]},
    {"canonical_name": "AI", "entity_type": "business_theme",
     "aliases": ["artificial intelligence", "AI ", "GenAI", "generative AI"]},
    {"canonical_name": "cost optimization", "entity_type": "business_theme",
     "aliases": ["cost optimization", "cost optimisation", "cost optimization initiatives"]},
    {"canonical_name": "operational efficiency", "entity_type": "business_theme",
     "aliases": ["operational efficiency"]},
    {"canonical_name": "productivity", "entity_type": "business_theme",
     "aliases": ["productivity", "productivity improvement"]},
    {"canonical_name": "merger synergy", "entity_type": "business_theme",
     "aliases": ["merger synergy", "merger synergies", "synergy"]},
    {"canonical_name": "profitable growth", "entity_type": "business_theme",
     "aliases": ["profitable growth"]},
    # ---- programs / initiatives ----
    {"canonical_name": "Large Deals program", "entity_type": "program",
     "aliases": ["large deals", "large deal", "deal wins"]},
    {"canonical_name": "GenAI initiative", "entity_type": "program",
     "aliases": ["GenAI", "generative AI initiative", "GenAI hackathon"]},
    # ---- risks ----
    {"canonical_name": "currency risk", "entity_type": "risk",
     "aliases": ["currency risk", "currency risks", "foreign exchange risk", "FX risk"]},
    {"canonical_name": "inflation", "entity_type": "risk",
     "aliases": ["inflation", "inflationary"]},
    {"canonical_name": "attrition", "entity_type": "risk",
     "aliases": ["attrition", "attrition rate", "voluntary attrition"]},
    {"canonical_name": "margin pressure", "entity_type": "risk",
     "aliases": ["margin pressure", "margin moderation", "ebitda margin pressure"]},
    {"canonical_name": "discretionary spending slowdown", "entity_type": "risk",
     "aliases": ["discretionary spend", "discretionary spending", "restrained client spending"]},
    {"canonical_name": "macroeconomic uncertainty", "entity_type": "risk",
     "aliases": ["macroeconomic", "macroeconomic uncertainty", "geopolitical"]},
    {"canonical_name": "cyber risk", "entity_type": "risk",
     "aliases": ["cybersecurity", "cyber security", "data breach", "cyber risk"]},
    # ---- capabilities / technologies ----
    {"canonical_name": "cloud", "entity_type": "capability",
     "aliases": ["cloud", "cloud computing"]},
    {"canonical_name": "data and analytics", "entity_type": "capability",
     "aliases": ["data and analytics", "data analytics"]},
    {"canonical_name": "cybersecurity", "entity_type": "capability",
     "aliases": ["cybersecurity", "cyber security"]},
    {"canonical_name": "digital engineering", "entity_type": "capability",
     "aliases": ["digital engineering"]},
    {"canonical_name": "quality engineering", "entity_type": "capability",
     "aliases": ["quality engineering", "quality engineering services"]},
    {"canonical_name": "generative AI", "entity_type": "capability",
     "aliases": ["generative AI", "GenAI", "Gen AI"]},
    {"canonical_name": "enterprise AI", "entity_type": "capability",
     "aliases": ["enterprise AI"]},
    # ---- verticals (industries served) ----
    {"canonical_name": "Banking and Financial Services", "entity_type": "vertical",
     "aliases": ["banking and financial services", "BFS", "banking, financial services"]},
    {"canonical_name": "Insurance", "entity_type": "vertical",
     "aliases": ["insurance"]},
    {"canonical_name": "Healthcare and Life Sciences", "entity_type": "vertical",
     "aliases": ["healthcare and life sciences", "life sciences"]},
    {"canonical_name": "Manufacturing", "entity_type": "vertical",
     "aliases": ["manufacturing"]},
    {"canonical_name": "Communications, Media and Entertainment", "entity_type": "vertical",
     "aliases": ["communications, media and entertainment", "communications media"]},
    {"canonical_name": "Energy and Utilities", "entity_type": "vertical",
     "aliases": ["energy and utilities", "utilities"]},
    {"canonical_name": "Technology, Media and Communications", "entity_type": "vertical",
     "aliases": ["technology, media and communications"]},
    # ---- geographies ----
    {"canonical_name": "North America", "entity_type": "geography",
     "aliases": ["north america"]},
    {"canonical_name": "Europe", "entity_type": "geography",
     "aliases": ["europe"]},
    {"canonical_name": "India", "entity_type": "geography",
     "aliases": ["india"]},
    {"canonical_name": "Rest of the World", "entity_type": "geography",
     "aliases": ["rest of the world", "rest of world"]},
]


def entity_id(entity_type: str, canonical: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", canonical.lower()).strip("_")
    return f"LTIM_{entity_type}_{slug}"


def _find_pages(document: Dict[str, Any], aliases: List[str]) -> List[int]:
    """Scan a parsed doc, return sorted pages where any alias matches."""
    found: Set[int] = set()
    for p in document["pages"]:
        text = "\n".join(b["text"] for b in p["blocks"])
        low = text.lower()
        for alias in aliases:
            a_low = alias.lower().strip()
            if not a_low:
                continue
            if re.search(r"\b" + re.escape(a_low) + r"\b", low):
                found.add(p["page"])
                break
    return sorted(found)


def extract_per_report(document: Dict[str, Any], fiscal_year: str) -> List[Dict[str, Any]]:
    document_id = "LTIM_" + fiscal_year.replace("-", "_") + "_Annual_Report"
    results = []
    for entry in ENTITY_VOCAB:
        canonical = entry["canonical_name"]
        etype = entry["entity_type"]
        pages = _find_pages(document, entry["aliases"])
        if not pages:
            continue
        results.append(
            {
                "entity_id": entity_id(etype, canonical),
                "name": canonical,
                "canonical_name": canonical,
                "entity_type": etype,
                "fiscal_year": fiscal_year,
                "pages": pages[:MAX_PAGES_RECORDED],
                "source": {
                    "document_id": document_id,
                    "page": pages[0],
                },
                "confidence": 1.0,
            }
        )
    return results


def extract_all() -> Dict[str, List[Dict[str, Any]]]:
    by_year: Dict[str, List[Dict[str, Any]]] = {}
    for fy in DOCS:
        doc = load_parsed(fy)
        by_year[fy] = extract_per_report(doc, fy)
    return by_year


def build_merged(by_year: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for fy, facets in by_year.items():
        for facet in facets:
            key = facet["entity_id"]
            if key not in merged:
                merged[key] = {
                    "entity_id": key,
                    "name": facet["name"],
                    "entity_type": facet["entity_type"],
                    "fiscal_years": [],
                    "pages_by_year": {},
                }
            record = merged[key]
            if fy not in record["fiscal_years"]:
                record["fiscal_years"].append(fy)
            record["pages_by_year"][fy] = facet["pages"]
    result = []
    for rec in merged.values():
        rec["fiscal_years"].sort()
        result.append(rec)
    result.sort(key=lambda r: (r["entity_type"], r["name"]))
    return result


def save(by_year: Dict[str, List[Dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fy, facets in by_year.items():
        document_id = "LTIM_" + fy.replace("-", "_") + "_Annual_Report"
        payload = {
            "document_id": document_id,
            "company": COMPANY,
            "ticker": TICKER,
            "fiscal_year": fy,
            "dataset_type": "narrative_entities",
            "entity_count": len(facets),
            "entities": facets,
        }
        with open(OUTPUT_DIR / f"{document_id}_entities.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    merged = build_merged(by_year)
    with open(OUTPUT_DIR / "LTIM_entities_multiyear.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "company": COMPANY,
                "ticker": TICKER,
                "dataset_type": "entities_multiyear",
                "entity_count": len(merged),
                "entities": merged,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    print("=" * 70)
    print("MULTI-YEAR ENTITY EXTRACTOR (grounded)")
    print("=" * 70)

    by_year = extract_all()
    for fy, facets in by_year.items():
        print(f"\n[{fy}] {len(facets)} entities")
        for e in facets:
            print(f"  {e['entity_type']:<16} {e['name']:<40} pages={e['pages'][:5]}")
    save(by_year)
    print("\nSaved entity files to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
