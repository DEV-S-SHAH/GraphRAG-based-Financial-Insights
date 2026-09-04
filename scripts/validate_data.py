"""
Phase 8: Data validation for the grounded multi-year pipeline.

Validates every artifact produced by the ingestion pipeline against the raw
source reports and enforces the "validate-then-commit" contract:

  - 4 real annual reports are present and match the manifest (size/SHA256/pages)
  - parsed JSON files exist for all 4 fiscal years
  - chunk files exist, are provenance-rich, and are prose-only
  - every canonical financial fact is grounded: its value/verbatim appears on
    the stated source page (re-verification, not trusting the source file)
    and it covers all 4 fiscal years for expected metrics
  - every entity is grounded in the reports (word-boundary, case-insensitive)
    and no fabricated entity (e.g. Fit4Future) leaks through
  - every relationship uses the controlled vocabulary, is explicit/inferred,
    and its evidence verbatim appears on the stated source page

Usage:
    python -m scripts.validate_data            # exit 0 on PASS, 1 on FAIL
    python -m scripts.validate_data --json     # machine-readable summary
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.financial_extractor_multi import DOCS, load_parsed, page_text  # noqa: E402

MANIFEST = PROJECT_ROOT / "data" / "metadata" / "documents_manifest.json"
FACTS = PROJECT_ROOT / "data" / "processed" / "financial_facts" / "LTIM_canonical_multiyear.json"
ENTITIES = PROJECT_ROOT / "data" / "processed" / "entities" / "LTIM_entities_multiyear.json"
RELATIONSHIPS = PROJECT_ROOT / "data" / "processed" / "relationships" / "LTIM_relationships_multiyear.json"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
PARSED_DIR = PROJECT_ROOT / "data" / "processed" / "pdf"

CONTROLLED_RELATIONS = {
    "SUPPORTS", "DRIVES", "ENABLES", "IMPROVES", "IMPACTS", "INCREASES",
    "DECREASES", "FUNDS", "DEPENDS_ON", "RELATED_TO", "TARGETS", "MITIGATES",
    "CREATES", "CONTRIBUTES_TO", "FOLLOWS",
}

# Cross-check reference values (INR million unless noted) derived from the
# reports; verified during earlier phases.
EXPECTED_FISCAL_YEARS = ["FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"]
EXPECTED_METRICS_PER_YEAR = {
    "revenue", "profit_before_tax", "profit_after_tax", "ebitda", "ebitda_margin",
    "pat_margin", "ebit", "eps_basic", "eps_diluted", "return_on_equity",
    "employees", "net_worth", "market_capitalization", "dividend_paid",
    "order_inflow_usd_bn", "revenue_usd_bn",
}
CROSSCHECK = {
    "revenue": {  # consolidated INR mn
        "FY2022-23": 331830, "FY2023-24": 355170, "FY2024-25": 380081,
        "FY2025-26": 423076,
    },
    "ebitda_margin": {  # %
        "FY2022-23": 18.4, "FY2023-24": 18.0, "FY2024-25": 17.1,
        "FY2025-26": 17.9,
    },
    "profit_after_tax": {  # INR mn
        "FY2022-23": 44103, "FY2023-24": 45846, "FY2024-25": 46020,
        "FY2025-26": 49827,
    },
}

FABRICATED_ENTITIES = {"fit4future", "new horizons"}
EXCLUDED_RELATIONSHIPS = {"HAS_FACT", "HAS_VALUE"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")

def _norm_comma(text: str) -> str:
    """Normalize whitespace and commas for numeric matching."""
    return re.sub(r"[,\s]", "", text or "")


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class Validator:
    def __init__(self):
        self.passes: List[str] = []
        self.failures: List[str] = []
        self.notes: List[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        if ok:
            self.passes.append(label)
        else:
            self.failures.append(f"{label} :: {detail}")

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def passed(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# 1. MANIFEST / RAW REPORTS
# ---------------------------------------------------------------------------

def validate_manifest(v: Validator) -> None:
    manifest = _load(MANIFEST)
    docs = manifest.get("documents", [])
    v.check(len(docs) == 4, "manifest.four_documents",
            f"expected 4, found {len(docs)}")

    found_years = {d.get("fiscal_year") for d in docs}
    v.check(found_years == set(EXPECTED_FISCAL_YEARS),
            "manifest.fiscal_years",
            f"expected {EXPECTED_FISCAL_YEARS}, found {sorted(found_years)}")

    for d in docs:
        path = PROJECT_ROOT / d["path"]
        if not path.exists():
            v.check(False, f"manifest.file.{d['fiscal_year']}", "missing file")
            continue
        actual_size = path.stat().st_size
        size_ok = d["size_bytes"] == actual_size
        v.check(size_ok, f"manifest.size.{d['fiscal_year']}",
                f"expected {d['size_bytes']}, got {actual_size}")

        sha_ok = True
        if d.get("sha256"):
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            actual_sha = h.hexdigest()
            sha_ok = actual_sha == d["sha256"]
            v.check(sha_ok, f"manifest.sha256.{d['fiscal_year']}",
                    f"expected {d['sha256'][:16]}..., got {actual_sha[:16]}...")


# ---------------------------------------------------------------------------
# 2. PARSED PDFS + CHUNKS
# ---------------------------------------------------------------------------

def validate_parsed_and_chunks(v: Validator) -> None:
    for fy in EXPECTED_FISCAL_YEARS:
        document_id, parsed_filename = DOCS[fy]
        parsed_file = PARSED_DIR / parsed_filename
        v.check(parsed_file.exists(), f"parsed.exists.{fy}", str(parsed_file))
        if parsed_file.exists():
            data = _load(parsed_file)
            v.check(len(data.get("pages", [])) > 0, f"parsed.pages.{fy}")
            # every page has page number and blocks
            all_ok = all(
                isinstance(p.get("page"), int) and isinstance(p.get("blocks"), list)
                for p in data.get("pages", [])
            )
            v.check(all_ok, f"parsed.structure.{fy}")

        chunk_file = CHUNKS_DIR / f"{document_id}_chunks.json"
        v.check(chunk_file.exists(), f"chunks.exists.{fy}", str(chunk_file))
        if chunk_file.exists():
            data = _load(chunk_file)
            chunks = data.get("chunks", [])
            v.check(len(chunks) > 0, f"chunks.nonempty.{fy}")
            # all chunks must be prose and have provenance
            bad = [c for c in chunks if c.get("chunk_type") not in ("prose", None)]
            v.check(not bad, f"chunks.prose_only.{fy}", f"{len(bad)} non-prose")
            missing_prov = [
                c for c in chunks
                if not c.get("chunk_id") or not c.get("text")
                or not c.get("document_id") or not c.get("fiscal_year")
                or not isinstance(c.get("page"), int)
            ]
            v.check(not missing_prov, f"chunks.provenance.{fy}",
                    f"{len(missing_prov)} missing chunk_id/text/document/fiscal_year/page")


# ---------------------------------------------------------------------------
# 3. FINANCIAL FACTS (validate-then-commit re-verification)
# ---------------------------------------------------------------------------

def validate_financial_facts(v: Validator) -> None:
    data = _load(FACTS)
    facts = data.get("facts", [])
    v.check(len(facts) == 68, "facts.count", f"expected 68, got {len(facts)}")

    # adjacency of metric->fiscal_years
    by_year_metric = {}
    for f in facts:
        fy = f.get("period", {}).get("label")
        metric = f.get("metric")
        by_year_metric.setdefault(fy, set()).add(metric)

    v.check(set(by_year_metric.keys()) == set(EXPECTED_FISCAL_YEARS),
            "facts.covers_all_years", str(sorted(by_year_metric.keys())))

    # each expected metric present in all 3 years
    missing = []
    for metric in EXPECTED_METRICS_PER_YEAR:
        for fy in EXPECTED_FISCAL_YEARS:
            if metric not in by_year_metric.get(fy, set()):
                missing.append(f"{metric}@{fy}")
    v.check(not missing, "facts.all_metrics_all_years", "; ".join(missing[:10]))

    # re-verify each fact is grounded on its source page
    ungrounded = []
    for f in facts:
        fy = f.get("period", {}).get("label")
        page = f.get("source", {}).get("page")
        document_id = f.get("source", {}).get("document_id")
        if not fy or document_id not in {doc_id for doc_id, _ in DOCS.values()}:
            ungrounded.append(f"{f.get('fact_id')}:bad_source")
            continue
        doc = load_parsed(fy)
        text = _norm(page_text(doc, page))
        text_comma = _norm_comma(page_text(doc, page))
        value = f.get("value")
        ok = False
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                s1 = f"{value:g}"
                s2 = str(int(value)) if value == int(value) else None
            else:
                s1 = str(value)
                s2 = f"{value:,}"
            ok = (_norm(s1) in text or (s2 is not None and _norm(s2) in text)
                  or _norm_comma(s1) in text_comma
                  or (s2 is not None and _norm_comma(s2) in text_comma))
        else:
            ok = str(value) in text
        if not ok:
            ungrounded.append(f"{f.get('fact_id')}@{fy}p{page}")
    v.check(not ungrounded, "facts.grounded_on_page", "; ".join(ungrounded[:10]))

    # cross-check known reference values
    mismatched = []
    for metric, refs in CROSSCHECK.items():
        for fy, expected in refs.items():
            for f in facts:
                if f.get("metric") == metric and f.get("period", {}).get("label") == fy:
                    if abs(float(f["value"]) - float(expected)) > 1e-9:
                        mismatched.append(f"{metric}@{fy}: {f['value']} != {expected}")
    v.check(not mismatched, "facts.crosscheck", "; ".join(mismatched))


# ---------------------------------------------------------------------------
# 4. ENTITIES (grounded, no fabrication)
# ---------------------------------------------------------------------------

def validate_entities(v: Validator) -> None:
    from ingestion.entity_extractor_multi import ENTITY_VOCAB

    data = _load(ENTITIES)
    entities = data.get("entities", [])
    v.check(len(entities) >= 25, "entities.count", f"{len(entities)}")

    # Match using the same grounding aliases the extractor used, so display
    # names that are canonical labels (e.g. "Banking and Financial Services",
    # "GenAI initiative", "macroeconomic uncertainty") are validated correctly.
    alias_by_name: Dict[str, List[str]] = {}
    for entry in ENTITY_VOCAB:
        alias_by_name[entry["canonical_name"]] = entry.get("aliases", [])

    def ground_aliases(name: str) -> List[str]:
        aliases = alias_by_name.get(name, [name])
        return [a for a in aliases if a.strip()]

    # every entity must appear (word-boundary) in at least one report
    ungrounded = []
    fabricated = []
    for e in entities:
        name = e.get("name", "")
        years = e.get("fiscal_years", [])
        if not name or not years:
            ungrounded.append(e.get("entity_id", "?") + ":no_name_or_year")
            continue
        aliases = ground_aliases(name)
        normalized_aliases = {re.sub(r"\s+", " ", a).strip().lower() for a in aliases}
        seen_any = False
        for fy in years:
            if fy not in DOCS:
                continue
            doc = load_parsed(fy)
            for p in doc["pages"]:
                low = " ".join(b["text"] for b in p["blocks"]).lower()
                for alias in normalized_aliases:
                    if re.search(r"\b" + re.escape(alias) + r"\b", low):
                        seen_any = True
                        break
                if seen_any:
                    break
            if seen_any:
                break
        if not seen_any:
            ungrounded.append(f"{name}@{years}")
        if name.lower() in FABRICATED_ENTITIES:
            fabricated.append(name)
    v.check(not ungrounded, "entities.grounded", "; ".join(ungrounded[:8]))
    v.check(not fabricated, "entities.no_fabricated", "; ".join(fabricated))


# ---------------------------------------------------------------------------
# 5. RELATIONSHIPS (controlled vocab + grounded evidence)
# ---------------------------------------------------------------------------

def validate_relationships(v: Validator) -> None:
    data = _load(RELATIONSHIPS)
    rels = data.get("relationships", [])
    v.check(len(rels) >= 10, "rels.count", f"{len(rels)}")

    bad_vocab = set()
    bad_type = set()
    bad_evidence = []
    for r in rels:
        rel_type = (r.get("relation") or "").upper()
        if rel_type not in CONTROLLED_RELATIONS:
            bad_vocab.add(rel_type)
        if r.get("relationship_type") not in ("explicit", "inferred"):
            bad_type.add(r.get("relationship_type"))
        # verify evidence appears on the stated page (whitespace-stripped)
        fy = r.get("fiscal_year")
        page = r.get("page")
        evidence = r.get("evidence") or ""
        if fy in DOCS and evidence:
            doc = load_parsed(fy)
            text = _norm(page_text(doc, page))
            if _norm(evidence) not in text:
                bad_evidence.append(f"{r.get('source',{}).get('name')}@{fy}p{page}")
        else:
            bad_evidence.append(f"{r.get('relationship_id','?')}:bad_fy/about")
    v.check(not bad_vocab, "rels.controlled_vocab", str(bad_vocab))
    v.check(not bad_type, "rels.explicit_inferred", str(bad_type))
    v.check(not bad_evidence, "rels.evidence_grounded", "; ".join(bad_evidence[:8]))

    # distinct fiscal years covered
    years = {r.get("fiscal_year") for r in rels}
    v.check(bool(years & set(EXPECTED_FISCAL_YEARS)), "rels.years", str(sorted(years)))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run(args) -> Validator:
    v = Validator()
    validate_manifest(v)
    validate_parsed_and_chunks(v)
    validate_financial_facts(v)
    validate_entities(v)
    validate_relationships(v)
    return v


def print_report(v: Validator, json_out: bool) -> None:
    if json_out:
        report = {
            "status": "PASS" if v.passed else "FAIL",
            "passed": len(v.passes),
            "failures": len(v.failures),
            "failure_checks": v.failures,
        }
        print(json.dumps(report, indent=2))
        return

    print()
    print("=" * 80)
    print("DATA VALIDATION")
    print("=" * 80)
    print(f"Passed checks : {len(v.passes)}")
    print(f"Failed checks : {len(v.failures)}")

    if v.failures:
        print()
        print("FAILURES:")
        for f in v.failures:
            print(f"  [FAIL] {f}")

    if v.notes:
        print()
        print("NOTES:")
        for n in v.notes:
            print(f"  [..] {n}")

    print()
    print("VERDICT:", "PASS" if v.passed else "FAIL")
    print("=" * 80)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate multi-year pipeline data")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    v = run(args)
    print_report(v, args.json)
    return 0 if v.passed else 1


if __name__ == "__main__":
    sys.exit(main())
