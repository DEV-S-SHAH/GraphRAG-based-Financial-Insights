"""
Context Fusion: Combine graph + vector retrieval into a
well-structured prompt context.

Sections:
- QUESTION
- GRAPH EVIDENCE
  - Financial Facts
  - Entities
  - Graph Paths
- DOCUMENT EVIDENCE
  - Page N: ...
- SOURCE METADATA
"""

from typing import Any, Dict, List


# ============================================================
# HELPERS
# ============================================================


def _format_financial_facts(facts: List[Dict[str, Any]]) -> str:

    if not facts:
        return "No financial facts retrieved."

    lines = []

    for f in facts:

        lines.append(
            f"- {f.get('metric')} = {f.get('value')} "
            f"{f.get('unit')} "
            f"(period: {f.get('period')}, "
            f"page: {f.get('page')}, "
            f"section: {f.get('section')})"
        )

    return "\n".join(lines)


def _format_entities(entities: List[Dict[str, Any]]) -> str:

    if not entities:
        return "No entities retrieved."

    lines = []

    for entity_result in entities:

        entity = entity_result.get("entity", {})

        if not entity:
            continue

        name = entity.get("entity")

        if not name:
            continue

        lines.append(f"- {name}")

        for rel in entity.get("outgoing", []):
            lines.append(
                f"    {name} --[{rel.get('relationship')}]--> "
                f"{rel.get('target')}"
            )

        for rel in entity.get("incoming", []):
            lines.append(
                f"    {rel.get('source')} --"
                f"[{rel.get('relationship')}]--> {name}"
            )

    return "\n".join(lines) if lines else "No entities."


def _format_graph_paths(paths: List[Dict[str, Any]]) -> str:

    if not paths:
        return "No graph paths retrieved."

    lines = []

    for path_data in paths:

        nodes = path_data.get(
            "nodes", path_data.get("path", [])
        )

        relationships = path_data.get(
            "relationships", []
        )

        chain_parts = []

        for i, node in enumerate(nodes):

            chain_parts.append(str(node))

            if i < len(relationships):
                chain_parts.append(
                    f"--[{relationships[i]}]-->"
                )

        lines.append("- " + " ".join(chain_parts))

    return "\n".join(lines)


def _format_document_evidence(
    vector_context: List[Dict[str, Any]]
) -> str:

    if not vector_context:
        return "No document evidence retrieved."

    # Deduplicate by page (keep highest score per page)
    by_page = {}

    for c in vector_context:

        page = c.get("page")

        current = by_page.get(page)

        if current is None or c.get(
            "score", 0
        ) > current.get("score", 0):
            by_page[page] = c

    # Order by page
    sorted_chunks = sorted(
        by_page.values(),
        key=lambda c: c.get("page", 0),
    )

    lines = []

    for c in sorted_chunks:

        page = c.get("page")
        text = c.get("text", "")

        snippet = text.strip()

        lines.append(
            f"[Page {page}]\n{snippet}"
        )

    return "\n\n".join(lines)


# ============================================================
# MAIN FORMATTER
# ============================================================


def build_context(result: Dict[str, Any]) -> str:
    """
    Build the fused context string from a HybridRetriever result.

    Structure:

    QUESTION
    ...

    GRAPH EVIDENCE
    ---------------
    FINANCIAL FACTS:
    ...

    ENTITIES:
    ...

    GRAPH PATHS:
    ...

    DOCUMENT EVIDENCE
    -----------------
    [Page N]
    ...

    SOURCE METADATA
    ---------------
    Document: ...
    Company: LTM Limited
    Ticker: LTM
    Period: FY2025-26
    """

    question = result.get("question", "")
    graph_context = result.get("graph_context", {})
    vector_context = result.get("vector_context", [])
    fyc = result.get("fiscal_year_context", {})

    parts = []

    # ---------- QUESTION ----------
    parts.append(f"QUESTION:\n{question}")

    # ---------- FISCAL-YEAR CONTEXT ----------
    if fyc:
        fiscal_parts = [
            "FISCAL-YEAR CONTEXT",
            "-" * 18,
            f'{fyc.get("scope_note", "")}',
        ]
        years = fyc.get("detected_years", [])
        if years:
            fiscal_parts.append(
                "Mentioned fiscal year(s): " + ", ".join(years)
            )
        fiscal_parts.append(
            "Available fiscal years: "
            + ", ".join(
                fyc.get("available_fiscal_years", [])
            )
        )
        parts.append("\n".join(fiscal_parts))

    # ---------- GRAPH EVIDENCE ----------
    graph_parts = ["GRAPH EVIDENCE", "-" * 15]

    graph_parts.append("FINANCIAL FACTS:")
    graph_parts.append(
        _format_financial_facts(
            graph_context.get("financial_facts", [])
        )
    )

    graph_parts.append("")
    graph_parts.append("ENTITIES:")
    graph_parts.append(
        _format_entities(
            graph_context.get("entities", [])
        )
    )

    graph_parts.append("")
    graph_parts.append("GRAPH PATHS:")
    graph_parts.append(
        _format_graph_paths(
            graph_context.get("semantic_paths", [])
        )
    )

    parts.append("\n".join(graph_parts))

    # ---------- DOCUMENT EVIDENCE ----------
    doc_parts = ["DOCUMENT EVIDENCE", "-" * 17]

    doc_parts.append(
        _format_document_evidence(vector_context)
    )

    parts.append("\n".join(doc_parts))

    # ---------- SOURCE METADATA ----------
    # Multi-year aware: derive fiscal years from retrieved facts / chunks.
    meta = {
        "company": "LTIMindtree Limited",
        "ticker": "LTIM",
        "fiscal_years_covered": "FY2022-23, FY2023-24, FY2024-25, FY2025-26",
    }

    period_labels = set()
    documents = set()
    for f in graph_context.get("financial_facts", []):
        if f.get("period"):
            period_labels.add(f["period"])
        doc = f.get("document")
        if doc:
            documents.add(doc)
    for c in vector_context:
        if c.get("fiscal_year") or c.get("period"):
            period_labels.add(
                c.get("fiscal_year") or c.get("period")
            )
        if c.get("document_id"):
            documents.add(c["document_id"])

    if period_labels:
        meta["fiscal_years_covered"] = ", ".join(
            sorted(period_labels)
        )
    if documents:
        meta["documents"] = ", ".join(sorted(documents))

    first = vector_context[0] if vector_context else None
    if first and isinstance(first.get("company"), dict):
        meta["company"] = first["company"].get(
            "name", meta["company"]
        )
    elif first and first.get("company"):
        meta["company"] = first["company"]

    meta_parts = ["SOURCE METADATA", "-" * 14]

    for key, value in meta.items():
        meta_parts.append(
            f"{key.replace('_', ' ').title()}: {value}"
        )

    parts.append("\n".join(meta_parts))

    return "\n\n".join(parts)
