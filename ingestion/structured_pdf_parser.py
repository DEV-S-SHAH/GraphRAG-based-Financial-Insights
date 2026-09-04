"""
Phase 3: Structure-aware PDF parser.

Extracts from each page of an annual report:
  - raw text blocks (paragraphs)
  - headings (detected by font size / style)
  - tables (via PyMuPDF find_tables when available)
  - section context (running section name)

Each page is broken into "blocks" with an ordered structure so that later
stages (chunking, financial extraction, entity/relationship extraction) can
retain page- and section-level provenance.

The parser is deliberately conservative: it never fabricates structure. It
uses PyMuPDF's reading-order blocks and `find_tables()` to capture tables,
falling back gracefully if table extraction is unavailable.

Usage:
    python -m ingestion.structured_pdf_parser
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw" / "annual_reports"
PROCESSED_DIR = ROOT / "data" / "processed" / "pdf"

# Common section headings used to split annual reports. Used only as a hint
# for the running-section tracker; headings actually detected in the text are
# preferred.
SECTION_HINTS = [
    "Board’s Report",
    "Board's Report",
    "Management Discussion and Analysis",
    "Management Discussion & Analysis",
    "Corporate Governance",
    "Business Responsibility",
    "Financial Statements",
    "Standalone Financial Statements",
    "Consolidated Financial Statements",
    "Risk Management",
    "Statutory Reports",
    "Notice",
    "Independent Auditor",
    "Key Performance Indicators",
    "Chairman’s Message",
    "CEO and Managing Director",
    "CEO's Message",
]

# Font-size thresholds (points) for heading detection. These are heuristics;
# we also treat short all-uppercase / title lines as candidate headings.
HEADING_FONT_THRESHOLD = 13.0
HEADING_MAX_CHARS = 90
HEADING_MAX_WORDS = 20


def tokenize_heading(text: str) -> Optional[str]:
    """Return a cleaned heading string if text looks like a heading, else None."""
    t = text.strip()
    if not t:
        return None
    if len(t) > HEADING_MAX_CHARS:
        return None
    words = t.split()
    if not words or len(words) > HEADING_MAX_WORDS:
        return None
    # Skip pure numeric / page marker lines
    if re.fullmatch(r"[\d\s.,()-]*", t):
        return None
    return t


def _font_stats(span_fonts: List[Tuple[float, bool]]) -> Tuple[float, bool]:
    """Given (size, bold) pairs, return (max_size, any_bold)."""
    sizes = [s for s, _ in span_fonts]
    bold = any(b for _, b in span_fonts)
    return (max(sizes) if sizes else 0.0), bold


def _is_bold_font(face: str) -> bool:
    low = face.lower()
    return any(k in low for k in ("bold", "semibold", "black", "medium"))


def _extract_table_text(table: Any) -> str:
    """Render a PyMuPDF Table object to simple column/row text."""
    try:
        data = table.extract()
    except Exception:
        return ""
    lines = []
    for row in data:
        cells = []
        for cell in row or []:
            if cell is None:
                cells.append("")
            else:
                cells.append(str(cell).strip())
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_page(
    page: "fitz.Page",
    page_number: int,
    extract_tables: bool = False,
) -> Dict[str, Any]:
    """Parse a single page into ordered blocks with structure hints."""
    blocks_out: List[Dict[str, Any]] = []

    # --- tables first (so their text is excluded from paragraph text) ---
    # NOTE: find_tables() is slow on dense desktop-format pages, so it is
    # opt-in via ``with_tables``. The financial extractor performs its own
    # targeted table parsing where numbers matter.
    table_texts = []
    if extract_tables:
        try:
            tables = page.find_tables()
            for t in tables.tables:
                text = _extract_table_text(t)
                if text.strip():
                    table_texts.append(text)
        except Exception:
            pass

    # --- text blocks in reading order ---
    all_blocks = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0:
            continue  # skip images
        spans = []
        span_fonts = []
        for line in block.get("lines", []):
            line_spans = "".join(
                s.get("text", "") for s in line.get("spans", [])
            )
            if not line_spans.strip():
                continue
            spans.append(line_spans)
            for s in line.get("spans", []):
                face = s.get("font", "")
                span_fonts.append(
                    (s.get("size", 0.0), _is_bold_font(face))
                )
        if not spans:
            continue
        text = "\n".join(spans).strip()
        if not text:
            continue
        max_size, any_bold = _font_stats(span_fonts)
        all_blocks.append((block, text, max_size, any_bold))

    # Classify each block as heading / paragraph / bullet
    for _block, text, max_size, any_bold in all_blocks:
        is_heading = (
            max_size >= HEADING_FONT_THRESHOLD
            and tokenize_heading(text) is not None
        ) or (
            any_bold
            and len(text) <= HEADING_MAX_CHARS
            and tokenize_heading(text) is not None
        )
        block_type = "heading" if is_heading else "paragraph"
        blocks_out.append(
            {
                "page": page_number,
                "type": block_type,
                "text": re.sub(r"[ \t]+", " ", text),
                "font_size": round(max_size, 1),
                "bold": any_bold,
            }
        )

    return {
        "page": page_number,
        "blocks": blocks_out,
        "tables": table_texts,
        "table_count": len(table_texts),
    }


def running_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign a running section name to each page based on detected headings."""
    section = "Front Matter"
    for page in pages:
        page_section = section
        for block in page["blocks"]:
            if block["type"] == "heading":
                candidate = tokenize_heading(block["text"])
                if candidate:
                    for hint in SECTION_HINTS:
                        if hint.lower() in candidate.lower():
                            section = hint
                            break
                    else:
                        # Generic headings still start a section if short/title-cased
                        if len(candidate.split()) <= 6:
                            section = candidate
                            break
            # first (top-most) heading wins as the page's section label
            if page_section == section and section != "Front Matter":
                pass
        # prefer the first heading found on the page for its section label
        found = None
        for block in page["blocks"]:
            if block["type"] == "heading":
                c = tokenize_heading(block["text"])
                if c:
                    for hint in SECTION_HINTS:
                        if hint.lower() in c.lower():
                            found = hint
                            break
                    if found is None and len(c.split()) <= 6:
                        found = c
                    break
        if found:
            page_section = found
            section = found
        page["section"] = page_section
    return pages


def parse_pdf(
    pdf_path: Path,
    document_id: str,
    with_tables: bool = False,
) -> Dict[str, Any]:
    """Parse a PDF into a structured document with prose, tables and sections."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        pages.append(parse_page(page, page_number + 1, extract_tables=with_tables))
    doc.close()

    pages = running_sections(pages)

    return {
        "document_id": document_id,
        "num_pages": len(pages),
        "pages": pages,
    }


def load_manifest() -> List[Dict[str, Any]]:
    manifest_path = ROOT / "data" / "metadata" / "documents_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)["documents"]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Parse annual report PDFs into structure.")
    ap.add_argument(
        "--with-tables",
        action="store_true",
        help="run slow find_tables() extraction (default off)",
    )
    args = ap.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_manifest()
    print("=" * 70)
    print("STRUCTURED PDF PARSER (multi-year)")
    print("=" * 70)

    for entry in documents:
        pdf_path = RAW_DIR / entry["filename"]
        document_id = entry["document_id"]
        out_path = PROCESSED_DIR / f"{document_id}_parsed.json"

        print(f"\n[{entry['fiscal_year']}] {pdf_path.name}", flush=True)
        if not pdf_path.exists():
            print("  SKIP: PDF not found")
            continue

        structured = parse_pdf(
            pdf_path,
            document_id,
            with_tables=args.with_tables,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)

        total_blocks = sum(len(p["blocks"]) for p in structured["pages"])
        total_tables = sum(p["table_count"] for p in structured["pages"])
        print(f"  pages={structured['num_pages']} blocks={total_blocks} "
              f"tables={total_tables}", flush=True)
        print(f"  -> {out_path.name}", flush=True)

    print("\nParser complete.")


if __name__ == "__main__":
    main()
