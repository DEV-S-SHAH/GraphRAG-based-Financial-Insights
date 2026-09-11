"""
Docling PDF Document Parser.

Parses annual reports and filings preserving:
- Tables (as markdown pipe-tables)
- Headings & hierarchy
- Sections (running section identification)
- Page numbers
- Document / year / company metadata & source URL

Includes graceful fallbacks and hybrid chunking support.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Common financial filing section headings
SECTION_HINTS = [
    "Corporate Overview",
    "Board's Report",
    "Board’s Report",
    "Management Discussion and Analysis",
    "Management Discussion & Analysis",
    "MD&A",
    "Corporate Governance",
    "Business Responsibility & Sustainability Report",
    "BRSR",
    "Financial Statements",
    "Standalone Financial Statements",
    "Consolidated Financial Statements",
    "Balance Sheet",
    "Statement of Profit and Loss",
    "Cash Flow Statement",
    "Notes to Financial Statements",
    "Risk Management",
    "Statutory Reports",
    "Independent Auditor's Report",
    "Key Performance Indicators",
    "Highlights",
    "Notice",
]


@dataclass
class ParsedBlock:
    page: int
    block_type: str  # "heading", "paragraph", "table", "list_item"
    text: str
    section: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    table_data: Optional[List[List[str]]] = None


@dataclass
class ParsedDocument:
    document_id: str
    company: str
    ticker: str
    fiscal_year: str
    source_url: str
    num_pages: int
    sections: List[str]
    pages: List[Dict[str, Any]]
    blocks: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]


class DoclingParser:
    """PDF parser leveraging Docling with structured table and heading preservation."""

    def __init__(self, use_docling_converter: bool = True):
        self.use_docling_converter = use_docling_converter
        self._docling_available = False
        if use_docling_converter:
            try:
                from docling.document_converter import DocumentConverter
                self._converter = DocumentConverter()
                self._docling_available = True
            except Exception as e:
                logger.warning(f"Docling converter initialization fallback: {e}")
                self._docling_available = False

    def parse_pdf(
        self,
        pdf_path: Path,
        document_id: str,
        company: str = "LTIMindtree Limited",
        ticker: str = "LTIM",
        fiscal_year: str = "FY2025-26",
        source_url: str = "",
        max_pages: Optional[int] = None,
    ) -> ParsedDocument:
        """Parse a PDF document into structured pages, blocks, and tables."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if pdf_path.stat().st_size == 0:
            raise ValueError(f"PDF file is empty (0 bytes): {pdf_path}")

        # Try Docling if enabled and small/medium document
        if self._docling_available:
            try:
                return self._parse_with_docling(
                    pdf_path, document_id, company, ticker, fiscal_year, source_url, max_pages
                )
            except Exception as e:
                logger.warning(f"Docling parsing encountered error: {e}. Falling back to structured parser.")

        try:
            return self._parse_with_structured_mupdf(
                pdf_path, document_id, company, ticker, fiscal_year, source_url, max_pages
            )
        except Exception as e:
            logger.error(f"Structured parser failed on {pdf_path}: {e}")
            raise ValueError(f"Corrupted or unreadable PDF document {pdf_path}: {e}")

    def _parse_with_docling(
        self,
        pdf_path: Path,
        document_id: str,
        company: str,
        ticker: str,
        fiscal_year: str,
        source_url: str,
        max_pages: Optional[int],
    ) -> ParsedDocument:
        """Use Docling's native DocumentConverter."""
        conv_res = self._converter.convert(str(pdf_path))
        doc = conv_res.document

        pages_data = {}
        all_blocks = []
        all_tables = []
        sections_found = set()

        current_section = "Overview"

        # Export tables
        for t_idx, table in enumerate(doc.tables):
            table_md = table.export_to_markdown() if hasattr(table, "export_to_markdown") else str(table)
            page_no = 1
            if hasattr(table, "prov") and table.prov and hasattr(table.prov[0], "page_no"):
                page_no = table.prov[0].page_no
            
            table_dict = {
                "table_id": f"{document_id}_p{page_no}_t{t_idx}",
                "page": page_no,
                "section": current_section,
                "markdown": table_md,
            }
            all_tables.append(table_dict)

        # Export elements / blocks
        for item, _level in doc.iterate_items():
            text = getattr(item, "text", "").strip()
            if not text:
                continue

            page_no = 1
            if hasattr(item, "prov") and item.prov and hasattr(item.prov[0], "page_no"):
                page_no = item.prov[0].page_no

            label = getattr(item, "label", "paragraph").lower()
            is_heading = "heading" in label or "title" in label or "header" in label

            if is_heading and len(text.split()) <= 10:
                current_section = text
                sections_found.add(text)

            b = ParsedBlock(
                page=page_no,
                block_type="heading" if is_heading else "paragraph",
                text=text,
                section=current_section,
            )
            all_blocks.append(asdict(b))

            if page_no not in pages_data:
                pages_data[page_no] = {"page": page_no, "section": current_section, "blocks": []}
            pages_data[page_no]["blocks"].append(asdict(b))

        num_pages = len(pages_data) if pages_data else 1

        return ParsedDocument(
            document_id=document_id,
            company=company,
            ticker=ticker,
            fiscal_year=fiscal_year,
            source_url=source_url,
            num_pages=num_pages,
            sections=sorted(list(sections_found)),
            pages=list(pages_data.values()),
            blocks=all_blocks,
            tables=all_tables,
        )

    def _parse_with_structured_mupdf(
        self,
        pdf_path: Path,
        document_id: str,
        company: str,
        ticker: str,
        fiscal_year: str,
        source_url: str,
        max_pages: Optional[int],
    ) -> ParsedDocument:
        """Fast, robust structured reading-order parser with table detection."""
        import fitz

        doc = fitz.open(str(pdf_path))
        num_pages = len(doc) if max_pages is None else min(len(doc), max_pages)

        pages = []
        all_blocks = []
        all_tables = []
        sections_found = set()

        current_section = "Corporate Overview"

        for page_idx in range(num_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Extract tables
            table_texts = []
            try:
                tables = page.find_tables()
                for t_idx, t in enumerate(tables.tables):
                    data = t.extract()
                    if data:
                        lines = []
                        for row in data:
                            cells = [str(c).strip().replace("\n", " ") if c is not None else "" for c in row]
                            lines.append("| " + " | ".join(cells) + " |")
                        if len(lines) >= 2:
                            header_sep = "| " + " | ".join(["---"] * len(data[0])) + " |"
                            table_md = lines[0] + "\n" + header_sep + "\n" + "\n".join(lines[1:])
                        else:
                            table_md = "\n".join(lines)
                        
                        table_texts.append(table_md)
                        all_tables.append({
                            "table_id": f"{document_id}_p{page_num}_t{t_idx}",
                            "page": page_num,
                            "section": current_section,
                            "markdown": table_md,
                        })
            except Exception:
                pass

            # 2. Extract text blocks in reading order
            page_blocks = []
            text_dict = page.get_text("dict", sort=True)
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue  # skip image blocks

                spans_text = []
                max_font_size = 0.0
                is_bold = False

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "")
                        if t.strip():
                            spans_text.append(t)
                            size = span.get("size", 0.0)
                            if size > max_font_size:
                                max_font_size = size
                            font_name = span.get("font", "").lower()
                            if any(k in font_name for k in ["bold", "black", "heavy", "semibold"]):
                                is_bold = True

                block_text = " ".join(spans_text).strip()
                if not block_text:
                    continue

                # Heading detection: font size >= 13 or bold and short
                is_heading = (max_font_size >= 13.0 and len(block_text.split()) <= 15) or (
                    is_bold and len(block_text.split()) <= 8 and not block_text.endswith(".")
                )

                if is_heading:
                    for hint in SECTION_HINTS:
                        if hint.lower() in block_text.lower():
                            current_section = hint
                            sections_found.add(hint)
                            break
                    else:
                        if len(block_text.split()) <= 6:
                            current_section = block_text
                            sections_found.add(block_text)

                pb = ParsedBlock(
                    page=page_num,
                    block_type="heading" if is_heading else "paragraph",
                    text=block_text,
                    section=current_section,
                    font_size=round(max_font_size, 1),
                    is_bold=is_bold,
                )
                page_blocks.append(asdict(pb))
                all_blocks.append(asdict(pb))

            pages.append({
                "page": page_num,
                "section": current_section,
                "blocks": page_blocks,
                "tables": table_texts,
            })

        doc.close()

        return ParsedDocument(
            document_id=document_id,
            company=company,
            ticker=ticker,
            fiscal_year=fiscal_year,
            source_url=source_url,
            num_pages=num_pages,
            sections=sorted(list(sections_found)),
            pages=pages,
            blocks=all_blocks,
            tables=all_tables,
        )
