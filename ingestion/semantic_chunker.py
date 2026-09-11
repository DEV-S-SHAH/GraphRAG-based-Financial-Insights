"""
Semantic Structure-Aware Chunker.

Transforms parsed documents into provenance-rich chunks that:
1. Preserve tabular structures intact (tables are not split across chunks)
2. Respect section and page boundaries
3. Attach deterministic unique IDs and complete metadata for PostgreSQL & Neo4j.
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional


@dataclass
class SemanticChunk:
    chunk_id: str
    document_id: str
    company: str
    ticker: str
    fiscal_year: str
    page: int
    section: str
    chunk_type: str  # "prose" | "table"
    text: str
    source_url: str = ""
    chunk_index: int = 0
    token_estimate: int = 0
    metadata: Dict[str, Any] = None


class SemanticChunker:
    """Structure-aware chunker that never tears tables apart and respects sections."""

    def __init__(self, target_chars: int = 1100, max_chars: int = 1500, min_chars: int = 30):
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.min_chars = min_chars

    def chunk_document(
        self,
        parsed_doc: Any,
        source_url: str = "",
    ) -> List[SemanticChunk]:
        """Convert a ParsedDocument or dict into a list of SemanticChunk objects."""
        if hasattr(parsed_doc, "document_id"):
            doc_id = parsed_doc.document_id
            company = parsed_doc.company
            ticker = parsed_doc.ticker
            fiscal_year = parsed_doc.fiscal_year
            pages = parsed_doc.pages
            source_url = source_url or parsed_doc.source_url
        else:
            doc_id = parsed_doc.get("document_id", "unknown_doc")
            company = parsed_doc.get("company", "LTIMindtree Limited")
            ticker = parsed_doc.get("ticker", "LTIM")
            fiscal_year = parsed_doc.get("fiscal_year", "FY2025-26")
            pages = parsed_doc.get("pages", [])
            source_url = source_url or parsed_doc.get("source_url", "")

        chunks: List[SemanticChunk] = []
        global_index = 0

        for page_data in pages:
            page_num = page_data.get("page", 1)
            section = page_data.get("section", "General")
            blocks = page_data.get("blocks", [])
            tables = page_data.get("tables", [])

            # 1. Table chunks: Each table preserved as a distinct chunk
            for t_idx, table_text in enumerate(tables):
                cleaned_table = table_text.strip()
                if cleaned_table and len(cleaned_table) >= 30:
                    chunk_id = f"{doc_id}_p{page_num}_t{t_idx}"
                    sc = SemanticChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        company=company,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        page=page_num,
                        section=section,
                        chunk_type="table",
                        text=cleaned_table,
                        source_url=source_url,
                        chunk_index=global_index,
                        token_estimate=len(cleaned_table) // 4,
                        metadata={
                            "is_table": True,
                            "table_index": t_idx,
                            "document_id": doc_id,
                            "company": company,
                            "ticker": ticker,
                            "fiscal_year": fiscal_year,
                            "page": page_num,
                            "section": section,
                            "source_url": source_url,
                        },
                    )
                    chunks.append(sc)
                    global_index += 1

            # 2. Prose chunks: Group paragraphs under the current section
            prose_paragraphs = []
            for b in blocks:
                t = b.get("text", "").strip() if isinstance(b, dict) else getattr(b, "text", "").strip()
                if not t:
                    continue
                # Filter out pure noise / running headers / page numbers
                if re.fullmatch(r"[\d\s.,|()/-]+", t) and len(t) < 15:
                    continue
                prose_paragraphs.append(t)

            if not prose_paragraphs:
                continue

            current_chunk_text = ""
            prose_counter = 0

            def flush_prose():
                nonlocal current_chunk_text, prose_counter, global_index
                text = current_chunk_text.strip()
                if text and len(text) >= self.min_chars:
                    chunk_id = f"{doc_id}_p{page_num}_s{prose_counter}"
                    sc = SemanticChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        company=company,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        page=page_num,
                        section=section,
                        chunk_type="prose",
                        text=text,
                        source_url=source_url,
                        chunk_index=global_index,
                        token_estimate=len(text) // 4,
                        metadata={
                            "is_table": False,
                            "section": section,
                            "document_id": doc_id,
                            "company": company,
                            "ticker": ticker,
                            "fiscal_year": fiscal_year,
                            "page": page_num,
                            "source_url": source_url,
                        },
                    )
                    chunks.append(sc)
                    prose_counter += 1
                    global_index += 1
                current_chunk_text = ""

            for para in prose_paragraphs:
                if current_chunk_text and (len(current_chunk_text) + len(para) > self.target_chars):
                    flush_prose()
                current_chunk_text += ("\n\n" if current_chunk_text else "") + para

            if current_chunk_text:
                flush_prose()

        return chunks
