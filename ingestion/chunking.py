"""
Phase 3 (part 2): Structure-aware chunking with provenance.

Reads the structured parsed JSON produced by ``structured_pdf_parser`` and
produces chunks. Each chunk keeps:
  - chunk_id            (stable, deterministic)
  - text
  - document_id, fiscal_year, page, section
  - chunk_type          (prose | table)
  - block_indices       (which parsed blocks map to this chunk)

Chunking policy
---------------
- We do NOT split across pages.
- Each page contributes "prose" chunks (paragraphs grouped up to target size)
  and one-or-more "table" chunks (each rendered table kept whole so financial
  tables are not torn apart).
- Section title is carried so that vector retrieval can be filtered by section.

Usage:
    python -m ingestion.chunking
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = ROOT / "data" / "processed" / "pdf"
CHUNKS_DIR = ROOT / "data" / "chunks"

TARGET_CHARS = 1100
MAX_CHARS = 1400
MIN_CHARS = 200

# Block types to exclude from prose (headers/footers noise patterns)
NOISE_RE = re.compile(
    r"^(page\s*\d+|infinite\s+possibilities|ltimindtree\s+limited\s*\|"
    r"integrated\s+annual\s+report|contents)$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\x07", "")).strip()


def _is_noise(text: str) -> bool:
    t = _clean(text)
    if not t:
        return True
    if len(t) > 400:
        return False
    if len(t) < 6:
        return True
    if NOISE_RE.match(t):
        return True
    # pure page-number lines like "  12  " or "1 | 2"
    if re.fullmatch(r"[\d\s.,|()-]+", t):
        return True
    return False


def _chunk_id(document_id: str, page: int, kind: str, idx: int) -> str:
    return f"{document_id}_p{page}_{kind}{idx}"


def chunk_document(
    structured: Dict[str, Any],
    fiscal_year: str,
    target_chars: int = TARGET_CHARS,
    max_chars: int = MAX_CHARS,
) -> List[Dict[str, Any]]:
    """Convert a structured parsed document into provenance-rich chunks."""
    document_id = structured["document_id"]
    chunks: List[Dict[str, Any]] = []

    for page_data in structured["pages"]:
        page = page_data["page"]
        section = _clean(page_data.get("section", ""))
        blocks = page_data.get("blocks", [])
        tables = page_data.get("tables", [])

        # --- table chunks (each table kept whole) ---
        for t_idx, table_text in enumerate(tables):
            clean = _clean(table_text)
            if clean:
                chunks.append(
                    {
                        "chunk_id": _chunk_id(document_id, page, "t", t_idx),
                        "text": clean,
                        "document_id": document_id,
                        "fiscal_year": fiscal_year,
                        "page": page,
                        "section": section,
                        "chunk_type": "table",
                    }
                )

        # --- prose chunks: group non-noisy, non-heading-only blocks ---
        prose_blocks = [
            (i, b)
            for i, b in enumerate(blocks)
            if not _is_noise(b.get("text", ""))
        ]
        if not prose_blocks:
            continue

        current = ""
        current_indices: List[int] = []
        counter = 0

        def flush():
            nonlocal current, current_indices, counter
            text = _clean(current)
            if text:
                chunks.append(
                    {
                        "chunk_id": _chunk_id(document_id, page, "s", counter),
                        "text": text,
                        "document_id": document_id,
                        "fiscal_year": fiscal_year,
                        "page": page,
                        "section": section,
                        "chunk_type": "prose",
                        "block_indices": current_indices,
                    }
                )
                counter += 1
            current = ""
            current_indices = []

        for i, block in prose_blocks:
            block_text = _clean(block.get("text", ""))
            if not block_text:
                continue
            if current and (len(current) + len(block_text)) > target_chars:
                flush()
            current += ("\n\n" if current else "") + block_text
            current_indices.append(i)

        if current:
            flush()

    return chunks


def load_manifest() -> List[Dict[str, Any]]:
    manifest_path = ROOT / "data" / "metadata" / "documents_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)["documents"]


def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_manifest()
    print("=" * 70)
    print("STRUCTURE-AWARE CHUNKING (multi-year)")
    print("=" * 70)

    totals = []
    for entry in documents:
        document_id = entry["document_id"]
        parsed_path = PROCESSED_DIR / f"{document_id}_parsed.json"
        out_path = CHUNKS_DIR / f"{document_id}_chunks.json"

        if not parsed_path.exists():
            print(f"[{entry['fiscal_year']}] SKIP (no parsed file)")
            continue

        with open(parsed_path, "r", encoding="utf-8") as f:
            structured = json.load(f)

        chunks = chunk_document(structured, entry["fiscal_year"])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"document_id": document_id, "chunks": chunks}, f, indent=2)

        types = {}
        for c in chunks:
            types[c["chunk_type"]] = types.get(c["chunk_type"], 0) + 1
        print(f"[{entry['fiscal_year']}] chunks={len(chunks)} {types}")
        totals.append(len(chunks))

    print(f"\nTotal chunks: {sum(totals)}")


if __name__ == "__main__":
    main()
