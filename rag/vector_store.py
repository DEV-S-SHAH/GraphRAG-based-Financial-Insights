"""
Vector Store: Page-aware chunking + FAISS index.

Builds a vector index from the normalized annual report.
Uses Ollama nomic-embed-text for local embeddings.

Usage:
    python -m rag.vector_store
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = (
    PROJECT_ROOT / "data" / "vectorstore"
)

# Multi-year (FY2022-23 / FY2023-24 / FY2024-25 / FY2025-26) FAISS index.
MULTIYEAR_VECTORSTORE_DIR = (
    PROJECT_ROOT / "data" / "vectorstore" / "multiyear"
)

# Directory holding the structure-aware, provenance-rich chunks produced by
# ingestion/chunking.py (one file per fiscal-year report).
MULTIYEAR_CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

FISCAL_YEARS = ["FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"]

NORMALIZED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normalized"
    / "LTM_FY2025-26_normalized.json"
)

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
OLLAMA_URL = "http://localhost:11434/api/embed"


# ============================================================
# TEXT SPLITTER
# ============================================================


class RecursiveCharacterTextSplitter:
    """
    Splits text into chunks by recursively trying
    separators from largest to smallest.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            self.separators = [
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ]
        else:
            self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if len(text) <= self.chunk_size:
            text = text.strip()
            if text:
                return [text]
            return []

        chunks = self._recursive_split(
            text, self.separators
        )

        return [
            c.strip()
            for c in chunks
            if c.strip()
        ]

    def _recursive_split(
        self, text: str, separators: List[str]
    ) -> List[str]:
        """Recursively split text."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._force_split(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return self._force_split(text)

        parts = text.split(sep)

        chunks = []

        current = ""

        for part in parts:

            candidate = (
                current + sep + part
                if current
                else part
            )

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.extend(
                        self._recursive_split(
                            current, remaining_seps
                        )
                    )
                current = part

        if current:
            chunks.extend(
                self._recursive_split(
                    current, remaining_seps
                )
            )

        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:

            overlapped = [chunks[0]]

            for i in range(1, len(chunks)):

                prev = chunks[i - 1]
                overlap_text = prev[
                    -self.chunk_overlap:
                ]

                # Find word boundary
                space_idx = overlap_text.find(" ")

                if space_idx > 0:
                    overlap_text = overlap_text[
                        space_idx + 1:
                    ]

                combined = overlap_text + chunks[i]

                if len(combined) <= self.chunk_size:
                    overlapped.append(combined)
                else:
                    overlapped.append(chunks[i])

            chunks = overlapped

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """Force split at chunk_size boundaries."""
        chunks = []

        start = 0

        while start < len(text):

            end = start + self.chunk_size

            if end < len(text):

                # Try to find a word boundary
                space = text.rfind(
                    " ", start, end
                )

                if space > start + self.chunk_size // 2:
                    end = space

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start = (
                end - self.chunk_overlap
                if self.chunk_overlap > 0
                else end
            )

            if start >= len(text):
                break

        return chunks


# ============================================================
# CHUNKING
# ============================================================


def create_chunks(
    normalized_path: Optional[Path] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Create page-aware chunks from normalized document.

    Each chunk preserves:
    - chunk_id
    - text
    - page number
    - document_id
    - company, ticker, period
    """

    if normalized_path is None:
        normalized_path = NORMALIZED_PATH

    with open(normalized_path, "r") as f:
        doc = json.load(f)

    document_id = doc.get(
        "document_id", "unknown"
    )
    company = doc.get("company", "LTM Limited")
    reporting_period = doc.get(
        "reporting_period", "FY2025-26"
    )

    pages = doc.get("pages", [])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    chunk_id = 0

    for page_data in pages:

        page_num = page_data.get("page", 0)
        text = page_data.get("text", "")

        if not text.strip():
            continue

        text_chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(text_chunks):

            chunks.append(
                {
                    "chunk_id": (
                        f"{document_id}_p{page_num}_c{i}"
                    ),
                    "text": chunk_text,
                    "page": page_num,
                    "document_id": document_id,
                    "company": company,
                    "ticker": "LTM",
                    "period": reporting_period,
                    "chunk_index": i,
                }
            )

            chunk_id += 1

    return chunks


# ============================================================
# EMBEDDING
# ============================================================


def get_embeddings(
    texts: List[str],
    batch_size: int = 64,
    ollama_url: str = OLLAMA_URL,
    model: str = EMBEDDING_MODEL,
) -> List[List[float]]:
    """
    Get embeddings from Ollama for a list of texts.
    Processes in batches to avoid timeouts.
    """

    all_embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i : i + batch_size]

        response = requests.post(
            ollama_url,
            json={"model": model, "input": batch},
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        all_embeddings.extend(data["embeddings"])

        if i + batch_size < len(texts):

            print(
                f"  Embedded {i + len(batch)}"
                f" / {len(texts)}"
            )

    return all_embeddings


# ============================================================
# BUILD INDEX
# ============================================================


def build_index(
    chunks: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[Path] = None,
    ollama_url: str = OLLAMA_URL,
    embedding_model: str = EMBEDDING_MODEL,
) -> Path:
    """
    Build FAISS index from chunks.
    Returns path to the output directory.
    """

    if output_dir is None:
        output_dir = VECTORSTORE_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        print("Creating chunks from normalized doc...")
        chunks = create_chunks()
        print(f"Created {len(chunks)} chunks.")

    if not chunks:
        raise ValueError("No chunks to index.")

    # Extract texts for embedding
    texts = [c["text"] for c in chunks]

    print(
        f"Embedding {len(texts)} chunks "
        f"with {embedding_model}..."
    )

    start = time.time()

    embeddings = get_embeddings(
        texts,
        ollama_url=ollama_url,
        model=embedding_model,
    )

    elapsed = time.time() - start

    print(
        f"Embedded in {elapsed:.1f}s "
        f"({len(texts) / max(elapsed, 0.01):.0f} "
        f"chunks/sec)"
    )

    # Convert to numpy array
    import numpy as np

    embedding_matrix = np.array(
        embeddings, dtype="float32"
    )

    # Build FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    # Normalize for cosine similarity
    faiss.normalize_L2(embedding_matrix)

    index.add(embedding_matrix)

    # Save index
    faiss.write_index(
        index,
        str(output_dir / "index.faiss"),
    )

    # Save metadata
    metadata = {
        "model": embedding_model,
        "dimension": EMBEDDING_DIM,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }

    with open(
        output_dir / "metadata.json", "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Index saved to {output_dir}"
    )
    print(f"  Chunks: {len(chunks)}")
    print(f"  Dimension: {EMBEDDING_DIM}")

    return output_dir


# ============================================================
# LOAD INDEX
# ============================================================


def load_index(
    index_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Load FAISS index and metadata.
    Returns (index, chunks).
    """

    if index_dir is None:
        index_dir = VECTORSTORE_DIR

    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}\n"
            "Run 'python -m rag.vector_store' first."
        )

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return index, metadata["chunks"]


# ============================================================
# MULTI-YEAR CHUNK LOADING
# ============================================================


def load_multiyear_chunks(
    chunks_dir: Optional[Path] = None,
    fiscal_years: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load all provenance-rich chunks from the multi-year pipeline.

    Each chunk already carries: chunk_id, text, document_id, fiscal_year,
    page, section, chunk_type. Returns a flat list across the three reports.
    """

    if chunks_dir is None:
        chunks_dir = MULTIYEAR_CHUNKS_DIR

    if fiscal_years is None:
        fiscal_years = FISCAL_YEARS

    document_ids = {
        "FY2022-23": "LTIM_FY2022-23_Annual_Report",
        "FY2023-24": "LTIM_FY2023-24_Annual_Report",
        "FY2024-25": "LTIM_FY2024-25_Annual_Report",
        "FY2025-26": "LTIM_FY2025-26_Annual_Report",
    }

    chunks: List[Dict[str, Any]] = []

    for fy in fiscal_years:

        doc_id = document_ids[fy]
        path = chunks_dir / f"{doc_id}_chunks.json"

        if not path.exists():
            raise FileNotFoundError(f"Chunk file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for chunk in data.get("chunks", []):
            chunk = dict(chunk)
            chunk["company"] = "LTIMindtree Limited"
            chunk["ticker"] = "LTIM"
            chunk["reporting_period"] = fy
            chunk.setdefault("fiscal_year", fy)
            chunks.append(chunk)

    return chunks


# ============================================================
# MULTI-YEAR INDEX BUILD
# ============================================================


def build_multiyear_index(
    output_dir: Optional[Path] = None,
    ollama_url: str = OLLAMA_URL,
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = 64,
) -> Path:
    """
    Build the FAISS index over the multi-year pipeline chunks.

    Uses Ollama nomic-embed-text (768-dim) embeddings and stores each chunk
    with its fiscal_year / page / section provenance.
    """

    if output_dir is None:
        output_dir = MULTIYEAR_VECTORSTORE_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_multiyear_chunks()

    print(f"Loaded {len(chunks)} multi-year chunks.")

    if not chunks:
        raise ValueError("No multi-year chunks to index.")

    texts = [c["text"] for c in chunks]

    print(
        f"Embedding {len(texts)} chunks "
        f"with {embedding_model} (dim {EMBEDDING_DIM})..."
    )

    start = time.time()

    embeddings = get_embeddings(
        texts,
        batch_size=batch_size,
        ollama_url=ollama_url,
        model=embedding_model,
    )

    elapsed = time.time() - start

    print(
        f"Embedded in {elapsed:.1f}s "
        f"({len(texts) / max(elapsed, 0.01):.0f} chunks/sec)"
    )

    import numpy as np

    embedding_matrix = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    faiss.normalize_L2(embedding_matrix)
    index.add(embedding_matrix)

    faiss.write_index(index, str(output_dir / "index.faiss"))

    metadata = {
        "model": embedding_model,
        "dimension": EMBEDDING_DIM,
        "num_chunks": len(chunks),
        "fiscal_years": FISCAL_YEARS,
        "chunks": chunks,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Multi-year index saved to {output_dir}")

    return output_dir


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":

    print("=" * 60)
    print("LTM VECTOR STORE BUILDER")
    print("=" * 60)

    output_dir = build_index()

    # Verify
    index, chunks = load_index(output_dir)

    print()
    print(f"Verification: index has "
          f"{index.ntotal} vectors")
    print(f"Metadata has {len(chunks)} chunks")
    print()
    print("Sample chunk:")
    print(f"  ID: {chunks[0]['chunk_id']}")
    print(f"  Page: {chunks[0]['page']}")
    print(f"  Text: {chunks[0]['text'][:120]}...")
