"""
Vector Retriever: Semantic search over annual report chunks.

Uses FAISS index + Ollama nomic-embed-text embeddings.

Usage:
    python -m rag.vector_retriever
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.vector_store import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    MULTIYEAR_VECTORSTORE_DIR,
    OLLAMA_URL,
    VECTORSTORE_DIR,
    load_index,
)


class VectorRetriever:
    """
    Semantic vector retrieval over LTM annual report.

    Uses FAISS for nearest-neighbor search and
    Ollama nomic-embed-text for query embedding.

    By default searches the multi-year index (FY2022-23 .. FY2025-26).
    Pass index_dir=... or use_multiyear=False to target another index.
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        embedding_model: str = EMBEDDING_MODEL,
        ollama_url: str = OLLAMA_URL,
        use_multiyear: bool = True,
    ):
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url

        if index_dir is None:
            if use_multiyear:
                index_dir = MULTIYEAR_VECTORSTORE_DIR
            else:
                index_dir = VECTORSTORE_DIR

        self.index, self.chunks = load_index(
            index_dir
        )

        self._verify_ollama()

    def _verify_ollama(self):
        """Verify Ollama is available."""
        try:
            resp = requests.get(
                self.ollama_url.replace(
                    "/api/embed", "/api/tags"
                ),
                timeout=5,
            )
            resp.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Ollama not available: {e}"
            )

    def _embed_query(
        self, query: str
    ) -> np.ndarray:
        """Embed a single query string."""
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.embedding_model,
                "input": [query],
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        embedding = np.array(
            data["embeddings"][0], dtype="float32"
        )

        return embedding

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a question.

        Returns list of dicts with:
        - text, page, document_id, company, period
        - score (cosine similarity)
        """

        query_embedding = self._embed_query(question)

        # Normalize for cosine similarity
        query_vec = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_vec)

        # Search
        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_vec, k
        )

        results = []

        for score, idx in zip(
            scores[0], indices[0]
        ):

            if idx < 0:
                continue

            if score < min_score:
                continue

            chunk = self.chunks[idx].copy()

            chunk["score"] = float(score)

            results.append(chunk)

        return results

    def retrieve_by_page(
        self,
        question: str,
        pages: List[int],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks filtered to specific pages.
        """

        all_results = self.retrieve(
            question, top_k=top_k * 3
        )

        filtered = [
            r
            for r in all_results
            if r["page"] in pages
        ]

        return filtered[:top_k]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LTM VECTOR RETRIEVER TEST")
    print("=" * 60)

    retriever = VectorRetriever()

    test_questions = [
        "What were LTM's major strategic priorities?",
        "What did management say about AI?",
        "What risks could affect revenue?",
        "EBITDA margin performance",
    ]

    for q in test_questions:

        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("-" * 60)

        results = retriever.retrieve(q, top_k=3)

        for i, r in enumerate(results):

            print(
                f"\n[{i+1}] Page {r['page']} "
                f"(score={r['score']:.3f})"
            )
            print(f"    {r['text'][:150]}...")
