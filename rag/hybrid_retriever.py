"""
Hybrid Retriever: Combines graph + vector retrieval.

Internally:
1. GraphQueryEngine for financial facts, entities, paths
2. VectorRetriever for document chunk search

Produces a unified structured retrieval result.

Usage:
    python -m rag.hybrid_retriever
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.query_engine import GraphQueryEngine
from rag.vector_retriever import VectorRetriever
from rag.fiscal_year import fiscal_year_context
from rag.query_router import (
    FINANCIAL_METRIC_WORDS,
    NARRATIVE_KEYWORDS,
    classify_question,
)


class HybridRetriever:
    """
    Combines graph (Neo4j) and vector (FAISS) retrieval.

    The `retrieve` method runs the selected retrievers
    based on classification and returns a unified result.
    """

    def __init__(
        self,
        graph_engine: Optional[GraphQueryEngine] = None,
        vector_retriever: Optional[VectorRetriever] = None,
    ):
        self.graph_engine = (
            graph_engine or GraphQueryEngine()
        )
        self.vector_retriever = (
            vector_retriever or VectorRetriever()
        )

        self._owns_graph = graph_engine is None

    def close(self):
        if self._owns_graph:
            try:
                self.graph_engine.close()
            except Exception:
                pass

    # ========================================================
    # EAGER CLASSIFICATION INPUTS
    # ========================================================

    def _detect_metric(self, question: str) -> Optional[str]:
        try:
            return self.graph_engine.detect_metric(question)
        except Exception:
            return None

    def _detect_entities(self, question: str) -> List[str]:
        try:
            return self.graph_engine.detect_entities(question)
        except Exception:
            return []

    # ========================================================
    # ROUTER
    # ========================================================

    def classify(
        self,
        question: str,
        force: Optional[str] = None,
    ) -> str:
        """Classify the question type."""
        metric = self._detect_metric(question)
        entities = self._detect_entities(question)

        return classify_question(
            question,
            detected_metric=metric,
            entities=entities,
            force=force,
        )

    # ========================================================
    # GRAPH RETRIEVAL
    # ========================================================

    def _retrieve_graph(
        self, question: str
    ) -> Dict[str, Any]:
        """Run graph retrieval."""
        return self.graph_engine.search(question)

    # ========================================================
    # VECTOR RETRIEVAL
    # ========================================================

    def _retrieve_vector(
        self, question: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Run vector retrieval."""
        return self.vector_retriever.retrieve(
            question, top_k=top_k
        )

    # ========================================================
    # MAIN RETRIEVE
    # ========================================================

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        force_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run retrieval based on question classification.

        Returns structured result:
        {
          "question",
          "question_type",
          "graph_context": {...},
          "vector_context": [...],
        }
        """

        question_type = self.classify(
            question, force=force_type
        )

        graph_context = {"financial_facts": [],
                         "entities": [],
                         "paths": [],
                         "semantic_paths": []}
        vector_context = []

        if question_type in ("financial", "graph", "hybrid"):
            graph_context = self._retrieve_graph(question)

        if question_type in ("narrative", "hybrid"):
            vector_context = self._retrieve_vector(
                question, top_k=top_k
            )

        # For hybrid and graph, we gather vector context
        # to enrich the graph with narrative text when
        # financial facts or entities point to specific pages.
        if question_type in ("graph",):
            # Also get vector text to enrich graph answers
            vector_context = self._retrieve_vector(
                question, top_k=top_k
            )

        # ---------- FISCAL-YEAR AWARENESS ----------
        fyc = fiscal_year_context(question)

        financial_facts = graph_context.get(
            "financial_facts", []
        )

        # When a single, specific fiscal year is named (and the question is
        # not comparative), keep only that year's facts so the answer is
        # scoped correctly. Comparative / multi-year questions keep all years.
        if (
            len(fyc["detected_years"]) == 1
            and not fyc["comparison"]
        ):
            target = fyc["detected_years"][0]
            financial_facts = [
                f for f in financial_facts
                if f.get("period") == target
            ]

        return {
            "question": question,
            "question_type": question_type,
            "fiscal_year_context": fyc,
            "graph_context": {
                "financial_facts": financial_facts,
                "entities": graph_context.get(
                    "entities", []
                ),
                "paths": graph_context.get(
                    "paths", []
                ),
                "semantic_paths": graph_context.get(
                    "semantic_paths", []
                ),
                "detected_metric": graph_context.get(
                    "detected_metric"
                ),
            },
            "vector_context": vector_context,
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LTM HYBRID RETRIEVER TEST")
    print("=" * 60)

    retriever = HybridRetriever()

    try:

        test_questions = [
            "What was LTM's EBITDA?",
            "What was LTM's EBITDA margin?",
            "How does cost optimization affect EBITDA margin?",
            "Starting from LTM's Fit4Future program, trace every available graph relationship that can eventually reach EBITDA margin.",
            "What strategic priorities did LTM discuss regarding AI and digital transformation?",
            "What risks could affect LTM's revenue and profitability?",
        ]

        for q in test_questions:

            print(f"\n{'='*60}")
            print(f"Q: {q}")
            print("-" * 60)

            result = retriever.retrieve(q)

            print(f"Type: {result['question_type']}")

            gc = result["graph_context"]

            facts = gc["financial_facts"]
            print(f"\nFinancial facts: {len(facts)}")
            for f in facts[:3]:
                print(
                    f"  {f.get('metric')} = "
                    f"{f.get('value')} {f.get('unit')}"
                    f" ({f.get('period')}, p{f.get('page')})"
                )

            sem = gc["semantic_paths"]
            print(f"\nSemantic paths: {len(sem)}")
            for p in sem[:3]:
                nodes = p.get("nodes", p.get("path", []))
                rels = p["relationships"]
                chain = " -> ".join(
                    f"{n} [{r}]"
                    for n, r in zip(
                        nodes, rels + [""]
                    )
                )
                print(f"  {chain}")

            vc = result["vector_context"]
            print(f"\nVector context: {len(vc)}")
            for c in vc[:3]:
                print(
                    f"  [p{c['page']}] "
                    f"(score={c['score']:.3f}) "
                    f"{c['text'][:80]}..."
                )

    finally:
        retriever.close()
