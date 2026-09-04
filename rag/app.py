"""
LTM Hybrid Graph + Vector RAG Application

Neo4j Graph Query -> FAISS Vector Retrieval -> Ollama

Run the complete RAG CLI:
    python -m rag.app

Usage:
    answer_question(question) -> structured result
"""

import sys
import os

# Make project root importable
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from rag.hybrid_retriever import HybridRetriever
from rag.context_fusion import build_context
from rag.answer_generator import AnswerGenerator


def format_context(graph_result):
    """
    Backward-compatible formatter for a graph-only
    GraphQueryEngine result. Used by legacy tests.
    """

    parts = []

    question = graph_result.get("question")
    if question:
        parts.append(f"Question:\n{question}")

    metric = graph_result.get("detected_metric")
    if metric:
        parts.append(f"Detected metric:\n{metric}")

    financial_facts = graph_result.get(
        "financial_facts", []
    )
    if financial_facts:
        fact_text = []

        for fact in financial_facts:
            line = (
                f"metric={fact.get('metric')}, "
                f"value={fact.get('value')}, "
                f"unit={fact.get('unit')}, "
                f"period={fact.get('period')}, "
                f"page={fact.get('page')}, "
                f"section={fact.get('section')}, "
                f"confidence={fact.get('confidence')}"
            )
            fact_text.append(line)

        parts.append(
            "FINANCIAL FACTS:\n" + "\n".join(fact_text)
        )

    entities = graph_result.get("entities", [])
    if entities:
        entity_text = []

        for entity_result in entities:
            entity = entity_result.get("entity", {})
            if not entity:
                continue
            entity_name = entity.get("entity")
            if not entity_name:
                continue
            entity_text.append(f"Entity: {entity_name}")
            for rel in entity.get("outgoing", []):
                entity_text.append(
                    f"  OUTGOING: "
                    f"{rel.get('relationship')} -> "
                    f"{rel.get('target')}"
                )
            for rel in entity.get("incoming", []):
                entity_text.append(
                    f"  INCOMING: "
                    f"{rel.get('source')} -> "
                    f"{rel.get('relationship')}"
                )

        if entity_text:
            parts.append(
                "ENTITIES:\n" + "\n".join(entity_text)
            )

    semantic_paths = graph_result.get("semantic_paths", [])
    if semantic_paths:
        path_text = []

        for path_data in semantic_paths:
            nodes_list = path_data.get(
                "nodes", path_data.get("path", [])
            )
            relationships = path_data.get(
                "relationships", []
            )
            chain_parts = []
            for i, node in enumerate(nodes_list):
                chain_parts.append(str(node))
                if i < len(relationships):
                    chain_parts.append(
                        f"--[{relationships[i]}]-->"
                    )
            path_text.append(
                "PATH: " + " ".join(chain_parts)
            )

        parts.append(
            "GRAPH SEMANTIC PATHS:\n"
            + "\n".join(path_text)
        )

    paths = graph_result.get("paths", [])
    if paths:
        path_text = []

        for path_data in paths:
            nodes_list = path_data.get("path", [])
            relationships = path_data.get(
                "relationships", []
            )
            chain_parts = []
            for i, node in enumerate(nodes_list):
                chain_parts.append(str(node))
                if i < len(relationships):
                    chain_parts.append(
                        f"--[{relationships[i]}]-->"
                    )
            path_text.append(
                "PATH: " + " ".join(chain_parts)
            )

        parts.append(
            "GRAPH PATHS:\n" + "\n".join(path_text)
        )

    return "\n\n".join(parts)


def answer_question(
    question: str,
    top_k: int = 5,
    force_type=None,
):
    """
    End-to-end hybrid RAG answer generation.

    1. Classify question
    2. Retrieve graph context (Neo4j)
    3. Retrieve vector context (FAISS)
    4. Fuse context
    5. Send to Ollama
    6. Return structured result
    """

    retriever = HybridRetriever()

    generator = AnswerGenerator()

    try:

        result = retriever.retrieve(
            question,
            top_k=top_k,
            force_type=force_type,
        )

        context = build_context(result)

        answer = generator.generate(context)

        # Build the structured response
        graph_context = result.get(
            "graph_context", {}
        )
        vector_context = result.get(
            "vector_context", []
        )

        return {
            "answer": answer,
            "question": question,
            "question_type": result.get(
                "question_type"
            ),
            "context": context,
            "financial_facts": graph_context.get(
                "financial_facts", []
            ),
            "entities": graph_context.get(
                "entities", []
            ),
            "graph_paths": graph_context.get(
                "semantic_paths",
                graph_context.get("paths", []),
            ),
            "document_chunks": vector_context,
            "graph_context": graph_context,
        }

    finally:

        try:
            retriever.close()
        except Exception:
            pass


if __name__ == "__main__":

    print("=" * 80)
    print("LTM HYBRID GRAPH + VECTOR RAG")
    print("=" * 80)

    while True:

        question = input(
            "\nQuestion (or 'exit'): "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:
            break

        if not question:
            continue

        try:

            result = answer_question(question)

            print("\n" + "=" * 80)
            print(
                f"ANSWER ({result['question_type']})"
            )
            print("=" * 80)

            print(result["answer"])

            print("\n" + "=" * 80)
            print("RECONSTRUCTED CONTEXT")
            print("=" * 80)

            print(result["context"])

        except Exception as e:

            print("\nERROR:")
            print(str(e))
