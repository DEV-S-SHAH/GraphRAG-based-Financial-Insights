"""
Retrieval Diagnostic & Test CLI.

Usage:
    python scripts/test_retrieval.py --query "What was the revenue and EBITDA trend?"
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.financial_qa_engine import FinancialQAEngine
from rag.hybrid_router import RetrievalStrategy


def main():
    parser = argparse.ArgumentParser(description="Test Financial GraphRAG retrieval.")
    parser.add_argument(
        "query_pos",
        nargs="?",
        default=None,
        help="Query to evaluate (positional)",
    )
    parser.add_argument(
        "--query",
        "-q",
        default=None,
        help="Query to evaluate (flag)",
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "vector", "graph", "hybrid"],
        default="auto",
        help="Retrieval strategy to use",
    )

    args = parser.parse_args()
    query_text = args.query or args.query_pos or "What was LTIMindtree revenue and EBITDA in FY2023-24?"

    strat_map = {
        "auto": None,
        "vector": RetrievalStrategy.VECTOR_SEARCH,
        "graph": RetrievalStrategy.CYPHER_STRUCTURED,
        "hybrid": RetrievalStrategy.HYBRID_FUSION,
    }

    qa = FinancialQAEngine()
    print("=" * 70)
    print(f"QUERY: {query_text}")
    print(f"STRATEGY: {args.strategy}")
    print("=" * 70)

    resp = qa.answer_question(query_text, strategy=strat_map[args.strategy])

    print(f"\n[ROUTING] Strategy Applied: {resp.strategy_used}")
    print(f"[METRICS] Latency: {resp.llm_response.latency_seconds}s | Prompt Tokens: {resp.llm_response.prompt_tokens} | Completion Tokens: {resp.llm_response.completion_tokens}")

    print("\n" + "=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(resp.answer)

    print("\n" + "=" * 70)
    print(f"CITATIONS ({len(resp.citations)}):")
    print("=" * 70)
    for c in resp.citations[:5]:
        print(f" - [{c.get('fiscal_year', 'General')}] {c.get('source_type')} (Page {c.get('page', '?')}, {c.get('section', '')}): {c.get('claim', '')[:80]}...")


if __name__ == "__main__":
    main()
