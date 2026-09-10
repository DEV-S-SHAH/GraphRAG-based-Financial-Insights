"""
Initialize PostgreSQL + pgvector and Neo4j Knowledge Graph.

Usage:
    python scripts/init_db.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.postgres_client import PostgresVectorClient
from database.financial_knowledge_graph import FinancialKnowledgeGraph


def main():
    print("=" * 60)
    print("FINANCIAL GRAPHRAG: DATABASE INITIALIZATION")
    print("=" * 60)

    # 1. PostgreSQL + pgvector
    print("\n[1/2] Initializing PostgreSQL + pgvector...")
    pg = PostgresVectorClient()
    stats = pg.get_stats()
    print("  PostgreSQL connection successful.")
    print("  pgvector extension and vector_chunks verified.")
    print(f"  Current DB Stats: {stats}")

    # 2. Neo4j Constraints and Indexes
    print("\n[2/2] Initializing Neo4j Schema & Constraints...")
    kg = FinancialKnowledgeGraph()
    kg.init_constraints_and_indexes()
    kg.build_full_graph()
    kg_summary = kg.get_graph_summary()
    print("  Neo4j connection successful.")
    print(f"  Total Nodes: {kg_summary['total_nodes']:,}")
    print(f"  Total Relationships: {kg_summary['total_relationships']:,}")
    kg.close()

    print("\n" + "=" * 60)
    print("DATABASE INITIALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
