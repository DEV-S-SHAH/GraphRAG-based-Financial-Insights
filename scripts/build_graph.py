"""
Build & Enrich Neo4j Financial Knowledge Graph CLI.

Creates:
- Company, AnnualReports, ReportingPeriods, SourcePages
- FinancialMetrics, FinancialValues with provenance
- BusinessSegments, Geographies, Risks, Strategies, Executives
- Semantic entity relationships: DRIVES, IMPROVES, IMPACTS, SUPPORTS, ENABLES

Usage:
    python scripts/build_graph.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.financial_knowledge_graph import FinancialKnowledgeGraph
import database.load_semantic_ground_truth as ground_truth_loader


def main():
    print("=" * 65)
    print("BUILDING FINANCIAL KNOWLEDGE GRAPH IN NEO4J")
    print("=" * 65)

    kg = FinancialKnowledgeGraph()
    print("\n[1/3] Initializing constraints & schema indexes...")
    kg.init_constraints_and_indexes()

    print("[2/3] Building full financial ontology & multi-year facts...")
    kg.build_full_graph()

    print("[3/3] Merging semantic ground-truth entities & relationships...")
    driver = kg.driver
    ground_truth_loader.load_missing_entities(driver)
    ground_truth_loader.load_relationships(driver)

    summary = kg.get_graph_summary()
    print("\n" + "=" * 65)
    print("KNOWLEDGE GRAPH BUILD COMPLETE")
    print(f"  • Total Nodes:         {summary['total_nodes']:,}")
    print(f"  • Total Relationships: {summary['total_relationships']:,}")
    print("=" * 65)
    kg.close()


if __name__ == "__main__":
    main()
