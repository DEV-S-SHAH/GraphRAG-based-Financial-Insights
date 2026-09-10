"""
Load the semantic ground-truth entities + relationships (from
database/semantic_relationships.py RELATIONSHIPS) into Neo4j.

This complements the multi-year loader so that the graph reflects the
idealized ground-truth semantic graph used by the evaluation framework:

  - Missing semantic nodes (Fit4Future, AI platforms, New Horizons, etc.)
    are created as :Entity nodes so GraphQueryEngine can detect & traverse them.
  - The 18 RELATIONSHIPS edges are merged (idempotent).

Usage:
    python -m database.load_semantic_ground_truth
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from neo4j import GraphDatabase

from database.semantic_relationships import RELATIONSHIPS

load_dotenv(PROJECT_ROOT / ".env")


def create_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        auth = os.getenv("NEO4J_AUTH", "")
        if "/" in auth:
            user, password = auth.split("/", 1)

    if not password:
        raise ValueError("Neo4j password is missing.")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("Neo4j connection successful.")
    return driver


# (name, [labels...], entity_id, entity_type)
MISSING_ENTITIES = [
    ("Fit4Future", ["Program"], "LTIM_program_fit4future", "program"),
    ("New Horizons", ["Program"], "LTIM_program_new_horizons", "program"),
    ("pyramid optimization", ["BusinessTheme"], "LTIM_business_theme_pyramid_optimization", "business_theme"),
    ("operating leverage", ["BusinessTheme"], "LTIM_business_theme_operating_leverage", "business_theme"),
    ("ecosystem partnerships", ["BusinessTheme"], "LTIM_business_theme_ecosystem_partnerships", "business_theme"),
    ("capability building", ["Capability"], "LTIM_capability_capability_building", "capability"),
    ("AI platforms", ["Technology"], "LTIM_technology_ai_platforms", "technology"),
    ("capital allocation", ["BusinessTheme"], "LTIM_business_theme_capital_allocation", "business_theme"),
    ("large deal wins", ["BusinessTheme"], "LTIM_business_theme_large_deal_wins", "business_theme"),
    ("client demand", ["BusinessTheme"], "LTIM_business_theme_client_demand", "business_theme"),
    ("transformation initiatives", ["BusinessTheme"], "LTIM_business_theme_transformation_initiatives", "business_theme"),
    ("recurring revenues", ["BusinessTheme"], "LTIM_business_theme_recurring_revenues", "business_theme"),
    ("labor codes", ["Risk"], "LTIM_risk_labor_codes", "risk"),
    ("order_inflow", ["FinancialMetric", "Metric"], "LTIM_metric_order_inflow", "metric"),
]


def load_missing_entities(driver):
    with driver.session() as session:
        for name, labels, eid, etype in MISSING_ENTITIES:
            label_str = ":".join(labels)
            session.run(
                f"""
                MERGE (e:Entity:{label_str} {{entity_id: $eid}})
                SET e.name = $name, e.entity_type = $etype
                """,
                eid=eid,
                name=name,
                etype=etype,
            )
        print(f"Ensured {len(MISSING_ENTITIES)} semantic entity nodes.")


def create_relationship(session, source, target, relationship):
    query = f"""
    MATCH (a {{name: $source}})
    MATCH (b {{name: $target}})
    MERGE (a)-[r:{relationship}]->(b)
    RETURN a.name AS source, type(r) AS relationship, b.name AS target
    """
    result = session.run(query, source=source, target=target)
    return result.single()


def load_relationships(driver):
    succeeded = 0
    failed = []
    with driver.session() as session:
        for source, target, relationship in RELATIONSHIPS:
            res = create_relationship(session, source, target, relationship)
            if res:
                succeeded += 1
                print(
                    f"  [OK ] {source:28} -{relationship:12}-> {target}"
                )
            else:
                failed.append((source, relationship, target))
                print(
                    f"  [SKIP] {source:28} -{relationship:12}-> {target}  (nodes missing)"
                )
    print(f"\nLoaded {succeeded} relationships, {len(failed)} skipped.")
    if failed:
        print("Skipped:", failed)


def main():
    driver = create_driver()
    try:
        load_missing_entities(driver)
        load_relationships(driver)
    finally:
        driver.close()
    print("\nSemantic ground-truth loading complete.")


if __name__ == "__main__":
    main()
