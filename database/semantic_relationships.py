import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


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

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password)
    )

    driver.verify_connectivity()

    return driver


RELATIONSHIPS = [
    (
        "AI",
        "digital transformation",
        "SUPPORTS",
    ),
    (
        "AI platforms",
        "digital transformation",
        "ENABLES",
    ),
    (
        "ecosystem partnerships",
        "AI platforms",
        "SUPPORTS",
    ),
    (
        "capability building",
        "AI platforms",
        "SUPPORTS",
    ),
    (
        "cost optimization",
        "operational efficiency",
        "IMPROVES",
    ),
    (
        "Fit4Future",
        "cost optimization",
        "DRIVES",
    ),
    (
        "Fit4Future",
        "operational efficiency",
        "IMPROVES",
    ),
    (
        "operational efficiency",
        "ebitda_margin",
        "SUPPORTS",
    ),
    (
        "pyramid optimization",
        "operating leverage",
        "IMPROVES",
    ),
    (
        "operating leverage",
        "ebitda_margin",
        "SUPPORTS",
    ),
    (
        "large deal wins",
        "order_inflow",
        "INCREASES",
    ),
    (
        "client demand",
        "order_inflow",
        "SUPPORTS",
    ),
    (
        "transformation initiatives",
        "order_inflow",
        "SUPPORTS",
    ),
    (
        "recurring revenues",
        "revenue",
        "SUPPORTS",
    ),
    (
        "New Horizons",
        "Fit4Future",
        "FOLLOWS",
    ),
    (
        "currency risk",
        "revenue_growth_usd",
        "IMPACTS",
    ),
    (
        "labor codes",
        "pat_margin",
        "IMPACTS",
    ),
    (
        "capital allocation",
        "AI platforms",
        "FUNDS",
    ),
]


def create_relationship(tx, source, target, relationship):
    query = """
    MATCH (a {name: $source})
    MATCH (b {name: $target})
    MERGE (a)-[r:%s]->(b)
    RETURN a.name AS source,
           type(r) AS relationship,
           b.name AS target
    """ % relationship

    result = tx.run(
        query,
        source=source,
        target=target,
    )

    return result.single()


def load_relationships():

    driver = create_driver()

    print("=" * 80)
    print("FINANCIAL KNOWLEDGE GRAPH: SEMANTIC RELATIONSHIP LOADER")
    print("=" * 80)

    with driver.session() as session:

        successful = 0

        for source, target, relationship in RELATIONSHIPS:

            result = session.execute_write(
                create_relationship,
                source,
                target,
                relationship,
            )

            if result:
                successful += 1

                print(
                    "[%02d/%02d] %-30s --%-15s--> %s"
                    % (
                        successful,
                        len(RELATIONSHIPS),
                        source,
                        relationship,
                        target,
                    )
                )

            else:
                print(
                    "SKIPPED: %s --%s--> %s"
                    % (
                        source,
                        relationship,
                        target,
                    )
                )

        print()
        print("=" * 80)
        print("SEMANTIC RELATIONSHIP VALIDATION")
        print("=" * 80)

        count = session.run(
            """
            MATCH ()-[r]->()
            RETURN count(r) AS count
            """
        ).single()["count"]

        print("Total relationships:", count)

        print()
        print("Sample semantic relationships:")

        rows = session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE type(r) IN [
                'SUPPORTS',
                'ENABLES',
                'IMPROVES',
                'DRIVES',
                'INCREASES',
                'IMPACTS',
                'FUNDS',
                'FOLLOWS'
            ]
            RETURN a.name AS source,
                   type(r) AS relationship,
                   b.name AS target
            ORDER BY source
            LIMIT 30
            """
        )

        for row in rows:
            print(
                "%-30s | %-15s | %s"
                % (
                    row["source"],
                    row["relationship"],
                    row["target"],
                )
            )

    driver.close()

    print()
    print("=" * 80)
    print("SEMANTIC RELATIONSHIP LOADING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    load_relationships()
