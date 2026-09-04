import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# ---------------------------------------------------------
# Neo4j Connection
# ---------------------------------------------------------

def create_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    # Support NEO4J_AUTH=neo4j/password123
    if not password:
        auth = os.getenv("NEO4J_AUTH")

        if auth and "/" in auth:
            user, password = auth.split("/", 1)

    if not password:
        raise ValueError(
            "Neo4j password missing. "
            "Set NEO4J_PASSWORD or NEO4J_AUTH in .env"
        )

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password)
    )

    driver.verify_connectivity()

    return driver


# ---------------------------------------------------------
# Query 1: Company Overview
# ---------------------------------------------------------

def get_company_overview(driver, ticker="LTM"):

    query = """
    MATCH (c:Company {ticker: $ticker})
    OPTIONAL MATCH (c)-[:HAS_METRIC]->(m:Metric)
    OPTIONAL MATCH (m)-[:HAS_VALUE]->(v:Value)

    RETURN
        c.name AS company,
        c.ticker AS ticker,
        count(DISTINCT m) AS metrics,
        count(DISTINCT v) AS values
    """

    with driver.session() as session:
        result = session.run(
            query,
            ticker=ticker
        )

        return result.single()


# ---------------------------------------------------------
# Query 2: Get Financial Metric
# ---------------------------------------------------------

def get_metric(driver, ticker, metric):

    query = """
    MATCH (c:Company {ticker: $ticker})
          -[:HAS_METRIC]->(m:Metric {name: $metric})
          -[:HAS_VALUE]->(v:Value)

    OPTIONAL MATCH (v)-[:FOR_PERIOD]->(p:Period)
    OPTIONAL MATCH (v)-[:FROM_DOCUMENT]->(d:Document)
    OPTIONAL MATCH (v)-[:FROM_PAGE]->(pg:Page)

    RETURN
        c.name AS company,
        c.ticker AS ticker,
        m.name AS metric,
        v.value AS value,
        v.unit AS unit,
        p.label AS period,
        d.document_id AS document,
        pg.page_number AS page,
        v.confidence AS confidence
    ORDER BY p.end DESC
    """

    with driver.session() as session:

        result = session.run(
            query,
            ticker=ticker,
            metric=metric
        )

        return result.data()


# ---------------------------------------------------------
# Query 3: Key Financial Metrics
# ---------------------------------------------------------

def get_key_metrics(driver, ticker="LTM"):

    metrics = [
        "revenue",
        "ebitda",
        "profit_after_tax",
        "eps_diluted",
        "return_on_equity",
        "net_worth",
        "market_capitalization"
    ]

    output = {}

    for metric in metrics:

        rows = get_metric(
            driver,
            ticker,
            metric
        )

        if rows:
            output[metric] = rows[0]

    return output


# ---------------------------------------------------------
# Query 4: Source-Grounded Financial Evidence
# ---------------------------------------------------------

def get_financial_evidence(driver, ticker="LTM"):

    query = """
    MATCH (c:Company {ticker: $ticker})
          -[:HAS_METRIC]->(m:Metric)
          -[:HAS_VALUE]->(v:Value)

    OPTIONAL MATCH (v)-[:FOR_PERIOD]->(p:Period)
    OPTIONAL MATCH (v)-[:FROM_DOCUMENT]->(d:Document)
    OPTIONAL MATCH (v)-[:FROM_PAGE]->(pg:Page)

    RETURN
        m.name AS metric,
        v.value AS value,
        v.unit AS unit,
        p.label AS period,
        d.document_id AS document,
        pg.page_number AS page,
        v.confidence AS confidence

    ORDER BY m.name
    """

    with driver.session() as session:

        result = session.run(
            query,
            ticker=ticker
        )

        return result.data()


# ---------------------------------------------------------
# Query 5: Relationships Around a Metric
# ---------------------------------------------------------

def get_metric_relationships(driver, ticker, metric):

    query = """
    MATCH (c:Company {ticker: $ticker})
          -[r1:HAS_METRIC]->
          (m:Metric {name: $metric})
          -[r2:HAS_VALUE]->
          (v:Value)

    OPTIONAL MATCH (v)-[r3:FOR_PERIOD]->(p:Period)
    OPTIONAL MATCH (v)-[r4:FROM_DOCUMENT]->(d:Document)
    OPTIONAL MATCH (v)-[r5:FROM_PAGE]->(pg:Page)

    RETURN
        c.name AS company,
        type(r1) AS company_relationship,
        m.name AS metric,
        type(r2) AS metric_relationship,
        v.value AS value,
        v.unit AS unit,
        type(r3) AS period_relationship,
        p.label AS period,
        type(r4) AS document_relationship,
        d.document_id AS document,
        type(r5) AS page_relationship,
        pg.page_number AS page
    """

    with driver.session() as session:

        result = session.run(
            query,
            ticker=ticker,
            metric=metric
        )

        return result.data()


# ---------------------------------------------------------
# Main Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print("LTM GRAPH QUERY LAYER")
    print("=" * 80)

    driver = create_driver()

    try:

        # ---------------------------------------------
        # Company overview
        # ---------------------------------------------

        overview = get_company_overview(
            driver,
            "LTM"
        )

        print("\nCOMPANY OVERVIEW")
        print("-" * 80)

        if overview:
            print(
                f"Company: {overview['company']}"
            )

            print(
                f"Ticker: {overview['ticker']}"
            )

            print(
                f"Metrics: {overview['metrics']}"
            )

            print(
                f"Values: {overview['values']}"
            )

        # ---------------------------------------------
        # Key metrics
        # ---------------------------------------------

        print("\nKEY FINANCIAL METRICS")
        print("-" * 80)

        key_metrics = get_key_metrics(
            driver,
            "LTM"
        )

        for metric, data in key_metrics.items():

            print(
                f"{metric:25} "
                f"{data['value']} "
                f"{data['unit']} "
                f"(page {data['page']})"
            )

        # ---------------------------------------------
        # Revenue evidence
        # ---------------------------------------------

        print("\nREVENUE EVIDENCE")
        print("-" * 80)

        revenue = get_metric(
            driver,
            "LTM",
            "revenue"
        )

        for row in revenue:
            print(row)

        # ---------------------------------------------
        # Metric relationships
        # ---------------------------------------------

        print("\nREVENUE RELATIONSHIPS")
        print("-" * 80)

        relationships = get_metric_relationships(
            driver,
            "LTM",
            "revenue"
        )

        for row in relationships:
            print(row)

    finally:

        driver.close()

    print("\n" + "=" * 80)
    print("GRAPH QUERY TEST COMPLETE")
    print("=" * 80)



