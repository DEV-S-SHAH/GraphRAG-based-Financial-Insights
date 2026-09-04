import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


BASE_DIR = Path(__file__).resolve().parent.parent

NARRATIVE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "financial_facts"
    / "LTM_FY2025-26_narrative_facts.json"
)


load_dotenv(BASE_DIR / ".env")


def create_driver():
    """
    Create Neo4j driver using the same credentials
    used by load_graph.py.
    """

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")

    password = os.getenv("NEO4J_PASSWORD")

    # Backward compatibility with NEO4J_AUTH=neo4j/password
    if not password:
        auth = os.getenv("NEO4J_AUTH")

        if auth and "/" in auth:
            auth_user, auth_password = auth.split("/", 1)

            if not os.getenv("NEO4J_USER"):
                user = auth_user

            password = auth_password

    if not password:
        raise ValueError(
            "Neo4j password missing. Set NEO4J_PASSWORD "
            "or NEO4J_AUTH in .env"
        )

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password),
    )

    driver.verify_connectivity()

    print("Neo4j connection successful.")

    return driver


def load_narrative_data():

    if not NARRATIVE_FILE.exists():
        raise FileNotFoundError(
            f"Narrative facts file not found:\n{NARRATIVE_FILE}"
        )

    with open(NARRATIVE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def create_constraints(driver):

    print("Creating narrative graph constraints...")

    queries = [

        """
        CREATE CONSTRAINT narrative_fact_id_unique IF NOT EXISTS
        FOR (n:NarrativeFact)
        REQUIRE n.fact_id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT narrative_metric_name_unique IF NOT EXISTS
        FOR (n:NarrativeMetric)
        REQUIRE n.name IS UNIQUE
        """,

    ]

    with driver.session() as session:

        for query in queries:
            session.run(query)


def create_narrative_fact(
    tx,
    company_name,
    ticker,
    reporting_period,
    fact,
):

    fact_id = fact.get("fact_id")

    if not fact_id:
        # Generate a stable ID if the extractor did not provide one.
        fact_id = (
            f"{ticker}_"
            f"{fact['metric']}_"
            f"{reporting_period}"
        )

    metric = fact["metric"]
    value = fact["value"]
    unit = fact.get("unit")

    source = fact.get("source", {})

    page = source.get("page")
    section = source.get("section")

    query = """

    MERGE (c:Company {
        ticker: $ticker
    })

    SET c.name = $company_name

    MERGE (p:ReportingPeriod {
        label: $reporting_period
    })

    MERGE (m:NarrativeMetric {
        name: $metric
    })

    MERGE (f:NarrativeFact {
        fact_id: $fact_id
    })

    SET
        f.metric = $metric,
        f.value = $value,
        f.unit = $unit,
        f.page = $page,
        f.section = $section,
        f.reporting_period = $reporting_period

    MERGE (c)-[:HAS_PERIOD]->(p)

    MERGE (c)-[:HAS_NARRATIVE_METRIC]->(m)

    MERGE (m)-[:HAS_FACT]->(f)

    MERGE (f)-[:FOR_PERIOD]->(p)

    RETURN f.fact_id AS fact_id

    """

    result = tx.run(
        query,
        company_name=company_name,
        ticker=ticker,
        reporting_period=reporting_period,
        metric=metric,
        value=value,
        unit=unit,
        page=page,
        section=section,
        fact_id=fact_id,
    )

    return result.single()["fact_id"]


def load_graph():

    print("=" * 80)
    print("LTM NARRATIVE GRAPH LOADER")
    print("=" * 80)

    data = load_narrative_data()

    facts = data["facts"]

    company = data.get(
        "company",
        "LTM Limited",
    )

    ticker = data.get(
        "ticker",
        "LTM",
    )

    reporting_period = data.get(
        "reporting_period",
        "FY2025-26",
    )

    print(f"Company: {company}")
    print(f"Ticker: {ticker}")
    print(f"Period: {reporting_period}")
    print(f"Narrative facts: {len(facts)}")

    driver = create_driver()

    try:

        create_constraints(driver)

        print("\nLoading narrative facts...")

        with driver.session() as session:

            for index, fact in enumerate(facts, start=1):

                fact_id = session.execute_write(
                    create_narrative_fact,
                    company,
                    ticker,
                    reporting_period,
                    fact,
                )

                print(
                    f"[{index:02d}/{len(facts)}] "
                    f"{fact['metric']:<40} "
                    f"= {fact['value']} "
                    f"{fact.get('unit', '')}"
                )

        validate_graph(driver)

    finally:

        driver.close()

    print("\n" + "=" * 80)
    print("NARRATIVE GRAPH LOADING COMPLETE")
    print("=" * 80)


def validate_graph(driver):

    print("\n" + "=" * 80)
    print("NARRATIVE GRAPH VALIDATION")
    print("=" * 80)

    queries = {

        "companies": """
            MATCH (n:Company)
            RETURN count(n) AS count
        """,

        "narrative_metrics": """
            MATCH (n:NarrativeMetric)
            RETURN count(n) AS count
        """,

        "narrative_facts": """
            MATCH (n:NarrativeFact)
            RETURN count(n) AS count
        """,

        "periods": """
            MATCH (n:ReportingPeriod)
            RETURN count(n) AS count
        """,

        "relationships": """
            MATCH ()-[r]->()
            RETURN count(r) AS count
        """,

    }

    with driver.session() as session:

        for name, query in queries.items():

            result = session.run(query).single()

            print(
                f"{name:<25} "
                f"{result['count']}"
            )


if __name__ == "__main__":
    load_graph()