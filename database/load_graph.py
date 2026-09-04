"""
LTM Financial Graph Loader

Loads canonical financial facts into Neo4j.

Graph structure:

(:Company)
    -[:REPORTED]->(:FinancialMetric)
    -[:HAS_VALUE]->(:FinancialValue)
    -[:FOR_PERIOD]->(:ReportingPeriod)
    -[:SUPPORTED_BY]->(:SourceDocument)
    -[:HAS_PAGE]->(:SourcePage)

The loader is designed to be:
- deterministic
- idempotent
- provenance-aware
- safe to run multiple times
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_facts"
    / "LTM_FY2025-26_canonical.json"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


def get_neo4j_config() -> Tuple[str, str, str]:
    """
    Read Neo4j configuration.

    Preferred configuration:

        NEO4J_URI=bolt://localhost:7687
        NEO4J_USER=neo4j
        NEO4J_PASSWORD=password123

    Existing project configuration also supports:

        NEO4J_AUTH=neo4j/password123
    """

    uri = os.getenv(
        "NEO4J_URI",
        "bolt://localhost:7687"
    )

    user = os.getenv(
        "NEO4J_USER",
        "neo4j"
    )

    password = os.getenv(
        "NEO4J_PASSWORD"
    )

    # --------------------------------------------------------
    # Backward compatibility with docker-compose configuration
    # --------------------------------------------------------

    if not password:

        neo4j_auth = os.getenv(
            "NEO4J_AUTH"
        )

        if neo4j_auth:

            if "/" not in neo4j_auth:
                raise ValueError(
                    "Invalid NEO4J_AUTH format. "
                    "Expected: username/password"
                )

            user, password = neo4j_auth.split(
                "/",
                1
            )

    if not password:

        raise ValueError(
            "Neo4j password is missing.\n"
            "Set NEO4J_PASSWORD in .env or use:\n"
            "NEO4J_AUTH=neo4j/your_password"
        )

    return uri, user, password


# ============================================================
# DRIVER
# ============================================================

def create_driver():
    """
    Create Neo4j driver.
    """

    uri, user, password = get_neo4j_config()

    driver = GraphDatabase.driver(
        uri,
        auth=(
            user,
            password
        )
    )

    return driver


# ============================================================
# LOAD CANONICAL DATA
# ============================================================

def load_canonical_data() -> Dict[str, Any]:
    """
    Load canonical financial facts JSON.
    """

    if not CANONICAL_FILE.exists():

        raise FileNotFoundError(
            f"Canonical financial facts not found:\n"
            f"{CANONICAL_FILE}"
        )

    with open(
        CANONICAL_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if "facts" not in data:

        raise ValueError(
            "Canonical JSON does not contain 'facts'."
        )

    return data


# ============================================================
# DATABASE CONSTRAINTS
# ============================================================

def create_constraints(driver) -> None:
    """
    Create uniqueness constraints.

    Neo4j 5.x syntax is used.
    """

    constraints = [

        """
        CREATE CONSTRAINT company_ticker_unique IF NOT EXISTS
        FOR (c:Company)
        REQUIRE c.ticker IS UNIQUE
        """,

        """
        CREATE CONSTRAINT metric_name_unique IF NOT EXISTS
        FOR (m:FinancialMetric)
        REQUIRE m.name IS UNIQUE
        """,

        """
        CREATE CONSTRAINT value_fact_id_unique IF NOT EXISTS
        FOR (v:FinancialValue)
        REQUIRE v.fact_id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT period_label_unique IF NOT EXISTS
        FOR (p:ReportingPeriod)
        REQUIRE p.label IS UNIQUE
        """,

        """
        CREATE CONSTRAINT document_id_unique IF NOT EXISTS
        FOR (d:SourceDocument)
        REQUIRE d.document_id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT page_unique IF NOT EXISTS
        FOR (p:SourcePage)
        REQUIRE p.page_key IS UNIQUE
        """
    ]

    with driver.session() as session:

        for query in constraints:

            session.run(query)


# ============================================================
# CREATE COMPANY
# ============================================================

def create_company(
    driver,
    company: str,
    ticker: str
) -> None:

    query = """
    MERGE (c:Company {
        ticker: $ticker
    })
    SET c.name = $company
    """

    with driver.session() as session:

        session.run(
            query,
            company=company,
            ticker=ticker
        )


# ============================================================
# CREATE REPORTING PERIOD
# ============================================================

def create_reporting_period(
    driver,
    period: Dict[str, Any]
) -> None:

    query = """
    MERGE (p:ReportingPeriod {
        label: $label
    })
    SET
        p.start = $start,
        p.end = $end
    """

    with driver.session() as session:

        session.run(
            query,
            label=period["label"],
            start=period.get("start"),
            end=period.get("end")
        )


# ============================================================
# CREATE SOURCE DOCUMENT
# ============================================================

def create_source_document(
    driver,
    document_id: str
) -> None:

    query = """
    MERGE (d:SourceDocument {
        document_id: $document_id
    })
    """

    with driver.session() as session:

        session.run(
            query,
            document_id=document_id
        )


# ============================================================
# CREATE SOURCE PAGE
# ============================================================

def create_source_page(
    driver,
    document_id: str,
    page: int,
    section: str = ""
) -> None:

    page_key = f"{document_id}:{page}"

    query = """
    MERGE (p:SourcePage {
        page_key: $page_key
    })
    SET
        p.document_id = $document_id,
        p.page = $page,
        p.section = $section

    WITH p

    MATCH (d:SourceDocument {
        document_id: $document_id
    })

    MERGE (d)-[:HAS_PAGE]->(p)
    """

    with driver.session() as session:

        session.run(
            query,
            page_key=page_key,
            document_id=document_id,
            page=page,
            section=section or ""
        )


# ============================================================
# CREATE FINANCIAL METRIC
# ============================================================

def create_metric(
    driver,
    metric: str
) -> None:

    query = """
    MERGE (m:FinancialMetric {
        name: $metric
    })
    """

    with driver.session() as session:

        session.run(
            query,
            metric=metric
        )


# ============================================================
# CREATE FINANCIAL VALUE
# ============================================================

def create_financial_value(
    driver,
    fact: Dict[str, Any]
) -> None:

    source = fact.get(
        "source",
        {}
    )

    period = fact.get(
        "period",
        {}
    )

    fact_id = fact["fact_id"]

    query = """
    MERGE (v:FinancialValue {
        fact_id: $fact_id
    })

    SET
        v.metric = $metric,
        v.value = $value,
        v.unit = $unit,
        v.company = $company,
        v.ticker = $ticker,
        v.statement_type = $statement_type,
        v.confidence = $confidence,
        v.extraction_method = $extraction_method

    WITH v

    MATCH (c:Company {
        ticker: $ticker
    })

    MATCH (m:FinancialMetric {
        name: $metric
    })

    MATCH (p:ReportingPeriod {
        label: $period_label
    })

    MATCH (d:SourceDocument {
        document_id: $document_id
    })

    MERGE (c)-[:REPORTED]->(m)

    MERGE (c)-[:HAS_VALUE]->(v)

    MERGE (v)-[:FOR_METRIC]->(m)

    MERGE (v)-[:FOR_PERIOD]->(p)

    MERGE (v)-[:SUPPORTED_BY]->(d)
    """

    parameters = {
        "fact_id": fact_id,
        "metric": fact["metric"],
        "value": fact["value"],
        "unit": fact.get("unit"),
        "company": fact.get("company"),
        "ticker": fact.get("ticker"),
        "statement_type": fact.get(
            "statement_type",
            "consolidated"
        ),
        "confidence": fact.get(
            "confidence",
            0.0
        ),
        "extraction_method": fact.get(
            "extraction_method"
        ),
        "period_label": period.get(
            "label"
        ),
        "document_id": source.get(
            "document_id"
        )
    }

    with driver.session() as session:

        session.run(
            query,
            **parameters
        )


# ============================================================
# LINK FINANCIAL VALUE TO SOURCE PAGE
# ============================================================

def link_value_to_source_page(
    driver,
    fact: Dict[str, Any]
) -> None:

    source = fact.get(
        "source",
        {}
    )

    document_id = source.get(
        "document_id"
    )

    page = source.get(
        "page"
    )

    if not document_id or page is None:
        return

    page_key = f"{document_id}:{page}"

    query = """
    MATCH (v:FinancialValue {
        fact_id: $fact_id
    })

    MATCH (p:SourcePage {
        page_key: $page_key
    })

    MERGE (v)-[:FOUND_ON]->(p)
    """

    with driver.session() as session:

        session.run(
            query,
            fact_id=fact["fact_id"],
            page_key=page_key
        )


# ============================================================
# CREATE FACT RELATIONSHIPS
# ============================================================

def create_fact_relationships(
    driver,
    facts: List[Dict[str, Any]]
) -> None:
    """
    Creates relationships between related financial facts.

    Example:

        Revenue
           |
           | RELATED_TO
           v
        EBITDA

    These relationships are intentionally lightweight at this
    stage. The semantic relationship layer will be expanded
    later for GraphRAG.
    """

    metric_groups = {
        "profitability": {
            "revenue",
            "ebitda",
            "profit_after_tax",
            "ebit",
            "profit_before_tax",
            "eps_basic",
            "eps_diluted",
        },

        "capital_efficiency": {
            "return_on_equity",
            "return_on_capital_employed",
            "net_worth",
        },

        "shareholder_returns": {
            "dividend_paid",
            "eps_basic",
            "eps_diluted",
        }
    }

    # Build metric -> groups mapping

    metric_to_groups = {}

    for group_name, metrics in metric_groups.items():

        for metric in metrics:

            metric_to_groups.setdefault(
                metric,
                []
            ).append(
                group_name
            )

    query = """
    MATCH (a:FinancialValue)
    MATCH (b:FinancialValue)

    WHERE
        a.fact_id < b.fact_id
        AND a.ticker = b.ticker
        AND a.metric IN $metrics
        AND b.metric IN $metrics

    WITH a, b

    MERGE (a)-[:RELATED_FINANCIAL_FACT]->(b)
    """

    metrics = list(
        metric_to_groups.keys()
    )

    with driver.session() as session:

        session.run(
            query,
            metrics=metrics
        )


# ============================================================
# VALIDATE GRAPH
# ============================================================

def validate_graph(driver) -> None:

    queries = {

        "companies": """
        MATCH (c:Company)
        RETURN count(c) AS count
        """,

        "metrics": """
        MATCH (m:FinancialMetric)
        RETURN count(m) AS count
        """,

        "values": """
        MATCH (v:FinancialValue)
        RETURN count(v) AS count
        """,

        "periods": """
        MATCH (p:ReportingPeriod)
        RETURN count(p) AS count
        """,

        "documents": """
        MATCH (d:SourceDocument)
        RETURN count(d) AS count
        """,

        "pages": """
        MATCH (p:SourcePage)
        RETURN count(p) AS count
        """,

        "relationships": """
        MATCH ()-[r]->()
        RETURN count(r) AS count
        """
    }

    print()
    print("=" * 80)
    print("GRAPH VALIDATION")
    print("=" * 80)

    with driver.session() as session:

        for name, query in queries.items():

            result = session.run(
                query
            ).single()

            count = result["count"]

            print(
                f"{name:<20} {count}"
            )


# ============================================================
# DISPLAY SAMPLE GRAPH DATA
# ============================================================

def display_sample_facts(driver) -> None:

    query = """
    MATCH
        (c:Company)-[:HAS_VALUE]->(v:FinancialValue)
        -[:FOR_METRIC]->(m:FinancialMetric)

    RETURN
        c.ticker AS ticker,
        m.name AS metric,
        v.value AS value,
        v.unit AS unit

    ORDER BY metric

    LIMIT 15
    """

    print()
    print("=" * 80)
    print("SAMPLE GRAPH DATA")
    print("=" * 80)

    with driver.session() as session:

        result = session.run(
            query
        )

        for record in result:

            print(
                f"{record['ticker']:<6} | "
                f"{record['metric']:<30} | "
                f"{record['value']} "
                f"{record['unit'] or ''}"
            )


# ============================================================
# MAIN GRAPH LOADER
# ============================================================

def load_graph() -> None:

    data = load_canonical_data()

    facts = data["facts"]

    print(
        f"Loading {len(facts)} financial facts..."
    )

    driver = create_driver()

    try:

        # ----------------------------------------------------
        # 1. Verify Neo4j connection
        # ----------------------------------------------------

        driver.verify_connectivity()

        print(
            "Neo4j connection successful."
        )

        # ----------------------------------------------------
        # 2. Create constraints
        # ----------------------------------------------------

        print(
            "Creating graph constraints..."
        )

        create_constraints(
            driver
        )

        # ----------------------------------------------------
        # 3. Company
        # ----------------------------------------------------

        company = data.get(
            "company"
        )

        ticker = data.get(
            "ticker"
        )

        create_company(
            driver,
            company,
            ticker
        )

        # ----------------------------------------------------
        # 4. Reporting period
        # ----------------------------------------------------

        reporting_period = data.get(
            "reporting_period"
        )

        first_period = None

        for fact in facts:

            if fact.get("period"):

                first_period = fact["period"]

                break

        if first_period:

            create_reporting_period(
                driver,
                first_period
            )

        # ----------------------------------------------------
        # 5. Source document
        # ----------------------------------------------------

        document_id = data.get(
            "document_id"
        )

        create_source_document(
            driver,
            document_id
        )

        # ----------------------------------------------------
        # 6. Load facts
        # ----------------------------------------------------

        for index, fact in enumerate(
            facts,
            start=1
        ):

            metric = fact["metric"]

            create_metric(
                driver,
                metric
            )

            source = fact.get(
                "source",
                {}
            )

            page = source.get(
                "page"
            )

            section = source.get(
                "section",
                ""
            )

            if page is not None:

                create_source_page(
                    driver,
                    source.get(
                        "document_id",
                        document_id
                    ),
                    page,
                    section
                )

            create_financial_value(
                driver,
                fact
            )

            link_value_to_source_page(
                driver,
                fact
            )

            print(
                f"[{index:02d}/{len(facts):02d}] "
                f"Loaded: "
                f"{metric:<30} "
                f"= {fact['value']} "
                f"{fact.get('unit', '')}"
            )

        # ----------------------------------------------------
        # 7. Financial relationships
        # ----------------------------------------------------

        print()
        print(
            "Creating financial relationships..."
        )

        create_fact_relationships(
            driver,
            facts
        )

        # ----------------------------------------------------
        # 8. Validate
        # ----------------------------------------------------

        validate_graph(
            driver
        )

        # ----------------------------------------------------
        # 9. Display sample
        # ----------------------------------------------------

        display_sample_facts(
            driver
        )

        print()
        print("=" * 80)
        print(
            "GRAPH LOADING COMPLETE"
        )
        print("=" * 80)

    finally:

        driver.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    load_graph()