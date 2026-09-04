import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENTITIES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "entities"
    / "LTM_FY2025-26_entities.json"
)

load_dotenv(BASE_DIR / ".env")


# ============================================================
# NEO4J CONNECTION
# ============================================================

def create_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    # Support NEO4J_AUTH=neo4j/password123 as well
    if not password:
        auth = os.getenv("NEO4J_AUTH")

        if auth and "/" in auth:
            auth_user, auth_password = auth.split("/", 1)

            if not user:
                user = auth_user

            password = auth_password

    if not password:
        raise ValueError(
            "NEO4J_PASSWORD is missing from .env"
        )

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password),
    )

    driver.verify_connectivity()

    print("Neo4j connection successful.")

    return driver


# ============================================================
# LOAD JSON
# ============================================================

def load_entities():
    if not ENTITIES_FILE.exists():
        raise FileNotFoundError(
            f"Entity file not found: {ENTITIES_FILE}"
        )

    with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# ============================================================
# CONSTRAINTS
# ============================================================

def create_constraints(session):

    print("Creating entity graph constraints...")

    queries = [

        """
        CREATE CONSTRAINT company_ticker_unique IF NOT EXISTS
        FOR (c:Company)
        REQUIRE c.ticker IS UNIQUE
        """,

        """
        CREATE CONSTRAINT entity_unique IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE e.entity_id IS UNIQUE
        """,

    ]

    for query in queries:
        session.run(query)


# ============================================================
# NORMALIZE ENTITY TYPE
# ============================================================

def normalize_entity_type(entity_type):

    mapping = {
        "strategy": "Strategy",
        "business_theme": "BusinessTheme",
        "program": "Program",
        "capability": "Capability",
        "risk": "Risk",
        "technology": "Technology",
        "initiative": "Initiative",
    }

    return mapping.get(
        entity_type.lower(),
        "Entity",
    )


# ============================================================
# CREATE ENTITY
# ============================================================

def create_entity(session, company, entity):

    entity_type = entity.get("entity_type", "entity")
    name = entity.get("name")

    page = entity.get("page")

    if not name:
        return

    label = normalize_entity_type(entity_type)

    entity_id = (
        f"{company['ticker']}_"
        f"{entity_type}_"
        f"{name.lower().replace(' ', '_')}"
    )

    query = f"""
    MERGE (c:Company {{
        ticker: $ticker
    }})

    SET c.name = $company_name

    MERGE (e:Entity:{label} {{
        entity_id: $entity_id
    }})

    SET
        e.name = $name,
        e.entity_type = $entity_type,
        e.page = $page

    MERGE (c)-[r:HAS_ENTITY]->(e)

    SET r.page = $page
    """

    session.run(
        query,
        ticker=company["ticker"],
        company_name=company["name"],
        entity_id=entity_id,
        name=name,
        entity_type=entity_type,
        page=page,
    )


# ============================================================
# CREATE RELATIONSHIPS BETWEEN ENTITIES
# ============================================================

def create_entity_relationships(session):

    print("Creating entity relationships...")

    # Strategy -> Capability
    session.run(
        """
        MATCH (s:Strategy), (c:Capability)
        WHERE
            toLower(s.name) CONTAINS "ai"
            AND
            toLower(c.name) CONTAINS "ai"
        MERGE (s)-[:ENABLES]->(c)
        """
    )

    # Strategy -> Program
    session.run(
        """
        MATCH (s:Strategy), (p:Program)
        WHERE
            toLower(s.name) CONTAINS "productivity"
            AND
            toLower(p.name) IN ["fit4future", "new horizons"]
        MERGE (s)-[:SUPPORTED_BY]->(p)
        """
    )

    # Strategy -> Business Theme
    session.run(
        """
        MATCH (s:Strategy), (b:BusinessTheme)
        WHERE
            toLower(s.name) = toLower(b.name)
        MERGE (s)-[:RELATES_TO]->(b)
        """
    )

    # Risk -> Strategy
    session.run(
        """
        MATCH (r:Risk), (s:Strategy)
        WHERE
            toLower(r.name) CONTAINS "currency"
            AND
            toLower(s.name) CONTAINS "cost"
        MERGE (r)-[:IMPACTS]->(s)
        """
    )

    # Risk -> Business Theme
    session.run(
        """
        MATCH (r:Risk), (b:BusinessTheme)
        WHERE
            toLower(r.name) CONTAINS "margin"
            AND
            toLower(b.name) CONTAINS "efficiency"
        MERGE (r)-[:IMPACTS]->(b)
        """
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_graph(session):

    print()
    print("=" * 80)
    print("ENTITY GRAPH VALIDATION")
    print("=" * 80)

    queries = {

        "companies": """
            MATCH (c:Company)
            RETURN count(c) AS count
        """,

        "entities": """
            MATCH (e:Entity)
            RETURN count(e) AS count
        """,

        "strategies": """
            MATCH (e:Strategy)
            RETURN count(e) AS count
        """,

        "capabilities": """
            MATCH (e:Capability)
            RETURN count(e) AS count
        """,

        "programs": """
            MATCH (e:Program)
            RETURN count(e) AS count
        """,

        "risks": """
            MATCH (e:Risk)
            RETURN count(e) AS count
        """,

        "business_themes": """
            MATCH (e:BusinessTheme)
            RETURN count(e) AS count
        """,

        "relationships": """
            MATCH ()-[r]->()
            RETURN count(r) AS count
        """,
    }

    for name, query in queries.items():

        result = session.run(query).single()

        print(
            f"{name:<25} {result['count']}"
        )


# ============================================================
# SAMPLE GRAPH
# ============================================================

def show_sample_graph(session):

    print()
    print("=" * 80)
    print("SAMPLE ENTITY GRAPH")
    print("=" * 80)

    result = session.run(
        """
        MATCH (c:Company)-[r:HAS_ENTITY]->(e:Entity)
        RETURN
            c.ticker AS company,
            e.entity_type AS type,
            e.name AS entity,
            e.page AS page
        ORDER BY e.entity_type, e.name
        LIMIT 30
        """
    )

    for record in result:

        print(
            f"{record['company']:<8} | "
            f"{record['type']:<20} | "
            f"{record['entity']:<35} | "
            f"page={record['page']}"
        )


# ============================================================
# MAIN
# ============================================================

def load_graph():

    data = load_entities()

    company = {
        "name": data.get(
            "company",
            "LTM Limited"
        ),
        "ticker": data.get(
            "ticker",
            "LTM"
        ),
    }

    entities = data.get("entities", [])

    print("=" * 80)
    print("LTM ENTITY GRAPH LOADER")
    print("=" * 80)

    print(
        f"Company: {company['name']}"
    )

    print(
        f"Ticker: {company['ticker']}"
    )

    print(
        f"Entities: {len(entities)}"
    )

    driver = create_driver()

    try:

        with driver.session() as session:

            create_constraints(session)

            print()
            print("Loading entities...")

            for index, entity in enumerate(
                entities,
                start=1
            ):

                create_entity(
                    session,
                    company,
                    entity
                )

                print(
                    f"[{index:02d}/{len(entities)}] "
                    f"{entity.get('entity_type', 'entity'):<20} "
                    f"{entity.get('name', '')}"
                )

            create_entity_relationships(
                session
            )

            validate_graph(session)

            show_sample_graph(session)

    finally:

        driver.close()

    print()
    print("=" * 80)
    print("ENTITY GRAPH LOADING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    load_graph()