import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def create_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    # Support existing NEO4J_AUTH=neo4j/password format
    if not password:
        auth = os.getenv("NEO4J_AUTH", "")

        if "/" in auth:
            user, password = auth.split("/", 1)

    if not password:
        raise ValueError("NEO4J_PASSWORD or NEO4J_AUTH is missing")

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password)
    )

    driver.verify_connectivity()
    print("Neo4j connection successful.")

    return driver


def find_financial_metric(driver, metric):
    """
    Retrieve a financial metric and its values,
    period and source page.
    """

    query = """
    MATCH (m:FinancialMetric)
    WHERE toLower(m.name) = toLower($metric)

    OPTIONAL MATCH (v:FinancialValue)-[:FOR_METRIC]->(m)
    OPTIONAL MATCH (v)-[:FOR_PERIOD]->(p:ReportingPeriod)
    OPTIONAL MATCH (v)-[:FOUND_ON]->(page:SourcePage)

    RETURN
        m.name AS metric,
        v.value AS value,
        v.unit AS unit,
        v.confidence AS confidence,
        v.extraction_method AS extraction_method,
        p.label AS period,
        page.page AS page,
        page.section AS section
    ORDER BY page.page
    """

    with driver.session() as session:
        return session.run(
            query,
            metric=metric
        ).data()


def find_entity(driver, entity_name):
    """
    Retrieve an entity and its semantic relationships.
    """

    query = """
    MATCH (e:Entity)
    WHERE toLower(e.name) = toLower($entity_name)

    OPTIONAL MATCH (e)-[r]->(target)
    OPTIONAL MATCH (source)-[r2]->(e)

    RETURN
        e.name AS entity,

        collect(DISTINCT {
            relationship: type(r),
            target: coalesce(
                target.name,
                target.metric
            )
        }) AS outgoing,

        collect(DISTINCT {
            relationship: type(r2),
            source: coalesce(
                source.name,
                source.metric
            )
        }) AS incoming
    """

    with driver.session() as session:
        result = session.run(
            query,
            entity_name=entity_name
        ).single()

        return dict(result) if result else None


def find_entity_paths(driver, entity_name, depth=3):
    """
    Retrieve connected semantic graph paths.
    """

    query = """
    MATCH p=(e:Entity)-[*1..3]->(target)
    WHERE toLower(e.name) = toLower($entity_name)

    RETURN
        [n IN nodes(p) |
            coalesce(
                n.name,
                n.metric,
                n.value
            )
        ] AS path,

        [r IN relationships(p) |
            type(r)
        ] AS relationships

    LIMIT 50
    """

    with driver.session() as session:
        return session.run(
            query,
            entity_name=entity_name
        ).data()


def search_metrics(driver, text):
    """
    Fuzzy search financial metrics.
    """

    query = """
    MATCH (m:FinancialMetric)
    WHERE toLower(m.name) CONTAINS toLower($text)

    RETURN m.name AS metric
    ORDER BY m.name
    """

    with driver.session() as session:
        return session.run(
            query,
            text=text
        ).data()


def search_entities(driver, text):
    """
    Fuzzy search graph entities.
    """

    query = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower($text)

    RETURN
        e.name AS name,
        e.entity_type AS entity_type,
        e.page AS page

    ORDER BY e.name
    """

    with driver.session() as session:
        return session.run(
            query,
            text=text
        ).data()


def retrieve_context(driver, query):
    """
    Basic Graph RAG retrieval.

    Searches for matching entities and financial metrics,
    then retrieves their graph context.
    """

    context = {
        "query": query,
        "entities": [],
        "metrics": [],
        "paths": []
    }

    # Search entities
    entity_results = search_entities(driver, query)

    for entity in entity_results:
        context["entities"].append(entity)

        paths = find_entity_paths(
            driver,
            entity["name"]
        )

        context["paths"].extend(paths)

    # Search metrics
    metric_results = search_metrics(driver, query)

    for metric in metric_results:
        context["metrics"].append(metric)

        facts = find_financial_metric(
            driver,
            metric["metric"]
        )

        context["metrics"][-1]["facts"] = facts

    return context


if __name__ == "__main__":

    driver = create_driver()

    print("=" * 80)
    print("LTM GRAPH RAG RETRIEVER")
    print("=" * 80)

    # Test 1
    print("\n[TEST 1] EBITDA")
    print("-" * 80)

    results = find_financial_metric(
        driver,
        "ebitda"
    )

    for result in results:
        print(result)

    # Test 2
    print("\n[TEST 2] Cost Optimization")
    print("-" * 80)

    entity = find_entity(
        driver,
        "cost optimization"
    )

    print(entity)

    # Test 3
    print("\n[TEST 3] Cost Optimization Paths")
    print("-" * 80)

    paths = find_entity_paths(
        driver,
        "cost optimization"
    )

    for path in paths:
        print(path)

    driver.close()

    print("\n" + "=" * 80)
    print("RETRIEVER TEST COMPLETE")
    print("=" * 80)