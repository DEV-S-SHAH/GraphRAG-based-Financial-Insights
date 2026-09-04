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

    # Support NEO4J_AUTH=neo4j/password123
    if not password:
        auth = os.getenv("NEO4J_AUTH", "")

        if "/" in auth:
            user, password = auth.split("/", 1)

    if not password:
        raise ValueError("Neo4j password is missing.")

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password),
    )

    driver.verify_connectivity()

    return driver


def find_entity(tx, query):
    """
    Find entities whose names approximately match
    words from the user's query.
    """

    result = tx.run(
        """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query)
           OR toLower($query) CONTAINS toLower(e.name)
        RETURN e.name AS name
        ORDER BY size(e.name) DESC
        LIMIT 10
        """,
        query=query,
    )

    return [record["name"] for record in result]


def retrieve_paths(tx, entity_name):
    """
    Retrieve up to 3-hop paths starting from
    the identified entity.
    """

    result = tx.run(
        """
        MATCH p=(e:Entity)-[*1..3]->(target)
        WHERE e.name = $entity_name
        RETURN
            [n IN nodes(p) |
                coalesce(n.name, n.metric, n.ticker)
            ] AS path,

            [r IN relationships(p) |
                type(r)
            ] AS relationships
        LIMIT 50
        """,
        entity_name=entity_name,
    )

    return [
        {
            "path": record["path"],
            "relationships": record["relationships"],
        }
        for record in result
    ]


def retrieve_financial_facts(tx, entity_name):
    """
    Retrieve financial facts connected to the
    semantic entity.
    """

    result = tx.run(
        """
        MATCH (e:Entity {name: $entity_name})
              -[*1..3]-
              (m:Metric)
              -[:HAS_FACT]->
              (f)
        RETURN DISTINCT
            m.name AS metric,
            f.value AS value,
            f.unit AS unit
        LIMIT 30
        """,
        entity_name=entity_name,
    )

    return [
        {
            "metric": record["metric"],
            "value": record["value"],
            "unit": record["unit"],
        }
        for record in result
    ]


def retrieve_company_facts(tx):
    """
    Retrieve the main financial facts for LTM.
    """

    result = tx.run(
        """
        MATCH (c:Company {ticker: "LTM"})
              -[*1..3]-
              (f:FinancialFact)

        RETURN DISTINCT
            f.metric AS metric,
            f.value AS value,
            f.unit AS unit
        ORDER BY f.metric
        LIMIT 100
        """
    )

    return [
        {
            "metric": record["metric"],
            "value": record["value"],
            "unit": record["unit"],
        }
        for record in result
    ]


def retrieve(query):
    """
    Main Graph RAG retrieval function.
    """

    driver = create_driver()

    response = {
        "query": query,
        "entities": [],
        "paths": [],
        "financial_facts": [],
    }

    with driver.session() as session:

        # --------------------------------------------------
        # 1. Entity identification
        # --------------------------------------------------

        entities = session.execute_read(
            find_entity,
            query,
        )

        response["entities"] = entities

        # --------------------------------------------------
        # 2. Multi-hop graph traversal
        # --------------------------------------------------

        for entity in entities:

            paths = session.execute_read(
                retrieve_paths,
                entity,
            )

            for path in paths:

                response["paths"].append(
                    {
                        "entity": entity,
                        "path": path["path"],
                        "relationships": path["relationships"],
                    }
                )

        # --------------------------------------------------
        # 3. Financial facts
        # --------------------------------------------------

        for entity in entities:

            facts = session.execute_read(
                retrieve_financial_facts,
                entity,
            )

            response["financial_facts"].extend(facts)

        # --------------------------------------------------
        # 4. If no specific entity was found,
        #    retrieve company-level financial facts.
        # --------------------------------------------------

        if not entities:

            response["financial_facts"] = session.execute_read(
                retrieve_company_facts
            )

    driver.close()

    return response


def print_results(response):

    print()
    print("=" * 80)
    print("GRAPH RAG RETRIEVAL")
    print("=" * 80)

    print()
    print("QUESTION:")
    print(response["query"])

    print()
    print("ENTITIES:")
    print("-" * 80)

    for entity in response["entities"]:
        print("  ", entity)

    print()
    print("GRAPH PATHS:")
    print("-" * 80)

    for item in response["paths"]:

        path = item["path"]
        relationships = item["relationships"]

        print()

        for i, node in enumerate(path):

            print(node, end="")

            if i < len(relationships):
                print(
                    " --[%s]--> "
                    % relationships[i],
                    end="",
                )

        print()

    print()
    print("FINANCIAL FACTS:")
    print("-" * 80)

    seen = set()

    for fact in response["financial_facts"]:

        key = (
            fact["metric"],
            fact["value"],
            fact["unit"],
        )

        if key in seen:
            continue

        seen.add(key)

        print(
            "%-35s = %-12s %s"
            % (
                fact["metric"],
                fact["value"],
                fact["unit"],
            )
        )

    print()
    print("=" * 80)
    print("RETRIEVAL COMPLETE")
    print("=" * 80)


if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    ).strip()

    result = retrieve(question)

    print_results(result)