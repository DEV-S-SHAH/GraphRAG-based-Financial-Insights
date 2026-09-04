import os

import psycopg
from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


def test_postgres():
    connection = psycopg.connect(
        host="localhost",
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

    with connection.cursor() as cursor:
        cursor.execute("SELECT version();")
        result = cursor.fetchone()

    connection.close()

    print("PostgreSQL connected:")
    print(result[0])


def test_neo4j():
    uri = "bolt://localhost:7687"
    auth = (
        "neo4j",
        os.getenv("NEO4J_AUTH").split("/", 1)[1],
    )

    driver = GraphDatabase.driver(uri, auth=auth)

    with driver.session() as session:
        result = session.run(
            'RETURN "Neo4j connected" AS message'
        ).single()

    driver.close()

    print(result["message"])


if __name__ == "__main__":
    test_postgres()
    test_neo4j()