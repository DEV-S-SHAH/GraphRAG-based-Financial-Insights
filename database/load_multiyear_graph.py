"""
Phase 7: Multi-year financial knowledge graph loader (Neo4j).

Loads the grounded multi-year pipeline (FY2022-23, FY2023-24, FY2024-25) into
Neo4j in a way that is compatible with the existing GraphQueryEngine retrieval
interface in rag/query_engine.py:

  - (c:Company)-[:HAS_VALUE]->(v:FinancialValue)-[:FOR_METRIC]->(m:FinancialMetric)
  - (v)-[:FOR_PERIOD]->(p:ReportingPeriod)
  - (v)-[:FOUND_ON]->(page:SourcePage)
  - (v)-[:SUPPORTED_BY]->(d:SourceDocument)
  - (c:Company)-[:REPORTED]->(m:FinancialMetric)
  - (:Entity) with semantic relationships: (:Entity)-[:REL_TYPE]->(:Entity | :FinancialMetric)

Semantic relationship types are drawn from the CONTROLLED vocabulary and carry
evidence + fiscal_year + relationship_type (explicit|inferred), so every edge is
provenance-aware and no causality is asserted without grounding.

The loader is:
  - deterministic
  - idempotent (MERGE + CREATE CONSTRAINT IF NOT EXISTS)
  - multi-year (all three reports coexist, distinguished by ReportingPeriod)

Usage:
    python -m database.load_multiyear_graph
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

FACTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_facts"
    / "LTIM_canonical_multiyear.json"
)
ENTITIES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "entities"
    / "LTIM_entities_multiyear.json"
)
RELATIONSHIPS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "relationships"
    / "LTIM_relationships_multiyear.json"
)
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

# Controlled relationship vocabulary (only these may become Neo4j rel types).
CONTROLLED_RELATIONS = {
    "SUPPORTS",
    "DRIVES",
    "ENABLES",
    "IMPROVES",
    "IMPACTS",
    "INCREASES",
    "DECREASES",
    "FUNDS",
    "DEPENDS_ON",
    "RELATED_TO",
    "TARGETS",
    "MITIGATES",
    "CREATES",
    "CONTRIBUTES_TO",
    "FOLLOWS",
}

ENTITY_TYPE_LABELS = {
    "strategy": "Strategy",
    "business_theme": "BusinessTheme",
    "program": "Program",
    "capability": "Capability",
    "risk": "Risk",
    "technology": "Technology",
    "initiative": "Initiative",
    "vertical": "Vertical",
    "geography": "Geography",
}


def get_neo4j_config():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        auth = os.getenv("NEO4J_AUTH", "")
        if auth and "/" in auth:
            user, password = auth.split("/", 1)

    if not password:
        raise ValueError(
            "Neo4j password is missing. Set NEO4J_PASSWORD or NEO4J_AUTH in .env"
        )

    return uri, user, password


def create_driver():
    uri, user, password = get_neo4j_config()
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("Neo4j connection successful.")
    return driver


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Constraints (idempotent)
# ---------------------------------------------------------------------------

def create_constraints(driver) -> None:
    print("Creating graph constraints...")
    constraints = [
        "CREATE CONSTRAINT company_ticker_unique IF NOT EXISTS "
        "FOR (c:Company) REQUIRE c.ticker IS UNIQUE",

        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",

        "CREATE CONSTRAINT financial_metric_name_unique IF NOT EXISTS "
        "FOR (m:FinancialMetric) REQUIRE m.name IS UNIQUE",

        "CREATE CONSTRAINT financial_value_fact_id_unique IF NOT EXISTS "
        "FOR (v:FinancialValue) REQUIRE v.fact_id IS UNIQUE",

        "CREATE CONSTRAINT reporting_period_label_unique IF NOT EXISTS "
        "FOR (p:ReportingPeriod) REQUIRE p.label IS UNIQUE",

        "CREATE CONSTRAINT source_document_id_unique IF NOT EXISTS "
        "FOR (d:SourceDocument) REQUIRE d.document_id IS UNIQUE",

        "CREATE CONSTRAINT source_page_key_unique IF NOT EXISTS "
        "FOR (p:SourcePage) REQUIRE p.page_key IS UNIQUE",

        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
    ]
    with driver.session() as session:
        for q in constraints:
            session.run(q)


# ---------------------------------------------------------------------------
# Clear the previous single-year graph (idempotent rebuild)
# ---------------------------------------------------------------------------

def clear_ltim_nodes(driver) -> None:
    print("Clearing previous LTIM graph nodes...")
    with driver.session() as session:
        # The current pipeline uses ticker "LTIM"; the retired single-year
        # pipeline used "LTM". Remove both so the rebuild is clean.
        session.run("MATCH (c:Company) WHERE c.ticker IN ['LTIM', 'LTM'] DETACH DELETE c")
        # Remove orphaned nodes that belong to the LTM pipeline labels.
        for label in ("Entity", "FinancialMetric", "FinancialValue",
                      "ReportingPeriod", "SourceDocument", "SourcePage",
                      "Chunk", "NarrativeFact", "NarrativeMetric"):
            try:
                session.run(f"MATCH (n:{label}) DETACH DELETE n")
            except Exception as exc:  # unknown label -> ignore
                print(f"  (skip clear {label}: {exc})")


# ---------------------------------------------------------------------------
# Company
# ---------------------------------------------------------------------------

def create_company(driver, company: str, ticker: str) -> None:
    with driver.session() as session:
        session.run(
            "MERGE (c:Company {ticker: $ticker}) SET c.name = $company",
            ticker=ticker,
            company=company,
        )


# ---------------------------------------------------------------------------
# Financial facts
# ---------------------------------------------------------------------------

def create_financial_graph(driver, facts_data: Dict[str, Any]) -> int:
    facts = facts_data["facts"]
    company = facts_data["company"]
    ticker = facts_data["ticker"]

    create_company(driver, company, ticker)

    count = 0
    with driver.session() as session:
        for fact in facts:
            metric = fact["metric"]
            fact_id = fact["fact_id"]
            period = fact.get("period") or {}
            source = fact.get("source") or {}
            document_id = source.get("document_id")
            page = source.get("page")
            section = source.get("section", "")
            page_key = f"{document_id}:{page}" if (document_id and page is not None) else None

            session.run(
                """
                MERGE (p:ReportingPeriod {label: $label})
                SET p.start = $start, p.end = $end

                MERGE (d:SourceDocument {document_id: $doc})
                SET d.company = $company, d.ticker = $ticker

                MERGE (m:FinancialMetric {name: $metric})
                SET m.category = $category

                MERGE (v:FinancialValue {fact_id: $fact_id})
                SET
                    v.metric = $metric,
                    v.value = $value,
                    v.unit = $unit,
                    v.company = $company,
                    v.ticker = $ticker,
                    v.statement_type = $statement_type,
                    v.confidence = $confidence,
                    v.extraction_method = $extraction_method

                WITH p, d, m, v

                MATCH (c:Company {ticker: $ticker})

                MERGE (c)-[:REPORTED]->(m)
                MERGE (c)-[:HAS_VALUE]->(v)
                MERGE (v)-[:FOR_METRIC]->(m)
                MERGE (v)-[:FOR_PERIOD]->(p)
                MERGE (v)-[:SUPPORTED_BY]->(d)
                """,
                label=period.get("label"),
                start=period.get("start"),
                end=period.get("end"),
                doc=document_id,
                company=company,
                ticker=ticker,
                metric=metric,
                category="profit_and_loss",
                fact_id=fact_id,
                value=fact.get("value"),
                unit=fact.get("unit"),
                statement_type=fact.get("statement_type", "consolidated"),
                confidence=fact.get("confidence", 0.0),
                extraction_method=fact.get("extraction_method"),
            )

            if page_key:
                session.run(
                    """
                    MERGE (pg:SourcePage {page_key: $page_key})
                    SET
                        pg.document_id = $document_id,
                        pg.page = $page,
                        pg.section = $section

                    WITH pg

                    MATCH (d:SourceDocument {document_id: $document_id})
                    MERGE (d)-[:HAS_PAGE]->(pg)

                    WITH pg

                    MATCH (v:FinancialValue {fact_id: $fact_id})
                    MERGE (v)-[:FOUND_ON]->(pg)
                    """,
                    page_key=page_key,
                    document_id=document_id,
                    page=page,
                    section=section or "",
                    fact_id=fact_id,
                )
            count += 1

    return count


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

def normalize_entity_type(entity_type: str) -> str:
    return ENTITY_TYPE_LABELS.get(
        entity_type.lower(), "Entity"
    )


def create_entities(driver, entities_data: Dict[str, Any]) -> int:
    entities = entities_data["entities"]
    ticker = entities_data["ticker"]
    company = entities_data["company"]

    count = 0
    with driver.session() as session:
        for ent in entities:
            eid = ent["entity_id"]
            name = ent["name"]
            etype = ent["entity_type"]
            label = normalize_entity_type(etype)
            fiscal_years = ent.get("fiscal_years", [])
            pages_by_year = ent.get("pages_by_year", {})

            session.run(
                f"""
                MERGE (e:Entity:{label} {{entity_id: $eid}})
                SET
                    e.name = $name,
                    e.entity_type = $etype,
                    e.fiscal_years = $fiscal_years,
                    e.pages_by_year = $pages

                WITH e

                MATCH (c:Company {{ticker: $ticker}})
                MERGE (c)-[:HAS_ENTITY]->(e)
                """,
                eid=eid,
                name=name,
                etype=etype,
                fiscal_years=fiscal_years,
                pages=json.dumps(pages_by_year, sort_keys=True),
                ticker=ticker,
                company=company,
            )
            count += 1
    return count


# ---------------------------------------------------------------------------
# Semantic relationships (controlled vocabulary, evidence-bearing)
# ---------------------------------------------------------------------------

def _safe_relationship_type(relation: str) -> Optional[str]:
    rel = (relation or "").strip().upper()
    if rel in CONTROLLED_RELATIONS:
        return rel
    return None


def create_relationships(driver, rel_data: Dict[str, Any]) -> int:
    relationships = rel_data["relationships"]

    # Aggregate relationships by (source, relation, target) so that the same
    # edge appearing in multiple fiscal years is stored ONCE but retains all
    # its per-year evidence (preserving multi-year provenance).
    aggregated: Dict[Any, Dict[str, Any]] = {}
    for rel in relationships:
        rel_type = _safe_relationship_type(rel["relation"])
        if not rel_type:
            print(f"  (skip unknown relation type: {rel.get('relation')})")
            continue

        src_id = rel["source"]["entity_id"]
        src_name = rel["source"]["name"]
        target_metric = (rel["target"] or {}).get("target_metric")
        target_eid = (rel["target"] or {}).get("entity_id")
        target_name = target_eid or target_metric

        key = (src_id, target_eid or ("metric:" + target_metric), rel_type)
        entry = aggregated.setdefault(key, {
            "rel_type": rel_type,
            "src_id": src_id,
            "src_name": src_name,
            "target_eid": target_eid,
            "target_metric": target_metric,
            "fiscal_years": [],
            "pages": [],
            "evidence_by_year": {},
            "relationship_types": [],
        })
        fy = rel.get("fiscal_year")
        if fy and fy not in entry["fiscal_years"]:
            entry["fiscal_years"].append(fy)
        entry["pages"].append(rel.get("page"))
        entry["relationship_types"].append(rel.get("relationship_type"))
        entry["evidence_by_year"][fy or "unknown"] = rel.get("evidence")

    count = 0
    with driver.session() as session:
        for entry in aggregated.values():
            rel_type = entry["rel_type"]
            props = {
                "fiscal_years": sorted(entry["fiscal_years"]),
                "pages": entry["pages"],
                "relationship_types": entry["relationship_types"],
                "evidence_by_year": json.dumps(entry["evidence_by_year"], sort_keys=True),
            }
            if entry["target_metric"]:
                session.run(
                    f"""
                    MERGE (a:Entity {{entity_id: $src_id}})
                    MERGE (m:FinancialMetric {{name: $metric}})
                    MERGE (a)-[r:{rel_type}]->(m)
                    SET r = $props
                    """,
                    src_id=entry["src_id"],
                    metric=entry["target_metric"],
                    props=props,
                )
            else:
                session.run(
                    f"""
                    MERGE (a:Entity {{entity_id: $src_id}})
                    MERGE (b:Entity {{entity_id: $tgt_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r = $props
                    """,
                    src_id=entry["src_id"],
                    tgt_id=entry["target_eid"],
                    props=props,
                )
            count += 1
    return count


# ---------------------------------------------------------------------------
# Narrative chunks (optional graph nodes for enrich/hybrid retrieval)
# ---------------------------------------------------------------------------

def create_chunks(driver) -> int:
    count = 0
    with driver.session() as session:
        for chunk_file in sorted(CHUNKS_DIR.glob("*_chunks.json")):
            data = _load_json(chunk_file)
            document_id = data.get("document_id")
            chunks = data.get("chunks", [])
            for chunk in chunks:
                session.run(
                    """
                    MERGE (ch:Chunk {chunk_id: $chunk_id})
                    SET
                        ch.text = $text,
                        ch.page = $page,
                        ch.section = $section,
                        ch.fiscal_year = $fiscal_year,
                        ch.chunk_type = $chunk_type,
                        ch.document_id = $document_id

                    WITH ch

                    MATCH (d:SourceDocument {document_id: $document_id})
                    MERGE (ch)-[:FROM_DOCUMENT]->(d)
                    """,
                    chunk_id=chunk["chunk_id"],
                    text=chunk.get("text", ""),
                    page=chunk.get("page"),
                    section=chunk.get("section"),
                    fiscal_year=chunk.get("fiscal_year") or chunk.get("reporting_period"),
                    chunk_type=chunk.get("chunk_type"),
                    document_id=document_id or chunk.get("document_id"),
                )
                count += 1
    return count


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_graph(driver) -> None:
    queries = {
        "companies": "MATCH (c:Company) RETURN count(c) AS c",
        "entities": "MATCH (e:Entity) RETURN count(e) AS c",
        "financial_metrics": "MATCH (m:FinancialMetric) RETURN count(m) AS c",
        "financial_values": "MATCH (v:FinancialValue) RETURN count(v) AS c",
        "periods": "MATCH (p:ReportingPeriod) RETURN count(p) AS c",
        "documents": "MATCH (d:SourceDocument) RETURN count(d) AS c",
        "pages": "MATCH (pg:SourcePage) RETURN count(pg) AS c",
        "chunks": "MATCH (ch:Chunk) RETURN count(ch) AS c",
        "semantic_edges": (
            "MATCH ()-[r]->(n) "
            "WHERE type(r) IN $rels RETURN count(r) AS c"
        ),
        "total_edges": "MATCH ()-[r]->() RETURN count(r) AS c",
    }
    rels = sorted(CONTROLLED_RELATIONS)
    print()
    print("=" * 80)
    print("GRAPH VALIDATION")
    print("=" * 80)
    with driver.session() as session:
        for name, query in queries.items():
            params = {"rels": rels} if "rels" in query else {}
            result = session.run(query, **params).single()
            print(f"{name:<22} {result['c']}")


def display_sample_paths(driver) -> None:
    print()
    print("=" * 80)
    print("SAMPLE SEMANTIC PATHS")
    print("=" * 80)
    query = """
    MATCH (c:Company {ticker: 'LTIM'})-[:HAS_ENTITY]->(e:Entity)-[r]->(target)
    RETURN
        e.name AS source,
        type(r) AS rel,
        coalesce(target.name, target.metric) AS tgt,
        r.fiscal_years AS fy,
        r.relationship_types AS rtype
    ORDER BY source
    LIMIT 40
    """
    with driver.session() as session:
        for record in session.run(query):
            print(
                f"{record['source']:<20} --{record['rel']}--> "
                f"{record['tgt']:<22} {record['fy']} [{record['rtype']}]"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_graph() -> None:
    facts_data = _load_json(FACTS_FILE)
    entities_data = _load_json(ENTITIES_FILE)
    rel_data = _load_json(RELATIONSHIPS_FILE)

    print("=" * 80)
    print("MULTI-YEAR KNOWLEDGE GRAPH LOADER")
    print("=" * 80)
    print(f"Fiscal years: {facts_data.get('fiscal_years')}")
    print(f"Facts: {len(facts_data.get('facts', []))}")
    print(f"Entities: {len(entities_data.get('entities', []))}")
    print(f"Relationships: {len(rel_data.get('relationships', []))}")

    driver = create_driver()

    try:
        create_constraints(driver)
        clear_ltim_nodes(driver)

        n_facts = create_financial_graph(driver, facts_data)
        n_entities = create_entities(driver, entities_data)
        n_rels = create_relationships(driver, rel_data)
        n_chunks = create_chunks(driver)

        print()
        print(f"Loaded {n_facts} financial facts")
        print(f"Loaded {n_entities} entities")
        print(f"Loaded {n_rels} semantic relationships")
        print(f"Loaded {n_chunks} narrative chunks")

        validate_graph(driver)
        display_sample_paths(driver)

        print()
        print("=" * 80)
        print("MULTI-YEAR GRAPH LOADING COMPLETE")
        print("=" * 80)

    finally:
        driver.close()
