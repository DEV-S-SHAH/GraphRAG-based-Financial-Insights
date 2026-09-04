import os
import re
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


# ============================================================
# EXCLUDED RELATIONSHIPS
# These are structural edges that do not represent
# meaningful semantic reasoning paths.
# ============================================================

EXCLUDED_RELATIONSHIPS = {
    "HAS_FACT",
    "HAS_VALUE",
    "RELATED_FINANCIAL_FACT",
    "HAS_ENTITY",
    "REPORTED",
    "FOR_METRIC",
    "FOR_PERIOD",
    "FOUND_ON",
    "SUPPORTED_BY",
}


# ============================================================
# CANONICAL METRIC RESOLUTION
# detect_metric() returns alias keys; these may differ from the
# canonical FinancialMetric node names used in Neo4j. This map
# resolves alias keys to their canonical Neo4j metric names so
# financial-fact lookups succeed.
# ============================================================

CANONICAL_METRIC_MAP = {
    "roe": "return_on_equity",
    "order_inflow": "order_inflow_usd_bn",
}


# ============================================================
# NEO4J CONNECTION
# ============================================================

def create_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")

    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        auth = os.getenv("NEO4J_AUTH")

        if auth and "/" in auth:
            auth_user, auth_password = auth.split("/", 1)

            user = auth_user
            password = auth_password

    if not password:
        raise ValueError(
            "NEO4J_PASSWORD is missing from .env "
            "and NEO4J_AUTH could not be parsed."
        )

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password)
    )

    driver.verify_connectivity()

    print("Neo4j connection successful.")

    return driver


# ============================================================
# GRAPH QUERY ENGINE
# ============================================================

class GraphQueryEngine:

    def __init__(self):
        self.driver = create_driver()

    def close(self):
        self.driver.close()

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(
            r"[^a-z0-9 ]+",
            " ",
            text.lower()
        ).strip()

    # ========================================================
    # METRIC DETECTION
    # ========================================================

    def detect_metric(self, question: str) -> Optional[str]:

        q = self.normalize(question)

        aliases = {

            "ebitda_margin": [
                "ebitda margin",
                "ebitda-margin",
                "profitability",
                "margin",
            ],

            "ebitda": [
                "ebitda"
            ],

            "revenue_growth_constant_currency": [
                "revenue growth constant currency",
                "constant currency growth",
            ],

            "revenue_growth_usd": [
                "revenue growth usd",
                "usd revenue growth",
            ],

            "revenue_growth_inr": [
                "revenue growth inr",
                "inr revenue growth",
            ],

            "profit_after_tax": [
                "profit after tax",
                "pat",
            ],

            "net_profit": [
                "net profit"
            ],

            "eps_diluted": [
                "diluted eps",
                "eps diluted",
            ],

            "eps_basic": [
                "basic eps"
            ],

            "revenue": [
                "revenue",
                "sales",
                "turnover",
            ],

            "market_capitalization": [
                "market capitalization",
                "market cap",
            ],

            "net_worth": [
                "net worth"
            ],

            "dividend_paid": [
                "dividend paid",
                "dividend",
            ],

            "employees": [
                "employees",
                "employee count",
            ],

            "return_on_equity": [
                "return on equity",
                "roe",
            ],

            "cash_and_investments": [
                "cash and investments"
            ],

            "current_ratio": [
                "current ratio"
            ],

            "ebit_margin": [
                "ebit margin"
            ],

            "pat_margin": [
                "pat margin",
                "net margin",
            ],

            "order_inflow": [
                "order inflow",
                "order book",
            ],

            "operating_cash_flow_conversion": [
                "operating cash flow conversion",
                "cash flow conversion",
            ],
        }

        # Longest phrase first for priority matching.
        matches = []

        for metric, phrases in aliases.items():

            for phrase in phrases:

                if phrase in q:

                    matches.append(
                        (len(phrase), metric)
                    )

        if not matches:
            return None

        matches.sort(reverse=True)

        return matches[0][1]

    # ========================================================
    # FINANCIAL FACT LOOKUP
    # ========================================================

    def get_financial_metric(
        self,
        metric: str
    ) -> List[Dict[str, Any]]:

        query = """
        MATCH (v:FinancialValue)-[:FOR_METRIC]->(m:FinancialMetric)
        WHERE toLower(m.name) = toLower($metric)

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

        ORDER BY confidence DESC, v.value DESC
        LIMIT 5
        """

        with self.driver.session() as session:

            records = session.run(
                query,
                metric=metric
            ).data()

        return records

    # ========================================================
    # ENTITY DETECTION
    # ========================================================

    def detect_entities(
        self,
        question: str
    ) -> List[str]:

        q = self.normalize(question)

        query = """
        MATCH (e:Entity)
        RETURN e.name AS name
        ORDER BY size(e.name) DESC
        """

        with self.driver.session() as session:

            records = session.run(query).data()

        found = []

        seen = set()

        for record in records:

            name = record["name"]

            name_lower = name.lower()

            if name_lower in seen:

                continue

            # Case-insensitive containment check
            if name_lower in q:

                found.append(name)

                seen.add(name_lower)

        return found

    # ========================================================
    # ENTITY RELATIONSHIPS
    # ========================================================

    def get_entity(
        self,
        entity_name: str
    ) -> Dict[str, Any]:

        query = """
        MATCH (e:Entity)

        WHERE toLower(e.name) = toLower($name)

        OPTIONAL MATCH (e)-[r1]->(target)

        OPTIONAL MATCH (source)-[r2]->(e)

        RETURN
            e.name AS entity,

            collect(
                DISTINCT CASE
                    WHEN r1 IS NOT NULL
                    THEN {
                        relationship: type(r1),
                        target: coalesce(
                            target.name,
                            target.metric,
                            target.fact_id
                        )
                    }
                    ELSE NULL
                END
            ) AS outgoing,

            collect(
                DISTINCT CASE
                    WHEN r2 IS NOT NULL
                    THEN {
                        source: coalesce(
                            source.name,
                            source.metric,
                            source.fact_id
                        ),
                        relationship: type(r2)
                    }
                    ELSE NULL
                END
            ) AS incoming
        """

        with self.driver.session() as session:

            record = session.run(
                query,
                name=entity_name
            ).single()

        if not record:
            return {}

        outgoing = [
            x for x in record["outgoing"]
            if x is not None
            and x.get("relationship") not in EXCLUDED_RELATIONSHIPS
        ]

        incoming = [
            x for x in record["incoming"]
            if x is not None
            and x.get("relationship") not in EXCLUDED_RELATIONSHIPS
        ]

        return {
            "entity": record["entity"],
            "outgoing": outgoing,
            "incoming": incoming
        }

    # ========================================================
    # GRAPH PATHS
    # ========================================================

    def get_paths(
        self,
        entity_name: str
    ) -> List[Dict[str, Any]]:

        query = """
        MATCH p=(e:Entity)-[*1..3]->(target)

        WHERE toLower(e.name) = toLower($name)

        RETURN
            [
                n IN nodes(p) |
                coalesce(
                    n.name,
                    n.metric,
                    n.fact_id
                )
            ] AS path,

            [
                r IN relationships(p) |
                type(r)
            ] AS relationships

        LIMIT 50
        """

        with self.driver.session() as session:

            records = session.run(
                query,
                name=entity_name
            ).data()

        return self._filter_and_deduplicate_paths(
            records,
            key_name="path"
        )

    # ========================================================
    # DIRECT SEMANTIC PATH
    # ========================================================

    def get_semantic_paths(
        self,
        entity_name: str,
        target_metric: str = None
    ) -> List[Dict[str, Any]]:

        if target_metric:

            query = """
            MATCH p=(e:Entity)-[*1..4]->(target:FinancialMetric)

            WHERE
                toLower(e.name) = toLower($entity)
                AND toLower(target.name) = toLower($metric)

            RETURN
                [
                    n IN nodes(p) |
                    coalesce(
                        n.name,
                        n.metric,
                        n.fact_id
                    )
                ] AS path,

                [
                    r IN relationships(p) |
                    type(r)
                ] AS relationships

            LIMIT 20
            """

            params = {
                "entity": entity_name,
                "metric": target_metric
            }

        else:

            query = """
            MATCH p=(e:Entity)-[*1..4]->(target)

            WHERE toLower(e.name) = toLower($entity)

            RETURN
                [
                    n IN nodes(p) |
                    coalesce(
                        n.name,
                        n.metric,
                        n.fact_id
                    )
                ] AS path,

                [
                    r IN relationships(p) |
                    type(r)
                ] AS relationships

            LIMIT 20
            """

            params = {
                "entity": entity_name
            }

        with self.driver.session() as session:

            records = session.run(
                query,
                **params
            ).data()

        return self._filter_and_deduplicate_paths(
            records,
            key_name="nodes"
        )

    # ========================================================
    # PATH FILTERING AND DEDUPLICATION
    # ========================================================

    @staticmethod
    def _filter_and_deduplicate_paths(
        paths: List[Dict[str, Any]],
        key_name: str = "path"
    ) -> List[Dict[str, Any]]:
        """
        Remove paths that:
        - contain excluded relationships (HAS_FACT, etc.)
        - contain None node names
        - are self-loops (single node)
        - are duplicates
        - are strict suffixes of longer paths
        """

        seen = set()
        result = []

        for path_data in paths:

            raw_path = path_data.get("path", [])
            relationships = path_data.get(
                "relationships", []
            )

            # Skip empty or single-node paths
            if len(raw_path) < 2:
                continue

            # Skip paths with None nodes
            if any(
                node is None or str(node).strip() == ""
                for node in raw_path
            ):
                continue

            # Skip paths with excluded relationships
            if any(
                rel in EXCLUDED_RELATIONSHIPS
                for rel in relationships
            ):
                continue

            # Skip self-loops (first and last node same)
            if len(raw_path) >= 2:
                if str(raw_path[0]).lower() == str(
                    raw_path[-1]
                ).lower():
                    # Allow if it has meaningful
                    # intermediate nodes
                    if len(raw_path) <= 2:
                        continue

            # Build deduplication key
            path_key = tuple(
                str(n).lower() for n in raw_path
            )
            rel_key = tuple(
                str(r) for r in relationships
            )
            key = (path_key, rel_key)

            if key in seen:
                continue

            seen.add(key)

            result.append(
                {
                    key_name: raw_path,
                    "relationships": relationships
                }
            )

        # Remove strict suffix paths: if path A is a
        # suffix of path B (same endpoint, shorter),
        # drop A.
        if len(result) > 1:

            filtered = []

            for i, candidate in enumerate(result):

                c_nodes = [
                    str(n).lower()
                    for n in candidate[key_name]
                ]

                is_suffix = False

                for j, other in enumerate(result):

                    if i == j:
                        continue

                    o_nodes = [
                        str(n).lower()
                        for n in other[key_name]
                    ]

                    # Candidate is a strict suffix if
                    # its node sequence equals the tail
                    # of a longer path
                    if (
                        len(c_nodes) < len(o_nodes)
                        and o_nodes[
                            len(o_nodes) - len(c_nodes):
                        ] == c_nodes
                    ):
                        is_suffix = True
                        break

                if not is_suffix:
                    filtered.append(candidate)

            result = filtered

        return result

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        question: str
    ) -> Dict[str, Any]:

        metric = self.detect_metric(question)

        # Resolve alias keys to canonical Neo4j metric names so
        # financial-fact lookups and semantic path targets match.
        canonical_metric = CANONICAL_METRIC_MAP.get(
            metric, metric
        )

        entities = self.detect_entities(question)

        # ----------------------------------------------------
        # FINANCIAL FACTS
        # ----------------------------------------------------

        financial_facts = []

        if canonical_metric:

            financial_facts = self.get_financial_metric(
                canonical_metric
            )

        # ----------------------------------------------------
        # ENTITY CONTEXT
        # ----------------------------------------------------

        entity_results = []

        all_semantic_paths = []
        all_paths = []

        for entity_name in entities:

            entity = self.get_entity(
                entity_name
            )

            paths = self.get_paths(
                entity_name
            )

            semantic_paths = self.get_semantic_paths(
                entity_name,
                canonical_metric
            )

            # Reverse-path discovery: when the detected entity is an
            # intermediate node on the way to the target metric, also
            # surface chains from the entities that DRIVE into it. This
            # lets questions that ask "what connects to X and reaches Y"
            # (without naming the upstream driver, e.g. Fit4Future)
            # recover the full multi-hop chain.
            if canonical_metric:
                for rel in entity.get("incoming", []):
                    source = rel.get("source")
                    if not isinstance(source, str):
                        continue
                    upstream_metric_paths = (
                        self.get_semantic_paths(
                            source,
                            canonical_metric
                        )
                    )
                    semantic_paths.extend(
                        upstream_metric_paths
                    )

            entity_results.append(
                {
                    "entity": entity,
                    "paths": paths,
                    "semantic_paths": semantic_paths
                }
            )

            all_paths.extend(paths)
            all_semantic_paths.extend(semantic_paths)

        # Deduplicate paths across entities
        all_paths = self._deduplicate_path_list(
            all_paths,
            key_name="path"
        )

        all_semantic_paths = (
            self._deduplicate_path_list(
                all_semantic_paths,
                key_name="nodes"
            )
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {
            "question": question,
            "detected_metric": canonical_metric,
            "financial_facts": financial_facts,
            "entities": entity_results,
            "paths": all_paths,
            "semantic_paths": all_semantic_paths,
        }

    @staticmethod
    def _deduplicate_path_list(
        paths: List[Dict[str, Any]],
        key_name: str = "path"
    ) -> List[Dict[str, Any]]:
        """Deduplicate a flat list of paths."""

        seen = set()
        result = []

        for path_data in paths:

            path_key = tuple(
                str(n).lower()
                for n in path_data.get(key_name, [])
            )
            rel_key = tuple(
                str(r)
                for r in path_data.get(
                    "relationships", []
                )
            )
            key = (path_key, rel_key)

            if key in seen:
                continue

            seen.add(key)
            result.append(path_data)

        return result


# ============================================================
# OUTPUT
# ============================================================

def print_result(result):

    print()
    print("=" * 80)
    print("RETRIEVED GRAPH CONTEXT")
    print("=" * 80)

    print()
    print("Question:")
    print(result["question"])

    print()
    print("Detected metric:")
    print(result["detected_metric"])

    # ========================================================
    # FINANCIAL FACTS
    # ========================================================

    print()
    print("-" * 80)
    print("FINANCIAL FACTS")
    print("-" * 80)

    facts = result["financial_facts"]

    if not facts:

        print("No financial facts found.")

    else:

        for fact in facts:

            print(
                f"{fact['metric']:<40} "
                f"{fact['value']} "
                f"{fact['unit']}"
            )

            print(
                f"  period={fact.get('period')} "
                f"page={fact.get('page')} "
                f"section={fact.get('section')} "
                f"confidence={fact.get('confidence')}"
            )

    # ========================================================
    # ENTITIES
    # ========================================================

    print()
    print("-" * 80)
    print("ENTITIES")
    print("-" * 80)

    if not result["entities"]:

        print("No entities found.")

    else:

        for entity_result in result["entities"]:

            entity = entity_result["entity"]

            if not entity:
                continue

            print()
            print(
                f"Entity: {entity.get('entity')}"
            )

            print()
            print("Outgoing:")

            for relation in entity.get(
                "outgoing",
                []
            ):

                print(
                    f"  --{relation['relationship']}--> "
                    f"{relation['target']}"
                )

            print()
            print("Incoming:")

            for relation in entity.get(
                "incoming",
                []
            ):

                print(
                    f"  {relation['source']} "
                    f"--{relation['relationship']}-->"
                )

            # =================================================
            # PATHS
            # =================================================

            paths = entity_result.get(
                "paths",
                []
            )

            if paths:

                print()
                print("Graph paths:")

                for path_data in paths:

                    path = path_data["path"]

                    relationships = path_data[
                        "relationships"
                    ]

                    for i, node in enumerate(path):

                        print(
                            str(node),
                            end=""
                        )

                        if i < len(relationships):

                            print(
                                f" --{relationships[i]}--> ",
                                end=""
                            )

                    print()

            # =================================================
            # SEMANTIC PATHS
            # =================================================

            semantic_paths = entity_result.get(
                "semantic_paths",
                []
            )

            if semantic_paths:

                print()
                print(
                    "Semantic paths to requested metric:"
                )

                for path_data in semantic_paths:

                    nodes_list = path_data.get(
                        "nodes",
                        path_data.get("path", [])
                    )

                    relationships = path_data[
                        "relationships"
                    ]

                    for i, node in enumerate(nodes_list):

                        print(
                            str(node),
                            end=""
                        )

                        if i < len(relationships):

                            print(
                                f" --{relationships[i]}--> ",
                                end=""
                            )

                    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("LTM GRAPH RAG QUERY ENGINE")
    print("=" * 80)

    engine = GraphQueryEngine()

    try:

        while True:

            question = input(
                "\nQuestion (or 'exit'): "
            ).strip()

            if question.lower() in {
                "exit",
                "quit",
                "q"
            }:

                break

            if not question:

                continue

            try:

                result = engine.search(
                    question
                )

                print_result(
                    result
                )

            except Exception as exc:

                print()
                print("=" * 80)
                print("QUERY ERROR")
                print("=" * 80)
                print(str(exc))

    finally:

        engine.close()

        print()
        print("=" * 80)
        print("GRAPH RAG QUERY ENGINE CLOSED")
        print("=" * 80)


if __name__ == "__main__":
    main()
