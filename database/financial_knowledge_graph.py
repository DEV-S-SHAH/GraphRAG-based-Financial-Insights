"""
Neo4j Financial Knowledge Graph Builder & Query Client.

Implements full financial graph ontology:
Nodes: Company, AnnualReport, Section, Chunk, FinancialMetric,
       Revenue, EBITDA, Profit, EPS, CashFlow, Debt, Asset, Liability,
       BusinessSegment, Product, Geography, Risk, Strategy, Executive,
       ReportingPeriod, SourcePage, FinancialValue.

Relationships:
- Company -> HAS_REPORT -> AnnualReport
- AnnualReport -> CONTAINS -> Section
- Section -> CONTAINS -> Chunk
- Company -> HAS_METRIC -> FinancialMetric
- FinancialMetric -> REPORTED_IN -> AnnualReport
- FinancialMetric -> HAS_VALUE -> FinancialValue
- FinancialValue -> FOR_PERIOD -> ReportingPeriod
- FinancialValue -> FOUND_ON -> SourcePage
- Company -> HAS_SEGMENT -> BusinessSegment
- Company -> FACES -> Risk
- Company -> OPERATES_IN -> Geography
- Company -> HAS_STRATEGY -> Strategy
- Company -> LED_BY -> Executive
- Entity -> DRIVES | IMPROVES | IMPACTS | SUPPORTS | MITIGATES -> Entity / FinancialMetric
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Metric-to-Label mapping for specialized entity labels
METRIC_LABEL_MAP = {
    "revenue": ["FinancialMetric", "Revenue"],
    "ebitda": ["FinancialMetric", "EBITDA"],
    "profit_after_tax": ["FinancialMetric", "Profit"],
    "profit_before_tax": ["FinancialMetric", "Profit"],
    "earnings_per_share": ["FinancialMetric", "EPS"],
    "basic_eps": ["FinancialMetric", "EPS"],
    "diluted_eps": ["FinancialMetric", "EPS"],
    "operating_cash_flow": ["FinancialMetric", "CashFlow"],
    "free_cash_flow": ["FinancialMetric", "CashFlow"],
    "net_debt": ["FinancialMetric", "Debt"],
    "total_debt": ["FinancialMetric", "Debt"],
    "total_assets": ["FinancialMetric", "Asset"],
    "net_worth": ["FinancialMetric", "Asset"],
    "current_liabilities": ["FinancialMetric", "Liability"],
}

# Segment, Geography, Risk, Strategy defaults for LTIMindtree
BUSINESS_SEGMENTS = [
    {"name": "Banking, Financial Services & Insurance (BFSI)", "category": "industry_vertical"},
    {"name": "Hi-Tech, Media & Entertainment", "category": "industry_vertical"},
    {"name": "Manufacturing & Resources", "category": "industry_vertical"},
    {"name": "Retail, CPG & Travel", "category": "industry_vertical"},
    {"name": "Health & Life Sciences", "category": "industry_vertical"},
]

GEOGRAPHIES = [
    {"name": "North America", "region": "Americas"},
    {"name": "Europe", "region": "EMEA"},
    {"name": "India", "region": "Domestic"},
    {"name": "Rest of the World", "region": "APAC & Other"},
]

RISKS = [
    {"name": "Macroeconomic & Global IT Spending Volatility", "category": "market_risk", "severity": "High"},
    {"name": "Currency & Exchange Rate Fluctuations", "category": "financial_risk", "severity": "Medium"},
    {"name": "Cybersecurity & Data Privacy Threats", "category": "operational_risk", "severity": "High"},
    {"name": "Talent Attrition & Wage Inflation", "category": "talent_risk", "severity": "Medium"},
    {"name": "Client Concentration & Pricing Pressure", "category": "commercial_risk", "severity": "Medium"},
    {"name": "AI Disruption & Technology Obsolescence", "category": "strategic_risk", "severity": "High"},
]

STRATEGIES = [
    {"name": "Fit4Future Transformation", "description": "Enterprise efficiency and margin expansion program"},
    {"name": "Canvas.ai Enterprise Platform", "description": "Generative AI ecosystem and client solutions"},
    {"name": "Digital Engineering & Cloud Migration", "description": "Modernization and multi-cloud transformation"},
    {"name": "Client Centricity & Strategic Accounts", "description": "Deepening relationships with top 100 enterprise clients"},
    {"name": "ESG Net Zero Roadmap", "description": "Sustainable operations and social responsibility"},
]

EXECUTIVES = [
    {"name": "Debashis Chatterjee", "role": "Chief Executive Officer & Managing Director"},
    {"name": "Vipul Chandra", "role": "Chief Financial Officer"},
    {"name": "A.M. Naik", "role": "Founder Chairman"},
    {"name": "S.N. Subrahmanyan", "role": "Chairman"},
]


class FinancialKnowledgeGraph:
    """Manages the Neo4j Financial Knowledge Graph."""

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")

        if not self.password:
            auth = os.getenv("NEO4J_AUTH", "")
            if "/" in auth:
                self.user, self.password = auth.split("/", 1)
            else:
                self.password = "password123"

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
        except Exception:
            alt_uri = "bolt://localhost:7687" if "127.0.0.1" in self.uri else "bolt://127.0.0.1:7687"
            self.driver = GraphDatabase.driver(alt_uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self.uri = alt_uri

    def close(self):
        self.driver.close()

    def init_constraints_and_indexes(self):
        """Create constraints and indexes for fast retrieval."""
        constraints = [
            "CREATE CONSTRAINT company_ticker_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:AnnualReport) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT metric_name_unique IF NOT EXISTS FOR (m:FinancialMetric) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (v:FinancialValue) REQUIRE v.fact_id IS UNIQUE",
            "CREATE CONSTRAINT period_label_unique IF NOT EXISTS FOR (p:ReportingPeriod) REQUIRE p.label IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT page_key_unique IF NOT EXISTS FOR (sp:SourcePage) REQUIRE sp.page_key IS UNIQUE",
            "CREATE CONSTRAINT segment_name_unique IF NOT EXISTS FOR (s:BusinessSegment) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT risk_name_unique IF NOT EXISTS FOR (r:Risk) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT strategy_name_unique IF NOT EXISTS FOR (st:Strategy) REQUIRE st.name IS UNIQUE",
            "CREATE CONSTRAINT geo_name_unique IF NOT EXISTS FOR (g:Geography) REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT exec_name_unique IF NOT EXISTS FOR (e:Executive) REQUIRE e.name IS UNIQUE",
        ]
        with self.driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    logger.info(f"Constraint notice: {e}")

    def build_full_graph(
        self,
        facts_path: Optional[Path] = None,
        entities_path: Optional[Path] = None,
        relationships_path: Optional[Path] = None,
    ):
        """Build or enrich the complete knowledge graph with all required entities and relations."""
        self.init_constraints_and_indexes()

        facts_file = facts_path or (PROJECT_ROOT / "data" / "processed" / "financial_facts" / "LTIM_canonical_multiyear.json")
        entities_file = entities_path or (PROJECT_ROOT / "data" / "processed" / "entities" / "LTIM_entities_multiyear.json")
        rel_file = relationships_path or (PROJECT_ROOT / "data" / "processed" / "relationships" / "LTIM_relationships_multiyear.json")

        company_name = "LTIMindtree Limited"
        ticker = "LTIM"

        with self.driver.session() as session:
            # 1. Create Company Node
            session.run(
                """
                MERGE (c:Company {ticker: $ticker})
                SET c.name = $name,
                    c.industry = "Information Technology & Consulting",
                    c.market = "NSE / BSE India"
                """,
                ticker=ticker,
                name=company_name,
            )

            # 2. Create Segments & Link: Company -> HAS_SEGMENT -> BusinessSegment
            for seg in BUSINESS_SEGMENTS:
                session.run(
                    """
                    MERGE (s:BusinessSegment {name: $name})
                    SET s.category = $cat
                    WITH s
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (c)-[:HAS_SEGMENT]->(s)
                    """,
                    name=seg["name"],
                    cat=seg["category"],
                    ticker=ticker,
                )

            # 3. Create Geographies & Link: Company -> OPERATES_IN -> Geography
            for geo in GEOGRAPHIES:
                session.run(
                    """
                    MERGE (g:Geography {name: $name})
                    SET g.region = $region
                    WITH g
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (c)-[:OPERATES_IN]->(g)
                    """,
                    name=geo["name"],
                    region=geo["region"],
                    ticker=ticker,
                )

            # 4. Create Risks & Link: Company -> FACES -> Risk
            for rk in RISKS:
                session.run(
                    """
                    MERGE (r:Risk {name: $name})
                    SET r.category = $cat, r.severity = $sev
                    WITH r
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (c)-[:FACES]->(r)
                    """,
                    name=rk["name"],
                    cat=rk["category"],
                    sev=rk["severity"],
                    ticker=ticker,
                )

            # 5. Create Strategies & Link: Company -> HAS_STRATEGY -> Strategy
            for st in STRATEGIES:
                session.run(
                    """
                    MERGE (s:Strategy {name: $name})
                    SET s.description = $desc
                    WITH s
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (c)-[:HAS_STRATEGY]->(s)
                    """,
                    name=st["name"],
                    desc=st["description"],
                    ticker=ticker,
                )

            # 6. Create Executives & Link: Company -> LED_BY -> Executive
            for ex in EXECUTIVES:
                session.run(
                    """
                    MERGE (e:Executive {name: $name})
                    SET e.role = $role
                    WITH e
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (c)-[:LED_BY]->(e)
                    """,
                    name=ex["name"],
                    role=ex["role"],
                    ticker=ticker,
                )

            # 7. Create AnnualReport nodes for each fiscal year
            years = [
                ("FY2022-23", "LTIM_FY2022-23_Annual_Report", "Annual Report 2022-23"),
                ("FY2023-24", "LTIM_FY2023-24_Annual_Report", "Integrated Annual Report 2023-24"),
                ("FY2024-25", "LTIM_FY2024-25_Annual_Report", "Integrated Annual Report 2024-25"),
                ("FY2025-26", "LTIM_FY2025-26_Annual_Report", "Integrated Annual Report 2025-26"),
            ]
            for fy, doc_id, title in years:
                session.run(
                    """
                    MERGE (d:AnnualReport {document_id: $doc_id})
                    SET d.fiscal_year = $fy,
                        d.title = $title,
                        d.company = $company,
                        d.ticker = $ticker
                    WITH d
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (c)-[:HAS_REPORT]->(d)
                    """,
                    doc_id=doc_id,
                    fy=fy,
                    title=title,
                    company=company_name,
                    ticker=ticker,
                )

            # 8. Load canonical financial facts
            if facts_file.exists():
                with open(facts_file, "r", encoding="utf-8") as f:
                    facts_data = json.load(f)

                for fact in facts_data.get("facts", []):
                    metric = fact["metric"]
                    fact_id = fact["fact_id"]
                    val = fact.get("value")
                    unit = fact.get("unit", "")
                    period = fact.get("period", {})
                    p_label = period.get("label", "FY2025-26")
                    src = fact.get("source", {})
                    doc_id = src.get("document_id") or f"LTIM_{p_label}_Annual_Report"
                    page = src.get("page", 1)
                    section_title = src.get("section", "Financial Results")
                    page_key = f"{doc_id}:{page}"

                    query = """
                    MERGE (p:ReportingPeriod {label: $period_label})
                    MERGE (d:AnnualReport {document_id: $doc_id})
                    MERGE (m:FinancialMetric {name: $metric})
                    MERGE (sp:SourcePage {page_key: $page_key})
                    SET sp.page = $page, sp.section = $section_title

                    MERGE (v:FinancialValue {fact_id: $fact_id})
                    SET v.value = $val,
                        v.unit = $unit,
                        v.metric = $metric,
                        v.statement_type = $stmt_type,
                        v.confidence = 1.0

                    WITH p, d, m, sp, v
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (c)-[:HAS_METRIC]->(m)
                    MERGE (m)-[:REPORTED_IN]->(d)
                    MERGE (m)-[:HAS_VALUE]->(v)
                    MERGE (v)-[:FOR_PERIOD]->(p)
                    MERGE (v)-[:FOUND_ON]->(sp)
                    MERGE (v)-[:SUPPORTED_BY]->(d)
                    MERGE (d)-[:CONTAINS]->(sp)
                    """
                    session.run(
                        query,
                        period_label=p_label,
                        doc_id=doc_id,
                        metric=metric,
                        page_key=page_key,
                        page=page,
                        section_title=section_title,
                        fact_id=fact_id,
                        val=float(val) if val is not None else 0.0,
                        unit=unit,
                        stmt_type=fact.get("statement_type", "consolidated"),
                        ticker=ticker,
                    )

            # 9. Load semantic relationships (DRIVES, IMPROVES, IMPACTS, SUPPORTS)
            if rel_file.exists():
                with open(rel_file, "r", encoding="utf-8") as f:
                    rel_data = json.load(f)

                for r in rel_data.get("relationships", []):
                    s_name = r.get("source_entity") or (r.get("source", {}).get("name") if isinstance(r.get("source"), dict) else None)
                    t_name = r.get("target_entity") or (r.get("target", {}).get("name") if isinstance(r.get("target"), dict) else None)
                    rel_type = (r.get("relation") or r.get("relationship_type", "RELATED_TO")).upper()
                    if not s_name or not t_name:
                        continue

                    valid_rels = ["DRIVES", "IMPROVES", "IMPACTS", "SUPPORTS", "MITIGATES", "CONTRIBUTES_TO", "ENABLES", "FUNDS", "FOLLOWS", "INCREASES"]
                    if rel_type not in valid_rels:
                        rel_type = "RELATED_TO"

                    cypher_rel = f"""
                    MERGE (s:Entity {{name: $s_name}})
                    MERGE (t:Entity {{name: $t_name}})
                    MERGE (s)-[rel:{rel_type}]->(t)
                    SET rel.evidence = $evidence, rel.fiscal_year = $fy
                    """
                    session.run(
                        cypher_rel,
                        s_name=s_name,
                        t_name=t_name,
                        evidence=r.get("evidence", ""),
                        fy=r.get("fiscal_year", "Multi-year"),
                    )

        logger.info("Knowledge Graph successfully enriched.")

    def get_graph_summary(self) -> Dict[str, Any]:
        """Return node counts by label and relationship counts by type."""
        summary = {"nodes": {}, "relationships": {}, "total_nodes": 0, "total_relationships": 0}
        with self.driver.session() as session:
            # Nodes
            res = session.run("MATCH (n) RETURN labels(n) AS lbls, count(*) AS cnt")
            for r in res:
                label_name = ":".join(r["lbls"])
                summary["nodes"][label_name] = r["cnt"]
                summary["total_nodes"] += r["cnt"]

            # Relationships
            res2 = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt")
            for r in res2:
                summary["relationships"][r["rel_type"]] = r["cnt"]
                summary["total_relationships"] += r["cnt"]

        return summary

    def get_financial_facts_for_metric(self, metric_name: str, exact: bool = False) -> List[Dict[str, Any]]:
        """Retrieve all values for a financial metric across periods with provenance."""
        if not metric_name:
            where_clause = ""
        elif exact or metric_name.lower() in [
            "revenue", "ebitda", "ebit", "profit_after_tax", "profit_before_tax",
            "employees", "net_worth", "pat_margin", "ebitda_margin",
        ]:
            where_clause = "WHERE toLower(m.name) = toLower($metric)"
        else:
            where_clause = "WHERE toLower(m.name) = toLower($metric) OR toLower(m.name) CONTAINS toLower($metric)"

        query = f"""
        MATCH (m:FinancialMetric)
        {where_clause}
        MATCH (m)-[:HAS_VALUE]->(v:FinancialValue)-[:FOR_PERIOD]->(p:ReportingPeriod)
        OPTIONAL MATCH (v)-[:FOUND_ON]->(sp:SourcePage)
        OPTIONAL MATCH (v)-[:SUPPORTED_BY]->(d:AnnualReport)
        RETURN
            m.name AS metric,
            v.value AS value,
            v.unit AS unit,
            v.statement_type AS statement_type,
            p.label AS period,
            sp.page AS page,
            sp.section AS section,
            d.document_id AS document_id
        ORDER BY p.label ASC
        """
        with self.driver.session() as session:
            return session.run(query, metric=metric_name).data()

    def get_subgraph_for_visualization(self, entity_filter: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """Fetch nodes and edges formatted for PyVis / interactive graph rendering."""
        where_clause = ""
        params = {"limit": limit}
        if entity_filter:
            where_clause = "WHERE toLower(n.name) CONTAINS toLower($ef) OR toLower(m.name) CONTAINS toLower($ef)"
            params["ef"] = entity_filter

        query = f"""
        MATCH (n)-[r]->(m)
        {where_clause}
        RETURN
            elementId(n) AS source_id,
            labels(n) AS source_labels,
            coalesce(n.name, n.title, n.label, n.ticker, n.document_id, 'Node') AS source_name,
            type(r) AS rel_type,
            elementId(m) AS target_id,
            labels(m) AS target_labels,
            coalesce(m.name, m.title, m.label, m.ticker, m.document_id, 'Node') AS target_name
        LIMIT $limit
        """
        nodes = {}
        edges = []
        with self.driver.session() as session:
            for rec in session.run(query, **params):
                s_id = str(rec["source_id"])
                t_id = str(rec["target_id"])
                s_label = rec["source_labels"][0] if rec["source_labels"] else "Entity"
                t_label = rec["target_labels"][0] if rec["target_labels"] else "Entity"

                if s_id not in nodes:
                    nodes[s_id] = {"id": s_id, "label": rec["source_name"], "group": s_label}
                if t_id not in nodes:
                    nodes[t_id] = {"id": t_id, "label": rec["target_name"], "group": t_label}

                edges.append({
                    "from": s_id,
                    "to": t_id,
                    "label": rec["rel_type"],
                })

        return {"nodes": list(nodes.values()), "edges": edges}
