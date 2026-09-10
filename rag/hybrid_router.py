"""
Hybrid GraphRAG Router & Retrieval Engine.

Routes queries intelligently between:
1. PostgreSQL pgvector search
2. Neo4j graph traversal (1..3 hops)
3. Cypher structured retrieval (for KPIs, YoY trends, multi-year comparisons)
4. Metadata-filtered retrieval (by fiscal year, section, company)
5. Hybrid fusion (combining vector context + graph facts + programmatic math)
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from database.postgres_client import PostgresVectorClient
from database.financial_knowledge_graph import FinancialKnowledgeGraph
from rag.embeddings_provider import EmbeddingsProvider, get_embeddings_provider
from rag.financial_math import FinancialMathEngine
from rag.fiscal_year import detect_fiscal_years

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    VECTOR_SEARCH = "vector_search"
    NEO4J_GRAPH = "neo4j_graph"
    CYPHER_STRUCTURED = "cypher_structured"
    METADATA_FILTERED = "metadata_filtered"
    HYBRID_FUSION = "hybrid_fusion"


@dataclass
class RoutingDecision:
    strategy: RetrievalStrategy
    detected_metric: Optional[str] = None
    fiscal_years: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    reasoning: str = ""
    cypher_query: Optional[str] = None


@dataclass
class RetrievalContext:
    question: str
    routing: RoutingDecision
    reported_facts: List[Dict[str, Any]]
    calculated_metrics: Dict[str, Any]
    graph_paths: List[Dict[str, Any]]
    vector_chunks: List[Dict[str, Any]]
    fused_context_text: str
    citations: List[Dict[str, Any]]
    latency_breakdown: Dict[str, float]


KNOWN_METRICS = {
    "revenue": ["revenue", "sales", "turnover", "income from operations"],
    "ebitda": ["ebitda", "operating profit", "ebitda margin"],
    "profit_after_tax": ["profit after tax", "pat", "net profit", "net income"],
    "profit_before_tax": ["profit before tax", "pbt"],
    "earnings_per_share": ["eps", "earnings per share", "diluted eps", "basic eps"],
    "dividend": ["dividend", "dividend per share"],
    "headcount": ["employee", "headcount", "workforce", "attrition"],
}

KNOWN_ENTITIES = [
    "fit4future", "canvas.ai", "ai", "cloud", "digital transformation",
    "banking", "bfsi", "hi-tech", "manufacturing", "esg",
    "cybersecurity", "currency volatility", "talent attrition",
]


class HybridRouter:
    """Intelligently routes questions based on financial intent, entities, and periods."""

    @staticmethod
    def route_query(question: str) -> RoutingDecision:
        q = question.lower().strip()
        detected_years = detect_fiscal_years(question)

        # 1. Detect metric
        detected_metric = None
        for canonical, syns in KNOWN_METRICS.items():
            if any(s in q for s in syns):
                detected_metric = canonical
                break

        # 2. Detect entities
        detected_entities = [e for e in KNOWN_ENTITIES if e in q]

        # 3. Detect relational language
        relational_keywords = [
            "relationship between", "how does", "how did", "trace", "path",
            "chain", "connected", "impact", "affect", "drive", "drives",
            "influence", "link", "leads to", "contribute",
        ]
        has_relational = any(kw in q for kw in relational_keywords)

        # 4. Detect narrative / prose language
        narrative_keywords = [
            "what does the report say", "management discussion", "md&a",
            "message from chairman", "ceo statement", "strategy", "initiatives",
            "overview of", "discuss", "outlook", "guidance", "priorities",
            "what risks", "risk factors", "esg",
        ]
        has_narrative = any(kw in q for kw in narrative_keywords)

        # Decision logic:
        # Case A: Questions combining narrative (e.g. risks, strategy) with metrics/relations -> HYBRID_FUSION
        has_risk_or_strategy = any(k in q for k in ["risk", "strategy", "priority", "initiative", "md&a"])
        if has_risk_or_strategy and (detected_metric or has_relational):
            return RoutingDecision(
                strategy=RetrievalStrategy.HYBRID_FUSION,
                detected_metric=detected_metric,
                fiscal_years=detected_years,
                entities=detected_entities,
                reasoning="Inquiry combines narrative factors (risks/strategy) with financial metrics/relations; routing to Hybrid Fusion.",
                cypher_query=f"MATCH (m:FinancialMetric)-[:HAS_VALUE]->(v) RETURN m, v" if detected_metric else None,
            )

        # Case B: Pure relational / entity paths
        if has_relational and (detected_entities or detected_metric):
            return RoutingDecision(
                strategy=RetrievalStrategy.NEO4J_GRAPH,
                detected_metric=detected_metric,
                fiscal_years=detected_years,
                entities=detected_entities,
                reasoning="Entity relationship or causal path traversal query targeting knowledge graph.",
                cypher_query="MATCH p=(s:Entity)-[*1..3]->(t) RETURN p LIMIT 25",
            )

        # Case B: Multi-year trend / KPI calculation (Revenue, EBITDA, Margins, YoY)
        is_calculation = any(k in q for k in ["trend", "growth", "yoy", "margin", "cagr", "compare", "performance"])
        if detected_metric and (is_calculation or len(detected_years) > 1 or not detected_years):
            return RoutingDecision(
                strategy=RetrievalStrategy.CYPHER_STRUCTURED,
                detected_metric=detected_metric,
                fiscal_years=detected_years,
                entities=detected_entities,
                reasoning="Quantitative KPI query across reporting periods; using Cypher structured facts + Python math engine.",
                cypher_query=f"MATCH (m:FinancialMetric {{name: '{detected_metric}'}})-[:HAS_VALUE]->(v)-[:FOR_PERIOD]->(p) RETURN m, v, p",
            )

        # Case C: Qualitative narrative / MD&A query
        if has_narrative and not detected_metric:
            sec_filter = "Risk" if "risk" in q else ("Strategy" if "strategy" in q else None)
            return RoutingDecision(
                strategy=RetrievalStrategy.METADATA_FILTERED if sec_filter else RetrievalStrategy.VECTOR_SEARCH,
                detected_metric=None,
                fiscal_years=detected_years,
                entities=detected_entities,
                reasoning=f"Narrative prose question; routing to PostgreSQL pgvector semantic search (filter={sec_filter}).",
            )

        # Case D: Hybrid fusion default for complex financial research
        return RoutingDecision(
            strategy=RetrievalStrategy.HYBRID_FUSION,
            detected_metric=detected_metric,
            fiscal_years=detected_years,
            entities=detected_entities,
            reasoning="Complex financial inquiry requiring fusion of Neo4j structured facts and pgvector document passages.",
            cypher_query=f"MATCH (m:FinancialMetric)-[:HAS_VALUE]->(v) RETURN m, v" if detected_metric else None,
        )


class HybridRetriever:
    """Executes multi-modal retrieval and fuses results with programmatic financial math."""

    def __init__(
        self,
        postgres_client: Optional[PostgresVectorClient] = None,
        neo4j_kg: Optional[FinancialKnowledgeGraph] = None,
        embeddings_provider: Optional[EmbeddingsProvider] = None,
    ):
        self.pg_client = postgres_client or PostgresVectorClient()
        self.kg = neo4j_kg or FinancialKnowledgeGraph()
        self.embedder = embeddings_provider or get_embeddings_provider()

    def retrieve(
        self,
        question: str,
        force_strategy: Optional[RetrievalStrategy] = None,
        top_k_vector: int = 5,
    ) -> RetrievalContext:
        """Perform end-to-end routing and retrieval."""
        latencies: Dict[str, float] = {}
        t_start = time.time()

        # 1. Routing
        routing = HybridRouter.route_query(question)
        if force_strategy:
            routing.strategy = force_strategy
            routing.reasoning = f"Strategy manually forced to {force_strategy}."
        latencies["routing"] = round(time.time() - t_start, 3)

        reported_facts: List[Dict[str, Any]] = []
        calculated_metrics: Dict[str, Any] = {}
        graph_paths: List[Dict[str, Any]] = []
        vector_chunks: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []

        # 2. Neo4j Graph Retrieval (when applicable)
        if routing.strategy in [
            RetrievalStrategy.CYPHER_STRUCTURED,
            RetrievalStrategy.NEO4J_GRAPH,
            RetrievalStrategy.HYBRID_FUSION,
        ]:
            t_graph = time.time()
            if routing.detected_metric:
                facts = self.kg.get_financial_facts_for_metric(routing.detected_metric)
                # Filter by years if specific year requested
                if routing.fiscal_years and len(routing.fiscal_years) == 1:
                    facts = [f for f in facts if f.get("period") in routing.fiscal_years]
                reported_facts = facts

                # Compute financial math
                if reported_facts:
                    calculated_metrics = FinancialMathEngine.analyze_multiyear_metric(reported_facts)

            # Query entity paths
            for ent in routing.entities:
                subg = self.kg.get_subgraph_for_visualization(entity_filter=ent, limit=10)
                if subg.get("edges"):
                    graph_paths.extend(subg["edges"])

            latencies["graph"] = round(time.time() - t_graph, 3)

        # 3. PostgreSQL Vector Retrieval (when applicable)
        if routing.strategy in [
            RetrievalStrategy.VECTOR_SEARCH,
            RetrievalStrategy.METADATA_FILTERED,
            RetrievalStrategy.HYBRID_FUSION,
        ]:
            t_vec = time.time()
            query_emb = self.embedder.embed_query(question)

            # Optional year filter if single year targeted
            target_fy = routing.fiscal_years[0] if len(routing.fiscal_years) == 1 else None
            chunks = self.pg_client.search_similarity(
                query_embedding=query_emb,
                top_k=top_k_vector,
                fiscal_year=target_fy,
            )
            vector_chunks = chunks
            latencies["vector"] = round(time.time() - t_vec, 3)

        # 4. Build Citations & Provenance
        # From graph facts
        for f in reported_facts:
            citations.append({
                "source_type": "Knowledge Graph Fact",
                "document_id": f.get("document_id", "LTIM Annual Report"),
                "fiscal_year": f.get("period", ""),
                "page": f.get("page", 1),
                "section": f.get("section", "Financial Results"),
                "claim": f"{f.get('metric')}: {f.get('value')} {f.get('unit')}",
            })

        # From vector chunks
        for c in vector_chunks:
            citations.append({
                "source_type": "Document Chunk",
                "chunk_id": c.get("chunk_id"),
                "document_id": c.get("document_id"),
                "fiscal_year": c.get("fiscal_year"),
                "page": c.get("page"),
                "section": c.get("section"),
                "score": c.get("score"),
                "claim": c.get("text", "")[:150] + "...",
            })

        # 5. Fuse Context Text
        fused_text = self._build_fused_context(
            question=question,
            routing=routing,
            facts=reported_facts,
            calculated=calculated_metrics,
            graph_paths=graph_paths,
            chunks=vector_chunks,
        )

        latencies["total"] = round(time.time() - t_start, 3)

        return RetrievalContext(
            question=question,
            routing=routing,
            reported_facts=reported_facts,
            calculated_metrics=calculated_metrics,
            graph_paths=graph_paths,
            vector_chunks=vector_chunks,
            fused_context_text=fused_text,
            citations=citations,
            latency_breakdown=latencies,
        )

    def _build_fused_context(
        self,
        question: str,
        routing: RoutingDecision,
        facts: List[Dict[str, Any]],
        calculated: Dict[str, Any],
        graph_paths: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> str:
        parts = [
            f"QUESTION: {question}",
            f"RETRIEVAL STRATEGY: {routing.strategy.value} (Reason: {routing.reasoning})",
        ]

        if routing.fiscal_years:
            parts.append(f"TARGET FISCAL YEARS: {', '.join(routing.fiscal_years)}")

        # 1. Reported facts
        parts.append("\n=== REPORTED FINANCIAL FACTS (Ground Truth from Filings) ===")
        if facts:
            for f in facts:
                val = f.get("value")
                unit = f.get("unit", "")
                period = f.get("period", "")
                page = f.get("page", "?")
                sec = f.get("section", "")
                parts.append(f"- Metric: {f.get('metric')} | Value: {val:,.2f} {unit} | Period: {period} | Page: {page} | Section: {sec}")
        else:
            parts.append("No direct structured facts found in Knowledge Graph.")

        # 2. Calculated Metrics (Python math)
        if calculated and calculated.get("yearly_data"):
            parts.append("\n=== PROGRAMMATICALLY CALCULATED METRICS (Python Math) ===")
            for entry in calculated["yearly_data"]:
                parts.append(f"- {entry['period']}: {entry['value']:,.2f} {entry['unit']} (YoY: {entry['yoy_display']})")
            if calculated.get("cagr") is not None:
                parts.append(f"- Compound Annual Growth Rate (CAGR): {calculated['cagr']:.2f}%")
            if calculated.get("overall_change_pct") is not None:
                parts.append(f"- Total Multi-Year Growth: {calculated['overall_change_pct']:+.2f}%")

        # 3. Graph Paths
        if graph_paths:
            parts.append("\n=== KNOWLEDGE GRAPH RELATIONSHIPS ===")
            for edge in graph_paths[:8]:
                parts.append(f"- ({edge.get('from')}) -[:{edge.get('label')}]-> ({edge.get('to')})")

        # 4. Document Chunks
        parts.append("\n=== DOCUMENT EVIDENCE (Annual Report Passages) ===")
        if chunks:
            for i, c in enumerate(chunks, 1):
                parts.append(
                    f"\n[Source Passage #{i}] Document: {c.get('document_id')} | Fiscal Year: {c.get('fiscal_year')} | Page: {c.get('page')} | Section: {c.get('section')} | Similarity: {c.get('score', 0):.3f}\n"
                    f"{c.get('text', '').strip()}"
                )
        else:
            parts.append("No document text chunks retrieved.")

        return "\n".join(parts)
