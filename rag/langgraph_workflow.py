"""
LangGraph Workflow Engine for Financial GraphRAG.

Implements a compiled StateGraph supporting four distinct retrieval & reasoning modes:
1. LLM without RAG (No RAG)
2. Traditional Vector RAG (PostgreSQL pgvector)
3. GraphRAG (Neo4j Cypher & multi-hop traversal)
4. Hybrid Vector + GraphRAG (fused graph facts + vector context + deterministic math)

Uses LangGraph StateGraph, deterministic financial math, and pluggable LLM backends.
"""

import logging
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from database.financial_knowledge_graph import FinancialKnowledgeGraph
from database.postgres_client import PostgresVectorClient
from rag.embeddings_provider import EmbeddingsProvider, get_embeddings_provider
from rag.financial_math import FinancialMathEngine
from rag.fiscal_year import detect_fiscal_years
from rag.hybrid_router import KNOWN_METRICS, KNOWN_ENTITIES
from rag.llm_provider import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_GROUNDED = """You are a senior financial analyst and corporate intelligence expert specialized in annual report analysis.

Your goal is to answer the user's question with absolute factual rigor using ONLY the provided evidence.

CRITICAL FINANCIAL GUIDELINES:
1. NEVER hallucinate or invent numbers. If a financial metric is not explicitly present in the REPORTED FINANCIAL FACTS or DOCUMENT EVIDENCE, clearly state that it is not disclosed in the retrieved filings.
2. DISTINGUISH THREE TIERS:
   - [REPORTED FACTS]: Exact numbers stated in the company filings. Always cite the fiscal year and source page number.
   - [CALCULATED METRICS]: Mathematical figures computed programmatically (e.g., YoY growth, margins, CAGR). Present them as calculations, not raw reported values.
   - [STRATEGIC INTERPRETATION]: Qualitative commentary on drivers, risks, initiatives, and business context drawn from MD&A and report passages.
3. For multi-year questions, compare the metrics across the relevant fiscal years and mention YoY growth or margin expansion/compression.
4. When citing evidence, include the Fiscal Year, Page Number, and Section Name.
5. If the evidence is insufficient to answer the question completely, clearly indicate what is known and what data is missing.
6. Refuse to give speculative investment advice. Focus purely on factual filing analysis.
"""

SYSTEM_PROMPT_NO_RAG = """You are an AI assistant answering questions about companies and financial reports based ONLY on your pre-trained internal knowledge. You do not have access to real-time filings, live documents, or retrieval databases."""


class FinancialGraphState(TypedDict, total=False):
    question: str
    mode: str  # "no_rag" | "vector_rag" | "graph_rag" | "hybrid"
    detected_metric: Optional[str]
    fiscal_years: List[str]
    entities: List[str]
    vector_chunks: List[Dict[str, Any]]
    graph_facts: List[Dict[str, Any]]
    graph_paths: List[Dict[str, Any]]
    calculated_metrics: Dict[str, Any]
    fused_context: str
    citations: List[Dict[str, Any]]
    answer: str
    latency_breakdown: Dict[str, float]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model_name: str
    provider_name: str
    evidence_summary: str
    quality_notes: str


class FinancialGraphRAGWorkflow:
    """Compiled LangGraph workflow for Financial GraphRAG."""

    def __init__(
        self,
        postgres_client: Optional[PostgresVectorClient] = None,
        neo4j_kg: Optional[FinancialKnowledgeGraph] = None,
        embeddings_provider: Optional[EmbeddingsProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.pg = postgres_client or PostgresVectorClient()
        self.kg = neo4j_kg or FinancialKnowledgeGraph()
        self.embedder = embeddings_provider or get_embeddings_provider()
        self.llm = llm_provider or get_llm_provider()
        self.app = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(FinancialGraphState)

        # Add Nodes
        workflow.add_node("router_node", self._router_node)
        workflow.add_node("no_rag_node", self._no_rag_node)
        workflow.add_node("vector_retrieval_node", self._vector_retrieval_node)
        workflow.add_node("graph_retrieval_node", self._graph_retrieval_node)
        workflow.add_node("hybrid_retrieval_node", self._hybrid_retrieval_node)
        workflow.add_node("financial_math_node", self._financial_math_node)
        workflow.add_node("context_fusion_node", self._context_fusion_node)
        workflow.add_node("generation_node", self._generation_node)

        # Entry Point
        workflow.set_entry_point("router_node")

        # Conditional branch from router
        workflow.add_conditional_edges(
            "router_node",
            self._route_condition,
            {
                "no_rag": "no_rag_node",
                "vector_rag": "vector_retrieval_node",
                "graph_rag": "graph_retrieval_node",
                "hybrid": "hybrid_retrieval_node",
            },
        )

        # No RAG terminates immediately after generation
        workflow.add_edge("no_rag_node", END)

        # Vector RAG goes directly to context fusion
        workflow.add_edge("vector_retrieval_node", "context_fusion_node")

        # Graph RAG goes to financial math then context fusion
        workflow.add_edge("graph_retrieval_node", "financial_math_node")

        # Hybrid goes to financial math then context fusion
        workflow.add_edge("hybrid_retrieval_node", "financial_math_node")

        workflow.add_edge("financial_math_node", "context_fusion_node")
        workflow.add_edge("context_fusion_node", "generation_node")
        workflow.add_edge("generation_node", END)

        return workflow.compile()

    def _route_condition(self, state: FinancialGraphState) -> str:
        mode = state.get("mode", "hybrid")
        if mode in ["no_rag", "vector_rag", "graph_rag", "hybrid"]:
            return mode
        return "hybrid"

    # --- Node Implementations ---

    def _router_node(self, state: FinancialGraphState) -> Dict[str, Any]:
        t0 = time.time()
        question = state["question"]
        q_low = question.lower().strip()

        # Detect fiscal years
        years = detect_fiscal_years(question)

        # Detect metric
        detected_metric = None
        for canonical, syns in KNOWN_METRICS.items():
            if any(s in q_low for s in syns):
                detected_metric = canonical
                break

        # Detect entities
        entities = [e for e in KNOWN_ENTITIES if e in q_low]

        latencies = state.get("latency_breakdown", {})
        latencies["router"] = round(time.time() - t0, 4)

        return {
            "detected_metric": detected_metric,
            "fiscal_years": years,
            "entities": entities,
            "latency_breakdown": latencies,
        }

    def _no_rag_node(self, state: FinancialGraphState) -> Dict[str, Any]:
        """Mode 1: LLM without RAG - directly asks LLM with zero retrieved context."""
        t0 = time.time()
        question = state["question"]
        prompt = f"""Answer the following question based only on your general knowledge. If you do not know the exact numbers from the annual reports, state what you know or explain that you lack access to the official filings.

QUESTION:
{question}"""

        resp = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT_NO_RAG, temperature=0.1)
        lat = round(time.time() - t0, 3)

        latencies = state.get("latency_breakdown", {})
        latencies["generation"] = lat
        latencies["total"] = round(sum(latencies.values()), 3)

        return {
            "answer": resp.content,
            "vector_chunks": [],
            "graph_facts": [],
            "graph_paths": [],
            "calculated_metrics": {},
            "fused_context": "No context retrieved (Zero-RAG baseline).",
            "citations": [],
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "cost_usd": resp.estimated_cost_usd,
            "model_name": resp.model,
            "provider_name": resp.provider,
            "latency_breakdown": latencies,
            "evidence_summary": "None (Model relies solely on parametric pre-training weights; prone to hallucination or lack of recent fiscal data).",
            "quality_notes": "Without RAG, the model lacks access to specific annual reports (FY23-FY26), leading to vague approximations, missing figures, or fabricated numbers.",
        }

    def _vector_retrieval_node(self, state: FinancialGraphState) -> Dict[str, Any]:
        """Mode 2: Traditional Vector RAG - retrieves chunks from pgvector."""
        t0 = time.time()
        question = state["question"]
        emb = self.embedder.embed_query(question)

        # Filter by year if specific year targeted
        target_fy = state.get("fiscal_years", [None])[0] if len(state.get("fiscal_years", [])) == 1 else None

        chunks = self.pg.search_similarity(
            query_embedding=emb,
            top_k=5,
            fiscal_year=target_fy,
        )

        citations = []
        for c in chunks:
            citations.append({
                "source_type": "Vector Chunk (pgvector)",
                "document_id": c.get("document_id"),
                "fiscal_year": c.get("fiscal_year"),
                "page": c.get("page"),
                "section": c.get("section"),
                "score": c.get("score"),
                "claim": c.get("text", "")[:140] + "...",
            })

        latencies = state.get("latency_breakdown", {})
        latencies["vector_retrieval"] = round(time.time() - t0, 3)

        return {
            "vector_chunks": chunks,
            "graph_facts": [],
            "graph_paths": [],
            "citations": citations,
            "latency_breakdown": latencies,
            "evidence_summary": f"Retrieved {len(chunks)} text chunks from PostgreSQL pgvector using 768-dim cosine similarity.",
            "quality_notes": "Vector RAG retrieves relevant document passages, but lacks global multi-year tabular aggregation and cannot reliably connect entity causal chains across pages.",
        }

    def _graph_retrieval_node(self, state: FinancialGraphState) -> Dict[str, Any]:
        """Mode 3: GraphRAG - retrieves structured facts and multi-hop paths from Neo4j."""
        t0 = time.time()
        metric = state.get("detected_metric")
        years = state.get("fiscal_years", [])
        entities = state.get("entities", [])

        facts = []
        if metric:
            facts = self.kg.get_financial_facts_for_metric(metric)
            if years and len(years) == 1:
                facts = [f for f in facts if f.get("period") in years]

        graph_paths = []
        for ent in entities:
            subg = self.kg.get_subgraph_for_visualization(entity_filter=ent, limit=8)
            if subg.get("edges"):
                graph_paths.extend(subg["edges"])

        if not graph_paths and metric:
            subg = self.kg.get_subgraph_for_visualization(entity_filter=metric, limit=8)
            if subg.get("edges"):
                graph_paths.extend(subg["edges"])

        citations = []
        for f in facts:
            citations.append({
                "source_type": "Neo4j Graph Fact",
                "document_id": f.get("document_id", "Annual Report"),
                "fiscal_year": f.get("period"),
                "page": f.get("page"),
                "section": f.get("section", "Financial Results"),
                "claim": f"{f.get('metric')}: {f.get('value')} {f.get('unit')}",
            })

        latencies = state.get("latency_breakdown", {})
        latencies["graph_retrieval"] = round(time.time() - t0, 3)

        return {
            "graph_facts": facts,
            "graph_paths": graph_paths,
            "vector_chunks": [],
            "citations": citations,
            "latency_breakdown": latencies,
            "evidence_summary": f"Retrieved {len(facts)} exact KPI facts across reporting periods and {len(graph_paths)} relationship edges from Neo4j.",
            "quality_notes": "GraphRAG guarantees exact, unhallucinated figures across all fiscal years and maps multi-hop causal relationships, but lacks qualitative MD&A narrative context.",
        }

    def _hybrid_retrieval_node(self, state: FinancialGraphState) -> Dict[str, Any]:
        """Mode 4: Hybrid Vector + GraphRAG - fuses both vector chunks and graph facts."""
        t_v0 = time.time()
        question = state["question"]
        emb = self.embedder.embed_query(question)
        target_fy = state.get("fiscal_years", [None])[0] if len(state.get("fiscal_years", [])) == 1 else None

        chunks = self.pg.search_similarity(
            query_embedding=emb,
            top_k=5,
            fiscal_year=target_fy,
        )
        lat_vec = round(time.time() - t_v0, 3)

        t_g0 = time.time()
        metric = state.get("detected_metric")
        years = state.get("fiscal_years", [])
        entities = state.get("entities", [])

        facts = []
        if metric:
            facts = self.kg.get_financial_facts_for_metric(metric)
            if years and len(years) == 1:
                facts = [f for f in facts if f.get("period") in years]

        graph_paths = []
        for ent in entities:
            subg = self.kg.get_subgraph_for_visualization(entity_filter=ent, limit=8)
            if subg.get("edges"):
                graph_paths.extend(subg["edges"])

        if not graph_paths and metric:
            subg = self.kg.get_subgraph_for_visualization(entity_filter=metric, limit=8)
            if subg.get("edges"):
                graph_paths.extend(subg["edges"])

        lat_graph = round(time.time() - t_g0, 3)

        citations = []
        for f in facts:
            citations.append({
                "source_type": "Neo4j Graph Fact",
                "document_id": f.get("document_id", "Annual Report"),
                "fiscal_year": f.get("period"),
                "page": f.get("page"),
                "section": f.get("section", "Financial Results"),
                "claim": f"{f.get('metric')}: {f.get('value')} {f.get('unit')}",
            })
        for c in chunks:
            citations.append({
                "source_type": "Vector Chunk (pgvector)",
                "document_id": c.get("document_id"),
                "fiscal_year": c.get("fiscal_year"),
                "page": c.get("page"),
                "section": c.get("section"),
                "score": c.get("score"),
                "claim": c.get("text", "")[:140] + "...",
            })

        latencies = state.get("latency_breakdown", {})
        latencies["vector_retrieval"] = lat_vec
        latencies["graph_retrieval"] = lat_graph

        return {
            "vector_chunks": chunks,
            "graph_facts": facts,
            "graph_paths": graph_paths,
            "citations": citations,
            "latency_breakdown": latencies,
            "evidence_summary": f"Fused {len(facts)} structured facts, {len(graph_paths)} graph relationships, and {len(chunks)} semantic passages.",
            "quality_notes": "Hybrid Vector + GraphRAG provides the gold standard: exact ground-truth values from Neo4j + programmatic math + rich qualitative MD&A commentary from pgvector.",
        }

    def _financial_math_node(self, state: FinancialGraphState) -> Dict[str, Any]:
        """Runs programmatic math on facts to calculate YoY growth, margins, CAGR."""
        t0 = time.time()
        facts = state.get("graph_facts", [])
        calculated = {}
        if facts:
            calculated = FinancialMathEngine.analyze_multiyear_metric(facts)

        latencies = state.get("latency_breakdown", {})
        latencies["financial_math"] = round(time.time() - t0, 4)
        return {"calculated_metrics": calculated, "latency_breakdown": latencies}

    def _context_fusion_node(self, state: FinancialGraphState) -> Dict[str, Any]:
        """Fuses all available evidence into an unambiguous structured prompt."""
        t0 = time.time()
        parts = [
            f"QUESTION: {state['question']}",
            f"EXECUTION MODE: {state.get('mode', 'hybrid')}",
        ]
        if state.get("fiscal_years"):
            parts.append(f"TARGET FISCAL YEARS: {', '.join(state['fiscal_years'])}")

        # 1. Structured facts
        facts = state.get("graph_facts", [])
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
            parts.append("No structured graph facts retrieved.")

        # 2. Calculated math
        calc = state.get("calculated_metrics", {})
        if calc and calc.get("yearly_data"):
            parts.append("\n=== PROGRAMMATICALLY CALCULATED METRICS (Python Math) ===")
            for entry in calc["yearly_data"]:
                parts.append(f"- {entry['period']}: {entry['value']:,.2f} {entry['unit']} (YoY: {entry['yoy_display']})")
            if calc.get("cagr") is not None:
                parts.append(f"- Multi-Year CAGR: {calc['cagr']:.2f}%")
            if calc.get("overall_change_pct") is not None:
                parts.append(f"- Total Multi-Year Growth: {calc['overall_change_pct']:+.2f}%")

        # 3. Graph paths
        paths = state.get("graph_paths", [])
        if paths:
            parts.append("\n=== KNOWLEDGE GRAPH RELATIONSHIPS ===")
            for p in paths[:8]:
                parts.append(f"- ({p.get('from')}) -[:{p.get('label')}]-> ({p.get('to')})")

        # 4. Document chunks
        chunks = state.get("vector_chunks", [])
        parts.append("\n=== DOCUMENT EVIDENCE (Annual Report Passages) ===")
        if chunks:
            for i, c in enumerate(chunks, 1):
                parts.append(
                    f"\n[Source Passage #{i}] Document: {c.get('document_id')} | Year: {c.get('fiscal_year')} | Page: {c.get('page')} | Section: {c.get('section')} | Similarity: {c.get('score', 0):.3f}\n"
                    f"{c.get('text', '').strip()}"
                )
        else:
            parts.append("No document text passages retrieved.")

        fused = "\n".join(parts)
        latencies = state.get("latency_breakdown", {})
        latencies["context_fusion"] = round(time.time() - t0, 4)
        return {"fused_context": fused, "latency_breakdown": latencies}

    def _generation_node(self, state: FinancialGraphState) -> Dict[str, Any]:
        """Calls LLM with fused context."""
        t0 = time.time()
        prompt = f"""Based on the following retrieved evidence from the company's filings, answer the question thoroughly, factually, and concisely.

USER QUESTION:
{state['question']}

{state['fused_context']}

STRUCTURE YOUR ANSWER AS FOLLOWS:
1. Direct Answer & Executive Summary
2. Ground-Truth Reported Numbers & Calculated Changes (with Fiscal Year and Page citations)
3. Strategic Context & Narrative Insights (from document passages and graph connections)
"""
        resp = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT_GROUNDED, temperature=0.0)
        lat = round(time.time() - t0, 3)

        latencies = state.get("latency_breakdown", {})
        latencies["generation"] = lat
        latencies["total"] = round(sum(latencies.values()), 3)

        return {
            "answer": resp.content,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "cost_usd": resp.estimated_cost_usd,
            "model_name": resp.model,
            "provider_name": resp.provider,
            "latency_breakdown": latencies,
        }

    # --- Public API ---

    def run(self, question: str, mode: str = "hybrid") -> Dict[str, Any]:
        """Execute single question with designated mode."""
        initial_state: FinancialGraphState = {
            "question": question,
            "mode": mode,
            "latency_breakdown": {},
        }
        return self.app.invoke(initial_state)

    def run_all_modes(self, question: str) -> Dict[str, Dict[str, Any]]:
        """
        Execute all 4 modes on the same question for direct side-by-side comparison:
        - no_rag
        - vector_rag
        - graph_rag
        - hybrid
        """
        results = {}
        for mode in ["no_rag", "vector_rag", "graph_rag", "hybrid"]:
            results[mode] = self.run(question=question, mode=mode)
        return results
