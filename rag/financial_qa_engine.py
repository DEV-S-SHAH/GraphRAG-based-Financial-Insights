"""
Financial QA Engine: Orchestrates Retrieval, Math, Evidence, and LLM Generation.

Strictly enforces:
1. Grounded generation (zero hallucinated numbers)
2. Clear provenance citations (document, fiscal year, page number, section)
3. Three-tier distinction: Reported Facts -> Calculated Metrics -> Qualitative Interpretation
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional
from rag.hybrid_router import HybridRetriever, RetrievalContext, RetrievalStrategy
from rag.llm_provider import LLMProvider, LLMResponse, get_llm_provider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior financial analyst and corporate intelligence expert specialized in annual report analysis.

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
6. Refuse to give speculative investment advice (e.g. "Should I buy/sell stock"). Focus purely on factual filing analysis.
"""


@dataclass
class FinancialQAResponse:
    question: str
    answer: str
    strategy_used: str
    reported_facts: List[Dict[str, Any]]
    calculated_metrics: Dict[str, Any]
    citations: List[Dict[str, Any]]
    retrieval_context: RetrievalContext
    llm_response: LLMResponse


class FinancialQAEngine:
    """End-to-end question answering engine for financial filings."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm_provider or get_llm_provider()

    def answer_question(
        self,
        question: str,
        strategy: Optional[RetrievalStrategy] = None,
        top_k: int = 5,
    ) -> FinancialQAResponse:
        """Run retrieval, calculation, prompt construction, and LLM generation."""
        # 1. Retrieve context
        ctx = self.retriever.retrieve(
            question=question,
            force_strategy=strategy,
            top_k_vector=top_k,
        )

        # 2. Build prompt
        user_prompt = f"""Based on the following retrieved evidence from the company's filings, answer the question thoroughly and concisely.

USER QUESTION:
{question}

{ctx.fused_context_text}

STRUCTURE YOUR ANSWER AS FOLLOWS:
1. Executive Summary / Direct Answer
2. Reported Financial Facts & Calculated Metrics (with Year and Page citations)
3. Strategic Insights & Qualitative Drivers (from document passages and graph connections)
"""

        # 3. Generate answer via LLM
        llm_resp = self.llm.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.0,
        )

        return FinancialQAResponse(
            question=question,
            answer=llm_resp.content,
            strategy_used=ctx.routing.strategy.value,
            reported_facts=ctx.reported_facts,
            calculated_metrics=ctx.calculated_metrics,
            citations=ctx.citations,
            retrieval_context=ctx,
            llm_response=llm_resp,
        )

    def compare_all_modes(self, question: str) -> Dict[str, Dict[str, Any]]:
        """
        Execute and compare all 4 paradigms on the exact same question:
        1. no_rag: LLM without RAG
        2. vector_rag: Traditional Vector RAG (pgvector)
        3. graph_rag: GraphRAG (Neo4j)
        4. hybrid: Hybrid Vector + GraphRAG
        """
        from rag.langgraph_workflow import FinancialGraphRAGWorkflow
        workflow = FinancialGraphRAGWorkflow(
            postgres_client=self.retriever.pg_client,
            neo4j_kg=self.retriever.kg,
            embeddings_provider=self.retriever.embedder,
            llm_provider=self.llm,
        )
        return workflow.run_all_modes(question)
