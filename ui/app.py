"""
Financial GraphRAG Platform: Production Research & Intelligence System.

Core Stack:
Docling → Semantic Structure Chunker → PostgreSQL + pgvector → Neo4j → Hybrid Retrieval → LLM → Streamlit

Modules:
1. ⚖️ RAG Comparison: 4-way side-by-side (No RAG vs Vector RAG vs GraphRAG vs Hybrid)
2. 💬 Financial Q&A: Grounded answers with 3-tier labeling (Reported -> Calculated -> Interpreted)
3. 📈 Financial Trends: Multi-year trend analysis, YoY growth, margins, CAGR, segment breakdown
4. 🕸️ Knowledge Graph: Interactive PyVis Neo4j visualization, entity filters, multi-hop path tracing
5. 🔍 Vector Search: Direct pgvector semantic search, cosine similarity scores, metadata filters
6. 📑 Document & DB Explorer: Chunks, metadata, embeddings, tables, documents, psql guide, SQL console
7. 📥 Data Ingestion: Idempotent PDF upload, report URLs, SEC EDGAR API discovery and indexing
8. 📊 Benchmark Dashboard: Empirical evaluation across all 4 paradigms with Plotly visual charts
"""

import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.financial_knowledge_graph import FinancialKnowledgeGraph
from database.postgres_client import PostgresVectorClient
import database.load_semantic_ground_truth as ground_truth_loader
from ingestion.pipeline import FinancialIngestionPipeline
from ingestion.sources.local_pdf import LocalPDFSource
from ingestion.sources.sec_edgar import SECEDGARSource
from ingestion.sources.web_source import WebFilingSource
from rag.financial_math import FinancialMathEngine
from rag.financial_qa_engine import FinancialQAEngine
from rag.hybrid_router import HybridRetriever, RetrievalStrategy
from rag.langgraph_workflow import FinancialGraphRAGWorkflow
from rag.llm_provider import get_llm_provider
from evaluation.benchmark import FinancialBenchmark, REALISTIC_FINANCIAL_QUESTIONS

logger = logging.getLogger(__name__)

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Financial GraphRAG Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise Dark-Mode Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #A0AEC0;
        margin-bottom: 1.2rem;
    }
    .tier-reported {
        background-color: #1A365D;
        border-left: 4px solid #3182CE;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .tier-calculated {
        background-color: #234E52;
        border-left: 4px solid #319795;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .tier-interpreted {
        background-color: #2D3748;
        border-left: 4px solid #805AD5;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .badge-no-rag {
        background-color: #742A2A;
        color: #FEB2B2;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8em;
    }
    .badge-vector {
        background-color: #2C5282;
        color: #90CDF4;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8em;
    }
    .badge-graph {
        background-color: #22543D;
        color: #9AE6B4;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8em;
    }
    .badge-hybrid {
        background-color: #553C9A;
        color: #D6BCFA;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8em;
    }
    .psql-card {
        background-color: #1A202C;
        border: 1px solid #4A5568;
        border-radius: 6px;
        padding: 14px 18px;
        font-family: monospace;
        font-size: 0.9em;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def auto_initialize_system():
    """
    Automatic initialization on startup:
    1. Ensures required directories exist
    2. Ensures PostgreSQL tables, pgvector extension, and HNSW index exist
    3. Ensures Neo4j constraints and indexes exist
    4. Auto-populates data if database is empty
    """
    # 1. Directories
    for d in ["data/raw/annual_reports", "data/chunks", "data/processed/financial_facts", "data/processed/entities", "data/processed/relationships"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    # 2. PostgreSQL + pgvector
    pg = PostgresVectorClient()

    # 3. Neo4j
    kg = FinancialKnowledgeGraph()
    kg.init_constraints_and_indexes()

    # Auto-populate if Neo4j is empty
    summary = kg.get_graph_summary()
    if summary.get("total_nodes", 0) < 10:
        kg.build_full_graph()
        try:
            ground_truth_loader.load_missing_entities(kg.driver)
            ground_truth_loader.load_relationships(kg.driver)
        except Exception:
            pass

    qa = FinancialQAEngine()
    wf = FinancialGraphRAGWorkflow(postgres_client=pg, neo4j_kg=kg)
    return pg, kg, qa, wf


try:
    pg_client, kg_client, qa_engine, workflow = auto_initialize_system()
    system_ready = True
except Exception as e:
    system_ready = False
    st.error(f"System initialization error: {e}")

# --- Sidebar ---
with st.sidebar:
    st.title("📈 Financial GraphRAG")
    st.caption("Docling • pgvector • Neo4j • LangGraph")

    st.markdown("### System Status")
    if system_ready:
        st.success("🟢 PostgreSQL + pgvector: Online")
        st.success("🟢 Neo4j Graph DB: Online")
    else:
        st.error("🔴 Storage Layer: Offline")

    llm_prov = os.getenv("LLM_PROVIDER", "ollama").upper()
    llm_mod = os.getenv("LLM_MODEL", "qwen2.5:3b")
    emb_prov = os.getenv("EMBEDDING_PROVIDER", "ollama").upper()
    emb_mod = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    emb_dim = pg_client.current_dimension if system_ready else int(os.getenv("EMBEDDING_DIM", "768"))
    st.info(f"🤖 LLM: **{llm_prov}** (`{llm_mod}`)")
    st.info(f"📐 Vectors: **{emb_dim}-dim** (`{emb_mod}`)")

    st.divider()
    stats = pg_client.get_stats() if system_ready else {}
    kg_sum = kg_client.get_graph_summary() if system_ready else {}
    st.metric("Annual Reports", f"{stats.get('documents', 4)}", "FY23 - FY26")
    st.metric("Indexed Chunks", f"{stats.get('vector_chunks', 0):,}", "pgvector HNSW")
    st.metric("Graph Entities", f"{kg_sum.get('total_nodes', 0):,}", f"{kg_sum.get('total_relationships', 0):,} edges")

    st.divider()
    st.markdown(
        """
        **Terminal Database Inspection (`psql`):**
        ```bash
        docker exec -it financial-postgres \\
          psql -U financial_user -d financial_db
        ```
        """
    )

# --- Navigation Tabs ---
tabs = st.tabs([
    "⚖️ RAG Comparison",
    "💬 Financial Q&A",
    "📈 Financial Trends",
    "🕸️ Knowledge Graph",
    "🔍 Vector Search",
    "📑 Document & DB Explorer",
    "📥 Data Ingestion",
    "📊 Benchmark Dashboard",
])

# =============================================================
# TAB 1: RAG COMPARISON (Side-by-Side 4 Paradigms)
# =============================================================
with tabs[0]:
    st.markdown('<div class="main-header">⚖️ Four-Way RAG Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Side-by-side demonstration proving what the LLM knows without RAG, what Vector RAG retrieves, what GraphRAG adds, and why Hybrid GraphRAG is superior for financial analysis.</div>',
        unsafe_allow_html=True,
    )

    preset_questions = [
        "How did LTIMindtree's consolidated revenue grow between FY2022-23 and FY2023-24, and what was the growth rate?",
        "What was LTIMindtree's EBITDA, profit before tax, and profit after tax in FY2023-24?",
        "What were LTIMindtree's EBITDA margin and PAT margin in FY2022-23 and FY2023-24?",
        "What are LTIMindtree's core industry business segments, and what role do BFSI and Hi-Tech play?",
        "What was LTIMindtree's net worth and dividend distribution across reporting periods?",
        "What key macroeconomic and operational risks are identified that could impact profitability?",
        "What strategic initiatives did management highlight regarding Fit4Future and the Canvas.ai platform?",
        "Compare LTIMindtree's revenue across all four fiscal years (FY23 through FY26) and analyze the multi-year trajectory.",
        "How does the Fit4Future program connect to cost optimization, operational efficiency, and EBITDA margin?",
    ]

    col_q1, col_btn = st.columns([4, 1])
    with col_q1:
        sel_preset = st.selectbox("Select a benchmark question:", ["-- Select realistic financial query --"] + preset_questions)
        q_default = sel_preset if sel_preset != "-- Select realistic financial query --" else preset_questions[0]
        user_comp_q = st.text_input("Or enter a custom financial inquiry:", value=q_default)
    with col_btn:
        st.write("")
        st.write("")
        run_4way = st.button("⚡ Compare All 4", type="primary", use_container_width=True)

    if run_4way and user_comp_q:
        with st.spinner("Executing LangGraph StateGraph across all 4 modes (No RAG, Vector RAG, GraphRAG, Hybrid)..."):
            t_start = time.time()
            res_all = workflow.run_all_modes(user_comp_q)
            t_total = round(time.time() - t_start, 2)

            st.success(f"Executed all 4 paradigms in **{t_total}s**.")

            # Summary Metrics Row
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown('<span class="badge-no-rag">1. LLM without RAG</span>', unsafe_allow_html=True)
                lat_no = res_all["no_rag"]["latency_breakdown"].get("total", 0.0)
                st.metric("Latency", f"{lat_no:.2f}s", "Zero Context")
                st.caption("Parametric weights only")
            with sc2:
                st.markdown('<span class="badge-vector">2. Traditional Vector RAG</span>', unsafe_allow_html=True)
                lat_vec = res_all["vector_rag"]["latency_breakdown"].get("total", 0.0)
                st.metric("Latency", f"{lat_vec:.2f}s", "pgvector Chunks")
                st.caption(f"{len(res_all['vector_rag'].get('vector_chunks', []))} chunks retrieved")
            with sc3:
                st.markdown('<span class="badge-graph">3. GraphRAG</span>', unsafe_allow_html=True)
                lat_g = res_all["graph_rag"]["latency_breakdown"].get("total", 0.0)
                st.metric("Latency", f"{lat_g:.2f}s", "Neo4j Knowledge Graph")
                st.caption(f"{len(res_all['graph_rag'].get('graph_facts', []))} facts + paths")
            with sc4:
                st.markdown('<span class="badge-hybrid">4. Hybrid GraphRAG</span>', unsafe_allow_html=True)
                lat_h = res_all["hybrid"]["latency_breakdown"].get("total", 0.0)
                st.metric("Latency", f"{lat_h:.2f}s", "Gold Standard")
                st.caption("Facts + Math + Passages")

            st.divider()

            # Side-by-Side 4 Columns
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("### 1. LLM without RAG")
                st.markdown(
                    """
                    > **Quality**: ⚠️ **Hallucination Risk**
                    > 
                    > Lacks access to actual annual reports. Gives generic approximations or admits lack of filing access.
                    """
                )
                with st.expander("📄 Generated Response", expanded=True):
                    st.write(res_all["no_rag"].get("answer", ""))
                with st.expander("🔍 Retrieved Evidence"):
                    st.info("Zero Evidence. Model has no retrieved context.")

            with c2:
                st.markdown("### 2. Traditional Vector RAG")
                st.markdown(
                    """
                    > **Quality**: 🟡 **Fragmented Prose**
                    > 
                    > Retrieves semantically close passages. Prone to missing multi-year tables or disconnected entity relationships.
                    """
                )
                with st.expander("📄 Generated Response", expanded=True):
                    st.write(res_all["vector_rag"].get("answer", ""))
                with st.expander("🔍 Retrieved Evidence"):
                    for i, ck in enumerate(res_all["vector_rag"].get("vector_chunks", []), 1):
                        st.markdown(f"**Chunk #{i}** (Page {ck.get('page')}, {ck.get('fiscal_year')}) | Sim: `{ck.get('score', 0):.3f}`")
                        st.caption(ck.get("text", "")[:120] + "...")

            with c3:
                st.markdown("### 3. GraphRAG")
                st.markdown(
                    """
                    > **Quality**: 🟢 **Exact Ground Truth**
                    > 
                    > Returns verified numbers from Neo4j across fiscal years and traverses multi-hop causal entity links.
                    """
                )
                with st.expander("📄 Generated Response", expanded=True):
                    st.write(res_all["graph_rag"].get("answer", ""))
                with st.expander("🔍 Retrieved Evidence"):
                    facts = res_all["graph_rag"].get("graph_facts", [])
                    if facts:
                        for f in facts:
                            st.markdown(f"• **{f.get('metric')}** ({f.get('period')}): `{f.get('value'):,}` {f.get('unit')} (Page {f.get('page')})")
                    else:
                        st.info("Traversed entity edges.")
                    paths = res_all["graph_rag"].get("graph_paths", [])
                    if paths:
                        st.markdown("**Graph Paths:**")
                        for p in paths[:5]:
                            st.caption(f"({p.get('from')}) -[:{p.get('label')}]-> ({p.get('to')})")

            with c4:
                st.markdown("### 4. Hybrid GraphRAG")
                st.markdown(
                    """
                    > **Quality**: 🌟 **Comprehensive & Grounded**
                    > 
                    > Combines exact graph facts, deterministic Python math, and qualitative MD&A passages with verbatim page citations.
                    """
                )
                with st.expander("📄 Generated Response", expanded=True):
                    st.write(res_all["hybrid"].get("answer", ""))
                with st.expander("🔍 Retrieved Evidence"):
                    st.markdown(f"**Graph Facts**: `{len(res_all['hybrid'].get('graph_facts', []))}` facts")
                    st.markdown(f"**Vector Passages**: `{len(res_all['hybrid'].get('vector_chunks', []))}` passages")
                    calc = res_all["hybrid"].get("calculated_metrics", {})
                    if calc and calc.get("yearly_data"):
                        st.markdown("**Programmatic Calculations:**")
                        for entry in calc["yearly_data"]:
                            st.caption(f"{entry['period']}: {entry['value']:,.0f} {entry['unit']} (YoY: {entry['yoy_display']})")

            st.divider()
            st.subheader("💡 Why GraphRAG Produces Better, More Contextual Answers")
            st.markdown(
                """
                1. **No RAG Fails on Proprietary Facts**: Pre-trained LLMs do not contain exact numbers from specific company filings (FY23–FY26), producing hallucinations or unhelpful refusals.
                2. **Vector RAG Suffers from Boundary Chunking**: Traditional semantic search retrieves isolated paragraphs, but financial annual reports express core truths across multi-year tables and cross-document footnotes that get fragmented.
                3. **GraphRAG Preserves Metric Lineage**: In Neo4j, every financial KPI (`Revenue`, `EBITDA`, `PAT`) is an explicit node linked to its `ReportingPeriod`, `AnnualReport`, and `SourcePage`, enabling deterministic multi-year lookups and multi-hop causal tracing (`Fit4Future → Cost Optimization → EBITDA Margin`).
                4. **Hybrid is the Production Gold Standard**: Combining exact graph facts, Python deterministic math (preventing arithmetic hallucinations), and rich MD&A narrative passages provides the complete picture.
                """
            )

# =============================================================
# TAB 2: FINANCIAL Q&A
# =============================================================
with tabs[1]:
    st.markdown('<div class="main-header">💬 Financial Q&A & Provenance Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Rigorous corporate intelligence with three-tier distinction: <strong>Reported Data → Calculated Metrics → Qualitative Interpretation</strong>.</div>',
        unsafe_allow_html=True,
    )

    col_q, col_s = st.columns([3, 1])
    with col_s:
        qa_strat = st.selectbox(
            "Retrieval Strategy",
            ["Auto-Route (Intelligent)", "Hybrid Graph+Vector", "Vector Only (pgvector)", "Graph Only (Neo4j Cypher)"],
            key="qa_strat_select",
        )
        s_map = {
            "Auto-Route (Intelligent)": None,
            "Hybrid Graph+Vector": RetrievalStrategy.HYBRID_FUSION,
            "Vector Only (pgvector)": RetrievalStrategy.VECTOR_SEARCH,
            "Graph Only (Neo4j Cypher)": RetrievalStrategy.CYPHER_STRUCTURED,
        }
    with col_q:
        qa_query = st.text_input(
            "Enter your financial question:",
            value="What was LTIMindtree's revenue and EBITDA in FY2023-24 and what were the reported margins?",
            key="qa_query_input",
        )

    if st.button("🚀 Analyze & Generate Answer", type="primary", key="qa_btn"):
        with st.spinner("Executing retrieval, deterministic calculations, and grounded synthesis..."):
            t0 = time.time()
            resp = qa_engine.answer_question(qa_query, strategy=s_map[qa_strat])
            elapsed = round(time.time() - t0, 2)

            st.success(f"Analysis completed in **{elapsed}s** using `{resp.strategy_used}`")

            # Three-Tier Distinction
            st.markdown(
                """
                <div class="tier-reported">
                    <strong>1. REPORTED FINANCIAL FACTS (Ground Truth from Filings)</strong><br>
                    Exact audited numbers directly stated in company filings.
                </div>
                <div class="tier-calculated">
                    <strong>2. PROGRAMMATICALLY CALCULATED METRICS (Python Math Engine)</strong><br>
                    YoY growth rates, margin %, and CAGR computed deterministically in Python to prevent LLM math hallucinations.
                </div>
                <div class="tier-interpreted">
                    <strong>3. STRATEGIC INTERPRETATION (LLM Synthesis)</strong><br>
                    Contextual commentary on business drivers, segment shifts, and risks drawn strictly from retrieved evidence.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("💡 Synthesized Answer")
            st.markdown(resp.answer)

            # Programmatic Calculations
            if resp.calculated_metrics and resp.calculated_metrics.get("yearly_data"):
                st.divider()
                st.subheader("🧮 Programmatically Calculated Metrics (Python Math)")
                calc_rows = []
                for entry in resp.calculated_metrics["yearly_data"]:
                    calc_rows.append({
                        "Fiscal Year": entry["period"],
                        "Reported Value": f"{entry['value']:,.2f} {entry['unit']}",
                        "YoY Growth Rate": entry["yoy_display"],
                        "Source Page": f"Page {entry.get('page', 'N/A')}",
                    })
                st.dataframe(pd.DataFrame(calc_rows), use_container_width=True)

                if resp.calculated_metrics.get("cagr"):
                    st.info(f"**Multi-Year CAGR**: `{resp.calculated_metrics['cagr']:.2f}%` | **Overall Change**: `{resp.calculated_metrics.get('overall_change_pct', 0):+.2f}%`")

            # Source Citations
            st.divider()
            st.subheader("📑 Verbatim Citations & Provenance")
            if resp.citations:
                c_cols = st.columns(2)
                for idx, cite in enumerate(resp.citations):
                    with c_cols[idx % 2]:
                        b_type = cite.get("source_type", "Citation")
                        with st.expander(f"Citation #{idx+1}: [{cite.get('fiscal_year', 'General')}] {b_type} (Page {cite.get('page', 'N/A')})", expanded=True):
                            st.markdown(f"**Document**: `{cite.get('document_id', 'Annual Report')}` | **Section**: `{cite.get('section', 'General')}`")
                            if cite.get("score"):
                                st.caption(f"Vector Similarity Score: `{cite.get('score'):.4f}`")
                            st.markdown(f"**Evidence Snippet**: \n> {cite.get('claim', '')}")
            else:
                st.info("No direct citations required.")

            # Automatic Knowledge Subgraph
            if resp.retrieval_context.graph_paths:
                st.divider()
                st.subheader("🕸️ Graph Traversal Evidence (Auto-Rendered)")
                for p in resp.retrieval_context.graph_paths[:6]:
                    st.caption(f"• `({p.get('from')})` ── **[{p.get('label')}]** ──> `({p.get('to')})`")

# =============================================================
# TAB 3: FINANCIAL TRENDS
# =============================================================
with tabs[2]:
    st.markdown('<div class="main-header">📈 Multi-Year Financial Performance & Trends</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Deterministic multi-year financial ratios, margin analysis, and segment performance across four fiscal years (FY23 through FY26).</div>',
        unsafe_allow_html=True,
    )

    rev_facts = kg_client.get_financial_facts_for_metric("revenue")
    ebitda_facts = kg_client.get_financial_facts_for_metric("ebitda")
    pat_facts = kg_client.get_financial_facts_for_metric("profit_after_tax")

    intel = FinancialMathEngine.build_financial_intelligence_report(rev_facts, ebitda_facts, pat_facts)

    # Multi-Year Overview Table
    st.subheader("📊 Multi-Year Comparison Matrix")
    st.markdown(FinancialMathEngine.format_as_markdown_table(intel))

    st.divider()

    # Interactive Charts
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("📈 Revenue & EBITDA Trends (INR Million)")
        plot_data = []
        for r in intel["revenue_analysis"].get("yearly_data", []):
            plot_data.append({"Fiscal Year": r["period"], "Metric": "Revenue", "Value": r["value"]})
        for e in intel["ebitda_analysis"].get("yearly_data", []):
            plot_data.append({"Fiscal Year": e["period"], "Metric": "EBITDA", "Value": e["value"]})
        for p in intel["pat_analysis"].get("yearly_data", []):
            plot_data.append({"Fiscal Year": p["period"], "Metric": "PAT (Net Profit)", "Value": p["value"]})

        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            fig1 = px.line(df_plot, x="Fiscal Year", y="Value", color="Metric", markers=True, title="Key Financial KPIs (Audited)")
            fig1.update_layout(height=380)
            st.plotly_chart(fig1, use_container_width=True)

    with col_c2:
        st.subheader("📉 Operating & Net Profit Margins (%)")
        margin_data = []
        for em in intel.get("ebitda_margins", []):
            margin_data.append({"Fiscal Year": em["period"], "Margin Type": "EBITDA Margin (%)", "Margin": em["margin_pct"]})
        for pm in intel.get("pat_margins", []):
            margin_data.append({"Fiscal Year": pm["period"], "Margin Type": "PAT Margin (%)", "Margin": pm["margin_pct"]})

        if margin_data:
            df_m = pd.DataFrame(margin_data)
            fig2 = px.bar(df_m, x="Fiscal Year", y="Margin", color="Margin Type", barmode="group", title="EBITDA Margin vs PAT Margin")
            fig2.update_layout(height=380, yaxis_ticksuffix="%")
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Segment Breakdown
    st.subheader("🏢 Business Segments & Geographies")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Industry Verticals**")
        segments = [
            {"Segment": "Banking, Financial Services & Insurance (BFSI)", "Contribution": "~37%", "Focus": "Digital banking, payments, risk transformation"},
            {"Segment": "Hi-Tech, Media & Entertainment", "Contribution": "~23%", "Focus": "Platform engineering, AI products, cloud SaaS"},
            {"Segment": "Manufacturing & Resources", "Contribution": "~18%", "Focus": "Industry 4.0, IoT, supply chain resilience"},
            {"Segment": "Retail, CPG & Travel", "Contribution": "~15%", "Focus": "Omnichannel experiences, customer analytics"},
            {"Segment": "Health & Life Sciences", "Contribution": "~7%", "Focus": "Clinical trials data, regulatory compliance"},
        ]
        st.dataframe(pd.DataFrame(segments), use_container_width=True)
    with col_s2:
        st.markdown("**Geographic Footprint**")
        geos = [
            {"Geography": "North America", "Share": "~72%", "Role": "Primary revenue engine"},
            {"Geography": "Europe", "Share": "~15%", "Role": "Key digital transformation market"},
            {"Geography": "India", "Share": "~8%", "Role": "Domestic enterprise & public sector"},
            {"Geography": "Rest of the World", "Share": "~5%", "Role": "APAC, Middle East expansion"},
        ]
        st.dataframe(pd.DataFrame(geos), use_container_width=True)

# =============================================================
# TAB 4: KNOWLEDGE GRAPH
# =============================================================
with tabs[3]:
    st.markdown('<div class="main-header">🕸️ Neo4j Financial Knowledge Graph</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive physics-based graph visualization of corporate entities, financial KPIs, business segments, risks, and strategic initiatives.</div>',
        unsafe_allow_html=True,
    )

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        kg_filter = st.text_input("Filter Subgraph by Entity, Metric, or Strategy:", value="", placeholder="e.g. EBITDA, Revenue, Fit4Future, BFSI, Risk...")
    with col_f2:
        kg_limit = st.slider("Max Nodes to Render", min_value=15, max_value=80, value=35)

    subgraph = kg_client.get_subgraph_for_visualization(
        entity_filter=kg_filter.strip() if kg_filter.strip() else None,
        limit=kg_limit,
    )

    st.markdown(f"**Rendered Subgraph**: `{len(subgraph['nodes'])}` Nodes, `{len(subgraph['edges'])}` Relationships")

    if subgraph["nodes"]:
        from pyvis.network import Network

        net = Network(height="520px", width="100%", bgcolor="#1A202C", font_color="#FFFFFF")
        net.force_atlas_2based()

        color_palette = {
            "Company": "#E53E3E",
            "AnnualReport": "#ED8936",
            "FinancialMetric": "#48BB78",
            "FinancialValue": "#38B2AC",
            "ReportingPeriod": "#4299E1",
            "BusinessSegment": "#9F7AEA",
            "Risk": "#F56565",
            "Strategy": "#ECC94B",
            "SourcePage": "#718096",
            "Entity": "#667EEA",
            "Capability": "#4FD1C5",
            "BusinessTheme": "#F6AD55",
            "Technology": "#B794F4",
            "Program": "#FC8181",
        }

        for n in subgraph["nodes"]:
            grp = n.get("group", "Entity")
            c = color_palette.get(grp, "#A0AEC0")
            net.add_node(
                n["id"],
                label=str(n["label"])[:25],
                title=f"Type: {grp}\nName: {n['label']}",
                color=c,
                size=24 if grp in ["Company", "FinancialMetric"] else 16,
            )

        for e in subgraph["edges"]:
            net.add_edge(e["from"], e["to"], title=e["label"], label=e["label"][:14])

        try:
            html_raw = net.generate_html()
        except Exception:
            html_file = PROJECT_ROOT / "data" / "graph_viz.html"
            html_file.parent.mkdir(parents=True, exist_ok=True)
            net.save_graph(str(html_file))
            with open(html_file, "r", encoding="utf-8") as f:
                html_raw = f.read()
            try:
                html_file.unlink(missing_ok=True)
            except Exception:
                pass

        components.html(html_raw, height=540)

    # Subgraph Table View
    with st.expander("🔍 View Raw Relationship Edges in Subgraph"):
        st.dataframe(pd.DataFrame(subgraph["edges"]), use_container_width=True)

# =============================================================
# TAB 5: VECTOR SEARCH
# =============================================================
with tabs[4]:
    st.markdown('<div class="main-header">🔍 PostgreSQL pgvector Semantic Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect raw vector retrieval on 768-dimensional embeddings using PostgreSQL cosine distance (<=>).</div>',
        unsafe_allow_html=True,
    )

    v_col1, v_col2, v_col3 = st.columns([3, 1, 1])
    with v_col1:
        v_query = st.text_input("Vector Search Query:", value="Generative AI Canvas platform enterprise client adoption", key="vec_search_q")
    with v_col2:
        v_year = st.selectbox("Filter Fiscal Year:", ["All", "FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"], key="vec_search_fy")
    with v_col3:
        v_type = st.selectbox("Chunk Type:", ["All", "prose", "table"], key="vec_search_t")

    v_topk = st.slider("Top K Chunks", min_value=1, max_value=15, value=5, key="vec_search_topk")

    if st.button("🔎 Search pgvector in PostgreSQL", type="primary", key="vec_search_btn"):
        with st.spinner("Generating query embedding and querying pgvector..."):
            t0 = time.time()
            emb = qa_engine.retriever.embedder.embed_query(v_query)
            chunks = pg_client.search_similarity(
                query_embedding=emb,
                top_k=v_topk,
                fiscal_year=v_year if v_year != "All" else None,
                chunk_type=v_type if v_type != "All" else None,
            )
            lat = round(time.time() - t0, 3)

            st.success(f"Retrieved **{len(chunks)} chunks** from PostgreSQL in **{lat}s**.")

            for i, c in enumerate(chunks, 1):
                badge = "📊 TABLE" if c.get("chunk_type") == "table" else "📄 PROSE"
                with st.expander(f"#{i} {badge} | Similarity: `{c.get('score'):.4f}` | {c.get('fiscal_year')} (Page {c.get('page')}) — {c.get('section')}"):
                    st.markdown(f"**Chunk ID**: `{c.get('chunk_id')}` | **Document**: `{c.get('document_id')}`")
                    st.markdown("### Content")
                    st.markdown(c.get("text", ""))
                    st.divider()
                    st.caption(f"Metadata: `{json.dumps(c.get('metadata', {}))}`")

# =============================================================
# TAB 6: DOCUMENT & DATABASE EXPLORER
# =============================================================
with tabs[5]:
    st.markdown('<div class="main-header">📑 Document & Database Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect documents, semantic chunks, JSON metadata, vector embeddings, extracted tables, and PostgreSQL stats.</div>',
        unsafe_allow_html=True,
    )

    # PSQL Terminal Connection Card
    st.markdown(
        """
        <div class="psql-card">
            <strong>Direct PostgreSQL Terminal Access (`psql`):</strong><br>
            <code>docker exec -it financial-postgres psql -U financial_user -d financial_db</code><br>
            <span style="color:#A0AEC0; font-size:0.85em;">Inspect pgvector tables, run EXPLAIN ANALYZE on cosine distance (<=>) queries, or view row stats directly.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    db_tabs = st.tabs([
        "📚 Ingested Documents",
        "📑 Chunks",
        "🏷️ Metadata",
        "📐 Embeddings",
        "📊 Extracted Tables",
        "📈 Financial Metrics",
        "💻 SQL Console",
    ])

    stats = pg_client.get_stats()

    # Sub-tab 1: Documents
    with db_tabs[0]:
        st.subheader("Ingested Documents & Coverage Manifest")
        sql_docs = "SELECT document_id, company, ticker, reporting_period, title, content_hash, metadata FROM documents;"
        try:
            cols, rows = pg_client.execute_query(sql_docs)
            st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading documents: {e}")

    # Sub-tab 2: Chunks
    with db_tabs[1]:
        st.subheader("Indexed Document Chunks")
        st.markdown(f"Total Chunks in PostgreSQL: **{stats.get('vector_chunks', 0):,}**")
        c_fy = st.selectbox("Filter Chunks by Fiscal Year:", ["All", "FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"], key="db_c_fy")
        c_search = st.text_input("Search Chunk Text:", value="", key="db_c_search")

        where_cond = []
        if c_fy != "All":
            where_cond.append(f"fiscal_year = '{c_fy}'")
        if c_search.strip():
            where_cond.append(f"text ILIKE '%{c_search.strip()}%'")

        where_str = ("WHERE " + " AND ".join(where_cond)) if where_cond else ""
        sql_chunks = f"SELECT chunk_id, document_id, fiscal_year, page, section, chunk_type, LEFT(text, 120) AS text_preview FROM vector_chunks {where_str} ORDER BY vector_id ASC LIMIT 25;"
        try:
            cols, rows = pg_client.execute_query(sql_chunks)
            st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading chunks: {e}")

    # Sub-tab 3: Metadata
    with db_tabs[2]:
        st.subheader("Chunk & Document Metadata")
        sql_meta = "SELECT chunk_id, fiscal_year, page, chunk_type, metadata FROM vector_chunks LIMIT 15;"
        cols, rows = pg_client.execute_query(sql_meta)
        st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True)

    # Sub-tab 4: Embeddings
    with db_tabs[3]:
        st.subheader("pgvector Vector Embeddings (768 Dimensions)")
        st.info("Vector type: `vector(768)` indexed with HNSW (`vector_cosine_ops`) for fast sub-millisecond search.")
        sql_emb = "SELECT chunk_id, fiscal_year, page, LEFT(text, 80) as text_snippet, embedding::text as raw_vector FROM vector_chunks LIMIT 5;"
        cols, rows = pg_client.execute_query(sql_emb)
        if rows:
            for r in rows:
                with st.expander(f"Vector for `{r[0]}` ({r[1]}, Page {r[2]}): {r[3]}..."):
                    vec_str = r[4]
                    try:
                        floats = [float(x) for x in vec_str.strip("[]").split(",")[:10]]
                        st.markdown(f"**First 10 Dimensions**: `{floats}`")
                        st.caption("Full vector length: 768 dimensions.")
                    except Exception:
                        st.code(vec_str[:150] + "...")

    # Sub-tab 5: Tables
    with db_tabs[4]:
        st.subheader("Extracted Financial Tables (Preserved Markdown Pipes)")
        sql_tbls = "SELECT chunk_id, fiscal_year, page, section, text FROM vector_chunks WHERE chunk_type = 'table' LIMIT 8;"
        cols, rows = pg_client.execute_query(sql_tbls)
        if rows:
            for r in rows:
                with st.expander(f"📊 Table: `{r[0]}` ({r[1]}, Page {r[2]} - {r[3]})"):
                    st.markdown(r[4])
        else:
            st.info("No table chunks found.")

    # Sub-tab 6: Financial Metrics
    with db_tabs[5]:
        st.subheader("Canonical Ground-Truth Financial Metrics")
        facts = kg_client.get_financial_facts_for_metric("")
        if facts:
            st.dataframe(pd.DataFrame(facts), use_container_width=True)
        else:
            st.info("No structured facts in Knowledge Graph.")

    # Sub-tab 7: SQL Console
    with db_tabs[6]:
        st.subheader("Read-Only SQL Console")
        default_sql = "SELECT fiscal_year, chunk_type, count(*) as count FROM vector_chunks GROUP BY fiscal_year, chunk_type ORDER BY fiscal_year;"
        sql_in = st.text_area("Enter SQL query (SELECT / EXPLAIN only):", value=default_sql, height=90)
        if st.button("▶️ Execute Query", key="sql_run_btn"):
            try:
                c_names, r_data = pg_client.execute_query(sql_in)
                st.dataframe(pd.DataFrame(r_data, columns=c_names), use_container_width=True)
                st.success(f"Returned {len(r_data)} rows.")
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")

# =============================================================
# TAB 7: DATA INGESTION
# =============================================================
with tabs[6]:
    st.markdown('<div class="main-header">📥 Document Ingestion Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Idempotent ingestion of annual reports via local PDF upload, filing URLs, or SEC EDGAR API discovery.</div>',
        unsafe_allow_html=True,
    )

    ingest_col1, ingest_col2 = st.columns(2)

    with ingest_col1:
        st.subheader("1. Add Local Annual Report PDF")
        uploaded_pdf = st.file_uploader("Upload Company Annual Report (PDF):", type=["pdf"])
        if uploaded_pdf is not None:
            save_path = PROJECT_ROOT / "data" / "raw" / "annual_reports" / uploaded_pdf.name
            with open(save_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.success(f"Saved `{uploaded_pdf.name}` to `data/raw/annual_reports/`")

        st.divider()

        st.subheader("2. Fetch from SEC EDGAR API")
        sec_ticker = st.text_input("Company Ticker for SEC 10-K Ingestion:", value="AAPL", help="Queries the public SEC EDGAR Submissions API")
        sec_max = st.number_input("Max 10-K Filings to Pull:", min_value=1, max_value=5, value=1)

    with ingest_col2:
        st.subheader("3. Execute Ingestion Pipeline")
        st.markdown(
            """
            The pipeline executes:
            1. **Duplicate Check**: SHA-256 content hashing (idempotent, skips duplicates).
            2. **Docling Parsing**: Preserves headings, reading-order, and markdown tables.
            3. **Semantic Chunking**: Structure-aware chunking without breaking tables.
            4. **Vector Embeddings**: 768-dim embeddings stored in PostgreSQL pgvector.
            5. **Knowledge Graph Enrichment**: Financial facts and entities loaded into Neo4j.
            """
        )

        ingest_mode = st.radio("Select Ingestion Scope:", ["Local PDFs (data/raw/annual_reports)", f"SEC EDGAR 10-K ({sec_ticker})", "All Sources"])

        if st.button("⚡ Run Ingestion Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running ingestion pipeline..."):
                sources = []
                raw_dir = PROJECT_ROOT / "data" / "raw" / "annual_reports"
                if "Local" in ingest_mode or "All" in ingest_mode:
                    sources.append(LocalPDFSource(directory_path=raw_dir))
                if "SEC" in ingest_mode or "All" in ingest_mode:
                    sources.append(SECEDGARSource(ticker=sec_ticker, max_filings=int(sec_max)))

                pipeline = FinancialIngestionPipeline()
                res = pipeline.run_ingestion(sources=sources, max_docs=5)
                st.success(f"Ingestion completed! Processed {res['processed']} documents, created {res['chunks_created']} semantic chunks.")
                st.rerun()

# =============================================================
# TAB 8: BENCHMARK DASHBOARD
# =============================================================
with tabs[7]:
    st.markdown('<div class="main-header">📊 Four-Way Benchmark Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Empirical evaluation comparing Answer Accuracy, Faithfulness, Citation Accuracy, Retrieval Recall, Precision, Latency, and Cost across all four paradigms.</div>',
        unsafe_allow_html=True,
    )

    bench_json = PROJECT_ROOT / "evaluation" / "benchmark_results.json"
    bench_data = None
    if bench_json.exists():
        with open(bench_json, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

    if bench_data and bench_data.get("summary"):
        summary_df = pd.DataFrame(bench_data["summary"])

        st.subheader("🏆 Summary Leaderboard")
        st.dataframe(summary_df, use_container_width=True)

        st.divider()

        # Visual Comparison Charts
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.subheader("📈 Accuracy, Faithfulness & Citations")
            fig_acc = px.bar(
                summary_df,
                x="Paradigm",
                y=["Answer Accuracy", "Faithfulness", "Citation Accuracy"],
                barmode="group",
                title="Quality & Grounding Metrics by Paradigm",
                color_discrete_sequence=["#4299E1", "#48BB78", "#9F7AEA"],
            )
            fig_acc.update_layout(height=360, yaxis_range=[0, 1.1])
            st.plotly_chart(fig_acc, use_container_width=True)

        with c_chart2:
            st.subheader("⚡ Latency & Cost Efficiency")
            fig_lat = px.bar(
                summary_df,
                x="Paradigm",
                y="Avg Latency (s)",
                color="Paradigm",
                title="Average Query Latency (Seconds)",
                color_discrete_sequence=["#FEB2B2", "#90CDF4", "#9AE6B4", "#D6BCFA"],
            )
            fig_lat.update_layout(height=360)
            st.plotly_chart(fig_lat, use_container_width=True)

        st.divider()

        # Detailed Question Breakdown
        st.subheader("🔍 Question-by-Question Detailed Results")
        for q_entry in bench_data.get("detailed", []):
            with st.expander(f"[{q_entry.get('category')}] {q_entry.get('question')}"):
                for m_name, m_data in q_entry.get("modes", {}).items():
                    st.markdown(f"**{m_name}** | Latency: `{m_data.get('latency_seconds', 0)}s` | Acc: `{m_data.get('answer_accuracy', 0)}` | Faith: `{m_data.get('faithfulness', 0)}` | Tokens: `{m_data.get('total_tokens', 0)}`")
                    st.caption(f"Snippet: {m_data.get('answer_snippet', '')}")

    st.divider()
    b_col1, b_col2 = st.columns([3, 1])
    with b_col1:
        st.write("Run a live benchmark evaluation across the realistic financial question set.")
    with b_col2:
        if st.button("⚡ Run Live 3-Question Benchmark", type="primary", key="bench_run_btn"):
            with st.spinner("Executing live 4-way evaluation benchmark across all models and tools..."):
                bm = FinancialBenchmark()
                bm.run_benchmark(max_questions=3)
                st.success("Benchmark completed! Reloading dashboard...")
                st.rerun()
