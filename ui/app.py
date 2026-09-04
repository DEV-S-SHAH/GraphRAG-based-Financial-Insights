"""
Phase 13: Streamlit UI for LTIMindtree multi-year GraphRAG.

Neo4j Graph + FAISS Vector + Ollama qwen2.5:3b
Four fiscal years: FY2022-23, FY2023-24, FY2024-25, FY2025-26.
"""

import sys
import os

import streamlit as st


# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from rag.app import answer_question


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="LTIMindtree GraphRAG",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📊 LTIMindtree Multi-Year GraphRAG")
st.caption(
    "Hybrid retrieval: Neo4j Graph + FAISS Vector + Ollama qwen2.5:3b"
    "  •  FY2022-23 | FY2023-24 | FY2024-25 | FY2025-26"
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("System")

    st.success("Neo4j Graph (multi-year)")
    st.success("FAISS Vector (4,570 chunks)")
    st.success("Ollama qwen2.5:3b")

    st.divider()

    st.markdown(
        """
### Data Coverage

| Fiscal Year | Report |
|---|---|
| FY2022-23 | Annual Report 2022-23 |
| FY2023-24 | Integrated Annual Report FY2023-24 |
| FY2024-25 | Integrated Annual Report FY2024-25 |
| FY2025-26 | Integrated Annual Report FY2025-26 |

### Facts

- 65 validated financial facts
- 32 grounded entities
- 19 evidence-backed relationships
- 6,206 narrative chunks
"""
    )

    st.divider()

    st.markdown(
        """
### Pipeline

Question
↓
Fiscal Year Detection
↓
Query Classification
↓
┌─────────────┬─────────────┐
↓             ↓
Neo4j Graph   FAISS Vector
Retrieval     Retrieval
↓             ↓
└─────────────┬─────────────┘
↓
Context Fusion
  (with fiscal-year scope)
↓
Ollama qwen2.5:3b
↓
Grounded Answer
  (evidence vs. inference)
"""
    )


# ---------------------------------------------------------
# Example questions
# ---------------------------------------------------------

st.subheader("Ask a question")

examples = [
    "What was LTIMindtree's EBITDA margin in FY2024-25?",
    "What was LTIMindtree's revenue in FY2023-24?",
    "Compare revenue across all fiscal years.",
    "How does cost optimization affect EBITDA margin?",
    "What risks could affect revenue and profitability?",
    "What strategic priorities did LTIMindtree discuss regarding AI?",
    "How did revenue change from FY2022-23 to FY2024-25?",
    "What was the EBITDA margin trend across four fiscal years?",
    "What was LTIMindtree's revenue and EBITDA margin in FY2025-26?",
    "What initiatives support order inflow?",
    "What did management say about GenAI productivity?",
    "What are the risks from macroeconomic uncertainty?",
]

selected = st.selectbox(
    "Example questions",
    ["Custom question"] + examples,
)


if selected != "Custom question":

    question = selected

else:

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "e.g. What was LTIMindtree's EBITDA margin in FY2024-25?"
        ),
    )


# ---------------------------------------------------------
# Ask button
# ---------------------------------------------------------

ask = st.button("Ask", type="primary")


if ask and question:

    with st.spinner("Retrieving + generating..."):

        try:

            result = answer_question(
                question,
                top_k=5,
            )

        except Exception as exc:

            st.error(f"Error: {exc}")
            st.stop()

    # ---- ANSWER ----
    answer = result.get("answer", "")
    fyc = result.get("fiscal_year_context", {})

    st.subheader("Answer")
    st.markdown(answer)

    # ---- FISCAL-YEAR CONTEXT ----
    if fyc:
        with st.expander("Fiscal Year Detection", expanded=False):
            st.write(f"**Detected years:** {', '.join(fyc.get('detected_years', [])) or 'None (all years)'}")
            st.write(f"**Comparison query:** {fyc.get('comparison', False)}")
            st.write(f"**Scope:** {fyc.get('scope_note', '')}")

    # ---- EVIDENCE ----
    st.subheader("Evidence Used")

    gc = result.get("graph_context", {})
    vc = result.get("document_chunks", [])

    facts = gc.get("financial_facts", [])
    sems = gc.get("semantic_paths", [])
    entities = gc.get("entities", [])

    cols = st.columns(3)

    with cols[0]:
        st.markdown("**Financial Facts**")
        if facts:
            for f in facts:
                period = f.get("period", "?")
                page = f.get("page", "?")
                st.write(
                    f"- {f.get('metric')} = "
                    f"{f.get('value')} {f.get('unit')} "
                    f"({period}, p{page})"
                )
        else:
            st.write("None")

    with cols[1]:
        st.markdown("**Semantic Paths**")
        if sems:
            for sp in sems:
                nodes = sp.get("nodes", sp.get("path", []))
                rels = sp.get("relationships", [])
                chain = " → ".join(
                    f"{n} [{r}]"
                    for n, r in zip(nodes, rels + [""])
                )
                st.write(f"- {chain}")
        else:
            st.write("None")

    with cols[2]:
        st.markdown("**Entities**")
        if entities:
            for ent in entities:
                e = ent.get("entity", {})
                name = e.get("entity") if isinstance(e, dict) else e
                if name:
                    outgoing = e.get("outgoing", [])
                    incoming = e.get("incoming", [])
                    rels = outgoing + incoming
                    rel_str = ", ".join(
                        r.get("relationship", "")
                        for r in rels
                    )
                    st.write(f"- **{name}** ({rel_str})")
        else:
            st.write("None")

    # ---- VECTOR EVIDENCE ----
    if vc:
        with st.expander("Document Evidence (top chunks)", expanded=False):
            for i, c in enumerate(vc[:5]):
                fy = c.get("fiscal_year", c.get("period", "?"))
                page = c.get("page", "?")
                score = c.get("score", 0)
                section = c.get("section", "")
                text = c.get("text", "")[:300]
                st.write(
                    f"**[{i+1}] {fy} p{page}** "
                    f"(score={score:.3f}, {section})"
                )
                st.write(text)
                st.write("---")
