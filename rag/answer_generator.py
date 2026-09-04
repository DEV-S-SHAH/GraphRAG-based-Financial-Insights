"""
Phase 12: Answer generation with evidence/inference distinction.

Sends fused context (graph + vector) to Ollama for answer generation.
Includes retry/timeout handling and evidence classification in the result.
"""

import time
import requests
from typing import Any, Dict, Optional


MAX_RETRIES = 3
RETRY_DELAY_S = 2
TIMEOUT_S = 120


class AnswerGenerator:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        self.model = model
        self.ollama_url = ollama_url

    # -----------------------------------------------------------
    # Ollama call with retry + timeout
    # -----------------------------------------------------------
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama with retries and exponential back-off."""
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=TIMEOUT_S,
                )
                resp.raise_for_status()
                data = resp.json()

                if "response" not in data:
                    raise RuntimeError(
                        f"Unexpected Ollama response: {data}"
                    )
                return data["response"].strip()

            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.ReadTimeout,
                RuntimeError,
            ) as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_S * attempt
                    time.sleep(delay)

        raise ConnectionError(
            f"Ollama call failed after {MAX_RETRIES} "
            f"attempts: {last_error}"
        )

    # -----------------------------------------------------------
    # Main generation
    # -----------------------------------------------------------
    def generate(self, context: str) -> str:
        """
        Generate an answer from the fused graph + vector context.

        The context string must contain:
        - the user question
        - fiscal-year context (detected years, scope note)
        - graph evidence (financial facts, entities, paths)
        - document evidence (retrieved chunks with page numbers)
        - source metadata
        """

        prompt = f"""You are a financial research assistant for LTIMindtree Limited (LTIM).

The company is LTIMindtree Limited (ticker: LTIM), formerly L&T Infotech merged with Mindtree. Data is available for four fiscal years: FY2022-23, FY2023-24, FY2024-25, and FY2025-26.

FISCAL-YEAR INSTRUCTIONS:
- Each financial fact includes a "period" field (e.g. FY2023-24). Report the specific period the fact belongs to.
- When the question refers to a specific year, answer only for that year's data.
- When asked to compare or show trends, present data for all available years and calculate the year-on-year change.
- Never mix values from different fiscal years without explicitly stating which year each number is from.

Evidence vs Inference:
- EVIDENCE: Numbers from FINANCIAL FACTS or verbatim text from DOCUMENT EVIDENCE. Always cite the period and page number.
- INFERRED: Conclusions drawn from GRAPH PATHS (entity relationships). State explicitly that these are inferred connections, not direct measurements.
- If both FINANCIAL FACTS and DOCUMENT EVIDENCE are empty for a question, say "Not enough evidence" and explain what additional information would be needed.

Answer the user's question using ONLY the supplied graph evidence and document evidence below. Do not use outside knowledge.

The context contains two kinds of evidence:

1. GRAPH EVIDENCE
   - FINANCIAL FACTS: exact structured values from the source document (metric, value, unit, period, page). These are direct facts. Report them precisely.
   - ENTITIES: named items in the knowledge graph (concepts, programs, initiatives). These are NOT financial metrics.
   - GRAPH PATHS: chains of entities connected by semantic relationships (e.g. IMPROVES, SUPPORTS, DRIVES). These represent conceptual CONNECTIONS in the knowledge graph, NOT proven causation. These carry evidence of indirect connection, not direct measured effect.

2. DOCUMENT EVIDENCE
   - Retrieved text passages from the annual report with page numbers.
   - This is DIRECT SOURCE material. Use it to answer narrative questions (risks, priorities, initiatives, outlook).

RULES:

1. Do not invent facts.
2. Distinguish between graph relationships (inferred) and factual financial evidence (direct).
3. If a graph relationship is indirect, explicitly say it is indirect.
4. Never claim a relationship means causation unless the source explicitly says so. A path like "A IMPROVES B SUPPORTS C" means the graph CONNECTS A to C, not that A caused C to change.
5. Cite page numbers whenever available.
6. If evidence is insufficient, say so.
7. Prefer exact financial values from structured graph facts over generated estimates.
8. Every financial answer should include: Metric, Value, Period, and Page when available.
9. For narrative/vector answers, quote or paraphrase the retrieved source text and provide the page number.
10. For multi-hop graph paths, explain each hop step by step.
11. DOCUMENT EVIDENCE is direct source material. If the DOCUMENT EVIDENCE contains text relevant to the question, USE it and paraphrase or quote it with the page number. The "not enough evidence" response should ONLY be used when BOTH the graph and document evidence sections are genuinely empty for the topic.
12. When the question asks about risks, priorities, initiatives, or other narrative topics and DOCUMENT EVIDENCE is provided, enumerate the specific items the document mentions. Do not refuse to answer just because the evidence is qualitative or prose.
13. When GRAPH PATHS are present in the context, you MUST address them as the primary evidence. For any question about how one thing "affects", "drives", "impacts" or "relates to" another, read the GRAPH PATHS section and explicitly walk the path hop by hop, describing the connection as indirect.
14. NEVER write "X increased/had a positive effect on/boosted Y" for a graph relationship. Graph relationships are directional CONNECTIONS in a knowledge graph, not measured effects. Always say "the graph represents X as IMPROVING Y" or "X is connected to Y through a SUPPORTS relationship".
15. Never invent or assume data for years not present in the context. If a value is available for FY2024-25 but not FY2023-24, state what is available.
16. NEVER give investment advice. Questions like "Should I buy X stock?", "Is X a good investment?", or "Will X grow next year?" must be answered with: "I cannot provide investment advice. I can only answer questions based on the data in the annual reports. Please consult a financial advisor for investment decisions."

CRITICAL BEHAVIOR EXAMPLE:
If asked "How does cost optimization affect EBITDA margin?" and the graph shows
"cost optimization --[DRIVES]--> ebitda_margin"
plus facts "ebitda_margin = 17.1% (FY2024-25, p8)" and "ebitda_margin = 18.0% (FY2023-24, p9)", respond like this:

  [Evidence] The graph directly connects cost optimization to EBITDA margin through a DRIVES relationship. However, this connection is an inferred relationship in the knowledge graph, not a quantified causal measurement.

  The reported EBITDA margins were:
  - FY2022-23: 18.4% (page 12)
  - FY2023-24: 18.0% (page 9)
  - FY2024-25: 17.1% (page 8)

  The graph establishes that cost optimization is associated with EBITDA margin, but the available evidence does not quantify how much cost optimization changed the EBITDA margin.

FUSED CONTEXT:
{context}

FINAL INSTRUCTION:
Read the DOCUMENT EVIDENCE section carefully. It contains verbatim passages from the annual report with page numbers. If the question is a narrative/informational question (priorities, risks, strategies, initiatives), then the ANSWER is found IN the DOCUMENT EVIDENCE — summarize and quote those passages and cite their page numbers.

Now answer the user's question directly and concisely.
"""

        return self._call_ollama(prompt)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 80)
    print("OLLAMA ANSWER GENERATOR TEST (multi-year)")
    print("=" * 80)

    generator = AnswerGenerator()

    test_context = """
QUESTION:
What was LTIMindtree's EBITDA in FY2023-24?

FISCAL-YEAR CONTEXT
------------------
The question mentions fiscal year(s): FY2023-24. Focus the answer on this year's data.

GRAPH EVIDENCE
---------------
FINANCIAL FACTS:
- ebitda = 63874 INR million (period: FY2023-24, page: 72, section: Key Performance Indicators)

ENTITIES:
No entities retrieved.

GRAPH PATHS:
No graph paths retrieved.

DOCUMENT EVIDENCE
-----------------
[Page 72]
EBITDA 63,874 55,685 8,189

SOURCE METADATA
---------------
Company: LTIMindtree Limited
Ticker: LTIM
Fiscal Years Covered: FY2023-24
"""

    print("\nTest context:")
    print(test_context)

    print("\nGenerating answer...")

    answer = generator.generate(
        test_context
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)
