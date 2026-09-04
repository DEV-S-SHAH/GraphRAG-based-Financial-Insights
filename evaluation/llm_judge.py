"""
LLM-as-Judge evaluator using local Ollama (qwen2.5:3b).

Sends question + ground truth + retrieved context + generated answer
to the LLM and asks it to rate on multiple dimensions.

If Ollama is unavailable, returns None scores.

Usage:
    from evaluation.llm_judge import LLMJudge
    judge = LLMJudge()
    result = judge.evaluate(question, ground_truth, context, answer)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


JUDGE_PROMPT_TEMPLATE = """You are an evaluation judge for a Graph RAG system answering questions about LTIMindtree Limited (LTIM).

You must evaluate the generated answer against the ground truth and retrieved context.

CRITICAL INSTRUCTIONS:
- Judge ONLY against the supplied ground truth and retrieved context.
- Do NOT use outside knowledge.
- If the ground truth says the answer is unavailable, the system should say so.
- Be strict about hallucination: any claim not supported by context is a failure.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Return ONLY valid JSON with these fields (scores 0-5, where 5 is best):
{{
    "correctness": <0-5>,
    "relevance": <0-5>,
    "faithfulness": <0-5>,
    "completeness": <0-5>,
    "unsupported_claims": <0-5>,
    "reason": "<brief explanation>"
}}

Rules for scoring:
- correctness: Does the answer match the ground truth facts?
- relevance: Does the answer address the specific question asked?
- faithfulness: Are all claims in the answer supported by the retrieved context?
- completeness: Does the answer cover all parts of the question?
- unsupported_claims: 5 = no unsupported claims, 0 = many unsupported claims

Return ONLY the JSON, nothing else."""


class LLMJudge:
    """
    Uses Ollama qwen2.5:3b as an LLM judge to evaluate
    answer quality across multiple dimensions.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        self.model = model
        self.ollama_url = ollama_url

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama with timeout."""
        import requests

        try:
            resp = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception:
            return None

    def _parse_json_response(
        self, response: str
    ) -> Optional[Dict[str, Any]]:
        """Extract JSON from the LLM response."""
        json_match = re.search(
            r"\{[^{}]*\"correctness\"[^{}]*\}",
            response,
            re.DOTALL,
        )
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def evaluate(
        self,
        question: str,
        ground_truth: str,
        context: str,
        answer: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single answer using the LLM judge.

        Returns structured JSON with scores 0-5 for each dimension,
        or None if Ollama is unavailable.
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            context=context[:3000],
            answer=answer[:1500],
        )

        response = self._call_ollama(prompt)

        if response is None:
            return None

        parsed = self._parse_json_response(response)

        if parsed is None:
            return None

        result = {
            "correctness": parsed.get("correctness", 0),
            "relevance": parsed.get("relevance", 0),
            "faithfulness": parsed.get("faithfulness", 0),
            "completeness": parsed.get("completeness", 0),
            "unsupported_claims": parsed.get("unsupported_claims", 0),
            "reason": parsed.get("reason", ""),
        }

        for key in [
            "correctness", "relevance", "faithfulness",
            "completeness", "unsupported_claims",
        ]:
            val = result[key]
            if isinstance(val, (int, float)):
                result[key] = max(0, min(5, val))
            else:
                result[key] = 0

        return result

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        import requests

        try:
            resp = requests.get(
                self.ollama_url.replace(
                    "/api/generate", "/api/tags"
                ),
                timeout=5,
            )
            resp.raise_for_status()
            return True
        except Exception:
            return False


def judge_format_context(
    graph_result: Dict[str, Any],
) -> str:
    """Build a context string from graph result for the judge."""
    parts = []

    facts = graph_result.get("financial_facts", [])
    if facts:
        parts.append("FINANCIAL FACTS:")
        for f in facts:
            parts.append(
                f"- {f.get('metric')} = {f.get('value')} "
                f"{f.get('unit')} ({f.get('period')})"
            )

    entities = graph_result.get("entities", [])
    if entities:
        parts.append("ENTITIES:")
        for ent in entities:
            e = ent.get("entity", {})
            if isinstance(e, dict) and e.get("entity"):
                parts.append(f"- {e['entity']}")
                for r in e.get("outgoing", []):
                    parts.append(
                        f"  {e['entity']} --[{r.get('relationship')}]--> {r.get('target')}"
                    )

    semantic_paths = graph_result.get("semantic_paths", [])
    if semantic_paths:
        parts.append("GRAPH PATHS:")
        for sp in semantic_paths:
            nodes = sp.get("nodes", sp.get("path", []))
            rels = sp.get("relationships", [])
            chain = " -> ".join(
                f"{n} [{r}]"
                for n, r in zip(nodes, rels + [""])
            )
            parts.append(f"- {chain}")

    return "\n".join(parts) if parts else "No context retrieved."
