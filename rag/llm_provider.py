"""
Unified LLM Abstraction Layer.

Supports:
1. Local LLM via Ollama (qwen2.5:3b - fits within < 4GB VRAM/RAM)
2. OpenAI (when OPENAI_API_KEY is configured)
3. Anthropic (when ANTHROPIC_API_KEY is configured)

Configured entirely through .env with automatic fallback.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    latency_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


class LLMProvider(ABC):
    """Abstract LLM Provider interface."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> LLMResponse:
        pass


class OllamaProvider(LLMProvider):
    """Local Ollama provider (e.g. qwen2.5:3b, <= 4 GB)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        model = model_name or os.getenv("LLM_MODEL", "qwen2.5:3b")
        super().__init__(model_name=model)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> LLMResponse:
        start_time = time.time()
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            latency = time.time() - start_time
            content = data.get("response", "").strip()

            prompt_tokens = data.get("prompt_eval_count", len(prompt) // 4)
            completion_tokens = data.get("eval_count", len(content) // 4)

            return LLMResponse(
                content=content,
                model=self.model_name,
                provider="ollama",
                latency_seconds=round(latency, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=0.0,  # Local execution is free
            )
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise ConnectionError(f"Ollama LLM call failed: {e}")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        model = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        super().__init__(model_name=model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> LLMResponse:
        start_time = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        latency = time.time() - start_time
        choice = resp.choices[0]
        content = choice.message.content or ""

        p_tok = resp.usage.prompt_tokens if resp.usage else len(prompt) // 4
        c_tok = resp.usage.completion_tokens if resp.usage else len(content) // 4
        
        # Estimate cost (e.g. gpt-4o-mini is ~$0.15/1M input, $0.60/1M output)
        cost = (p_tok * 0.00000015) + (c_tok * 0.00000060)

        return LLMResponse(
            content=content.strip(),
            model=self.model_name,
            provider="openai",
            latency_seconds=round(latency, 2),
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            estimated_cost_usd=round(cost, 6),
        )


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        model = model_name or os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022")
        super().__init__(model_name=model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        from anthropic import Anthropic
        self.client = Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> LLMResponse:
        start_time = time.time()
        resp = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - start_time
        content = resp.content[0].text if resp.content else ""
        p_tok = resp.usage.input_tokens
        c_tok = resp.usage.output_tokens
        
        # Claude-3.5 Haiku is ~$0.80/1M input, $4.00/1M output
        cost = (p_tok * 0.00000080) + (c_tok * 0.00000400)

        return LLMResponse(
            content=content.strip(),
            model=self.model_name,
            provider="anthropic",
            latency_seconds=round(latency, 2),
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            estimated_cost_usd=round(cost, 6),
        )


def get_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMProvider:
    """
    Factory function returning the configured LLM provider.
    Priority: explicit arg -> .env LLM_PROVIDER -> API key presence -> Ollama fallback.
    """
    chosen = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()

    if chosen == "openai":
        try:
            return OpenAIProvider(model_name=model)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider ({e}). Falling back to Ollama.")
            return OllamaProvider(model_name=model)

    elif chosen == "anthropic":
        try:
            return AnthropicProvider(model_name=model)
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider ({e}). Falling back to Ollama.")
            return OllamaProvider(model_name=model)

    return OllamaProvider(model_name=model)
