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

    def __init__(self, model_name: str, provider_name: str = "custom"):
        self.model_name = model_name
        self.provider_name = provider_name

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> LLMResponse:
        pass


def _is_remote_model(name: Optional[str]) -> bool:
    if not name:
        return False
    low = name.lower()
    return any(low.startswith(p) for p in ["gpt-", "claude-", "o1-", "o3-", "text-davinci"])


def _get_default_ollama_model(requested_model: Optional[str] = None) -> str:
    if requested_model and not _is_remote_model(requested_model):
        return requested_model
    env_model = os.getenv("LLM_MODEL", "qwen2.5:3b")
    if _is_remote_model(env_model):
        return "qwen2.5:3b"
    return env_model


def _retry_call(fn, max_retries: int = 3, initial_delay: float = 1.0):
    """Execute a function with exponential backoff on failure."""
    delay = initial_delay
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
    raise last_err


class OllamaProvider(LLMProvider):
    """Local Ollama provider (e.g. qwen2.5:3b, <= 4 GB)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        model = _get_default_ollama_model(model_name)
        super().__init__(model_name=model, provider_name="ollama")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

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

        def _make_request():
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()

        try:
            data = _retry_call(_make_request, max_retries=2, initial_delay=1.0)
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
            raise ConnectionError(
                f"Ollama LLM call failed on {self.base_url} (model '{self.model_name}'): {e}. "
                "Ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull qwen2.5:3b`)."
            )


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 60.0):
        model = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        if not _is_remote_model(model):
            model = "gpt-4o-mini"
        super().__init__(model_name=model, provider_name="openai")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key or not self.api_key.strip():
            raise ValueError("OPENAI_API_KEY is not configured or empty.")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> LLMResponse:
        start_time = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _call_api():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )

        resp = _retry_call(_call_api, max_retries=3, initial_delay=1.0)
        latency = time.time() - start_time
        choice = resp.choices[0]
        content = choice.message.content or ""

        p_tok = resp.usage.prompt_tokens if resp.usage else len(prompt) // 4
        c_tok = resp.usage.completion_tokens if resp.usage else len(content) // 4

        # Dynamic pricing estimation
        if "gpt-4o-mini" in self.model_name:
            cost = (p_tok * 0.00000015) + (c_tok * 0.00000060)
        elif "gpt-4o" in self.model_name:
            cost = (p_tok * 0.00000250) + (c_tok * 0.00001000)
        else:
            cost = (p_tok * 0.00000050) + (c_tok * 0.00000150)

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

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 60.0):
        model = model_name or os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022")
        if not _is_remote_model(model):
            model = "claude-3-5-haiku-20241022"
        super().__init__(model_name=model, provider_name="anthropic")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key or not self.api_key.strip():
            raise ValueError("ANTHROPIC_API_KEY is not configured or empty.")
        from anthropic import Anthropic
        self.client = Anthropic(api_key=self.api_key, timeout=timeout)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> LLMResponse:
        start_time = time.time()

        def _call_api():
            return self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )

        resp = _retry_call(_call_api, max_retries=3, initial_delay=1.0)
        latency = time.time() - start_time
        content = resp.content[0].text if resp.content else ""
        p_tok = resp.usage.input_tokens
        c_tok = resp.usage.output_tokens

        # Dynamic pricing estimation
        if "sonnet" in self.model_name:
            cost = (p_tok * 0.00000300) + (c_tok * 0.00001500)
        else:
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
            return OllamaProvider(model_name=_get_default_ollama_model(model))

    elif chosen == "anthropic":
        try:
            return AnthropicProvider(model_name=model)
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider ({e}). Falling back to Ollama.")
            return OllamaProvider(model_name=_get_default_ollama_model(model))

    return OllamaProvider(model_name=_get_default_ollama_model(model))
