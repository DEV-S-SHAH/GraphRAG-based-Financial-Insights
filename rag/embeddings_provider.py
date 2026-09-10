"""
Unified Embeddings Abstraction Layer.

Supports:
1. Local Ollama embeddings (nomic-embed-text: 768 dimensions)
2. OpenAI embeddings (text-embedding-3-small: 1536 or reduced)
3. HuggingFace / SentenceTransformers (local fallback)
"""

from abc import ABC, abstractmethod
import os
from typing import List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class EmbeddingsProvider(ABC):
    """Abstract Embeddings Provider interface."""

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class OllamaEmbeddings(EmbeddingsProvider):
    """Local Ollama embeddings using nomic-embed-text (768-dim)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: int = 768,
    ):
        model = model_name or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        super().__init__(model_name=model, dimension=dimension)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        all_embeddings = []
        url = f"{self.base_url}/api/embed"

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = requests.post(
                url,
                json={"model": self.model_name, "input": batch},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            all_embeddings.extend(data["embeddings"])

        return all_embeddings


class OpenAIEmbeddings(EmbeddingsProvider):
    """OpenAI API embeddings."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: int = 768,
    ):
        model = model_name or "text-embedding-3-small"
        super().__init__(model_name=model, dimension=dimension)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not configured.")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def embed_texts(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.dimension,
            )
            batch_emb = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_emb)
        return all_embeddings


def get_embeddings_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> EmbeddingsProvider:
    """Factory function for embeddings."""
    chosen = (provider or os.getenv("EMBEDDING_PROVIDER", "ollama")).lower()

    if chosen == "openai":
        try:
            return OpenAIEmbeddings(model_name=model)
        except Exception:
            return OllamaEmbeddings(model_name=model)

    return OllamaEmbeddings(model_name=model)
