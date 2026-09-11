"""
Unified Embeddings Abstraction Layer for Financial GraphRAG.

Supports:
1. Local Ollama embeddings:
   - bge-m3 (1024 dimensions - State-of-the-art dense retrieval model, Default)
   - mxbai-embed-large (1024 dimensions - Top MTEB benchmark model)
   - nomic-embed-text (768 dimensions - Lightweight model)
2. OpenAI embeddings (text-embedding-3-small / text-embedding-3-large with Matryoshka dimensions)
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
    """Local Ollama embeddings supporting bge-m3, mxbai-embed-large, and nomic-embed-text."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        model = model_name or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

        # Automatically determine embedding dimension based on model
        if dimension is not None:
            dim = dimension
        elif "bge-m3" in model or "mxbai" in model or "bge-large" in model:
            dim = 1024
        elif "nomic" in model:
            dim = 768
        else:
            dim = int(os.getenv("EMBEDDING_DIM", "768"))

        super().__init__(model_name=model, dimension=dim)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        all_embeddings = []
        url = f"{self.base_url}/api/embed"

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = requests.post(
                    url,
                    json={"model": self.model_name, "input": batch},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    all_embeddings.extend(data["embeddings"])
                    continue
            except Exception:
                pass

            # Fallback to single-item /api/embeddings for compatibility
            legacy_url = f"{self.base_url}/api/embeddings"
            for t in batch:
                resp = requests.post(
                    legacy_url,
                    json={"model": self.model_name, "prompt": t},
                    timeout=60,
                )
                resp.raise_for_status()
                all_embeddings.append(resp.json()["embedding"])

        # Validate dimension matches expected
        if all_embeddings and len(all_embeddings[0]) != self.dimension:
            self.dimension = len(all_embeddings[0])

        return all_embeddings


class OpenAIEmbeddings(EmbeddingsProvider):
    """OpenAI API embeddings (text-embedding-3-small/large with Matryoshka dimension)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        model = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        dim = dimension or int(os.getenv("EMBEDDING_DIM", "1024"))
        super().__init__(model_name=model, dimension=dim)
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
    dimension: Optional[int] = None,
) -> EmbeddingsProvider:
    """Factory function for embeddings provider."""
    chosen = (provider or os.getenv("EMBEDDING_PROVIDER", "ollama")).lower()

    if chosen == "openai":
        try:
            return OpenAIEmbeddings(model_name=model, dimension=dimension)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings ({e}). Falling back to Ollama.")
            ollama_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
            if ollama_model.startswith("text-embedding-"):
                ollama_model = "nomic-embed-text"
            return OllamaEmbeddings(model_name=ollama_model, dimension=dimension or 768)

    return OllamaEmbeddings(model_name=model, dimension=dimension)
