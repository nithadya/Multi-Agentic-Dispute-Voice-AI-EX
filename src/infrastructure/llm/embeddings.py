"""
Embedding providers.

Two flavours, picked by call site:

  ``get_default_embeddings()`` — OpenAI ``text-embedding-3-*`` (1536 dims).
      Used for the long-term memory store and the RAG knowledge-base
      index, where semantic quality across a large corpus matters more
      than per-query latency.

  ``get_local_embedder()`` — sentence-transformers ``all-MiniLM-L6-v2``
      (384 dims). Used **only** for the CAG cache short-circuit on the
      chat hot path: zero network latency, ~30 ms per query vs. ~1 s
      for the OpenAI round-trip from Sri Lanka. The model is loaded
      once and re-used for the lifetime of the process.
"""

from threading import Lock
from typing import Any, List, Optional

from langchain_openai import OpenAIEmbeddings
from loguru import logger

from infrastructure.config import EMBEDDING_MODEL, PROVIDER, OPENROUTER_BASE_URL, get_api_key


# ── OpenAI (default) ────────────────────────────────────────────────

def get_default_embeddings(
    batch_size: int = 100,
    show_progress: bool = False,
    **kwargs: Any,
) -> OpenAIEmbeddings:
    """OpenAI embeddings — 1536/3072-dim, network call to api.openai.com."""
    llm_kwargs: dict[str, Any] = dict(
        model=EMBEDDING_MODEL,
        show_progress_bar=show_progress,
        **kwargs,
    )

    api_key = get_api_key(PROVIDER)
    if api_key:
        llm_kwargs["openai_api_key"] = api_key
        
    if PROVIDER == "openrouter":
        llm_kwargs["openai_api_base"] = OPENROUTER_BASE_URL

    return OpenAIEmbeddings(**llm_kwargs)


# ── Local (CAG cache only) ──────────────────────────────────────────

LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBED_DIM = 384

_local_singleton: Optional["LocalEmbedder"] = None
_local_lock = Lock()


class LocalEmbedder:
    """
    Lightweight wrapper that gives a ``sentence-transformers`` model the
    minimum interface our code expects from an embedder:

        - ``embed_query(text: str) -> list[float]``
        - ``embed_documents(texts: list[str]) -> list[list[float]]``

    These two methods cover both the LangChain ``Embeddings`` protocol
    used by the RAG layer and the ad-hoc usage by ``CAGCache``.
    """

    def __init__(self, model_name: str = LOCAL_EMBED_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model '{}' (cold load is one-time)…", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = LOCAL_EMBED_DIM
        logger.info("✓ Local embedder ready (dim={})", self.dim)

    def embed_query(self, text: str) -> List[float]:
        # normalize_embeddings=True so cosine similarity behaves
        # identically to the OpenAI vectors we used previously.
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()


def get_local_embedder() -> LocalEmbedder:
    """Return the process-wide singleton local embedder, lazily loaded."""
    global _local_singleton
    if _local_singleton is None:
        with _local_lock:
            if _local_singleton is None:
                _local_singleton = LocalEmbedder()
    return _local_singleton
