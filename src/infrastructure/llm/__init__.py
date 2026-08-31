"""
LLM provider wrappers — 3-model architecture.

  get_router_llm()     → gpt-4o-mini (routing)
  get_extractor_llm()  → llama-3.1-8b-instant via Groq (extraction)
  get_chat_llm()       → gemini-2.0-flash (synthesis)
  get_default_embeddings() → text-embedding-3-small (embeddings)
"""

from .llm_provider import (
    get_chat_llm,
    get_fast_chat_llm,
    get_router_llm,
    get_extractor_llm,
)
from .embeddings import get_default_embeddings

__all__ = [
    "get_chat_llm",
    "get_fast_chat_llm",
    "get_router_llm",
    "get_extractor_llm",
    "get_default_embeddings",
]
