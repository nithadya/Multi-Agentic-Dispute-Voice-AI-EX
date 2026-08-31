"""
Cache-Augmented Generation (CAG) — semantic vector cache backed by Qdrant.

Architecture:
    Uses a dedicated Qdrant collection (``cag_cache``) separate from the
    RAG knowledge-base collection (``disputeai``).  Each cached entry is a
    Qdrant point whose vector is the query embedding.

How it works:
    1. Every cached entry is a Qdrant point containing:
       - **vector**         : embedded query (float32)
       - **payload.query**  : original question (text)
       - **payload.answer** : generated response (text)
       - **payload.evidence_urls** : JSON-encoded source list
       - **payload.ts**     : unix timestamp (for TTL filtering)
    2. ``get(query)`` embeds the query, runs KNN-1 search;
       if cosine similarity ≥ threshold → cache **HIT**.
    3. ``set(query, response)`` embeds the query, upserts point so
       future *semantically similar* queries hit the cache.

Why Qdrant over Redis Stack:
    - Standard Redis Cloud (free tier) does NOT include the RediSearch
      module required for FT.CREATE / FT.SEARCH vector search.
    - Qdrant Cloud is already provisioned for RAG — adding a second
      collection is zero-cost and requires no extra infrastructure.
    - Same sub-millisecond HNSW vector search, same cosine metric.

Why semantic > hash-based (MD5):
    - "What is the leave policy?"  →  "How many leave days do staff get?"
      Same meaning, different words.  MD5 would MISS both;
      semantic search catches the paraphrase with ~0.95 cosine similarity.

Dependencies:
    - Qdrant Cloud (already configured for RAG)
    - An embedder with ``embed_query(text) → List[float]``
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger


# ── Defaults ─────────────────────────────────────────────────

_DEFAULT_COLLECTION = "cag_cache"


class CAGCache:
    """
    Qdrant-backed semantic cache for pre-computed RAG responses.

    Each point stores:
        vector   → query embedding (float32)
        payload  → query, answer, evidence_urls, ts

    Lookup is a KNN-1 search; a hit is declared when
    ``cosine_similarity ≥ similarity_threshold``.

    Usage::

        cache = CAGCache(embedder=embedder)

        # Semantic lookup
        cached = cache.get("How many leave days?")
        if cached:
            return cached["answer"]

        # Store new entry (semantically indexed)
        cache.set("What is the leave policy?", {"answer": "...", "evidence_urls": [...]})
    """

    def __init__(
        self,
        embedder: Any,
        collection_name: Optional[str] = None,
        dim: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Initialise the semantic CAG cache.

        Args:
            embedder: Object with ``embed_query(text) -> List[float]``.
            collection_name: Qdrant collection for CAG (default: ``cag_cache``).
            dim: Embedding dimension (auto-detected from config if None).
            similarity_threshold: Min cosine similarity for a hit (0.90–0.95).
            ttl_seconds: Entries older than this are ignored; 0 → no expiry.
        """
        from infrastructure.config import (
            CAG_SIMILARITY_THRESHOLD,
            CAG_CACHE_TTL,
            EMBEDDING_DIM,
            CAG_COLLECTION_NAME,
        )

        self.embedder = embedder
        self.collection_name = collection_name or CAG_COLLECTION_NAME
        self.dim = dim or EMBEDDING_DIM
        self.similarity_threshold = similarity_threshold or CAG_SIMILARITY_THRESHOLD
        self.ttl_seconds = ttl_seconds or CAG_CACHE_TTL

        # Ensure the Qdrant collection exists
        self._available = False
        try:
            from infrastructure.db.qdrant_client import (
                get_qdrant_client,
                collection_exists,
            )

            self._client = get_qdrant_client()
            if not collection_exists(self.collection_name):
                self._create_collection()
            self._available = True
            logger.info(
                "✓ CAG cache ready (Qdrant collection='{}', dim={}, threshold={:.2f})",
                self.collection_name,
                self.dim,
                self.similarity_threshold,
            )
        except Exception as exc:
            logger.warning(
                "CAG cache DISABLED — Qdrant unavailable: {}. "
                "All lookups will miss; every query runs full RAG.",
                exc,
            )

    # ── collection management ─────────────────────────────────

    def _create_collection(self) -> None:
        """Create the Qdrant collection for CAG cache."""
        from qdrant_client.http.models import Distance, VectorParams

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE,
                on_disk=False,  # keep in-memory for speed
            ),
        )
        logger.info(
            "Created CAG cache collection '{}' (dim={}, COSINE)",
            self.collection_name,
            self.dim,
        )

    # ── public API ────────────────────────────────────────────

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Semantic cache lookup via KNN-1 search.

        Embeds *query*, searches the Qdrant HNSW index, and returns
        the cached response if ``cosine_similarity ≥ threshold``.

        Args:
            query: Natural-language question.

        Returns:
            Dict with ``query``, ``answer``, ``evidence_urls``, ``ts``,
            ``score`` — or ``None`` on miss.
        """
        if not self._available:
            return None

        # Embed query
        try:
            query_vec = self.embedder.embed_query(query)
        except Exception as exc:
            logger.warning("CAG embed failed: {}", exc)
            return None

        # KNN-1 search
        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=1,
                score_threshold=self.similarity_threshold,
            )
        except Exception as exc:
            logger.warning("CAG cache GET error: {}", exc)
            return None

        if not response.points:
            return None

        hit = response.points[0]
        similarity = hit.score
        payload = hit.payload or {}

        # TTL filtering: skip entries older than ttl_seconds
        if self.ttl_seconds and self.ttl_seconds > 0:
            entry_ts = payload.get("ts", 0)
            if entry_ts and (time.time() - float(entry_ts)) > self.ttl_seconds:
                logger.debug("CAG cache expired: '{}' → deleting point {}", query[:50], hit.id)
                try:
                    self._client.delete(
                        collection_name=self.collection_name,
                        points_selector=[hit.id],
                    )
                except Exception as exc:
                    logger.warning("Failed to delete expired CAG entry: {}", exc)
                return None

        cached_query = payload.get("query", "")
        logger.info(
            "CAG cache HIT (sim={:.3f}): '{}' → matched '{}'",
            similarity,
            query[:50],
            cached_query[:50],
        )

        evidence_raw = payload.get("evidence_urls", "[]")
        try:
            evidence_urls = json.loads(evidence_raw) if isinstance(evidence_raw, str) else evidence_raw
        except (json.JSONDecodeError, TypeError):
            evidence_urls = []

        return {
            "query": cached_query,
            "answer": payload.get("answer", ""),
            "evidence_urls": evidence_urls,
            "ts": float(payload.get("ts", 0)),
            "score": similarity,
        }

    def set(self, query: str, response: Dict[str, Any]) -> None:
        """
        Cache a response, indexed by the query's embedding.

        Args:
            query: Original user query.
            response: Dict with ``answer`` and optionally ``evidence_urls``.
        """
        if not self._available:
            return

        # Embed query
        try:
            query_vec = self.embedder.embed_query(query)
        except Exception as exc:
            logger.warning("CAG embed failed on SET: {}", exc)
            return

        from qdrant_client.http.models import PointStruct

        # First, search and destroy any existing entries for this exact same semantic query
        # This prevents the cache from filling up with duplicates over time when TTL expires
        try:
            existing = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=10, # Check for a few in case duplicates already exist
                score_threshold=0.99, # practically identical queries
            )
            if existing.points:
                existing_ids = [p.id for p in existing.points]
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=existing_ids,
                )
                logger.debug("CAG cache replaced {} existing duplicates for '{}'", len(existing_ids), query[:50])
        except Exception as exc:
            logger.warning("CAG cache failed to clean existing duplicates: {}", exc)

        point_id = str(uuid.uuid4())
        payload = {
            "query": query,
            "answer": response.get("answer", ""),
            "evidence_urls": json.dumps(response.get("evidence_urls", [])),
            "ts": time.time(),
        }

        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=query_vec, payload=payload)],
            )
            logger.debug("CAG cache SET: '{}' → point={}", query[:60], point_id)
        except Exception as exc:
            logger.warning("CAG cache SET error: {}", exc)

    def clear(self) -> None:
        """
        Drop and recreate the CAG cache collection.

        All cached entries are removed. The collection is recreated
        immediately so the cache is ready for new entries.
        """
        if not self._available:
            return

        try:
            self._client.delete_collection(self.collection_name)
            logger.info("Dropped CAG cache collection '{}'", self.collection_name)
        except Exception:
            pass  # Collection may not exist

        self._create_collection()
        logger.info("CAG cache cleared and collection recreated")

    def stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dict with ``total_cached``, ``backend``, ``collection``,
            ``similarity_threshold``, ``ttl_seconds``.
        """
        return {
            "total_cached": self._count(),
            "backend": "qdrant",
            "collection": self.collection_name,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds,
            "available": self._available,
        }

    # ── internal helpers ──────────────────────────────────────

    def _count(self) -> int:
        """Return number of cached entries."""
        if not self._available:
            return 0
        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    # ── dunder helpers ────────────────────────────────────────

    def __len__(self) -> int:
        return self._count()

    def __contains__(self, query: str) -> bool:
        return self.get(query) is not None

    def __repr__(self) -> str:
        return (
            f"CAGCache(collection='{self.collection_name}', "
            f"threshold={self.similarity_threshold}, "
            f"ttl={self.ttl_seconds}s, "
            f"entries={self._count()}, "
            f"backend='qdrant')"
        )
