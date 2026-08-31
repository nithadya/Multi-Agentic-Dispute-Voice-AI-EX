"""
RAG Tool -- Hybrid policy retrieval via Graph RAG (Neo4j) + Vector CRAG (Qdrant).

Architecture (updated for E-Commerce Dispute Resolution):
    Query --> CAGService (semantic cache)
              --> Qdrant cag_cache (KNN-1)
              --> HIT? Return instantly
              --> MISS? --> HybridRAGService
                           --> GraphRAGService (Neo4j Cypher multi-hop)
                           --> CRAGService (Qdrant vector CRAG)
                           --> Fusion (graph ranked higher)
              --> Cache fused result

Fallback: if Neo4j is not configured (GRAPH_RAG_ENABLED=False or no NEO4J_URI),
falls back to pure vector CRAG (original behaviour).
"""

from loguru import logger
import time
from typing import Any, Dict, List, Optional

from infrastructure.config import (
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    CRAG_CONFIDENCE_THRESHOLD,
    CRAG_EXPANDED_K,
    GRAPH_RAG_ENABLED,
)
from infrastructure.observability import observe, update_current_observation


class RAGTool:
    """
    Internal-KB retrieval tool backed by CAGService (cache) + CRAGService (CRAG).

    The tool is a thin wrapper that:
    1. Builds the service stack (CAGCache -> CRAGService -> CAGService)
    2. Delegates all retrieval to CAGService.generate()
    3. Handles warm-up, stats, and dispatch routing
    """

    def __init__(
        self,
        embedder: Any,
        llm: Optional[Any] = None,
    ) -> None:
        self.embedder = embedder
        self.llm = llm

        from services.chat_service.cag_cache import CAGCache
        from services.chat_service.rag_service import QdrantRetriever, RAGService
        from services.chat_service.crag_service import CRAGService
        from services.chat_service.cag_service import CAGService

        self._cache = CAGCache(embedder=embedder)
        self._cag_service: Optional[CAGService] = None
        self._hybrid_rag = None  # HybridRAGService (init below if graph enabled)

        if llm is not None:
            retriever = QdrantRetriever(
                embedder=embedder,
                top_k=TOP_K_RESULTS,
                score_threshold=SIMILARITY_THRESHOLD,
            )

            crag_service = CRAGService(
                retriever=retriever,
                llm=llm,
                initial_k=TOP_K_RESULTS,
                expanded_k=CRAG_EXPANDED_K,
            )

            self._cag_service = CAGService(
                crag_service=crag_service,
                cache=self._cache,
            )

            # ── Graph RAG init (conditional) ────────────────────────────
            # Only initialise if NEO4J_URI is configured. Falls back to
            # pure vector CRAG if graph is unavailable — zero disruption.
            if GRAPH_RAG_ENABLED:
                try:
                    from infrastructure.db.neo4j_client import neo4j_available
                    if neo4j_available():
                        from services.chat_service.graph_rag_service import GraphRAGService
                        from services.chat_service.hybrid_rag_service import HybridRAGService
                        graph_rag = GraphRAGService(llm=llm)
                        self._hybrid_rag = HybridRAGService(
                            graph_rag=graph_rag,
                            vector_rag=crag_service,
                            llm=llm,
                        )
                        logger.success("HybridRAGService initialised: Neo4j Graph + Qdrant Vector")
                    else:
                        logger.info("Graph RAG disabled: NEO4J_URI not set. Using Vector CRAG only.")
                except Exception as exc:
                    logger.warning("Graph RAG init failed (using vector CRAG only): {}", exc)

            logger.info(
                "RAGTool initialised: CAG cache -> {} (k={}, expanded_k={}, threshold={:.2f})",
                "HybridRAG" if self._hybrid_rag else "CRAG",
                TOP_K_RESULTS,
                CRAG_EXPANDED_K,
                CRAG_CONFIDENCE_THRESHOLD,
            )
        else:
            logger.info("RAGTool initialised in raw-chunk mode (no LLM)")

    @observe(name="rag_search")
    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        threshold: float = SIMILARITY_THRESHOLD,
        use_cache: bool = True,
        vendor_id: Optional[str] = None,
    ) -> str:
        """
        Retrieve + generate an answer from policy knowledge base.

        Pipeline: CAG cache check -> HybridRAG (Graph+Vector) -> answer
        Falls back to pure Vector CRAG if graph is not configured.
        """
        # ── Hybrid Graph + Vector path (when Neo4j is available) ────
        if self._hybrid_rag is not None:
            try:
                result = self._hybrid_rag.search(query, vendor_id=vendor_id)
                answer = result.get("answer", "")
                source = result.get("retrieval_source", "hybrid")
                logger.debug("HybridRAG search: source={}", source)
                if answer:
                    return answer
            except Exception as exc:
                logger.warning("HybridRAG search failed, falling back to CRAG: {}", exc)

        # ── Pure Vector CRAG fallback ────────────────────────────────
        if self._cag_service is not None:
            try:
                # We append vendor context to the query if provided
                v_query = f"[Vendor: {vendor_id}] {query}" if vendor_id else query
                result = self._cag_service.generate(v_query, use_cache=use_cache)
                return result.get("answer", "") or "No relevant policy found in the knowledge base."
            except Exception as exc:
                logger.error("CRAG search failed: {}", exc)
                return "Policy retrieval error. Please try again."

        return "No relevant policy found. (RAG service not initialised)"

    def _raw_search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> str:
        """Fallback: direct Qdrant search returning formatted chunks (no LLM)."""
        from infrastructure.db.qdrant_client import search_chunks

        try:
            query_vec = self.embedder.embed_query(query)
        except Exception as exc:
            logger.error("Embedding query failed: {}", exc)
            return f"RAG embedding error: {exc}"

        try:
            results = search_chunks(
                query_vector=query_vec,
                top_k=top_k,
                score_threshold=threshold,
            )
        except Exception as exc:
            logger.error("Qdrant search failed: {}", exc)
            return f"RAG search error: {exc}"

        if not results:
            return ""

        seen_parents: set = set()
        lines: List[str] = [f"Internal KB results ({len(results)} chunks):"]

        for idx, hit in enumerate(results, 1):
            parent_id = hit.get("parent_id")
            if parent_id and parent_id in seen_parents:
                continue
            if parent_id:
                seen_parents.add(parent_id)

            sim = f"{hit['score']:.2f}"
            title = hit.get("title") or "Untitled"
            url = hit.get("url") or "N/A"
            text = hit.get("parent_text", hit["chunk_text"])
            lines.append(f"\n--- Chunk {idx} (similarity {sim}) ---")
            lines.append(f"Source: {title} ({url})")
            lines.append(text)

        return "\n".join(lines)

    def warm_cache(self, queries: List[str]) -> int:
        """Pre-populate CAG cache with common queries via CRAG pipeline."""
        if self._cag_service is None:
            return 0
        return self._cag_service.warm_cache(queries)

    def cache_stats(self) -> Dict[str, Any]:
        return self._cache.stats()

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("CAG cache cleared")

    def dispatch(self, action: str, params: Dict[str, Any]) -> str:
        """
        Dispatch a RAG action.

        Resilient to router slip-ups: filters kwargs the action doesn't
        accept (so an extra ``time_frame`` or similar doesn't crash) and
        gives a useful error string if the query is missing instead of
        500-erroring.
        """
        import inspect

        handlers = {
            "search": self.search,
            "cache_stats": lambda: f"CAG cache: {self.cache_stats()}",
            "clear_cache": lambda: (self.clear_cache(), "CAG cache cleared.")[1],
        }
        handler = handlers.get(action)
        if handler is None:
            return f"Unknown RAG action: {action}. Available: {list(handlers)}"

        if action == "search":
            sig = inspect.signature(self.search)
            accepted = {p.name for p in sig.parameters.values()}
            clean = {k: v for k, v in (params or {}).items() if k in accepted and v is not None}
            if not clean.get("query"):
                # The router didn't extract a query — return a clear message
                # the synth can turn into "what would you like me to search?"
                return (
                    "RAG search requires a query string but none was provided. "
                    "Please rephrase the question with a specific topic."
                )
            return self.search(**clean)

        return handler()
