"""
CAG (Cache-Augmented Generation) service combining caching with CRAG.

Pipeline:
    Query --> Semantic Cache (Qdrant cag_cache KNN-1)
          --> HIT? Return instantly (0ms, $0)
          --> MISS? --> CRAGService (self-correcting retrieval)
                    --> Cache the result for future hits
                    --> Return answer
"""

from loguru import logger
from typing import Any, Dict, List
import time

from services.chat_service.cag_cache import CAGCache
from services.chat_service.crag_service import CRAGService
from infrastructure.observability import observe, update_current_observation


class CAGService:
    """
    Cache-Augmented Generation backed by Corrective RAG.

    Layer 1: Semantic cache (Qdrant cag_cache) -- instant, $0
    Layer 2: CRAG (confidence-gated retrieval) -- self-correcting
    """

    def __init__(self, crag_service: CRAGService, cache: CAGCache):
        self.crag_service = crag_service
        self.cache = cache

    @observe(name="cag_generate")
    def generate(
        self,
        query: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate answer with CAG + CRAG pipeline.

        1. Check semantic cache (cosine >= 0.90 = HIT)
        2. On miss: run CRAGService (confidence-gated retrieval)
        3. Cache the result for future semantic hits
        """
        start = time.time()

        if use_cache:
            cached = self.cache.get(query)
            if cached:
                latency_ms = int((time.time() - start) * 1000)
                logger.info(
                    "CAG cache HIT (score={:.3f}) for: {}",
                    cached.get("score", 0),
                    query[:60],
                )
                update_current_observation(
                    input=query,
                    output=cached["answer"][:500],
                    metadata={
                        "cache_hit": True,
                        "cache_score": cached.get("score", 0),
                        "latency_ms": latency_ms,
                    },
                )
                return {
                    "answer": cached["answer"],
                    "evidence_urls": cached.get("evidence_urls", []),
                    "cache_hit": True,
                    "cache_score": cached.get("score", 0),
                    "generation_time": 0.0,
                }

        # Cache miss -- run CRAG (self-correcting retrieval)
        crag_result = self.crag_service.generate(query, verbose=False)

        answer = crag_result.get("answer", "")
        evidence_urls = crag_result.get("evidence_urls", [])

        result: Dict[str, Any] = {
            "answer": answer,
            "evidence_urls": evidence_urls,
            "cache_hit": False,
            "confidence_initial": crag_result.get("confidence_initial", 0),
            "confidence_final": crag_result.get("confidence_final", 0),
            "correction_applied": crag_result.get("correction_applied", False),
            "generation_time": crag_result.get("generation_time", 0),
            "num_docs": crag_result.get("docs_used", 0),
        }

        # Confidence gate: CRAG's ``confidence_final`` is a 0-1 score
        # over the retrieval grounding. When it's below the CRAG
        # confidence threshold, the synthesised answer is built from
        # weak evidence — exactly the path that produces apologetic
        # "I don't know" prose. Caching it would poison the CAG.
        # Trust the score, not string matching.
        from infrastructure.config import CRAG_CONFIDENCE_THRESHOLD
        _confident = result.get("confidence_final", 0) >= CRAG_CONFIDENCE_THRESHOLD

        if use_cache and answer and _confident:
            # Strip the leading "Hello {name}," prefix (if any) before
            # caching — cached answers are shared across users, so
            # personalisation in the synth output would leak user A's
            # name into user B's reply on a cache hit.
            import re as _re
            _greet_re = _re.compile(
                r"^\s*(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))"
                r"[ ,\-]+[A-Z][a-zA-Z']*[ ,\-]*\n?",
                _re.IGNORECASE,
            )
            cached_answer = _greet_re.sub("", answer, count=1).lstrip()
            self.cache.set(query, {"answer": cached_answer, "evidence_urls": evidence_urls})
            logger.info("CAG cache MISS -> cached for: {}", query[:60])
        elif use_cache and answer:
            logger.debug(
                "CAG cache: not caching low-confidence answer (conf={:.2f}) for: {}",
                result.get("confidence_final", 0), query[:60],
            )

        latency_ms = int((time.time() - start) * 1000)
        update_current_observation(
            input=query,
            output=answer[:500] if answer else "No results",
            metadata={
                "cache_hit": False,
                "latency_ms": latency_ms,
                "correction_applied": result["correction_applied"],
                "confidence_final": result["confidence_final"],
            },
        )

        return result

    def warm_cache(self, queries: List[Any]) -> int:
        """Pre-populate cache with common queries via CRAG pipeline or hardcoded answers."""
        cached_count = 0
        for item in queries:
            if isinstance(item, dict) and "query" in item and "answer" in item:
                query = item["query"]
                if query not in self.cache:
                    self.cache.set(query, {"answer": item["answer"], "evidence_urls": []})
                    cached_count += 1
            elif isinstance(item, str):
                query = item
                if query not in self.cache:
                    self.generate(query, use_cache=True)
                    cached_count += 1
        return cached_count

    def cache_stats(self) -> Dict[str, Any]:
        return self.cache.stats()

    def clear_cache(self) -> None:
        self.cache.clear()


__all__ = ["CAGService"]
