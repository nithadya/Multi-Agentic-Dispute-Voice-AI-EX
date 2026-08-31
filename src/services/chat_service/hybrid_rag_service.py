"""
Hybrid RAG Service — Fuses Graph RAG (Neo4j) + Vector CRAG (Qdrant).

This is the core retrieval engine for the e-commerce dispute resolution system.
It addresses the "RAG Bottleneck" identified in the design by combining:

  1. Graph RAG (Neo4j): Multi-hop Cypher traversal of vendor policy rules
     → handles conditional logic chains ("If X then Y unless Z")

  2. Vector CRAG (Qdrant): Semantic similarity search over policy documents
     → handles policy text semantics and edge cases not in the graph

Fusion strategy:
  - Graph results ranked HIGHER for structured policy questions
  - Vector results fill gaps when graph returns low/no results
  - Combined context passed to LLM for final decision generation
  - CAG cache still applies — hybrid result is cached for future reuse
"""

import asyncio
import time
from typing import Any, Dict, Optional

from loguru import logger

from infrastructure.config import GRAPH_RAG_ENABLED
from infrastructure.observability import observe


class HybridRAGService:
    """
    Fuses Neo4j Graph RAG + Qdrant Vector CRAG for policy-aware retrieval.

    Used by RAGTool.search() when GRAPH_RAG_ENABLED=True and Neo4j is available.
    Falls back gracefully to pure vector CRAG when Neo4j is unavailable.
    """

    def __init__(self, graph_rag: Any, vector_rag: Any, llm: Optional[Any] = None) -> None:
        """
        Args:
            graph_rag: GraphRAGService instance (Neo4j policy traversal)
            vector_rag: CRAGService instance (Qdrant semantic search)
            llm: Optional LLM for hybrid synthesis (falls back to graph answer if None)
        """
        self.graph_rag = graph_rag
        self.vector_rag = vector_rag
        self.llm = llm

    @observe(name="hybrid_rag_search")
    def search(self, query: str, vendor_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run both Graph RAG and Vector CRAG, then fuse the results.

        Returns:
            answer: Final fused answer
            graph_context: Context from Neo4j (empty string if unavailable)
            vector_context: Context from Qdrant
            retrieval_source: "graph" | "vector" | "hybrid"
            latency_ms: Total retrieval time
        """
        start = time.time()
        graph_result = {}
        vector_answer = ""

        # ── Concurrent Retrieval (Graph + Vector) ─────────────────────
        import concurrent.futures

        def _run_graph():
            if not GRAPH_RAG_ENABLED:
                return {}
            try:
                res = self.graph_rag.query(query, verbose=False, vendor_id=vendor_id)
                logger.debug(
                    "Graph RAG: {} retrieval attempts, context_len={}",
                    res.get("retrieval_attempts", 0),
                    len(res.get("policy_context", "")),
                )
                return res
            except Exception as exc:
                logger.warning("Graph RAG failed (falling back to vector): {}", exc)
                return {}

        def _run_vector():
            try:
                v_query = f"[Vendor: {vendor_id}] {query}" if vendor_id else query
                return self.vector_rag.generate(v_query, verbose=False)
            except Exception as exc:
                logger.warning("Vector CRAG failed: {}", exc)
                return {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_graph = executor.submit(_run_graph)
            future_vector = executor.submit(_run_vector)
            graph_result = future_graph.result()
            vec_result = future_vector.result()

        vector_answer = vec_result.get("answer", "")
        # ── Fusion logic ──────────────────────────────────────────────
        graph_context = graph_result.get("policy_context", "")
        graph_answer = graph_result.get("answer", "")

        elapsed_ms = int((time.time() - start) * 1000)

        if graph_context and vector_answer:
            # Both succeeded → fuse
            retrieval_source = "hybrid"
            answer = self._fuse(query, graph_answer, vector_answer, graph_context)
        elif graph_context:
            # Only graph succeeded
            retrieval_source = "graph"
            answer = graph_answer
        elif vector_answer:
            # Only vector succeeded
            retrieval_source = "vector"
            answer = vector_answer
        else:
            retrieval_source = "none"
            answer = (
                "I could not find a specific policy for this dispute in either the "
                "knowledge graph or the policy document database. "
                "Please escalate to a human dispute resolution agent."
            )

        logger.info(
            "HybridRAG: source={}, graph_ctx_len={}, latency={}ms",
            retrieval_source,
            len(graph_context),
            elapsed_ms,
        )

        return {
            "answer": answer,
            "graph_context": graph_context,
            "vector_context": vector_answer,
            "retrieval_source": retrieval_source,
            "graph_entities": graph_result.get("entities", {}),
            "graph_retrieval_attempts": graph_result.get("retrieval_attempts", 0),
            "latency_ms": elapsed_ms,
        }

    def _fuse(
        self,
        query: str,
        graph_answer: str,
        vector_answer: str,
        graph_context: str,
    ) -> str:
        """
        Merge graph and vector answers.

        Strategy:
        - If an LLM is available: ask it to synthesize both sources
        - Otherwise: prepend the graph answer (more structured) before vector
        """
        if self.llm is None:
            # Simple concatenation with clear source attribution
            return (
                f"📊 **Policy Graph Decision** (Neo4j):\n{graph_answer}\n\n"
                f"📄 **Policy Document Context** (Semantic Search):\n{vector_answer}"
            )

        FUSION_PROMPT = (
            "You are an E-Commerce Dispute Resolution AI. Synthesize these two policy "
            "sources into ONE coherent, policy-grounded decision:\n\n"
            "GRAPH POLICY DECISION (structured rules — higher authority):\n{graph}\n\n"
            "VECTOR POLICY CONTEXT (semantic document search):\n{vector}\n\n"
            "DISPUTE QUERY:\n{query}\n\n"
            "Provide a single, clear decision (APPROVE/PARTIAL_REFUND/REJECT/ESCALATE) "
            "citing the most relevant policy from either source."
        )

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = ChatPromptTemplate.from_messages([
                ("human", FUSION_PROMPT),
            ])
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({
                "graph": graph_answer[:2000],
                "vector": vector_answer[:1500],
                "query": query,
            })
        except Exception as exc:
            logger.warning("Fusion LLM failed, using graph answer: {}", exc)
            return graph_answer or vector_answer


__all__ = ["HybridRAGService"]
