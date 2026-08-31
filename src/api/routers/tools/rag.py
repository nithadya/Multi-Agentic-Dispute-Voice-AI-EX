"""
RAG tool — direct REST access to ``src/agents/tools/rag_tool.py``.

RAG retrieval is sync (Qdrant KNN + CRAG LLM generation), so every call
is dispatched onto the async thread pool. The underlying ``RAGTool``
manages its own CAG cache + CRAG retriever — these endpoints don't
bypass those layers.
"""

import asyncio
import time

from fastapi import APIRouter, Depends

from api.deps import get_rag_tool
from api.schemas import RAGResponse, RAGSearchRequest, RAGStatsResponse


router = APIRouter(prefix="/tools/rag", tags=["Tools — RAG"])


@router.post("/search", response_model=RAGResponse)
async def search(req: RAGSearchRequest, rag=Depends(get_rag_tool)) -> RAGResponse:
    t0 = time.perf_counter()
    result = await asyncio.to_thread(
        rag.search,
        query=req.query,
        top_k=req.top_k,
        threshold=req.threshold,
        use_cache=req.use_cache,
    )
    return RAGResponse(result=result, latency_ms=int((time.perf_counter() - t0) * 1000))


@router.get("/stats", response_model=RAGStatsResponse)
async def stats(rag=Depends(get_rag_tool)) -> RAGStatsResponse:
    cache = getattr(rag, "_cache", None)
    if cache is None:
        return RAGStatsResponse(stats={"available": False})
    data = await asyncio.to_thread(cache.stats)
    return RAGStatsResponse(stats=data)
