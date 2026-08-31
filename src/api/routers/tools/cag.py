"""
CAG (Cache-Augmented Generation) — REST access to the semantic cache.

Same four operations that ``cag_server.py`` exposes over MCP: ``get``,
``set``, ``stats``, ``clear``. ``clear`` is destructive and untouched
behind any auth in v1 — guard it at the ingress if that matters.
"""

import asyncio

from fastapi import APIRouter, Depends

from api.deps import get_cag_cache
from api.schemas import (
    CAGClearResponse,
    CAGGetRequest,
    CAGGetResponse,
    CAGSetRequest,
    CAGSetResponse,
    CAGStatsResponse,
)


router = APIRouter(prefix="/tools/cag", tags=["Tools — CAG"])


@router.post("/get", response_model=CAGGetResponse)
async def cag_get(req: CAGGetRequest, cag=Depends(get_cag_cache)) -> CAGGetResponse:
    hit = await asyncio.to_thread(cag.get, req.query)
    if hit is None:
        return CAGGetResponse(hit=False)
    return CAGGetResponse(
        hit=True,
        query=hit.get("query", ""),
        answer=hit.get("answer", ""),
        evidence_urls=hit.get("evidence_urls", []) or [],
        score=float(hit.get("score", 0.0)),
        ts=float(hit.get("ts", 0.0)),
    )


@router.post("/set", response_model=CAGSetResponse)
async def cag_set(req: CAGSetRequest, cag=Depends(get_cag_cache)) -> CAGSetResponse:
    await asyncio.to_thread(
        cag.set,
        req.query,
        {"answer": req.answer, "evidence_urls": req.evidence_urls},
    )
    return CAGSetResponse(cached=True, query=req.query)


@router.get("/stats", response_model=CAGStatsResponse)
async def cag_stats(cag=Depends(get_cag_cache)) -> CAGStatsResponse:
    stats = await asyncio.to_thread(cag.stats)
    return CAGStatsResponse(stats=stats)


@router.post("/clear", response_model=CAGClearResponse)
async def cag_clear(cag=Depends(get_cag_cache)) -> CAGClearResponse:
    await asyncio.to_thread(cag.clear)
    return CAGClearResponse(cleared=True)
