"""Web search tool — direct REST access to the Tavily-backed WebSearchTool."""

import asyncio
import time

from fastapi import APIRouter, Depends

from api.deps import get_web_tool
from api.schemas import WebSearchRequest, WebSearchResponse


router = APIRouter(prefix="/tools/web_search", tags=["Tools — Web"])


@router.post("", response_model=WebSearchResponse)
async def web_search(req: WebSearchRequest, web=Depends(get_web_tool)) -> WebSearchResponse:
    t0 = time.perf_counter()
    result = await asyncio.to_thread(web.search, query=req.query, max_results=req.max_results)
    return WebSearchResponse(result=result, latency_ms=int((time.perf_counter() - t0) * 1000))
