"""
Memory tools — REST access to the 4-tier memory system.

Endpoints mirror the MCP ``memory_server`` surface:

  POST /tools/memory/recall            hybrid ST + LT recall
  GET  /tools/memory/facts/{user_id}   list all LT facts
  POST /tools/memory/store_fact        manually add a LT fact
  POST /tools/memory/distill           force distillation on the latest turns
"""

import asyncio
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_distiller, get_lt_store, get_recaller, get_st_store
from api.schemas import (
    FactItem,
    MemoryDistillRequest,
    MemoryDistillResponse,
    MemoryFactsResponse,
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemoryStoreFactRequest,
    MemoryStoreFactResponse,
    TurnItem,
)


router = APIRouter(prefix="/tools/memory", tags=["Tools — Memory"])


def _fact_to_item(f) -> FactItem:
    return FactItem(
        id=getattr(f, "id", None),
        text=getattr(f, "text", "") or "",
        tags=getattr(f, "tags", []) or [],
        score=float(getattr(f, "score", 0.0)),
    )


@router.post("/recall", response_model=MemoryRecallResponse)
async def recall(
    req: MemoryRecallRequest,
    recaller=Depends(get_recaller),
) -> MemoryRecallResponse:
    """Hybrid recall: recent ST turns + top-K semantic LT facts for the query."""
    st_turns, lt_facts = await asyncio.to_thread(
        recaller.recall,
        user_id=req.user_id,
        session_id=req.session_id,
        query=req.query,
    )
    return MemoryRecallResponse(
        st_turns=[
            TurnItem(
                role=getattr(t, "role", "user"),
                content=getattr(t, "content", ""),
                ts=float(getattr(t, "ts", 0.0)),
            )
            for t in (st_turns or [])
        ],
        lt_facts=[_fact_to_item(f) for f in (lt_facts or [])],
    )


@router.get("/facts/{user_id}", response_model=MemoryFactsResponse)
async def list_facts(user_id: str, lt_store=Depends(get_lt_store)) -> MemoryFactsResponse:
    facts = await asyncio.to_thread(lt_store.get_all_facts, user_id)
    items = [_fact_to_item(f) for f in (facts or [])]
    return MemoryFactsResponse(user_id=user_id, fact_count=len(items), facts=items)


@router.post("/store_fact", response_model=MemoryStoreFactResponse)
async def store_fact(
    req: MemoryStoreFactRequest,
    lt_store=Depends(get_lt_store),
) -> MemoryStoreFactResponse:
    """Manually insert a LT fact — bypasses the LLM distiller."""
    from memory.schemas import MemoryFact

    now = time.time()
    fact = MemoryFact(
        id=str(uuid.uuid4()),
        user_id=req.user_id,
        text=req.text,
        score=0.5,
        tags=req.tags or [],
        created_at=now,
        last_used_at=now,
        ttl_at=None,
        pin=False,
    )
    await asyncio.to_thread(lt_store.upsert, [fact])
    return MemoryStoreFactResponse(stored=True, fact_id=fact.id)


@router.post("/distill", response_model=MemoryDistillResponse)
async def distill(
    req: MemoryDistillRequest,
    st_store=Depends(get_st_store),
    distiller=Depends(get_distiller),
) -> MemoryDistillResponse:
    """
    Force a distillation pass on the most recent turns.

    Returns a zero-count response if the policy says not to distill. Useful
    for testing and manual curation; the agent also auto-distills inside
    the LangGraph ``save_memory`` node.
    """
    recent = await asyncio.to_thread(st_store.recent, req.user_id, req.session_id, 10)
    if not recent:
        return MemoryDistillResponse(triggered=False, distilled_count=0)

    should = distiller.should_distill(recent)
    if not should:
        return MemoryDistillResponse(triggered=False, distilled_count=0)

    try:
        facts = await asyncio.to_thread(distiller.distill, req.user_id, recent)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Distill failed: {exc}")

    return MemoryDistillResponse(triggered=True, distilled_count=len(facts or []))
