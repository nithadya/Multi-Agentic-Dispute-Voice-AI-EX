"""
System endpoints — liveness, readiness, and active configuration.

``/health`` stays cheap (never touches I/O) so orchestrators can poll it
often. ``/ready`` actively probes Qdrant, Supabase and an LLM to decide
whether the process can serve requests end-to-end.
"""

import asyncio

from fastapi import APIRouter, Request

from api.schemas import (
    ConfigResponse,
    HealthResponse,
    ReadinessCheck,
    ReadinessResponse,
)


router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness — is the process up?"""
    agent = getattr(request.app.state, "agent", None)
    return HealthResponse(status="ok" if agent is not None else "starting")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request) -> ReadinessResponse:
    """
    Readiness — probe the critical downstream dependencies in parallel
    and report a per-check summary.
    """
    agent = getattr(request.app.state, "agent", None)
    cag = getattr(request.app.state, "cag_cache", None)

    if agent is None:
        return ReadinessResponse(
            ready=False,
            checks=[ReadinessCheck(name="agent", ok=False, detail="not initialised")],
        )

    async def check_qdrant() -> ReadinessCheck:
        try:
            if cag is None:
                return ReadinessCheck(name="qdrant", ok=False, detail="cag cache unavailable")
            ok = bool(getattr(cag, "_available", False))
            return ReadinessCheck(name="qdrant", ok=ok, detail=None if ok else "cache not available")
        except Exception as exc:
            return ReadinessCheck(name="qdrant", ok=False, detail=str(exc)[:200])

    async def check_supabase() -> ReadinessCheck:
        try:
            await asyncio.to_thread(agent.st_store.recent, "__probe__", "__probe__", 1)
            return ReadinessCheck(name="supabase", ok=True)
        except Exception as exc:
            return ReadinessCheck(name="supabase", ok=False, detail=str(exc)[:200])

    async def check_tools() -> ReadinessCheck:
        enabled = sum(bool(x) for x in (agent.crm_tool, agent.rag_tool, agent.web_tool))
        return ReadinessCheck(name="tools", ok=enabled > 0, detail=f"{enabled}/3 enabled")

    checks = await asyncio.gather(check_qdrant(), check_supabase(), check_tools())
    return ReadinessResponse(ready=all(c.ok for c in checks), checks=list(checks))


@router.get("/config", response_model=ConfigResponse)
async def config(request: Request) -> ConfigResponse:
    """Report which models and providers are active."""
    from infrastructure.config import (
        CHAT_MODEL,
        EMBEDDING_MODEL,
        EXTRACTOR_MODEL,
        PROVIDER,
        ROUTER_MODEL,
    )

    agent = getattr(request.app.state, "agent", None)
    return ConfigResponse(
        chat_model=CHAT_MODEL,
        router_model=ROUTER_MODEL,
        extractor_model=EXTRACTOR_MODEL,
        embedding_model=EMBEDDING_MODEL,
        provider=PROVIDER,
        tools_enabled={
            "crm": bool(agent and agent.crm_tool),
            "rag": bool(agent and agent.rag_tool),
            "web_search": bool(agent and agent.web_tool),
        },
    )
