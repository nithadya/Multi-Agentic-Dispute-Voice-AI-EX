"""
Dependency-injection helpers.

All heavyweight objects (LLM clients, Qdrant/Supabase sessions, the agent
orchestrator, the CAG cache, the web crawler) are built once during the
FastAPI lifespan and stashed on ``app.state``. Routers import these
``get_*`` helpers to fetch them per-request.

Every getter raises HTTP 503 ``Service Unavailable`` if the startup hook
has not completed yet — this keeps handlers simple (no None checks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request


if TYPE_CHECKING:
    from agents.orchestrator import AgentOrchestrator


def _state_attr(request: Request, name: str) -> Any:
    value = getattr(request.app.state, name, None)
    if value is None:
        raise HTTPException(
            status_code=503,
            detail=f"API not ready — {name} is still initialising",
        )
    return value


def get_agent(request: Request) -> "AgentOrchestrator":
    """The LangGraph-backed agent orchestrator."""
    return _state_attr(request, "agent")


def get_crm_tool(request: Request):
    agent = get_agent(request)
    if agent.crm_tool is None:
        raise HTTPException(status_code=503, detail="CRM tool not configured")
    return agent.crm_tool


def get_rag_tool(request: Request):
    agent = get_agent(request)
    if agent.rag_tool is None:
        raise HTTPException(status_code=503, detail="RAG tool not configured")
    return agent.rag_tool


def get_web_tool(request: Request):
    agent = get_agent(request)
    if agent.web_tool is None:
        raise HTTPException(status_code=503, detail="Web search tool not configured")
    return agent.web_tool


def get_cag_cache(request: Request):
    """CAGCache instance — exposed for direct REST access + chat short-circuit."""
    return _state_attr(request, "cag_cache")


def get_st_store(request: Request):
    return get_agent(request).st_store


def get_lt_store(request: Request):
    return get_agent(request).lt_store


def get_recaller(request: Request):
    return get_agent(request).recaller


def get_distiller(request: Request):
    return get_agent(request).distiller


def get_embedder(request: Request):
    return _state_attr(request, "embedder")
