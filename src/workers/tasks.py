"""
Arq background tasks — the *non-blocking* post-turn bookkeeping for chat.

Each task is fire-and-forget from the API's perspective. The API enqueues
a job and returns immediately; this worker process pulls jobs off Redis,
executes them, and Arq handles retries + dead-letter on persistent failure.

Tasks defined here:

- `save_chat_turn`     — write user + assistant turn into short-term store
- `auto_title_session` — if title still default + enough turns, LLM-rename
- `distill_facts`      — every Nth turn, extract LT memory facts via LLM

Run with:

    arq src.workers.tasks.WorkerSettings

Or via the Makefile target:

    make worker

The API side queues jobs via `enqueue_chat_bookkeeping(...)` in
`src/workers/enqueue.py` — single small wrapper that knows the
Redis URL and the function names.
"""

from __future__ import annotations

import os
import time
from typing import Any

from arq.connections import RedisSettings
from loguru import logger

from memory.schemas import ConversationTurn


# ── Redis connection settings ─────────────────────────────────────────

def _redis_settings() -> RedisSettings:
    """
    Build the Arq Redis connection settings from REDIS_URL.

    Accepts either a full URL (`redis://host:6379/0`) — the common form
    in compose / cloud — or falls back to host/port for legacy setups.
    """
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    # Arq's RedisSettings doesn't take a URL directly; parse it ourselves.
    # Minimal parse — full URL handling lives in `redis-py` and we just
    # forward to it via env when the API enqueues.
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "redis",
        port=parsed.port or 6379,
        database=int((parsed.path or "/0").lstrip("/") or "0"),
        password=parsed.password,
    )


# ── Orchestrator lifecycle (shared across tasks in this worker) ───────

async def _startup(ctx: dict) -> None:
    """
    Build the orchestrator once when the worker boots, reuse across jobs.

    This mirrors how the API does it via FastAPI lifespan — building the
    orchestrator costs ~5–10 s (Supabase + Qdrant + MCP wiring) and we
    don't want to pay that per job.
    """
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))

    from infrastructure.log import setup_logging
    setup_logging()

    logger.info("Arq worker boot — building agent orchestrator...")
    from agents.orchestrator import build_agent
    ctx["orchestrator"] = build_agent()
    logger.success("Arq worker ready — orchestrator + tools wired")


async def _shutdown(ctx: dict) -> None:
    """Drain any final tasks + close connections."""
    logger.info("Arq worker shutting down")


# ── Tasks ─────────────────────────────────────────────────────────────

async def save_chat_turn(
    ctx: dict,
    *,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> dict:
    """
    Persist one chat turn (user + assistant) into the short-term store.

    Idempotent at the row level — duplicate enqueues will write duplicate
    turns; the API side is responsible for not double-enqueueing the same
    turn. (Arq retries on failure call this function again, so the cost
    of a duplicate row is preferred over losing the turn.)
    """
    orchestrator = ctx["orchestrator"]
    now = time.time()
    orchestrator.st_store.add(
        user_id, session_id,
        ConversationTurn(
            user_id=user_id, session_id=session_id,
            role="user", content=user_message, ts=now,
        ),
    )
    orchestrator.st_store.add(
        user_id, session_id,
        ConversationTurn(
            user_id=user_id, session_id=session_id,
            role="assistant", content=assistant_message, ts=now,
        ),
    )
    logger.debug(f"save_chat_turn: {session_id} (user_id={user_id})")
    return {"status": "ok", "session_id": session_id}


async def auto_title_session(
    ctx: dict,
    *,
    user_id: str,
    session_id: str,
) -> dict:
    """
    Maybe LLM-rename a session if it still has the default title.

    Delegates to `maybe_auto_title_sync` — the exact same helper the
    API uses for inline scheduling. Wrapping it in a worker task means:

    - The LLM call (Groq, ~80–200 ms) doesn't block the API response.
    - Failures get retried automatically by Arq.
    - Cost is metered on the worker, not the API request budget.
    """
    orchestrator = ctx["orchestrator"]
    llm = getattr(orchestrator, "llm_fast", None) or orchestrator.llm_chat

    # Lazy import — avoids pulling FastAPI into the worker process boot
    # unless we actually need it.
    from api.routers.chat_sessions import maybe_auto_title_sync
    maybe_auto_title_sync(
        session_id=session_id,
        user_id=user_id,
        st_store=orchestrator.st_store,
        llm=llm,
    )
    return {"status": "ok"}


async def distill_facts(
    ctx: dict,
    *,
    user_id: str,
    session_id: str,
) -> dict:
    """
    Run long-term distillation on recent turns if the heuristic says so.

    Safe to enqueue after every turn — the distiller's `should_distill`
    check gates the actual LLM call so most enqueues are cheap no-ops.
    """
    orchestrator = ctx["orchestrator"]
    try:
        recent = orchestrator.st_store.recent(user_id, session_id, k=5)
        if orchestrator.distiller.should_distill(recent):
            logger.info(f"distill_facts: triggering LT distillation for {user_id}")
            orchestrator.distiller.distill(user_id, recent)
    except Exception as e:
        logger.warning(f"distill_facts failed (non-fatal): {e}")
    return {"status": "ok"}


# ── Arq worker config ─────────────────────────────────────────────────

class WorkerSettings:
    """
    Arq picks this up via `arq src.workers.tasks.WorkerSettings`.

    `functions` is the registry of callable tasks. `redis_settings` tells
    Arq where to find the broker. `on_startup`/`on_shutdown` build and
    tear down shared resources (the orchestrator).
    """

    functions = [save_chat_turn, auto_title_session, distill_facts]
    redis_settings = _redis_settings()
    on_startup = _startup
    on_shutdown = _shutdown

    # Generous defaults — these workloads are I/O bound (Supabase + Groq)
    # and benefit from concurrent execution.
    max_jobs = 10
    job_timeout = 60                # any single task > 60 s is a bug
    keep_result = 60 * 5            # 5-min result history for debugging
    health_check_interval = 30
