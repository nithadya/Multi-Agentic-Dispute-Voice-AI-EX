"""
Tiny enqueue helper used by the API to push jobs onto the Arq queue.

Pattern:

    from workers.enqueue import enqueue_chat_bookkeeping
    await enqueue_chat_bookkeeping(user_id, session_id, user_msg, assistant_msg)

Failure mode is graceful: if Redis is unreachable, we log a warning and
fall back to the synchronous BackgroundTasks path the API already uses.
That keeps local development (no Redis container) working and avoids
silent loss of memory if the queue is down in production.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from loguru import logger


@lru_cache(maxsize=1)
def _redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://redis:6379/0")


@lru_cache(maxsize=1)
def _worker_enabled() -> bool:
    """
    Whether to actually enqueue jobs. Off by default until the worker
    is provably running — flipping this to True is the cutover from
    inline FastAPI BackgroundTasks to the Arq queue.

    Set ARQ_WORKER_ENABLED=true in the production env once the worker
    container is up; leave unset/false in local dev to keep the
    Week-14 behaviour.
    """
    return os.getenv("ARQ_WORKER_ENABLED", "false").lower() in ("true", "1", "yes")


_pool = None  # module-level connection pool, lazy-initialised


async def _get_pool():
    """Lazily build the Arq Redis pool. One pool per API process."""
    global _pool
    if _pool is None:
        from arq import create_pool
        from arq.connections import RedisSettings
        from urllib.parse import urlparse
        parsed = urlparse(_redis_url())
        _pool = await create_pool(RedisSettings(
            host=parsed.hostname or "redis",
            port=parsed.port or 6379,
            database=int((parsed.path or "/0").lstrip("/") or "0"),
            password=parsed.password,
        ))
    return _pool


async def enqueue_chat_bookkeeping(
    *,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> bool:
    """
    Enqueue the three post-turn jobs:

        1. save_chat_turn       — persist user + assistant turn
        2. auto_title_session   — maybe LLM-rename the session
        3. distill_facts        — maybe extract LT facts

    Returns True if all three jobs were enqueued, False otherwise (caller
    should fall back to the inline FastAPI BackgroundTasks path).
    """
    if not _worker_enabled():
        return False

    try:
        pool = await _get_pool()
        await pool.enqueue_job(
            "save_chat_turn",
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )
        await pool.enqueue_job(
            "auto_title_session",
            user_id=user_id,
            session_id=session_id,
        )
        await pool.enqueue_job(
            "distill_facts",
            user_id=user_id,
            session_id=session_id,
        )
        return True
    except Exception as e:
        logger.warning(
            f"Arq enqueue failed (falling back to inline BackgroundTasks): {e}"
        )
        return False
