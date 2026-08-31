"""
Chat-session metadata endpoints — power the ChatGPT-style sidebar.

Each row is a conversation thread for a customer. Short-term turns
themselves still live in ``st_turns`` and are fetched via
``GET /sessions/{sid}/turns`` (chat router) — this module only manages
the parent metadata: title, last activity, archived flag.

  GET    /chat_sessions?user_id=…           list non-archived sessions
  POST   /chat_sessions                      create a new session row
  PATCH  /chat_sessions/{session_id}         rename / archive
  DELETE /chat_sessions/{session_id}         hard-delete + cascade ST turns

The chat hot path also calls ``touch_session()`` from this module after
a successful reply, to keep ``last_message_at`` fresh.
"""

import asyncio
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from sqlalchemy import desc, text
from sqlalchemy.orm import sessionmaker

from infrastructure.db import get_sql_engine
from infrastructure.db.crm_models import ChatSession

from api.schemas import (
    ChatSessionCreateRequest,
    ChatSessionListResponse,
    ChatSessionMeta,
    ChatSessionUpdateRequest,
)


router = APIRouter(prefix="/chat_sessions", tags=["Chat sessions"])


# ── helpers ─────────────────────────────────────────────────────────

def _session_db():
    return sessionmaker(bind=get_sql_engine(), autoflush=False, expire_on_commit=False)()


def _to_meta(row: ChatSession) -> ChatSessionMeta:
    return ChatSessionMeta(
        session_id=row.session_id,
        customer_id=row.customer_id,
        title=row.title,
        last_message_at=row.last_message_at,
        created_at=int(row.created_at or 0),
        updated_at=int(row.updated_at or 0),
        archived=int(row.archived or 0),
    )


def _gen_session_id() -> str:
    return f"s_{uuid.uuid4().hex[:10]}"


def _default_title() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        from infrastructure.config import TIMEZONE
    except Exception:
        TIMEZONE = "Asia/Colombo"
    now = datetime.now(ZoneInfo(TIMEZONE))
    return f"Conversation {now.strftime('%Y-%m-%d %H:%M')}"


def _is_default_title(title: str) -> bool:
    """True if the title still looks auto-generated ("Conversation 2026-…")."""
    if not title:
        return True
    return title.startswith("Conversation ") and any(c.isdigit() for c in title)


def maybe_auto_title_sync(
    *,
    session_id: str,
    user_id: str,
    st_store,
    llm,
    min_turns: int = 4,
) -> None:
    """
    Generate a short LLM title from the conversation if the session's
    title is still the auto-generated default and there's enough content
    to summarise. Idempotent and silent on failure — never raises.

    Designed to be called from the background save path after every
    turn. The default-title check ensures we only fire once per session.

    Parameters
    ----------
    session_id : str
    user_id    : str
    st_store   : the orchestrator's short-term store (has .recent())
    llm        : a LangChain Chat LLM (we use llm_fast — Groq llama-3.1-8b)
    min_turns  : minimum ST turns required before titling (default 4 ≈ 2 exchanges)
    """
    s = _session_db()
    try:
        row = s.get(ChatSession, session_id)
        if row is None or not _is_default_title(row.title):
            return  # session unknown OR already user/LLM-titled

        try:
            recent = st_store.recent(user_id, session_id, k=8)
        except Exception:
            return
        if len(recent) < min_turns:
            return

        # Build a small snippet. Strip our internal [interrupted] tag so
        # it doesn't leak into the title.
        snippet_lines = []
        for t in recent:
            content = (t.content or "").replace("[interrupted]", "").strip()
            if not content:
                continue
            snippet_lines.append(f"{t.role}: {content[:200]}")
        snippet = "\n".join(snippet_lines)
        if not snippet:
            return

        from langchain_core.messages import SystemMessage, HumanMessage
        sys_msg = SystemMessage(content=(
            "You write very short chat-window titles. Read the snippet "
            "and reply with a 3-to-6-word title that captures the topic. "
            "No quotes. No punctuation at the end. Title case."
        ))
        user_msg = HumanMessage(content=f"Conversation snippet:\n\n{snippet}")

        try:
            resp = llm.invoke([sys_msg, user_msg])
            title = (getattr(resp, "content", None) or "").strip().strip('"').strip("'").strip()
            # Some models append a period — strip it.
            if title.endswith("."):
                title = title[:-1].strip()
        except Exception as e:
            logger.debug(f"auto-title LLM call failed: {e}")
            return

        if not title or len(title) > 80:
            return

        row.title = title
        row.updated_at = int(time.time())
        s.commit()
        logger.info(f"auto-titled session {session_id} → {title!r}")
    except Exception as exc:
        logger.debug(f"maybe_auto_title_sync failed for {session_id}: {exc}")
        s.rollback()
    finally:
        s.close()


def touch_session_sync(customer_id: str, session_id: str) -> None:
    """
    Ensure a chat_sessions row exists for (customer_id, session_id) and
    bump its ``last_message_at`` to now.

    Called from the chat hot path on every successful reply. Idempotent
    — if the session was created by the UI via ``POST /chat_sessions``,
    we just update the timestamp; if a chat request arrived for an
    unknown session_id (e.g. legacy localStorage one), we materialise
    a row so the sidebar can pick it up.
    """
    s = _session_db()
    try:
        now = int(time.time())
        row = s.get(ChatSession, session_id)
        if row is None:
            row = ChatSession(
                session_id=session_id,
                customer_id=customer_id,
                title=_default_title(),
                last_message_at=now,
                created_at=now,
                updated_at=now,
                archived=0,
            )
            s.add(row)
        else:
            row.last_message_at = now
            row.updated_at = now
        s.commit()
    except Exception as exc:
        logger.warning("touch_session failed for {}/{}: {}", customer_id, session_id, exc)
        s.rollback()
    finally:
        s.close()


# ── endpoints ───────────────────────────────────────────────────────

@router.get("", response_model=ChatSessionListResponse)
async def list_sessions(
    user_id: str = Query(..., min_length=1, description="customer_id"),
    include_archived: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
) -> ChatSessionListResponse:
    """List a customer's sessions, newest activity first."""

    def _query() -> list[ChatSession]:
        s = _session_db()
        try:
            q = s.query(ChatSession).filter(ChatSession.customer_id == user_id)
            if not include_archived:
                q = q.filter(ChatSession.archived == 0)
            q = q.order_by(
                desc(ChatSession.last_message_at),
                desc(ChatSession.created_at),
            ).limit(limit)
            return list(q.all())
        finally:
            s.close()

    rows = await asyncio.to_thread(_query)
    return ChatSessionListResponse(sessions=[_to_meta(r) for r in rows])


@router.post("", response_model=ChatSessionMeta, status_code=201)
async def create_session(req: ChatSessionCreateRequest) -> ChatSessionMeta:
    """Create a new session row. ``session_id`` is auto-generated if not supplied."""

    def _insert() -> ChatSession:
        sid = req.session_id or _gen_session_id()
        s = _session_db()
        try:
            existing = s.get(ChatSession, sid)
            if existing is not None:
                # Treat as upsert: same id, same customer → return existing
                if existing.customer_id != req.user_id:
                    raise ValueError(f"session_id {sid} already belongs to another customer")
                return existing
            now = int(time.time())
            row = ChatSession(
                session_id=sid,
                customer_id=req.user_id,
                title=req.title or _default_title(),
                last_message_at=None,
                created_at=now,
                updated_at=now,
                archived=0,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            return row
        finally:
            s.close()

    try:
        row = await asyncio.to_thread(_insert)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _to_meta(row)


@router.patch("/{session_id}", response_model=ChatSessionMeta)
async def update_session(session_id: str, req: ChatSessionUpdateRequest) -> ChatSessionMeta:
    def _update() -> Optional[ChatSession]:
        s = _session_db()
        try:
            row = s.get(ChatSession, session_id)
            if row is None:
                return None
            if req.title is not None:
                row.title = req.title.strip() or row.title
            if req.archived is not None:
                row.archived = int(bool(req.archived))
            row.updated_at = int(time.time())
            s.commit()
            s.refresh(row)
            return row
        finally:
            s.close()

    row = await asyncio.to_thread(_update)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _to_meta(row)


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    """Hard-delete a session and its short-term turns."""

    def _delete() -> bool:
        s = _session_db()
        try:
            row = s.get(ChatSession, session_id)
            if row is None:
                return False
            # Cascade: drop ST turns for this session too
            s.execute(
                text("DELETE FROM st_turns WHERE session_id = :sid"),
                {"sid": session_id},
            )
            s.delete(row)
            s.commit()
            return True
        finally:
            s.close()

    ok = await asyncio.to_thread(_delete)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")

    # Drop any in-process warm cache for this session
    cache = getattr(request.app.state, "session_cache", None)
    if cache is not None:
        for key in list(cache.keys()):
            if key[1] == session_id:
                cache.pop(key, None)

    return {"deleted": True, "session_id": session_id}
