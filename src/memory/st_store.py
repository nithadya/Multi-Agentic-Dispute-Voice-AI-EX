"""
Short-term memory store — Supabase backend.

Implements a ring buffer with TTL for recent conversation turns.
Turns are stored in the ``st_turns`` table in Supabase PostgreSQL.
"""

import json
from loguru import logger
import time
from typing import List, Optional
from datetime import datetime, timedelta
from memory.schemas import ConversationTurn
from sqlalchemy import text


class ShortTermMemoryStore:
    """
    Short-term memory store backed by Supabase PostgreSQL.

    Stores the most recent *N* conversation turns per session,
    automatically trimming older entries and expiring via TTL.
    """

    def __init__(self, supabase_session_factory=None):
        if not supabase_session_factory:
            from infrastructure.db.supabase_client import get_supabase_session
            supabase_session_factory = get_supabase_session
        self.supabase_session_factory = supabase_session_factory
        logger.debug("Using Supabase backend for short-term memory")

    def add(self, user_id: str, session_id: str, turn: ConversationTurn) -> None:
        """Add a conversation turn (alias for append with default config)."""
        from infrastructure.config import ST_MAX_TURNS, ST_TTL_SECONDS

        if not hasattr(turn, 'user_id') or not turn.user_id:
            turn.user_id = user_id
        if not hasattr(turn, 'session_id') or not turn.session_id:
            turn.session_id = session_id

        self.append(turn, max_turns=ST_MAX_TURNS, ttl_seconds=ST_TTL_SECONDS)

    def append(self, turn: ConversationTurn, max_turns: int, ttl_seconds: int) -> None:
        """Append a conversation turn to short-term memory."""
        session = self.supabase_session_factory()
        try:
            ttl_at = datetime.now() + timedelta(seconds=ttl_seconds)
            session.execute(
                text("""
                    INSERT INTO st_turns (user_id, session_id, role, content, ttl_at)
                    VALUES (:user_id, :session_id, :role, :content, :ttl_at)
                """),
                {
                    "user_id": turn.user_id,
                    "session_id": turn.session_id,
                    "role": turn.role,
                    "content": turn.content,
                    "ttl_at": ttl_at,
                }
            )
            session.commit()
            # Trim to max_turns (ring buffer)
            session.execute(
                text("""
                    DELETE FROM st_turns
                    WHERE id IN (
                        SELECT id FROM st_turns
                        WHERE user_id = :user_id AND session_id = :session_id
                        ORDER BY created_at DESC
                        OFFSET :max_turns
                    )
                """),
                {
                    "user_id": turn.user_id,
                    "session_id": turn.session_id,
                    "max_turns": max_turns,
                }
            )
            session.commit()
            logger.debug("Appended turn to ST: {}/{}", turn.user_id, turn.session_id)
        except Exception as e:
            session.rollback()
            logger.error("Failed to append turn to Supabase: {}", e)
            raise
        finally:
            session.close()

    def recent(self, user_id: str, session_id: str, k: int) -> List[ConversationTurn]:
        """Retrieve the most recent *k* conversation turns for a session."""
        session = self.supabase_session_factory()
        try:
            results = session.execute(
                text("""
                    SELECT role, content, created_at
                    FROM st_turns
                    WHERE user_id = :user_id 
                        AND session_id = :session_id
                        AND (ttl_at IS NULL OR ttl_at > NOW())
                    ORDER BY created_at DESC
                    LIMIT :k
                """),
                {"user_id": user_id, "session_id": session_id, "k": k}
            ).fetchall()

            turns = []
            for row in reversed(results):
                turn = ConversationTurn(
                    role=row.role,
                    content=row.content,
                    user_id=user_id,
                    session_id=session_id,
                    ts=row.created_at.timestamp() if hasattr(row.created_at, 'timestamp') else time.time()
                )
                turns.append(turn)
            return turns
        except Exception as e:
            logger.error("Failed to retrieve turns from Supabase: {}", e)
            return []
        finally:
            session.close()

    def history(self, user_id: str, session_id: str, k: int) -> List[ConversationTurn]:
        """Retrieve the most recent *k* turns for a session, for DISPLAY.
        Unlike :meth:`recent` — which powers the orchestrator's active
        short-term context and therefore hides turns past their TTL — this
        returns the stored turns **regardless of ``ttl_at``**. The sidebar
        must be able to reopen a conversation and show its history days or
        weeks later, as long as the rows still exist in ``st_turns``.
        """
        session = self.supabase_session_factory()
        try:
            results = session.execute(
                text("""
                    SELECT role, content, created_at
                    FROM st_turns
                    WHERE user_id = :user_id
                        AND session_id = :session_id
                    ORDER BY created_at DESC
                    LIMIT :k
                """),
                {"user_id": user_id, "session_id": session_id, "k": k}
            ).fetchall()

            turns = []
            for row in reversed(results):
                turns.append(ConversationTurn(
                    role=row.role,
                    content=row.content,
                    user_id=user_id,
                    session_id=session_id,
                    ts=row.created_at.timestamp() if hasattr(row.created_at, 'timestamp') else time.time(),
                ))
            return turns
        except Exception as e:
            logger.error("Failed to retrieve session history from Supabase: {}", e)
            return []
        finally:
            session.close()

    def clear(self, user_id: str, session_id: str) -> None:
        """Clear all turns for a session."""
        session = self.supabase_session_factory()
        try:
            session.execute(
                text("""
                    DELETE FROM st_turns
                    WHERE user_id = :user_id AND session_id = :session_id
                """),
                {"user_id": user_id, "session_id": session_id}
            )
            session.commit()
            logger.info("Cleared ST for: {}/{}", user_id, session_id)
        except Exception as e:
            session.rollback()
            logger.error("Failed to clear ST: {}", e)
            raise
        finally:
            session.close()
