"""
Memory MCP Server — exposes the 4-tier memory system over MCP.

Wraps:
  - ShortTermMemoryStore  (st_store.py)  — recent conversation turns
  - LongTermMemoryStore   (lt_store.py)  — pgvector semantic facts
  - MemoryRecaller        (memory_ops.py) — unified ST + LT recall

This is the "unique angle" server — most MCP tutorials only show stateless
tools. This one gives any MCP client (LangGraph, Claude Desktop, etc.)
persistent, semantic memory across sessions.

Run standalone:
    python -m mcp_servers.memory_server

Inspect:
    npx @modelcontextprotocol/inspector python -m mcp_servers.memory_server
"""

import os
import sys
import time
import uuid

# Ensure src/ is importable when run as a script
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from mcp.server.fastmcp import FastMCP

from memory.st_store import ShortTermMemoryStore
from memory.lt_store import LongTermMemoryStore
from memory.memory_ops import MemoryRecaller
from memory.schemas import ConversationTurn, MemoryFact
from infrastructure.llm import get_default_embeddings


# ── Server + lazy-initialised stores ────────────────────────────

mcp = FastMCP("disputeai-memory")

_st: ShortTermMemoryStore | None = None
_lt: LongTermMemoryStore | None = None
_recaller: MemoryRecaller | None = None


def _init() -> tuple[ShortTermMemoryStore, LongTermMemoryStore, MemoryRecaller]:
    global _st, _lt, _recaller
    if _st is None:
        logger.info("Initialising memory stores inside MCP server...")
        embedder = get_default_embeddings()
        _st = ShortTermMemoryStore()
        _lt = LongTermMemoryStore(embedder)
        _recaller = MemoryRecaller(_st, _lt)
    return _st, _lt, _recaller


# ── MCP Tools ───────────────────────────────────────────────────

@mcp.tool()
def recall_context(
    user_id: str,
    session_id: str,
    query: str,
    k_st: int = 6,
    k_lt: int = 5,
) -> str:
    """
    Unified recall: recent conversation turns + semantically relevant
    long-term facts for a given user and query.

    This is the primary memory tool — use it to load context before
    answering a question about a user.
    """
    _, _, recaller = _init()
    try:
        st_turns, lt_facts = recaller.recall(
            user_id=user_id,
            session_id=session_id,
            query=query,
            k_st=k_st,
            k_lt=k_lt,
        )
        out = recaller.format_context(st_turns)
        if lt_facts:
            out += "\n=== LONG-TERM FACTS ===\n"
            for f in lt_facts:
                out += f"- {f.text}\n"
        return out or "(no memory found)"
    except Exception as e:
        logger.error(f"recall_context failed: {e}")
        return f"Error recalling memory: {e}"


@mcp.tool()
def get_recent_turns(user_id: str, session_id: str, k: int = 10) -> str:
    """
    Fetch the last k conversation turns for a user's session from
    short-term memory.
    """
    st, _, _ = _init()
    try:
        turns = st.recent(user_id, session_id, k=k)
        if not turns:
            return "(no recent turns)"
        return "\n".join(f"{t.role}: {t.content}" for t in turns)
    except Exception as e:
        logger.error(f"get_recent_turns failed: {e}")
        return f"Error fetching turns: {e}"


@mcp.tool()
def add_turn(user_id: str, session_id: str, role: str, content: str) -> str:
    """
    Append a conversation turn to short-term memory.

    Args:
        user_id: User identifier
        session_id: Session identifier
        role: 'user' or 'assistant'
        content: Message text
    """
    st, _, _ = _init()
    try:
        turn = ConversationTurn(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
            ts=time.time(),
        )
        st.add(user_id, session_id, turn)
        return f"Stored {role} turn for {user_id}/{session_id}"
    except Exception as e:
        logger.error(f"add_turn failed: {e}")
        return f"Error adding turn: {e}"


@mcp.tool()
def search_facts(user_id: str, query: str, k: int = 5) -> str:
    """
    Semantic search over a user's long-term facts (pgvector).

    Returns the top-k most relevant facts for the query.
    """
    _, lt, _ = _init()
    try:
        from infrastructure.config import LT_SIM_THRESHOLD
        facts = lt.query(user_id=user_id, query_text=query, k=k, threshold=LT_SIM_THRESHOLD)
        if not facts:
            return "(no matching facts)"
        return "\n".join(f"- {f.text}" for f in facts)
    except Exception as e:
        logger.error(f"search_facts failed: {e}")
        return f"Error searching facts: {e}"


@mcp.tool()
def store_fact(user_id: str, text: str, tags: list[str] | None = None) -> str:
    """
    Store a new long-term semantic fact about a user.

    Use this when the user says things like "remember that..." or
    when a fact should persist across sessions (allergies, preferences,
    medications, etc.).
    """
    _, lt, _ = _init()
    try:
        now = time.time()
        fact = MemoryFact(
            id=str(uuid.uuid4()),
            user_id=user_id,
            text=text,
            score=1.0,
            tags=tags or [],
            created_at=now,
            last_used_at=now,
            ttl_at=None,
            pin=False,
        )
        lt.upsert([fact])
        return f"Stored fact for {user_id}: {text[:80]}"
    except Exception as e:
        logger.error(f"store_fact failed: {e}")
        return f"Error storing fact: {e}"


@mcp.tool()
def list_facts(user_id: str) -> str:
    """List all long-term facts stored for a user."""
    _, lt, _ = _init()
    try:
        facts = lt.get_all_facts(user_id)
        if not facts:
            return "(no facts stored)"
        return "\n".join(f"- [{f.id[:8]}] {f.text}" for f in facts)
    except Exception as e:
        logger.error(f"list_facts failed: {e}")
        return f"Error listing facts: {e}"


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting disputeai-memory MCP server on stdio...")
    mcp.run()
