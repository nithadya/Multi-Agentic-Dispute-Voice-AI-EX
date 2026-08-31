"""
Supabase client — REST API + SQLAlchemy helpers.

Provides:
- ``get_supabase_client()``  — Supabase REST/Auth/Realtime client
- ``get_supabase_engine()``  — Delegates to sql_client.get_sql_engine()
- ``get_supabase_session()`` — Delegates to sql_client.get_session()
- Utility functions: test_connection, pgvector check, schema init, RLS, etc.

The canonical SQLAlchemy engine/session lives in ``sql_client.py``.
This module re-exports those functions under Supabase-prefixed names
for convenience so existing imports continue to work.
"""

import os
from loguru import logger
from typing import Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from supabase import create_client, Client

from infrastructure.config import EMBEDDING_DIM

# Canonical engine/session — single source of truth
from .sql_client import get_sql_engine, get_session
# ---------------------------------------------------------------------------
# Supabase REST client (Auth, Realtime, Storage)
# ---------------------------------------------------------------------------

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """
    Get or create Supabase REST client (for Auth, Realtime, Storage).

    Returns:
        Supabase Client instance
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env file"
        )

    _supabase_client = create_client(supabase_url, supabase_key)
    logger.info(f"✓ Supabase client created: {supabase_url}")

    return _supabase_client


# ---------------------------------------------------------------------------
# SQLAlchemy delegates — thin wrappers over sql_client
# ---------------------------------------------------------------------------


def get_supabase_engine():
    """
    Return the canonical SQLAlchemy engine (delegates to sql_client).

    Kept for backward-compatible imports.
    """
    return get_sql_engine()


def get_supabase_session() -> Session:
    """
    Return a new SQLAlchemy session (delegates to sql_client).

    Kept for backward-compatible imports.
    """
    return get_session()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def test_connection() -> bool:
    """Test Supabase PostgreSQL connection."""
    try:
        engine = get_sql_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

        logger.info("✅ Supabase connection test: SUCCESS")
        return True

    except Exception as e:
        logger.error(f"❌ Supabase connection test: FAILED - {e}")
        return False


def check_pgvector_installed() -> bool:
    """Check if pgvector extension is installed in Supabase."""
    try:
        engine = get_sql_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            installed = result.scalar() == "vector"

        if installed:
            logger.info("✅ pgvector extension: INSTALLED")
        else:
            logger.warning("⚠️  pgvector extension: NOT INSTALLED")
            logger.warning("   Run in Supabase SQL Editor: CREATE EXTENSION vector;")

        return installed

    except Exception as e:
        logger.error(f"❌ Failed to check pgvector: {e}")
        return False


def init_supabase_schema():
    """
    Initialize Supabase schema dynamically from config.

    Creates all tables, indexes, functions, and RLS policies.
    Vector dimensions are loaded from config.EMBEDDING_DIM.
    """
    from .supabase_schema import generate_supabase_schema

    sql_content = generate_supabase_schema()
    engine = get_sql_engine()

    try:
        with engine.begin() as conn:
            conn.execute(text(sql_content))

        logger.info(
            f"✅ Supabase schema initialised (vector dim={EMBEDDING_DIM})"
        )

    except Exception as e:
        logger.error(f"❌ Failed to initialise schema: {e}")
        raise


def set_user_context(user_id: str):
    """
    Set user context for Row Level Security (RLS).

    Args:
        user_id: User identifier
    """
    engine = get_sql_engine()

    with engine.connect() as conn:
        conn.execute(text(f"SET app.user_id = '{user_id}'"))


def validate_schema_dimensions():
    """
    Validate that Supabase schema vector dimensions match config.EMBEDDING_DIM.
    """
    try:
        engine = get_sql_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT atttypmod
                    FROM pg_attribute
                    WHERE attrelid = 'mem_facts'::regclass
                    AND attname = 'embedding'
                """)
            ).scalar()

            if result:
                db_dim = result
                if db_dim != EMBEDDING_DIM:
                    raise ValueError(
                        f"❌ Schema dimension mismatch!\n"
                        f"   Database: vector({db_dim})\n"
                        f"   Config:   EMBEDDING_DIM={EMBEDDING_DIM}\n"
                        f"   Fix: Run 'make clean-supabase && make init-supabase'"
                    )

                logger.info(f"✓ Schema validation passed: vector({EMBEDDING_DIM})")
            else:
                logger.warning(
                    "⚠️ Could not validate schema dimensions (table may not exist)"
                )

    except Exception as e:
        logger.warning(f"⚠️ Schema validation skipped: {e}")


# ---------------------------------------------------------------------------
# Health check on import
# ---------------------------------------------------------------------------

# NOTE: Import-time health checks (test_connection, check_pgvector_installed,
# validate_schema_dimensions) have been removed to prevent 100-second blocking
# DNS timeouts in Docker WSL2 environments during LiveKit worker startup.
