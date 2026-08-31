"""
SQL client — canonical SQLAlchemy engine and session for Supabase PostgreSQL.

Provides:
- Singleton engine backed by SUPABASE_DB_URL
- Session factory
- SQLAlchemy table definitions for memory tables (mem_facts, mem_episodes)
"""

from loguru import logger
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Boolean,
    Text,
    Integer,
    MetaData,
    Table,
    DateTime,
    UUID,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from pgvector.sqlalchemy import Vector
from typing import Optional
import os

from infrastructure.config import EMBEDDING_DIM
# Singleton engine
_engine: Optional[object] = None
_SessionLocal: Optional[object] = None

# Metadata
metadata = MetaData()

# ============================================================================
# MEMORY TABLES (Supabase PostgreSQL + pgvector)
# ============================================================================

# Memory facts table (Long-term semantic memory)
mem_facts_table = Table(
    "mem_facts",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("user_id", Text, nullable=False, index=True),
    Column("text", Text, nullable=False),
    Column("embedding", Vector(EMBEDDING_DIM), nullable=True),  # pgvector column
    Column("score", Float, nullable=False),
    Column("tags", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
    Column("last_used_at", DateTime(timezone=True), nullable=True, server_default=text("NOW()")),
    Column("ttl_at", DateTime(timezone=True), nullable=True),
    Column("pin", Boolean, nullable=False, server_default=text("FALSE")),
    Column("deleted", Boolean, nullable=False, server_default=text("FALSE"), index=True),
)

# Memory episodes table (Long-term episodic memory)
mem_episodes_table = Table(
    "mem_episodes",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("user_id", Text, nullable=False, index=True),
    Column("session_id", Text, nullable=False, index=True),
    Column("summary", Text, nullable=False),
    Column("summary_embedding", Vector(EMBEDDING_DIM), nullable=True),  # pgvector column
    Column("topic_tags", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("start_at", DateTime(timezone=True), nullable=False),
    Column("end_at", DateTime(timezone=True), nullable=False),
    Column("turn_count", Integer, nullable=False),
    Column("turns", JSONB, nullable=False),  # Full conversation as JSON
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
)


def get_sql_engine():
    """
    Get SQLAlchemy engine for Supabase PostgreSQL.

    The default Supabase ``.pooler.supabase.com`` URL we expect in
    ``SUPABASE_DB_URL`` should point at the **transaction-mode pooler
    on port 6543**, not the session-mode pooler on 5432.  Session-mode
    caps at ~15 clients per project, which a chat workload exhausts in
    minutes; the transaction pooler handles thousands.

    Engine config:
      - ``pool_size`` / ``max_overflow``: small client-side pool; each
        connection is a "session" with the pooler that shares an
        underlying Postgres connection across many transactions.
      - ``pool_pre_ping=True``: catches dead connections before use.
      - ``pool_recycle=300``: refresh hourly idle connections to avoid
        the pooler's idle-disconnect.
    """
    global _engine
    if _engine is None:
        db_url = os.getenv("SUPABASE_DB_URL")

        if not db_url:
            raise ValueError(
                "SUPABASE_DB_URL must be set in .env file. "
                "Format: postgresql://postgres:[password]@db.xxxxx.supabase.co:6543/postgres "
                "(use port 6543 for the transaction-mode pooler — 5432 caps at 15 clients)."
            )

        if ":5432" in db_url:
            logger.warning(
                "SUPABASE_DB_URL points at port 5432 (session-mode pooler, max 15 clients). "
                "Switch to port 6543 (transaction pooler) for chat-grade concurrency."
            )

        _engine = create_engine(
            db_url,
            pool_size=8,
            max_overflow=12,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False,
        )
        logger.info("✓ Supabase SQL engine created")
    return _engine


def get_session():
    """
    Get SQLAlchemy session for Supabase.
    
    Returns:
        SQLAlchemy session
    """
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_sql_engine()
        _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return _SessionLocal()


def create_tables():
    """
    Create database tables if they don't exist.
    
    Note: In production with Supabase, tables should be created via SQL Editor
    using sql/supabase_schema.sql for full pgvector support and RLS policies.
    """
    engine = get_sql_engine()
    metadata.create_all(bind=engine)
    logger.info("✓ Database tables created/verified")


def test_connection():
    """
    Test Supabase database connection.
    
    Returns:
        True if connection successful, False otherwise
    """
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
