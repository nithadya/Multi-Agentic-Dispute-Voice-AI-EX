"""
Database clients for the Agentic Memory system.

2-tier storage architecture:
    🟡 Warm  — Qdrant Cloud  → RAG KB vectors + CAG semantic cache
    🟢 Cold  — Supabase PG   → ST Memory + LT Memory (pgvector) + CRM (relational)
"""

from .sql_client import get_sql_engine, create_tables, get_session
from .supabase_client import (
    get_supabase_client,
    get_supabase_engine,
    get_supabase_session,
    test_connection,
    check_pgvector_installed,
    init_supabase_schema,
)
from .crm_init import init_crm_schema, check_crm_schema
from .qdrant_client import (
    get_qdrant_client,
    ensure_collection,
    delete_collection,
    collection_info,
    upsert_chunks,
    search_chunks,
    count_points,
    collection_exists,
    ensure_kb_ingested,
)

__all__ = [
    # Supabase (ST memory + Long-term memory + CRM)
    "get_sql_engine",
    "get_session",
    "create_tables",
    "get_supabase_client",
    "get_supabase_engine",
    "get_supabase_session",
    "test_connection",
    "check_pgvector_installed",
    "init_supabase_schema",

    # CRM
    "init_crm_schema",
    "check_crm_schema",

    # Qdrant Cloud (RAG KB vectors + CAG semantic cache)
    "get_qdrant_client",
    "ensure_collection",
    "delete_collection",
    "collection_info",
    "upsert_chunks",
    "search_chunks",
    "count_points",
    "collection_exists",
    "ensure_kb_ingested",
]
