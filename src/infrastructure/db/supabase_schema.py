"""
Dynamic Supabase schema generator - reads dimensions from config.

This ensures the database schema always matches config.EMBEDDING_DIM
without any hardcoded values in SQL files.
"""

from infrastructure.config import EMBEDDING_DIM, EMBEDDING_MODEL


def generate_supabase_schema() -> str:
    """
    Generate Supabase schema DDL dynamically from config.
    
    Returns:
        SQL DDL string with vector dimensions from config.EMBEDDING_DIM
    """
    
    return f"""-- ============================================================================
-- Supabase Schema: Memory System + E-commerce Dispute Resolution
-- PostgreSQL 15+ with pgvector extension
-- ============================================================================
-- 
-- ⚠️ DYNAMICALLY GENERATED FROM CONFIG
-- Embedding Model: {EMBEDDING_MODEL}
-- Vector Dimensions: {EMBEDDING_DIM}
-- 
-- This schema is generated programmatically to ensure dimensions
-- always match config.EMBEDDING_DIM (single source of truth).
-- 
-- ============================================================================

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- SHORT-TERM MEMORY (Supabase backend — ring buffer with TTL)
-- ============================================================================

CREATE TABLE IF NOT EXISTS st_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ttl_at TIMESTAMPTZ  -- Auto-cleanup after this time (default 24h from created_at)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_st_turns_user_session ON st_turns (user_id, session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_st_turns_ttl ON st_turns (ttl_at) WHERE ttl_at IS NOT NULL;

COMMENT ON TABLE st_turns IS 'Short-term conversation memory — ring buffer with TTL';

-- ============================================================================
-- LONG-TERM SEMANTIC MEMORY (pgvector facts)
-- ============================================================================

CREATE TABLE IF NOT EXISTS mem_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM}),
    score REAL NOT NULL CHECK (score >= 0 AND score <= 1),
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ DEFAULT NOW(),
    ttl_at TIMESTAMPTZ,
    pin BOOLEAN DEFAULT FALSE,
    deleted BOOLEAN DEFAULT FALSE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_mem_facts_user_id ON mem_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_facts_score ON mem_facts(score DESC);
CREATE INDEX IF NOT EXISTS idx_mem_facts_deleted ON mem_facts(deleted) WHERE deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mem_facts_ttl ON mem_facts(ttl_at) WHERE ttl_at IS NOT NULL;

-- pgvector index
CREATE INDEX IF NOT EXISTS idx_mem_facts_embedding 
ON mem_facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Helper function for semantic search
CREATE OR REPLACE FUNCTION search_mem_facts(
    query_embedding vector({EMBEDDING_DIM}),
    query_user_id TEXT,
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    user_id TEXT,
    text TEXT,
    score REAL,
    tags JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.id,
        f.user_id,
        f.text,
        f.score,
        f.tags,
        1 - (f.embedding <=> query_embedding) AS similarity
    FROM mem_facts f
    WHERE f.user_id = query_user_id
        AND f.deleted = FALSE
        AND (f.ttl_at IS NULL OR f.ttl_at > NOW())
        AND 1 - (f.embedding <=> query_embedding) >= match_threshold
    ORDER BY f.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- LONG-TERM EPISODIC MEMORY (pgvector episodes)
-- ============================================================================

CREATE TABLE IF NOT EXISTS mem_episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    summary_embedding vector({EMBEDDING_DIM}),
    topic_tags JSONB DEFAULT '[]'::jsonb,
    start_at TIMESTAMPTZ NOT NULL,
    end_at TIMESTAMPTZ NOT NULL,
    turn_count INTEGER NOT NULL CHECK (turn_count > 0),
    turns JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_mem_episodes_user_id ON mem_episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_episodes_session_id ON mem_episodes(session_id);

-- pgvector index
CREATE INDEX IF NOT EXISTS idx_mem_episodes_embedding 
ON mem_episodes USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 100);

-- Helper function for semantic search
CREATE OR REPLACE FUNCTION search_mem_episodes(
    query_embedding vector({EMBEDDING_DIM}),
    query_user_id TEXT,
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    user_id TEXT,
    session_id TEXT,
    summary TEXT,
    topic_tags JSONB,
    start_at TIMESTAMPTZ,
    end_at TIMESTAMPTZ,
    turn_count INTEGER,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id,
        e.user_id,
        e.session_id,
        e.summary,
        e.topic_tags,
        e.start_at,
        e.end_at,
        e.turn_count,
        1 - (e.summary_embedding <=> query_embedding) AS similarity
    FROM mem_episodes e
    WHERE e.user_id = query_user_id
        AND 1 - (e.summary_embedding <=> query_embedding) >= match_threshold
    ORDER BY e.summary_embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PROCEDURAL MEMORY (How-to knowledge)
-- ============================================================================

CREATE TABLE IF NOT EXISTS mem_procedures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    context_when TEXT,
    steps JSONB NOT NULL,
    conditions JSONB,
    examples JSONB,
    embedding vector({EMBEDDING_DIM}),
    category TEXT,
    active BOOLEAN DEFAULT TRUE,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- CHAT SESSIONS (Matches ORM ChatSession)
-- ============================================================================

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    title TEXT NOT NULL,
    last_message_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_customer_id ON chat_sessions(customer_id);

-- ============================================================================
-- E-COMMERCE: CUSTOMERS
-- ============================================================================

CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,
    external_user_id TEXT NOT NULL UNIQUE,  -- Phone number without '+'
    name TEXT NOT NULL,
    email TEXT,
    tier TEXT DEFAULT 'standard',
    total_orders INTEGER DEFAULT 0,
    dispute_count INTEGER DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_customers_external_user_id ON customers(external_user_id);

-- ============================================================================
-- E-COMMERCE: VENDORS
-- ============================================================================

CREATE TABLE IF NOT EXISTS vendors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT,
    active INTEGER NOT NULL DEFAULT 1
);

-- ============================================================================
-- E-COMMERCE: ORDERS
-- ============================================================================

CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    order_number TEXT UNIQUE NOT NULL,
    customer_id TEXT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    vendor_id TEXT REFERENCES vendors(id) ON DELETE SET NULL,
    status TEXT NOT NULL,
    total_amount REAL NOT NULL,
    currency TEXT DEFAULT 'LKR',
    purchase_date TIMESTAMPTZ,
    items JSONB DEFAULT '[]'::jsonb,
    shipping_address TEXT,
    tracking_number TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);

-- ============================================================================
-- E-COMMERCE: DISPUTES
-- ============================================================================

CREATE TABLE IF NOT EXISTS disputes (
    id TEXT PRIMARY KEY,
    dispute_number TEXT UNIQUE NOT NULL,
    order_id TEXT NOT NULL,
    customer_id TEXT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    complaint_text TEXT,
    evidence_urls JSONB DEFAULT '[]'::jsonb,
    decision TEXT,
    refund_amount REAL,
    currency TEXT DEFAULT 'LKR',
    resolution_notes TEXT,
    customer_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_disputes_customer ON disputes(customer_id);
CREATE INDEX IF NOT EXISTS idx_disputes_number ON disputes(dispute_number);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) - Production Ready
-- ============================================================================

ALTER TABLE mem_facts ENABLE ROW LEVEL SECURITY;
ALTER TABLE mem_episodes ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own facts" ON mem_facts;
DROP POLICY IF EXISTS "Users can manage their own facts" ON mem_facts;
DROP POLICY IF EXISTS "Users can view their own episodes" ON mem_episodes;
DROP POLICY IF EXISTS "Users can manage their own episodes" ON mem_episodes;

CREATE POLICY "Users can view their own facts"
    ON mem_facts FOR SELECT
    USING (user_id = current_setting('app.user_id', TRUE));

CREATE POLICY "Users can manage their own facts"
    ON mem_facts FOR ALL
    USING (user_id = current_setting('app.user_id', TRUE));

CREATE POLICY "Users can view their own episodes"
    ON mem_episodes FOR SELECT
    USING (user_id = current_setting('app.user_id', TRUE));

CREATE POLICY "Users can manage their own episodes"
    ON mem_episodes FOR ALL
    USING (user_id = current_setting('app.user_id', TRUE));

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE mem_facts IS 'Long-term semantic memory facts';
COMMENT ON TABLE mem_episodes IS 'Long-term episodic memory';
COMMENT ON TABLE customers IS 'E-commerce customer records';
COMMENT ON TABLE vendors IS 'E-commerce vendor records';
COMMENT ON TABLE orders IS 'E-commerce orders';
COMMENT ON TABLE disputes IS 'E-commerce disputes';

-- ============================================================================
-- COMPLETION
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '✅ Supabase E-commerce schema created successfully!';
    RAISE NOTICE '📊 Tables created: st_turns, mem_facts, mem_episodes, customers, orders, disputes, vendors';
    RAISE NOTICE '🎯 Ready for production use!';
END $$;
"""
