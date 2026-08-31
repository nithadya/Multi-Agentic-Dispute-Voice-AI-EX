"""
Application configuration - loads from YAML param files.

CONFIGURATION POLICY:
====================
Configuration is loaded from config/param.yaml and config/models.yaml.
Secrets (API keys) live ONLY in .env and are loaded via os.getenv().

Supports multiple LLM providers via OpenRouter unified API or direct providers:
- OpenRouter (unified multi-provider access)
- OpenAI (direct)
- Anthropic (direct)
- Google/Gemini (direct)
- Groq (direct)
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml
from loguru import logger

# ========================================
# Project Paths
# ========================================

# Get project root (parent of src/infrastructure/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

# ========================================
# YAML Config Loading
# ========================================

def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_nested(d: Dict, *keys, default=None):
    """Get nested dictionary value safely."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


# Load configs
_PARAMS = _load_yaml("param.yaml")
_MODELS = _load_yaml("models.yaml")

# ========================================
# Provider Configuration
# ========================================

PROVIDER = _get_nested(_PARAMS, "provider", "default", default="openrouter")
MODEL_TIER = _get_nested(_PARAMS, "provider", "tier", default="general")
OPENROUTER_BASE_URL = _get_nested(_PARAMS, "provider", "openrouter_base_url",
                                   default="https://openrouter.ai/api/v1")

# ========================================
# Model Names (from models.yaml)
# ========================================

def get_chat_model(provider: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Get chat model name for specified provider and tier."""
    provider = provider or PROVIDER
    tier = tier or MODEL_TIER

    # Handle provider name mapping
    if provider == "google":
        provider = "google"  # Keep as-is for models.yaml
    elif provider == "gemini":
        provider = "google"  # Alias

    return _get_nested(_MODELS, provider, "chat", tier, default="openai/gpt-4o-mini")


EMBEDDING_TIER = _get_nested(_PARAMS, "embedding", "tier", default="default")


def get_embedding_model(provider: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Get embedding model name for specified provider and tier."""
    provider = provider or PROVIDER
    tier = tier or EMBEDDING_TIER

    # Handle provider name mapping
    if provider == "google" or provider == "gemini":
        provider = "google"

    return _get_nested(_MODELS, provider, "embedding", tier, default="openai/text-embedding-3-small")


# ========================================
# 3-Model Architecture
# ========================================
# Each role uses the best model for its task:
#   Routing:    gpt-4o-mini     — reliable JSON output, fast
#   Extraction: llama-3.1-8b    — ultra-fast via Groq, structured output
#   Chat:       gemini-2.0-flash — high quality synthesis, generous context

# Router uses the same 70B model as the fast synth path — same Groq
# latency profile (~400-600ms TTFT, ~200 tok/s), but a 70B classifier
# is materially better at picking up self-referential intent like
# "do I have an appointment today" → crm. The 8B model was misrouting
# these to "direct".
ROUTER_MODEL = "llama-3.3-70b-versatile"
ROUTER_PROVIDER = "groq"

EXTRACTOR_MODEL = "llama-3.1-8b-instant"
EXTRACTOR_PROVIDER = "groq"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

CHAT_MODEL = "gpt-4o-mini"
CHAT_PROVIDER = "openai"

# Fast LLM for the "direct" route (greetings/concierge).
# llama-3.3-70b-versatile on Groq:
#   - 70B parameters → proper instruction-following, no Sanskrit
#     hallucinations like the 8B model produced
#   - LPU hardware → ~500-800ms TTFT, ~200 tok/s sustained
#   - Same provider as the router (Groq), no proxy hop
FAST_CHAT_MODEL = "gpt-4o-mini"
FAST_CHAT_PROVIDER = "openai"

EMBEDDING_MODEL = get_embedding_model()

# Legacy alias
OPENAI_CHAT_MODEL = CHAT_MODEL

# ========================================
# Embedding Dimensions
# ========================================

# ⚠️ IMPORTANT: EMBEDDING_DIM must match the model's output dimensions
#
# Supported models:
#   - "text-embedding-3-small"  → 1536 dims (recommended for Qdrant)
#   - "text-embedding-3-large"  → 3072 dims
#   - "text-embedding-ada-002"  → 1536 dims (legacy)
#
# Qdrant supports any dimension size (no limit like Supabase)
EMBEDDING_DIM = 1536  # Default for text-embedding-3-small

# Auto-detect dimension from model name
if "large" in EMBEDDING_MODEL.lower():
    EMBEDDING_DIM = 3072
elif "small" in EMBEDDING_MODEL.lower() or "ada" in EMBEDDING_MODEL.lower():
    EMBEDDING_DIM = 1536

# ========================================
# LLM Defaults
# ========================================

LLM_TEMPERATURE = _get_nested(_PARAMS, "llm", "temperature", default=0.0)
LLM_MAX_TOKENS = _get_nested(_PARAMS, "llm", "max_tokens", default=2000)
LLM_STREAMING = _get_nested(_PARAMS, "llm", "streaming", default=False)

# ========================================
# Embedding Defaults
# ========================================

EMBEDDING_BATCH_SIZE = _get_nested(_PARAMS, "embedding", "batch_size", default=100)
EMBEDDING_SHOW_PROGRESS = _get_nested(_PARAMS, "embedding", "show_progress", default=False)

# ========================================
# Project Paths (from param.yaml)
# ========================================

DATA_DIR = _PROJECT_ROOT / _get_nested(_PARAMS, "paths", "data_dir", default="data")
KB_DIR = _PROJECT_ROOT / _get_nested(_PARAMS, "paths", "kb_dir", default="data/knowledge_base")

# Alternative ingestion sources (used by pipeline.py loader map)
JSONL_DIR = DATA_DIR / "jsonl"
MARKDOWN_DIR = _PROJECT_ROOT / _get_nested(_PARAMS, "paths", "markdown_dir", default="data/disputeai_markdown")

# NOTE: Both RAG KB and CAG cache live in Qdrant Cloud (separate collections).
# NOTE: ST memory lives in Supabase (st_turns table).

# ========================================
# Chunking Configuration
# ========================================

# Fixed-size chunking
FIXED_CHUNK_SIZE = _get_nested(_PARAMS, "chunking", "fixed", "chunk_size", default=800)
FIXED_CHUNK_OVERLAP = _get_nested(_PARAMS, "chunking", "fixed", "chunk_overlap", default=100)

# Semantic chunking
SEMANTIC_MAX_CHUNK_SIZE = _get_nested(_PARAMS, "chunking", "semantic", "max_chunk_size", default=1000)
SEMANTIC_MIN_CHUNK_SIZE = _get_nested(_PARAMS, "chunking", "semantic", "min_chunk_size", default=200)

# Sliding-window chunking
SLIDING_WINDOW_SIZE = _get_nested(_PARAMS, "chunking", "sliding", "window_size", default=512)
SLIDING_STRIDE_SIZE = _get_nested(_PARAMS, "chunking", "sliding", "stride_size", default=256)

# Parent-child chunking
PARENT_CHUNK_SIZE = _get_nested(_PARAMS, "chunking", "parent_child", "parent_size", default=1200)
CHILD_CHUNK_SIZE = _get_nested(_PARAMS, "chunking", "parent_child", "child_size", default=250)
CHILD_OVERLAP = _get_nested(_PARAMS, "chunking", "parent_child", "child_overlap", default=50)

# Late chunking
LATE_CHUNK_BASE_SIZE = _get_nested(_PARAMS, "chunking", "late", "base_size", default=1000)
LATE_CHUNK_SPLIT_SIZE = _get_nested(_PARAMS, "chunking", "late", "split_size", default=300)
LATE_CHUNK_CONTEXT_WINDOW = _get_nested(_PARAMS, "chunking", "late", "context_window", default=150)

# ========================================
# Retrieval Configuration
# ========================================

TOP_K_RESULTS = _get_nested(_PARAMS, "retrieval", "top_k", default=4)
SIMILARITY_THRESHOLD = _get_nested(_PARAMS, "retrieval", "similarity_threshold", default=0.7)

# ========================================
# CAG Configuration (Qdrant Semantic Cache)
# ========================================

CAG_COLLECTION_NAME = _get_nested(_PARAMS, "cag", "collection_name", default="cag_cache")
CAG_SIMILARITY_THRESHOLD = _get_nested(_PARAMS, "cag", "similarity_threshold", default=0.90)
CAG_CACHE_TTL = _get_nested(_PARAMS, "cag", "cache_ttl", default=86400)  # 24h
CAG_CACHE_MAX_SIZE = _get_nested(_PARAMS, "cag", "max_cache_size", default=1000)

# ========================================
# CRAG Configuration
# ========================================

CRAG_CONFIDENCE_THRESHOLD = _get_nested(_PARAMS, "crag", "confidence_threshold", default=0.6)
CRAG_EXPANDED_K = _get_nested(_PARAMS, "crag", "expanded_k", default=8)

# ========================================
# Crawling Configuration
# ========================================

CRAWL_MAX_DEPTH = _get_nested(_PARAMS, "crawling", "max_depth", default=3)
CRAWL_DELAY_SECONDS = _get_nested(_PARAMS, "crawling", "delay_seconds", default=2.0)
CRAWL_MAX_PAGES = _get_nested(_PARAMS, "crawling", "max_pages", default=100)

# ========================================
# Memory Configuration (NOT in YAML - kept as constants)
# ========================================

# Timezone (used by web_search_tool for timestamp display)
TIMEZONE = "Asia/Colombo"

# Short-term memory (Supabase st_turns table)
ST_MAX_TURNS = 30
ST_TTL_SECONDS = 60 * 60 * 24  # 24 hours

# Long-term memory (Supabase Postgres + pgvector)
LT_TOP_K = 5
LT_SIM_THRESHOLD = 0.30  # Lowered from 0.65 for better recall
LT_TTL_SECONDS = 60 * 60 * 24 * 90  # 90 days
LT_DECAY_HALF_LIFE_DAYS = 30
MEM_COLLECTION = "mem_vectors"

# ========================================
# Reminders Configuration (FUTURE — not implemented in Week 07)
# Will be re-added when reminders_service is wired in.
# REM_TZ, REM_POLL_SECONDS, REM_DEFAULT_CHANNEL, QUIET_HOURS,
# REMINDER_OFFSETS_SECONDS, CRM_HORIZON_DAYS
# ========================================

# ========================================
# Database URLs
# ========================================

# Short-Term Memory: Supabase only (Redis introduced in a later week)

# ========================================
# Qdrant Cloud Configuration
# ========================================

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "ecommerce_policies")

# ========================================
# Neo4j Graph Database (Graph RAG — Week 11 inspired)
# ========================================
# Stores e-commerce vendor policies as a knowledge graph.
# Enables multi-hop Cypher reasoning that standard vector RAG cannot do.

NEO4J_URI = os.getenv("NEO4J_URI", None)
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", None)
NEO4J_DATABASE = _get_nested(_PARAMS, "graph", "database", default="neo4j")
GRAPH_RAG_ENABLED = _get_nested(_PARAMS, "graph", "enabled", default=False)
GRAPH_MAX_HOPS = _get_nested(_PARAMS, "graph", "max_hops", default=2)
GRAPH_TOP_K_NODES = _get_nested(_PARAMS, "graph", "top_k_nodes", default=10)
GRAPH_CRAG_MAX_RETRIES = _get_nested(_PARAMS, "graph", "crag_max_retries", default=2)

# ========================================
# FAQ Loading (optional)
# ========================================

def load_faqs() -> list:
    """
    Load known FAQs from config/faqs.yaml (if exists).

    Returns:
        List of FAQ items (can be strings or dictionaries)
    """
    faqs_config = _load_yaml("faqs.yaml")
    if not faqs_config:
        return []

    all_faqs = []
    # Flatten all categories into a single list
    for category, items in faqs_config.items():
        if isinstance(items, list):
            all_faqs.extend(items)
    
    return all_faqs


# Pre-load FAQs for easy access (empty if file doesn't exist)
KNOWN_FAQS = load_faqs()

# ========================================
# Helper Functions
# ========================================

def get_api_key(provider: Optional[str] = None) -> Optional[str]:
    """Get API key for the specified provider."""
    provider = provider or PROVIDER
    key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",  # Alias
        "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "tavily": "TAVILY_API_KEY",
    }
    env_var = key_map.get(provider, f"{provider.upper()}_API_KEY")
    return os.getenv(env_var)


def validate() -> None:
    """
    Validate configuration and create required directories.

    Raises:
        ValueError: If required secrets are missing
        OSError: If directories cannot be created
    """
    # Check required secrets based on provider
    api_key = get_api_key()
    if not api_key:
        key_name = "OPENROUTER_API_KEY" if PROVIDER == "openrouter" else f"{PROVIDER.upper()}_API_KEY"
        raise ValueError(
            f"❌ Missing required secret: {key_name}\n"
            f"Please add it to your .env file."
        )

    # Create required directories (only active ones)
    required_dirs = [DATA_DIR, KB_DIR]

    for dir_path in required_dirs:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise OSError(f"❌ Cannot create directory {dir_path}: {e}")


def dump() -> None:
    """Print all active non-secret configuration values for debugging."""
    logger.info("\n" + "=" * 60)
    logger.info("CONFIGURATION (NON-SECRETS ONLY)")
    logger.info("=" * 60)

    logger.info("\n🌐 Provider:")
    logger.info(f"   Provider: {PROVIDER}")
    logger.info(f"   Model Tier: {MODEL_TIER}")
    logger.info(f"   Chat Model: {CHAT_MODEL}")
    logger.info(f"   Embedding Model: {EMBEDDING_MODEL}")
    logger.info(f"   Embedding Dimensions: {EMBEDDING_DIM}")

    logger.info("\n📁 Directories & Storage:")
    logger.info(f"   Data Root: {DATA_DIR}")
    logger.info(f"   Knowledge Base: {KB_DIR}")
    logger.info(f"   🟡 RAG Vectors: Qdrant Cloud ({QDRANT_COLLECTION_NAME})")
    logger.info(f"   🟡 CAG Cache: Qdrant Cloud ({CAG_COLLECTION_NAME})")
    logger.info(f"   🟢 ST + LT Memory + CRM: Supabase PostgreSQL")

    logger.info("\n🔧 Chunking:")
    logger.info(f"   Fixed Size: {FIXED_CHUNK_SIZE} tokens")
    logger.info(f"   Fixed Overlap: {FIXED_CHUNK_OVERLAP} tokens")
    logger.info(f"   Sliding Window: {SLIDING_WINDOW_SIZE} tokens")
    logger.info(f"   Sliding Stride: {SLIDING_STRIDE_SIZE} tokens")
    logger.info(f"   Parent-Child: {CHILD_CHUNK_SIZE} → {PARENT_CHUNK_SIZE} tokens")
    logger.info(f"   Late Chunk: {LATE_CHUNK_BASE_SIZE} → {LATE_CHUNK_SPLIT_SIZE} tokens")

    logger.info("\n🔍 Retrieval:")
    logger.info(f"   Top-K Results: {TOP_K_RESULTS}")
    logger.info(f"   Similarity Threshold: {SIMILARITY_THRESHOLD}")

    logger.info("\n💾 CAG (Semantic Cache — Qdrant):")
    logger.info(f"   Collection: {CAG_COLLECTION_NAME}")
    logger.info(f"   Similarity Threshold: {CAG_SIMILARITY_THRESHOLD}")
    logger.info(f"   TTL (seconds): {CAG_CACHE_TTL}")
    logger.info(f"   Max Cache Size: {CAG_CACHE_MAX_SIZE}")

    logger.success("\n🎯 CRAG:")
    logger.info(f"   Confidence Threshold: {CRAG_CONFIDENCE_THRESHOLD}")
    logger.info(f"   Expanded K: {CRAG_EXPANDED_K}")

    logger.info("\n🧠 Memory:")
    logger.info(f"   Short-term Max Turns: {ST_MAX_TURNS}")
    logger.info(f"   Long-term Top-K: {LT_TOP_K}")
    logger.info(f"   Long-term Threshold: {LT_SIM_THRESHOLD}")

    logger.info("\n🗄️  Qdrant:")
    logger.info(f"   Collection: {QDRANT_COLLECTION_NAME}")
    logger.success(f"   URL: {'✅ Set' if QDRANT_URL else '❌ Not set'}")
    logger.success(f"   API Key: {'✅ Set' if QDRANT_API_KEY else '❌ Not set'}")

    logger.info("\n" + "=" * 60 + "\n")


def get_all_models() -> Dict[str, Any]:
    """Return all available models from models.yaml."""
    return _MODELS


def get_config() -> Dict[str, Any]:
    """Return full config dictionary."""
    return _PARAMS
