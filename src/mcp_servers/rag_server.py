"""
RAG MCP Server — exposes the policy database over MCP.

Wraps `src/agents/tools/rag_tool.py` which provides:
  - CAG cache (semantic dedup via Qdrant)
  - CRAG (corrective retrieval-augmented generation)
  - Qdrant KB search (parent-child chunks)

Transport: stdio

Run standalone:
    python -m mcp_servers.rag_server

Inspect:
    npx @modelcontextprotocol/inspector python -m mcp_servers.rag_server
"""

import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from mcp.server.fastmcp import FastMCP

from agents.tools.rag_tool import RAGTool
from infrastructure.llm import get_chat_llm, get_default_embeddings


# ── Server + lazy init ──────────────────────────────────────────

mcp = FastMCP("disputeai-kb")

_rag: RAGTool | None = None


def _get_rag() -> RAGTool:
    global _rag
    if _rag is None:
        logger.info("Initialising RAGTool inside MCP server...")
        embedder = get_default_embeddings()
        llm = get_chat_llm(temperature=0)
        _rag = RAGTool(embedder=embedder, llm=llm)
    return _rag


# ── MCP Tools ───────────────────────────────────────────────────

@mcp.tool()
def search_policy_kb(query: str) -> str:
    """
    Search the DisputeAI knowledge base.

    Uses the full CAG + CRAG pipeline:
    1. Check semantic cache (CAG) for a previous similar query
    2. If cache miss, retrieve from Qdrant KB (parent-child chunks)
    3. Apply confidence gate (CRAG) — expand search if low confidence
    4. Generate answer via LLM
    5. Cache the result for future queries

    Use this for store policies, procedures, services, departments,
    and any medical/operational information about DisputeAI.
    """
    return _get_rag().dispatch("search", {"query": query})


@mcp.tool()
def cache_stats() -> str:
    """Return CAG cache statistics (hit rate, entry count)."""
    return _get_rag().dispatch("cache_stats", {})


@mcp.tool()
def clear_cache() -> str:
    """Clear the CAG semantic cache. Use when KB content has been updated."""
    return _get_rag().dispatch("clear_cache", {})


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting disputeai-kb MCP server on stdio...")
    mcp.run()
