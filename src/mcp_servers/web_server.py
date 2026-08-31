"""
Web Search MCP Server — exposes Tavily real-time search over MCP.

Wraps `src/agents/tools/web_search_tool.py`. The simplest MCP server
in the project — one tool, ~20 lines of wrapper code.

Transport: stdio

Run standalone:
    python -m mcp_servers.web_server

Inspect:
    npx @modelcontextprotocol/inspector python -m mcp_servers.web_server
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

from agents.tools.web_search_tool import WebSearchTool


# ── Server + lazy init ──────────────────────────────────────────

mcp = FastMCP("disputeai-web")

_web: WebSearchTool | None = None


def _get_web() -> WebSearchTool:
    global _web
    if _web is None:
        logger.info("Initialising WebSearchTool inside MCP server...")
        _web = WebSearchTool()
    return _web


# ── MCP Tools ───────────────────────────────────────────────────

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for real-time information using Tavily.

    Use this for questions that need up-to-date data not found in the
    policy database: visiting hours, directions, current events,
    live status, news, etc.

    Returns a summary + ranked web sources with snippets and URLs.
    """
    return _get_web().dispatch("search", {"query": query, "max_results": max_results})


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting disputeai-web MCP server on stdio...")
    mcp.run()
