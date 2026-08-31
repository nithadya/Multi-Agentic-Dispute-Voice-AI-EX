"""
Web Crawler MCP Server — exposes the Playwright BFS crawler over MCP.

Wraps ``src/services/ingest_service/web_crawler.py``. The crawler is
heavyweight (spins up a headless Chromium via Playwright) so the tool
is async and returns structured documents the client can feed into
any ingestion pipeline — not just the store one.

Transport: stdio

Run standalone:
    python -m mcp_servers.crawler_server

Inspect:
    npx @modelcontextprotocol/inspector python -m mcp_servers.crawler_server
"""

import os
import sys
from typing import Any, Dict, List, Optional

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from mcp.server.fastmcp import FastMCP

from services.ingest_service.web_crawler import DisputeAIWebCrawler


mcp = FastMCP("disputeai-crawler")


@mcp.tool()
async def crawl(
    start_urls: List[str],
    base_url: str,
    max_depth: int = 2,
    exclude_patterns: Optional[List[str]] = None,
    request_delay: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    BFS-crawl a website with a headless Chromium browser and return
    structured markdown documents.

    Args:
        start_urls: Seed URLs to begin the crawl from.
        base_url: Domain boundary — only URLs starting with this
                  prefix are followed (keeps the crawl on-site).
        max_depth: How many hops from the seeds to traverse.
        exclude_patterns: Substrings that, if present in a URL, skip
                          it (e.g. ["/login", "/admin"]).
        request_delay: Seconds between page loads (politeness).

    Returns a list of dicts, each with: ``url``, ``title``,
    ``headings``, ``content`` (markdown), ``links``, ``depth_level``.
    Pages with under 100 chars of extracted content are dropped.
    """
    crawler = DisputeAIWebCrawler(
        base_url=base_url,
        max_depth=max_depth,
        exclude_patterns=exclude_patterns or [],
    )
    docs = await crawler.crawl_async(start_urls, request_delay=request_delay)
    logger.info("Crawl finished: {} documents", len(docs))
    return docs


if __name__ == "__main__":
    logger.info("Starting disputeai-crawler MCP server on stdio...")
    mcp.run()
