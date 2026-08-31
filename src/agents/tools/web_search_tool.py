"""
Web Search Tool — Tavily-powered real-time web search.

Used when the agent needs external, up-to-date information that
is not available in the internal knowledge base (Qdrant/RAG).
Typical triggers: store hours, live status, news, directions.
"""

from loguru import logger
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from infrastructure.config import TIMEZONE
from infrastructure.observability import observe, update_current_observation
class WebSearchTool:
    """
    Tavily-powered web search tool.

    Returns formatted text suitable for injection into a synthesiser prompt.
    """

    def __init__(
        self,
        max_results: int = 5,
        search_depth: str = "advanced",
        prefer_domains: Optional[List[str]] = None,
    ) -> None:
        import os

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set in .env")

        from tavily import TavilyClient

        self.client = TavilyClient(api_key=api_key)
        self.max_results = max_results
        self.search_depth = search_depth
        self.prefer_domains = prefer_domains or [".lk", ".gov", ".org"]
        self.timezone = ZoneInfo(TIMEZONE)

    # ── core search ───────────────────────────────────────────

    @observe(name="web_search")
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> str:
        """
        Run a web search and return a formatted text summary.

        Traced via LangFuse so Tavily latency is visible in the dashboard.
        """
        start = time.time()

        update_current_observation(input=query)

        try:
            response = self.client.search(
                query=query,
                max_results=max_results or self.max_results,
                search_depth=self.search_depth,
                include_answer=True,
                include_raw_content=False,
            )
        except Exception as exc:
            logger.error("Web search failed: {}", exc)
            return f"Web search failed: {exc}"

        latency_ms = int((time.time() - start) * 1000)
        results = response.get("results", [])

        update_current_observation(
            metadata={
                "latency_ms": latency_ms,
                "result_count": len(results),
                "search_depth": self.search_depth,
            },
        )

        if not results:
            return "No web results found for your query."

        # ── rank & format ─────────────────────────────────────
        lines: List[str] = []

        answer = response.get("answer")
        if answer:
            lines.append(f"Summary: {answer}\n")

        lines.append("Web sources:")
        for idx, item in enumerate(results[: self.max_results], 1):
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            snippet = content[:300] + "…" if len(content) > 300 else content
            lines.append(f"  {idx}. {title}\n     {snippet}\n     URL: {url}")

        checked_at = datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M %Z")
        lines.append(f"\n(checked at {checked_at}, {latency_ms}ms)")

        return "\n".join(lines)

    # ── dispatch ──────────────────────────────────────────────

    def dispatch(self, action: str, params: Dict[str, Any]) -> str:
        """
        Dispatch (only action is ``search``).

        Kept for symmetry with CRMTool.dispatch().
        """
        if action != "search":
            return f"Unknown web_search action: {action}"
        return self.search(**params)
