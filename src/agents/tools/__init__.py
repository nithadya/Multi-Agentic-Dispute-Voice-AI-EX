"""
Agent tools — CRM, Web Search, and RAG.

Each tool exposes a ``dispatch(action, params)`` method so the
orchestrator can call them uniformly.
"""

from .order_tool import OrderTool as CRMTool
from .rag_tool import RAGTool
from .web_search_tool import WebSearchTool

__all__ = [
    "CRMTool",
    "RAGTool",
    "WebSearchTool",
]
