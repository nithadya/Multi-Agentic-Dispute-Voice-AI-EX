"""
Agentic Routing Engine — the core agent module.

Public API:
    build_agent()        → AgentOrchestrator (fully wired, ready to chat)
    AgentOrchestrator    → main orchestrator class
    AgentResponse        → response dataclass
    QueryRouter          → intent classifier
    RouteDecision        → single routing result dataclass
    MultiRouteDecision   → multi-route container (fan-out support)
"""

from .orchestrator import AgentOrchestrator, AgentResponse, build_agent
from .router import QueryRouter, RouteDecision, MultiRouteDecision

__all__ = [
    "AgentOrchestrator",
    "AgentResponse",
    "QueryRouter",
    "RouteDecision",
    "MultiRouteDecision",
    "build_agent",
]
