"""
Query Router — LLM-based intent classification.

Takes a user message + memory context and returns a ``MultiRouteDecision``
containing one or more ``RouteDecision`` objects.  When the user query
contains multiple independent intents (e.g. "Check my return status AND
what is the refund policy?") the router returns multiple
routes so the orchestrator can fan out to parallel agent nodes.
"""

import json
from loguru import logger
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.prompts.agent_prompts import build_router_prompt
from infrastructure.observability import observe, update_current_observation

# Valid routes
VALID_ROUTES = {"order", "rag", "web_search", "direct"}

# Valid ORDER (CRM) sub-actions — e-commerce dispute domain
VALID_CRM_ACTIONS = {
    "lookup_order",
    "lookup_customer",
    "submit_dispute",
    "check_dispute_status",
    "update_dispute",
    "calculate_refund",
    "list_disputes",
}

# Maximum routes per query (safety cap)
MAX_ROUTES = 3


@dataclass
class RouteDecision:
    """
    A single routing decision for one intent.

    Attributes:
        route: Primary route (crm | rag | web_search | direct).
        confidence: Router's self-assessed confidence [0-1].
        reasoning: One-line explanation of the routing decision.
        action: CRM sub-action (only when route == crm).
        params: Extracted parameters for the tool.
    """

    route: str = "direct"
    confidence: float = 0.0
    reasoning: str = ""
    action: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiRouteDecision:
    """
    Container for one or more RouteDecision objects.

    Single-intent queries produce ``decisions`` with one element.
    Multi-intent queries (e.g. "file a return request AND tell me
    about the refund policy") produce multiple elements, enabling
    LangGraph fan-out to parallel agent nodes.
    """

    decisions: List[RouteDecision] = field(default_factory=list)

    @property
    def is_multi_route(self) -> bool:
        return len(self.decisions) > 1

    @property
    def primary(self) -> RouteDecision:
        """First (or only) decision — backward compatibility."""
        return self.decisions[0] if self.decisions else RouteDecision()


class QueryRouter:
    """
    Routes user queries to the appropriate tool path.

    Uses an LLM call with structured JSON output to classify intent.
    Falls back to ``direct`` on parse errors.
    """

    def __init__(self, llm: Any) -> None:
        """
        Args:
            llm: A LangChain ``ChatOpenAI`` (or compatible) instance.
        """
        self.llm = llm

    @observe(name="router", as_type="generation")
    def route(
        self,
        user_message: str,
        memory_context: str = "",
    ) -> MultiRouteDecision:
        """Synchronous routing — kept for the LangGraph orchestrator path."""
        return self._call(user_message, memory_context, async_call=False)

    @observe(name="router", as_type="generation")
    async def aroute(
        self,
        user_message: str,
        memory_context: str = "",
    ) -> MultiRouteDecision:
        """
        Async router — used by the API hot path so it can run concurrently
        with CAG lookup and memory recall via ``asyncio.gather``.

        Identical logic to ``route()`` but awaits ``llm.ainvoke`` instead
        of blocking on ``llm.invoke``.
        """
        return await self._acall(user_message, memory_context)

    # ── Internal sync/async cores ──────────────────────────────────

    def _build_messages(self, user_message: str, memory_context: str):
        system_prompt, user_prompt = build_router_prompt(
            user_message=user_message,
            memory_context=memory_context,
        )
        update_current_observation(
            input=user_prompt[:1000],
            model=self._model_name(),
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _record_usage(self, content: str, response) -> None:
        usage = {}
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata or {}
            token_usage = meta.get("token_usage") or meta.get("usage", {})
            if token_usage:
                usage = {
                    "input": token_usage.get("prompt_tokens", 0),
                    "output": token_usage.get("completion_tokens", 0),
                    "total": token_usage.get("total_tokens", 0),
                }
        update_current_observation(
            output=content[:500],
            usage=usage if usage else None,
        )

    @staticmethod
    def _content(response) -> str:
        return response.content if hasattr(response, "content") else str(response)

    def _call(self, user_message: str, memory_context: str, async_call: bool = False):
        # async_call kept for symmetry; sync path is used by the LangGraph nodes.
        try:
            response = self.llm.invoke(self._build_messages(user_message, memory_context))
            content = self._content(response)
            self._record_usage(content, response)
        except Exception as exc:
            logger.error("Router LLM call failed: {}", exc)
            return MultiRouteDecision(decisions=[
                RouteDecision(route="direct", confidence=0.0, reasoning=f"Router LLM error: {exc}")
            ])
        return self._parse_response(content)

    async def _acall(self, user_message: str, memory_context: str):
        try:
            response = await self.llm.ainvoke(self._build_messages(user_message, memory_context))
            content = self._content(response)
            self._record_usage(content, response)
        except Exception as exc:
            logger.error("Router LLM async call failed: {}", exc)
            return MultiRouteDecision(decisions=[
                RouteDecision(route="direct", confidence=0.0, reasoning=f"Router LLM error: {exc}")
            ])
        return self._parse_response(content)

    def _model_name(self) -> str:
        """Extract model name from the LLM for LangFuse metadata."""
        llm = getattr(self.llm, "runnable", self.llm)
        if hasattr(llm, "model_name"):
            return llm.model_name
        if hasattr(llm, "model"):
            return llm.model
        return "unknown"

    # ── parsing ───────────────────────────────────────────────

    def _parse_response(self, raw: str) -> MultiRouteDecision:
        """
        Parse the JSON response from the router LLM.

        Supports two formats:
          - Multi-route (new):  ``{"routes": [{...}, {...}]}``
          - Single-route (old): ``{"route": "crm", ...}``

        The old format is auto-wrapped into a single-element list
        for full backward compatibility.
        """
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # Locate JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            logger.warning("Router output is not JSON; falling back to direct.")
            return MultiRouteDecision(decisions=[
                RouteDecision(route="direct", confidence=0.0,
                              reasoning="Failed to parse router output as JSON.")
            ])

        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            logger.warning("Router JSON parse error: {}", exc)
            return MultiRouteDecision(decisions=[
                RouteDecision(route="direct", confidence=0.0,
                              reasoning=f"JSON parse error: {exc}")
            ])

        # ── Normalise to a list of route dicts ──────────────────
        if "routes" in data and isinstance(data["routes"], list):
            # New multi-route format
            route_dicts = data["routes"][:MAX_ROUTES]
        else:
            # Old single-route format — wrap in list
            route_dicts = [data]

        # ── Build RouteDecision objects ──────────────────────────
        decisions: List[RouteDecision] = []
        seen_routes: set = set()

        for rd in route_dicts:
            route = rd.get("route", "direct")
            if route not in VALID_ROUTES:
                logger.warning("Invalid route '{}'; skipping.", route)
                continue
            # Deduplicate (same route appearing twice)
            if route in seen_routes:
                continue
            seen_routes.add(route)

            action = rd.get("action")
            if route == "order" and action not in VALID_CRM_ACTIONS:
                logger.warning(
                    "Invalid order action '{}'; defaulting to lookup_order.", action
                )
                action = "lookup_order"

            decisions.append(RouteDecision(
                route=route,
                confidence=float(rd.get("confidence", 0.5)),
                reasoning=rd.get("reasoning", ""),
                action=action if route == "order" else None,
                params=rd.get("params", {}),
            ))

        # Fallback if nothing valid was parsed
        if not decisions:
            decisions = [RouteDecision(route="direct", confidence=0.0,
                                       reasoning="No valid routes parsed.")]

        return MultiRouteDecision(decisions=decisions)
