"""
Decision LangGraph — parallel classifiers + decision node.

Replaces the inline ``asyncio.gather(guardrail, route, cag)`` block
that used to live in the chat router. Same wall-clock latency
(parallel execution), but the routing decisions are now formalised
as graph nodes — easier to inspect on a Langfuse trace, easier to
extend with more parallel checks (toxicity, PII), and the chat
router shrinks to a single ``ainvoke`` call.

Topology::

    START
      ├── guardrail_node    (≈150 ms, Llama 3.1 8B)
      ├── router_node       (≈800 ms, Llama 3.3 70B)
      └── cag_node          (≈300 ms, KNN-1 Qdrant search)
              │ (fan-in — LangGraph waits for all three)
              ▼
          decide_node       (sets ``verdict``: out_of_scope | cache_hit | proceed)
              │
              ▼
             END

The slowest of the three classifiers sets the floor (≈800 ms — the
router). Guardrail and CAG hide behind it.

State writes are disjoint per node (each writes to its own field),
so LangGraph's default replace-reducer is fine without a custom
merger.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Dict, Literal, Optional, TypedDict

from loguru import logger
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from agents.guardrail import Guardrail
from agents.router import MultiRouteDecision, QueryRouter, RouteDecision


# ──────────────────────────────────────────────────────────────────
# State schema
# ──────────────────────────────────────────────────────────────────


GuardrailVerdict = Literal["in_scope", "out_of_scope"]
DecisionVerdict = Literal["out_of_scope", "cache_hit", "proceed"]


class DecisionState(TypedDict, total=False):
    """Mutable state passed between nodes.

    Inputs are filled by the chat router before ``ainvoke``.
    Each parallel node writes to a single dedicated key, so the
    default replace-reducer is safe (no concurrent writes to the
    same field). The ``decide`` node reads everything and produces
    the final ``verdict``.
    """

    # ── inputs ─────────────────────────────────────────────────
    message: str
    router_context: str

    # ── parallel node outputs ─────────────────────────────────
    guardrail: GuardrailVerdict
    decision: MultiRouteDecision
    cag_hit: Optional[Dict[str, Any]]

    # ── per-stage timings (filled by each node) ───────────────
    guardrail_ms: int
    route_ms: int
    cag_ms: int

    # ── final verdict (set by decide_node) ────────────────────
    verdict: DecisionVerdict
    primary_route: str  # convenience copy of decision.decisions[0].route


# Type alias for the SSE callback we pass through RunnableConfig.
# Signature mirrors the chat router's ``EmitFn``: a coroutine that
# accepts a JSON-serialisable event dict and returns None.
EmitFn = Callable[[Dict[str, Any]], Awaitable[None]]


# ──────────────────────────────────────────────────────────────────
# Node implementations
# ──────────────────────────────────────────────────────────────────


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _emit_from_config(config: Optional[RunnableConfig]) -> EmitFn:
    """Pull the per-request emit callback out of LangGraph's
    ``configurable`` dict. Returns a no-op if not provided so nodes
    can call ``await emit(...)`` unconditionally."""
    if config and (cfg := config.get("configurable")):
        fn = cfg.get("emit")
        if fn is not None:
            return fn

    async def _noop(_: Dict[str, Any]) -> None:
        return None

    return _noop


def _stage_label_safe(stage: str) -> str:
    """Friendly label lookup. Imports lazily so this module doesn't
    pull in api package symbols at import time (the orchestrator
    instantiates the graph during agent build, before the FastAPI
    app object exists)."""
    try:
        from api.event_labels import stage_label
        return stage_label(stage)
    except Exception:
        return stage.replace("_", " ").capitalize()


def make_guardrail_node(guardrail: Guardrail):
    """Closure factory: returns a node bound to the given Guardrail.

    The closure captures ``guardrail`` so the graph builder doesn't
    need to know about the orchestrator's instance state.
    """

    async def guardrail_node(
        state: DecisionState,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        emit = _emit_from_config(config)
        t0 = time.perf_counter()
        await emit({"type": "stage_start", "stage": "guardrail",
                    "label": _stage_label_safe("guardrail")})
        try:
            verdict = await guardrail.aclassify(state["message"])
        except Exception as exc:
            logger.warning("Guardrail node failed (defaulting in_scope): {}", exc)
            verdict = "in_scope"
        ms = _ms(t0)
        await emit({"type": "stage_done", "stage": "guardrail", "ms": ms,
                    "detail": {"verdict": verdict}})
        return {"guardrail": verdict, "guardrail_ms": ms}

    return guardrail_node


def make_router_node(router: QueryRouter):
    """Closure factory: router_node bound to the given QueryRouter."""

    async def router_node(
        state: DecisionState,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        emit = _emit_from_config(config)
        t0 = time.perf_counter()
        await emit({"type": "stage_start", "stage": "route",
                    "label": _stage_label_safe("route")})
        try:
            decision = await router.aroute(
                state["message"], state.get("router_context", "")
            )
        except Exception as exc:
            logger.warning("Router node failed (defaulting direct): {}", exc)
            decision = MultiRouteDecision(
                decisions=[RouteDecision(route="direct", confidence=0.0)]
            )
        ms = _ms(t0)
        primary = decision.decisions[0] if decision.decisions else None
        await emit({"type": "stage_done", "stage": "route", "ms": ms,
                    "detail": {
                        "route": primary.route if primary else "direct",
                        "action": primary.action if primary else None,
                        "reasoning": (primary.reasoning or "")[:160] if primary else "",
                    }})
        return {"decision": decision, "route_ms": ms}

    return router_node


def make_cag_node(cag_getter: Callable[[], Any]):
    """Closure factory: cag_node bound to a *getter* callable that
    returns the live CAG cache.

    Late-binding matters here: the FastAPI lifespan constructs the
    local-embedder CAG cache *after* the orchestrator is built, then
    attaches it to the orchestrator. Capturing a getter (rather than
    the cache itself) lets the graph compile early while still
    seeing the real cache at call time.
    """

    async def cag_node(
        state: DecisionState,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        import asyncio

        emit = _emit_from_config(config)
        t0 = time.perf_counter()
        await emit({"type": "stage_start", "stage": "cache",
                    "label": _stage_label_safe("cache")})
        cag = cag_getter()
        if cag is None:
            ms = _ms(t0)
            await emit({"type": "stage_done", "stage": "cache", "ms": ms,
                        "detail": {"hit": False, "skipped": "cag_unavailable"}})
            return {"cag_hit": None, "cag_ms": ms}
        try:
            hit = await asyncio.to_thread(cag.get, state["message"])
        except Exception as exc:
            logger.warning("CAG lookup node failed: {}", exc)
            hit = None
        ms = _ms(t0)
        await emit({"type": "stage_done", "stage": "cache", "ms": ms,
                    "detail": {"hit": hit is not None}})
        return {"cag_hit": hit, "cag_ms": ms}

    return cag_node


def decide_node(
    state: DecisionState,
    config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """Pure decision logic — no I/O, no awaiting.

    Reads the three classifier outputs and computes a single verdict:

      ``out_of_scope`` — guardrail rejected the message; chat router
            returns the templated refusal and skips tool + synth.

      ``cache_hit`` — CAG returned a hit AND the router chose a
            cache-eligible route (rag or direct). The chat router
            returns the cached answer directly.

      ``proceed`` — anything else; the chat router runs the chosen
            tool and the synth LLM.

    The route gate (``primary_route in {rag, direct}``) is the
    structural fix that prevents personal CRM queries from
    false-matching a generic FAQ at low cosine similarity. Without
    it, a question like "Do I have an appointment today?" could
    match the FAQ entry "How early should I arrive for my
    appointment?" and return the wrong answer; with the gate, CRM
    routes always run the tool against the patient's own data.
    """
    guardrail_v = state.get("guardrail", "in_scope")
    decision = state.get("decision")
    cag_hit = state.get("cag_hit")

    primary = (
        decision.decisions[0]
        if decision and decision.decisions
        else None
    )
    primary_route = primary.route if primary else "direct"

    if guardrail_v == "out_of_scope":
        verdict: DecisionVerdict = "out_of_scope"
    elif cag_hit is not None and primary_route in {"rag", "direct"}:
        verdict = "cache_hit"
    else:
        verdict = "proceed"

    return {"verdict": verdict, "primary_route": primary_route}


# ──────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────


def build_decision_graph(
    *,
    guardrail: Guardrail,
    router: QueryRouter,
    cag_getter: Callable[[], Any],
):
    """Build and compile the decision graph.

    Three classifier nodes fan out from START in parallel; the
    decide node fans them in and produces the verdict. Returns a
    compiled LangGraph runnable that the chat router invokes per
    request via ``ainvoke(state, config={"configurable": {"emit": ...}})``.

    ``cag_getter`` is a zero-arg callable returning the live CAG
    cache. The indirection is required because the cache is built
    by the FastAPI lifespan *after* the orchestrator (and therefore
    this graph) is constructed.
    """
    g = StateGraph(DecisionState)

    g.add_node("guardrail", make_guardrail_node(guardrail))
    g.add_node("router",    make_router_node(router))
    g.add_node("cag",       make_cag_node(cag_getter))
    g.add_node("decide",    decide_node)

    # Parallel start — all three classifiers begin at once.
    g.add_edge(START, "guardrail")
    g.add_edge(START, "router")
    g.add_edge(START, "cag")

    # Fan-in to decide — LangGraph waits for ALL incoming edges.
    g.add_edge("guardrail", "decide")
    g.add_edge("router",    "decide")
    g.add_edge("cag",       "decide")

    g.add_edge("decide", END)

    return g.compile()
