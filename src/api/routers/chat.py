"""
Conversational chat endpoints — concurrent hot path.

Architecture (no shortcuts, real LLM at every gate):

    POST /chat
        │
        │  ── Phase 1 — concurrent fan-out via asyncio.gather ─────────
        │       cag_lookup    (~150 ms — embed + Qdrant KNN-1)
        │       recall        (~250 ms — ST + LT semantic)
        │       router        (~250 ms — Groq Llama 3.1 8B JSON classify)
        │
        │  → CAG hit?  ────────►  cancel recall + route, store turn,
        │                          return cag_hit answer
        │
        │  → CAG miss
        │       │
        │       │  ── Phase 2 — sequential, depends on router ─────────
        │       │       tool dispatch (CRM | RAG | Web)   if route ≠ direct
        │       │       synthesise                          (Groq for direct,
        │       │                                            Gemini for tool routes)
        │       │
        │       │  ── Phase 3 — block on ST store, then return ───────
        │       │       persist turn pair    (~30 ms)
        │       │
        │       └──►  ChatResponse(answer, route, latency_ms, timings={…})
        │
        │  ── Phase 4 — BackgroundTasks (after response flushes) ──────
        │       maybe_distill   (LLM extraction, only on policy trigger)
        │       cag_set         (warm cache for similar future queries)

Every node's wall-clock is captured into the response's ``timings`` dict
so the UI/students can see exactly where the latency lives.
"""

import asyncio
import json
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from api.deps import get_agent, get_cag_cache, get_st_store
from api.event_labels import stage_label, tool_label
from api.schemas import (
    ChatRequest,
    ChatResetRequest,
    ChatResetResponse,
    ChatResponse,
    SessionTurnsResponse,
    SessionWarmupRequest,
    SessionWarmupResponse,
    TurnItem,
)


# Type alias for the event emitter passed through the pipeline.
EmitFn = Callable[[Dict[str, Any]], Awaitable[None]]


async def _noop_emit(_event: Dict[str, Any]) -> None:
    """Default emitter for the non-streaming path."""
    return None


router = APIRouter(tags=["Chat"])


# ── Helpers ──────────────────────────────────────────────────────────

def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _store_turn_pair(st_store, user_id: str, session_id: str, user_msg: str, answer: str) -> None:
    from memory.schemas import ConversationTurn
    now = time.time()
    st_store.add(user_id, session_id, ConversationTurn(
        user_id=user_id, session_id=session_id, role="user", content=user_msg, ts=now,
    ))
    st_store.add(user_id, session_id, ConversationTurn(
        user_id=user_id, session_id=session_id, role="assistant", content=answer, ts=now,
    ))


# ── Session cache helpers ───────────────────────────────────────────

# Maximum turns we keep warm in the in-memory session cache. Recall
# already budgets to ~4 turns, so 6 is plenty of headroom.
_CACHE_MAX_TURNS = 6


# CRM actions whose answers are customer-agnostic (depend only on the
# store's reference data, not on the user). These are safe to share
# across customers via the CAG semantic cache, exactly like RAG answers.
# Customer-specific actions (lookup_customer, create/cancel/reschedule)
# stay OFF this list — caching them would leak User A's data to User B.
_CACHEABLE_CRM_ACTIONS = {
    "list_specialties",
    "list_locations",
    "search_doctors",
}


def _cache_key(user_id: str, session_id: str):
    return (user_id, session_id)


def _append_to_cached_turns(app_state, user_id: str, session_id: str,
                             user_msg: str, answer: str) -> None:
    """Mirror the background DB write into the warm cache, so the next
    chat request finds an up-to-date ST context without a Supabase round-trip."""
    from memory.schemas import ConversationTurn

    cache = getattr(app_state, "session_cache", None)
    if cache is None:
        return
    entry = cache.get(_cache_key(user_id, session_id))
    if entry is None:
        return
    now = time.time()
    turns = list(entry.get("st_turns") or [])
    turns.append(ConversationTurn(
        user_id=user_id, session_id=session_id, role="user", content=user_msg, ts=now,
    ))
    turns.append(ConversationTurn(
        user_id=user_id, session_id=session_id, role="assistant", content=answer, ts=now,
    ))
    entry["st_turns"] = turns[-_CACHE_MAX_TURNS:]


# All previous string-match guards (fallback markers, clarification
# markers, personal-query markers, two-tier CAG threshold) were
# replaced by structural gates: the guardrail node decides scope, and
# the CAG hit gate (route ∈ {rag, direct}) prevents cross-intent
# false matches without any keyword heuristics.


_GREETING_PREFIX_RE = re.compile(
    r"^\s*(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))[ ,\-]+"
    r"[A-Z][a-zA-Z']*[ ,\-]*\n?",
    re.IGNORECASE,
)


def _strip_greeting_prefix(answer: str) -> str:
    """Remove a leading "Hello {name}," / "Hi {name}," etc. from a
    synthesised answer. Cached answers are shared across users, so
    they must be name-free — otherwise user A's name leaks into user
    B's reply when B's question hits the same cache entry.

    The body is preserved verbatim. We only strip the very first line
    and only if it matches the greeting-with-name pattern.
    """
    if not answer:
        return answer
    return _GREETING_PREFIX_RE.sub("", answer, count=1).lstrip()


def _safe_cag_set(cag, query: str, answer: str) -> None:
    """Background CAG write. ``cag.set`` raises on Qdrant transient
    errors; this wrapper logs and continues so a flaky write never
    fails a request. Fallback / clarification responses are filtered
    upstream (the chat path only calls this for cache-eligible routes
    with non-empty tool output, and ``cag_service.generate`` only
    writes when CRAG returned hits with sufficient confidence)."""
    sanitised = _strip_greeting_prefix(answer)
    try:
        cag.set(query, {"answer": sanitised, "evidence_urls": []})
    except Exception as exc:
        logger.warning("Background CAG set failed: {}", exc)


# How often automatic distillation fires. Distill is an LLM call +
# Qdrant upsert + LT dedup against existing facts — running it on every
# single turn was producing 1 fact per turn and getting deduped 95% of
# the time. We now distill at most every Nth turn unless the user
# explicitly says "remember"/"from now on", in which case we fire
# immediately because the intent is unambiguous.
_DISTILL_EVERY_N_TURNS = 4
_DISTILL_KEYWORDS = ("remember", "from now on", "remind me", "always", "never")
_distill_counters: Dict[tuple, int] = {}


def _maybe_distill(distiller, st_store, user_id: str, session_id: str) -> None:
    try:
        # Look at just the latest user+assistant pair (k=2) — anything
        # older has already been processed in a previous distill pass.
        # Less work for the dedup path, fewer redundant facts produced.
        recent = st_store.recent(user_id, session_id, k=2)
        if not recent:
            return

        explicit = any(
            any(kw in (t.content or "").lower() for kw in _DISTILL_KEYWORDS)
            for t in recent
        )

        if not explicit:
            key = (user_id, session_id)
            count = _distill_counters.get(key, 0) + 1
            _distill_counters[key] = count
            # Only fire on every Nth turn (4, 8, 12, …)
            if count % _DISTILL_EVERY_N_TURNS != 0:
                return

        if distiller.should_distill(recent):
            distiller.distill(user_id, recent)
    except Exception as exc:
        logger.warning("Background distill failed: {}", exc)


def _system_prompt_for(route: str) -> str:
    from agents.prompts.agent_prompts import LANGFUSE_PROMPT_NAMES, _SYNTHESISER_SYSTEM_FALLBACK
    from infrastructure.observability import fetch_prompt
    
    return fetch_prompt(
        LANGFUSE_PROMPT_NAMES["synthesiser_system"],
        fallback=_SYNTHESISER_SYSTEM_FALLBACK,
    )


# Hard-coded grounding guard appended to every synth system prompt.
# This is the *thin* version: the architecture (router, guardrail,
# route-gated CAG) handles intent classification. Synth's only job
# is to compose a reply from TOOL OUTPUT — these rules just keep it
# honest. Pure greetings never reach this prompt because the router
# routes them through ``direct``, which has its own concierge persona.
STYLE_OVERRIDE = (
    "=== GROUNDING (HARD CONSTRAINTS) ===\n"
    "\n"
    "1. Language: clear professional English. Never use Sanskrit / Hindi /\n"
    "   Tamil greetings (Namaskaram, Namaste, Vanakkam, Ayubowan).\n"
    "\n"
    "2. The customer is already authenticated. Never ask for identity\n"
    "   info that USER CONTEXT already lists (name, email, phone, etc.).\n"
    "\n"
    "3. SOURCES OF TRUTH (priority order):\n"
    "   a. TOOL OUTPUT — the only valid source for orders, refund amounts,\n"
    "      disputes, tracking, and policies.\n"
    "   b. USER CONTEXT — the customer's profile.\n"
    "   c. MEMORY CONTEXT — what was said, not a fact database.\n"
    "\n"
    "4. ZERO HALLUCINATION: never invent order numbers, refund amounts,\n"
    "   tracking numbers, or policies. If TOOL OUTPUT is exactly '(no tool output)'\n"
    "   you have no data — say so plainly: 'I don't have that on file.'\n"
    "   HOWEVER, if TOOL OUTPUT indicates a lookup failed (e.g., 'No order found'), you MUST politely ask the user for their phone number or order ID.\n"
    "\n"
    "5. STATUS HONESTY: each order in TOOL OUTPUT carries a Status.\n"
    "   Only valid active orders are subject to dispute resolution.\n"
    "\n"
    "6. TABLES: when TOOL OUTPUT contains a markdown table, preserve it\n"
    "   verbatim — the UI renders it.\n"
    "\n"
    "7. ERROR TRANSPARENCY: when TOOL OUTPUT contains a failure message,\n"
    "   relay the real reason instead of hiding it behind technical excuses.\n"
)


def _customer_to_dict(p) -> Dict[str, Any]:
    """Stable, plain-Python view of a Customer ORM row."""
    return {
        "customer_id": p.customer_id,
        "full_name": p.full_name,
        "phone": p.phone or "",
        "dob": p.dob or "",
        "gender": p.gender or "",
        "email": p.email or "",
        "notes": p.notes or "",
    }


def _fetch_customer_sync(customer_id: str) -> Optional[Dict[str, Any]]:
    """Indexed PK lookup against customers. Off-loaded to a worker thread."""
    if not customer_id or customer_id == "UNKNOWN":
        return None
    try:
        from infrastructure.db.supabase_client import get_supabase_client
        db = get_supabase_client()
        res = db.table("customers").select("*").eq("id", customer_id).single().execute()
        c = res.data
        if not c:
            return None
        return {
            "customer_id": c.get("id"),
            "full_name": c.get("name"),
            "phone": c.get("external_user_id"),
            "email": c.get("email"),
            "tier": c.get("tier"),
            "notes": f"Total Orders: {c.get('total_orders')} | Disputes: {c.get('dispute_count')}",
        }
    except Exception:
        return None


def _fetch_recent_orders_sync(
    customer_id: str, *, limit: int = 8,
) -> List[Dict[str, Any]]:
    """Return recent orders for the customer context."""
    if not customer_id or customer_id == "UNKNOWN":
        return []
    try:
        from infrastructure.db.supabase_client import get_supabase_client
        db = get_supabase_client()
        res = db.table("orders").select("id, order_number, status, purchase_date, items, total_amount, currency").eq("customer_id", customer_id).order("purchase_date", desc=True).limit(limit).execute()
        orders = []
        for o in (res.data or []):
            items_str = ", ".join(f"{i.get('name', 'Item')} x{i.get('qty', 1)}" for i in (o.get("items") or []))
            orders.append({
                "order_id": o.get("order_number", o.get("id")),
                "when": o.get("purchase_date", "Unknown")[:10],
                "items": items_str,
                "total_amount": f"{o.get('currency', 'LKR')} {o.get('total_amount', 0):.2f}",
                "status": o.get("status", "Unknown"),
            })
        return orders
    except Exception:
        return []


# Regex used to extract order_id values from CRM tool output. The
# CRM tool renders order ids inline as `B...` markdown code spans
# (e.g. ``Your order ID is `B4d7a1b...```) and inside disambiguation
# tables that include a "Order ID" column. This pattern matches both.
_BOOKING_ID_RE = re.compile(
    r"`?(?P<id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})`?",
    re.IGNORECASE,
)


def _extract_order_ids(tool_output: str) -> List[str]:
    """Parse order_id values from CRM tool output, in order of
    appearance. Used to capture *what was just shown to the user* so
    the next turn's router can resolve referential demonstratives
    ("this", "it") to a concrete id."""
    if not tool_output:
        return []
    seen: List[str] = []
    for m in _BOOKING_ID_RE.finditer(tool_output):
        bid = m.group("id")
        if bid not in seen:
            seen.append(bid)
    return seen


def _user_context_block(
    customer: Optional[Dict[str, Any]],
    fallback_customer_id: str,
    session_id: str,
) -> str:
    """
    Build the per-request USER CONTEXT preface.

    Includes:
      - today's date + local time in the store timezone (for "today",
        "this week" comparisons),
      - the **full** authenticated customer profile (name, phone, DOB,
        gender, email, notes) — so the synth LLM never asks for
        information that was already collected at login,
      - an explicit instruction reminding the model not to re-ask.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        from infrastructure.config import TIMEZONE
    except Exception:
        TIMEZONE = "Asia/Colombo"

    now = datetime.now(ZoneInfo(TIMEZONE))
    head = (
        "=== USER CONTEXT ===\n"
        f"- Today is {now.strftime('%A, %Y-%m-%d')} ({TIMEZONE}); "
        f"current local time {now.strftime('%H:%M')}.\n"
        f"- Conversation session_id is {session_id}.\n"
    )

    if customer:
        identity = (
            "- The authenticated customer profile (already on file — "
            "DO NOT ASK FOR ANY OF THESE):\n"
            f"    name:       {customer.get('full_name') or '(unknown)'}\n"
            f"    phone:      {customer.get('phone') or '(none)'}\n"
            f"    dob:        {customer.get('dob') or '(none)'}\n"
            f"    gender:     {customer.get('gender') or '(none)'}\n"
            f"    email:      {customer.get('email') or '(none)'}\n"
            f"    notes:      {customer.get('notes') or '(none)'}\n"
            f"    customer_id: {customer.get('customer_id')}\n"
            "- Use the customer's first name when greeting them.\n"
        )
    else:
        identity = (
            f"- user_id is {fallback_customer_id} but the profile row "
            "could not be loaded. Since you don't have their details, YOU MUST ASK the user for their phone number or order ID if you need to look up their orders.\n"
        )

    return head + identity


def _current_trace_id() -> Optional[str]:
    try:
        from infrastructure.observability import get_langfuse
        client = get_langfuse()
        if client is None:
            return None
        tracer = getattr(client, "tracer", None)
        span = getattr(tracer, "current_span", None) if tracer else None
        return getattr(span, "trace_id", None) if span else None
    except Exception:
        return None
async def _dispatch_tool(agent, decision, *, customer_id: str, fallback_query: str = "") -> str:
    route = decision.route
    params = dict(decision.params or {})

    if route in ("order", "crm") and agent.crm_tool is not None:
        action = decision.action or "lookup_order"
        # Self-referential order actions get the logged-in customer's id
        if action in {"lookup_order", "lookup_customer", "submit_dispute", "list_disputes"}:
            params.setdefault("customer_id", customer_id)
            params.setdefault("customer_id", customer_id)
        return await asyncio.to_thread(agent.crm_tool.dispatch, action, params)

    if route == "rag" and agent.rag_tool is not None:
        if not params.get("query") and fallback_query:
            params["query"] = fallback_query
        return await asyncio.to_thread(agent.rag_tool.dispatch, "search", params)

    if route == "web_search" and agent.web_tool is not None:
        if not params.get("query") and fallback_query:
            params["query"] = fallback_query
        return await asyncio.to_thread(agent.web_tool.dispatch, "search", params)

    return ""


# ── Shared chat pipeline ─────────────────────────────────────────────
#
# Both the non-streaming ``POST /chat`` endpoint and the streaming
# ``POST /chat/stream`` endpoint call ``_run_chat_pipeline``. The only
# difference is the ``emit`` callback they pass:
#
#   - ``/chat`` passes ``_noop_emit`` (events are discarded).
#   - ``/chat/stream`` passes a queue-pushing emit and SSE-streams the
#     events to the client.
#
# Every phase boundary in the pipeline calls ``await emit(event)`` so
# the streaming client can render a live chain-of-thought timeline.
async def _run_chat_pipeline(
    req: ChatRequest,
    *,
    request: Request,
    background: BackgroundTasks,
    emit: EmitFn,
) -> ChatResponse:
    t_total = time.perf_counter()
    timings: Dict[str, int] = {}
    model_used: Optional[str] = None

    agent = get_agent(request)
    cag = get_cag_cache(request)
    st_store = get_st_store(request)

    # ── Phase 1 — concurrent fan-out ─────────────────────────────
    # CAG lookup, memory recall and router classification all start at
    # the same time. The router runs on the empty memory-context path —
    # for the queries it actually matters on (greetings, simple Q&A) the
    # context doesn't affect routing, and the latency win is large. The
    # full memory_context still gets fed into the synthesiser below.

    # NOTE: the previous ``_cag_task`` / ``_guardrail_task`` /
    # ``_route_task`` inline coroutines have moved into the decision
    # LangGraph (see ``agents/decision_graph.py``). They live there
    # as graph nodes — guardrail_node, router_node, cag_node — that
    # fan out from START in parallel and fan in to a decide_node.
    # The chat router invokes the graph once via ``ainvoke`` instead
    # of building three asyncio tasks.

    async def _recall_st_task():
        """ST recall only — fast (one Supabase round-trip, no embed)."""
        t0 = time.perf_counter()
        await emit({"type": "stage_start", "stage": "recall_st",
                    "label": stage_label("recall_st")})
        try:
            st_turns = await asyncio.to_thread(
                agent.st_store.recent, req.user_id, req.session_id, 6,
            )
            ctx = agent.recaller.format_context(st_turns)
        except Exception as exc:
            logger.warning("ST recall failed: {}", exc)
            ctx = ""
        ms = _ms(t0)
        timings["recall_st"] = ms
        await emit({"type": "stage_done", "stage": "recall_st", "ms": ms})
        return ctx

    async def _recall_lt_async() -> str:
        """LT semantic recall — only run when the route can use it."""
        t0 = time.perf_counter()
        await emit({"type": "stage_start", "stage": "recall_lt",
                    "label": stage_label("recall_lt")})
        try:
            from infrastructure.config import LT_SIM_THRESHOLD
            lt_facts = await asyncio.to_thread(
                agent.lt_store.query,
                user_id=req.user_id,
                query_text=req.message,
                k=5,
                threshold=LT_SIM_THRESHOLD,
            )
            if not lt_facts:
                ms = _ms(t0)
                timings["recall_lt"] = ms
                await emit({"type": "stage_done", "stage": "recall_lt", "ms": ms,
                            "detail": {"facts": 0}})
                return ""
            ctx = "\n=== LONG-TERM FACTS ===\n"
            for f in lt_facts:
                ctx += f"- {getattr(f, 'text', '')}\n"
        except Exception as exc:
            logger.warning("LT recall failed: {}", exc)
            ctx = ""
        ms = _ms(t0)
        timings["recall_lt"] = ms
        await emit({"type": "stage_done", "stage": "recall_lt", "ms": ms,
                    "detail": {"facts": len(lt_facts) if 'lt_facts' in locals() and lt_facts else 0}})
        return ctx

    # Build a tiny router context from the warm session cache (if any).
    # The router runs in parallel with CAG and recall, so we cannot wait
    # on the live ST fetch — but the warm cache already holds the last
    # 6 turns, and that's exactly what helps the router interpret
    # follow-ups like "Yeah, I made it on Monday" in the context of the
    # previous order attempt. Empty when cache is cold.
    cache_for_router = request.app.state.session_cache.get(
        _cache_key(req.user_id, req.session_id)
    )
    recent_turns = []
    if cache_for_router is not None:
        recent_turns = (cache_for_router.get("st_turns") or [])[-4:]
    if not recent_turns:
        # Cold warm-cache: a DIFFERENT api replica handled the previous turn
        # (the warm cache is per-task, in-memory), or no warmup ran. Without
        # this fallback the router fires with ZERO history and can't resolve
        # follow-ups — "what are their timings?", "cancel that one" — so it
        # misroutes them. The cache is a latency optimisation; this fetch
        # guarantees the router always sees the conversation, so memory works
        # no matter which replica serves the turn.
        try:
            live = await asyncio.to_thread(
                st_store.recent, req.user_id, req.session_id, 4
            )
            recent_turns = live or []
        except Exception as _e:  # noqa: BLE001
            logger.debug(f"router-context fallback fetch failed: {_e}")
            recent_turns = []
    router_context = (
        agent.recaller.format_context(recent_turns) if recent_turns else ""
    )

    # Deterministic customer-profile hint. Two structured blocks are
    # injected ahead of the recent ST turns:
    #
    #   CUSTOMER PROFILE — every recent non-cancelled order, oldest
    #   first. The router sees the *full* set of candidates so it can
    #   resolve "reschedule this" without biasing toward any one row.
    #
    #   RECENTLY SHOWN — the order_ids that appeared in the *previous*
    #   bot turn's CRM output. When the user uses a referential
    #   demonstrative ("this", "it", "the one"), this is the answer.
    recent = (cache_for_router or {}).get("recent_orders")
    if recent is None:
        try:
            recent = await asyncio.to_thread(_fetch_recent_orders_sync, req.user_id)
            if cache_for_router is not None:
                cache_for_router["recent_orders"] = recent
        except Exception:
            recent = []

    profile_block_parts: List[str] = []

    # 1) RECENTLY SHOWN — what the user just looked at
    visible_ids: List[str] = (cache_for_router or {}).get("visible_orders") or []
    if visible_ids and recent:
        # Render only the recent rows whose ids match what was
        # displayed last turn. Keeps the block tight and aligned to
        # the user's referent.
        by_id = {b["order_id"]: b for b in recent}
        shown = [by_id[i] for i in visible_ids if i in by_id]
        if shown:
            lines = [
                f"- order_id={b['order_id']}  {b['when']}  "
                f"with {b['items']} ({b['total_amount']}) [{b['status']}]"
                for b in shown
            ]
            profile_block_parts.append(
                "=== RECENTLY SHOWN ===\n"
                "These are the orders the user just saw in the previous turn. "
                "When the user says \"this\" / \"it\" / \"the one\" in a CRM "
                "request, the referent is in this list — pass the matching "
                "order_id explicitly.\n"
                + "\n".join(lines)
                + "\n"
            )

    # 2) CUSTOMER PROFILE — the full recent list (deterministic
    # source of truth for availability defaults + reschedule targets)
    if recent:
        lines = [
            f"- order_id={b['order_id']}  {b['when']}  "
            f"with {b['items']} ({b['total_amount']}) [{b['status']}]"
            for b in recent
        ]
        # Default total_amount: use the soonest recent order's total_amount
        default_total_amount = recent[0].get("total_amount")
        profile_block_parts.append(
            "=== CUSTOMER PROFILE ===\n"
            f"Upcoming orders ({len(recent)} total, soonest first):\n"
            + "\n".join(lines) + "\n"
            f"Default total_amount for availability questions: {default_total_amount}\n"
        )

    if profile_block_parts:
        router_context = "\n".join(profile_block_parts) + "\n" + (router_context or "")

    # NOTE: the previous ``_route_task`` inline coroutine has moved
    # into the decision LangGraph (see ``router_node`` in
    # ``agents/decision_graph.py``). Routing now happens inside the
    # graph, alongside guardrail and cag, fanning into the decide
    # node that produces the verdict the chat router branches on.

    async def _customer_task():
        """Fetch the authenticated customer's full profile in parallel.

        The lookup is an indexed PK query — runs alongside CAG/recall/
        router, so its ~300-500 ms RTT (Sri Lanka → Supabase US-East)
        fits inside the existing parallel window without adding to
        wall-clock latency on the chat hot path.
        """
        t0 = time.perf_counter()
        await emit({"type": "stage_start", "stage": "customer",
                    "label": stage_label("customer")})
        try:
            data = await asyncio.to_thread(_fetch_customer_sync, req.user_id)
        except Exception as exc:
            logger.warning("Customer profile fetch failed: {}", exc)
            data = None
        ms = _ms(t0)
        timings["customer"] = ms
        await emit({"type": "stage_done", "stage": "customer", "ms": ms,
                    "detail": {"loaded": data is not None}})
        return data

    # ── Warm-cache short-circuit for ST + customer ───────────────
    # When the session was preloaded via /sessions/warmup, we already
    # hold the customer profile and the most recent ST turns in memory.
    # That removes two Supabase round-trips (~600-1000 ms from Sri Lanka)
    # from every chat. Fall through to live fetches when the cache is
    # cold (first request before warmup, after /chat/reset, etc.).
    cache_entry = request.app.state.session_cache.get(
        _cache_key(req.user_id, req.session_id)
    )
    # ── Phase 1 — parallel classifiers via the decision LangGraph ─
    # Three classifier nodes (guardrail, router, cag) fan out from
    # START in parallel; a decide_node fans them in and produces a
    # single ``verdict`` (out_of_scope | cache_hit | proceed). Same
    # wall-clock latency as the previous asyncio.gather (parallel
    # execution, slowest sets the floor at ~800 ms), but the routing
    # decisions are now formalised as graph nodes — easier to inspect
    # on a Langfuse trace and trivial to extend with more parallel
    # checks later. See ``agents/decision_graph.py`` for the topology.
    decision_graph_task = asyncio.create_task(
        agent.decision_graph.ainvoke(
            {
                "message": req.message,
                "router_context": router_context,
            },
            config={"configurable": {"emit": emit}},
        )
    )

    # Customer lookup + ST recall stay outside the decision graph —
    # they're preparation tasks (only used if verdict=proceed) and
    # they need the warm-cache short-circuit logic that's specific to
    # the request session. Run them in parallel with the graph.
    if cache_entry is not None:
        cached_customer = cache_entry.get("customer")
        cached_turns = cache_entry.get("st_turns") or []
        st_context_cached = agent.recaller.format_context(cached_turns)
        timings["recall_st"] = 0
        timings["customer"] = 0

        async def _customer_task():
            await emit({"type": "stage_done", "stage": "customer", "ms": 0,
                        "detail": {"cached": True, "loaded": cached_customer is not None}})
            return cached_customer

        async def _recall_st_task():
            await emit({"type": "stage_done", "stage": "recall_st", "ms": 0,
                        "detail": {"cached": True, "turns": len(cached_turns)}})
            return st_context_cached

        recall_task = asyncio.create_task(_recall_st_task())
        customer_task = asyncio.create_task(_customer_task())
    else:
        recall_task = asyncio.create_task(_recall_st_task())
        customer_task = asyncio.create_task(_customer_task())

    # Wait for the decision graph to finish — its decide_node has
    # already aggregated the three signals into a single verdict.
    decision_state = await decision_graph_task
    guardrail_verdict = decision_state.get("guardrail", "in_scope")
    decision = decision_state.get("decision")
    cag_hit = decision_state.get("cag_hit")
    timings["guardrail"] = decision_state.get("guardrail_ms", 0)
    timings["route"] = decision_state.get("route_ms", 0)
    timings["cag"] = decision_state.get("cag_ms", 0)

    # ── Phase 1a — Guardrail short-circuit (out-of-domain query) ──
    # Out-of-scope queries return a templated refusal: no router LLM,
    # no tool call, no synth. This is what stops "who is the president
    # of the USA" from ever reaching Tavily.
    if guardrail_verdict == "out_of_scope":
        from agents.guardrail import OUT_OF_SCOPE_REPLY
        # Cancel concurrent work we no longer need (route already
        # awaited above; recall + customer may still be running).
        recall_task.cancel()
        customer_task.cancel()
        for t in (recall_task, customer_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        await emit({"type": "tool_invoke", "route": "out_of_scope",
                    "action": None, "label": tool_label("out_of_scope")})
        await emit({"type": "tool_done", "route": "out_of_scope",
                    "action": None, "ms": 0,
                    "summary": "domain guardrail declined"})

        _append_to_cached_turns(
            request.app.state, req.user_id, req.session_id,
            req.message, OUT_OF_SCOPE_REPLY,
        )
        background.add_task(
            _store_turn_pair, st_store, req.user_id, req.session_id,
            req.message, OUT_OF_SCOPE_REPLY,
        )
        from api.routers.chat_sessions import touch_session_sync as _touch
        from api.routers.chat_sessions import maybe_auto_title_sync as _autotitle
        background.add_task(_touch, req.user_id, req.session_id)
        # Auto-title — fires once per session after enough turns.
        background.add_task(
            _autotitle,
            session_id=req.session_id,
            user_id=req.user_id,
            st_store=st_store,
            llm=getattr(agent, "llm_fast", None) or agent.llm_chat,
        )
        timings["save"] = 0

        return ChatResponse(
            answer=OUT_OF_SCOPE_REPLY,
            route="out_of_scope",
            routes=["out_of_scope"],
            cached=False,
            latency_ms=_ms(t_total),
            trace_id=_current_trace_id(),
            timings=timings,
            model_used="guardrail",
        )

    # ── Phase 1b — CAG hit short-circuit ─────────────────────────
    # Only route-eligible cache hits return cached. The cache holds
    # generic FAQ-style answers; using a hit when the router picked
    # ``crm`` or ``web`` would risk returning a stale FAQ in place of
    # the customer's own data or a live external lookup. Letting the
    # router gate this kills the personal-query false-match class
    # (e.g. "Do I have an appointment today?" matching a generic FAQ
    # at 0.70 cosine) without any string heuristics.
    _primary = (decision.decisions[0] if decision and decision.decisions else None)
    _route = (_primary.route if _primary else "direct")
    cache_eligible = cag_hit is not None and _route in {"rag", "direct"}

    if cache_eligible:
        # Cancel concurrent work we no longer need (route already
        # awaited above; recall + customer may still be running).
        recall_task.cancel()
        customer_task.cancel()
        for t in (recall_task, customer_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        answer = cag_hit.get("answer", "") or ""
        await emit({"type": "tool_invoke", "route": "cag_hit", "action": None,
                    "label": tool_label("cag_hit")})
        await emit({"type": "tool_done", "route": "cag_hit", "action": None,
                    "ms": 0, "summary": "answer served from semantic cache"})
        # Mirror the turn into the warm cache + background DB write
        _append_to_cached_turns(
            request.app.state, req.user_id, req.session_id, req.message, answer,
        )
        background.add_task(
            _store_turn_pair, st_store, req.user_id, req.session_id, req.message, answer,
        )
        from api.routers.chat_sessions import touch_session_sync as _touch
        from api.routers.chat_sessions import maybe_auto_title_sync as _autotitle
        background.add_task(_touch, req.user_id, req.session_id)
        # Auto-title — fires once per session after enough turns.
        background.add_task(
            _autotitle,
            session_id=req.session_id,
            user_id=req.user_id,
            st_store=st_store,
            llm=getattr(agent, "llm_fast", None) or agent.llm_chat,
        )
        timings["save"] = 0

        return ChatResponse(
            answer=answer,
            route="cag_hit",
            routes=["cag_hit"],
            cached=True,
            latency_ms=_ms(t_total),
            trace_id=_current_trace_id(),
            timings=timings,
            model_used="cache",
        )

    # ── CAG miss — collect the remaining ST recall + customer profile
    # The router was already awaited above for the cache-gate decision,
    # so we only need ST + customer now.
    st_context, customer = await asyncio.gather(recall_task, customer_task)

    decisions = decision.decisions or []
    if not decisions:
        from agents.router import RouteDecision
        decisions = [RouteDecision(route="direct", confidence=0.0)]
    primary = decisions[0]
    routes_taken: List[str] = [d.route for d in decisions]
    primary_route_label = "multi" if len(set(routes_taken)) > 1 else primary.route

    # ── Phase 1.5 + 2 — LT recall AND tool dispatch run in parallel ─
    # Both stages depend on the router result but are independent of
    # each other. Running them concurrently saves ~max(lt, tool) on
    # tool-backed routes — typically 1-1.5 s on slow links.
    needs_lt = (len(decisions) > 1) or (primary.route != "direct")
    needs_tool = (len(decisions) > 1) or (primary.route != "direct")

    async def _do_lt():
        if not needs_lt:
            timings["recall_lt"] = 0
            return ""
        return await _recall_lt_async()

    async def _dispatch_one(d) -> str:
        # Per-decision wrapper that emits friendly tool_invoke / tool_done
        # so the chain-of-thought UI shows "Looking up your appointments
        # in the CRM" instead of "running tool".
        t0 = time.perf_counter()
        await emit({
            "type": "tool_invoke",
            "route": d.route,
            "action": d.action,
            "label": tool_label(d.route, d.action),
        })
        out = await _dispatch_tool(
            agent, d,
            customer_id=req.user_id,
            fallback_query=req.message,
        )
        ms = _ms(t0)
        # Best-effort summary: first non-empty line of the tool output,
        # capped so the chip stays one line in the UI.
        first_line = ""
        if out:
            for line in (out or "").splitlines():
                if line.strip():
                    first_line = line.strip()[:120]
                    break
        await emit({
            "type": "tool_done",
            "route": d.route,
            "action": d.action,
            "ms": ms,
            "summary": first_line,
        })
        return out

    async def _do_tools():
        if not needs_tool:
            timings["tool"] = 0
            return [""]
        t0 = time.perf_counter()
        try:
            if len(decisions) > 1:
                outs = await asyncio.gather(*[_dispatch_one(d) for d in decisions])
            else:
                outs = [await _dispatch_one(primary)]
        finally:
            timings["tool"] = _ms(t0)
        return outs

    lt_context, tool_outputs = await asyncio.gather(_do_lt(), _do_tools())
    memory_context = (st_context or "") + (lt_context or "")

    # Combine tool outputs for the synthesiser (labelled by route)
    if any(tool_outputs):
        combined_tool_output = "\n\n".join(
            f"=== {d.route.upper()} RESULT ===\n{out}"
            for d, out in zip(decisions, tool_outputs) if out
        )
    else:
        combined_tool_output = ""
        
    with open("debug_tool_output.txt", "w") as f:
        f.write(combined_tool_output or "EMPTY")

    # Capture the order_ids that were just shown to the user so the
    # *next* turn's router can resolve referential demonstratives
    # ("reschedule this", "cancel it") to a concrete order. This is
    # the structural fix for the wrong-order-rescheduled bug: the
    # router stops guessing from "most recent", and instead consults
    # the exact list of rows the user can see in the chat.
    cache_for_router_post = request.app.state.session_cache.get(
        _cache_key(req.user_id, req.session_id)
    )
    if cache_for_router_post is not None:
        crm_outputs = [
            out for d, out in zip(decisions, tool_outputs)
            if out and d.route in ("crm", "order")
        ]
        if crm_outputs:
            ids = _extract_order_ids("\n".join(crm_outputs))
            if ids:
                cache_for_router_post["visible_orders"] = ids
        # Also invalidate the recent-orders cache after any
        # mutating CRM action so the next turn re-reads fresh state.
        if any(d.route in ("crm", "order") and d.action in
               {"create_order", "cancel_order", "reschedule_order"}
               for d in decisions):
            cache_for_router_post.pop("recent_orders", None)

    # ── Phase 3 — synthesise ─────────────────────────────────────
    # Direct route → fast Groq model. Tool routes → flagship Gemini for
    # quality. Multi-route uses Gemini too (it's merging multiple sources).
    t0 = time.perf_counter()
    use_fast = (len(decisions) == 1 and primary.route == "direct")
    synth_llm = agent.llm_fast if use_fast else agent.llm_chat
    _synth_llm_inst = getattr(synth_llm, "runnable", synth_llm)
    model_used = getattr(_synth_llm_inst, "model_name", None) or getattr(_synth_llm_inst, "model", "unknown")

    system_prompt = _system_prompt_for(primary.route)
    user_context = _user_context_block(customer, req.user_id, req.session_id)
    system_content = (
        f"{system_prompt}\n\n"
        f"{user_context}\n"
        f"{STYLE_OVERRIDE}\n"
        f"=== MEMORY CONTEXT ===\n{memory_context}\n\n"
        f"=== TOOL OUTPUT ===\n{combined_tool_output or '(no tool output)'}"
    )
    logger.info(f"LLM SYSTEM CONTENT:\n{system_content}")

    await emit({
        "type": "stage_start",
        "stage": "synth",
        "label": stage_label("synth"),
        "detail": {"model": model_used},
    })
    
    logger.info(f"=== SYNTH PROMPT ===\nSystem: {system_content}\nUser: {req.message}\n===================")
    
    try:
        response = await synth_llm.ainvoke([
            SystemMessage(content=system_content),
            HumanMessage(content=req.message),
        ])
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.exception("Synth LLM failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Synth error: {exc}")
    timings["synth"] = _ms(t0)
    await emit({
        "type": "stage_done",
        "stage": "synth",
        "ms": timings["synth"],
        "detail": {"model": model_used},
    })

    # ── Phase 4 — fire-and-forget writes + warm-cache write-through ─
    # Background tasks: ST store and distill always run. The CAG cache
    # is populated ONLY for routes whose answers are customer-agnostic
    # (RAG = store KB / FAQs). Direct and CRM answers are personal —
    # caching them poisons the cache: User B's lookup would return
    # User A's answer. Web is too time-sensitive to cache.
    if answer:
        _append_to_cached_turns(
            request.app.state, req.user_id, req.session_id, req.message, answer,
        )
        background.add_task(
            _store_turn_pair, st_store, req.user_id, req.session_id, req.message, answer,
        )
        background.add_task(_maybe_distill, agent.distiller, st_store, req.user_id, req.session_id)
        # Mark session as recently active (creates the row if missing).
        from api.routers.chat_sessions import touch_session_sync as _touch
        from api.routers.chat_sessions import maybe_auto_title_sync as _autotitle
        background.add_task(_touch, req.user_id, req.session_id)
        # Auto-title — fires once per session after enough turns.
        background.add_task(
            _autotitle,
            session_id=req.session_id,
            user_id=req.user_id,
            st_store=st_store,
            llm=getattr(agent, "llm_fast", None) or agent.llm_chat,
        )
        # CAG-safe routes: single-route RAG (storeal KB) OR single-route
        # CRM where the action is customer-agnostic (store reference).
        cag_safe = (
            len(decisions) == 1 and (
                primary.route == "rag"
                or (primary.route == "crm" and primary.action in _CACHEABLE_CRM_ACTIONS)
            )
        )
        if cag_safe:
            background.add_task(_safe_cag_set, cag, req.message, answer)
    timings["save"] = 0  # background — user does not wait

    return ChatResponse(
        answer=answer,
        route=primary_route_label,
        routes=routes_taken,
        cached=False,
        latency_ms=_ms(t_total),
        trace_id=_current_trace_id(),
        timings=timings,
        model_used=model_used,
    )


# ── POST /chat — non-streaming (sync contract) ──────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    background: BackgroundTasks,
) -> ChatResponse:
    """Run the pipeline and return the full response in one shot."""
    return await _run_chat_pipeline(
        req, request=request, background=background, emit=_noop_emit,
    )


# ── POST /chat/stream — chain-of-thought via Server-Sent Events ─────

@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    background: BackgroundTasks,
) -> StreamingResponse:
    """
    Stream the chain of thought as the pipeline runs.

    The transport is SSE — one ``data: {…}\\n\\n`` event per phase
    transition, ending with a ``{"type":"final", …}`` payload that
    contains the same fields as the non-streaming ``/chat`` response.
    Clients that just want the answer can ignore everything except the
    final event.

    Same backend logic as ``/chat``; only the emit callback differs.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event: Dict[str, Any]) -> None:
        await queue.put(event)

    async def run() -> None:
        try:
            final = await _run_chat_pipeline(
                req, request=request, background=background, emit=emit,
            )
            await queue.put({
                "type": "final",
                "answer": final.answer,
                "route": final.route,
                "routes": final.routes,
                "cached": final.cached,
                "latency_ms": final.latency_ms,
                "trace_id": final.trace_id,
                "timings": final.timings,
                "model_used": final.model_used,
            })
        except HTTPException as exc:
            await queue.put({"type": "error", "status": exc.status_code,
                             "message": str(exc.detail)})
        except Exception as exc:
            logger.exception("Streaming chat failed: {}", exc)
            await queue.put({"type": "error", "status": 500, "message": str(exc)})
        finally:
            await queue.put(None)  # sentinel — closes the stream

    asyncio.create_task(run())

    async def event_generator():
        # Initial keep-alive comment so proxies open the stream immediately.
        yield ": stream-open\n\n"
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache",
            "x-accel-buffering": "no",  # disable proxy buffering
            "connection": "keep-alive",
        },
    )


# ── POST /chat/reset ─────────────────────────────────────────────────

@router.post("/chat/reset", response_model=ChatResetResponse)
async def chat_reset(
    req: ChatResetRequest,
    request: Request,
    st_store=Depends(get_st_store),
) -> ChatResetResponse:
    """Clear the short-term memory for a single ``(user_id, session_id)`` pair."""
    await asyncio.to_thread(st_store.clear, req.user_id, req.session_id)
    request.app.state.session_cache.pop(_cache_key(req.user_id, req.session_id), None)
    return ChatResetResponse(cleared=True, user_id=req.user_id, session_id=req.session_id)


# ── POST /sessions/warmup ────────────────────────────────────────────

@router.post("/sessions/warmup", response_model=SessionWarmupResponse)
async def session_warmup(
    req: SessionWarmupRequest,
    request: Request,
) -> SessionWarmupResponse:
    """
    Preload the customer profile + recent ST turns into the in-memory
    session cache. The UI calls this immediately after login (and on
    every session switch) so the first chat message doesn't pay for
    those round-trips.

    This runs the customer lookup and ST recall in parallel — the dead
    time between login and first message is enough to mask both.
    """
    t0 = time.perf_counter()
    agent = get_agent(request)

    async def _fetch_customer():
        try:
            return await asyncio.to_thread(_fetch_customer_sync, req.user_id)
        except Exception as exc:
            logger.warning("warmup: customer fetch failed: {}", exc)
            return None

    async def _fetch_st():
        try:
            turns = await asyncio.to_thread(
                agent.st_store.recent, req.user_id, req.session_id, _CACHE_MAX_TURNS,
            )
            return list(turns or [])
        except Exception as exc:
            logger.warning("warmup: ST fetch failed: {}", exc)
            return []

    async def _fetch_recent():
        try:
            return await asyncio.to_thread(
                _fetch_recent_orders_sync, req.user_id,
            )
        except Exception as exc:
            logger.warning("warmup: recent-orders fetch failed: {}", exc)
            return []

    customer, st_turns, recent = await asyncio.gather(
        _fetch_customer(), _fetch_st(), _fetch_recent(),
    )

    request.app.state.session_cache[_cache_key(req.user_id, req.session_id)] = {
        "customer": customer,
        "st_turns": st_turns,
        "recent_orders": recent,
        "visible_orders": [],
    }

    return SessionWarmupResponse(
        warmed=True,
        customer_loaded=customer is not None,
        st_turn_count=len(st_turns),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ── GET /sessions/{sid}/turns ────────────────────────────────────────

@router.get("/sessions/{session_id}/turns", response_model=SessionTurnsResponse)
async def session_turns(
    session_id: str,
    user_id: str,
    limit: int = 20,
    st_store=Depends(get_st_store),
) -> SessionTurnsResponse:
    turns = await asyncio.to_thread(st_store.history, user_id, session_id, limit)
    items = [
        TurnItem(
            role=getattr(t, "role", "user"),
            content=getattr(t, "content", ""),
            ts=float(getattr(t, "ts", 0.0)),
        )
        for t in (turns or [])
    ]
    return SessionTurnsResponse(
        user_id=user_id,
        session_id=session_id,
        turn_count=len(items),
        turns=items,
    )
