"""
Agent Orchestrator — LangGraph Multi-Agent State Machine.

Week 10 refactor: the Week 7 linear orchestrator is now a LangGraph StateGraph
with multi-route fan-out support.

Architecture (Supervisor-Worker pattern with fan-out):
    recall → supervisor → [classification_agent, investigation_agent, resolution_agent]  (1 or more in parallel)
                                  ↘         ↓         ↙
                              merge_responses  (fan-in + synthesize)
                                      ↓
                              save_memory → END

Multi-route support:
    When a user asks a compound question (e.g. "Check my return status AND
    what's the refund policy?"), the router returns multiple
    RouteDecisions. The supervisor fans out to the relevant agent nodes
    in parallel via LangGraph's native fan-out. The merge_responses node
    combines all agent outputs into one coherent answer.

    For single-route queries (the common case), only one agent runs and
    merge_responses passes through without an extra LLM call.

Prompt management:
    Sub-agent prompts (classification, investigation, resolution, merge) are defined in
    agents/prompts/agent_prompts.py with LangFuse integration + local fallbacks.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.router import QueryRouter, RouteDecision, MultiRouteDecision
from agents.prompts.agent_prompts import (
    build_admin_agent_prompt,
    build_clinical_agent_prompt,
    build_direct_agent_prompt,
    build_merge_prompt,
)
from memory.schemas import ConversationTurn
from infrastructure.observability import (
    observe,
    update_current_trace,
    update_current_observation,
)


@dataclass
class AgentResponse:
    """
    Complete agent response with metadata for the UI/Notebooks.
    """
    answer: str
    route: str = "direct"
    routes: List[str] = field(default_factory=list)   # all routes taken (multi-route)
    action: Optional[str] = None
    tool_output: str = ""
    memory_context: str = ""
    latency_ms: int = 0


class AgentOrchestrator:
    """
    Orchestrates the multi-agent system using a LangGraph StateGraph.

    Supports both single-route and multi-route (fan-out) queries.
    """

    def __init__(
        self,
        llm_chat: Any,
        llm_router: Any,
        st_store: Any,
        lt_store: Any,
        recaller: Any,
        distiller: Any,
        crm_tool: Optional[Any] = None,
        rag_tool: Optional[Any] = None,
        web_tool: Optional[Any] = None,
        llm_fast: Optional[Any] = None,
        llm_guardrail: Optional[Any] = None,
    ) -> None:
        self.llm_chat = llm_chat
        # Fast LLM (Groq) for direct/concierge replies. Falls back to the
        # main chat LLM if the caller didn't supply one.
        self.llm_fast = llm_fast or llm_chat
        self.st_store = st_store
        self.lt_store = lt_store
        self.recaller = recaller
        self.distiller = distiller

        self.crm_tool = crm_tool
        self.rag_tool = rag_tool
        self.web_tool = web_tool

        self.router = QueryRouter(llm_router)
        # Domain guardrail — decides if the message is in-scope for a
        # store health assistant. Runs in parallel with the router
        # so its latency is hidden inside the gather. Falls back to
        # the router LLM if no dedicated guardrail LLM is provided.
        from agents.guardrail import Guardrail
        self.guardrail = Guardrail(llm_guardrail or llm_router)

        # CAG cache — owned by the FastAPI lifespan (it depends on the
        # local-embedder build path) and attached after the
        # orchestrator is constructed. The decision graph below reads
        # this attribute lazily, so the late binding is fine.
        self.cag_cache: Any = None

        # Build the multi-agent state machine (legacy fan-out graph
        # used by Week 10's recall → supervisor → … → save_memory
        # pipeline).
        self.graph = self._build_graph()

        # Build the decision LangGraph — three parallel classifiers
        # (guardrail, router, cag) → fan-in to a decide node. The
        # chat-API hot path invokes this graph for every request and
        # branches on ``state["verdict"]``. Latency is identical to
        # the old asyncio.gather fan-out (parallel nodes, max ≈800 ms
        # set by the router LLM call) but the routing decisions are
        # now formalised as graph nodes — easier to inspect on a
        # Langfuse trace and trivial to extend with more parallel
        # checks (toxicity, PII, etc.) later.
        self.decision_graph = self._build_decision_graph()

    def _build_decision_graph(self):
        """Compile the parallel-classifier LangGraph used by the chat
        API hot path. See ``agents.decision_graph`` for the topology
        and node behaviour. The CAG cache is read via a getter
        closure so the graph survives the late-binding pattern used
        by the FastAPI lifespan.
        """
        from agents.decision_graph import build_decision_graph
        return build_decision_graph(
            guardrail=self.guardrail,
            router=self.router,
            cag_getter=lambda: self.cag_cache,
        )

    # ── Graph Construction ──────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """
        Construct the LangGraph state machine.

        Topology:
            recall → supervisor → [classification | investigation | resolution]  (fan-out)
                                        ↘     ↓     ↙
                                     merge_responses  (fan-in)
                                           ↓
                                      save_memory → END
        """
        workflow = StateGraph(AgentState)

        # 1. Define Nodes
        workflow.add_node("recall", self.recall_node)
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("admin_agent", self.admin_agent_node)
        workflow.add_node("clinical_agent", self.clinical_agent_node)
        workflow.add_node("direct_agent", self.direct_agent_node)
        workflow.add_node("merge_responses", self.merge_responses_node)
        workflow.add_node("save_memory", self.store_and_distill_node)

        # 2. Define Edges (The Pipeline)
        workflow.set_entry_point("recall")
        workflow.add_edge("recall", "supervisor")

        # Conditional routing from Supervisor (supports fan-out)
        # supervisor_routing() returns str for single-route, list[str] for multi-route
        workflow.add_conditional_edges(
            "supervisor",
            self.supervisor_routing,
            {
                "admin": "admin_agent",
                "clinical": "clinical_agent",
                "direct": "direct_agent"
            }
        )

        # All agents converge to merge_responses (fan-in point)
        workflow.add_edge("admin_agent", "merge_responses")
        workflow.add_edge("clinical_agent", "merge_responses")
        workflow.add_edge("direct_agent", "merge_responses")

        # Merge → save → end
        workflow.add_edge("merge_responses", "save_memory")
        workflow.add_edge("save_memory", END)

        return workflow.compile()

    # ── Node Implementations ────────────────────────────────────────

    @observe(name="node_recall")
    def recall_node(self, state: AgentState) -> Dict:
        """Reads conversation history and long-term facts into the state."""
        user_message = state["messages"][-1].content
        user_id = state["user_id"]
        session_id = state["session_id"]

        try:
            st_turns, lt_facts = self.recaller.recall(
                user_id=user_id,
                session_id=session_id,
                query=user_message
            )
            memory_context = self.recaller.format_context(st_turns)
            semantic_facts = [f.to_dict() if hasattr(f, 'to_dict') else vars(f) for f in lt_facts]

            return {
                "memory_context": memory_context,
                "semantic_facts": semantic_facts
            }
        except Exception as e:
            logger.warning(f"Recall node failed: {e}")
            return {"memory_context": "(memory offline)"}

    @observe(name="node_supervisor")
    def supervisor_node(self, state: AgentState) -> Dict:
        """
        Classifies intent and chooses which specialized agent(s) to call.

        For multi-intent queries, returns multiple route decisions so the
        graph can fan out to parallel agent nodes.
        """
        user_message = state["messages"][-1].content
        memory_context = state.get("memory_context", "")

        # Augment context with LT facts for the Router
        facts = state.get("semantic_facts", [])
        if facts:
            memory_context += "\n=== LONG-TERM FACTS ===\n"
            for f in facts:
                memory_context += f"- {f.get('text', '')}\n"

        # Router now returns MultiRouteDecision
        multi_decision = self.router.route(user_message, memory_context)

        # Serialise all decisions for the state
        route_decisions = [
            {
                "route": d.route,
                "action": d.action,
                "params": d.params or {},
                "reasoning": d.reasoning,
            }
            for d in multi_decision.decisions
        ]

        return {
            # Full list of decisions (multi-route)
            "route_decisions": route_decisions,
            # Primary decision (backward compat)
            "route_decision": route_decisions[0],
        }

    def supervisor_routing(self, state: AgentState) -> Union[str, List[str]]:
        """
        Map RouteDecision route strings to graph node names.

        Router outputs:  crm | rag | web_search | direct
        Graph nodes:     classification_agent | investigation_agent | resolution_agent

        Returns a single string for single-route (standard conditional edge)
        or a list of strings for multi-route (LangGraph fan-out).

        Note: web_search is handled by direct_agent, which checks
        route_decision internally to decide whether to call Tavily.
        """
        route_map = {
            "order": "admin",
            "crm": "admin",  # backward compatibility
            "rag": "clinical",
            "web_search": "direct",
            "direct": "direct",
        }

        decisions = state.get("route_decisions", [])
        if not decisions:
            return "direct"

        # Map routes to node names, deduplicate, preserve order
        node_names = []
        seen = set()
        for d in decisions:
            node = route_map.get(d.get("route", "direct"), "direct")
            if node not in seen:
                node_names.append(node)
                seen.add(node)

        # Single route → return string (no fan-out)
        # Multiple routes → return list (LangGraph fan-out)
        if len(node_names) == 1:
            return node_names[0]
        return node_names

    @observe(name="node_admin_agent")
    def admin_agent_node(self, state: AgentState) -> Dict:
        """Specialized Agent for Order Lookup and Dispute Processing."""
        # Find the order-specific decision from route_decisions
        decisions = state.get("route_decisions", [])
        order_decision = next(
            (d for d in decisions if d.get("route") in ("order", "crm")),
            state.get("route_decision", {})
        )
        action = order_decision.get("action", "lookup_order")
        params = dict(order_decision.get("params") or {})

        # Fallback: auto-inject phone if the user message is just a phone number
        user_message = state["messages"][-1].content
        clean_msg = user_message.strip().replace(" ", "").replace("+", "")
        if "phone" not in params and clean_msg.isdigit() and len(clean_msg) >= 9:
            params["phone"] = clean_msg

        # Auto-inject customer_id/patient_id from session
        if "customer_id" not in params and state.get("user_id"):
            params["customer_id"] = state["user_id"]
        if "patient_id" not in params and state.get("user_id"):
            params["patient_id"] = state["user_id"]

        system_prompt = build_admin_agent_prompt()

        if not self.crm_tool:
            tool_output = "Order Tool unavailable."
        else:
            tool_output = self.crm_tool.dispatch(action, params)

        answer = self._generate_agent_response(state, system_prompt, tool_output)

        return {
            "messages": [AIMessage(content=answer)],
            "tool_output": tool_output,
            "final_answer": answer,
            "agent_outputs": [{"route": "order", "tool_output": tool_output, "answer": answer}],
        }

    @observe(name="node_clinical_agent")
    def clinical_agent_node(self, state: AgentState) -> Dict:
        """Specialized Agent for Policy Verification and Dispute History."""
        # Find the RAG-specific decision from route_decisions
        decisions = state.get("route_decisions", [])
        rag_decision = next(
            (d for d in decisions if d.get("route") == "rag"),
            state.get("route_decision", {})
        )
        params = rag_decision.get("params", {})
        query = params.get("query", state["messages"][-1].content)

        system_prompt = build_clinical_agent_prompt()

        # Inject semantic facts for customer context
        facts = state.get("semantic_facts", [])
        kb_context = ""
        if facts:
            kb_context += "\n=== CUSTOMER DISPUTE HISTORY ===\n"
            for f in facts:
                kb_context += f"- {f.get('text', '')}\n"

        if not self.rag_tool:
            tool_output = "RAG Tool unavailable."
        else:
            tool_output = self.rag_tool.dispatch("search", {"query": query})

        answer = self._generate_agent_response(state, system_prompt, tool_output, extra_context=kb_context)

        return {
            "messages": [AIMessage(content=answer)],
            "tool_output": tool_output,
            "final_answer": answer,
            "agent_outputs": [{"route": "rag", "tool_output": tool_output, "answer": answer}],
        }

    @observe(name="node_direct_agent")
    def direct_agent_node(self, state: AgentState) -> Dict:
        """Specialized Agent for greetings and general inquiries."""
        system_prompt = build_direct_agent_prompt()

        # Check if any decision routes to web_search
        decisions = state.get("route_decisions", [])
        web_decision = next(
            (d for d in decisions if d.get("route") == "web_search"),
            None
        )

        tool_output = ""
        route_label = "direct"
        if web_decision and self.web_tool:
            params = web_decision.get("params", {})
            query = params.get("query", state["messages"][-1].content)
            tool_output = self.web_tool.dispatch("search", {"query": query})
            route_label = "web_search"

        answer = self._generate_agent_response(state, system_prompt, tool_output)

        return {
            "messages": [AIMessage(content=answer)],
            "tool_output": tool_output,
            "final_answer": answer,
            "agent_outputs": [{"route": route_label, "tool_output": tool_output, "answer": answer}],
        }

    @observe(name="node_merge_responses")
    def merge_responses_node(self, state: AgentState) -> Dict:
        """
        Fan-in node: merges outputs from parallel agent nodes.

        Single-route:  passes through (no extra LLM call, zero latency overhead).
        Multi-route:   calls the merge synthesiser to produce one coherent response.
        """
        agent_outputs = state.get("agent_outputs", [])

        # Single agent → pass through (backward compatible, no overhead)
        if len(agent_outputs) <= 1:
            return {}

        # Multi-agent → synthesize into one response
        logger.info(f"Merging {len(agent_outputs)} agent outputs into unified response")

        user_message = state["messages"][0].content
        memory_context = state.get("memory_context", "")

        # Build labelled tool output sections for the synthesiser
        combined_tool_output = ""
        for out in agent_outputs:
            route = out.get("route", "unknown").upper()
            answer = out.get("answer", "")
            combined_tool_output += f"=== {route} AGENT RESULT ===\n{answer}\n\n"

        system_prompt = build_merge_prompt()

        system_content = (
            f"{system_prompt}\n\n"
            f"=== MEMORY CONTEXT ===\n{memory_context}\n\n"
            f"=== AGENT RESULTS TO MERGE ===\n{combined_tool_output}"
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_message),
        ]

        response = self.llm_chat.invoke(messages)
        merged_answer = response.content if hasattr(response, "content") else str(response)

        # Combine all tool outputs
        all_tool_output = "\n---\n".join(
            out.get("tool_output", "") for out in agent_outputs if out.get("tool_output")
        )

        return {
            "final_answer": merged_answer,
            "tool_output": all_tool_output,
            "messages": [AIMessage(content=merged_answer)],
        }

    @observe(name="node_save_memory")
    def store_and_distill_node(self, state: AgentState) -> Dict:
        """Saves messages to short-term and extracts long-term facts."""
        user_message = state["messages"][0].content
        answer = state["final_answer"]
        user_id = state["user_id"]
        session_id = state["session_id"]

        # Store ST turns
        now = time.time()
        self.st_store.add(user_id, session_id, ConversationTurn(user_id=user_id, session_id=session_id, role="user", content=user_message, ts=now))
        self.st_store.add(user_id, session_id, ConversationTurn(user_id=user_id, session_id=session_id, role="assistant", content=answer, ts=now))

        # Distill if needed
        try:
            recent = self.st_store.recent(user_id, session_id, k=5)
            if self.distiller.should_distill(recent):
                logger.info(f"Distilling new facts for {user_id}...")
                self.distiller.distill(user_id, recent)
                return {"should_distill": True}
        except Exception as e:
            logger.warning(f"Distillation failed: {e}")

        return {"should_distill": False}

    # ── Core Helpers ──────────────────────────────────────────────

    def _generate_agent_response(self, state: AgentState, system_prompt: str, tool_output: str, extra_context: str = "") -> str:
        """
        Standard LLM call for all sub-agents.

        Each sub-agent calls its prompt builder (e.g. build_admin_agent_prompt())
        which fetches from LangFuse Prompt Management with local fallbacks.
        The system_prompt passed here is already the fully composed prompt.
        """
        user_message = state["messages"][-1].content
        memory_context = state.get("memory_context", "") + extra_context

        system_content = (
            f"{system_prompt}\n\n"
            f"=== MEMORY CONTEXT ===\n{memory_context}\n\n"
            f"=== TOOL OUTPUT ===\n{tool_output}"
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_message),
        ]

        response = self.llm_chat.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    # ── Entry Point ───────────────────────────────────────────────

    @observe(name="agent_chat")
    def chat(self, user_message: str, user_id: str, session_id: str) -> AgentResponse:
        """Run the graph for one interaction."""
        t0 = time.time()

        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id,
            "session_id": session_id,
            "agent_outputs": [],  # initialise the fan-out collector
        }

        # Run the LangGraph state machine
        final_state = self.graph.invoke(initial_state)

        latency = int((time.time() - t0) * 1000)

        # Extract all routes taken
        route_decisions = final_state.get("route_decisions", [])
        all_routes = [d.get("route", "direct") for d in route_decisions]
        primary = route_decisions[0] if route_decisions else {"route": "direct"}

        return AgentResponse(
            answer=final_state["final_answer"],
            route=primary.get("route", "direct"),
            routes=all_routes,
            action=primary.get("action"),
            tool_output=final_state.get("tool_output", ""),
            memory_context=final_state.get("memory_context", ""),
            latency_ms=latency
        )

    # ── Async Entry Point (for FastAPI) ──────────────────────────

    async def achat(self, user_message: str, user_id: str, session_id: str) -> AgentResponse:
        """
        Async version of chat() — uses graph.ainvoke() for non-blocking execution.

        Identical logic to chat(), but awaits ainvoke() so the FastAPI event
        loop isn't blocked during LLM calls.  Notebooks can keep using chat().
        """
        t0 = time.time()

        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id,
            "session_id": session_id,
            "agent_outputs": [],
        }

        # Non-blocking graph execution
        final_state = await self.graph.ainvoke(initial_state)

        latency = int((time.time() - t0) * 1000)

        route_decisions = final_state.get("route_decisions", [])
        all_routes = [d.get("route", "direct") for d in route_decisions]
        primary = route_decisions[0] if route_decisions else {"route": "direct"}

        return AgentResponse(
            answer=final_state["final_answer"],
            route=primary.get("route", "direct"),
            routes=all_routes,
            action=primary.get("action"),
            tool_output=final_state.get("tool_output", ""),
            memory_context=final_state.get("memory_context", ""),
            latency_ms=latency
        )

    # ── Voice Fast Path ────────────────────────────────────────
    #
    # Bypass the multi-agent graph entirely and stream tokens from a
    # single fast-LLM call. This is the right shape for realtime voice:
    # the full graph (router LLM → parallel agents → merge LLM) makes
    # 3+ sequential LLM calls before any audio plays. For phone-call
    # style turns the user can't see citations or rich tool output
    # anyway, so the multi-agent richness is mostly wasted.
    #
    # Latency budget after this change:
    #   STT first-final (~200-400ms) + LLM first token (~300-500ms)
    #   + TTS first audio (~200-400ms)  ≈  700-1300ms perceived.
    #
    # Memory writes happen in the background after streaming ends —
    # they don't block the user-perceived latency.

    # NOTE: deliberately NOT decorated with @observe.
    # Langfuse v4's @observe doesn't cleanly handle async generators in
    # all versions — it can buffer yields until the function returns,
    # which would *single-handedly* fake-disable streaming and explain
    # multi-second first-token latencies even when the LLM itself is
    # streaming fine. The adapter (voice/adapter.py) opens a manual
    # Langfuse span around the consumption of this generator instead,
    # which is the safe pattern for streaming functions.
    async def achat_stream_fast(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
    ):
        """Single-LLM streaming path for voice. Yields:

            ("token",   str)             — speakable text chunk
            ("partial", str)              — running concatenation, sent
                                           after each token so the
                                           caller can record what was
                                           ACTUALLY emitted in case of
                                           barge-in cancellation
            ("final",   AgentResponse)    — emitted once, after the last
                                           token

        On barge-in (`asyncio.CancelledError`), the generator terminates
        without ever yielding "final". Callers MUST treat the last
        "partial" they received as the truth of what was said. See
        voice/adapter.py for how this is wired into LiveKit.
        """
        import asyncio as _asyncio
        from langchain_core.messages import SystemMessage, HumanMessage

        t_start = time.perf_counter()

        # ── 1+2) Memory fetch + Router — run IN PARALLEL ──
        # Previously these were sequential: memory fetch (~0-600ms) then
        # router LLM (~400-800ms). Running them concurrently saves
        # ~400-800ms per turn (whichever is slower becomes free).
        # The router doesn't need memory_context to classify intent —
        # it uses only the user_message. Memory context is injected into
        # the synthesis prompt AFTER both complete.
        memory_context = ""
        tool_output = ""
        route = "direct"
        t_mem_start = time.perf_counter()

        async def _fetch_memory():
            """Fetch conversation memory (instant from cache, ~600ms cap from DB)."""
            try:
                cache = getattr(self, "_voice_ctx_cache", None)
                if cache is None:
                    cache = self._voice_ctx_cache = {}
                cached_turns = cache.get(session_id)
                if cached_turns:
                    return self.recaller.format_context(cached_turns)
                try:
                    recent = await _asyncio.wait_for(
                        _asyncio.to_thread(self.st_store.recent, user_id, session_id, 6),
                        timeout=0.6,
                    )
                    if recent:
                        cache[session_id] = list(recent)
                        return self.recaller.format_context(recent)
                except _asyncio.TimeoutError:
                    logger.warning("voice: first-turn memory fetch slow (>0.6s) — proceeding")
                except Exception as e:
                    logger.debug(f"voice: memory fetch failed (non-fatal): {e}")
            except Exception as e:
                logger.debug(f"voice: cache read failed: {e}")
            return ""

        async def _route_and_dispatch():
            """Run the intent router + dispatch tool if needed."""
            _route = "direct"
            _tool_output = ""
            try:
                decision = await self.router.aroute(user_message, "")
                primary = (decision.decisions[0]
                           if getattr(decision, "decisions", None) else None)
                _route = (primary.route if primary else "direct") or "direct"
                if _route in ("order", "crm", "rag", "web_search") and primary is not None:
                    _tool_output = await self._voice_dispatch_tool(
                        primary, customer_id=user_id, fallback_query=user_message,
                    )
            except Exception as e:  # noqa: BLE001 — routing must never break the call
                logger.warning(f"voice: routing/tool dispatch failed (non-fatal): {e}")
            return _route, _tool_output

        # Fire both concurrently — the slower one is "free".
        (memory_context, (route, tool_output)) = await _asyncio.gather(
            _fetch_memory(),
            _route_and_dispatch(),
        )
        t_route_done = time.perf_counter()
        t_mem_done = t_route_done  # both finished at the same wall-clock time

        # ── 3) Build the voice-shaped prompt (grounded when a tool ran) ──
        system = (
            "You are the DisputeAI voice assistant on a live phone call. "
            "Keep replies short, warm, conversational — under three sentences. "
            "No markdown, no tables, no bullet points, no asterisks. The caller is "
            "listening, not reading; read order IDs, dates, and prices naturally. "
            "If a customer is upset or aggressive, remain calm, empathetic, and "
            "focus on the solution.\n\n"
            "Store policy quick facts (use these for general questions):\n"
            "- Standard items can be returned within 30 days of delivery.\n"
            "- Electronics have a 14-day return window.\n"
            "- Furniture returns are subject to a 15% restocking fee.\n"
            "- General customer support phone number: +1 800 555 0199.\n"
            "Use these quick facts ONLY for general policy questions. For specific "
            "customer orders, disputes, and live details, always use the INFORMATION "
            "block below (the live system), never these.\n\n"
        )
        if tool_output:
            system += (
                "Answer using ONLY the information below — it is the live source "
                "of truth from our database. Do NOT invent order details, dates, "
                "prices, or policy details. If it does not contain the answer, say "
                "so plainly and offer to help another way.\n\n"
                f"=== INFORMATION ===\n{tool_output}\n\n"
            )
        system += f"=== RECENT CONVERSATION ===\n{memory_context}"
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_message),
        ]
        t_prompt_built = time.perf_counter()

        # ── 4) Stream tokens from the fast LLM ──
        llm = self.llm_fast or self.llm_chat
        answer_parts: list[str] = []
        running = ""
        t_first_chunk: float | None = None
        was_cancelled = False

        try:
            async for chunk in llm.astream(messages):
                content = getattr(chunk, "content", None)
                if not content:
                    continue
                if t_first_chunk is None:
                    t_first_chunk = time.perf_counter()
                    logger.info(
                        "⏱  achat_stream_fast: "
                        f"mem={int((t_mem_done - t_mem_start) * 1000)}ms, "
                        f"route+tool={int((t_route_done - t_mem_done) * 1000)}ms (route={route}), "
                        f"prompt={int((t_prompt_built - t_route_done) * 1000)}ms, "
                        f"pre_llm_total={int((t_prompt_built - t_start) * 1000)}ms, "
                        f"first_llm_chunk={int((t_first_chunk - t_prompt_built) * 1000)}ms"
                    )
                answer_parts.append(content)
                running += content
                yield ("token", content)
                yield ("partial", running)
                await _asyncio.sleep(0)
        except _asyncio.CancelledError:
            # Barge-in. The caller has already received our partials;
            # they know exactly what was said. Fall through to the
            # finally block which persists that partial answer.
            was_cancelled = True
            logger.info(
                f"voice: cancelled mid-stream after "
                f"{int((time.perf_counter() - t_start) * 1000)} ms "
                f"({len(answer_parts)} chunks, "
                f"{len(running)} chars spoken so far)"
            )

        answer_final = "".join(answer_parts).strip()
        latency = int((time.perf_counter() - t_start) * 1000)

        # ── 5) Persist the turn — FIRE AND FORGET ──
        # Previously this was `await`ed, adding 200-1500ms between
        # turns on slow networks. The in-memory context cache is
        # updated synchronously inside _save_voice_turn_async (instant),
        # so the NEXT turn has full context. The DB write drifts into
        # the background — if the call hangs up before it completes,
        # the worst case is one lost turn in persistent memory.
        if answer_final:
            _asyncio.create_task(
                self._save_voice_turn_async(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=answer_final,
                    was_interrupted=was_cancelled,
                )
            )

        # If we were cancelled, do NOT yield "final" — the consumer
        # already handled the cancellation via the async-generator
        # contract.
        if was_cancelled:
            return

        yield (
            "final",
            AgentResponse(
                answer=answer_final,
                route="voice_fast",
                routes=["voice_fast"],
                action=None,
                tool_output="",
                memory_context=memory_context,
                latency_ms=latency,
            ),
        )

    # ── Voice tool dispatch ─────────────────────────────────────

    async def _voice_dispatch_tool(self, decision, *, customer_id, fallback_query=""):
        """Run the tool for a voice ``RouteDecision`` — mirrors the text
        path's ``_dispatch_tool`` so voice answers are grounded in the SAME
        live data (CRM / RAG / web) instead of being invented by a bare LLM.
        Returns the tool's plain-text output (the synthesis prompt then feeds
        it to the LLM as the source of truth). Never raises — on any failure
        it returns ``""`` and the caller falls back to an ungrounded reply.
        """
        import asyncio as _asyncio
        route = getattr(decision, "route", "direct")
        params = dict(getattr(decision, "params", None) or {})
        try:
            if route in ("order", "crm") and self.crm_tool is not None:
                action = getattr(decision, "action", None) or "lookup_order"
                # Self-referential Order actions get the caller's customer_id when
                # the router didn't extract one (e.g. "where is my refund").
                if action in {"lookup_order", "lookup_customer", "submit_dispute",
                              "check_dispute_status", "list_disputes", "update_dispute"}:
                    params.setdefault("customer_id", customer_id)
                    params.setdefault("patient_id", customer_id)
                return await _asyncio.to_thread(self.crm_tool.dispatch, action, params)
            if route == "rag" and self.rag_tool is not None:
                if not params.get("query") and fallback_query:
                    params["query"] = fallback_query
                return await _asyncio.to_thread(self.rag_tool.dispatch, "search", params)
            if route == "web_search" and self.web_tool is not None:
                if not params.get("query") and fallback_query:
                    params["query"] = fallback_query
                return await _asyncio.to_thread(self.web_tool.dispatch, "search", params)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"voice: tool dispatch ({route}) failed: {e}")
        return ""

    # ── Voice memory persistence ────────────────────────────────

    async def _save_voice_turn_async(
        self,
        *,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        was_interrupted: bool,
    ) -> None:
        """Persist a voice turn to short-term + (eventually) long-term
        memory. Runs off-thread (so it never parks the event loop) but is
        AWAITED to completion so the write can't be dropped when the call's
        loop tears down on hang-up.

        The flow mirrors what the full ``store_and_distill_node`` does
        for the text path — but it never blocks TTS or the next turn.

        For barge-in turns we tag the stored assistant message with a
        ``[interrupted]`` marker so downstream distillation knows it
        was cut short and shouldn't extract long-term facts from it.
        """
        import asyncio as _asyncio

        # Tag interrupted assistant utterances so distillation can skip
        # them. The marker is invisible to the LLM in subsequent turns
        # because we strip it during context formatting if needed.
        stored_assistant = (
            f"[interrupted] {assistant_message}"
            if was_interrupted and assistant_message
            else assistant_message
        )

        def _do_save():
            try:
                now = time.time()
                self.st_store.add(
                    user_id, session_id,
                    ConversationTurn(
                        user_id=user_id, session_id=session_id,
                        role="user", content=user_message, ts=now,
                    ),
                )
                self.st_store.add(
                    user_id, session_id,
                    ConversationTurn(
                        user_id=user_id, session_id=session_id,
                        role="assistant", content=stored_assistant, ts=now,
                    ),
                )
                # Keep the per-call in-memory context fresh so the NEXT turn
                # gets memory INSTANTLY (no Supabase round-trip) — this is what
                # keeps voice latency low across a conversation.
                try:
                    _cache = getattr(self, "_voice_ctx_cache", None)
                    if _cache is None:
                        _cache = self._voice_ctx_cache = {}
                    _turns = list(_cache.get(session_id) or [])
                    _turns.append(ConversationTurn(
                        user_id=user_id, session_id=session_id,
                        role="user", content=user_message, ts=now))
                    _turns.append(ConversationTurn(
                        user_id=user_id, session_id=session_id,
                        role="assistant", content=stored_assistant, ts=now))
                    _cache[session_id] = _turns[-6:]
                except Exception:
                    pass
                # Materialise a chat_sessions row so the call appears in
                # the UI sidebar (the "voice-" session_id prefix is the
                # marker the frontend uses to split voice from text).
                # Use the SAME working Supabase session that just saved the
                # turns — NOT touch_session_sync(), whose get_sql_engine()
                # is unreliable inside the voice-worker process and silently
                # failed there, leaving voice calls invisible in the sidebar.
                try:
                    from infrastructure.db.crm_models import ChatSession
                    sess = self.st_store.supabase_session_factory()
                    try:
                        now_i = int(now)
                        row = sess.get(ChatSession, session_id)
                        if row is None:
                            sess.add(ChatSession(
                                session_id=session_id,
                                patient_id=user_id,
                                title=((user_message or "").strip()[:48] or "Voice call"),
                                last_message_at=now_i,
                                created_at=now_i,
                                updated_at=now_i,
                                archived=0,
                            ))
                        else:
                            row.last_message_at = now_i
                            row.updated_at = now_i
                        sess.commit()
                    finally:
                        sess.close()
                except Exception as e:
                    logger.warning(f"voice: session header create failed: {e}")

                # Auto-title after a few turns of real content. Only fires
                # once per session (the helper checks if the title is
                # still the default before doing the LLM call).
                try:
                    from api.routers.chat_sessions import maybe_auto_title_sync
                    maybe_auto_title_sync(
                        session_id=session_id,
                        user_id=user_id,
                        st_store=self.st_store,
                        llm=self.llm_fast or self.llm_chat,
                    )
                except Exception as e:
                    logger.debug(f"voice: auto-title failed (non-fatal): {e}")
                # Long-term distillation — only on clean (un-interrupted)
                # turns, and only when there's enough new conversation
                # for the distiller's heuristics to extract a useful
                # fact.
                if not was_interrupted:
                    try:
                        recent = self.st_store.recent(user_id, session_id, k=5)
                        if self.distiller.should_distill(recent):
                            logger.debug(
                                f"voice: distilling LT facts for {user_id}"
                            )
                            self.distiller.distill(user_id, recent)
                    except Exception as e:
                        logger.debug(
                            f"voice: distillation failed (non-fatal): {e}"
                        )
            except Exception as e:
                logger.warning(f"voice: memory write failed: {e}")

        # Run the save to COMPLETION: off-thread (so the event loop isn't
        # blocked by the DB write) but AWAITED so it can't be dropped when
        # the call's loop tears down on hang-up.
        try:
            await _asyncio.to_thread(_do_save)
        except Exception as e:
            logger.warning(f"voice: save task failed: {e}")

    # ── Async Streaming Entry Point (for Voice / LiveKit) ────────

    async def achat_stream(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
    ):
        """
        Streaming variant of ``achat`` for the voice pipeline.

        Yields ``(kind, payload)`` tuples:

            ("token", str)        — a chunk of the assistant's text answer
            ("final", AgentResponse) — emitted once, after the last token

        The voice adapter (``LangGraphLLMAdapter``) consumes the tokens and
        forwards each as a ``ChatChunk`` to LiveKit's TTS plugin, which begins
        synthesising audio as soon as the first sentence-sized chunk arrives.
        That cuts time-to-first-audio dramatically vs. waiting for the full
        ``final_answer`` string.

        Implementation note (Week 15 stage 1):

            Today we run the graph to completion via ``ainvoke`` and then chunk
            the final answer into sentence/phrase units, yielding each with a
            small ``asyncio.sleep(0)`` so TTS can pipeline synthesis. This
            preserves the multi-agent richness (router / fan-out / merge) and
            keeps state behaviour identical to ``achat``.

            Week 15 stage 2 (follow-up) replaces this with native
            ``astream_events`` token streaming tagged ``voice_output`` on the
            synthesiser node — first-token latency drops further. The adapter
            interface stays unchanged.

        Cancellation:

            If the consumer cancels (LiveKit barge-in), the ``CancelledError``
            propagates here and aborts the graph mid-flight.
        """
        import re
        import asyncio as _asyncio

        t0 = time.time()

        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id,
            "session_id": session_id,
            "agent_outputs": [],
        }

        # Run the graph to completion. Cancellation propagates through ainvoke.
        final_state = await self.graph.ainvoke(initial_state)

        latency = int((time.time() - t0) * 1000)

        route_decisions = final_state.get("route_decisions", [])
        all_routes = [d.get("route", "direct") for d in route_decisions]
        primary = route_decisions[0] if route_decisions else {"route": "direct"}

        answer = final_state.get("final_answer", "") or ""

        # Chunk by sentence first, then by phrase if a sentence is long.
        # This is the unit TTS will receive — keep them speakable.
        # Pattern keeps the trailing punctuation with the sentence.
        sentences = re.findall(r"[^.!?\n]+[.!?]?(?:\s+|$)", answer) or [answer]

        for sentence in sentences:
            if not sentence.strip():
                continue
            # For long sentences, sub-chunk on commas/semicolons to start TTS
            # earlier. ~80 chars is a comfortable phrase length.
            if len(sentence) > 80:
                parts = re.split(r"(?<=[,;:])\s+", sentence)
                for part in parts:
                    if part.strip():
                        yield ("token", part if part.endswith(" ") else part + " ")
                        # Yield to the event loop so TTS can pull the chunk.
                        await _asyncio.sleep(0)
            else:
                yield ("token", sentence)
                await _asyncio.sleep(0)

        # Emit the final AgentResponse for memory / observability bookkeeping.
        yield (
            "final",
            AgentResponse(
                answer=answer,
                route=primary.get("route", "direct"),
                routes=all_routes,
                action=primary.get("action"),
                tool_output=final_state.get("tool_output", ""),
                memory_context=final_state.get("memory_context", ""),
                latency_ms=latency,
            ),
        )


# ── Factory function ──────────────────────────────────────────

def build_agent(enable_crm: bool = True, enable_rag: bool = True, enable_web: bool = True) -> AgentOrchestrator:
    """Builds the Multi-Agent Orchestrator."""
    from infrastructure.llm import (
        get_chat_llm,
        get_fast_chat_llm,
        get_router_llm,
        get_extractor_llm,
        get_default_embeddings,
    )
    from memory.st_store import ShortTermMemoryStore
    from memory.lt_store import LongTermMemoryStore
    from memory.memory_ops import MemoryRecaller, MemoryDistiller

    llm_chat = get_chat_llm(temperature=0)
    # Direct/concierge replies — keep deterministic so the model can't
    # invent foreign greetings, made-up names, etc. Style is enforced
    # via the explicit override appended in chat.py.
    llm_fast = get_fast_chat_llm(temperature=0)
    llm_router = get_router_llm(temperature=0)
    llm_extractor = get_extractor_llm(temperature=0)
    embedder = get_default_embeddings()

    st_store = ShortTermMemoryStore()
    lt_store = LongTermMemoryStore(embedder)
    recaller = MemoryRecaller(st_store, lt_store)
    distiller = MemoryDistiller(llm_extractor, lt_store)

    crm_tool = None
    if enable_crm:
        try:
            from agents.tools import CRMTool
            crm_tool = CRMTool()
            logger.info("CRM tool initialised")
        except Exception as e:
            logger.warning(f"CRM tool unavailable: {e}")

    rag_tool = None
    if enable_rag:
        try:
            from agents.tools import RAGTool
            rag_tool = RAGTool(embedder=embedder, llm=llm_fast)
            logger.info("RAG tool initialised")
        except Exception as e:
            logger.warning(f"RAG tool unavailable: {e}")

    web_tool = None
    if enable_web:
        try:
            from agents.tools import WebSearchTool
            web_tool = WebSearchTool()
            logger.info("Web search tool initialised")
        except Exception as e:
            logger.warning(f"Web search tool unavailable: {e}")

    return AgentOrchestrator(
        llm_chat=llm_chat,
        llm_fast=llm_fast,
        llm_router=llm_router,
        # Guardrail uses the extractor LLM (Llama 3.1 8B Instant on
        # Groq) — ~150 ms binary classification, an order of magnitude
        # cheaper than the 70B router for the same call cadence.
        llm_guardrail=llm_extractor,
        st_store=st_store,
        lt_store=lt_store,
        recaller=recaller,
        distiller=distiller,
        crm_tool=crm_tool,
        rag_tool=rag_tool,
        web_tool=web_tool
    )


# ── MCP-backed factory (teaching demo) ─────────────────────────

def _mcp_invoke_sync(tool, params: dict) -> str:
    """
    Bridge async MCP tool invocation to sync context.

    MCP tools from langchain-mcp-adapters are async-only (ainvoke).
    Graph nodes run synchronously, so we bridge here.
    Works in both Jupyter (nest_asyncio) and plain scripts.
    """
    import asyncio
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    try:
        loop = asyncio.get_running_loop()
        raw = loop.run_until_complete(tool.ainvoke(clean))
    except RuntimeError:
        raw = asyncio.run(tool.ainvoke(clean))

    # MCP returns list[{type, text}] — extract plain text
    if isinstance(raw, list):
        return "\n".join(
            item.get("text", str(item))
            for item in raw
            if isinstance(item, dict)
        ) or str(raw)
    return str(raw)


class _MCPCRMToolAdapter:
    """
    Adapter: MCP CRM tools → CRMTool.dispatch(action, params) interface.
    classification_agent_node calls dispatch() — has no idea it's MCP.
    """

    _ACTION_TO_TOOL = {
        "lookup_patient": "lookup_patient",
        "search_doctors": "search_doctors",
        "create_booking": "create_booking",
        "cancel_booking": "cancel_booking",
        "reschedule_booking": "reschedule_booking",
    }

    def __init__(self, tools_by_name: dict):
        self._tools = tools_by_name

    def dispatch(self, action: str, params: dict) -> str:
        tool_name = self._ACTION_TO_TOOL.get(action)
        if not tool_name or tool_name not in self._tools:
            return (
                f"Unknown or unavailable CRM action via MCP: {action}. "
                f"Available: {list(self._tools.keys())}"
            )
        try:
            return _mcp_invoke_sync(self._tools[tool_name], params)
        except Exception as exc:
            logger.error(f"MCP CRM tool '{tool_name}' failed: {exc}")
            return f"Error calling MCP CRM tool '{tool_name}': {exc}"


class _MCPRAGToolAdapter:
    """
    Adapter: MCP RAG tools → RAGTool.dispatch(action, params) interface.
    investigation_agent_node calls dispatch() — has no idea it's MCP.
    """

    _ACTION_TO_TOOL = {
        "search": "search_policy_kb",
        "cache_stats": "cache_stats",
        "clear_cache": "clear_cache",
    }

    def __init__(self, tools_by_name: dict):
        self._tools = tools_by_name

    def dispatch(self, action: str, params: dict) -> str:
        tool_name = self._ACTION_TO_TOOL.get(action)
        if not tool_name or tool_name not in self._tools:
            return (
                f"Unknown or unavailable RAG action via MCP: {action}. "
                f"Available: {list(self._ACTION_TO_TOOL.keys())}"
            )
        try:
            return _mcp_invoke_sync(self._tools[tool_name], params)
        except Exception as exc:
            logger.error(f"MCP RAG tool '{tool_name}' failed: {exc}")
            return f"Error calling MCP RAG tool '{tool_name}': {exc}"


class _MCPWebToolAdapter:
    """
    Adapter: MCP Web tool → WebSearchTool.dispatch(action, params) interface.
    resolution_agent_node calls dispatch() — has no idea it's MCP.
    """

    _ACTION_TO_TOOL = {
        "search": "web_search",
    }

    def __init__(self, tools_by_name: dict):
        self._tools = tools_by_name

    def dispatch(self, action: str, params: dict) -> str:
        tool_name = self._ACTION_TO_TOOL.get(action)
        if not tool_name or tool_name not in self._tools:
            return (
                f"Unknown or unavailable web action via MCP: {action}. "
                f"Available: {list(self._ACTION_TO_TOOL.keys())}"
            )
        try:
            return _mcp_invoke_sync(self._tools[tool_name], params)
        except Exception as exc:
            logger.error(f"MCP Web tool '{tool_name}' failed: {exc}")
            return f"Error calling MCP Web tool '{tool_name}': {exc}"


async def build_agent_mcp() -> AgentOrchestrator:
    """
    MCP-backed variant of build_agent() — ALL tools via MCP.

    Wires the orchestrator to 5 MCP servers:
      1. disputeai-crm     (custom)        →  admin_agent_node
      2. disputeai-memory  (custom)        →  discovered, available
      3. disputeai-kb      (custom)        →  clinical_agent_node
      4. disputeai-web     (custom)        →  direct_agent_node
      5. postgres         (off-the-shelf) →  discovered, available

    The LangGraph topology, routing, and memory nodes are UNCHANGED.
    Only the tool integration boundary moved to MCP.
    """
    from infrastructure.llm import (
        get_chat_llm, get_router_llm, get_extractor_llm, get_default_embeddings,
    )
    from memory.st_store import ShortTermMemoryStore
    from memory.lt_store import LongTermMemoryStore
    from memory.memory_ops import MemoryRecaller, MemoryDistiller
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from mcp_servers.mcp_config import build_mcp_server_config

    llm_chat = get_chat_llm(temperature=0)
    llm_router = get_router_llm(temperature=0)
    llm_extractor = get_extractor_llm(temperature=0)
    embedder = get_default_embeddings()

    st_store = ShortTermMemoryStore()
    lt_store = LongTermMemoryStore(embedder)
    recaller = MemoryRecaller(st_store, lt_store)
    distiller = MemoryDistiller(llm_extractor, lt_store)

    # ── Spin up MCP client and load tools from all servers ──────
    server_config = build_mcp_server_config()
    logger.info(f"Connecting to MCP servers: {list(server_config.keys())}")
    mcp_client = MultiServerMCPClient(server_config)

    all_tools = await mcp_client.get_tools()
    tools_by_name = {t.name: t for t in all_tools}
    logger.info(f"Loaded {len(all_tools)} tools via MCP: {list(tools_by_name.keys())}")

    # ── Wrap MCP tools in adapters expected by the graph nodes ──
    crm_tool = _MCPCRMToolAdapter(tools_by_name)
    logger.info("CRM tool backed by disputeai-crm MCP server")

    rag_tool = _MCPRAGToolAdapter(tools_by_name)
    logger.info("RAG tool backed by disputeai-kb MCP server")

    web_tool = _MCPWebToolAdapter(tools_by_name)
    logger.info("Web tool backed by disputeai-web MCP server")

    orchestrator = AgentOrchestrator(
        llm_chat=llm_chat,
        llm_router=llm_router,
        llm_guardrail=llm_extractor,
        st_store=st_store,
        lt_store=lt_store,
        recaller=recaller,
        distiller=distiller,
        crm_tool=crm_tool,
        rag_tool=rag_tool,
        web_tool=web_tool,
    )

    # Attach the client so callers can reach extra MCP tools (e.g. memory, postgres)
    # and so it can be cleanly shut down at process exit.
    orchestrator.mcp_client = mcp_client
    orchestrator.mcp_tools = tools_by_name

    return orchestrator
