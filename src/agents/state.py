"""
AgentState — the shared state dictionary for the LangGraph StateGraph.

Every node in the graph reads from and writes back to this TypedDict.
Think of it as the "conveyor belt" that carries data through the pipeline.

Upgraded for E-Commerce Dispute Resolution:
    Added dispute-specific fields (dispute_id, dispute_type, severity, etc.)
    on top of the existing memory + routing + fan-out infrastructure.

Multi-route support:
    ``agent_outputs`` uses ``operator.add`` as a reducer so that when
    LangGraph fans out to parallel agent nodes, each node appends its
    result and the lists are automatically concatenated on fan-in.
"""

import operator
from typing import Annotated, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Standard LangGraph conversation history (add_messages merges lists)
    messages: Annotated[list[AnyMessage], add_messages]

    # User / session identifiers (passed through every node)
    user_id: str
    session_id: str

    # ── Memory ────────────────────────────────────────────────────────────────
    # Short-Term: formatted recent conversation turns (text string)
    memory_context: Optional[str]
    # Long-Term: raw fact objects [{fact, score, tags}] — NOT pre-stringified
    # Each sub-agent decides independently what to inject into its own prompt.
    semantic_facts: Optional[list[dict]]

    # ── Routing ───────────────────────────────────────────────────────────────
    # Single route decision (backward compat — always the primary/first route)
    route_decision: Optional[dict]
    # All route decisions for multi-route queries (list of dicts)
    route_decisions: Optional[list[dict]]

    # ── Tool & Response ───────────────────────────────────────────────────────
    tool_output: Optional[str]    # Raw tool output (single-route compat)
    final_answer: Optional[str]   # User-facing response

    # ── Multi-Agent Fan-Out Collector ─────────────────────────────────────────
    # Each agent node appends {"route": str, "tool_output": str, "answer": str}
    # The operator.add reducer concatenates lists from parallel branches on fan-in
    agent_outputs: Annotated[list[dict], operator.add]

    # ── Memory Write Control ───────────────────────────────────────────────────
    should_distill: Optional[bool]  # True → trigger LT fact extraction

    # ── E-Commerce Dispute Resolution Fields ──────────────────────────────────
    # Populated by the classification_agent, used by investigation + resolution agents
    dispute_id: Optional[str]              # Unique dispute reference ID (e.g. "DIS-2024-001")
    dispute_type: Optional[str]            # return | refund | damaged | missing | fraud | warranty
    severity: Optional[str]               # low | medium | high | critical
    order_metadata: Optional[Dict]         # Order details from order lookup tool
    policy_context: Optional[str]          # Raw policy context from Hybrid GraphRAG
    decision: Optional[str]               # APPROVE | PARTIAL_REFUND | REJECT | ESCALATE
    refund_amount: Optional[float]         # Calculated refund amount (0.0 if rejected)
    escalate: Optional[bool]              # True → route to escalation_agent
    required_evidence: Optional[List[str]] # Evidence the customer must provide
