"""
Pydantic request / response schemas for the E-Commerce Dispute Resolution API.

Organized by router: chat, health, disputes, graph, and per-tool groups
(order/rag/web/cag/memory/crawl). All schemas are validated at the FastAPI
boundary so internal code can assume clean inputs.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# Chat
# ═══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """POST /chat"""
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, description="User's natural-language message")


class ChatResponse(BaseModel):
    """POST /chat response."""
    answer: str
    route: Literal["cag_hit", "crm", "order", "rag", "web_search", "direct", "multi", "out_of_scope"]
    routes: List[str] = Field(default_factory=list, description="All routes taken for multi-intent queries")
    cached: bool = False
    latency_ms: int = 0
    trace_id: Optional[str] = None
    timings: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-node wall-clock latency in ms (cag, recall, route, tool, synth, save).",
    )
    model_used: Optional[str] = Field(
        default=None,
        description="Which LLM produced the final answer (fast / chat).",
    )


class ChatResetRequest(BaseModel):
    """POST /chat/reset — clear ST memory for a session."""
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class ChatResetResponse(BaseModel):
    cleared: bool = True
    user_id: str
    session_id: str


class TurnItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    ts: float


class SessionTurnsResponse(BaseModel):
    user_id: str
    session_id: str
    turn_count: int
    turns: List[TurnItem] = Field(default_factory=list)


class SessionWarmupRequest(BaseModel):
    """POST /sessions/warmup — preload patient + ST turns into the server cache."""
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class SessionWarmupResponse(BaseModel):
    warmed: bool = True
    patient_loaded: bool = False
    st_turn_count: int = 0
    latency_ms: int = 0


# ── Chat sessions (ChatGPT-style sidebar) ──────────────────────────

class ChatSessionMeta(BaseModel):
    """One row in the customer's session list."""
    session_id: str
    customer_id: str
    title: str
    last_message_at: Optional[int] = None
    created_at: int
    updated_at: int
    archived: int = 0


class ChatSessionListResponse(BaseModel):
    sessions: List[ChatSessionMeta] = Field(default_factory=list)


class ChatSessionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    title: Optional[str] = None             # backend supplies a default if absent
    session_id: Optional[str] = None        # client may suggest one; backend dedupes


class ChatSessionUpdateRequest(BaseModel):
    title: Optional[str] = None
    archived: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════
# Health / System
# ═══════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: Literal["ok", "starting", "degraded"] = "ok"


class ReadinessCheck(BaseModel):
    name: str
    ok: bool
    detail: Optional[str] = None


class ReadinessResponse(BaseModel):
    ready: bool
    checks: List[ReadinessCheck] = Field(default_factory=list)


class ConfigResponse(BaseModel):
    chat_model: str
    router_model: str
    extractor_model: str
    embedding_model: str
    provider: str
    tools_enabled: Dict[str, bool]


# ═══════════════════════════════════════════════════════════════════
# Order Tool (E-Commerce)
# ═══════════════════════════════════════════════════════════════

class OrderLookupRequest(BaseModel):
    order_id: Optional[str] = None
    customer_id: Optional[str] = None


class OrderToolResponse(BaseModel):
    result: str
    latency_ms: int = 0


# ═══════════════════════════════════════════════════════════════
# Dispute Schemas (core domain models)
# ═══════════════════════════════════════════════════════════════

DisputeTypeEnum = Literal["return", "refund", "damaged", "missing", "fraud", "warranty", "chargeback", "other"]
DisputeStatusEnum = Literal["submitted", "investigating", "resolved", "escalated", "closed"]
DisputeDecisionEnum = Literal["APPROVE", "PARTIAL_REFUND", "REJECT", "ESCALATE", "PENDING"]


class DisputeCreate(BaseModel):
    """POST /disputes — Submit a new dispute."""
    customer_id: str = Field(..., min_length=1)
    order_id: str = Field(..., min_length=1)
    complaint_text: str = Field(..., min_length=10, description="Description of the dispute")
    dispute_type: DisputeTypeEnum = "other"
    evidence_urls: List[str] = Field(default_factory=list)


class DisputeUpdate(BaseModel):
    """PATCH /disputes/{id} — Add evidence or update notes."""
    evidence_urls: Optional[List[str]] = None
    notes: Optional[str] = None


class DisputeResponse(BaseModel):
    """Dispute record returned by POST /disputes and GET /disputes/{id}."""
    dispute_id: str
    order_id: str
    customer_id: str
    dispute_type: str
    status: DisputeStatusEnum = "submitted"
    decision: Optional[DisputeDecisionEnum] = None
    refund_amount: Optional[float] = None
    currency: str = "USD"
    reasoning: Optional[str] = None
    policy_reference: Optional[str] = None
    required_evidence: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DisputeListResponse(BaseModel):
    disputes: List[DisputeResponse] = Field(default_factory=list)
    total: int = 0


# ═══════════════════════════════════════════════════════════════
# Graph RAG (Neo4j query endpoint)
# ═══════════════════════════════════════════════════════════════

class GraphQueryRequest(BaseModel):
    """POST /graph/query — Execute a parameterized Cypher query (admin/debug)."""
    cypher: str = Field(..., min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)


class GraphQueryResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    latency_ms: int = 0


class GraphRAGQueryRequest(BaseModel):
    """POST /graph/rag — Run the Agentic Graph RAG pipeline for a dispute query."""
    query: str = Field(..., min_length=1)


class GraphRAGQueryResponse(BaseModel):
    answer: str
    policy_context: str = ""
    entities: Dict[str, Any] = Field(default_factory=dict)
    retrieval_attempts: int = 1
    retrieval_source: str = "graph"  # graph | vector | hybrid
    latency_ms: int = 0


class GraphSchemaResponse(BaseModel):
    labels: Dict[str, int] = Field(default_factory=dict)
    relationship_types: Dict[str, int] = Field(default_factory=dict)
    total_nodes: int = 0
    total_relationships: int = 0


# ═══════════════════════════════════════════════════════════════════
# RAG tool
# ═══════════════════════════════════════════════════════════════════

class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 4
    threshold: float = 0.5
    use_cache: bool = True


class RAGResponse(BaseModel):
    result: str
    latency_ms: int = 0


class RAGStatsResponse(BaseModel):
    stats: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Web search tool
# ═══════════════════════════════════════════════════════════════════

class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = 5


class WebSearchResponse(BaseModel):
    result: str
    latency_ms: int = 0


# ═══════════════════════════════════════════════════════════════════
# CAG cache tool
# ═══════════════════════════════════════════════════════════════════

class CAGGetRequest(BaseModel):
    query: str = Field(..., min_length=1)


class CAGGetResponse(BaseModel):
    hit: bool
    query: str = ""
    answer: str = ""
    evidence_urls: List[str] = Field(default_factory=list)
    score: float = 0.0
    ts: float = 0.0


class CAGSetRequest(BaseModel):
    query: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    evidence_urls: List[str] = Field(default_factory=list)


class CAGSetResponse(BaseModel):
    cached: bool = True
    query: str


class CAGStatsResponse(BaseModel):
    stats: Dict[str, Any] = Field(default_factory=dict)


class CAGClearResponse(BaseModel):
    cleared: bool = True


# ═══════════════════════════════════════════════════════════════════
# Memory tool
# ═══════════════════════════════════════════════════════════════════

class MemoryRecallRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class FactItem(BaseModel):
    id: Optional[str] = None
    text: str
    tags: List[str] = Field(default_factory=list)
    score: float = 0.0


class MemoryRecallResponse(BaseModel):
    st_turns: List[TurnItem] = Field(default_factory=list)
    lt_facts: List[FactItem] = Field(default_factory=list)


class MemoryFactsResponse(BaseModel):
    user_id: str
    fact_count: int
    facts: List[FactItem] = Field(default_factory=list)


class MemoryStoreFactRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)


class MemoryStoreFactResponse(BaseModel):
    stored: bool = True
    fact_id: Optional[str] = None


class MemoryDistillRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class MemoryDistillResponse(BaseModel):
    distilled_count: int = 0
    triggered: bool


# ═══════════════════════════════════════════════════════════════════
# Crawler tool
# ═══════════════════════════════════════════════════════════════════

class CrawlRequest(BaseModel):
    start_urls: List[str] = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    max_depth: int = 2
    exclude_patterns: List[str] = Field(default_factory=list)
    request_delay: float = 2.0


class CrawledDoc(BaseModel):
    url: str
    title: str = ""
    headings: List[str] = Field(default_factory=list)
    content: str = ""
    depth_level: int = 0


class CrawlResponse(BaseModel):
    doc_count: int
    docs: List[CrawledDoc] = Field(default_factory=list)
    latency_ms: int = 0


# ═══════════════════════════════════════════════════════════════════
# Customers (phone-based identity, not auth)
# ═══════════════════════════════════════════════════════════════════

class CustomerLookupRequest(BaseModel):
    """POST /customers/lookup — phone-based "who is this?"."""
    phone: str = Field(..., min_length=4, description="Any common phone format; normalized server-side")


class CustomerRegisterRequest(BaseModel):
    """POST /customers/register — required fields at sign-up."""
    name: str = Field(..., min_length=1, max_length=200)
    phone: str = Field(..., min_length=4)


class CustomerUpdateRequest(BaseModel):
    """PUT /customers/{customer_id} — profile screen edits."""
    email: Optional[str] = Field(default=None, max_length=200)
    tier: Optional[str] = Field(default=None, max_length=50)


class CustomerResponse(BaseModel):
    """Canonical customer record returned by every /customers/* endpoint."""
    id: str
    name: str
    external_user_id: str       # display form ("+94…")
    email: Optional[str] = None
    tier: str = 'standard'
    total_orders: int = 0
    dispute_count: int = 0
    active: int = 1
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
# Errors
# ═══════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
# Webhooks (Supabase Sync)
# ═══════════════════════════════════════════════════════════════════

class VendorPolicyRecord(BaseModel):
    """Represents a row in the vendor_policies Supabase table."""
    id: str
    vendor_id: str
    policy_name: str = ""
    policy_type: str
    max_return_days: Optional[int] = None
    refund_type: Optional[str] = None
    restocking_fee_percent: Optional[float] = None
    conditions: Optional[Dict[str, Any]] = Field(default_factory=dict)
    approved_by_admin: bool = False

class WebhookPayload(BaseModel):
    """Standard Supabase database webhook payload."""
    type: Literal["INSERT", "UPDATE", "DELETE"]
    table: str
    record: VendorPolicyRecord
    old_record: Optional[Dict[str, Any]] = None
