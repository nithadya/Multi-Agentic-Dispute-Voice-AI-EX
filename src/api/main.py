"""
FastAPI application for the E-Commerce Dispute Resolution AI.

Architecture:
    Sync HTTP contract · async internals · CAG short-circuit for common
    queries · Hybrid GraphRAG (Neo4j + Qdrant) on cache miss · multi-agent
    LangGraph pipeline for dispute resolution.

Endpoints:
    POST  /chat                          — main conversational endpoint
    POST  /chat/reset                    — clear session ST memory
    GET   /sessions/{sid}/turns          — debug: dump recent ST turns
    GET   /health, /ready, /config       — system status

    POST  /disputes                      — submit new dispute
    GET   /disputes/{id}                 — check dispute status + decision
    PATCH /disputes/{id}                 — add evidence / notes
    GET   /disputes                      — list customer disputes

    POST  /graph/query                   — raw Cypher query (admin)
    POST  /graph/rag                     — Agentic Graph RAG query
    GET   /graph/schema                  — graph schema introspection

    POST  /tools/rag/search              — internal KB retrieval
    GET   /tools/rag/stats               — RAG cache stats
    POST  /tools/web_search              — Tavily real-time search
    POST  /tools/cag/{get,set,clear}     — semantic cache CRUD
    POST  /tools/memory/{recall,store_fact,distill}
    POST  /tools/crawl                   — Playwright BFS crawl

Start:
    cd src && uvicorn api.main:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""

import asyncio
import os
import pathlib
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Ensure src/ is on the path regardless of launch cwd
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv
load_dotenv()

from api.middleware import install_middleware
from api.routers import chat as chat_router
from api.routers import chat_sessions as chat_sessions_router
from api.routers import health as health_router
from api.routers import customers as customers_router
from api.routers import disputes as disputes_router
from api.routers import graph as graph_router
from api.routers import voice as voice_router
from api.routers import webhooks as webhooks_router
from api.routers.tools import cag as cag_router
from api.routers.tools import crawl as crawl_router
from api.routers.tools import memory as memory_router
from api.routers.tools import rag as rag_router
from api.routers.tools import web as web_router


# ── Lifespan — warm heavy singletons before accepting traffic ────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build the agent (LLM clients, DB pools, Qdrant, embedder) once at
    startup. Runs in a worker thread to keep the event loop responsive.

    Warmup steps performed before accepting traffic:
      1. Pre-fetch every Langfuse prompt → eliminates per-request hops.
      2. Run a one-token LLM call against the router model → warms the
         Groq HTTP pool / TCP / TLS, removing cold-start tax on the first
         user request.
      3. Run a one-vector embed → warms OpenAI embeddings client.
    """
    logger.info("Starting DisputeAI E-Commerce Specialist API...")

    from agents.orchestrator import build_agent
    from agents.prompts.agent_prompts import LANGFUSE_PROMPT_NAMES as AGENT_PROMPTS
    from memory.prompts import LANGFUSE_PROMPT_NAMES as MEMORY_PROMPTS
    from infrastructure.llm.embeddings import get_default_embeddings, get_local_embedder
    from infrastructure.observability import prefetch_prompts
    from services.chat_service.cag_cache import CAGCache

    # Every Langfuse prompt name used anywhere in the codebase. Listing
    # them here means none ever incurs a network hop on the request path
    # — the per-call ``fetch_prompt()`` always finds it in cache.
    ALL_PROMPT_NAMES = list(AGENT_PROMPTS.values()) + list(MEMORY_PROMPTS.values())

    # Heavy, mostly I/O-bound — off the event loop
    agent = await asyncio.to_thread(
        build_agent, enable_crm=True, enable_rag=True, enable_web=True,
    )
    embedder = agent.rag_tool.embedder if agent.rag_tool else get_default_embeddings()

    # ── Local-embedded CAG cache ─────────────────────────────────
    # The chat hot path's CAG short-circuit lookup originally ran on a local
    # sentence-transformers model. We have switched it to use the default
    # OpenAI embedder to free up ~400MB of RAM and prevent Docker OOM crashes
    # when the Voice agent and GraphRAG are running simultaneously.
    local_embedder = await asyncio.to_thread(get_default_embeddings)

    # Drop any stale cag_cache_local from the previous container run
    # before warmup. The collection lives on Qdrant Cloud (persistent),
    # so old "I apologize / not available" answers from earlier code
    # paths out-rank the real FAQ entries we're about to load. A
    # clean recreate every startup is cheap (collection is small) and
    # makes FAQ correctness deterministic.
    try:
        from infrastructure.db.qdrant_client import get_qdrant_client, collection_exists
        _qc = get_qdrant_client()
        if collection_exists("cag_cache_local"):
            await asyncio.to_thread(_qc.delete_collection, "cag_cache_local")
            logger.info("CAG cache_local: dropped stale collection for clean warmup")
    except Exception as _exc:
        logger.warning("CAG cache_local purge skipped: {}", _exc)

    cag_cache = await asyncio.to_thread(
        CAGCache,
        local_embedder,
        "cag_cache_local",
        1536, # OpenAI embedding dimension
    )

    # Make the RAG tool's internal cache point at the same local cache —
    # this keeps the /chat short-circuit and the RAG pipeline coherent
    # (a query the agent answered via RAG can still hit the cache on a
    # subsequent /chat call).
    if agent.rag_tool is not None:
        agent.rag_tool._cache = cag_cache
        cag_service = getattr(agent.rag_tool, "_cag_service", None)
        if cag_service is not None:
            cag_service.cache = cag_cache

    # Attach the cache to the orchestrator so the decision LangGraph's
    # cag_node (which reads ``agent.cag_cache`` via a getter closure)
    # can perform semantic lookups during the parallel classification
    # phase. The graph itself was compiled at orchestrator-build time
    # before this cache existed; the late binding is intentional.
    agent.cag_cache = cag_cache

    app.state.agent = agent
    app.state.embedder = embedder              # remote, for LT memory + RAG index
    app.state.local_embedder = local_embedder  # local, for CAG short-circuit
    app.state.cag_cache = cag_cache

    # ── Per-session warm cache ──────────────────────────────────
    # Keyed by ``(patient_id, session_id)``. Populated by the
    # ``/sessions/warmup`` endpoint at login (and on session switch),
    # then maintained write-through by the chat hot path.
    #
    # Holds:
    #   patient:  full Patient row as a dict
    #   st_turns: most recent ConversationTurns
    #
    # When the chat endpoint finds an entry here it can skip both the
    # patient PK lookup and the ST recall round-trip — those happen
    # during the dead time between login and first message.
    app.state.session_cache = {}

    # ── Warmup: prompts + LLM pools + embedder ──────────────────
    async def _warmup_prompts():
        try:
            await asyncio.to_thread(prefetch_prompts, ALL_PROMPT_NAMES)
        except Exception as exc:
            logger.warning("Prompt prefetch failed: {}", exc)

    async def _warmup_router():
        try:
            await agent.router.aroute("ping", "")
        except Exception as exc:
            logger.debug("Router warmup error (non-fatal): {}", exc)

    async def _warmup_fast_llm():
        try:
            await agent.llm_fast.ainvoke("hi")
        except Exception as exc:
            logger.debug("Fast LLM warmup error (non-fatal): {}", exc)

    async def _warmup_embedder():
        try:
            await asyncio.to_thread(embedder.embed_query, "warmup")
        except Exception as exc:
            logger.debug("Embedder warmup error (non-fatal): {}", exc)

    async def _warmup_faqs_cache():
        """
        Seed CAG with the curated patient FAQs from ``config/faqs.yaml``.
        Each entry already has a hand-written ``answer`` so we bypass
        CRAG and write directly into the cache — sub-200 ms hits for
        common patient questions without any LLM call.
        """
        if cag_cache is None:
            return
        try:
            from infrastructure.config import KNOWN_FAQS
        except Exception as exc:
            logger.debug("FAQ load skipped: {}", exc)
            return

        faqs = [
            f for f in (KNOWN_FAQS or [])
            if isinstance(f, dict) and f.get("query") and f.get("answer")
        ]
        if not faqs:
            return

        cached = 0
        for entry in faqs:
            try:
                await asyncio.to_thread(
                    cag_cache.set,
                    str(entry["query"]),
                    {"answer": str(entry["answer"]), "evidence_urls": []},
                )
                cached += 1
            except Exception as exc:
                logger.debug("CAG FAQ warmup failed for '{}': {}",
                             entry.get("query", "?")[:60], exc)
        if cached:
            logger.info("Pre-warmed CAG with {}/{} FAQ entries from faqs.yaml",
                        cached, len(faqs))



    # Run all warmups concurrently — total time is max(individual), not sum
    await asyncio.gather(
        _warmup_prompts(),
        _warmup_router(),
        _warmup_fast_llm(),
        _warmup_embedder(),
        _warmup_faqs_cache(),
    )

    logger.success("API ready — agent + CAG cache online (warmup complete)")

    try:
        yield
    finally:
        logger.info("Shutting down API...")
        mcp_client = getattr(agent, "mcp_client", None)
        if mcp_client is not None:
            try:
                close = getattr(mcp_client, "aclose", None) or getattr(mcp_client, "close", None)
                if close:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as exc:
                logger.warning("MCP client shutdown raised: {}", exc)


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="E-Commerce Dispute Resolution AI",
    description=(
        "Autonomous Hierarchical Multi-Agent Framework with Hybrid GraphRAG "
        "(Neo4j + Qdrant) for e-commerce dispute resolution. "
        "Inspired by AEE Week 11 Graph RAG."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # teaching project — tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

install_middleware(app)


# ── SPA ⇄ API path bridge ─────────────────────────────────────────────
# The React SPA (served at /) calls the backend over relative ``/api/*``
# paths — that's how the Vite dev proxy is configured, and keeping the
# same convention in production means the browser hits a single origin
# (no CORS, one ALB hostname). The real API routes have no ``/api``
# prefix (``/chat``, ``/voice/token`` …), so this middleware strips the
# leading ``/api`` before routing. Runs before the router, so the rewrite
# is transparent to every endpoint.
@app.middleware("http")
async def _strip_api_prefix(request, call_next):
    path = request.scope.get("path", "")
    if path == "/api" or path == "/api/":
        request.scope["path"] = "/"
        request.scope["raw_path"] = b"/"
    elif path.startswith("/api/"):
        new_path = path[4:]  # drop the leading "/api"
        request.scope["path"] = new_path
        request.scope["raw_path"] = new_path.encode("utf-8")
    return await call_next(request)


# Keep browsers from serving stale dynamic content. Two cases:
#   - HTML (the SPA index) → ``no-cache`` so a freshly deployed bundle hash
#     is always picked up on the next load. A cached index.html was pinning
#     users to an old JS bundle.
#   - JSON (every API response) → ``no-store`` so the live session list /
#     patient data is never cached; the sidebar always reflects the DB. A
#     cached empty session-list response was the root cause of a sidebar
#     that looked permanently empty while the database had the data.
# Hashed JS/CSS assets are immutable, so they are left cacheable.
@app.middleware("http")
async def _cache_control(request, call_next):
    response = await call_next(request)
    ctype = response.headers.get("content-type", "")
    if ctype.startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    elif ctype.startswith("application/json"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ── Routers ──────────────────────────────────────────────────────────

app.include_router(health_router.router)
app.include_router(customers_router.router)
app.include_router(disputes_router.router)
app.include_router(graph_router.router)
app.include_router(chat_sessions_router.router)
app.include_router(chat_router.router)
app.include_router(rag_router.router)
app.include_router(web_router.router)
app.include_router(cag_router.router)
app.include_router(memory_router.router)
app.include_router(crawl_router.router)
app.include_router(voice_router.router)
app.include_router(webhooks_router.router)


# ── SPA static files ──────────────────────────────────────────────────
# In the container the compiled React build lives at /app/ui_dist (copied
# by docker/api/Dockerfile from the Stage-0 node build). __file__ is
# /app/src/api/main.py, so parents[2] == /app. When the build is present
# we mount it at / with ``html=True`` so "/" serves index.html and
# /assets/* serve the bundles. This mount is registered LAST, so every
# real route (/chat, /health, /docs, …) is matched first; only unmatched
# paths fall through to the SPA. Locally (no build) we keep a JSON root.
_UI_DIST = pathlib.Path(__file__).resolve().parents[2] / "ui_dist"

if _UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_UI_DIST), html=True), name="spa")
    logger.info("SPA mounted from {}", _UI_DIST)
else:
    @app.get("/", tags=["System"])
    async def root():
        """Friendly landing page pointer (no SPA build present)."""
        return {
            "service": "DisputeAI E-Commerce Specialist API",
            "version": app.version,
            "docs": "/docs",
            "redoc": "/redoc",
        }
