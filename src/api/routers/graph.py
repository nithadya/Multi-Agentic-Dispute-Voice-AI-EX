"""
Graph RAG router — Direct Neo4j graph queries + Agentic GraphRAG endpoint.

POST  /graph/query   — Execute raw Cypher (admin/debug only)
POST  /graph/rag     — Run Agentic Graph RAG pipeline for a policy query
GET   /graph/schema  — Introspect the graph schema (node labels, rel types, counts)
"""

import time

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from api.schemas import (
    GraphQueryRequest,
    GraphQueryResponse,
    GraphRAGQueryRequest,
    GraphRAGQueryResponse,
    GraphSchemaResponse,
)

router = APIRouter(prefix="/graph", tags=["graph"])


def _get_neo4j():
    """Get the Neo4j client — raises 503 if not configured."""
    try:
        from infrastructure.db.neo4j_client import get_neo4j_client, neo4j_available
        if not neo4j_available():
            raise HTTPException(
                503,
                detail="Neo4j is not configured. Set NEO4J_URI and NEO4J_PASSWORD in .env",
            )
        return get_neo4j_client()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, detail=f"Neo4j connection failed: {exc}")


@router.post("/query", response_model=GraphQueryResponse, summary="Execute Cypher query")
async def graph_query(body: GraphQueryRequest):
    """
    Execute a parameterized Cypher query against the policy knowledge graph.

    **Admin/debug only** — not for production use.

    Example:
    ```json
    {
        "cypher": "MATCH (dt:DisputeType)-[:GOVERNED_BY]->(p:VendorPolicy) RETURN dt.name, p.name LIMIT 10",
        "params": {}
    }
    ```
    """
    start = time.time()
    client = _get_neo4j()

    try:
        results = client.query(body.cypher, body.params)
    except Exception as exc:
        logger.error("Graph query failed: {}", exc)
        raise HTTPException(400, detail=f"Cypher query error: {exc}")

    return GraphQueryResponse(
        results=results,
        row_count=len(results),
        latency_ms=int((time.time() - start) * 1000),
    )


@router.post("/rag", response_model=GraphRAGQueryResponse, summary="Agentic Graph RAG query")
async def graph_rag_query(body: GraphRAGQueryRequest, request: Request):
    """
    Run the full Agentic Graph RAG pipeline (Week 11 inspired) for a policy question.

    Workflow:
    1. Entity Extraction  → LLM extracts dispute_type, product_category
    2. Graph Retrieval    → Cypher single-hop + multi-hop over vendor policies
    3. Grader             → Relevance check
    4. Rewriter           → CRAG loop (max 2 retries)
    5. Generator          → Policy-grounded decision

    Returns the decision + full graph context for transparency.
    """
    from infrastructure.config import GRAPH_RAG_ENABLED
    if not GRAPH_RAG_ENABLED:
        raise HTTPException(503, detail="Graph RAG is not enabled (GRAPH_RAG_ENABLED=false)")

    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(503, "Agent not ready yet.")

    try:
        from services.chat_service.graph_rag_service import GraphRAGService
        graph_service = GraphRAGService(llm=agent.llm)
        result = graph_service.query(body.query, verbose=False)

        return GraphRAGQueryResponse(
            answer=result["answer"],
            policy_context=result.get("policy_context", ""),
            entities=result.get("entities", {}),
            retrieval_attempts=result.get("retrieval_attempts", 1),
            retrieval_source="graph",
            latency_ms=result.get("latency_ms", 0),
        )
    except Exception as exc:
        logger.error("Graph RAG query failed: {}", exc)
        raise HTTPException(500, detail=f"Graph RAG error: {exc}")


@router.get("/schema", response_model=GraphSchemaResponse, summary="Graph schema info")
async def graph_schema():
    """
    Introspect the Neo4j graph schema.
    Returns node labels, relationship types, and counts.
    """
    start = time.time()
    client = _get_neo4j()

    try:
        schema = client.get_schema_info()
        return GraphSchemaResponse(
            labels=schema.get("labels", {}),
            relationship_types=schema.get("relationship_types", {}),
            total_nodes=schema.get("total_nodes", 0),
            total_relationships=schema.get("total_relationships", 0),
        )
    except Exception as exc:
        logger.error("Schema introspection failed: {}", exc)
        raise HTTPException(500, detail=f"Schema error: {exc}")
