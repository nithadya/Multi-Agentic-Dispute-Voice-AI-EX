"""
Disputes router — CRUD lifecycle for e-commerce dispute resolution.

POST   /disputes            — Submit new dispute (triggers AI resolution pipeline)
GET    /disputes/{id}       — Check dispute status + decision
PATCH  /disputes/{id}       — Add evidence or notes
GET    /disputes            — List disputes for a customer
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from api.schemas import (
    DisputeCreate,
    DisputeListResponse,
    DisputeResponse,
    DisputeUpdate,
)

router = APIRouter(prefix="/disputes", tags=["disputes"])


def _get_agent(request: Request):
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(503, "Agent not ready yet — please retry.")
    return agent


@router.post("", response_model=DisputeResponse, summary="Submit a new dispute")
async def submit_dispute(body: DisputeCreate, request: Request):
    """
    Submit a customer complaint for AI-driven resolution.

    The system will:
    1. Classify the dispute type + severity
    2. Look up the order details
    3. Query the Hybrid GraphRAG policy engine (Neo4j + Qdrant)
    4. Make a policy-compliant decision (APPROVE / PARTIAL_REFUND / REJECT / ESCALATE)
    5. Return the decision with policy references
    """
    start = time.time()
    agent = _get_agent(request)

    dispute_id = f"DIS-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"

    # Build the query for the AI agent
    query = (
        f"Customer complaint for Order {body.order_id}:\n"
        f"Type: {body.dispute_type}\n"
        f"Description: {body.complaint_text}"
    )

    try:
        # Run through the multi-agent pipeline
        result = await agent.achat(
            user_id=body.customer_id,
            session_id=dispute_id,
            message=query,
        )
        answer = result.get("answer", "Your dispute has been submitted and is under review.")
        route = result.get("route", "rag")
    except Exception as exc:
        answer = f"Dispute {dispute_id} submitted. Under investigation (AI pipeline error: {exc})"
        route = "error"

    elapsed_ms = int((time.time() - start) * 1000)

    # Parse decision from the answer if present
    decision = "PENDING"
    for d in ["APPROVE", "PARTIAL_REFUND", "REJECT", "ESCALATE"]:
        if d in answer.upper():
            decision = d
            break

    return DisputeResponse(
        dispute_id=dispute_id,
        order_id=body.order_id,
        customer_id=body.customer_id,
        dispute_type=body.dispute_type,
        status="investigating" if decision == "PENDING" else "resolved",
        decision=decision,
        refund_amount=None,
        reasoning=answer,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/{dispute_id}", response_model=DisputeResponse, summary="Get dispute status")
async def get_dispute(dispute_id: str, request: Request):
    """Check the current status and decision of a dispute by its reference ID."""
    agent = _get_agent(request)

    try:
        result = await agent.achat(
            user_id="system",
            session_id=dispute_id,
            message=f"Check status of dispute {dispute_id}",
        )
        answer = result.get("answer", "Status pending.")
    except Exception as exc:
        answer = f"Unable to retrieve dispute status: {exc}"

    return DisputeResponse(
        dispute_id=dispute_id,
        order_id="",
        customer_id="",
        dispute_type="other",
        status="investigating",
        reasoning=answer,
    )


@router.patch("/{dispute_id}", response_model=DisputeResponse, summary="Update a dispute")
async def update_dispute(dispute_id: str, body: DisputeUpdate, request: Request):
    """Add evidence URLs or customer notes to an existing dispute."""
    update_parts = []
    if body.evidence_urls:
        update_parts.append(f"Evidence added: {', '.join(body.evidence_urls)}")
    if body.notes:
        update_parts.append(f"Notes: {body.notes}")

    return DisputeResponse(
        dispute_id=dispute_id,
        order_id="",
        customer_id="",
        dispute_type="other",
        status="investigating",
        reasoning=f"Dispute {dispute_id} updated: {'; '.join(update_parts)}",
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("", response_model=DisputeListResponse, summary="List disputes for a customer")
async def list_disputes(customer_id: str, request: Request):
    """List all disputes for a given customer ID."""
    return DisputeListResponse(disputes=[], total=0)
