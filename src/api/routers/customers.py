"""
Customer identity endpoints — phone-based "login", no auth.

Four operations:

  POST /customers/lookup        body {phone}     → 200 CustomerResponse | 404
  POST /customers/register      body {name, phone}
                                                 → 201 CustomerResponse | 409
  GET  /customers/{customer_id}                  → 200 CustomerResponse | 404
  PUT  /customers/{customer_id}  body {email?, tier?}
                                                 → 200 CustomerResponse | 404

Uniqueness is enforced at the database level (``customers.external_user_id`` has a
``UNIQUE`` constraint).
"""

import asyncio
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from infrastructure.db import get_sql_engine
from infrastructure.db.crm_models import Customer

from api.schemas import (
    CustomerLookupRequest,
    CustomerRegisterRequest,
    CustomerResponse,
    CustomerUpdateRequest,
)
from api.utils import normalize_phone


router = APIRouter(prefix="/customers", tags=["Customers"])


# ── Helpers ──────────────────────────────────────────────────────────

def _session():
    """One short-lived session per call. Engine is a process-wide singleton."""
    engine = get_sql_engine()
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()


def _to_response(c: Customer) -> CustomerResponse:
    return CustomerResponse(
        id=c.id,
        name=c.name,
        external_user_id=c.external_user_id,
        email=c.email,
        tier=c.tier or 'standard',
        total_orders=c.total_orders or 0,
        dispute_count=c.dispute_count or 0,
        active=int(c.active or 0),
        created_at=str(c.created_at) if hasattr(c, 'created_at') else None,
        updated_at=str(c.updated_at) if hasattr(c, 'updated_at') else None,
    )


def _find_by_phone(canonical: str) -> Optional[Customer]:
    """Sync DB read — call via ``asyncio.to_thread``."""
    s = _session()
    try:
        return s.query(Customer).filter(Customer.external_user_id == canonical).first()
    finally:
        s.close()


def _find_by_id(customer_id: str) -> Optional[Customer]:
    s = _session()
    try:
        return s.query(Customer).filter(Customer.id == customer_id).first()
    finally:
        s.close()


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/lookup", response_model=CustomerResponse)
async def lookup(req: CustomerLookupRequest) -> CustomerResponse:
    """Find a customer by phone (any common format, normalized server-side)."""
    canonical = normalize_phone(req.phone)
    if not canonical:
        raise HTTPException(status_code=422, detail="Invalid phone number")

    customer = await asyncio.to_thread(_find_by_phone, canonical)
    if customer is None:
        raise HTTPException(status_code=404, detail="No customer with that phone")
    return _to_response(customer)


@router.post("/register", response_model=CustomerResponse, status_code=201)
async def register(req: CustomerRegisterRequest) -> CustomerResponse:
    """Create a new customer row. Returns 409 if the phone is already taken."""
    canonical = normalize_phone(req.phone)
    if not canonical:
        raise HTTPException(status_code=422, detail="Invalid phone number")

    def _insert() -> Customer:
        s = _session()
        try:
            # We don't store created_at since TIMESTAMPTZ handles it, or we can just let DB handle
            c = Customer(
                id=str(uuid.uuid4()),
                external_user_id=canonical,
                name=req.name.strip(),
                email=None,
                tier='standard',
                total_orders=0,
                dispute_count=0,
                active=1,
            )
            s.add(c)
            s.commit()
            s.refresh(c)
            return c
        finally:
            s.close()

    try:
        customer = await asyncio.to_thread(_insert)
    except IntegrityError as exc:
        logger.info("Customer register conflict for {}: {}", canonical, exc.orig)
        raise HTTPException(status_code=409, detail="That phone is already registered")
    except Exception as exc:
        logger.exception("Customer register failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Could not register: {exc}")

    return _to_response(customer)


@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(customer_id: str) -> CustomerResponse:
    """Fetch a full customer profile by id."""
    customer = await asyncio.to_thread(_find_by_id, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return _to_response(customer)


@router.put("/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: str,
    req: CustomerUpdateRequest,
    request: Request,
) -> CustomerResponse:
    """Update editable profile fields (``email``, ``tier``)."""

    def _update() -> Optional[Customer]:
        s = _session()
        try:
            c = s.query(Customer).filter(Customer.id == customer_id).first()
            if c is None:
                return None
            if req.email is not None:
                c.email = req.email.strip() or None
            if req.tier is not None:
                c.tier = req.tier.strip() or None
            s.commit()
            s.refresh(c)
            return c
        finally:
            s.close()

    customer = await asyncio.to_thread(_update)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    # Invalidate any warm session cache entries for this customer
    cache = getattr(request.app.state, "session_cache", None)
    if cache is not None:
        for key in list(cache.keys()):
            if key[0] == customer_id:
                cache.pop(key, None)

    return _to_response(customer)
