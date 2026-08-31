"""
CRM tool — direct REST access to ``src/agents/tools/crm_tool.py``.

Each endpoint wraps one ``CRMTool`` action and returns the tool's plain-
text output. Clients that want structured data can call the underlying
Supabase directly; these endpoints exist so the same surface available
to the agent (and to MCP hosts via ``crm_server.py``) is reachable over
HTTP.
"""

import asyncio
import time

from fastapi import APIRouter, Depends

from api.deps import get_crm_tool
from api.schemas import (
    CRMCancelBookingRequest,
    CRMCreateBookingRequest,
    CRMLookupPatientRequest,
    CRMRescheduleBookingRequest,
    CRMResponse,
    CRMSearchDoctorsRequest,
)


router = APIRouter(prefix="/tools/crm", tags=["Tools — CRM"])


async def _run(fn, /, **kwargs) -> CRMResponse:
    """Run a sync CRM method in a worker thread and wrap the result."""
    t0 = time.perf_counter()
    result = await asyncio.to_thread(fn, **kwargs)
    return CRMResponse(result=result, latency_ms=int((time.perf_counter() - t0) * 1000))


@router.post("/lookup_patient", response_model=CRMResponse)
async def lookup_patient(
    req: CRMLookupPatientRequest,
    crm=Depends(get_crm_tool),
) -> CRMResponse:
    return await _run(
        crm.lookup_patient,
        phone=req.phone,
        name=req.name,
        patient_id=req.patient_id,
        external_user_id=req.external_user_id,
    )


@router.post("/search_doctors", response_model=CRMResponse)
async def search_doctors(
    req: CRMSearchDoctorsRequest,
    crm=Depends(get_crm_tool),
) -> CRMResponse:
    # NOTE: CRMTool.search_doctors() supports specialty/location/doctor_name only.
    # The availability filters were never implemented on the tool, so forwarding
    # them raised TypeError → HTTP 500. Match the tool's real signature.
    return await _run(
        crm.search_doctors,
        specialty=req.specialty,
        location=req.location,
        doctor_name=req.doctor_name,
    )


@router.post("/create_booking", response_model=CRMResponse)
async def create_booking(
    req: CRMCreateBookingRequest,
    crm=Depends(get_crm_tool),
) -> CRMResponse:
    return await _run(
        crm.create_booking,
        patient_id=req.patient_id,
        doctor_id=req.doctor_id,
        start_time=req.start_time,
        duration_minutes=req.duration_minutes,
        notes=req.notes,
    )


@router.post("/cancel_booking", response_model=CRMResponse)
async def cancel_booking(
    req: CRMCancelBookingRequest,
    crm=Depends(get_crm_tool),
) -> CRMResponse:
    return await _run(crm.cancel_booking, booking_id=req.booking_id, reason=req.reason)


@router.post("/reschedule_booking", response_model=CRMResponse)
async def reschedule_booking(
    req: CRMRescheduleBookingRequest,
    crm=Depends(get_crm_tool),
) -> CRMResponse:
    return await _run(
        crm.reschedule_booking,
        booking_id=req.booking_id,
        new_start_time=req.new_start_time,
    )
