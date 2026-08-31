"""
CRM MCP Server — exposes the existing CRMTool over the Model Context Protocol.

This is a thin wrapper around `src/agents/tools/crm_tool.py`. The business
logic (SQLAlchemy queries, conflict checks, Supabase writes) stays exactly
where it was — this file only adds the MCP transport layer so that ANY MCP
client (LangGraph agent, Claude Desktop, Cursor, MCP Inspector) can call it.

Transport: stdio (default for local servers)

Run standalone:
    python -m mcp_servers.crm_server

Inspect interactively:
    npx @modelcontextprotocol/inspector python -m mcp_servers.crm_server

IMPORTANT — stdio gotcha:
    MCP over stdio uses stdout for the JSON-RPC protocol. NEVER `print()`
    to stdout from inside this process. All logging must go to stderr
    (loguru's default sink is stderr, so we're safe).
"""

import os
import sys

# Ensure src/ is importable when run as a script
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from mcp.server.fastmcp import FastMCP

from agents.tools.crm_tool import CRMTool


# ── Server + tool instance ──────────────────────────────────────

mcp = FastMCP("disputeai-crm")

# Lazy-init so that --help / import-only uses don't hit the DB
_crm: CRMTool | None = None


def _get_crm() -> CRMTool:
    global _crm
    if _crm is None:
        logger.info("Initialising CRMTool inside MCP server...")
        _crm = CRMTool()
    return _crm


# ── MCP Tools ───────────────────────────────────────────────────

@mcp.tool()
def lookup_patient(
    phone: str | None = None,
    name: str | None = None,
    patient_id: str | None = None,
) -> str:
    """
    Find a patient record by phone, name, or patient_id.

    Returns patient details plus their upcoming bookings (if any).
    Supply exactly one of: phone, name, or patient_id.
    """
    return _get_crm().dispatch(
        "lookup_patient",
        {"phone": phone, "name": name, "patient_id": patient_id},
    )


@mcp.tool()
def search_doctors(
    specialty: str | None = None,
    location: str | None = None,
    name: str | None = None,
) -> str:
    """
    Search doctors by specialty (e.g. 'cardiology'), location, or name.

    Returns a list of matching active doctors with their IDs.
    """
    return _get_crm().dispatch(
        "search_doctors",
        {"specialty": specialty, "location": location, "name": name},
    )


@mcp.tool()
def create_booking(
    patient_id: str,
    doctor_id: str,
    location_id: str,
    start_time: str,
    duration_minutes: int = 30,
    title: str = "Consultation",
    reason: str | None = None,
) -> str:
    """
    Book a new appointment.

    Args:
        patient_id: Patient identifier (from lookup_patient)
        doctor_id: Doctor identifier (from search_doctors)
        location_id: Location identifier
        start_time: Datetime string in format 'YYYY-MM-DD HH:MM'
        duration_minutes: Visit duration (default 30)
        title: Booking title (default 'Consultation')
        reason: Optional reason for the visit
    """
    return _get_crm().dispatch(
        "create_booking",
        {
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "location_id": location_id,
            "start_time": start_time,
            "duration_minutes": duration_minutes,
            "title": title,
            "reason": reason,
        },
    )


@mcp.tool()
def cancel_booking(booking_id: str) -> str:
    """Cancel an existing booking by its booking_id."""
    return _get_crm().dispatch("cancel_booking", {"booking_id": booking_id})


@mcp.tool()
def reschedule_booking(
    booking_id: str,
    new_start_time: str,
    duration_minutes: int = 30,
) -> str:
    """
    Reschedule an existing booking to a new start time.

    Args:
        booking_id: Existing booking ID
        new_start_time: Datetime string 'YYYY-MM-DD HH:MM'
        duration_minutes: New duration (default 30)
    """
    return _get_crm().dispatch(
        "reschedule_booking",
        {
            "booking_id": booking_id,
            "new_start_time": new_start_time,
            "duration_minutes": duration_minutes,
        },
    )


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting disputeai-crm MCP server on stdio...")
    mcp.run()
