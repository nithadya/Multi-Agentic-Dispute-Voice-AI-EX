"""
Friendly labels for the chain-of-thought timeline.

Maps internal stage / tool identifiers to human-readable strings the
UI shows users while their request is in flight. Keeps the wording
in one place so the chain-of-thought never says "lookup_patient" but
"Looking up your appointments in the CRM".
"""

from typing import Optional, Tuple


STAGE_LABELS: dict[str, str] = {
    "cache":     "Looking in cache for similar questions",
    "recall_st": "Loading your conversation history",
    "recall_lt": "Searching your long-term memory",
    "patient":   "Verifying your patient profile",
    "route":     "Routing your question to the right tool",
    "guardrail": "Checking the question is in scope",
    "tool":      "Running the chosen tool",
    "synth":     "Composing your reply",
    "save":      "Saving the conversation",
}


# Tool-level labels keyed by (route, action). action is None for routes
# without a sub-action (rag / web_search / direct).
_TOOL_LABELS: dict[Tuple[str, Optional[str]], str] = {
    ("crm", "lookup_patient"):     "Looking up your appointments in the CRM",
    ("crm", "search_doctors"):     "Searching the doctor directory",
    ("crm", "create_booking"):     "Creating your booking in the CRM",
    ("crm", "cancel_booking"):     "Cancelling the booking in the CRM",
    ("crm", "reschedule_booking"): "Rescheduling your booking in the CRM",
    ("crm", "list_specialties"):   "Pulling the list of departments",
    ("crm", "list_locations"):     "Pulling the list of branches",
    ("crm", "check_doctor_availability"): "Checking the doctor's open time slots",
    ("rag", None):                 "Searching the policy database",
    ("web_search", None):          "Searching live web sources (Tavily)",
    ("direct", None):              "Composing a direct reply",
    ("multi", None):               "Running multiple tools in parallel",
    ("cag_hit", None):             "Returning a cached answer",
    ("out_of_scope", None):        "Politely declining — outside store domain",
}


def tool_label(route: str, action: Optional[str] = None) -> str:
    """Friendly label for a single tool invocation."""
    return (
        _TOOL_LABELS.get((route, action))
        or _TOOL_LABELS.get((route, None))
        or f"Running {route}{' / ' + action if action else ''}"
    )


def stage_label(stage: str) -> str:
    """Friendly label for a pipeline stage."""
    return STAGE_LABELS.get(stage, stage.replace("_", " ").capitalize())
