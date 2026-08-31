"""
Seed procedural memory with common healthcare workflows.

If ``sql/06_procedures.sql`` exists, loads from that file (deterministic,
identical for all students).  Otherwise falls back to inline definitions.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from infrastructure.log import setup_logging
from memory.procedural_store import ProceduralMemoryStore
from infrastructure.db import create_tables

PROJECT_ROOT = Path(__file__).parent.parent
SQL_SEED_FILE = PROJECT_ROOT / "sql" / "06_procedures.sql"


def _seed_from_sql() -> bool:
    """Load procedural memory from pre-exported SQL file. Returns True on success."""
    if not SQL_SEED_FILE.exists():
        return False

    logger.info("📂 Found sql/06_procedures.sql — loading deterministic data")

    try:
        from sqlalchemy import text
        from infrastructure.db.sql_client import get_sql_engine

        engine = get_sql_engine()
        sql_content = SQL_SEED_FILE.read_text(encoding="utf-8")

        # Strip comment lines BEFORE splitting by semicolon
        lines = [
            line for line in sql_content.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        cleaned_sql = "\n".join(lines)
        statements = [
            s.strip() for s in cleaned_sql.split(";")
            if s.strip()
        ]

        row_count = 0
        with engine.connect() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
                if stmt.upper().startswith("INSERT"):
                    row_count += 1
            conn.commit()

        logger.info(f"  ✅ mem_procedures: {row_count} rows loaded from sql/06_procedures.sql")

        # ── Backfill embeddings (SQL file doesn't include them) ───
        _backfill_embeddings(engine)

        return True
    except Exception as exc:
        logger.error(f"❌ SQL seed failed: {exc}")
        logger.info("   Falling back to inline definitions...")
        return False


def _backfill_embeddings(engine) -> None:
    """Compute and store embeddings for procedures that have NULL embeddings."""
    from sqlalchemy import text
    from infrastructure.llm import get_default_embeddings

    embedder = get_default_embeddings()

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, description, context_when FROM mem_procedures "
            "WHERE embedding IS NULL AND active = TRUE"
        )).fetchall()

        if not rows:
            logger.info("  ✅ All procedures already have embeddings")
            return

        logger.info(f"  🔄 Backfilling embeddings for {len(rows)} procedures...")
        for row in rows:
            embed_text = f"{row.description}. Context: {row.context_when or 'General'}"
            embedding = embedder.embed_query(embed_text)
            conn.execute(
                text(
                    "UPDATE mem_procedures SET embedding = CAST(:emb AS vector) "
                    "WHERE id = :proc_id"
                ),
                {"emb": str(embedding), "proc_id": str(row.id)},
            )
        conn.commit()
        logger.info(f"  ✅ Backfilled {len(rows)} procedure embeddings")


def seed_procedure_if_not_exists(store, name, **kwargs):
    """Helper to seed a procedure only if it doesn't exist."""
    if store.get_procedure_by_name(name):
        logger.info(f"⏭️  Skipping '{name}' - already exists")
        return None
    else:
        proc_id = store.store_procedure(name=name, **kwargs)
        logger.info(f"✅ Added '{name}'")
        return proc_id


def seed_procedures():
    """Seed common healthcare procedures."""
    
    logger.info("🌱 Seeding procedural memory...")
    
    # Ensure tables exist (only if Supabase is configured)
    try:
        create_tables()
        logger.info("✅ Tables verified/created")
    except ValueError as e:
        logger.error(f"❌ Supabase not configured: {e}")
        logger.info("💡 Please set SUPABASE_DB_URL in your .env file to use procedural memory")
        return
    
    # ── Try SQL-first (deterministic) ──────────────────────────
    if _seed_from_sql():
        logger.info("🎯 Procedural memory ready (from SQL)!")
        return
    
    # ── Fallback: inline definitions ───────────────────────────
    logger.info("⚙️  No SQL seed file — using inline definitions")
    
    try:
        store = ProceduralMemoryStore()
    except Exception as e:
        logger.error(f"❌ Failed to initialize ProceduralMemoryStore: {e}")
        logger.info("💡 Make sure Supabase is configured and pgvector extension is enabled")
        return
    
    # =========================================================================
    # 1. BOOK NEW APPOINTMENT
    # =========================================================================
    
    seed_procedure_if_not_exists(
        store,
        name="book_new_appointment",
        description="Book a new appointment for a patient with a doctor at a location",
        context_when="When a patient requests to schedule a new appointment",
        category="booking",
        steps=[
            {
                "order": 1,
                "action": "identify_patient",
                "description": "Verify patient identity using phone number or patient ID. If not found, suggest registration."
            },
            {
                "order": 2,
                "action": "identify_requirements",
                "description": "Ask for: specialty needed, preferred doctor (if any), preferred location, date/time preferences, reason for visit."
            },
            {
                "order": 3,
                "action": "check_availability",
                "description": "Query CRM database for available doctors matching specialty and location within requested timeframe."
            },
            {
                "order": 4,
                "action": "present_options",
                "description": "Present 2-3 available time slots to the patient. Include doctor name, specialty, location, and timing."
            },
            {
                "order": 5,
                "action": "confirm_selection",
                "description": "Ask patient to confirm their choice. Verify all details are correct."
            },
            {
                "order": 6,
                "action": "create_booking",
                "description": "Create booking record in CRM with status='CONFIRMED' and source='MEMORY'."
            },
            {
                "order": 7,
                "action": "send_confirmation",
                "description": "Confirm booking details to patient: appointment ID, date, time, doctor, location, instructions."
            },
            {
                "order": 8,
                "action": "store_in_memory",
                "description": "Store appointment details in long-term memory for future reference."
            }
        ],
        conditions={
            "preconditions": [
                "Patient must be registered in CRM",
                "Doctor must be active and available",
                "Location must be operational"
            ],
            "constraints": [
                "Appointments must be at least 1 hour in the future",
                "No double-booking for same doctor/time slot"
            ]
        },
        examples=[
            "Patient: I need to see a cardiologist next week",
            "Patient: Can I book an appointment with Dr. Silva?",
            "Patient: I want to schedule a checkup at the main store"
        ]
    )
    
    # =========================================================================
    # 2. RESCHEDULE EXISTING APPOINTMENT
    # =========================================================================
    
    seed_procedure_if_not_exists(

    
        store,
        name="reschedule_appointment",
        description="Reschedule an existing appointment to a new date/time",
        context_when="When a patient requests to change an existing appointment time",
        category="booking",
        steps=[
            {
                "order": 1,
                "action": "identify_appointment",
                "description": "Ask for booking ID or appointment details (doctor, date). Query CRM to find the booking."
            },
            {
                "order": 2,
                "action": "verify_ownership",
                "description": "Confirm the appointment belongs to the current patient."
            },
            {
                "order": 3,
                "action": "check_status",
                "description": "Verify appointment status is 'CONFIRMED' or 'PENDING' (not 'CANCELLED' or 'COMPLETED')."
            },
            {
                "order": 4,
                "action": "ask_new_preferences",
                "description": "Ask patient for new preferred date/time. Check if they want same doctor/location."
            },
            {
                "order": 5,
                "action": "check_new_availability",
                "description": "Query CRM for available slots matching new preferences."
            },
            {
                "order": 6,
                "action": "present_options",
                "description": "Present available alternatives to patient."
            },
            {
                "order": 7,
                "action": "confirm_new_time",
                "description": "Ask patient to confirm new appointment time."
            },
            {
                "order": 8,
                "action": "update_booking",
                "description": "Update CRM booking record with new date/time. Set status='RESCHEDULED'."
            },
            {
                "order": 9,
                "action": "send_confirmation",
                "description": "Send updated appointment details to patient."
            },
            {
                "order": 10,
                "action": "update_memory",
                "description": "Update long-term memory with new appointment details."
            }
        ],
        conditions={
            "preconditions": [
                "Appointment must exist in CRM",
                "Appointment must not be in the past",
                "Appointment status must be active"
            ],
            "constraints": [
                "New time must be at least 2 hours in the future",
                "Must maintain minimum 24h notice (configurable)"
            ]
        },
        examples=[
            "Patient: I need to reschedule my appointment with Dr. Perera",
            "Patient: Can I move my appointment to next Monday?",
            "Patient: Something came up, I can't make it tomorrow"
        ]
    )
    
    # =========================================================================
    # 3. CANCEL APPOINTMENT
    # =========================================================================
    
    seed_procedure_if_not_exists(

    
        store,
        name="cancel_appointment",
        description="Cancel an existing appointment",
        context_when="When a patient wants to cancel a scheduled appointment",
        category="booking",
        steps=[
            {
                "order": 1,
                "action": "identify_appointment",
                "description": "Ask for booking ID or appointment details. Query CRM to find the booking."
            },
            {
                "order": 2,
                "action": "verify_ownership",
                "description": "Confirm the appointment belongs to the current patient."
            },
            {
                "order": 3,
                "action": "check_status",
                "description": "Verify appointment is not already cancelled or completed."
            },
            {
                "order": 4,
                "action": "ask_reason",
                "description": "Optionally ask for cancellation reason (for record-keeping)."
            },
            {
                "order": 5,
                "action": "confirm_cancellation",
                "description": "Ask patient to confirm they want to cancel the appointment. Mention cancellation policy if applicable."
            },
            {
                "order": 6,
                "action": "update_booking",
                "description": "Update CRM booking record with status='CANCELLED' and add cancellation reason to notes."
            },
            {
                "order": 7,
                "action": "send_confirmation",
                "description": "Confirm cancellation to patient. Mention rescheduling options if relevant."
            },
            {
                "order": 8,
                "action": "update_memory",
                "description": "Update long-term memory to reflect cancelled status."
            }
        ],
        conditions={
            "preconditions": [
                "Appointment must exist",
                "Appointment must not be completed"
            ],
            "constraints": [
                "Last-minute cancellations may incur fees (policy-dependent)"
            ]
        },
        examples=[
            "Patient: I need to cancel my appointment tomorrow",
            "Patient: Cancel my booking with Dr. Silva please",
            "Patient: I can't make it to my appointment"
        ]
    )
    
    # =========================================================================
    # 4. CHECK APPOINTMENT STATUS
    # =========================================================================
    
    seed_procedure_if_not_exists(

    
        store,
        name="check_appointment_status",
        description="Look up and provide status of patient's appointments",
        context_when="When a patient asks about their upcoming or past appointments",
        category="inquiry",
        steps=[
            {
                "order": 1,
                "action": "identify_patient",
                "description": "Verify patient identity using phone number or patient ID."
            },
            {
                "order": 2,
                "action": "query_appointments",
                "description": "Query CRM for patient's appointments. Filter by timeframe (upcoming, past, all) based on query."
            },
            {
                "order": 3,
                "action": "format_results",
                "description": "Format appointments in readable format: date, time, doctor, location, status, reason."
            },
            {
                "order": 4,
                "action": "present_to_patient",
                "description": "Present appointment list to patient. Highlight upcoming appointments."
            },
            {
                "order": 5,
                "action": "offer_actions",
                "description": "Ask if patient wants to reschedule, cancel, or get more details about any appointment."
            }
        ],
        conditions={
            "preconditions": [
                "Patient must be registered"
            ]
        },
        examples=[
            "Patient: What appointments do I have coming up?",
            "Patient: When is my next doctor visit?",
            "Patient: Show me my appointment history"
        ]
    )
    
    # =========================================================================
    # 5. FIND A DOCTOR
    # =========================================================================
    
    seed_procedure_if_not_exists(

    
        store,
        name="find_doctor",
        description="Help patient find a suitable doctor by specialty or name",
        context_when="When a patient needs to find a doctor but hasn't decided on booking yet",
        category="inquiry",
        steps=[
            {
                "order": 1,
                "action": "understand_needs",
                "description": "Ask patient what they're looking for: specialty, doctor name, location preferences, any specific requirements."
            },
            {
                "order": 2,
                "action": "query_doctors",
                "description": "Query CRM database for matching doctors. Filter by specialty, location, active status."
            },
            {
                "order": 3,
                "action": "present_options",
                "description": "Present 3-5 matching doctors with: name, specialty, locations, general availability."
            },
            {
                "order": 4,
                "action": "offer_booking",
                "description": "Ask if patient wants to book an appointment with any of the listed doctors."
            },
            {
                "order": 5,
                "action": "follow_up",
                "description": "If yes, transition to 'book_new_appointment' procedure. If no, ask if they need more information."
            }
        ],
        conditions={
            "preconditions": []
        },
        examples=[
            "Patient: I need a cardiologist",
            "Patient: Do you have Dr. Fernando?",
            "Patient: Who are your pediatricians in Colombo?"
        ]
    )
    
    # =========================================================================
    # 6. UPDATE PATIENT INFORMATION
    # =========================================================================
    
    seed_procedure_if_not_exists(

    
        store,
        name="update_patient_info",
        description="Update patient's personal information in CRM",
        context_when="When a patient wants to change their contact details, address, or medical info",
        category="administrative",
        steps=[
            {
                "order": 1,
                "action": "identify_patient",
                "description": "Verify patient identity."
            },
            {
                "order": 2,
                "action": "ask_what_to_update",
                "description": "Ask which information needs updating: phone, email, address, emergency contact, etc."
            },
            {
                "order": 3,
                "action": "collect_new_info",
                "description": "Collect the new information from patient. Validate format (e.g., phone number, email)."
            },
            {
                "order": 4,
                "action": "confirm_changes",
                "description": "Repeat back the changes to patient for confirmation."
            },
            {
                "order": 5,
                "action": "update_crm",
                "description": "Update patient record in CRM database."
            },
            {
                "order": 6,
                "action": "confirm_success",
                "description": "Confirm to patient that information has been updated."
            },
            {
                "order": 7,
                "action": "update_memory",
                "description": "Store the update in long-term memory as a fact."
            }
        ],
        conditions={
            "preconditions": [
                "Patient must be registered"
            ],
            "constraints": [
                "Some fields may require verification (e.g., changing registered phone number)"
            ]
        },
        examples=[
            "Patient: I changed my phone number",
            "Patient: Can you update my email address?",
            "Patient: I moved to a new address"
        ]
    )
    
    logger.info("✅ Seeded 6 common healthcare procedures")
    
    # List all procedures
    all_procs = store.list_all_procedures()
    logger.info(f"\n📋 Procedures in database:")
    for proc in all_procs:
        logger.info(f"  - {proc.name} ({proc.category}): {proc.description}")
    
    logger.info("\n🎯 Procedural memory ready!")


if __name__ == "__main__":
    setup_logging()
    seed_procedures()

