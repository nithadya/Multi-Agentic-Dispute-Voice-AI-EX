"""
CRM Tool — appointments, doctors, bookings, store reference.

Seven actions exposed to the routing engine:

  1. lookup_patient      — list/check appointments with optional filters
                            (start_date, end_date, doctor_name, specialty, status)
  2. search_doctors      — find doctors by specialty / location / name
  3. create_booking      — book a new appointment
  4. cancel_booking      — cancel by booking_id OR by (doctor_name + dates)
  5. reschedule_booking  — move an appointment to a new ISO datetime
  6. list_specialties    — every active department + active doctor count
  7. list_locations      — every active branch / clinic / lab

Design contract with the router:
    - The router LLM is responsible for resolving natural language into
      typed inputs: ``start_date`` / ``end_date`` (YYYY-MM-DD) and
      ``start_at`` / ``new_start_at`` (full ISO datetime including
      timezone). The tool itself does NO keyword-based parsing of
      "today", "tomorrow", "next monday" etc. — that's the router's job.
    - Patient identity (``patient_id``) is auto-injected at the API layer
      from the authenticated session. Tools never trust client-supplied
      patient ids except as a "find this *other* patient" lookup, which
      we don't expose to chat.

All tool output is plain markdown — when more than one booking is
returned, it's a markdown table so the UI renders it natively.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import sessionmaker

from infrastructure.config import TIMEZONE
from infrastructure.db import get_sql_engine
from infrastructure.db.crm_models import Booking, Doctor, Location, Patient, Specialty
from infrastructure.observability import observe, update_current_observation


# Booking statuses considered "active" (not cancelled / completed / no-show).
_ACTIVE_STATUSES = ["PENDING", "CONFIRMED", "RESCHEDULED"]


class CRMTool:
    """CRM tool — every public method is one routable action."""

    def __init__(self) -> None:
        self.engine = get_sql_engine()
        self._tz = ZoneInfo(TIMEZONE)

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _session(self):
        return sessionmaker(bind=self.engine, expire_on_commit=False)()

    def _epoch_to_local(self, epoch: int) -> str:
        """Format epoch as 'YYYY-MM-DD HH:MM' in the store timezone."""
        return datetime.fromtimestamp(epoch, self._tz).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _doctor_label(full_name: Optional[str]) -> str:
        """
        Render a doctor's name with exactly one ``Dr.`` prefix.

        Some seed rows already start with ``Dr.``; naïvely prepending
        another one produced "Dr. Dr. Kusal Fernando" in the UI.
        """
        if not full_name:
            return "—"
        name = full_name.strip()
        lower = name.lower()
        if lower.startswith("dr.") or lower.startswith("dr "):
            return name
        return f"Dr. {name}"

    def _parse_iso_datetime(self, s: str) -> Optional[int]:
        """
        Parse ``YYYY-MM-DDTHH:MM:SS+HH:MM`` (or naïve ``YYYY-MM-DD HH:MM``)
        into a UTC epoch. Returns None on failure.

        The router emits full ISO timestamps; we accept both forms
        defensively so a slightly different shape doesn't crash the
        request.
        """
        if not s:
            return None
        text = s.strip().replace(" ", "T")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            # second-chance for "YYYY-MM-DD HH:MM"
            try:
                dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
            except ValueError:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._tz)
        return int(dt.timestamp())

    def _date_window(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Optional[tuple[int, int]]:
        """
        Convert a (start_date, end_date) pair (each ``YYYY-MM-DD`` or None)
        into a (start_epoch, end_epoch) window in the store timezone.

        - start_date: midnight at start of that day (inclusive).
        - end_date  : 23:59:59 at end of that day (inclusive).
        Returns None if both are None (caller treats as "no window").
        """
        if not start_date and not end_date:
            return None
        try:
            sd = datetime.fromisoformat(start_date).date() if start_date else None
            ed = datetime.fromisoformat(end_date).date() if end_date else None
        except ValueError:
            return None

        if sd is None and ed is not None:
            sd = ed
        if ed is None and sd is not None:
            ed = sd

        start = datetime.combine(sd, datetime.min.time(), tzinfo=self._tz)
        end = datetime.combine(ed, datetime.max.time(), tzinfo=self._tz)
        return int(start.timestamp()), int(end.timestamp())

    def _bookings_query(
        self,
        session,
        *,
        patient_id: str,
        start_epoch: Optional[int] = None,
        end_epoch: Optional[int] = None,
        doctor_name: Optional[str] = None,
        specialty: Optional[str] = None,
        status: str = "all",
    ):
        """
        Build the canonical bookings query used by lookup, cancel, and
        reschedule. Joins doctors + specialties + locations once so we
        can filter on doctor / specialty names at the SQL level instead
        of doing N+1 lookups.
        """
        q = (
            session.query(Booking)
            .join(Doctor, Doctor.doctor_id == Booking.doctor_id)
            .outerjoin(Specialty, Specialty.specialty_id == Doctor.specialty_id)
            .outerjoin(Location, Location.location_id == Booking.location_id)
            .filter(Booking.patient_id == patient_id)
        )

        if start_epoch is not None:
            q = q.filter(Booking.start_at >= start_epoch)
        if end_epoch is not None:
            q = q.filter(Booking.start_at <= end_epoch)

        if doctor_name:
            q = q.filter(Doctor.full_name.ilike(f"%{doctor_name}%"))
        if specialty:
            q = q.filter(Specialty.name.ilike(f"%{specialty}%"))

        status_lc = (status or "not_cancelled").lower()
        if status_lc == "active":
            q = q.filter(Booking.status.in_(_ACTIVE_STATUSES))
        elif status_lc == "completed":
            q = q.filter(Booking.status == "COMPLETED")
        elif status_lc == "cancelled":
            q = q.filter(Booking.status == "CANCELLED")
        elif status_lc == "not_cancelled":
            # Default for end-user queries: hide cancelled rows entirely
            # so "do I have any bookings?" doesn't surface ghosts.
            q = q.filter(Booking.status != "CANCELLED")
        # status="all" → no filter (still available when explicitly asked)

        return q

    def _format_booking_row(self, session, bk: Booking) -> Dict[str, str]:
        """Single booking → flat dict used to build markdown tables."""
        doctor = session.get(Doctor, bk.doctor_id)
        loc = session.get(Location, bk.location_id)
        spec = doctor.specialty.name if (doctor and doctor.specialty) else "—"
        return {
            "when": self._epoch_to_local(bk.start_at),
            "doctor": self._doctor_label(doctor.full_name if doctor else None),
            "specialty": spec,
            "location": loc.name if loc else "—",
            "status": bk.status,
        }

    def _bookings_to_md_table(self, rows: List[Dict[str, str]]) -> str:
        """
        Render a list of formatted booking rows as a markdown table.
        IDs are deliberately omitted — the user does not need to see
        opaque UUIDs and they crowded the columns. Disambiguation works
        by date + doctor instead.
        """
        if not rows:
            return ""
        header = "| When | Doctor | Specialty | Location | Status |"
        sep = "|---|---|---|---|---|"
        lines = [header, sep]
        for r in rows:
            lines.append(
                f"| {r['when']} | {r['doctor']} | {r['specialty']} | "
                f"{r['location']} | {r['status']} |"
            )
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    # Patient resolution — UUID or external_user_id (phone)
    # ─────────────────────────────────────────────────────────────

    def _resolve_patient(self, session, key: Optional[str]) -> Optional[Patient]:
        """
        Look up a Patient by either the canonical UUID
        (``patients.patient_id``) or the phone-style
        ``external_user_id`` that the orchestrator injects from
        ``state["user_id"]``.

        Returns the Patient row or ``None``. Callers should always
        re-bind their local ``patient_id`` to ``patient.patient_id``
        before passing it to downstream queries / inserts so foreign
        keys land on the canonical UUID.
        """
        if not key:
            return None
        # 1) primary key (UUID)
        patient = session.get(Patient, key)
        if patient is not None:
            return patient
        # 2) external_user_id (phone) — what voice/text orchestrator pass
        return (
            session.query(Patient)
            .filter(Patient.external_user_id == key)
            .first()
        )

    # ─────────────────────────────────────────────────────────────
    # 1. lookup_patient — list/check appointments with filters
    # ─────────────────────────────────────────────────────────────

    @observe(name="crm_lookup_patient")
    def lookup_patient(
        self,
        patient_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        doctor_name: Optional[str] = None,
        specialty: Optional[str] = None,
        status: str = "not_cancelled",
        limit: int = 10,
        # Legacy fall-throughs — accepted but only used as last resort.
        phone: Optional[str] = None,
        name: Optional[str] = None,
        external_user_id: Optional[str] = None,
    ) -> str:
        """
        List bookings for a patient.

        All filters are optional. The router LLM is responsible for
        translating natural language into ``start_date`` / ``end_date``
        (ISO ``YYYY-MM-DD``) and the typed string filters. This tool
        does no keyword parsing of "today", "last 3 months", etc.
        """
        session = self._session()
        try:
            # Resolve the patient — primary path is patient_id (injected
            # by the orchestrator from session). May be a UUID or a phone.
            patient: Optional[Patient] = self._resolve_patient(session, patient_id)
            # Legacy fall-throughs (kept for direct API / test callers)
            if patient is None and external_user_id:
                patient = self._resolve_patient(session, external_user_id)
            if patient is None and phone:
                patient = session.query(Patient).filter(
                    Patient.phone == phone
                ).first()
            if patient is None and name:
                patient = session.query(Patient).filter(
                    Patient.full_name.ilike(f"%{name}%")
                ).first()
            if patient is None:
                return "No patient found."
            # From here on, always use the canonical UUID.
            patient_id = patient.patient_id

            window = self._date_window(start_date, end_date)
            start_epoch, end_epoch = (window if window else (None, None))

            q = self._bookings_query(
                session,
                patient_id=patient.patient_id,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                doctor_name=doctor_name,
                specialty=specialty,
                status=status,
            ).order_by(Booking.start_at).limit(limit)

            bookings: List[Booking] = q.all()

            # Build a description of what filters were applied so the
            # synthesiser can faithfully report them back.
            applied = []
            if window:
                applied.append(f"window {start_date or '…'} → {end_date or '…'}")
            if doctor_name:
                applied.append(f"doctor~'{doctor_name}'")
            if specialty:
                applied.append(f"specialty~'{specialty}'")
            if (status or "all").lower() != "all":
                applied.append(f"status={status.lower()}")
            applied_str = "; ".join(applied) if applied else "no filters"

            if not bookings:
                return (
                    f"PATIENT: {patient.full_name}\n"
                    f"FILTERS: {applied_str}\n"
                    f"RESULT: 0 bookings match."
                )

            rows = [self._format_booking_row(session, bk) for bk in bookings]
            now = int(time.time())
            past_count = sum(1 for bk in bookings if bk.start_at < now)
            future_count = len(bookings) - past_count

            return (
                f"PATIENT: {patient.full_name}\n"
                f"FILTERS: {applied_str}\n"
                f"RESULT: {len(bookings)} booking(s) "
                f"({past_count} past, {future_count} upcoming).\n\n"
                + self._bookings_to_md_table(rows)
            )

        except Exception as exc:
            logger.exception("lookup_patient failed: {}", exc)
            return f"Error looking up bookings: {exc}"
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────
    # 2. search_doctors — find doctors
    # ─────────────────────────────────────────────────────────────

    @observe(name="crm_search_doctors")
    def search_doctors(
        self,
        specialty: Optional[str] = None,
        location: Optional[str] = None,
        doctor_name: Optional[str] = None,
        # Legacy alias accepted by the router historically:
        name: Optional[str] = None,
    ) -> str:
        """Search active doctors. All filters optional."""
        session = self._session()
        try:
            q = (
                session.query(Doctor)
                .outerjoin(Specialty, Specialty.specialty_id == Doctor.specialty_id)
                .filter(Doctor.active == 1)
            )
            if specialty:
                q = q.filter(Specialty.name.ilike(f"%{specialty}%"))
            name_filter = doctor_name or name
            if name_filter:
                q = q.filter(Doctor.full_name.ilike(f"%{name_filter}%"))
            if location:
                # join through bookings → locations to find doctors who practice there
                loc_sub = (
                    session.query(Location.location_id)
                    .filter(Location.name.ilike(f"%{location}%"))
                    .subquery()
                )
                booking_docs = (
                    session.query(Booking.doctor_id)
                    .filter(Booking.location_id.in_(loc_sub.select()))
                    .distinct()
                    .subquery()
                )
                q = q.filter(Doctor.doctor_id.in_(booking_docs.select()))

            # Count first (cheap, no JOIN explosion) — answers
            # "how many cardiologists" without enumerating all of them.
            total = q.with_entities(func.count(Doctor.doctor_id)).scalar() or 0
            doctors: List[Doctor] = q.order_by(Doctor.full_name).limit(20).all()
            if total == 0:
                applied = []
                if specialty:    applied.append(f"specialty~'{specialty}'")
                if name_filter:  applied.append(f"name~'{name_filter}'")
                if location:     applied.append(f"location~'{location}'")
                return (
                    f"No active doctors match the filters: "
                    f"{', '.join(applied) or 'none'}."
                )

            rows = []
            for doc in doctors:
                spec = doc.specialty.name if doc.specialty else "General"
                rows.append({
                    "name": self._doctor_label(doc.full_name),
                    "specialty": spec,
                    "phone": doc.phone or "—",
                })

            header = "| Name | Specialty | Phone |"
            sep = "|---|---|---|"
            lines = [header, sep] + [
                f"| {r['name']} | {r['specialty']} | {r['phone']} |"
                for r in rows
            ]
            cap_note = ""
            if total > len(rows):
                cap_note = f" (showing first {len(rows)} — refine the search to narrow down)"
            return f"DOCTORS FOUND: {total}{cap_note}\n\n" + "\n".join(lines)

        except Exception as exc:
            logger.exception("search_doctors failed: {}", exc)
            return f"Error searching doctors: {exc}"
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────
    # 6. list_specialties — every active department + doctor counts
    # ─────────────────────────────────────────────────────────────

    @observe(name="crm_list_specialties")
    def list_specialties(self) -> str:
        """
        Return every specialty offered by the store, with the count
        of currently active doctors in each. Patient-agnostic — safe to
        cache across all users.
        """
        session = self._session()
        try:
            # LEFT JOIN so specialties with zero active doctors still appear
            # (and are clearly flagged with a count of 0).
            rows = (
                session.query(
                    Specialty.name,
                    func.count(Doctor.doctor_id).filter(Doctor.active == 1).label("n"),
                )
                .outerjoin(Doctor, Doctor.specialty_id == Specialty.specialty_id)
                .group_by(Specialty.specialty_id, Specialty.name)
                .order_by(Specialty.name)
                .all()
            )

            if not rows:
                return "No specialties are currently configured."

            lines = ["| Specialty | Active doctors |", "|---|---|"]
            for name, n in rows:
                lines.append(f"| {name} | {n} |")
            total_specialties = len(rows)
            total_docs = sum(n for _, n in rows)
            return (
                f"SPECIALTIES: {total_specialties} departments, "
                f"{total_docs} active doctors total.\n\n"
                + "\n".join(lines)
            )

        except Exception as exc:
            logger.exception("list_specialties failed: {}", exc)
            return f"Error listing specialties: {exc}"
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────
    # 7. list_locations — every active branch
    # ─────────────────────────────────────────────────────────────

    @observe(name="crm_list_locations")
    def list_locations(self) -> str:
        """
        Return every active branch / clinic / lab. Patient-agnostic —
        safe to cache across all users.
        """
        session = self._session()
        try:
            locs: List[Location] = (
                session.query(Location)
                .filter(Location.active == 1)
                .order_by(Location.type, Location.name)
                .all()
            )
            if not locs:
                return "No active locations are currently configured."

            lines = ["| Name | Type | Address | Timezone |", "|---|---|---|---|"]
            for loc in locs:
                lines.append(
                    f"| {loc.name} | {loc.type} | "
                    f"{(loc.address or '—').replace('|', '/')} | {loc.tz} |"
                )
            return f"LOCATIONS: {len(locs)} active.\n\n" + "\n".join(lines)

        except Exception as exc:
            logger.exception("list_locations failed: {}", exc)
            return f"Error listing locations: {exc}"
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────
    # 3. create_booking — book a new appointment
    # ─────────────────────────────────────────────────────────────

    @observe(name="crm_create_booking")
    def create_booking(
        self,
        patient_id: str,
        start_at: Optional[str] = None,
        doctor_name: Optional[str] = None,
        specialty: Optional[str] = None,
        doctor_id: Optional[str] = None,
        location_name: Optional[str] = None,
        location_id: Optional[str] = None,
        duration_minutes: int = 30,
        title: Optional[str] = None,
        reason: Optional[str] = None,
        # Legacy router params we silently accept:
        start_time: Optional[str] = None,
        new_start_time: Optional[str] = None,
    ) -> str:
        """
        Create a new booking. ``start_at`` is a full ISO datetime with
        timezone — the router LLM is responsible for producing it from
        natural language ("tomorrow 10am" → '2026-04-26T10:00:00+05:30').

        Doctor resolution: ``doctor_id`` > ``doctor_name`` (substring) >
        ``specialty`` (first match by alpha order). If multiple doctors
        match the name, the response asks the synthesiser to clarify.
        """
        session = self._session()
        try:
            # ── 0. Resolve patient ─────────────────────────────
            patient: Optional[Patient] = self._resolve_patient(session, patient_id)
            if patient is None:
                return (
                    f"Cannot book — no patient found with id/phone "
                    f"'{patient_id}'. Please register the patient first."
                )
            # Always persist the canonical UUID in bookings.patient_id
            patient_id = patient.patient_id

            # ── 1. Parse start time ────────────────────────────
            iso = start_at or start_time or new_start_time
            start_epoch = self._parse_iso_datetime(iso) if iso else None
            if start_epoch is None:
                return (
                    "Cannot book — no valid start time provided. "
                    "Please specify a date and time, e.g. 'tomorrow at 10am' "
                    "or 'May 5 at 14:00'."
                )
            now = int(time.time())
            if start_epoch <= now:
                return (
                    f"Cannot book at {self._epoch_to_local(start_epoch)} — "
                    "appointments must be scheduled at a future date and "
                    "time. Please give a specific future day (e.g. tomorrow, "
                    "next Monday) and time of day (e.g. 10:00, 14:30)."
                )

            # ── 2. Resolve doctor ─────────────────────────────
            doctor: Optional[Doctor] = None
            if doctor_id:
                doctor = session.get(Doctor, doctor_id)
            elif doctor_name:
                docs = (
                    session.query(Doctor)
                    .outerjoin(Specialty, Specialty.specialty_id == Doctor.specialty_id)
                    .filter(Doctor.active == 1)
                    .filter(Doctor.full_name.ilike(f"%{doctor_name}%"))
                )
                if specialty:
                    docs = docs.filter(Specialty.name.ilike(f"%{specialty}%"))
                doc_list = docs.order_by(Doctor.full_name).limit(5).all()
                if not doc_list:
                    return (
                        f"No active doctor matching '{doctor_name}'"
                        + (f" in '{specialty}'" if specialty else "")
                        + ". Please be more specific or try another name."
                    )
                if len(doc_list) > 1:
                    rows = [
                        f"| {self._doctor_label(d.full_name)} | "
                        f"{d.specialty.name if d.specialty else '—'} | "
                        f"{d.phone or '—'} |"
                        for d in doc_list
                    ]
                    return (
                        f"Multiple doctors match '{doctor_name}'. "
                        "Please clarify which one (by full name or specialty):\n\n"
                        "| Name | Specialty | Phone |\n"
                        "|---|---|---|\n" + "\n".join(rows)
                    )
                doctor = doc_list[0]
            elif specialty:
                doctor = (
                    session.query(Doctor)
                    .outerjoin(Specialty, Specialty.specialty_id == Doctor.specialty_id)
                    .filter(Doctor.active == 1)
                    .filter(Specialty.name.ilike(f"%{specialty}%"))
                    .order_by(Doctor.full_name)
                    .first()
                )
                if doctor is None:
                    return f"No active doctor found in specialty '{specialty}'."
            else:
                return (
                    "Cannot book — no doctor or specialty provided. "
                    "Please specify which doctor or specialty you'd like."
                )

            # ── 3. Resolve location ────────────────────────────
            # Priority for picking a default when the router didn't
            # specify one:
            #   1. Explicit location_id from router.
            #   2. Explicit location_name (substring match).
            #   3. The doctor's most-used non-LAB location (consultations
            #      should land in an Electronics/Clothing, not a lab — even if the
            #      seed data has lab bookings).
            #   4. Any active Electronics or Clothing.
            #   5. Any active location at all (last resort).
            #
            # The earlier "last_bk.location" default produced "Central
            # Lab" for dermatology consultations because the seed data
            # had lab rows; the type-filter below avoids that.
            loc: Optional[Location] = None
            if location_id:
                loc = session.get(Location, location_id)
            elif location_name:
                loc = (
                    session.query(Location)
                    .filter(Location.name.ilike(f"%{location_name}%"))
                    .first()
                )
            if loc is None:
                # Most-used non-LAB location for this doctor
                doctor_loc = (
                    session.query(Booking, Location)
                    .join(Location, Location.location_id == Booking.location_id)
                    .filter(Booking.doctor_id == doctor.doctor_id)
                    .filter(Location.type.in_(("Electronics", "Clothing", "Standard")))
                    .order_by(Booking.start_at.desc())
                    .first()
                )
                if doctor_loc is not None:
                    loc = doctor_loc[1]
            if loc is None:
                # Any active Electronics or Clothing
                loc = (
                    session.query(Location)
                    .filter(Location.active == 1)
                    .filter(Location.type.in_(("Electronics", "Clothing")))
                    .order_by(Location.name)
                    .first()
                )
            if loc is None:
                # Last resort — any active location
                loc = (
                    session.query(Location)
                    .filter(Location.active == 1)
                    .order_by(Location.name)
                    .first()
                )
            if loc is None:
                return "Cannot book — no available location configured."

            # ── 4. Insert ──────────────────────────────────────
            booking_id = str(uuid.uuid4())
            spec_label = doctor.specialty.name if doctor.specialty else "General"
            booking = Booking(
                booking_id=booking_id,
                patient_id=patient_id,
                doctor_id=doctor.doctor_id,
                location_id=loc.location_id,
                title=title or f"{spec_label} appointment",
                reason=reason,
                start_at=start_epoch,
                end_at=start_epoch + max(15, int(duration_minutes)) * 60,
                status="PENDING",
                source="CRM",
                created_at=now,
                updated_at=now,
            )
            session.add(booking)
            session.commit()

            return (
                "BOOKING CREATED.\n"
                f"- When: {self._epoch_to_local(booking.start_at)} "
                f"(duration {duration_minutes} min)\n"
                f"- Doctor: {self._doctor_label(doctor.full_name)} ({spec_label})\n"
                f"- Location: {loc.name}\n"
                f"- Status: {booking.status}\n"
                f"- Booking ID: `{booking_id}`"
            )

        except Exception as exc:
            session.rollback()
            logger.exception("create_booking failed: {}", exc)
            return f"Error creating booking: {exc}"
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────
    # 4. cancel_booking — by booking_id OR by description
    # ─────────────────────────────────────────────────────────────

    @observe(name="crm_cancel_booking")
    def cancel_booking(
        self,
        patient_id: Optional[str] = None,
        booking_id: Optional[str] = None,
        doctor_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        specialty: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        """
        Cancel by ``booking_id`` (preferred) or by disambiguators
        (doctor / date window / specialty). Returns a list back to the
        synthesiser when multiple bookings match — the user can then
        narrow down on the next turn.
        """
        session = self._session()
        try:
            # Resolve patient_id (UUID or phone) up front so the ownership
            # comparison below uses the canonical UUID.
            if patient_id:
                resolved = self._resolve_patient(session, patient_id)
                if resolved is None:
                    return (
                        f"Cannot cancel — no patient with id/phone '{patient_id}'."
                    )
                patient_id = resolved.patient_id

            target: Optional[Booking] = None

            if booking_id:
                target = session.get(Booking, booking_id)
                if target is None:
                    return f"Booking `{booking_id}` not found."
                if patient_id and target.patient_id != patient_id:
                    return "That booking does not belong to the current patient."
            else:
                if not patient_id:
                    return "Cannot cancel — no patient context."
                # If the user didn't specify a date window, default to
                # FUTURE only — past bookings cannot be cancelled, and
                # surfacing them in the disambiguation list confuses
                # both the user and the synthesiser.
                window = self._date_window(start_date, end_date)
                if window:
                    start_epoch, end_epoch = window
                else:
                    start_epoch, end_epoch = int(time.time()), None

                q = self._bookings_query(
                    session,
                    patient_id=patient_id,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    doctor_name=doctor_name,
                    specialty=specialty,
                    status="active",
                ).order_by(Booking.start_at)
                matches: List[Booking] = q.limit(10).all()
                if not matches:
                    return (
                        "No upcoming active booking matches that description "
                        "(filters: "
                        f"doctor~'{doctor_name or '—'}', "
                        f"window {start_date or 'now'} → {end_date or '…'}, "
                        f"specialty~'{specialty or '—'}')."
                    )
                if len(matches) > 1:
                    rows = [self._format_booking_row(session, b) for b in matches]
                    return (
                        f"{len(matches)} upcoming bookings match — please clarify which to cancel "
                        "(by date or specialty):\n\n"
                        + self._bookings_to_md_table(rows)
                    )
                target = matches[0]

            if target.status == "CANCELLED":
                return f"Booking `{target.booking_id}` is already cancelled."

            old_status = target.status
            old_when = self._epoch_to_local(target.start_at)
            doctor = session.get(Doctor, target.doctor_id)

            target.status = "CANCELLED"
            target.updated_at = int(time.time())
            session.commit()

            return (
                "BOOKING CANCELLED.\n"
                f"- Was: {old_when} with "
                f"{self._doctor_label(doctor.full_name if doctor else None)}\n"
                f"- Previous status: {old_status}\n"
                f"- Booking ID: `{target.booking_id}`"
                + (f"\n- Reason: {reason}" if reason else "")
            )

        except Exception as exc:
            session.rollback()
            logger.exception("cancel_booking failed: {}", exc)
            return f"Error cancelling booking: {exc}"
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────
    # 5. reschedule_booking — move to a new ISO datetime
    # ─────────────────────────────────────────────────────────────

    @observe(name="crm_reschedule_booking")
    def reschedule_booking(
        self,
        patient_id: Optional[str] = None,
        new_start_at: Optional[str] = None,
        new_duration_minutes: Optional[int] = None,
        new_doctor_name: Optional[str] = None,
        new_doctor_id: Optional[str] = None,
        booking_id: Optional[str] = None,
        doctor_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        specialty: Optional[str] = None,
        # Legacy:
        new_start_time: Optional[str] = None,
    ) -> str:
        """
        Move an existing booking. Three things can change in one call:

          • ``new_start_at``    — the appointment's date/time (required if
                                  no ``new_doctor_*`` is provided)
          • ``new_duration_minutes`` — slot length (default: keep existing)
          • ``new_doctor_name`` / ``new_doctor_id`` — swap the doctor
                                  while keeping the row + booking_id

        Patient-anchor params (``booking_id``, or ``doctor_name`` /
        ``start_date`` / ``end_date`` / ``specialty``) match the same
        rules as ``cancel_booking`` for selecting which booking to
        update. The selection params describe the EXISTING booking;
        the ``new_*`` params describe what to change it to.
        """
        session = self._session()
        try:
            # Resolve patient_id (UUID or phone) up front so the ownership
            # comparison below uses the canonical UUID.
            if patient_id:
                resolved = self._resolve_patient(session, patient_id)
                if resolved is None:
                    return (
                        f"Cannot reschedule — no patient with id/phone '{patient_id}'."
                    )
                patient_id = resolved.patient_id

            iso = new_start_at or new_start_time
            new_epoch = self._parse_iso_datetime(iso) if iso else None
            doctor_change_requested = bool(new_doctor_id or new_doctor_name)

            # At least one of (new time, new doctor) must be specified.
            if new_epoch is None and not doctor_change_requested:
                return (
                    "Cannot reschedule — please specify what to change "
                    "(a new date/time, a new doctor, or both)."
                )
            if new_epoch is not None and new_epoch < int(time.time()):
                return (
                    f"Cannot reschedule to {self._epoch_to_local(new_epoch)} — "
                    "that time is in the past."
                )

            # ── 1. Resolve the new doctor (if a swap was requested) ──
            new_doctor: Optional[Doctor] = None
            if doctor_change_requested:
                if new_doctor_id:
                    new_doctor = session.get(Doctor, new_doctor_id)
                elif new_doctor_name:
                    new_doctor = (
                        session.query(Doctor)
                        .filter(Doctor.full_name.ilike(f"%{new_doctor_name}%"))
                        .filter(Doctor.active == 1)
                        .first()
                    )
                if new_doctor is None:
                    return (
                        f"Could not find a doctor matching "
                        f"{(new_doctor_name or new_doctor_id)!r}. "
                        "Try `search_doctors` to see who is available."
                    )

            # ── 2. Find the existing booking to update ──
            target: Optional[Booking] = None
            if booking_id:
                target = session.get(Booking, booking_id)
                if target is None:
                    return f"Booking `{booking_id}` not found."
                if patient_id and target.patient_id != patient_id:
                    return "That booking does not belong to the current patient."
            else:
                if not patient_id:
                    return "Cannot reschedule — no patient context."
                # Same future-only default as cancel: rescheduling a
                # past booking is meaningless.
                window = self._date_window(start_date, end_date)
                if window:
                    start_epoch, end_epoch = window
                else:
                    start_epoch, end_epoch = int(time.time()), None

                q = self._bookings_query(
                    session,
                    patient_id=patient_id,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    doctor_name=doctor_name,
                    specialty=specialty,
                    status="active",
                ).order_by(Booking.start_at)
                matches: List[Booking] = q.limit(10).all()
                if not matches:
                    return "No upcoming active booking matches that description."
                if len(matches) > 1:
                    rows = [self._format_booking_row(session, b) for b in matches]
                    return (
                        f"{len(matches)} upcoming bookings match — please clarify which to reschedule "
                        "(by date or specialty):\n\n"
                        + self._bookings_to_md_table(rows)
                    )
                target = matches[0]

            if target.status in ("CANCELLED", "COMPLETED"):
                return f"Cannot reschedule a {target.status} booking."

            # ── 3. Apply changes inside a single transaction ──
            old_when = self._epoch_to_local(target.start_at)
            old_doctor = session.get(Doctor, target.doctor_id)
            old_doctor_label = self._doctor_label(
                old_doctor.full_name if old_doctor else None
            )

            duration_min = new_duration_minutes or max(
                15, int((target.end_at - target.start_at) / 60)
            )
            if new_epoch is not None:
                target.start_at = new_epoch
                target.end_at = new_epoch + duration_min * 60
            if new_doctor is not None:
                target.doctor_id = new_doctor.doctor_id
            target.status = "RESCHEDULED"
            target.updated_at = int(time.time())
            session.commit()

            current_doctor = new_doctor or old_doctor
            new_doctor_label = self._doctor_label(
                current_doctor.full_name if current_doctor else None
            )

            # ── 4. Build a response that names the actual changes ──
            change_lines: List[str] = []
            if new_epoch is not None:
                change_lines.append(
                    f"- Time: {old_when}  →  {self._epoch_to_local(target.start_at)} "
                    f"(duration {duration_min} min)"
                )
            else:
                change_lines.append(
                    f"- Time: unchanged ({self._epoch_to_local(target.start_at)})"
                )
            if new_doctor is not None and (
                old_doctor is None or old_doctor.doctor_id != new_doctor.doctor_id
            ):
                change_lines.append(
                    f"- Doctor: {old_doctor_label}  →  {new_doctor_label}"
                )
            else:
                change_lines.append(f"- Doctor: unchanged ({new_doctor_label})")

            return (
                "BOOKING RESCHEDULED.\n"
                + "\n".join(change_lines)
                + f"\n- Booking ID: `{target.booking_id}`"
            )

        except Exception as exc:
            session.rollback()
            logger.exception("reschedule_booking failed: {}", exc)
            return f"Error rescheduling booking: {exc}"
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────
    # Doctor availability — open slots on a specific date
    # ─────────────────────────────────────────────────────────────

    # Specialist consultant clinic hours (per the DisputeAI FAQ entries
    # in config/faqs.yaml). All open-slot computation lives inside
    # this band; outside it the doctor is considered closed.
    _Clothing_OPEN_HOUR = 14   # 14:00
    _Clothing_CLOSE_HOUR = 21  # 21:00 (last slot starts at 20:30)
    _DEFAULT_SLOT_MIN = 30

    def check_doctor_availability(
        self,
        doctor_name: Optional[str] = None,
        doctor_id: Optional[str] = None,
        date: Optional[str] = None,                  # "YYYY-MM-DD"
        slot_minutes: Optional[int] = None,
    ) -> str:
        """
        Return a markdown table of OPEN time slots for a doctor on a
        given date.

        How it works:
          1. Resolve the doctor by ``doctor_id`` (preferred) or
             ``doctor_name`` (case-insensitive ``ilike`` match).
          2. Pull every non-cancelled booking for that doctor on the
             given date in Asia/Colombo local time.
          3. Walk the clinic-hour band (14:00–21:00) in fixed-size
             slots and keep the ones that don't overlap any booking.

        Notes:
          - Past slots (relative to "now" in Colombo) are filtered out.
          - The DisputeAI clinic-hour band is sourced from the FAQ
            entries; if the store's hours change we adjust the
            class-level constants and not the call sites.
        """
        from datetime import datetime, timedelta, timezone, time as dtime

        if not (doctor_name or doctor_id):
            return "Cannot check availability — please specify a doctor."
        if not date:
            return "Cannot check availability — please specify a date (YYYY-MM-DD)."

        # Parse the date strictly. We accept YYYY-MM-DD only — the
        # router resolves natural language dates upstream.
        try:
            day = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return f"Invalid date {date!r} — expected YYYY-MM-DD."

        slot_min = slot_minutes or self._DEFAULT_SLOT_MIN
        local = timezone(timedelta(hours=5, minutes=30))  # Asia/Colombo

        session = self._session()
        try:
            # ── 1. Resolve doctor ──────────────────────────────
            doctor: Optional[Doctor] = None
            if doctor_id:
                doctor = session.get(Doctor, doctor_id)
            elif doctor_name:
                doctor = (
                    session.query(Doctor)
                    .filter(Doctor.full_name.ilike(f"%{doctor_name}%"))
                    .filter(Doctor.active == 1)
                    .first()
                )
            if doctor is None:
                return (
                    f"Could not find doctor matching {(doctor_name or doctor_id)!r}. "
                    "Try `search_doctors` to see who is available."
                )

            # ── 2. Compute the day's epoch window (local midnight → midnight) ──
            day_start_local = datetime.combine(day, dtime(0, 0), tzinfo=local)
            day_end_local = day_start_local + timedelta(days=1)
            day_start_epoch = int(day_start_local.timestamp())
            day_end_epoch = int(day_end_local.timestamp())

            # ── 3. Pull this doctor's existing bookings for the day ──
            existing = (
                session.query(Booking)
                .filter(Booking.doctor_id == doctor.doctor_id)
                .filter(Booking.status != "CANCELLED")
                .filter(Booking.start_at >= day_start_epoch)
                .filter(Booking.start_at < day_end_epoch)
                .order_by(Booking.start_at)
                .all()
            )
            booked_intervals = [(b.start_at, b.end_at) for b in existing]

            # ── 4. Walk the clinic-hour band, emit free slots ──
            now_epoch = int(time.time())
            clinic_open = datetime.combine(
                day, dtime(self._Clothing_OPEN_HOUR, 0), tzinfo=local
            )
            clinic_close = datetime.combine(
                day, dtime(self._Clothing_CLOSE_HOUR, 0), tzinfo=local
            )

            free_slots: List[str] = []
            cursor = clinic_open
            slot_delta = timedelta(minutes=slot_min)
            while cursor + slot_delta <= clinic_close:
                start_epoch = int(cursor.timestamp())
                end_epoch = int((cursor + slot_delta).timestamp())
                # Skip slots that are already in the past
                if end_epoch <= now_epoch:
                    cursor += slot_delta
                    continue
                # Conflict if the candidate overlaps any booked
                # interval. Standard half-open overlap check:
                #   a_start < b_end AND b_start < a_end
                conflicts = any(
                    start_epoch < be and bs < end_epoch
                    for (bs, be) in booked_intervals
                )
                if not conflicts:
                    free_slots.append(cursor.strftime("%H:%M"))
                cursor += slot_delta

            doctor_label = self._doctor_label(doctor.full_name)
            header = (
                f"Open slots for {doctor_label} on {date} "
                f"(clinic hours {self._Clothing_OPEN_HOUR:02d}:00–"
                f"{self._Clothing_CLOSE_HOUR:02d}:00, "
                f"{slot_min}-minute appointments):\n\n"
            )

            if not free_slots:
                if booked_intervals:
                    return (
                        header
                        + f"No open slots — {doctor_label} is fully booked "
                        + f"({len(booked_intervals)} appointments) for that day.\n"
                    )
                # No bookings AND no free slots means the date is in the
                # past relative to now — clinic-hours band has elapsed.
                return (
                    header
                    + "No open slots — that day is already past.\n"
                )

            # Render as a 4-column table for legibility on small screens
            cols = 4
            rows = [free_slots[i : i + cols] for i in range(0, len(free_slots), cols)]
            table = "| " + " | ".join([" "] * cols) + " |\n"
            table += "|" + "|".join(["---"] * cols) + "|\n"
            for row in rows:
                padded = row + [""] * (cols - len(row))
                table += "| " + " | ".join(padded) + " |\n"

            footer = (
                f"\n{len(free_slots)} open slot(s) available. "
                "Reply with a time and I'll book it."
            )
            return header + table + footer

        except Exception as exc:
            logger.exception("check_doctor_availability failed: {}", exc)
            return f"Error checking availability: {exc}"
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────
    # Dispatch (resilient to router hallucinations)
    # ─────────────────────────────────────────────────────────────

    @observe(name="crm_dispatch")
    def dispatch(self, action: str, params: Dict[str, Any]) -> str:
        """
        Dispatch a CRM action, dropping any kwargs the router emitted
        that the handler doesn't accept (e.g. an old ``time_frame``
        enum). The synth still gets clean TOOL OUTPUT.
        """
        import inspect

        handler_map = {
            "lookup_patient": self.lookup_patient,
            "search_doctors": self.search_doctors,
            "create_booking": self.create_booking,
            "cancel_booking": self.cancel_booking,
            "reschedule_booking": self.reschedule_booking,
            "list_specialties": self.list_specialties,
            "list_locations": self.list_locations,
            "check_doctor_availability": self.check_doctor_availability,
        }
        handler = handler_map.get(action)
        if handler is None:
            return f"Unknown CRM action: {action}. Available: {list(handler_map.keys())}"

        sig = inspect.signature(handler)
        accepted = {p.name for p in sig.parameters.values()}
        clean = {k: v for k, v in (params or {}).items() if k in accepted and v is not None}
        dropped = sorted(set((params or {}).keys()) - accepted)
        if dropped:
            logger.debug(
                "CRM dispatch[{}]: dropped unsupported param(s) {} from router output",
                action, dropped,
            )

        update_current_observation(
            input=f"action={action} params={clean}"
                  + (f" (dropped {dropped})" if dropped else ""),
        )

        start = time.time()
        result = handler(**clean)
        latency_ms = int((time.time() - start) * 1000)

        update_current_observation(
            output=(result or "")[:500],
            metadata={"action": action, "latency_ms": latency_ms,
                      "dropped_params": dropped or None},
        )

        return result
