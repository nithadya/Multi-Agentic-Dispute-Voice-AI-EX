"""
Audit + migrate the `patients` table to enforce uniqueness on phone and email.

Phase 1 (always): scan for duplicate non-NULL values.
  - phone: exact-match duplicates
  - email: case-insensitive duplicates (LOWER(email))

Phase 2 (only with --apply): if no duplicates exist, add the constraints:
    ALTER TABLE patients ADD CONSTRAINT patients_phone_unique UNIQUE (phone);
    CREATE UNIQUE INDEX patients_email_lower_unique ON patients (LOWER(email));

Multiple NULLs are still allowed in both columns — patients without a
phone or email are fine.

Run from the project root:
    PYTHONPATH=src .venv/bin/python scripts/check_patient_uniques.py
    PYTHONPATH=src .venv/bin/python scripts/check_patient_uniques.py --apply
"""

import argparse
import sys

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text

from infrastructure.db import get_sql_engine


PHONE_CONSTRAINT = "patients_phone_unique"
EMAIL_INDEX = "patients_email_lower_unique"


def query_duplicates(conn, column: str, group_expr: str | None = None):
    """Return list of (value, count) tuples for duplicate non-NULL values."""
    expr = group_expr or column
    sql = f"""
        SELECT {expr} AS val, COUNT(*) AS n
        FROM patients
        WHERE {column} IS NOT NULL AND {column} <> ''
        GROUP BY {expr}
        HAVING COUNT(*) > 1
        ORDER BY n DESC, val
    """
    return conn.execute(text(sql)).fetchall()


def report(label: str, rows) -> bool:
    if not rows:
        print(f"  ✓ {label}: no duplicates")
        return True
    print(f"  ✗ {label}: {len(rows)} duplicate value(s)")
    for val, n in rows[:10]:
        print(f"      {val!r:40s}  ×{n}")
    if len(rows) > 10:
        print(f"      … and {len(rows) - 10} more")
    return False


def constraint_exists(conn, name: str) -> bool:
    return conn.execute(
        text("SELECT 1 FROM pg_constraint WHERE conname = :n LIMIT 1"),
        {"n": name},
    ).first() is not None


def index_exists(conn, name: str) -> bool:
    return conn.execute(
        text("SELECT 1 FROM pg_indexes WHERE indexname = :n LIMIT 1"),
        {"n": name},
    ).first() is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Add the UNIQUE constraints if (and only if) the table is duplicate-free.",
    )
    parser.add_argument(
        "--phone-only",
        action="store_true",
        help="Skip the email check; only enforce uniqueness on phone.",
    )
    args = parser.parse_args()

    engine = get_sql_engine()

    # ── Audit phase (read-only) ──────────────────────────────────
    with engine.connect() as conn:
        print("=== patients table — uniqueness audit ===\n")

        print("Phone (exact-match duplicates):")
        phone_clean = report("phone", query_duplicates(conn, "phone"))

        if args.phone_only:
            email_clean = True
            print("\n(skipping email — --phone-only)")
        else:
            print("\nEmail (case-insensitive duplicates — LOWER(email)):")
            email_clean = report("email", query_duplicates(conn, "email", "LOWER(email)"))

    if not (phone_clean and email_clean):
        print(
            "\n⚠️  Cannot add UNIQUE constraints — dedupe first.\n"
            "    Inspect/merge the rows above in Supabase, then re-run."
        )
        sys.exit(1)

    print("\n=== Constraints that will be added ===")
    print(f"  ALTER TABLE patients ADD CONSTRAINT {PHONE_CONSTRAINT} UNIQUE (phone);")
    if not args.phone_only:
        print(f"  CREATE UNIQUE INDEX {EMAIL_INDEX} ON patients (LOWER(email));")
    print()

    if not args.apply:
        print("Audit-only run. Re-run with --apply to execute.")
        return

    # ── Apply phase (its own transactional connection) ───────────
    print("Applying...")
    with engine.begin() as conn:
        if constraint_exists(conn, PHONE_CONSTRAINT):
            print(f"  • {PHONE_CONSTRAINT} already exists — skipping")
        else:
            conn.execute(text(
                f"ALTER TABLE patients ADD CONSTRAINT {PHONE_CONSTRAINT} UNIQUE (phone)"
            ))
            print(f"  ✓ added UNIQUE constraint on patients.phone")

        if not args.phone_only:
            if index_exists(conn, EMAIL_INDEX):
                print(f"  • {EMAIL_INDEX} already exists — skipping")
            else:
                conn.execute(text(
                    f"CREATE UNIQUE INDEX {EMAIL_INDEX} ON patients (LOWER(email))"
                ))
                print(f"  ✓ added UNIQUE INDEX on LOWER(patients.email)")

    print("\nDone — uniqueness now enforced.")


if __name__ == "__main__":
    main()
