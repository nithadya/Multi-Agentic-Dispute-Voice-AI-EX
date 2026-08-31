"""
Shared helpers for API routers.

Currently houses phone-number normalization. The seeded patients table
stores phones in two parallel columns:

    phone             "+94781030736"     ← display, with leading "+"
    external_user_id  "94781030736"      ← search key, digits-only

Whatever a user types ("078 103 0736", "+94 78 103 0736", "94781030736")
must collapse to the same canonical form for both lookup and unique-
constraint purposes. ``normalize_phone()`` handles the common Sri Lanka
patterns and falls back to a plain digit-strip for international numbers.
"""

from __future__ import annotations


def normalize_phone(raw: str | None) -> str:
    """
    Reduce a user-entered phone string to its canonical digits-only form.

    Rules (Sri Lanka specific, falls through cleanly for others):
      - Strip every non-digit character.
      - 10 digits starting with ``0`` (local SL format) → drop the leading
        ``0`` and prepend ``94``.  e.g. ``"0781030736"`` → ``"94781030736"``.
      - 11 digits starting with ``94`` → keep as-is (matches seed data).
      - Anything else → return the digit-strip unchanged.

    The result is what we store in ``patients.external_user_id`` and what
    we compare against when the user "logs in" by phone.
    """
    if not raw:
        return ""
    digits = "".join(c for c in raw if c.isdigit())
    if len(digits) == 10 and digits.startswith("0"):
        return "94" + digits[1:]
    return digits


def display_phone(canonical: str | None) -> str:
    """
    Render a canonical phone number with the customary ``+`` prefix.

    Inverse of ``normalize_phone()`` for display purposes only — does not
    re-introduce spacing. Returns an empty string for empty input.
    """
    if not canonical:
        return ""
    return "+" + canonical if not canonical.startswith("+") else canonical
