"""Pre-migration probe for Plan 2a tournament schema invariants.

Exposes `check_tournament_schema_ready(connection) -> list[str]` for use
from the Alembic upgrade() step and from the __main__ block below.

Returns a list of violation descriptions. Empty list = safe to migrate.
Any non-empty return = migration MUST abort (not silently continue).

Usage from CLI (pre-deploy staging check):

    python -m atp.dashboard.migrations.probes.check_tournament_invariants

Reads ATP_DATABASE_URL from env. Exits 0 if clean, exits 1 with violation
list if not. Exits 2 if ATP_DATABASE_URL is unset (fail-loud default to
prevent silent misconfiguration on staging).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Connection, Row, create_engine, text

from atp.dashboard.migrations.url_helpers import as_sync_url


def _rows(conn: Connection, sql: str) -> Sequence[Row[Any]]:
    return conn.execute(text(sql)).all()


def check_tournament_schema_ready(connection: Connection) -> list[str]:
    """Run all probes; return list of human-readable violation descriptions."""
    violations: list[str] = []

    # Probe 1: Participant.user_id NOT NULL precondition
    rows = _rows(
        connection,
        "SELECT COUNT(*) FROM tournament_participants WHERE user_id IS NULL",
    )
    null_user_id_count = rows[0][0]
    if null_user_id_count > 0:
        violations.append(
            f"P1: {null_user_id_count} tournament_participants rows have "
            f"user_id IS NULL. Plan 2a requires user_id NOT NULL. "
            f"Resolution: DELETE the anonymous rows or backfill them to "
            f"known user_ids before re-running upgrade."
        )

    # Probe 2: FK orphan check (SQLite default does not enforce FK)
    rows = _rows(
        connection,
        """
        SELECT COUNT(*) FROM tournament_participants p
        WHERE p.user_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM users u WHERE u.id = p.user_id)
        """,
    )
    orphan_count = rows[0][0]
    if orphan_count > 0:
        violations.append(
            f"P2: {orphan_count} tournament_participants rows reference a "
            f"user_id that does not exist in the users table. This is an "
            f"FK integrity violation that SQLite silently allows. "
            f"Resolution: DELETE FROM tournament_participants WHERE user_id "
            f"NOT IN (SELECT id FROM users)."
        )

    # Probe 3: uq_participant_tournament_user precondition
    rows = _rows(
        connection,
        """
        SELECT tournament_id, user_id, COUNT(*) as cnt
        FROM tournament_participants
        WHERE user_id IS NOT NULL
        GROUP BY tournament_id, user_id
        HAVING COUNT(*) > 1
        """,
    )
    dup_participant_rows = list(rows)
    if dup_participant_rows:
        examples = ", ".join(
            f"(tournament={t}, user={u}, count={c})"
            for t, u, c in dup_participant_rows[:5]
        )
        violations.append(
            f"P3: {len(dup_participant_rows)} (tournament_id, user_id) pairs "
            f"have duplicate participant rows. Examples: {examples}. "
            f"Plan 2a requires uq_participant_tournament_user. "
            f"Resolution: manually deduplicate, keeping the row with the "
            f"earliest joined_at."
        )

    # Probe 4: uq_action_round_participant precondition
    rows = _rows(
        connection,
        """
        SELECT round_id, participant_id, COUNT(*) as cnt
        FROM tournament_actions
        GROUP BY round_id, participant_id
        HAVING COUNT(*) > 1
        """,
    )
    dup_action_rows = list(rows)
    if dup_action_rows:
        examples = ", ".join(
            f"(round={r}, participant={p}, count={c})"
            for r, p, c in dup_action_rows[:5]
        )
        violations.append(
            f"P4: {len(dup_action_rows)} (round_id, participant_id) pairs "
            f"have duplicate action rows. Examples: {examples}. "
            f"Plan 2a requires uq_action_round_participant. "
            f"Resolution: manually deduplicate."
        )

    # Probe 5: uq_round_tournament_number precondition
    rows = _rows(
        connection,
        """
        SELECT tournament_id, round_number, COUNT(*) as cnt
        FROM tournament_rounds
        GROUP BY tournament_id, round_number
        HAVING COUNT(*) > 1
        """,
    )
    dup_round_rows = list(rows)
    if dup_round_rows:
        examples = ", ".join(
            f"(tournament={t}, round={r}, count={c})" for t, r, c in dup_round_rows[:5]
        )
        violations.append(
            f"P5: {len(dup_round_rows)} (tournament_id, round_number) pairs "
            f"have duplicate round rows. Examples: {examples}. "
            f"Plan 2a requires uq_round_tournament_number."
        )

    # Probe 6: uq_participant_user_active precondition (relaxed via JOIN)
    rows = _rows(
        connection,
        """
        SELECT p.user_id, COUNT(*) as cnt
        FROM tournament_participants p
        JOIN tournaments t ON p.tournament_id = t.id
        WHERE p.user_id IS NOT NULL
          AND t.status IN ('pending', 'active')
        GROUP BY p.user_id
        HAVING COUNT(*) > 1
        """,
    )
    dup_active_rows = list(rows)
    if dup_active_rows:
        examples = ", ".join(f"(user={u}, count={c})" for u, c in dup_active_rows[:5])
        violations.append(
            f"P6: {len(dup_active_rows)} users are currently in more than "
            f"one pending/active tournament. Examples: {examples}. "
            f"Plan 2a enforces 1-active-per-user via "
            f"uq_participant_user_active. Resolution: transition stale "
            f"tournaments to completed/cancelled status directly via SQL, "
            f"or DELETE stale participant rows before re-running upgrade. "
            f"Do NOT attempt to set released_at directly — the column does "
            f"not exist at probe time."
        )

    return violations


def _main() -> int:
    db_url = os.environ.get("ATP_DATABASE_URL")
    if not db_url:
        print(
            "FAIL: ATP_DATABASE_URL environment variable is not set. "
            "Pre-deploy probe requires an explicit database URL — there "
            "is no default to prevent silent misconfiguration.",
            file=sys.stderr,
        )
        return 2

    # Operators typically export the same ATP_DATABASE_URL the app reads,
    # which uses an async driver. Strip the async suffix so the sync
    # engine constructed below doesn't crash with MissingGreenlet.
    engine = create_engine(as_sync_url(db_url))
    try:
        with engine.connect() as conn:
            violations = check_tournament_schema_ready(conn)
    finally:
        engine.dispose()

    if not violations:
        print("OK: all tournament schema invariants satisfied")
        return 0

    print(f"FAIL: {len(violations)} violations found:")
    for v in violations:
        print(f"  - {v}")
    print()
    print("See migration file header for the full probe->resolution playbook.")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
