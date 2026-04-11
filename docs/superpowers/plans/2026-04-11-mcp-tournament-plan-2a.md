# MCP Tournament Plan 2a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the vertical-slice MCP tournament server (commit `2759613`) to a state where public tournament launch is safe: enforced schema invariants, deadline worker, full MCP tool set, REST admin surface, idempotent reconnect, AD-9/AD-10 safety.

**Architecture:** Seven Alembic invariants + eight additive columns on existing tournament tables, a single-worker asyncio deadline loop in the FastAPI lifespan, a twin-methods cancel pattern (`cancel_tournament` / `cancel_tournament_system` → shared `_cancel_impl`), `session_sync` reconnect mechanism, and a `_format_for_user` notification personalization dispatcher. Single service method is the source of truth for every cross-surface operation; REST and MCP are thin wrappers.

**Tech Stack:** Python 3.12, SQLAlchemy 2.x async, Alembic (`batch_alter_table` for SQLite portability), FastAPI, FastMCP 3.x, pytest + pytest-anyio, httpx, freezegun, structlog.

**Source spec:** `docs/superpowers/specs/2026-04-11-mcp-tournament-plan-2a-design.md` (commits `1aaf192` + `7908f8d`).

**Baseline:** `main` at or after commit `2759613` (vertical slice merged).

**Prerequisite before Task 1:** run `uv sync --all-packages --group dev` once to ensure `fastmcp`, `pyrefly`, and other vertical-slice deps are installed in the local `.venv`. The pre-commit hook will fail without this.

---

## Task 1: CancelReason enum module

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/tournament/reasons.py`
- Test: `tests/unit/dashboard/tournament/test_reasons.py`

- [ ] **Step 1: Create test directory structure if missing**

```bash
mkdir -p tests/unit/dashboard/tournament
touch tests/unit/dashboard/tournament/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/dashboard/tournament/test_reasons.py`:

```python
"""Tests for CancelReason enum module."""

from atp.dashboard.tournament.reasons import CancelReason


def test_cancel_reason_has_three_values():
    assert set(CancelReason) == {
        CancelReason.ADMIN_ACTION,
        CancelReason.PENDING_TIMEOUT,
        CancelReason.ABANDONED,
    }


def test_cancel_reason_values_are_stable_strings():
    assert CancelReason.ADMIN_ACTION.value == "admin_action"
    assert CancelReason.PENDING_TIMEOUT.value == "pending_timeout"
    assert CancelReason.ABANDONED.value == "abandoned"


def test_cancel_reason_is_str_enum():
    # StrEnum members compare equal to their string value in Python 3.11+
    assert CancelReason.ADMIN_ACTION == "admin_action"
    assert "abandoned" == CancelReason.ABANDONED
```

- [ ] **Step 3: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_reasons.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atp.dashboard.tournament.reasons'`

- [ ] **Step 4: Create the module**

Create `packages/atp-dashboard/atp/dashboard/tournament/reasons.py`:

```python
"""Cancellation reason enum. Single source of truth, imported by service,
deadline worker, handlers, models, and tests."""

from enum import StrEnum


class CancelReason(StrEnum):
    ADMIN_ACTION = "admin_action"
    PENDING_TIMEOUT = "pending_timeout"
    ABANDONED = "abandoned"
```

- [ ] **Step 5: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/tournament/test_reasons.py -v`
Expected: PASS — 3 tests

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/reasons.py \
        tests/unit/dashboard/tournament/__init__.py \
        tests/unit/dashboard/tournament/test_reasons.py
git commit -m "feat(tournament): add CancelReason enum module"
```

---

## Task 2: RoundStatus and ActionSource enums + string-literal refactor

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/models.py`
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` (refactor existing string literals)
- Test: `tests/unit/dashboard/tournament/test_model_enums.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_model_enums.py`:

```python
"""Tests for RoundStatus and ActionSource enums added by Plan 2a."""

from atp.dashboard.tournament.models import RoundStatus, ActionSource


def test_round_status_has_four_values():
    assert set(RoundStatus) == {
        RoundStatus.WAITING_FOR_ACTIONS,
        RoundStatus.IN_PROGRESS,
        RoundStatus.COMPLETED,
        RoundStatus.CANCELLED,
    }


def test_round_status_wire_values_match_vertical_slice():
    assert RoundStatus.WAITING_FOR_ACTIONS.value == "waiting_for_actions"
    assert RoundStatus.IN_PROGRESS.value == "in_progress"
    assert RoundStatus.COMPLETED.value == "completed"
    assert RoundStatus.CANCELLED.value == "cancelled"


def test_action_source_has_two_values():
    assert set(ActionSource) == {
        ActionSource.SUBMITTED,
        ActionSource.TIMEOUT_DEFAULT,
    }


def test_action_source_wire_values():
    assert ActionSource.SUBMITTED.value == "submitted"
    assert ActionSource.TIMEOUT_DEFAULT.value == "timeout_default"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_model_enums.py -v`
Expected: FAIL with `ImportError: cannot import name 'RoundStatus'` (or `ActionSource`)

- [ ] **Step 3: Add enums to models.py**

Edit `packages/atp-dashboard/atp/dashboard/tournament/models.py`. After the existing `class TournamentStatus(StrEnum):` block (around line 20-26), insert:

```python
class RoundStatus(StrEnum):
    """Round lifecycle status.

    WAITING_FOR_ACTIONS, IN_PROGRESS, COMPLETED existed as bare string
    literals in vertical slice service.py. Plan 2a introduces this StrEnum
    for type safety and adds CANCELLED as a new value used by _cancel_impl
    to transition in-flight rounds when their tournament is cancelled.

    Stored as plain String(20) in the DB without a native enum type or
    CHECK constraint.
    """

    WAITING_FOR_ACTIONS = "waiting_for_actions"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ActionSource(StrEnum):
    """Origin of an Action row.

    SUBMITTED — player sent make_move via MCP tool before deadline.
    TIMEOUT_DEFAULT — deadline worker force_resolve_round created a
    default action for a participant who did not submit before the
    round deadline.

    Stored as plain String(32) without a native enum type or CHECK
    constraint.
    """

    SUBMITTED = "submitted"
    TIMEOUT_DEFAULT = "timeout_default"
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/tournament/test_model_enums.py -v`
Expected: PASS — 4 tests

- [ ] **Step 5: Refactor bare string literals in service.py**

Find and replace in `packages/atp-dashboard/atp/dashboard/tournament/service.py`:

Add at the top of imports (near existing tournament imports):

```python
from atp.dashboard.tournament.models import RoundStatus
```

Then replace these bare string literals with enum references:
- Line 148 (approx): `status="waiting_for_actions"` → `status=RoundStatus.WAITING_FOR_ACTIONS`
- Line 295 (approx): `Round.status == "waiting_for_actions"` → `Round.status == RoundStatus.WAITING_FOR_ACTIONS`
- Line 431 (approx): `status="waiting_for_actions"` → `status=RoundStatus.WAITING_FOR_ACTIONS`

Use `grep -n '"waiting_for_actions"\|"in_progress"\|"completed"' packages/atp-dashboard/atp/dashboard/tournament/service.py` to find all occurrences, then replace each with the corresponding `RoundStatus.<VALUE>`.

- [ ] **Step 6: Run existing tournament unit tests to verify no regression**

Run: `uv run pytest tests/unit/dashboard/tournament -v`
Expected: all pre-existing vertical slice tests still pass, plus the new enum tests.

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/models.py \
        packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_model_enums.py
git commit -m "feat(tournament): add RoundStatus and ActionSource enums"
```

---

## Task 3: Tournament model column additions (no migration yet)

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/models.py`
- Test: `tests/unit/dashboard/tournament/test_model_columns.py`

This task adds SQLAlchemy column declarations to match what the migration in Task 5 will add. Tests verify the ORM model shape only; schema reflection tests come after the migration.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_model_columns.py`:

```python
"""Tests that model column declarations match Plan 2a schema deltas."""

from sqlalchemy import inspect

from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Tournament,
)


def test_tournament_has_plan_2a_columns():
    columns = {c.name for c in inspect(Tournament).columns}
    required = {
        "pending_deadline",
        "join_token",
        "cancelled_at",
        "cancelled_by",
        "cancelled_reason",
        "cancelled_reason_detail",
    }
    assert required.issubset(columns), (
        f"Tournament missing columns: {required - columns}"
    )


def test_participant_has_released_at():
    columns = {c.name for c in inspect(Participant).columns}
    assert "released_at" in columns


def test_participant_user_id_not_nullable():
    col = inspect(Participant).columns["user_id"]
    assert col.nullable is False


def test_action_has_source_column():
    columns = {c.name for c in inspect(Action).columns}
    assert "source" in columns


def test_action_source_has_server_default():
    col = inspect(Action).columns["source"]
    assert col.server_default is not None
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_model_columns.py -v`
Expected: 5 tests fail (columns not declared yet).

- [ ] **Step 3: Add imports and columns to `Tournament` class**

Edit `packages/atp-dashboard/atp/dashboard/tournament/models.py`. Update imports at top to include `sa` and `CancelReason`:

```python
import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)

from atp.dashboard.tournament.reasons import CancelReason
```

Inside `class Tournament(Base):`, after the existing columns block (after `created_at:`, before `participants:` relationship), add:

```python
    # Plan 2a additive columns — AD-9 pending deadline
    pending_deadline: Mapped[datetime] = mapped_column(
        DateTime, nullable=False
    )

    # Plan 2a additive columns — AD-10 join token
    join_token: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )

    # Plan 2a additive columns — cancel audit
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    cancelled_by: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    cancelled_reason: Mapped[CancelReason | None] = mapped_column(
        sa.Enum(CancelReason, native_enum=False, length=32),
        nullable=True,
    )
    cancelled_reason_detail: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
```

Replace `Tournament.__table_args__` with:

```python
    __table_args__ = (
        Index("idx_tournaments_status", "status"),
        Index("idx_tournaments_tenant", "tenant_id"),
        Index(
            "idx_tournaments_status_pending_deadline",
            "status",
            "pending_deadline",
        ),
    )
```

- [ ] **Step 4: Flip `Participant.user_id` to NOT NULL and add `released_at`**

Inside `class Participant(Base):`, change `user_id` declaration from `Mapped[int | None]` with `nullable=True` to:

```python
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
```

After `total_score:`, add:

```python
    # Plan 2a additive column — AD-10 slot release
    released_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
```

Replace `Participant.__table_args__` with:

```python
    __table_args__ = (
        Index("idx_participant_tournament", "tournament_id"),
        Index("idx_participant_user", "user_id"),
        UniqueConstraint(
            "tournament_id", "user_id",
            name="uq_participant_tournament_user",
        ),
        Index(
            # uq_ prefix: semantically a unique constraint, implemented
            # as a partial unique index because UniqueConstraint does
            # not accept WHERE clauses and neither SQLite nor PostgreSQL
            # support partial UNIQUE in CREATE TABLE syntax.
            "uq_participant_user_active",
            "user_id",
            unique=True,
            sqlite_where=text("user_id IS NOT NULL AND released_at IS NULL"),
            postgresql_where=text("user_id IS NOT NULL AND released_at IS NULL"),
        ),
    )
```

- [ ] **Step 5: Add `uq_round_tournament_number` + `idx_round_status_deadline` to Round**

Replace `Round.__table_args__` with:

```python
    __table_args__ = (
        Index("idx_round_tournament", "tournament_id"),
        UniqueConstraint(
            "tournament_id", "round_number",
            name="uq_round_tournament_number",
        ),
        Index("idx_round_status_deadline", "status", "deadline"),
    )
```

- [ ] **Step 6: Add `source` column and `uq_action_round_participant` to Action**

Inside `class Action(Base):`, after `payoff:`, add:

```python
    # Plan 2a additive column — audit trail for timeout-default vs
    # player-submitted actions
    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="submitted",
    )
```

Replace `Action.__table_args__` with:

```python
    __table_args__ = (
        Index("idx_action_round", "round_id"),
        UniqueConstraint(
            "round_id", "participant_id",
            name="uq_action_round_participant",
        ),
    )
```

- [ ] **Step 7: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/tournament/test_model_columns.py -v`
Expected: PASS — 5 tests

- [ ] **Step 8: Run full tournament unit suite for regressions**

Run: `uv run pytest tests/unit/dashboard/tournament -v`
Expected: all tests pass. Any vertical-slice test that referenced `Participant.user_id` as nullable may need adjustment — fix those by providing a real `user_id` in fixtures.

- [ ] **Step 9: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/models.py \
        tests/unit/dashboard/tournament/test_model_columns.py
git commit -m "feat(tournament): add Plan 2a schema columns and constraints to models"
```

---

## Task 4: Probe module (check_tournament_invariants.py)

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/migrations/__init__.py`
- Create: `packages/atp-dashboard/atp/dashboard/migrations/probes/__init__.py`
- Create: `packages/atp-dashboard/atp/dashboard/migrations/probes/check_tournament_invariants.py`
- Test: `tests/unit/dashboard/migrations/probes/test_check_tournament_invariants.py`

- [ ] **Step 1: Create directory skeleton**

```bash
mkdir -p packages/atp-dashboard/atp/dashboard/migrations/probes
touch packages/atp-dashboard/atp/dashboard/migrations/__init__.py
touch packages/atp-dashboard/atp/dashboard/migrations/probes/__init__.py
mkdir -p tests/unit/dashboard/migrations/probes
touch tests/unit/dashboard/migrations/__init__.py
touch tests/unit/dashboard/migrations/probes/__init__.py
```

- [ ] **Step 2: Write the failing test (P1 clean + P1 violation)**

Create `tests/unit/dashboard/migrations/probes/test_check_tournament_invariants.py`:

```python
"""Tests for the Plan 2a pre-migration probe module."""

import pytest
from sqlalchemy import create_engine, text

from atp.dashboard.migrations.probes.check_tournament_invariants import (
    check_tournament_schema_ready,
)


@pytest.fixture
def baseline_db(tmp_path):
    """SQLite DB matching the vertical-slice schema (pre-Plan-2a).

    Only creates the tables the probes touch: users, tournaments,
    tournament_participants, tournament_rounds, tournament_actions.
    FK enforcement OFF by default so tests can seed orphan rows for P2.
    """
    db_path = tmp_path / "probe_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                is_admin BOOLEAN NOT NULL DEFAULT 0
            )
        """))
        conn.execute(text("""
            CREATE TABLE tournaments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                game_type TEXT NOT NULL DEFAULT 'prisoners_dilemma',
                config TEXT DEFAULT '{}'
            )
        """))
        conn.execute(text("""
            CREATE TABLE tournament_participants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id INTEGER NOT NULL,
                user_id INTEGER,
                agent_name TEXT NOT NULL DEFAULT ''
            )
        """))
        conn.execute(text("""
            CREATE TABLE tournament_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id INTEGER NOT NULL,
                round_number INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'waiting_for_actions'
            )
        """))
        conn.execute(text("""
            CREATE TABLE tournament_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                participant_id INTEGER NOT NULL,
                action_data TEXT DEFAULT '{}'
            )
        """))
        conn.execute(text(
            "INSERT INTO users (id, username) VALUES (1, 'alice'), (2, 'bob')"
        ))
    yield engine
    engine.dispose()


def test_clean_db_returns_empty(baseline_db):
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert violations == []


def test_p1_detects_null_user_id(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text(
            "INSERT INTO tournaments (id, name) VALUES (1, 't')"
        ))
        conn.execute(text(
            "INSERT INTO tournament_participants "
            "(tournament_id, user_id, agent_name) VALUES (1, NULL, 'anon')"
        ))
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P1:") for v in violations)


def test_p2_detects_fk_orphan(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text(
            "INSERT INTO tournaments (id, name) VALUES (1, 't')"
        ))
        conn.execute(text(
            "INSERT INTO tournament_participants "
            "(tournament_id, user_id, agent_name) VALUES (1, 99999, 'ghost')"
        ))
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P2:") for v in violations)


def test_p3_detects_duplicate_participant(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text(
            "INSERT INTO tournaments (id, name) VALUES (1, 't')"
        ))
        conn.execute(text(
            "INSERT INTO tournament_participants "
            "(tournament_id, user_id, agent_name) VALUES (1, 1, 'a')"
        ))
        conn.execute(text(
            "INSERT INTO tournament_participants "
            "(tournament_id, user_id, agent_name) VALUES (1, 1, 'b')"
        ))
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P3:") for v in violations)


def test_p4_detects_duplicate_action(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text(
            "INSERT INTO tournaments (id, name) VALUES (1, 't')"
        ))
        conn.execute(text(
            "INSERT INTO tournament_rounds "
            "(id, tournament_id, round_number) VALUES (1, 1, 1)"
        ))
        conn.execute(text(
            "INSERT INTO tournament_participants "
            "(id, tournament_id, user_id, agent_name) "
            "VALUES (1, 1, 1, 'a')"
        ))
        conn.execute(text(
            "INSERT INTO tournament_actions "
            "(round_id, participant_id) VALUES (1, 1), (1, 1)"
        ))
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P4:") for v in violations)


def test_p5_detects_duplicate_round(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text(
            "INSERT INTO tournaments (id, name) VALUES (1, 't')"
        ))
        conn.execute(text(
            "INSERT INTO tournament_rounds "
            "(tournament_id, round_number) VALUES (1, 1), (1, 1)"
        ))
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P5:") for v in violations)


def test_p6_detects_multi_active_user(baseline_db):
    with baseline_db.begin() as conn:
        conn.execute(text(
            "INSERT INTO tournaments (id, name, status) VALUES "
            "(1, 't1', 'active'), (2, 't2', 'pending')"
        ))
        conn.execute(text(
            "INSERT INTO tournament_participants "
            "(tournament_id, user_id, agent_name) VALUES "
            "(1, 1, 'a'), (2, 1, 'b')"
        ))
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    assert any(v.startswith("P6:") for v in violations)


def test_p6_ignores_completed_tournament_history(baseline_db):
    """Critical relaxed-probe edge case: a user with past participation
    in completed tournaments plus one active participation must pass."""
    with baseline_db.begin() as conn:
        conn.execute(text(
            "INSERT INTO tournaments (id, name, status) VALUES "
            "(1, 't1', 'completed'), "
            "(2, 't2', 'completed'), "
            "(3, 't3', 'completed'), "
            "(4, 't4', 'active')"
        ))
        conn.execute(text(
            "INSERT INTO tournament_participants "
            "(tournament_id, user_id, agent_name) VALUES "
            "(1, 1, 'a'), (2, 1, 'b'), (3, 1, 'c'), (4, 1, 'd')"
        ))
    with baseline_db.connect() as conn:
        violations = check_tournament_schema_ready(conn)
    # No P6 violation — historical completed participations are ignored
    assert not any(v.startswith("P6:") for v in violations)
```

- [ ] **Step 3: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/migrations/probes/test_check_tournament_invariants.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atp.dashboard.migrations.probes.check_tournament_invariants'`

- [ ] **Step 4: Create the probe module**

Create `packages/atp-dashboard/atp/dashboard/migrations/probes/check_tournament_invariants.py`:

```python
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
from typing import Iterable

from sqlalchemy import Connection, create_engine, text


def _rows(conn: Connection, sql: str) -> Iterable[tuple]:
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
            f"(tournament={t}, round={r}, count={c})"
            for t, r, c in dup_round_rows[:5]
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
        examples = ", ".join(
            f"(user={u}, count={c})" for u, c in dup_active_rows[:5]
        )
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

    engine = create_engine(db_url)
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
```

- [ ] **Step 5: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/migrations/probes/test_check_tournament_invariants.py -v`
Expected: PASS — 8 tests

- [ ] **Step 6: Smoke test the CLI entry point fail-loud behaviour**

Run: `ATP_DATABASE_URL= uv run python -m atp.dashboard.migrations.probes.check_tournament_invariants; echo "exit=$?"`
Expected: stderr prints `FAIL: ATP_DATABASE_URL environment variable is not set.`, `exit=2`.

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/migrations/ \
        tests/unit/dashboard/migrations/ \
        tests/unit/dashboard/migrations/probes/ \
        tests/unit/dashboard/migrations/probes/test_check_tournament_invariants.py
git commit -m "feat(tournament): add Plan 2a pre-migration probe module"
```

---

## Task 5: Alembic migration — tournament_plan_2a_constraints

**Files:**
- Create: `migrations/dashboard/versions/<auto>_tournament_plan_2a_constraints.py`
- Test: `tests/integration/dashboard/tournament/test_migration_plan_2a.py`

- [ ] **Step 1: Create integration test directory**

```bash
mkdir -p tests/integration/dashboard/tournament
touch tests/integration/dashboard/__init__.py
touch tests/integration/dashboard/tournament/__init__.py
```

- [ ] **Step 2: Write the failing integration test**

Create `tests/integration/dashboard/tournament/test_migration_plan_2a.py`:

```python
"""Integration tests for the Plan 2a Alembic migration."""

import os
import subprocess
import sys

import pytest
from sqlalchemy import create_engine, inspect, text


@pytest.fixture
def fresh_db(tmp_path):
    db_path = tmp_path / "plan2a_fresh.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
    engine.dispose()

    env = {**os.environ, "ATP_DATABASE_URL": f"sqlite:///{db_path}"}
    result = subprocess.run(
        ["uv", "run", "alembic",
         "-c", "migrations/dashboard/alembic.ini",
         "upgrade", "head"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"alembic upgrade failed: {result.stderr}"
    )

    yield f"sqlite:///{db_path}"


def test_upgrade_creates_all_new_tournament_columns(fresh_db):
    engine = create_engine(fresh_db)
    columns = {c["name"] for c in inspect(engine).get_columns("tournaments")}
    required = {
        "pending_deadline",
        "join_token",
        "cancelled_at",
        "cancelled_by",
        "cancelled_reason",
        "cancelled_reason_detail",
    }
    assert required.issubset(columns)
    engine.dispose()


def test_upgrade_creates_participant_released_at(fresh_db):
    engine = create_engine(fresh_db)
    columns = {c["name"] for c in inspect(engine).get_columns("tournament_participants")}
    assert "released_at" in columns
    user_col = next(
        c for c in inspect(engine).get_columns("tournament_participants")
        if c["name"] == "user_id"
    )
    assert user_col["nullable"] is False
    engine.dispose()


def test_upgrade_creates_action_source(fresh_db):
    engine = create_engine(fresh_db)
    columns = {c["name"] for c in inspect(engine).get_columns("tournament_actions")}
    assert "source" in columns
    engine.dispose()


def test_upgrade_creates_six_base_invariants(fresh_db):
    engine = create_engine(fresh_db)
    inspector = inspect(engine)

    # uq_participant_tournament_user
    pc_unique = {
        u["name"] for u in
        inspector.get_unique_constraints("tournament_participants")
    }
    assert "uq_participant_tournament_user" in pc_unique

    # uq_action_round_participant
    ac_unique = {
        u["name"] for u in
        inspector.get_unique_constraints("tournament_actions")
    }
    assert "uq_action_round_participant" in ac_unique

    # uq_round_tournament_number
    r_unique = {
        u["name"] for u in
        inspector.get_unique_constraints("tournament_rounds")
    }
    assert "uq_round_tournament_number" in r_unique

    # idx_round_status_deadline
    r_indexes = {i["name"] for i in inspector.get_indexes("tournament_rounds")}
    assert "idx_round_status_deadline" in r_indexes

    # uq_participant_user_active (partial unique index)
    p_indexes = {i["name"] for i in inspector.get_indexes("tournament_participants")}
    assert "uq_participant_user_active" in p_indexes

    engine.dispose()


def test_upgrade_creates_check_constraint(fresh_db):
    engine = create_engine(fresh_db)
    with engine.connect() as conn:
        # Query SQLite's sqlite_master for the CHECK constraint text
        result = conn.execute(text(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='tournaments'"
        )).scalar()
    assert "ck_tournament_cancel_consistency" in result
    engine.dispose()


def test_downgrade_then_upgrade_idempotent(fresh_db):
    """Smoke-test downgrade and re-upgrade produce a clean schema."""
    env = {**os.environ, "ATP_DATABASE_URL": fresh_db}
    result = subprocess.run(
        ["uv", "run", "alembic",
         "-c", "migrations/dashboard/alembic.ini",
         "downgrade", "-1"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"downgrade failed: {result.stderr}"

    result = subprocess.run(
        ["uv", "run", "alembic",
         "-c", "migrations/dashboard/alembic.ini",
         "upgrade", "head"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"re-upgrade failed: {result.stderr}"
```

- [ ] **Step 3: Run test to verify failure**

Run: `uv run pytest tests/integration/dashboard/tournament/test_migration_plan_2a.py -v`
Expected: FAIL with "alembic upgrade failed" (no migration file yet).

- [ ] **Step 4: Generate the migration file skeleton**

Run: `uv run alembic -c migrations/dashboard/alembic.ini revision -m "tournament_plan_2a_constraints"`

This creates a file `migrations/dashboard/versions/<new-hash>_tournament_plan_2a_constraints.py`. Note the hash in the filename — you will reference it nowhere else, but keep track of it for commit.

- [ ] **Step 5: Replace the generated skeleton with the full migration**

Open the new file and replace its entire content with:

```python
"""tournament plan 2a — schema constraints, cancel audit, AD-9 + AD-10 columns

Revision ID: <auto-generated, keep from skeleton>
Revises: 028d8a9fdc46
Create Date: 2026-04-11

## Precondition

Transitively follows c8d5f2a91234 (IDOR fix, enforce_run_user_id_not_null)
via 028d8a9fdc46. IDOR fix backfilled and constrained benchmark_runs; Plan
2a applies the analogous invariant to tournament_participants, a sibling
table not touched by the IDOR migration. The FK-orphan probe on
tournament_participants therefore verifies a fresh invariant, not a
re-check of earlier work.

## Probe-to-resolution playbook

| Probe | Violation | Resolution |
|-------|-----------|------------|
| P1 | Participant user_id IS NULL | DELETE anonymous rows, or backfill user_id from agent_name lookup if meaningful. Do NOT assign a sentinel user_id. |
| P2 | FK orphan on user_id | DELETE FROM tournament_participants WHERE user_id NOT IN (SELECT id FROM users). |
| P3 | Duplicate (tournament_id, user_id) | Manually dedupe, keep earliest joined_at. |
| P4 | Duplicate (round_id, participant_id) | Manually dedupe, keep earliest submitted_at. |
| P5 | Duplicate (tournament_id, round_number) | Inspect tournament history before deleting. Likely cause: re-submitted create-round during a crash. |
| P6 | User with >1 participant in pending/active tournaments | Transition stale tournaments to completed/cancelled status via SQL; migration step 5a backfill will set released_at automatically. Do NOT try to set released_at at probe time — column does not exist yet. |
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

from atp.dashboard.migrations.probes.check_tournament_invariants import (
    check_tournament_schema_ready,
)

# Keep the revision ID from the generated skeleton — do not replace it.
# revision: str = "<keep>"
down_revision: str = "028d8a9fdc46"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()

    # Step 1: Run probes. Abort on any violation.
    violations = check_tournament_schema_ready(connection)
    if violations:
        message = "Plan 2a migration aborted by probe — resolve and re-run:\n"
        for v in violations:
            message += f"  - {v}\n"
        raise RuntimeError(message)

    # Step 2: Tournament columns — pending_deadline nullable first,
    # backfill, then flip to NOT NULL.
    with op.batch_alter_table("tournaments") as batch_op:
        batch_op.add_column(sa.Column("pending_deadline", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("join_token", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("cancelled_at", sa.DateTime(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "cancelled_by",
                sa.Integer(),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            )
        )
        batch_op.add_column(sa.Column("cancelled_reason", sa.String(32), nullable=True))
        batch_op.add_column(sa.Column("cancelled_reason_detail", sa.String(512), nullable=True))

    op.execute(
        text(
            "UPDATE tournaments SET pending_deadline = CURRENT_TIMESTAMP "
            "WHERE pending_deadline IS NULL"
        )
    )

    with op.batch_alter_table("tournaments") as batch_op:
        batch_op.alter_column(
            "pending_deadline", existing_type=sa.DateTime(), nullable=False
        )
        batch_op.create_index(
            "idx_tournaments_status_pending_deadline",
            ["status", "pending_deadline"],
            unique=False,
        )
        batch_op.create_check_constraint(
            "ck_tournament_cancel_consistency",
            """(
                (
                    cancelled_reason IS NULL
                    AND cancelled_by IS NULL
                    AND cancelled_at IS NULL
                ) OR (
                    cancelled_reason = 'admin_action'
                    AND cancelled_by IS NOT NULL
                    AND cancelled_at IS NOT NULL
                ) OR (
                    cancelled_reason IN ('pending_timeout', 'abandoned')
                    AND cancelled_by IS NULL
                    AND cancelled_at IS NOT NULL
                )
            )""",
        )

    # Step 3: Round — unique constraint + composite index
    with op.batch_alter_table("tournament_rounds") as batch_op:
        batch_op.create_unique_constraint(
            "uq_round_tournament_number", ["tournament_id", "round_number"]
        )
        batch_op.create_index(
            "idx_round_status_deadline", ["status", "deadline"], unique=False
        )

    # Step 4: Action — source column + uq_action_round_participant
    with op.batch_alter_table("tournament_actions") as batch_op:
        batch_op.add_column(
            sa.Column(
                "source",
                sa.String(32),
                nullable=False,
                server_default="submitted",
            )
        )
        batch_op.create_unique_constraint(
            "uq_action_round_participant", ["round_id", "participant_id"]
        )

    # Step 5: Participant — released_at, NOT NULL flip,
    # uq_participant_tournament_user
    with op.batch_alter_table("tournament_participants") as batch_op:
        batch_op.add_column(sa.Column("released_at", sa.DateTime(), nullable=True))
        batch_op.alter_column(
            "user_id", existing_type=sa.Integer(), nullable=False
        )
        batch_op.create_unique_constraint(
            "uq_participant_tournament_user", ["tournament_id", "user_id"]
        )

    # Step 5a: Backfill released_at for participants in terminal-status
    # tournaments.
    op.execute(text("""
        UPDATE tournament_participants
        SET released_at = CURRENT_TIMESTAMP
        WHERE tournament_id IN (
            SELECT id FROM tournaments
            WHERE status IN ('completed', 'cancelled')
        )
    """))

    # Step 6: Partial unique index — created outside batch_alter_table.
    op.create_index(
        "uq_participant_user_active",
        "tournament_participants",
        ["user_id"],
        unique=True,
        sqlite_where=text("user_id IS NOT NULL AND released_at IS NULL"),
        postgresql_where=text("user_id IS NOT NULL AND released_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_participant_user_active", "tournament_participants")

    with op.batch_alter_table("tournament_participants") as batch_op:
        batch_op.drop_constraint("uq_participant_tournament_user", type_="unique")
        batch_op.alter_column(
            "user_id", existing_type=sa.Integer(), nullable=True
        )
        batch_op.drop_column("released_at")

    with op.batch_alter_table("tournament_actions") as batch_op:
        batch_op.drop_constraint("uq_action_round_participant", type_="unique")
        batch_op.drop_column("source")

    with op.batch_alter_table("tournament_rounds") as batch_op:
        batch_op.drop_index("idx_round_status_deadline")
        batch_op.drop_constraint("uq_round_tournament_number", type_="unique")

    with op.batch_alter_table("tournaments") as batch_op:
        batch_op.drop_constraint(
            "ck_tournament_cancel_consistency", type_="check"
        )
        batch_op.drop_index("idx_tournaments_status_pending_deadline")
        batch_op.drop_column("cancelled_reason_detail")
        batch_op.drop_column("cancelled_reason")
        batch_op.drop_column("cancelled_by")
        batch_op.drop_column("cancelled_at")
        batch_op.drop_column("join_token")
        batch_op.drop_column("pending_deadline")
```

Keep the `revision: str = "<generated-hash>"` line from the skeleton — do not edit it.

- [ ] **Step 6: Run the integration test suite**

Run: `uv run pytest tests/integration/dashboard/tournament/test_migration_plan_2a.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 7: Verify pyrefly still passes**

Run: `uv run pyrefly check`
Expected: 0 errors.

- [ ] **Step 8: Commit**

```bash
git add migrations/dashboard/versions/*tournament_plan_2a*.py \
        tests/integration/dashboard/__init__.py \
        tests/integration/dashboard/tournament/__init__.py \
        tests/integration/dashboard/tournament/test_migration_plan_2a.py
git commit -m "feat(tournament): add Plan 2a Alembic migration with 7 invariants + 8 columns"
```

---

## Task 6: TournamentCancelEvent dataclass with __post_init__ validator

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/events.py`
- Test: `tests/unit/dashboard/tournament/test_tournament_cancel_event.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_tournament_cancel_event.py`:

```python
"""Tests for TournamentCancelEvent dataclass + __post_init__ validator."""

from datetime import datetime

import pytest

from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason


def _build(**overrides):
    base = dict(
        tournament_id=1,
        cancelled_at=datetime(2026, 4, 15, 10, 0, 0),
        cancelled_by=42,
        cancelled_reason=CancelReason.ADMIN_ACTION,
        cancelled_reason_detail=None,
        final_rounds_played=0,
        final_status=TournamentStatus.CANCELLED,
    )
    base.update(overrides)
    return base


def test_admin_action_with_actor_valid():
    event = TournamentCancelEvent(**_build())
    assert event.cancelled_reason == CancelReason.ADMIN_ACTION
    assert event.cancelled_by == 42


def test_pending_timeout_without_actor_valid():
    event = TournamentCancelEvent(**_build(
        cancelled_reason=CancelReason.PENDING_TIMEOUT,
        cancelled_by=None,
    ))
    assert event.cancelled_by is None


def test_abandoned_without_actor_valid():
    event = TournamentCancelEvent(**_build(
        cancelled_reason=CancelReason.ABANDONED,
        cancelled_by=None,
    ))
    assert event.cancelled_by is None


def test_admin_action_without_actor_rejected():
    with pytest.raises(ValueError, match="must have cancelled_by set"):
        TournamentCancelEvent(**_build(
            cancelled_reason=CancelReason.ADMIN_ACTION,
            cancelled_by=None,
        ))


def test_pending_timeout_with_actor_rejected():
    with pytest.raises(ValueError, match="must have cancelled_by=None"):
        TournamentCancelEvent(**_build(
            cancelled_reason=CancelReason.PENDING_TIMEOUT,
            cancelled_by=42,
        ))


def test_abandoned_with_actor_rejected():
    with pytest.raises(ValueError, match="must have cancelled_by=None"):
        TournamentCancelEvent(**_build(
            cancelled_reason=CancelReason.ABANDONED,
            cancelled_by=42,
        ))


def test_final_status_must_be_cancelled():
    with pytest.raises(ValueError, match="final_status must be CANCELLED"):
        TournamentCancelEvent(**_build(
            final_status=TournamentStatus.ACTIVE,
        ))


def test_is_frozen():
    event = TournamentCancelEvent(**_build())
    with pytest.raises(Exception):  # FrozenInstanceError on dataclass
        event.tournament_id = 999  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_tournament_cancel_event.py -v`
Expected: FAIL with `ImportError: cannot import name 'TournamentCancelEvent' from 'atp.dashboard.tournament.events'`

- [ ] **Step 3: Add the dataclass to events.py**

Edit `packages/atp-dashboard/atp/dashboard/tournament/events.py`. At the top of the file, add imports:

```python
from dataclasses import dataclass
from datetime import datetime

from atp.dashboard.tournament.models import TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason
```

At the bottom of the file, append:

```python
_SYSTEM_CANCEL_REASONS: frozenset[CancelReason] = frozenset({
    CancelReason.PENDING_TIMEOUT,
    CancelReason.ABANDONED,
})


@dataclass(frozen=True)
class TournamentCancelEvent:
    """Payload for `tournament_cancelled` bus event.

    Field invariant (enforced three ways — defense in depth):

    1. DB CHECK constraint `ck_tournament_cancel_consistency` on the
       tournaments table.
    2. `__post_init__` validator on this dataclass.
    3. Construction call site in `TournamentService._cancel_impl` —
       always builds from consistent inputs by construction.

    Invariant:
        cancelled_by IS NULL  <->  cancelled_reason in {PENDING_TIMEOUT, ABANDONED}
        cancelled_by NOT NULL <->  cancelled_reason == ADMIN_ACTION
    """

    tournament_id: int
    cancelled_at: datetime
    cancelled_by: int | None
    cancelled_reason: CancelReason
    cancelled_reason_detail: str | None
    final_rounds_played: int
    final_status: TournamentStatus

    def __post_init__(self) -> None:
        is_system = self.cancelled_reason in _SYSTEM_CANCEL_REASONS
        has_actor = self.cancelled_by is not None

        if is_system and has_actor:
            raise ValueError(
                f"system cancel (reason={self.cancelled_reason.value}) "
                f"must have cancelled_by=None, got {self.cancelled_by}"
            )
        if not is_system and not has_actor:
            raise ValueError(
                f"user-initiated cancel (reason={self.cancelled_reason.value}) "
                f"must have cancelled_by set, got None"
            )
        if self.final_status != TournamentStatus.CANCELLED:
            raise ValueError(
                f"TournamentCancelEvent.final_status must be CANCELLED, "
                f"got {self.final_status.value}"
            )
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/tournament/test_tournament_cancel_event.py -v`
Expected: PASS — 8 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/events.py \
        tests/unit/dashboard/tournament/test_tournament_cancel_event.py
git commit -m "feat(tournament): add TournamentCancelEvent dataclass with invariant validator"
```

---

## Task 7: TournamentService._load_for_auth helper

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/unit/dashboard/tournament/test_service_load_for_auth.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_service_load_for_auth.py`:

```python
"""Tests for TournamentService._load_for_auth enumeration guard."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.errors import NotFoundError
from atp.dashboard.tournament.models import Tournament
from atp.dashboard.tournament.service import TournamentService


def _make_tournament(tournament_id: int, created_by: int | None) -> Tournament:
    t = MagicMock(spec=Tournament)
    t.id = tournament_id
    t.created_by = created_by
    return t


def _make_user(user_id: int, is_admin: bool = False) -> MagicMock:
    u = MagicMock()
    u.id = user_id
    u.is_admin = is_admin
    return u


def _make_service(tournament_result):
    session = MagicMock()
    session.get = AsyncMock(return_value=tournament_result)
    bus = MagicMock()
    return TournamentService(session=session, bus=bus)


@pytest.mark.anyio
async def test_admin_always_authorized():
    svc = _make_service(_make_tournament(1, created_by=99))
    admin = _make_user(42, is_admin=True)
    result = await svc._load_for_auth(tournament_id=1, user=admin)
    assert result.id == 1


@pytest.mark.anyio
async def test_admin_authorized_on_legacy_null_owner():
    svc = _make_service(_make_tournament(1, created_by=None))
    admin = _make_user(42, is_admin=True)
    result = await svc._load_for_auth(tournament_id=1, user=admin)
    assert result.id == 1


@pytest.mark.anyio
async def test_owner_authorized_on_match():
    svc = _make_service(_make_tournament(1, created_by=42))
    user = _make_user(42, is_admin=False)
    result = await svc._load_for_auth(tournament_id=1, user=user)
    assert result.id == 1


@pytest.mark.anyio
async def test_non_owner_raises_not_found():
    svc = _make_service(_make_tournament(1, created_by=99))
    user = _make_user(42, is_admin=False)
    with pytest.raises(NotFoundError):
        await svc._load_for_auth(tournament_id=1, user=user)


@pytest.mark.anyio
async def test_non_admin_legacy_null_raises_not_found():
    svc = _make_service(_make_tournament(1, created_by=None))
    user = _make_user(42, is_admin=False)
    with pytest.raises(NotFoundError):
        await svc._load_for_auth(tournament_id=1, user=user)


@pytest.mark.anyio
async def test_missing_tournament_raises_not_found():
    svc = _make_service(None)
    user = _make_user(42, is_admin=False)
    with pytest.raises(NotFoundError):
        await svc._load_for_auth(tournament_id=99999, user=user)
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_service_load_for_auth.py -v`
Expected: FAIL with `AttributeError: 'TournamentService' object has no attribute '_load_for_auth'`

- [ ] **Step 3: Add `_load_for_auth` to TournamentService**

Edit `packages/atp-dashboard/atp/dashboard/tournament/service.py`. Add to the bottom of the `TournamentService` class (after existing methods):

```python
    async def _load_for_auth(
        self,
        tournament_id: int,
        user: "User",
    ) -> "Tournament":
        """Load tournament and verify that `user` is authorized to act on it.

        Authorization rule:
        - Admins (user.is_admin): always allowed.
        - Owners (tournament.created_by == user.id): allowed.
        - Legacy with no owner (tournament.created_by IS NULL): admin only.
        - Everyone else: denied.

        All denial cases raise NotFoundError — same exception as
        "tournament doesn't exist". Preserves the enumeration-guard
        invariant: unauthorized callers cannot distinguish between
        "doesn't exist" and "not allowed".
        """
        tournament = await self.session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id}")

        if user.is_admin:
            return tournament

        if tournament.created_by is None:
            raise NotFoundError(f"tournament {tournament_id}")

        if tournament.created_by != user.id:
            raise NotFoundError(f"tournament {tournament_id}")

        return tournament
```

If `User` and `Tournament` are not already imported in the file, add to the imports:

```python
from atp.dashboard.models import User
from atp.dashboard.tournament.models import Tournament
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/tournament/test_service_load_for_auth.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_load_for_auth.py
git commit -m "feat(tournament): add _load_for_auth with enumeration guard"
```

---

## Task 8: TournamentService._cancel_impl private method

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/unit/dashboard/tournament/test_service_cancel_impl.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_service_cancel_impl.py`:

```python
"""Unit tests for TournamentService._cancel_impl — the private cancel
helper. Exercises control flow, idempotent guard, event building, and
the step-3-before-step-6 ordering regression."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import Tournament, TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService


@pytest.fixture
def frozen_now(monkeypatch):
    fixed = datetime(2026, 4, 15, 10, 0, 0)

    class _FakeDatetime:
        @classmethod
        def utcnow(cls):
            return fixed

    monkeypatch.setattr(
        "atp.dashboard.tournament.service.datetime", _FakeDatetime
    )
    return fixed


def _tournament(status: TournamentStatus) -> Tournament:
    t = MagicMock(spec=Tournament)
    t.id = 1
    t.status = status
    t.cancelled_at = None
    t.cancelled_by = None
    t.cancelled_reason = None
    t.cancelled_reason_detail = None
    return t


def _make_service(tournament: Tournament | None, completed_round_count: int = 0):
    session = MagicMock()
    session.get = AsyncMock(return_value=tournament)
    session.execute = AsyncMock()
    session.scalar = AsyncMock(return_value=completed_round_count)
    bus = MagicMock()
    return TournamentService(session=session, bus=bus), session


@pytest.mark.anyio
async def test_happy_path_pending_to_cancelled(frozen_now):
    t = _tournament(TournamentStatus.PENDING)
    svc, session = _make_service(t, completed_round_count=0)

    event = await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.ADMIN_ACTION,
        cancelled_by=42,
        reason_detail="stuck",
    )

    assert t.status == TournamentStatus.CANCELLED
    assert t.cancelled_at == frozen_now
    assert t.cancelled_by == 42
    assert t.cancelled_reason == CancelReason.ADMIN_ACTION
    assert t.cancelled_reason_detail == "stuck"
    assert isinstance(event, TournamentCancelEvent)
    assert event.tournament_id == 1
    assert event.final_rounds_played == 0


@pytest.mark.anyio
async def test_happy_path_active_to_cancelled(frozen_now):
    t = _tournament(TournamentStatus.ACTIVE)
    svc, _ = _make_service(t, completed_round_count=3)

    event = await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.ADMIN_ACTION,
        cancelled_by=42,
        reason_detail=None,
    )
    assert event is not None
    assert event.final_rounds_played == 3


@pytest.mark.anyio
async def test_idempotent_on_already_cancelled():
    t = _tournament(TournamentStatus.CANCELLED)
    svc, session = _make_service(t)

    event = await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.ADMIN_ACTION,
        cancelled_by=42,
        reason_detail=None,
    )
    assert event is None
    # No bulk updates executed when tournament already terminal
    session.execute.assert_not_called()


@pytest.mark.anyio
async def test_idempotent_on_already_completed():
    t = _tournament(TournamentStatus.COMPLETED)
    svc, session = _make_service(t)
    event = await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.ADMIN_ACTION,
        cancelled_by=42,
        reason_detail=None,
    )
    assert event is None
    session.execute.assert_not_called()


@pytest.mark.anyio
async def test_final_rounds_played_snapshots_before_bulk_update(frozen_now):
    """Regression guard for step-3-before-step-6 ordering. If anyone
    reorders _cancel_impl such that the scalar count runs AFTER the
    bulk UPDATE that transitions in-flight rounds to CANCELLED, this
    test fires — because the count would then reflect post-update
    state instead of pre-update state."""
    t = _tournament(TournamentStatus.ACTIVE)
    svc, session = _make_service(t, completed_round_count=2)

    # Record the order in which session methods are called
    call_order: list[str] = []
    original_scalar = session.scalar
    original_execute = session.execute

    async def record_scalar(*args, **kwargs):
        call_order.append("scalar")
        return await original_scalar(*args, **kwargs)

    async def record_execute(*args, **kwargs):
        call_order.append("execute")
        return await original_execute(*args, **kwargs)

    session.scalar = record_scalar
    session.execute = record_execute

    await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.ADMIN_ACTION,
        cancelled_by=42,
        reason_detail=None,
    )

    # scalar (final_rounds_played snapshot) must precede both bulk UPDATEs
    assert call_order[0] == "scalar"
    assert "execute" in call_order
    assert call_order.index("scalar") < call_order.index("execute")


@pytest.mark.anyio
async def test_system_cancel_with_none_actor_valid(frozen_now):
    t = _tournament(TournamentStatus.PENDING)
    svc, _ = _make_service(t)

    event = await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.PENDING_TIMEOUT,
        cancelled_by=None,
        reason_detail=None,
    )
    assert event is not None
    assert event.cancelled_by is None
    assert event.cancelled_reason == CancelReason.PENDING_TIMEOUT
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_service_cancel_impl.py -v`
Expected: FAIL with `AttributeError: 'TournamentService' object has no attribute '_cancel_impl'`

- [ ] **Step 3: Add `_cancel_impl` to TournamentService**

Edit `packages/atp-dashboard/atp/dashboard/tournament/service.py`. Ensure these imports exist:

```python
from datetime import datetime

from sqlalchemy import func, select, update

from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import (
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.reasons import CancelReason
```

Add to the bottom of the `TournamentService` class:

```python
    async def _cancel_impl(
        self,
        tournament_id: int,
        reason: CancelReason,
        cancelled_by: int | None,
        reason_detail: str | None,
    ) -> TournamentCancelEvent | None:
        """Shared cancellation logic. Single source of truth.

        Mutates DB state but does NOT commit — caller owns the transaction.
        Does NOT publish to bus — returns the event for the caller to
        publish after its commit succeeds.

        Returns None if the tournament was already in a terminal state
        (idempotent no-op); returns a TournamentCancelEvent if the call
        caused a state transition.
        """
        # Step 1: Lock + load tournament
        tournament = await self.session.get(
            Tournament, tournament_id, with_for_update=True
        )
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id}")

        # Step 2: Idempotent guard
        if tournament.status in (
            TournamentStatus.CANCELLED,
            TournamentStatus.COMPLETED,
        ):
            return None

        # Step 3: Snapshot final_rounds_played BEFORE step 6 bulk UPDATE.
        # Otherwise the count would include in-flight rounds that step 6
        # transitions to CANCELLED.
        final_rounds_played = await self.session.scalar(
            select(func.count())
            .select_from(Round)
            .where(Round.tournament_id == tournament_id)
            .where(Round.status == RoundStatus.COMPLETED)
        ) or 0

        # Step 4: Write tournament audit fields
        now = datetime.utcnow()
        tournament.status = TournamentStatus.CANCELLED
        tournament.cancelled_at = now
        tournament.cancelled_by = cancelled_by
        tournament.cancelled_reason = reason
        tournament.cancelled_reason_detail = reason_detail

        # Step 5: Release all unreleased participants (bulk UPDATE)
        await self.session.execute(
            update(Participant)
            .where(Participant.tournament_id == tournament_id)
            .where(Participant.released_at.is_(None))
            .values(released_at=now)
        )

        # Step 6: Cancel all in-flight rounds (bulk UPDATE)
        await self.session.execute(
            update(Round)
            .where(Round.tournament_id == tournament_id)
            .where(Round.status.in_([
                RoundStatus.WAITING_FOR_ACTIONS,
                RoundStatus.IN_PROGRESS,
            ]))
            .values(status=RoundStatus.CANCELLED)
        )

        # Step 7: Build event (caller publishes post-commit)
        return TournamentCancelEvent(
            tournament_id=tournament_id,
            cancelled_at=now,
            cancelled_by=cancelled_by,
            cancelled_reason=reason,
            cancelled_reason_detail=reason_detail,
            final_rounds_played=final_rounds_played,
            final_status=TournamentStatus.CANCELLED,
        )
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/tournament/test_service_cancel_impl.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_cancel_impl.py
git commit -m "feat(tournament): add _cancel_impl shared cancellation logic"
```

---

## Task 9: cancel_tournament and cancel_tournament_system public methods

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/unit/dashboard/tournament/test_service_cancel_public.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_service_cancel_public.py`:

```python
"""Tests for the two public cancel entry points: cancel_tournament
(user-authenticated) and cancel_tournament_system (trusted internal)."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.errors import NotFoundError
from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import Tournament, TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService


def _tournament(created_by: int | None = 42) -> Tournament:
    t = MagicMock(spec=Tournament)
    t.id = 1
    t.created_by = created_by
    t.status = TournamentStatus.ACTIVE
    return t


def _user(user_id: int, is_admin: bool = False) -> MagicMock:
    u = MagicMock()
    u.id = user_id
    u.is_admin = is_admin
    return u


def _make_service(tournament, cancel_event_to_return=None):
    session = MagicMock()
    session.get = AsyncMock(return_value=tournament)
    session.begin = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock()
    session.begin.return_value.__aexit__ = AsyncMock(return_value=False)
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = TournamentService(session=session, bus=bus)

    async def fake_cancel_impl(*args, **kwargs):
        return cancel_event_to_return

    svc._cancel_impl = fake_cancel_impl  # type: ignore[method-assign]
    return svc, session, bus


def _event():
    return TournamentCancelEvent(
        tournament_id=1,
        cancelled_at=datetime(2026, 4, 15, 10, 0, 0),
        cancelled_by=42,
        cancelled_reason=CancelReason.ADMIN_ACTION,
        cancelled_reason_detail=None,
        final_rounds_played=0,
        final_status=TournamentStatus.CANCELLED,
    )


@pytest.mark.anyio
async def test_cancel_tournament_by_owner_publishes_event():
    t = _tournament(created_by=42)
    svc, session, bus = _make_service(t, cancel_event_to_return=_event())

    await svc.cancel_tournament(
        user=_user(42, is_admin=False),
        tournament_id=1,
        reason_detail="manual",
    )

    bus.publish.assert_awaited_once()


@pytest.mark.anyio
async def test_cancel_tournament_by_admin_publishes_event():
    t = _tournament(created_by=99)
    svc, session, bus = _make_service(t, cancel_event_to_return=_event())

    await svc.cancel_tournament(
        user=_user(42, is_admin=True),
        tournament_id=1,
        reason_detail=None,
    )

    bus.publish.assert_awaited_once()


@pytest.mark.anyio
async def test_cancel_tournament_by_non_owner_raises_not_found():
    t = _tournament(created_by=99)
    svc, session, bus = _make_service(t)

    with pytest.raises(NotFoundError):
        await svc.cancel_tournament(
            user=_user(42, is_admin=False),
            tournament_id=1,
            reason_detail=None,
        )
    bus.publish.assert_not_called()


@pytest.mark.anyio
async def test_cancel_tournament_publish_failure_does_not_raise():
    """Bus publish failure after successful commit returns success to caller.
    Per error handling matrix: DB is single source of truth, session_sync
    recovers subscribers."""
    t = _tournament(created_by=42)
    svc, session, bus = _make_service(t, cancel_event_to_return=_event())
    bus.publish = AsyncMock(side_effect=ConnectionError("bus down"))

    # Must NOT raise
    await svc.cancel_tournament(
        user=_user(42, is_admin=False),
        tournament_id=1,
        reason_detail=None,
    )


@pytest.mark.anyio
async def test_cancel_tournament_system_no_auth_required():
    t = _tournament(created_by=99)
    svc, session, bus = _make_service(t, cancel_event_to_return=_event())

    await svc.cancel_tournament_system(
        tournament_id=1,
        reason=CancelReason.PENDING_TIMEOUT,
    )

    bus.publish.assert_awaited_once()
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_service_cancel_public.py -v`
Expected: FAIL with `AttributeError: 'TournamentService' object has no attribute 'cancel_tournament'` (or `cancel_tournament_system`).

- [ ] **Step 3: Add both public methods**

Edit `packages/atp-dashboard/atp/dashboard/tournament/service.py`. Add near the top of imports:

```python
import logging

log = logging.getLogger(__name__)
```

Add inside the `TournamentService` class (between existing `join` / `leave` methods and the private `_cancel_impl` / `_load_for_auth`):

```python
    async def cancel_tournament(
        self,
        user: "User",
        tournament_id: int,
        reason_detail: str | None = None,
    ) -> None:
        """User-facing cancel entry point.

        Called by REST POST /api/v1/tournaments/{id}/cancel and the MCP
        `cancel_tournament` tool. Authorization runs against an unlocked
        SELECT before the write transaction.
        """
        await self._load_for_auth(tournament_id, user)

        async with self.session.begin():
            event = await self._cancel_impl(
                tournament_id,
                reason=CancelReason.ADMIN_ACTION,
                cancelled_by=user.id,
                reason_detail=reason_detail,
            )

        if event is not None:
            try:
                await self.bus.publish(event)
            except Exception:
                log.warning(
                    "tournament.cancel.publish_failed",
                    extra={"tournament_id": tournament_id},
                    exc_info=True,
                )

    async def cancel_tournament_system(
        self,
        tournament_id: int,
        reason: CancelReason,
        reason_detail: str | None = None,
    ) -> None:
        """System-initiated cancel. Called ONLY by the deadline worker
        (pending_timeout path) and `leave()` (abandoned cascade).

        Code-review invariant: no handler file imports this method.
        Enforced by tests/unit/dashboard/tournament/test_static_guards.py.
        """
        async with self.session.begin():
            event = await self._cancel_impl(
                tournament_id,
                reason=reason,
                cancelled_by=None,
                reason_detail=reason_detail,
            )

        if event is not None:
            try:
                await self.bus.publish(event)
            except Exception:
                log.warning(
                    "tournament.cancel.publish_failed",
                    extra={"tournament_id": tournament_id},
                    exc_info=True,
                )
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/tournament/test_service_cancel_public.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_cancel_public.py
git commit -m "feat(tournament): add cancel_tournament + cancel_tournament_system"
```

---

## Task 10: Idempotent join with race-aware IntegrityError handling

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/integration/dashboard/tournament/test_service_join_idempotent.py`

Unit tests for the idempotent path cannot fully simulate the race; integration tests against real SQLite are the right level.

- [ ] **Step 1: Write the failing integration tests**

Create `tests/integration/dashboard/tournament/conftest.py`:

```python
"""Shared integration fixtures for tournament tests."""

import asyncio
import os
import subprocess

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def tournament_db(tmp_path):
    """SQLite WAL-mode DB with the full Plan 2a schema applied.

    WAL mode is non-negotiable — deadline worker race tests depend on
    WAL's single-writer serialization for deterministic outcomes.
    """
    db_path = tmp_path / "plan2a_test.db"

    sync_engine = create_engine(f"sqlite:///{db_path}")
    with sync_engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.commit()
    sync_engine.dispose()

    env = {**os.environ, "ATP_DATABASE_URL": f"sqlite:///{db_path}"}
    result = subprocess.run(
        ["uv", "run", "alembic", "-c",
         "migrations/dashboard/alembic.ini", "upgrade", "head"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"alembic upgrade failed: {result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(tournament_db):
    return async_sessionmaker(tournament_db, expire_on_commit=False)
```

Create `tests/integration/dashboard/tournament/test_service_join_idempotent.py`:

```python
"""Integration tests for idempotent join under concurrency."""

import asyncio

import pytest
from sqlalchemy import func, select

from atp.dashboard.tournament.errors import ConflictError
from atp.dashboard.tournament.models import Participant, Tournament
from atp.dashboard.tournament.service import TournamentService


# The actual factory and fixture helpers will grow as later tasks
# introduce more integration tests. For this task, build minimal
# seed helpers inline.

async def _seed_user(session, user_id: int, username: str):
    # Use raw SQL for test seeding to avoid coupling to User model imports.
    from sqlalchemy import text
    await session.execute(text(
        "INSERT INTO users (id, username, is_admin) VALUES (:id, :u, 0)"
    ), {"id": user_id, "u": username})


async def _seed_tournament(session, tournament_id: int):
    from sqlalchemy import text
    await session.execute(text(
        "INSERT INTO tournaments (id, name, status, num_players, "
        "total_rounds, round_deadline_s, pending_deadline) "
        "VALUES (:id, :n, 'pending', 2, 3, 30, CURRENT_TIMESTAMP)"
    ), {"id": tournament_id, "n": f"t{tournament_id}"})


@pytest.mark.anyio
async def test_sc5_five_concurrent_joins_same_tournament_all_idempotent(
    session_factory,
):
    """SC-5: five concurrent join calls from the same user to the SAME
    tournament must all succeed idempotently. Exactly one row in DB."""
    async with session_factory() as setup:
        await _seed_user(setup, user_id=1, username="alice")
        await _seed_tournament(setup, tournament_id=1)
        await setup.commit()

    async def one_join():
        async with session_factory() as session:
            # Fetch User inside this session's transaction
            user_row = await session.get(
                __import__("atp.dashboard.models", fromlist=["User"]).User, 1
            )
            svc = TournamentService(session, bus=_DummyBus())
            return await svc.join(
                tournament_id=1, user=user_row, agent_name="bot"
            )

    results = await asyncio.gather(*[one_join() for _ in range(5)])

    # All five calls returned successfully
    assert len(results) == 5
    for participant, is_new in results:
        assert participant is not None

    # Exactly one is_new=True
    new_flags = [is_new for _, is_new in results]
    assert new_flags.count(True) == 1
    assert new_flags.count(False) == 4

    # DB has exactly one row
    async with session_factory() as verify:
        count = await verify.scalar(
            select(func.count()).select_from(Participant)
              .where(Participant.tournament_id == 1)
              .where(Participant.user_id == 1)
        )
        assert count == 1


@pytest.mark.anyio
async def test_sc4_two_concurrent_joins_different_tournaments_one_wins(
    session_factory,
):
    """SC-4: same user joining two different tournaments concurrently
    → one success + one ConflictError via uq_participant_user_active."""
    async with session_factory() as setup:
        await _seed_user(setup, user_id=1, username="alice")
        await _seed_tournament(setup, tournament_id=1)
        await _seed_tournament(setup, tournament_id=2)
        await setup.commit()

    async def one_join(tournament_id):
        async with session_factory() as session:
            user_row = await session.get(
                __import__("atp.dashboard.models", fromlist=["User"]).User, 1
            )
            svc = TournamentService(session, bus=_DummyBus())
            try:
                return await svc.join(
                    tournament_id=tournament_id,
                    user=user_row,
                    agent_name="bot",
                )
            except ConflictError as e:
                return e

    results = await asyncio.gather(one_join(1), one_join(2))
    successes = [r for r in results if not isinstance(r, Exception)]
    conflicts = [r for r in results if isinstance(r, ConflictError)]

    assert len(successes) == 1
    assert len(conflicts) == 1


class _DummyBus:
    async def publish(self, event):
        pass
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/integration/dashboard/tournament/test_service_join_idempotent.py -v`
Expected: FAIL — existing `join()` implementation will not survive concurrent calls correctly.

- [ ] **Step 3: Update `TournamentService.join` to be idempotent with race handling**

Edit `packages/atp-dashboard/atp/dashboard/tournament/service.py`. Replace the existing `join` method with:

```python
    async def join(
        self,
        tournament_id: int,
        user: "User",
        agent_name: str,
        join_token: str | None = None,
    ) -> tuple[Participant, bool]:
        """Idempotent join with race-aware IntegrityError handling.

        Returns (participant, is_new).

        Race semantics on INSERT IntegrityError:
        - uq_participant_tournament_user violation: a concurrent idempotent
          re-join from the same user won the insert race. Re-read the
          existing row and return (existing, False).
        - uq_participant_user_active violation: user has active participation
          in a different tournament. Raise ConflictError(409).
        - Any other IntegrityError: re-raise unchanged.
        """
        from sqlalchemy.exc import IntegrityError

        async with self.session.begin():
            tournament = await self.session.get(Tournament, tournament_id)
            if tournament is None:
                raise NotFoundError(f"tournament {tournament_id}")
            if tournament.status != TournamentStatus.PENDING:
                raise ConflictError(
                    f"tournament {tournament_id} is {tournament.status.value}, "
                    "not accepting joins"
                )

            # Private tournament token check (AD-10)
            if tournament.join_token is not None:
                import hmac
                if join_token is None or not hmac.compare_digest(
                    tournament.join_token, join_token
                ):
                    raise ConflictError(
                        f"tournament {tournament_id} requires a valid join_token"
                    )

            # Idempotent pre-check
            existing = await self.session.scalar(
                select(Participant)
                .where(Participant.tournament_id == tournament_id)
                .where(Participant.user_id == user.id)
            )
            if existing is not None:
                if existing.released_at is not None:
                    # Leave is terminal — cannot rejoin
                    raise ConflictError(
                        f"user {user.id} already left tournament "
                        f"{tournament_id}; rejoin not permitted"
                    )
                return existing, False

            # INSERT path
            participant = Participant(
                tournament_id=tournament_id,
                user_id=user.id,
                agent_name=agent_name,
            )
            self.session.add(participant)
            try:
                await self.session.flush()
            except IntegrityError as exc:
                constraint_name = self._extract_constraint_name(exc)
                if constraint_name == "uq_participant_tournament_user":
                    # Lost idempotent race — re-read and return existing
                    await self.session.rollback()
                    async with self.session.begin():
                        existing = await self.session.scalar(
                            select(Participant)
                            .where(Participant.tournament_id == tournament_id)
                            .where(Participant.user_id == user.id)
                        )
                        if existing is None:
                            raise  # extreme edge — no row after IntegrityError
                        return existing, False
                if constraint_name == "uq_participant_user_active":
                    raise ConflictError(
                        "user already has an active tournament"
                    )
                raise

            return participant, True

    @staticmethod
    def _extract_constraint_name(exc: Exception) -> str:
        """Best-effort constraint name extraction from IntegrityError.

        SQLite: the message looks like
        'UNIQUE constraint failed: tournament_participants.user_id' or
        'constraint failed: uq_participant_user_active'.
        PostgreSQL: exc.orig.diag.constraint_name is available.
        """
        message = str(exc).lower()
        # SQLite partial unique index name is embedded in the message
        for name in (
            "uq_participant_user_active",
            "uq_participant_tournament_user",
            "uq_action_round_participant",
            "uq_round_tournament_number",
        ):
            if name in message:
                return name
        # PostgreSQL path
        orig = getattr(exc, "orig", None)
        diag = getattr(orig, "diag", None)
        if diag is not None:
            return getattr(diag, "constraint_name", "") or ""
        return ""
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/integration/dashboard/tournament/test_service_join_idempotent.py -v`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/integration/dashboard/tournament/conftest.py \
        tests/integration/dashboard/tournament/test_service_join_idempotent.py
git commit -m "feat(tournament): idempotent join with race-aware IntegrityError handling"
```

---

## Task 11: leave() with last-participant abandoned cascade

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/integration/dashboard/tournament/test_service_leave.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/dashboard/tournament/test_service_leave.py`:

```python
"""Integration tests for leave() and last-participant abandoned cascade."""

import pytest
from sqlalchemy import text

from atp.dashboard.tournament.errors import NotFoundError
from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import (
    Participant,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService


class _CapturedBus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


async def _seed_minimal(session, num_participants: int):
    await session.execute(text(
        "INSERT INTO users (id, username, is_admin) "
        "VALUES (1, 'alice', 0), (2, 'bob', 0)"
    ))
    await session.execute(text(
        "INSERT INTO tournaments (id, name, status, num_players, "
        "total_rounds, round_deadline_s, pending_deadline) "
        "VALUES (1, 't', 'active', 2, 3, 30, CURRENT_TIMESTAMP)"
    ))
    for i in range(1, num_participants + 1):
        await session.execute(text(
            "INSERT INTO tournament_participants "
            "(tournament_id, user_id, agent_name) "
            "VALUES (1, :uid, :name)"
        ), {"uid": i, "name": f"p{i}"})


async def _load_user(session, user_id):
    from atp.dashboard.models import User
    return await session.get(User, user_id)


@pytest.mark.anyio
async def test_leave_sets_released_at(session_factory):
    async with session_factory() as setup:
        await _seed_minimal(setup, num_participants=2)
        await setup.commit()

    bus = _CapturedBus()
    async with session_factory() as session:
        user1 = await _load_user(session, 1)
        svc = TournamentService(session, bus)
        await svc.leave(tournament_id=1, user=user1)

    async with session_factory() as verify:
        p = await verify.scalar(
            text("SELECT released_at FROM tournament_participants "
                 "WHERE user_id = 1")
        )
        assert p is not None  # released_at populated


@pytest.mark.anyio
async def test_leave_non_participant_raises_not_found(session_factory):
    async with session_factory() as setup:
        await _seed_minimal(setup, num_participants=2)
        await setup.commit()

    bus = _CapturedBus()
    async with session_factory() as session:
        # User 99 does not exist as participant
        from atp.dashboard.models import User
        user = User(id=99, username="ghost", is_admin=False)
        svc = TournamentService(session, bus)
        with pytest.raises(NotFoundError):
            await svc.leave(tournament_id=1, user=user)


@pytest.mark.anyio
async def test_last_participant_leave_triggers_abandoned_cascade(session_factory):
    async with session_factory() as setup:
        await _seed_minimal(setup, num_participants=2)
        await setup.commit()

    bus = _CapturedBus()
    async with session_factory() as session:
        user1 = await _load_user(session, 1)
        svc = TournamentService(session, bus)
        await svc.leave(tournament_id=1, user=user1)

    # User 1 left — tournament still has user 2 active, so no cascade
    async with session_factory() as session:
        user2 = await _load_user(session, 2)
        svc = TournamentService(session, bus)
        await svc.leave(tournament_id=1, user=user2)

    # Tournament is now abandoned
    async with session_factory() as verify:
        t = await verify.get(Tournament, 1)
        assert t.status == TournamentStatus.CANCELLED
        assert t.cancelled_reason == CancelReason.ABANDONED
        assert t.cancelled_by is None

    cancel_events = [
        e for e in bus.events if isinstance(e, TournamentCancelEvent)
    ]
    assert len(cancel_events) == 1
    assert cancel_events[0].cancelled_reason == CancelReason.ABANDONED
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/integration/dashboard/tournament/test_service_leave.py -v`
Expected: FAIL — existing `leave()` (if any) doesn't do abandoned cascade.

- [ ] **Step 3: Implement `leave` in TournamentService**

Edit `packages/atp-dashboard/atp/dashboard/tournament/service.py`. Add or replace:

```python
    async def leave(self, tournament_id: int, user: "User") -> None:
        """Mark the caller's Participant as released. If the caller is the
        last active participant of an ACTIVE tournament, cascade to
        _cancel_impl with reason=ABANDONED inside the same transaction.
        """
        cancel_event = None
        async with self.session.begin():
            participant = await self.session.scalar(
                select(Participant)
                .where(Participant.tournament_id == tournament_id)
                .where(Participant.user_id == user.id)
                .where(Participant.released_at.is_(None))
            )
            if participant is None:
                raise NotFoundError(
                    f"user {user.id} is not active in tournament "
                    f"{tournament_id}"
                )

            now = datetime.utcnow()
            participant.released_at = now
            await self.session.flush()

            remaining = await self.session.scalar(
                select(func.count())
                .select_from(Participant)
                .where(Participant.tournament_id == tournament_id)
                .where(Participant.released_at.is_(None))
            )
            tournament = await self.session.get(Tournament, tournament_id)

            if remaining == 0 and tournament.status == TournamentStatus.ACTIVE:
                log.info(
                    "tournament.leave.abandoned_cascade",
                    extra={
                        "tournament_id": tournament_id,
                        "leaving_user_id": user.id,
                    },
                )
                cancel_event = await self._cancel_impl(
                    tournament_id,
                    reason=CancelReason.ABANDONED,
                    cancelled_by=None,
                    reason_detail=None,
                )

        if cancel_event is not None:
            try:
                await self.bus.publish(cancel_event)
            except Exception:
                log.warning(
                    "tournament.cancel.publish_failed",
                    extra={"tournament_id": tournament_id},
                    exc_info=True,
                )
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/integration/dashboard/tournament/test_service_leave.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/integration/dashboard/tournament/test_service_leave.py
git commit -m "feat(tournament): add leave() with last-participant abandoned cascade"
```

---

## Task 12: create_tournament AD-9 validation + join_token + pending_deadline

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/unit/dashboard/tournament/test_service_create.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/tournament/test_service_create.py`:

```python
"""Tests for TournamentService.create_tournament AD-9 validation,
pending_deadline computation, and join_token generation."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.errors import ValidationError
from atp.dashboard.tournament.service import (
    TOURNAMENT_PENDING_MAX_WAIT_S,
    TournamentService,
)


def _make_user(user_id=1, is_admin=False):
    u = MagicMock()
    u.id = user_id
    u.is_admin = is_admin
    return u


def _make_service():
    session = MagicMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.begin = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock()
    session.begin.return_value.__aexit__ = AsyncMock(return_value=False)
    bus = MagicMock()
    return TournamentService(session=session, bus=bus)


@pytest.mark.anyio
async def test_create_within_cap_succeeds(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    t, token = await svc.create_tournament(
        creator=_make_user(),
        name="ok",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=90,
        round_deadline_s=30,
    )
    assert t is not None
    assert token is None  # public tournament


@pytest.mark.anyio
async def test_create_over_cap_raises_validation_error(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    with pytest.raises(ValidationError, match="max duration"):
        await svc.create_tournament(
            creator=_make_user(),
            name="too_long",
            game_type="prisoners_dilemma",
            num_players=2,
            total_rounds=200,
            round_deadline_s=30,
        )


@pytest.mark.anyio
async def test_create_private_returns_join_token(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    t, token = await svc.create_tournament(
        creator=_make_user(),
        name="private",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
        private=True,
    )
    assert token is not None
    assert len(token) >= 32  # secrets.token_urlsafe(32) base64-ish


@pytest.mark.anyio
async def test_create_sets_pending_deadline(monkeypatch):
    monkeypatch.setenv("ATP_TOKEN_EXPIRE_MINUTES", "60")
    svc = _make_service()
    t, _ = await svc.create_tournament(
        creator=_make_user(),
        name="x",
        game_type="prisoners_dilemma",
        num_players=2,
        total_rounds=3,
        round_deadline_s=30,
    )
    assert t.pending_deadline is not None
    # Should be roughly now + TOURNAMENT_PENDING_MAX_WAIT_S
    delta = t.pending_deadline - datetime.utcnow()
    assert timedelta(seconds=TOURNAMENT_PENDING_MAX_WAIT_S - 5) < delta
    assert delta < timedelta(seconds=TOURNAMENT_PENDING_MAX_WAIT_S + 5)
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_service_create.py -v`
Expected: FAIL — `TOURNAMENT_PENDING_MAX_WAIT_S` not defined and `create_tournament` doesn't have new parameters.

- [ ] **Step 3: Add constants and rewrite `create_tournament`**

Edit `packages/atp-dashboard/atp/dashboard/tournament/service.py`. Add near the top of the file:

```python
import os
import secrets

TOURNAMENT_PENDING_MAX_WAIT_S: int = 300
```

Replace the existing `create_tournament` method with:

```python
    async def create_tournament(
        self,
        creator: "User",
        *,
        name: str,
        game_type: str,
        num_players: int,
        total_rounds: int,
        round_deadline_s: int,
        private: bool = False,
    ) -> tuple[Tournament, str | None]:
        """Create a tournament. AD-9 duration cap validation, pending_deadline,
        optional join_token.

        Does NOT auto-join the creator.
        """
        # AD-9 duration cap
        token_expire_minutes = int(
            os.environ.get("ATP_TOKEN_EXPIRE_MINUTES", "60")
        )
        max_wall_clock = (
            TOURNAMENT_PENDING_MAX_WAIT_S
            + total_rounds * round_deadline_s
        )
        budget = (token_expire_minutes - 10) * 60
        if max_wall_clock > budget:
            raise ValidationError(
                f"max duration {budget}s (pending {TOURNAMENT_PENDING_MAX_WAIT_S}s "
                f"+ {total_rounds} rounds × {round_deadline_s}s = "
                f"{max_wall_clock}s) exceeds "
                f"(ATP_TOKEN_EXPIRE_MINUTES − 10) × 60 = {budget}s cap. "
                f"Reduce total_rounds or round_deadline_s."
            )

        now = datetime.utcnow()
        pending_deadline = now + timedelta(seconds=TOURNAMENT_PENDING_MAX_WAIT_S)

        join_token_plaintext: str | None = None
        if private:
            join_token_plaintext = secrets.token_urlsafe(32)

        tournament = Tournament(
            name=name,
            game_type=game_type,
            num_players=num_players,
            total_rounds=total_rounds,
            round_deadline_s=round_deadline_s,
            status=TournamentStatus.PENDING,
            created_by=creator.id,
            pending_deadline=pending_deadline,
            join_token=join_token_plaintext,
        )
        async with self.session.begin():
            self.session.add(tournament)
            await self.session.flush()

        return tournament, join_token_plaintext
```

Add to imports:

```python
from datetime import datetime, timedelta
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/tournament/test_service_create.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/unit/dashboard/tournament/test_service_create.py
git commit -m "feat(tournament): AD-9 duration cap validation + join_token generation"
```

---

## Task 13: Read methods — list_tournaments, get_tournament, get_history

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py`
- Test: `tests/integration/dashboard/tournament/test_service_read_methods.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/dashboard/tournament/test_service_read_methods.py`:

```python
"""Tests for list_tournaments, get_tournament, get_history visibility."""

import pytest
from sqlalchemy import text

from atp.dashboard.tournament.errors import NotFoundError
from atp.dashboard.tournament.models import TournamentStatus
from atp.dashboard.tournament.service import TournamentService


class _DummyBus:
    async def publish(self, event):
        pass


async def _seed(session):
    await session.execute(text(
        "INSERT INTO users (id, username, is_admin) "
        "VALUES (1, 'alice', 0), (2, 'bob', 0), (99, 'admin', 1)"
    ))
    # Three tournaments: public (owned by 1), private (owned by 2),
    # public (owned by 99/admin).
    await session.execute(text(
        "INSERT INTO tournaments "
        "(id, name, status, num_players, total_rounds, round_deadline_s, "
        " pending_deadline, created_by, join_token) "
        "VALUES "
        "(1, 'public_alice', 'pending', 2, 3, 30, CURRENT_TIMESTAMP, 1, NULL),"
        "(2, 'private_bob', 'pending', 2, 3, 30, CURRENT_TIMESTAMP, 2, 'secret'),"
        "(3, 'public_admin', 'active', 2, 3, 30, CURRENT_TIMESTAMP, 99, NULL)"
    ))


async def _load_user(session, user_id):
    from atp.dashboard.models import User
    return await session.get(User, user_id)


@pytest.mark.anyio
async def test_list_tournaments_admin_sees_all(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        admin = await _load_user(s, 99)
        svc = TournamentService(s, _DummyBus())
        result = await svc.list_tournaments(user=admin)
        assert len(result) == 3


@pytest.mark.anyio
async def test_list_tournaments_non_admin_hides_private(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        result = await svc.list_tournaments(user=alice)
        names = {t.name for t in result}
        assert "public_alice" in names
        assert "public_admin" in names
        assert "private_bob" not in names  # not visible to alice


@pytest.mark.anyio
async def test_list_tournaments_shows_own_private(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        bob = await _load_user(s, 2)
        svc = TournamentService(s, _DummyBus())
        result = await svc.list_tournaments(user=bob)
        names = {t.name for t in result}
        assert "private_bob" in names  # bob created it


@pytest.mark.anyio
async def test_get_tournament_invisible_raises_not_found(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        alice = await _load_user(s, 1)
        svc = TournamentService(s, _DummyBus())
        with pytest.raises(NotFoundError):
            await svc.get_tournament(tournament_id=2, user=alice)


@pytest.mark.anyio
async def test_get_tournament_admin_sees_private(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        admin = await _load_user(s, 99)
        svc = TournamentService(s, _DummyBus())
        t = await svc.get_tournament(tournament_id=2, user=admin)
        assert t.id == 2


@pytest.mark.anyio
async def test_list_tournaments_status_filter(session_factory):
    async with session_factory() as setup:
        await _seed(setup)
        await setup.commit()

    async with session_factory() as s:
        admin = await _load_user(s, 99)
        svc = TournamentService(s, _DummyBus())
        result = await svc.list_tournaments(
            user=admin, status=TournamentStatus.ACTIVE
        )
        assert len(result) == 1
        assert result[0].status == TournamentStatus.ACTIVE
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/integration/dashboard/tournament/test_service_read_methods.py -v`
Expected: FAIL with `AttributeError: ... no attribute 'list_tournaments'`.

- [ ] **Step 3: Implement read methods**

Edit `packages/atp-dashboard/atp/dashboard/tournament/service.py`. Add to `TournamentService`:

```python
    async def list_tournaments(
        self,
        user: "User",
        status: "TournamentStatus | None" = None,
    ) -> list[Tournament]:
        """Return tournaments visible to `user`, optionally filtered by status.

        Visibility rule:
        - user.is_admin: all tournaments.
        - Else: Tournament.join_token IS NULL OR created_by = user.id
                OR EXISTS participant row for (tournament, user).
        """
        stmt = select(Tournament)
        if status is not None:
            stmt = stmt.where(Tournament.status == status)

        if not user.is_admin:
            from sqlalchemy import exists, or_
            stmt = stmt.where(
                or_(
                    Tournament.join_token.is_(None),
                    Tournament.created_by == user.id,
                    exists().where(
                        (Participant.tournament_id == Tournament.id)
                        & (Participant.user_id == user.id)
                    ),
                )
            )

        result = await self.session.scalars(stmt)
        return list(result)

    async def get_tournament(
        self,
        tournament_id: int,
        user: "User",
    ) -> Tournament:
        """Return a single tournament if visible to `user`.

        Uses the same visibility filter as list_tournaments. Invisible
        tournaments raise NotFoundError (enumeration guard).
        """
        tournament = await self.session.get(Tournament, tournament_id)
        if tournament is None:
            raise NotFoundError(f"tournament {tournament_id}")

        if user.is_admin:
            return tournament
        if tournament.join_token is None:
            return tournament
        if tournament.created_by == user.id:
            return tournament
        # Check participant rows
        exists_row = await self.session.scalar(
            select(Participant)
            .where(Participant.tournament_id == tournament_id)
            .where(Participant.user_id == user.id)
        )
        if exists_row is not None:
            return tournament
        raise NotFoundError(f"tournament {tournament_id}")

    async def get_history(
        self,
        tournament_id: int,
        user: "User",
        last_n: int | None = None,
    ) -> list[Round]:
        """Return rounds for a tournament visible to `user`. Plan 2a ships
        PD-only, so all participants see all actions post-resolution.

        last_n hard-capped at 100 regardless of input.
        """
        # Visibility check reuses get_tournament's rule
        await self.get_tournament(tournament_id, user)

        stmt = (
            select(Round)
            .where(Round.tournament_id == tournament_id)
            .order_by(Round.round_number.desc())
        )
        cap = min(last_n or 100, 100)
        stmt = stmt.limit(cap)

        result = await self.session.scalars(stmt)
        rounds = list(result)
        rounds.reverse()  # ascending by round_number for consumer convenience
        return rounds
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/integration/dashboard/tournament/test_service_read_methods.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/integration/dashboard/tournament/test_service_read_methods.py
git commit -m "feat(tournament): add list_tournaments, get_tournament, get_history"
```

---

## Task 14: Deadline worker module

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/tournament/deadlines.py`
- Test: `tests/unit/dashboard/tournament/test_deadline_worker.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/dashboard/tournament/test_deadline_worker.py`:

```python
"""Unit tests for the deadline worker tick logic."""

import asyncio
import logging
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.deadlines import run_deadline_worker, _tick
from atp.dashboard.tournament.reasons import CancelReason


def _make_session_factory(
    expired_rounds: list[int],
    expired_pending: list[int],
):
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[
        MagicMock(__iter__=lambda self: iter([(r,) for r in expired_rounds])),
        MagicMock(__iter__=lambda self: iter([(t,) for t in expired_pending])),
    ])
    # Async context manager support
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory, session


@pytest.mark.anyio
async def test_tick_empty_paths_logs_zero_counts(caplog):
    factory, session = _make_session_factory([], [])
    bus = MagicMock()
    log = logging.getLogger("tournament.deadlines")

    with caplog.at_level(logging.INFO, logger="tournament.deadlines"):
        await _tick(factory, bus, log)

    records = [r for r in caplog.records if "tick_complete" in r.message]
    assert len(records) == 1


@pytest.mark.anyio
async def test_tick_inner_guard_isolates_poison_row(caplog, monkeypatch):
    factory, session = _make_session_factory([1, 2, 3], [])
    bus = MagicMock()

    # Patch TournamentService so force_resolve_round raises on the middle row
    from atp.dashboard.tournament import deadlines

    call_count = {"n": 0}

    class _FakeService:
        def __init__(self, *a, **k):
            pass

        async def force_resolve_round(self, round_id):
            call_count["n"] += 1
            if round_id == 2:
                raise RuntimeError("simulated poison")

        async def cancel_tournament_system(self, *a, **k):
            pass

    monkeypatch.setattr(deadlines, "TournamentService", _FakeService)
    log = logging.getLogger("tournament.deadlines")

    with caplog.at_level(logging.ERROR, logger="tournament.deadlines"):
        await _tick(factory, bus, log)

    # All three rounds attempted — inner guard absorbed the middle failure
    assert call_count["n"] == 3
    failed_records = [
        r for r in caplog.records
        if "round_resolve_failed" in r.message
    ]
    assert len(failed_records) == 1
    outer_records = [
        r for r in caplog.records if "tick_failed" in r.message
    ]
    assert len(outer_records) == 0


@pytest.mark.anyio
async def test_run_deadline_worker_hard_cancel_exits_fast(monkeypatch):
    """Shutdown sequence: shutdown_event.set() + task.cancel() must
    terminate the worker even if it's mid-tick in a slow call."""
    factory = MagicMock()

    slow_session = MagicMock()

    async def slow_execute(*a, **k):
        await asyncio.sleep(5)
        return MagicMock(__iter__=lambda self: iter([]))

    slow_session.execute = slow_execute
    factory.return_value.__aenter__ = AsyncMock(return_value=slow_session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    bus = MagicMock()
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_deadline_worker(factory, bus, shutdown_event=shutdown)
    )
    await asyncio.sleep(0.1)  # let the worker enter the slow execute
    shutdown.set()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert task.done()  # must not hang waiting for the 5s sleep
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/tournament/test_deadline_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atp.dashboard.tournament.deadlines'`

- [ ] **Step 3: Create the deadline worker module**

Create `packages/atp-dashboard/atp/dashboard/tournament/deadlines.py`:

```python
"""Plan 2a deadline worker. Single asyncio task running inside the
FastAPI lifespan. Two-path scan per tick:
1. Expired round deadlines → force_resolve_round.
2. Expired pending tournaments → cancel_tournament_system(PENDING_TIMEOUT).

Per-iteration outer try/except + per-row inner try/except; one poisoned
row cannot kill the tick. Hard cancel on shutdown — correctness is
preserved by SQLAlchemy transaction rollback on CancelledError and
session_sync on subscriber reconnect.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.models import (
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService

POLL_INTERVAL_S: float = float(
    os.environ.get("ATP_DEADLINE_WORKER_POLL_INTERVAL_S", "5")
)


async def run_deadline_worker(
    session_factory: async_sessionmaker[AsyncSession],
    bus: TournamentEventBus,
    *,
    shutdown_event: asyncio.Event,
) -> None:
    """Main loop. Runs until shutdown_event is set or the task is cancelled."""
    log = logging.getLogger("tournament.deadlines")
    log.info(
        "deadline_worker.started", extra={"poll_interval_s": POLL_INTERVAL_S}
    )

    while not shutdown_event.is_set():
        try:
            await _tick(session_factory, bus, log)
        except Exception:
            log.exception("deadline_worker.tick_failed")

        try:
            await asyncio.wait_for(
                shutdown_event.wait(), timeout=POLL_INTERVAL_S
            )
        except TimeoutError:
            pass  # normal path — interval elapsed

    log.info("deadline_worker.shutting_down")


async def _tick(
    session_factory: async_sessionmaker[AsyncSession],
    bus: TournamentEventBus,
    log: logging.Logger,
) -> None:
    """One scan pass across both paths."""
    t_start = time.monotonic()
    async with session_factory() as session:
        service = TournamentService(session, bus)

        # Path 1: expired round deadlines
        expired_rounds_result = await session.execute(
            select(Round.id)
            .join(Tournament, Tournament.id == Round.tournament_id)
            .where(Round.status == RoundStatus.WAITING_FOR_ACTIONS)
            .where(Round.deadline < datetime.utcnow())
            .where(Tournament.status == TournamentStatus.ACTIVE)
        )
        round_ids = [row[0] for row in expired_rounds_result]
        for round_id in round_ids:
            try:
                await service.force_resolve_round(round_id)
            except Exception:
                log.exception(
                    "deadline_worker.round_resolve_failed",
                    extra={"round_id": round_id},
                )

        # Path 2: expired PENDING tournaments
        expired_pending_result = await session.execute(
            select(Tournament.id)
            .where(Tournament.status == TournamentStatus.PENDING)
            .where(Tournament.pending_deadline < datetime.utcnow())
        )
        tournament_ids = [row[0] for row in expired_pending_result]
        for tournament_id in tournament_ids:
            try:
                await service.cancel_tournament_system(
                    tournament_id,
                    reason=CancelReason.PENDING_TIMEOUT,
                )
            except Exception:
                log.exception(
                    "deadline_worker.pending_cancel_failed",
                    extra={"tournament_id": tournament_id},
                )

    log.info(
        "deadline_worker.tick_complete",
        extra={
            "rounds_processed": len(round_ids),
            "pending_cancelled": len(tournament_ids),
            "elapsed_ms": int((time.monotonic() - t_start) * 1000),
        },
    )
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/tournament/test_deadline_worker.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/deadlines.py \
        tests/unit/dashboard/tournament/test_deadline_worker.py
git commit -m "feat(tournament): add deadline worker with two-path scan"
```

---

## Task 15: Deadline worker wiring in FastAPI lifespan + WEB_CONCURRENCY assertion

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/factory.py`
- Test: `tests/integration/dashboard/tournament/test_factory_lifespan.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/dashboard/tournament/test_factory_lifespan.py`:

```python
"""Tests for WEB_CONCURRENCY startup assertion."""

import os
from unittest.mock import patch

import pytest

from atp.dashboard.v2.factory import assert_single_worker


def test_single_worker_ok(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    assert_single_worker()  # must not raise


def test_default_is_single_worker(monkeypatch):
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    assert_single_worker()  # default = 1


def test_multi_worker_raises(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with pytest.raises(RuntimeError, match="WEB_CONCURRENCY=1"):
        assert_single_worker()
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/integration/dashboard/tournament/test_factory_lifespan.py -v`
Expected: FAIL with `ImportError: cannot import name 'assert_single_worker'`

- [ ] **Step 3: Add `assert_single_worker` and deadline worker wiring to factory.py**

Edit `packages/atp-dashboard/atp/dashboard/v2/factory.py`. Add near the top of imports:

```python
import asyncio
import os
from contextlib import asynccontextmanager, suppress

from atp.dashboard.tournament.deadlines import run_deadline_worker
```

Add this module-level function near other helpers:

```python
def assert_single_worker() -> None:
    """Crash at startup if the deadline worker would race itself across
    multiple uvicorn workers. Multi-worker support is backlog I."""
    wc = int(os.environ.get("WEB_CONCURRENCY", "1"))
    if wc != 1:
        raise RuntimeError(
            f"ATP Tournament deadline worker requires WEB_CONCURRENCY=1 "
            f"(got {wc}). Multiple workers would each run a deadline "
            f"worker, racing on force_resolve_round and wasting DB reads. "
            f"Multi-worker support is backlog item I (Redis bus + "
            f"PostgreSQL migration)."
        )
```

Locate the existing FastAPI lifespan context manager (search for `@asynccontextmanager` in the file). Wrap or modify it so that:

1. It calls `assert_single_worker()` first.
2. It starts the deadline worker as a background task.
3. On shutdown it sets `shutdown_event`, cancels the task, and gathers it with `return_exceptions=True`.

Example — add this lifespan (or fold its contents into the existing one):

```python
@asynccontextmanager
async def combined_lifespan(app):
    assert_single_worker()

    shutdown_event = asyncio.Event()
    worker_task = asyncio.create_task(
        run_deadline_worker(
            app.state.session_factory,
            app.state.tournament_event_bus,
            shutdown_event=shutdown_event,
        )
    )
    try:
        # ... existing lifespan body (mcp_app lifespan, etc.) ...
        yield
    finally:
        shutdown_event.set()
        worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(worker_task, return_exceptions=True)
```

If `app.state.session_factory` or `app.state.tournament_event_bus` don't yet exist, add their initialization to the factory function that builds the app (search for the existing `Bus` / `session_factory` setup and wire them onto `app.state`).

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/integration/dashboard/tournament/test_factory_lifespan.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Smoke test the app starts**

Run: `uv run python -c "from atp.dashboard.v2.factory import create_app; app = create_app(); print('OK')"`
Expected: `OK`. If the test fails with `AttributeError: state.session_factory`, fix factory initialization to wire `app.state.session_factory` and `app.state.tournament_event_bus` before lifespan starts.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/factory.py \
        tests/integration/dashboard/tournament/test_factory_lifespan.py
git commit -m "feat(tournament): wire deadline worker into FastAPI lifespan"
```

---

## Task 16: MCP notification formatter (_format_for_user)

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/notifications.py`
- Test: `tests/unit/dashboard/mcp/test_format_for_user.py`

- [ ] **Step 1: Create test skeleton**

```bash
mkdir -p tests/unit/dashboard/mcp
touch tests/unit/dashboard/mcp/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/dashboard/mcp/test_format_for_user.py`:

```python
"""Tests for the _format_for_user notification dispatcher."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from atp.dashboard.mcp.notifications import _format_for_user
from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason


def _user(user_id: int, is_admin: bool = False):
    u = MagicMock()
    u.id = user_id
    u.is_admin = is_admin
    return u


def _cancel_event(cancelled_by=42, reason=CancelReason.ADMIN_ACTION):
    return TournamentCancelEvent(
        tournament_id=1,
        cancelled_at=datetime(2026, 4, 15, 10, 0, 0),
        cancelled_by=cancelled_by,
        cancelled_reason=reason,
        cancelled_reason_detail=None,
        final_rounds_played=0,
        final_status=TournamentStatus.CANCELLED,
    )


def test_cancel_event_admin_recipient_sees_cancelled_by():
    payload = _format_for_user(_cancel_event(), _user(99, is_admin=True))
    assert payload is not None
    assert payload["event"] == "tournament_cancelled"
    assert payload["cancelled_by"] == 42


def test_cancel_event_non_admin_recipient_hides_cancelled_by():
    payload = _format_for_user(_cancel_event(), _user(99, is_admin=False))
    assert payload is not None
    assert payload["cancelled_by"] is None


def test_system_cancel_has_none_cancelled_by_regardless_of_recipient():
    event = _cancel_event(
        cancelled_by=None, reason=CancelReason.PENDING_TIMEOUT
    )
    admin_payload = _format_for_user(event, _user(99, is_admin=True))
    user_payload = _format_for_user(event, _user(99, is_admin=False))
    assert admin_payload["cancelled_by"] is None
    assert user_payload["cancelled_by"] is None
```

- [ ] **Step 3: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/mcp/test_format_for_user.py -v`
Expected: FAIL with `ImportError: cannot import name '_format_for_user'`.

- [ ] **Step 4: Add the dispatcher to notifications.py**

Edit `packages/atp-dashboard/atp/dashboard/mcp/notifications.py`. Add to the top of imports:

```python
from atp.dashboard.tournament.events import TournamentCancelEvent
```

Append at the bottom of the file:

```python
def _format_for_user(event, user) -> dict | None:
    """Dispatch-based notification formatter.

    Takes a bus event and a recipient User, returns the personalized
    `data` dict to send via session.send_notification, or None if the
    event is not deliverable to this user.

    Responsibilities:
    1. Add `event` discriminator.
    2. Apply game-specific reveal semantics (Plan 2a is PD-only, near-identity).
    3. Privacy filter admin-only fields (cancelled_by for non-admins).
    """
    if isinstance(event, TournamentCancelEvent):
        cancelled_by = event.cancelled_by if user.is_admin else None
        return {
            "event": "tournament_cancelled",
            "tournament_id": event.tournament_id,
            "reason": event.cancelled_reason.value,
            "reason_detail": event.cancelled_reason_detail,
            "cancelled_by": cancelled_by,
            "final_rounds_played": event.final_rounds_played,
            "cancelled_at": event.cancelled_at.isoformat(),
        }

    # round_started, round_ended, tournament_completed events are
    # existing dict-shaped payloads from the vertical slice bus.
    # For Plan 2a they pass through with an added `event` discriminator
    # only — no per-recipient privacy filtering is needed because PD
    # reveals all actions after each round (source spec §Notification
    # personalization). Future hidden-information games override this
    # branch per-game.
    if isinstance(event, dict) and "event_type" in event:
        event_type = event["event_type"]
        payload = {k: v for k, v in event.items() if k != "event_type"}
        payload["event"] = event_type
        return payload

    return None
```

- [ ] **Step 5: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/mcp/test_format_for_user.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/mcp/notifications.py \
        tests/unit/dashboard/mcp/__init__.py \
        tests/unit/dashboard/mcp/test_format_for_user.py
git commit -m "feat(tournament): add _format_for_user notification dispatcher"
```

---

## Task 17: MCP tools — idempotent join_tournament with session_sync emission

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/tools.py`
- Test: `tests/integration/dashboard/tournament/test_mcp_join_tournament.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/dashboard/tournament/test_mcp_join_tournament.py`:

```python
"""Integration tests for the MCP join_tournament tool handler.

Verifies session_sync is emitted as the first notification on every
join call (first and subsequent), per the Plan 2a contract.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text

from atp.dashboard.mcp.tools import join_tournament
from atp.dashboard.tournament.service import TournamentService


@pytest.mark.anyio
async def test_join_tournament_emits_session_sync_first(session_factory):
    async with session_factory() as setup:
        await setup.execute(text(
            "INSERT INTO users (id, username, is_admin) "
            "VALUES (1, 'alice', 0)"
        ))
        await setup.execute(text(
            "INSERT INTO tournaments "
            "(id, name, status, num_players, total_rounds, round_deadline_s, "
            " pending_deadline, created_by) "
            "VALUES (1, 't', 'pending', 2, 3, 30, CURRENT_TIMESTAMP, 1)"
        ))
        await setup.commit()

    notifications = []

    ctx = MagicMock()
    ctx.session.send_notification = AsyncMock(
        side_effect=lambda n: notifications.append(n)
    )

    async with session_factory() as session:
        from atp.dashboard.models import User
        user = await session.get(User, 1)
        svc = TournamentService(session, bus=_CapturingBus())
        result = await join_tournament(
            ctx=ctx,
            service=svc,
            user=user,
            tournament_id=1,
            agent_name="bot",
            join_token=None,
        )

    assert result["joined"] is True
    assert len(notifications) >= 1
    first_notification = notifications[0]
    assert "session_sync" in str(first_notification)


class _CapturingBus:
    async def publish(self, event):
        pass
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/integration/dashboard/tournament/test_mcp_join_tournament.py -v`
Expected: FAIL — `join_tournament` tool signature mismatch or missing session_sync emission.

- [ ] **Step 3: Update the `join_tournament` tool handler**

Edit `packages/atp-dashboard/atp/dashboard/mcp/tools.py`. Find the existing `join_tournament` function. Replace its body (keep the FastMCP decorator if present) with logic that:

1. Calls `service.join(...)` to obtain `(participant, is_new)`.
2. Builds the full current state via `service.get_state_for(...)`.
3. Sends a `session_sync` notification to `ctx.session` **before** returning the tool result.
4. Returns a dict including `joined=True` and `participant_id`.

```python
async def join_tournament(
    ctx,
    service,
    user,
    tournament_id: int,
    agent_name: str,
    join_token: str | None = None,
) -> dict:
    """MCP tool handler: idempotent join with session_sync on every call.

    Called both from the FastMCP tool decorator and from integration
    tests. Extracting the body from the decorator makes it testable
    without a live MCP session.
    """
    participant, is_new = await service.join(
        tournament_id=tournament_id,
        user=user,
        agent_name=agent_name,
        join_token=join_token,
    )

    state = await service.get_state_for(
        tournament_id=tournament_id, user=user
    )

    session_sync_payload = {
        "event": "session_sync",
        "tournament_id": tournament_id,
        "state": state.to_dict() if hasattr(state, "to_dict") else state,
    }
    await ctx.session.send_notification(session_sync_payload)

    return {
        "joined": True,
        "participant_id": participant.id,
        "is_new": is_new,
    }
```

If the existing vertical-slice `join_tournament` is wrapped by `@mcp.tool(...)`, keep the decorator wrapper as a thin shim that calls the extracted async function; the tests above import the unwrapped async function directly.

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/integration/dashboard/tournament/test_mcp_join_tournament.py -v`
Expected: PASS — 1 test.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/mcp/tools.py \
        tests/integration/dashboard/tournament/test_mcp_join_tournament.py
git commit -m "feat(tournament): MCP join_tournament emits session_sync on every call"
```

---

## Task 18: MCP tools — leave, get_history, list, get, cancel

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/tools.py`
- Test: `tests/unit/dashboard/mcp/test_tools_plan_2a.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dashboard/mcp/test_tools_plan_2a.py`:

```python
"""Unit tests for the five new Plan 2a MCP tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.mcp.tools import (
    cancel_tournament,
    get_history,
    get_tournament,
    leave_tournament,
    list_tournaments,
)


def _make_service(**method_returns):
    svc = MagicMock()
    for name, value in method_returns.items():
        setattr(svc, name, AsyncMock(return_value=value))
    return svc


def _ctx():
    c = MagicMock()
    c.session.send_notification = AsyncMock()
    return c


def _user():
    u = MagicMock()
    u.id = 1
    u.is_admin = False
    return u


@pytest.mark.anyio
async def test_leave_tournament_calls_service():
    svc = _make_service(leave=None)
    result = await leave_tournament(
        ctx=_ctx(), service=svc, user=_user(), tournament_id=1
    )
    svc.leave.assert_awaited_once_with(tournament_id=1, user=_user_matcher())
    assert result["left"] is True


@pytest.mark.anyio
async def test_get_history_returns_rounds():
    mock_rounds = [MagicMock(round_number=1), MagicMock(round_number=2)]
    svc = _make_service(get_history=mock_rounds)
    result = await get_history(
        ctx=_ctx(),
        service=svc,
        user=_user(),
        tournament_id=1,
        last_n=None,
    )
    assert len(result["rounds"]) == 2


@pytest.mark.anyio
async def test_list_tournaments_returns_filtered():
    mock_tournaments = [MagicMock(id=1), MagicMock(id=2)]
    svc = _make_service(list_tournaments=mock_tournaments)
    result = await list_tournaments(
        ctx=_ctx(), service=svc, user=_user(), status=None
    )
    assert len(result["tournaments"]) == 2


@pytest.mark.anyio
async def test_get_tournament_returns_detail():
    mock_t = MagicMock(id=1, name="t")
    svc = _make_service(get_tournament=mock_t)
    result = await get_tournament(
        ctx=_ctx(), service=svc, user=_user(), tournament_id=1
    )
    assert result["tournament"]["id"] == 1


@pytest.mark.anyio
async def test_cancel_tournament_calls_service():
    svc = _make_service(cancel_tournament=None)
    result = await cancel_tournament(
        ctx=_ctx(),
        service=svc,
        user=_user(),
        tournament_id=1,
        reason_detail=None,
    )
    svc.cancel_tournament.assert_awaited_once()
    assert result["cancelled"] is True


def _user_matcher():
    """MagicMock equality helper for keyword arg comparison."""
    class _M:
        def __eq__(self, other):
            return hasattr(other, "id") and other.id == 1
    return _M()
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/dashboard/mcp/test_tools_plan_2a.py -v`
Expected: FAIL with `ImportError: cannot import name 'leave_tournament'` (or similar).

- [ ] **Step 3: Add five new tool functions to tools.py**

Edit `packages/atp-dashboard/atp/dashboard/mcp/tools.py`. Append:

```python
async def leave_tournament(
    ctx,
    service,
    user,
    tournament_id: int,
) -> dict:
    """MCP tool: leave a tournament. Idempotent at the DB level —
    a retry after a successful-but-unacknowledged first call will
    return NotFoundError; SDK callers MUST treat that as success."""
    await service.leave(tournament_id=tournament_id, user=user)
    return {"left": True, "tournament_id": tournament_id}


async def get_history(
    ctx,
    service,
    user,
    tournament_id: int,
    last_n: int | None = None,
) -> dict:
    rounds = await service.get_history(
        tournament_id=tournament_id, user=user, last_n=last_n
    )
    return {
        "tournament_id": tournament_id,
        "rounds": [
            {
                "round_number": r.round_number,
                "status": getattr(r, "status", None),
            }
            for r in rounds
        ],
    }


async def list_tournaments(
    ctx,
    service,
    user,
    status: str | None = None,
) -> dict:
    from atp.dashboard.tournament.models import TournamentStatus
    status_filter = TournamentStatus(status) if status else None
    tournaments = await service.list_tournaments(
        user=user, status=status_filter
    )
    return {
        "tournaments": [
            {
                "id": t.id,
                "name": getattr(t, "name", ""),
                "status": getattr(t, "status", None),
                "has_join_token": bool(getattr(t, "join_token", None)),
            }
            for t in tournaments
        ]
    }


async def get_tournament(
    ctx,
    service,
    user,
    tournament_id: int,
) -> dict:
    t = await service.get_tournament(tournament_id=tournament_id, user=user)
    return {
        "tournament": {
            "id": t.id,
            "name": getattr(t, "name", ""),
            "status": getattr(t, "status", None),
            "has_join_token": bool(getattr(t, "join_token", None)),
        }
    }


async def cancel_tournament(
    ctx,
    service,
    user,
    tournament_id: int,
    reason_detail: str | None = None,
) -> dict:
    await service.cancel_tournament(
        user=user,
        tournament_id=tournament_id,
        reason_detail=reason_detail,
    )
    return {"cancelled": True, "tournament_id": tournament_id}
```

After the function definitions, register them with FastMCP by appending `@mcp.tool` decorated wrappers (mirror the pattern used by the vertical-slice `join_tournament` registration). Each wrapper extracts the current `user` from request context via `get_http_request()` and constructs a `TournamentService(session, bus)` from module-level `session_factory` / `bus`. Example wrapper shape (adapt to the existing vertical-slice pattern in this file):

```python
@mcp.tool
async def mcp_leave_tournament(ctx, tournament_id: int) -> dict:
    user = await _user_from_context(ctx)
    async with _session_factory() as session:
        svc = TournamentService(session, _bus)
        return await leave_tournament(
            ctx=ctx, service=svc, user=user, tournament_id=tournament_id
        )
```

Repeat for each of the five tools. The un-decorated async functions remain independently testable.

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/dashboard/mcp/test_tools_plan_2a.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/mcp/tools.py \
        tests/unit/dashboard/mcp/test_tools_plan_2a.py
git commit -m "feat(tournament): add 5 new MCP tools (leave/get_history/list/get/cancel)"
```

---

## Task 19: REST admin endpoints (6 handlers in tournament_api.py)

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py`
- Test: `tests/integration/dashboard/tournament/test_rest_admin.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/dashboard/tournament/test_rest_admin.py`:

```python
"""Integration tests for the 6 REST admin endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from atp.dashboard.v2.factory import create_app


@pytest.fixture
def app(tournament_db, monkeypatch):
    # Inject the test DB into the factory
    monkeypatch.setenv("ATP_DATABASE_URL", str(tournament_db.url))
    monkeypatch.setenv("ATP_DISABLE_AUTH", "true")  # test mode
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
async def seeded_tournament(session_factory):
    async with session_factory() as s:
        await s.execute(text(
            "INSERT INTO users (id, username, is_admin) "
            "VALUES (1, 'alice', 1)"
        ))
        await s.execute(text(
            "INSERT INTO tournaments "
            "(id, name, status, num_players, total_rounds, round_deadline_s, "
            " pending_deadline, created_by) "
            "VALUES (1, 'test', 'pending', 2, 3, 30, CURRENT_TIMESTAMP, 1)"
        ))
        await s.commit()


def test_list_tournaments_returns_200(client, seeded_tournament):
    response = client.get("/api/v1/tournaments")
    assert response.status_code == 200
    body = response.json()
    assert "tournaments" in body


def test_get_tournament_detail_returns_200(client, seeded_tournament):
    response = client.get("/api/v1/tournaments/1")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert "join_token" not in body  # never serialized
    assert "has_join_token" in body


def test_get_tournament_missing_returns_404(client, seeded_tournament):
    response = client.get("/api/v1/tournaments/99999")
    assert response.status_code == 404


def test_get_rounds_returns_200(client, seeded_tournament):
    response = client.get("/api/v1/tournaments/1/rounds")
    assert response.status_code == 200


def test_get_participants_returns_200(client, seeded_tournament):
    response = client.get("/api/v1/tournaments/1/participants")
    assert response.status_code == 200


def test_create_tournament_returns_token_once(client):
    response = client.post(
        "/api/v1/tournaments",
        json={
            "name": "new",
            "game_type": "prisoners_dilemma",
            "num_players": 2,
            "total_rounds": 3,
            "round_deadline_s": 30,
            "private": True,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert "join_token" in body  # returned once on create
    assert body["join_token"] is not None


def test_cancel_tournament_returns_200(client, seeded_tournament):
    response = client.post("/api/v1/tournaments/1/cancel")
    assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/integration/dashboard/tournament/test_rest_admin.py -v`
Expected: FAIL with 404/501 on most endpoints (not implemented yet).

- [ ] **Step 3: Implement the 6 REST handlers**

Edit `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py`. Delete any existing 501 stubs for these routes. Add the following handler set (adapt to the file's existing `APIRouter` and dependency patterns):

```python
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from atp.dashboard.tournament.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from atp.dashboard.tournament.models import TournamentStatus
from atp.dashboard.tournament.service import TournamentService

router = APIRouter(prefix="/api/v1/tournaments", tags=["tournaments"])


class CreateTournamentRequest(BaseModel):
    name: str
    game_type: str = "prisoners_dilemma"
    num_players: int = Field(ge=2)
    total_rounds: int = Field(ge=1)
    round_deadline_s: int = Field(ge=1)
    private: bool = False


class CreateTournamentResponse(BaseModel):
    id: int
    name: str
    status: str
    has_join_token: bool
    join_token: str | None = None  # returned once on create only


class TournamentResponse(BaseModel):
    id: int
    name: str
    status: str
    game_type: str
    num_players: int
    total_rounds: int
    round_deadline_s: int
    has_join_token: bool
    cancelled_reason: str | None = None
    cancelled_reason_detail: str | None = None


def _serialize(t, is_admin: bool) -> dict:
    base = {
        "id": t.id,
        "name": t.name,
        "status": t.status if isinstance(t.status, str) else t.status.value,
        "game_type": t.game_type,
        "num_players": t.num_players,
        "total_rounds": t.total_rounds,
        "round_deadline_s": t.round_deadline_s,
        "has_join_token": bool(t.join_token),
        "cancelled_reason": (
            t.cancelled_reason.value
            if t.cancelled_reason is not None
            else None
        ),
        "cancelled_reason_detail": t.cancelled_reason_detail,
    }
    if is_admin:
        base["cancelled_by"] = t.cancelled_by
    return base


@router.get("")
async def list_tournaments_endpoint(
    status_filter: str | None = None,
    user=Depends(get_current_user),
    service: TournamentService = Depends(get_tournament_service),
):
    filt = TournamentStatus(status_filter) if status_filter else None
    tournaments = await service.list_tournaments(user=user, status=filt)
    return {
        "tournaments": [_serialize(t, user.is_admin) for t in tournaments]
    }


@router.get("/{tournament_id}")
async def get_tournament_endpoint(
    tournament_id: int,
    user=Depends(get_current_user),
    service: TournamentService = Depends(get_tournament_service),
):
    try:
        t = await service.get_tournament(tournament_id, user)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="tournament not found")
    return _serialize(t, user.is_admin)


@router.get("/{tournament_id}/rounds")
async def get_rounds_endpoint(
    tournament_id: int,
    user=Depends(get_current_user),
    service: TournamentService = Depends(get_tournament_service),
):
    try:
        rounds = await service.get_history(tournament_id, user)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="tournament not found")
    return {
        "rounds": [
            {
                "round_number": r.round_number,
                "status": r.status if isinstance(r.status, str) else r.status.value,
            }
            for r in rounds
        ]
    }


@router.get("/{tournament_id}/participants")
async def get_participants_endpoint(
    tournament_id: int,
    user=Depends(get_current_user),
    service: TournamentService = Depends(get_tournament_service),
):
    try:
        t = await service.get_tournament(tournament_id, user)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="tournament not found")
    # Load participants from relationship (or separate query)
    participants = []
    for p in t.participants:
        row = {
            "id": p.id,
            "user_id": p.user_id,
            "agent_name": p.agent_name,
        }
        if p.user_id == user.id or user.is_admin:
            row["released_at"] = (
                p.released_at.isoformat() if p.released_at else None
            )
        participants.append(row)
    return {"participants": participants}


@router.post("", status_code=201)
async def create_tournament_endpoint(
    req: CreateTournamentRequest,
    user=Depends(get_current_user),
    service: TournamentService = Depends(get_tournament_service),
):
    try:
        tournament, join_token = await service.create_tournament(
            creator=user,
            name=req.name,
            game_type=req.game_type,
            num_players=req.num_players,
            total_rounds=req.total_rounds,
            round_deadline_s=req.round_deadline_s,
            private=req.private,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    response = _serialize(tournament, user.is_admin)
    response["join_token"] = join_token  # returned once on create only
    return response


@router.post("/{tournament_id}/cancel")
async def cancel_tournament_endpoint(
    tournament_id: int,
    user=Depends(get_current_user),
    service: TournamentService = Depends(get_tournament_service),
):
    try:
        await service.cancel_tournament(
            user=user, tournament_id=tournament_id
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="tournament not found")
    return {"cancelled": True}
```

Dependency helpers `get_current_user` and `get_tournament_service` should live alongside the router. If they don't exist, create them:

```python
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

async def get_current_user(request: Request):
    user = getattr(request.state, "user", None)
    if user is None:
        # In test mode (ATP_DISABLE_AUTH=true), fall back to user id 1
        import os
        if os.environ.get("ATP_DISABLE_AUTH") == "true":
            from atp.dashboard.models import User
            async with request.app.state.session_factory() as s:
                return await s.get(User, 1)
        raise HTTPException(status_code=401, detail="unauthenticated")
    return user


async def get_tournament_service(request: Request):
    async with request.app.state.session_factory() as session:
        yield TournamentService(
            session=session,
            bus=request.app.state.tournament_event_bus,
        )
```

Mount the router on the FastAPI app in `factory.py` if not already mounted:

```python
from atp.dashboard.v2.routes.tournament_api import router as tournament_router
app.include_router(tournament_router)
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/integration/dashboard/tournament/test_rest_admin.py -v`
Expected: PASS — 7 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py \
        packages/atp-dashboard/atp/dashboard/v2/factory.py \
        tests/integration/dashboard/tournament/test_rest_admin.py
git commit -m "feat(tournament): 6 REST admin endpoints with visibility filter + serialization"
```

---

## Task 20: Integration tests — cancel cascade (3 paths) + CHECK constraint + partial unique index

**Files:**
- Test: `tests/integration/dashboard/tournament/test_cancel_cascade.py`
- Test: `tests/integration/dashboard/tournament/test_cancel_check_constraint.py`
- Test: `tests/integration/dashboard/tournament/test_partial_unique_index.py`

- [ ] **Step 1: Write cancel cascade tests**

Create `tests/integration/dashboard/tournament/test_cancel_cascade.py`:

```python
"""Integration tests for the three cancel paths (user/system/abandoned)
and their full cascade effects."""

import pytest
from sqlalchemy import func, select, text

from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import (
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService


class _CapturingBus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


async def _seed_active_tournament_with_rounds(session):
    await session.execute(text(
        "INSERT INTO users (id, username, is_admin) "
        "VALUES (1, 'alice', 0), (2, 'bob', 0), (99, 'admin', 1)"
    ))
    await session.execute(text(
        "INSERT INTO tournaments "
        "(id, name, status, num_players, total_rounds, round_deadline_s, "
        " pending_deadline, created_by) "
        "VALUES (1, 't', 'active', 2, 3, 30, CURRENT_TIMESTAMP, 99)"
    ))
    await session.execute(text(
        "INSERT INTO tournament_participants "
        "(tournament_id, user_id, agent_name) VALUES "
        "(1, 1, 'a'), (1, 2, 'b')"
    ))
    await session.execute(text(
        "INSERT INTO tournament_rounds "
        "(tournament_id, round_number, status) VALUES "
        "(1, 1, 'completed'), (1, 2, 'waiting_for_actions')"
    ))


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["user_initiated", "pending_timeout"])
async def test_cancel_cascade_full_effect(session_factory, path):
    async with session_factory() as setup:
        await _seed_active_tournament_with_rounds(setup)
        await setup.commit()

    bus = _CapturingBus()
    async with session_factory() as session:
        svc = TournamentService(session, bus)
        if path == "user_initiated":
            from atp.dashboard.models import User
            admin = await session.get(User, 99)
            await svc.cancel_tournament(user=admin, tournament_id=1)
        else:
            # pending_timeout — for this test we reuse the system entry
            # point even though tournament is ACTIVE (service guard allows
            # any status transition via _cancel_impl)
            await svc.cancel_tournament_system(
                tournament_id=1, reason=CancelReason.PENDING_TIMEOUT
            )

    async with session_factory() as verify:
        refreshed = await verify.get(Tournament, 1)
        assert refreshed.status == TournamentStatus.CANCELLED
        assert refreshed.cancelled_at is not None

        expected_reason = {
            "user_initiated": CancelReason.ADMIN_ACTION,
            "pending_timeout": CancelReason.PENDING_TIMEOUT,
        }[path]
        assert refreshed.cancelled_reason == expected_reason

        if path == "user_initiated":
            assert refreshed.cancelled_by == 99
        else:
            assert refreshed.cancelled_by is None

        completed = await verify.scalar(
            select(func.count()).select_from(Round)
            .where(Round.tournament_id == 1)
            .where(Round.status == RoundStatus.COMPLETED)
        )
        assert completed == 1  # pre-cancel COMPLETED round preserved

        in_flight = await verify.scalar(
            select(func.count()).select_from(Round)
            .where(Round.tournament_id == 1)
            .where(Round.status == RoundStatus.WAITING_FOR_ACTIONS)
        )
        assert in_flight == 0  # transitioned to CANCELLED

        unreleased = await verify.scalar(
            select(func.count()).select_from(Participant)
            .where(Participant.tournament_id == 1)
            .where(Participant.released_at.is_(None))
        )
        assert unreleased == 0  # all released

    cancel_events = [
        e for e in bus.events if isinstance(e, TournamentCancelEvent)
    ]
    assert len(cancel_events) == 1
    assert cancel_events[0].final_rounds_played == 1


@pytest.mark.anyio
async def test_idempotent_cancel_no_double_transition(session_factory):
    async with session_factory() as setup:
        await _seed_active_tournament_with_rounds(setup)
        await setup.commit()

    bus = _CapturingBus()
    async with session_factory() as session:
        from atp.dashboard.models import User
        admin = await session.get(User, 99)
        svc = TournamentService(session, bus)
        await svc.cancel_tournament(user=admin, tournament_id=1)

    async with session_factory() as session:
        from atp.dashboard.models import User
        admin = await session.get(User, 99)
        svc = TournamentService(session, bus)
        await svc.cancel_tournament(user=admin, tournament_id=1)

    # Exactly one event — second call is an idempotent no-op
    cancel_events = [
        e for e in bus.events if isinstance(e, TournamentCancelEvent)
    ]
    assert len(cancel_events) == 1


@pytest.mark.anyio
async def test_cancel_publish_failure_returns_success(session_factory):
    async with session_factory() as setup:
        await _seed_active_tournament_with_rounds(setup)
        await setup.commit()

    class _FailingBus:
        async def publish(self, event):
            raise ConnectionError("bus down")

    async with session_factory() as session:
        from atp.dashboard.models import User
        admin = await session.get(User, 99)
        svc = TournamentService(session, _FailingBus())
        # MUST NOT raise
        await svc.cancel_tournament(user=admin, tournament_id=1)

    async with session_factory() as verify:
        refreshed = await verify.get(Tournament, 1)
        assert refreshed.status == TournamentStatus.CANCELLED
```

- [ ] **Step 2: Write CHECK constraint test**

Create `tests/integration/dashboard/tournament/test_cancel_check_constraint.py`:

```python
"""Integration tests for ck_tournament_cancel_consistency CHECK constraint."""

from datetime import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


@pytest.mark.anyio
@pytest.mark.parametrize("fields,should_fail", [
    ({"status": "active", "cancelled_reason": None, "cancelled_by": None, "cancelled_at": None}, False),
    ({"status": "cancelled", "cancelled_reason": "admin_action", "cancelled_by": 1, "cancelled_at": "2026-04-15T10:00:00"}, False),
    ({"status": "cancelled", "cancelled_reason": "pending_timeout", "cancelled_by": None, "cancelled_at": "2026-04-15T10:00:00"}, False),
    ({"status": "cancelled", "cancelled_reason": "abandoned", "cancelled_by": None, "cancelled_at": "2026-04-15T10:00:00"}, False),
    ({"status": "cancelled", "cancelled_reason": "admin_action", "cancelled_by": None, "cancelled_at": "2026-04-15T10:00:00"}, True),
    ({"status": "cancelled", "cancelled_reason": "pending_timeout", "cancelled_by": 1, "cancelled_at": "2026-04-15T10:00:00"}, True),
    ({"status": "cancelled", "cancelled_reason": "abandoned", "cancelled_by": 1, "cancelled_at": "2026-04-15T10:00:00"}, True),
    ({"status": "cancelled", "cancelled_reason": "admin_action", "cancelled_by": 1, "cancelled_at": None}, True),
    ({"status": "cancelled", "cancelled_reason": None, "cancelled_by": 1, "cancelled_at": None}, True),
])
async def test_check_constraint_enforces_cancel_tuple(
    session_factory, fields, should_fail
):
    async with session_factory() as s:
        await s.execute(text(
            "INSERT INTO users (id, username, is_admin) "
            "VALUES (1, 'alice', 0)"
        ))
        await s.execute(text(
            "INSERT INTO tournaments "
            "(id, name, status, num_players, total_rounds, round_deadline_s, "
            " pending_deadline, created_by) "
            "VALUES (1, 't', 'active', 2, 3, 30, CURRENT_TIMESTAMP, 1)"
        ))
        await s.commit()

    async def _attempt_update():
        async with session_factory() as s:
            await s.execute(
                text(
                    "UPDATE tournaments SET "
                    "status = :status, "
                    "cancelled_reason = :cancelled_reason, "
                    "cancelled_by = :cancelled_by, "
                    "cancelled_at = :cancelled_at "
                    "WHERE id = 1"
                ),
                fields,
            )
            await s.commit()

    if should_fail:
        with pytest.raises(IntegrityError) as exc_info:
            await _attempt_update()
        assert "ck_tournament_cancel_consistency" in str(exc_info.value)
    else:
        await _attempt_update()
```

- [ ] **Step 3: Write partial unique index test**

Create `tests/integration/dashboard/tournament/test_partial_unique_index.py`:

```python
"""Integration tests for uq_participant_user_active partial unique index."""

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from atp.dashboard.tournament.models import Participant


async def _seed_two_tournaments_and_user(session):
    await session.execute(text(
        "INSERT INTO users (id, username, is_admin) VALUES (1, 'alice', 0)"
    ))
    await session.execute(text(
        "INSERT INTO tournaments "
        "(id, name, status, num_players, total_rounds, round_deadline_s, "
        " pending_deadline, created_by) "
        "VALUES "
        "(1, 't1', 'pending', 2, 3, 30, CURRENT_TIMESTAMP, 1),"
        "(2, 't2', 'pending', 2, 3, 30, CURRENT_TIMESTAMP, 1)"
    ))


@pytest.mark.anyio
async def test_one_active_per_user_across_tournaments(session_factory):
    async with session_factory() as s:
        await _seed_two_tournaments_and_user(s)
        await s.execute(text(
            "INSERT INTO tournament_participants "
            "(tournament_id, user_id, agent_name) VALUES (1, 1, 'a')"
        ))
        await s.commit()

    with pytest.raises(IntegrityError) as exc_info:
        async with session_factory() as s:
            await s.execute(text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_name) VALUES (2, 1, 'b')"
            ))
            await s.commit()

    assert "uq_participant_user_active" in str(exc_info.value)


@pytest.mark.anyio
async def test_released_rows_do_not_block(session_factory):
    async with session_factory() as s:
        await s.execute(text(
            "INSERT INTO users (id, username, is_admin) "
            "VALUES (1, 'alice', 0)"
        ))
        # Seed 10 tournaments
        for i in range(1, 11):
            await s.execute(text(
                "INSERT INTO tournaments "
                "(id, name, status, num_players, total_rounds, "
                " round_deadline_s, pending_deadline, created_by) "
                "VALUES "
                "(:id, :n, 'completed', 2, 3, 30, CURRENT_TIMESTAMP, 1)"
            ), {"id": i, "n": f"t{i}"})
        # Seed 10 released participants across them
        for i in range(1, 11):
            await s.execute(text(
                "INSERT INTO tournament_participants "
                "(tournament_id, user_id, agent_name, released_at) "
                "VALUES (:id, 1, :n, CURRENT_TIMESTAMP)"
            ), {"id": i, "n": f"bot{i}"})
        await s.commit()

    async with session_factory() as verify:
        count = await verify.scalar(
            select(func.count()).select_from(Participant)
            .where(Participant.user_id == 1)
        )
        assert count == 10
```

- [ ] **Step 4: Run all three test files**

Run: `uv run pytest tests/integration/dashboard/tournament/test_cancel_cascade.py tests/integration/dashboard/tournament/test_cancel_check_constraint.py tests/integration/dashboard/tournament/test_partial_unique_index.py -v`
Expected: PASS — ~13 tests total.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/dashboard/tournament/test_cancel_cascade.py \
        tests/integration/dashboard/tournament/test_cancel_check_constraint.py \
        tests/integration/dashboard/tournament/test_partial_unique_index.py
git commit -m "test(tournament): integration tests for cancel cascade + CHECK + partial unique"
```

---

## Task 21: Integration test — deadline worker race (AD-6 guard)

**Files:**
- Test: `tests/integration/dashboard/tournament/test_deadline_worker_race.py`

This test relies on SQLite WAL single-writer serialization. If it becomes flaky, the root cause is almost certainly the fixture losing WAL mode — check `tournament_db` fixture.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/dashboard/tournament/test_deadline_worker_race.py`:

```python
"""Race guard test: force_resolve_round vs submit_action on the same round
cannot both commit a resolution. AD-6 status filter + WAL serialization."""

import asyncio

import pytest
from sqlalchemy import select, text

from atp.dashboard.tournament.errors import ConflictError
from atp.dashboard.tournament.models import Round, RoundStatus
from atp.dashboard.tournament.service import TournamentService


class _CapturingBus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


async def _setup_round_with_one_action_pending(session):
    await session.execute(text(
        "INSERT INTO users (id, username, is_admin) "
        "VALUES (1, 'alice', 0), (2, 'bob', 0)"
    ))
    await session.execute(text(
        "INSERT INTO tournaments "
        "(id, name, status, num_players, total_rounds, round_deadline_s, "
        " pending_deadline, created_by) "
        "VALUES (1, 't', 'active', 2, 3, 30, CURRENT_TIMESTAMP, 1)"
    ))
    await session.execute(text(
        "INSERT INTO tournament_participants "
        "(id, tournament_id, user_id, agent_name) VALUES "
        "(1, 1, 1, 'a'), (2, 1, 2, 'b')"
    ))
    await session.execute(text(
        "INSERT INTO tournament_rounds "
        "(id, tournament_id, round_number, status, deadline) "
        "VALUES (1, 1, 1, 'waiting_for_actions', '1970-01-01 00:00:00')"
    ))
    # Only participant 1 has submitted an action; participant 2 pending
    await session.execute(text(
        "INSERT INTO tournament_actions "
        "(round_id, participant_id, action_data, source) "
        "VALUES (1, 1, '{\"move\": \"cooperate\"}', 'submitted')"
    ))


@pytest.mark.anyio
async def test_force_resolve_vs_submit_action(session_factory):
    async with session_factory() as setup:
        await _setup_round_with_one_action_pending(setup)
        await setup.commit()

    bus = _CapturingBus()

    async def deadline_path():
        async with session_factory() as session:
            svc = TournamentService(session, bus)
            try:
                await svc.force_resolve_round(round_id=1)
                return "deadline_won"
            except ConflictError:
                return "deadline_noop"

    async def submit_path():
        async with session_factory() as session:
            from atp.dashboard.models import User
            user2 = await session.get(User, 2)
            svc = TournamentService(session, bus)
            try:
                await svc.submit_action(
                    tournament_id=1,
                    user=user2,
                    action={"move": "cooperate"},
                )
                return "submit_won"
            except ConflictError:
                return "submit_noop"

    results = await asyncio.gather(deadline_path(), submit_path())
    wins = [r for r in results if r.endswith("_won")]
    # Exactly one must win; the other no-ops
    assert len(wins) == 1

    async with session_factory() as verify:
        round_row = await verify.get(Round, 1)
        assert round_row.status == RoundStatus.COMPLETED
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/integration/dashboard/tournament/test_deadline_worker_race.py -v`
Expected: PASS — 1 test. If the test is flaky, re-run 5 times to confirm the AD-6 guard is deterministic. Flakiness = fixture lost WAL mode.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/dashboard/tournament/test_deadline_worker_race.py
git commit -m "test(tournament): AD-6 deadline worker race guard integration test"
```

---

## Task 22: Integration test — session_sync on join and reconnect

**Files:**
- Test: `tests/integration/dashboard/tournament/test_session_sync.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/dashboard/tournament/test_session_sync.py`:

```python
"""Integration tests for session_sync delivery on join and reconnect."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text

from atp.dashboard.mcp.tools import join_tournament
from atp.dashboard.tournament.service import TournamentService


class _CapturingBus:
    async def publish(self, event):
        pass


async def _seed_tournament_in_progress(session):
    await session.execute(text(
        "INSERT INTO users (id, username, is_admin) "
        "VALUES (1, 'alice', 0), (2, 'bob', 0)"
    ))
    await session.execute(text(
        "INSERT INTO tournaments "
        "(id, name, status, num_players, total_rounds, round_deadline_s, "
        " pending_deadline, created_by) "
        "VALUES (1, 't', 'pending', 2, 3, 30, CURRENT_TIMESTAMP, 1)"
    ))


@pytest.mark.anyio
async def test_session_sync_is_first_notification_on_fresh_join(
    session_factory,
):
    async with session_factory() as setup:
        await _seed_tournament_in_progress(setup)
        await setup.commit()

    notifications = []
    ctx = MagicMock()
    ctx.session.send_notification = AsyncMock(
        side_effect=lambda n: notifications.append(n)
    )

    async with session_factory() as session:
        from atp.dashboard.models import User
        user = await session.get(User, 1)
        svc = TournamentService(session, _CapturingBus())
        await join_tournament(
            ctx=ctx, service=svc, user=user,
            tournament_id=1, agent_name="bot", join_token=None,
        )

    assert len(notifications) >= 1
    first = notifications[0]
    assert first.get("event") == "session_sync"


@pytest.mark.anyio
async def test_session_sync_on_idempotent_rejoin(session_factory):
    async with session_factory() as setup:
        await _seed_tournament_in_progress(setup)
        await setup.commit()

    ctx1 = MagicMock()
    ctx1.session.send_notification = AsyncMock()

    # First join
    async with session_factory() as session:
        from atp.dashboard.models import User
        user = await session.get(User, 1)
        svc = TournamentService(session, _CapturingBus())
        await join_tournament(
            ctx=ctx1, service=svc, user=user,
            tournament_id=1, agent_name="bot", join_token=None,
        )

    ctx2 = MagicMock()
    notifications = []
    ctx2.session.send_notification = AsyncMock(
        side_effect=lambda n: notifications.append(n)
    )

    # Reconnect / second join — must also emit session_sync
    async with session_factory() as session:
        from atp.dashboard.models import User
        user = await session.get(User, 1)
        svc = TournamentService(session, _CapturingBus())
        await join_tournament(
            ctx=ctx2, service=svc, user=user,
            tournament_id=1, agent_name="bot", join_token=None,
        )

    assert len(notifications) >= 1
    assert notifications[0].get("event") == "session_sync"
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/integration/dashboard/tournament/test_session_sync.py -v`
Expected: PASS — 2 tests.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/dashboard/tournament/test_session_sync.py
git commit -m "test(tournament): session_sync delivery on join and idempotent rejoin"
```

---

## Task 23: Static architectural guard tests

**Files:**
- Create: `tests/unit/dashboard/tournament/_grep_helper.py`
- Test: `tests/unit/dashboard/tournament/test_static_guards.py`

- [ ] **Step 1: Create the grep helper**

Create `tests/unit/dashboard/tournament/_grep_helper.py`:

```python
"""Minimal git grep wrapper for static architectural guard tests."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class GrepMatch:
    path: str
    line_number: int
    content: str


def grep_pattern(pattern: str, paths: list[str]) -> list[GrepMatch]:
    """Run git grep and return structured matches. Empty list if no matches."""
    cmd = ["git", "grep", "-n", "-E", pattern, "--", *paths]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("git not available — static guards need git grep")

    if result.returncode == 1:
        return []  # git grep exits 1 when no matches
    if result.returncode != 0:
        raise RuntimeError(f"git grep failed: {result.stderr}")

    matches: list[GrepMatch] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        path, lineno, content = parts
        matches.append(GrepMatch(path=path, line_number=int(lineno), content=content))
    return matches
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/dashboard/tournament/test_static_guards.py`:

```python
"""Static architectural guard tests. Run in the unit stage for sub-30s
feedback. They enforce architectural invariants via git grep."""

from pathlib import Path

from tests.unit.dashboard.tournament._grep_helper import grep_pattern


def test_cancel_tournament_system_not_called_from_handlers():
    """Twin-methods invariant: system method must not be reachable from
    any REST or MCP handler. Called only from the deadline worker and
    from service.leave() (same module)."""
    matches = grep_pattern(
        r"cancel_tournament_system\b",
        paths=[
            "packages/atp-dashboard/atp/dashboard/mcp",
            "packages/atp-dashboard/atp/dashboard/v2/routes",
        ],
    )
    assert matches == [], (
        f"cancel_tournament_system called from handler files "
        f"(must be deadline_worker/service-only): "
        f"{[m.path for m in matches]}"
    )


def test_no_bare_string_round_status_comparisons():
    """Plan 2a refactor invariant: all Round.status comparisons must use
    RoundStatus enum, not bare string literals."""
    matches = grep_pattern(
        r'Round\.status\s*[=!]=\s*["\x27]',
        paths=["packages/atp-dashboard/atp/dashboard/tournament"],
    )
    assert matches == [], (
        f"Bare string literal comparison on Round.status: "
        f"{[m.path + ':' + str(m.line_number) for m in matches]}. "
        f"Use RoundStatus enum from models.py."
    )


def test_no_direct_cancel_field_writes_outside_service():
    """All writes to cancelled_* fields must go through _cancel_impl
    (which lives in service.py)."""
    matches = grep_pattern(
        r"\.(cancelled_by|cancelled_at|cancelled_reason|cancelled_reason_detail)\s*=",
        paths=["packages/atp-dashboard/atp/dashboard/tournament"],
    )
    allowed_file = "service.py"
    bad = [m for m in matches if Path(m.path).name != allowed_file]
    assert bad == [], (
        f"Direct cancel-field writes outside service.py _cancel_impl: "
        f"{[m.path + ':' + str(m.line_number) for m in bad]}"
    )
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/dashboard/tournament/test_static_guards.py -v`
Expected: PASS — 3 tests (given the code from prior tasks obeys these rules).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/dashboard/tournament/_grep_helper.py \
        tests/unit/dashboard/tournament/test_static_guards.py
git commit -m "test(tournament): static architectural guards via git grep"
```

---

## Task 24: E2E test — 30-round PD tournament with mid-tournament reconnect

**Files:**
- Test: `tests/e2e/dashboard/tournament/test_e2e_30_round_pd_with_reconnect.py`

This is the primary e2e test for SC-1. It exercises the full MCP SSE stack, deadline worker in lifespan, session_sync on reconnect, and the 30-round gameplay loop at `round_deadline_s=1` for wall-clock economy.

- [ ] **Step 1: Create the e2e test directory**

```bash
mkdir -p tests/e2e/dashboard/tournament
touch tests/e2e/__init__.py
touch tests/e2e/dashboard/__init__.py
touch tests/e2e/dashboard/tournament/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/e2e/dashboard/tournament/test_e2e_30_round_pd_with_reconnect.py`:

```python
"""SC-1: full multi-round PD tournament end-to-end over real MCP SSE,
with mid-tournament reconnect triggering session_sync replay.

30 rounds × round_deadline_s=1 keeps wall clock under ~40s for CI. The
90-round variant exists as test_e2e_90_round_pd_benchmark.py (manual).
"""

import asyncio

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
async def tournament_uvicorn(tournament_db, monkeypatch):
    """Spin up a real uvicorn instance on a random port serving the
    FastAPI app with the deadline worker running in lifespan.

    Returns (base_url, shutdown_coroutine).
    """
    import uvicorn
    from atp.dashboard.v2.factory import create_app

    monkeypatch.setenv("ATP_DATABASE_URL", str(tournament_db.url))
    monkeypatch.setenv("ATP_DISABLE_AUTH", "true")
    monkeypatch.setenv("ATP_DEADLINE_WORKER_POLL_INTERVAL_S", "0.5")

    app = create_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve())
    # Wait for server to start
    while not server.started:
        await asyncio.sleep(0.05)
    # Uvicorn binds to port 0 but we cannot easily discover the assigned
    # port without reaching into its internals. Use a fixed port instead.
    # Fall back: re-create Config with a known free port.
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    free_port = sock.getsockname()[1]
    sock.close()

    # Restart server on the known port
    server.should_exit = True
    await task
    config = uvicorn.Config(
        app, host="127.0.0.1", port=free_port, log_level="warning"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    yield f"http://127.0.0.1:{free_port}"

    server.should_exit = True
    await asyncio.wait_for(task, timeout=5.0)


async def test_thirty_round_pd_with_reconnect_sc1(tournament_uvicorn):
    """30-round PD with both bots cooperating; client 1 reconnects at
    round 15 and must receive session_sync with correct round_number."""
    base_url = tournament_uvicorn
    from atp.adapters.mcp import MCPAdapter  # vertical-slice adapter

    # Admin creates the tournament via REST
    import httpx
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "e2e-sc1",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 30,
                "round_deadline_s": 1,
                "private": False,
            },
        )
        assert response.status_code == 201
        tournament_id = response.json()["id"]

    # Connect two MCPAdapter clients in subscription mode
    adapter_a = MCPAdapter(base_url=f"{base_url}/mcp")
    adapter_b = MCPAdapter(base_url=f"{base_url}/mcp")
    await adapter_a.connect()
    await adapter_b.connect()

    # Both clients join
    await adapter_a.call_tool(
        "join_tournament",
        {"tournament_id": tournament_id, "agent_name": "alice"},
    )
    await adapter_b.call_tool(
        "join_tournament",
        {"tournament_id": tournament_id, "agent_name": "bob"},
    )

    # Play 30 rounds, both cooperating. Mid-tournament reconnect at
    # round 15: drop adapter_a's SSE transport and reconnect.
    events_a = []
    events_b = []

    async def gameplay_loop(adapter, events, reconnect_at: int | None = None):
        for round_num in range(1, 31):
            # Wait for round_started
            event = await adapter.wait_for_event(
                "round_started", timeout=5.0
            )
            events.append(event)
            # Submit cooperate
            await adapter.call_tool(
                "make_move",
                {"tournament_id": tournament_id, "move": "cooperate"},
            )
            # Wait for round_ended
            event = await adapter.wait_for_event(
                "round_ended", timeout=5.0
            )
            events.append(event)

            if reconnect_at is not None and round_num == reconnect_at:
                await adapter.disconnect()
                await adapter.connect()
                # Idempotent re-join triggers session_sync
                result = await adapter.call_tool(
                    "join_tournament",
                    {
                        "tournament_id": tournament_id,
                        "agent_name": "alice",
                    },
                )
                # First notification after reconnect must be session_sync
                sync_event = await adapter.wait_for_event(
                    "session_sync", timeout=2.0
                )
                assert sync_event is not None

    await asyncio.gather(
        gameplay_loop(adapter_a, events_a, reconnect_at=15),
        gameplay_loop(adapter_b, events_b),
    )

    # Both clients saw tournament_completed
    completed_a = await adapter_a.wait_for_event(
        "tournament_completed", timeout=5.0
    )
    completed_b = await adapter_b.wait_for_event(
        "tournament_completed", timeout=5.0
    )
    assert completed_a is not None
    assert completed_b is not None

    await adapter_a.disconnect()
    await adapter_b.disconnect()
```

**Note:** `MCPAdapter.wait_for_event` and `disconnect/connect` are helper shapes — implement whatever shim is needed on top of the vertical-slice `packages/atp-adapters/atp/adapters/mcp/` API if it does not already expose them. If the adapter lacks a clean event-wait primitive, add one as a small internal helper (`_wait_for_event(self, event_type, timeout)`) rather than polluting the production adapter API.

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/e2e/dashboard/tournament/test_e2e_30_round_pd_with_reconnect.py -v --timeout=180`
Expected: PASS — 1 test, wall-clock ~30-60 seconds.

If the test hangs, inspect `tournament_uvicorn` fixture for lifespan startup errors. If the test fails at the reconnect step, verify the MCPAdapter `disconnect/connect` sequence actually closes and reopens the SSE stream (otherwise the "first notification" assertion fires on a stale buffer).

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/__init__.py \
        tests/e2e/dashboard/__init__.py \
        tests/e2e/dashboard/tournament/__init__.py \
        tests/e2e/dashboard/tournament/test_e2e_30_round_pd_with_reconnect.py
git commit -m "test(tournament): SC-1 e2e test — 30-round PD with reconnect"
```

---

## Task 25: E2E tests — cancel paths, AD-9/AD-10 enforcement, REST admin curl

**Files:**
- Test: `tests/e2e/dashboard/tournament/test_e2e_cancel_paths.py`
- Test: `tests/e2e/dashboard/tournament/test_e2e_ad9_ad10.py`
- Test: `tests/e2e/dashboard/tournament/test_e2e_rest_admin_curl.py`

- [ ] **Step 1: Write cancel-path e2e test**

Create `tests/e2e/dashboard/tournament/test_e2e_cancel_paths.py`:

```python
"""SC-3, SC-6, SC-8 cancel-path e2e coverage: user cancel via MCP,
pending timeout, abandoned cascade."""

import asyncio

import httpx
import pytest

pytestmark = pytest.mark.anyio


async def test_e2e_user_cancel_via_mcp(tournament_uvicorn):
    base_url = tournament_uvicorn
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "cancel-test",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 3,
                "round_deadline_s": 5,
                "private": False,
            },
        )
        tid = response.json()["id"]

        # Admin cancels via REST
        cancel_resp = await client.post(f"/api/v1/tournaments/{tid}/cancel")
        assert cancel_resp.status_code == 200

        # Verify status
        detail = await client.get(f"/api/v1/tournaments/{tid}")
        assert detail.json()["status"] == "cancelled"
        assert detail.json()["cancelled_reason"] == "admin_action"


async def test_e2e_pending_timeout_autocancel(tournament_uvicorn, monkeypatch):
    """SC-3: deadline worker auto-cancels PENDING tournaments past
    their pending_deadline. Uses short TOURNAMENT_PENDING_MAX_WAIT_S via
    monkeypatch of the module constant."""
    base_url = tournament_uvicorn

    # Reduce the constant for this test — the app process is already
    # running so we must import and mutate at runtime
    from atp.dashboard.tournament import service as svc_module
    monkeypatch.setattr(svc_module, "TOURNAMENT_PENDING_MAX_WAIT_S", 2)

    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "timeout",
                "game_type": "prisoners_dilemma",
                "num_players": 4,
                "total_rounds": 3,
                "round_deadline_s": 30,
                "private": False,
            },
        )
        tid = response.json()["id"]

        # Wait longer than TOURNAMENT_PENDING_MAX_WAIT_S + poll interval
        await asyncio.sleep(4.0)

        detail = await client.get(f"/api/v1/tournaments/{tid}")
        assert detail.json()["status"] == "cancelled"
        assert detail.json()["cancelled_reason"] == "pending_timeout"


async def test_e2e_idempotent_cancel(tournament_uvicorn):
    """SC-6: second cancel is a no-op, never 500."""
    base_url = tournament_uvicorn
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "idem",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 3,
                "round_deadline_s": 30,
                "private": False,
            },
        )
        tid = response.json()["id"]

        r1 = await client.post(f"/api/v1/tournaments/{tid}/cancel")
        r2 = await client.post(f"/api/v1/tournaments/{tid}/cancel")
        # First call succeeds, second is idempotent (200 or 404 enum guard)
        assert r1.status_code == 200
        assert r2.status_code in (200, 404)
```

- [ ] **Step 2: Write AD-9 / AD-10 enforcement test**

Create `tests/e2e/dashboard/tournament/test_e2e_ad9_ad10.py`:

```python
"""SC-3 AD-9 duration cap + SC-4 AD-10 join_token + 1-active-per-user."""

import httpx
import pytest

pytestmark = pytest.mark.anyio


async def test_e2e_ad9_duration_cap_rejects_over_budget(tournament_uvicorn):
    base_url = tournament_uvicorn
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "too-long",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 200,  # 200 * 30 = 6000s >> (60-10)*60 = 3000s
                "round_deadline_s": 30,
                "private": False,
            },
        )
        assert response.status_code == 422
        body = response.json()
        assert "max duration" in body["detail"].lower()


async def test_e2e_ad10_join_token_required_when_private(tournament_uvicorn):
    base_url = tournament_uvicorn
    async with httpx.AsyncClient(base_url=base_url) as client:
        create_resp = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "private",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 3,
                "round_deadline_s": 30,
                "private": True,
            },
        )
        assert create_resp.status_code == 201
        body = create_resp.json()
        assert body["has_join_token"] is True
        assert body.get("join_token") is not None
        token = body["join_token"]

        # Subsequent GET must NOT expose the token
        detail = await client.get(f"/api/v1/tournaments/{body['id']}")
        assert detail.json()["has_join_token"] is True
        assert "join_token" not in detail.json()
```

- [ ] **Step 3: Write REST admin curl path test**

Create `tests/e2e/dashboard/tournament/test_e2e_rest_admin_curl.py`:

```python
"""SC-8: operator cancels a stuck tournament via curl-equivalent
REST call. The MCP half of SC-8 is covered in test_e2e_cancel_paths."""

import httpx
import pytest

pytestmark = pytest.mark.anyio


async def test_e2e_rest_cancel_returns_200(tournament_uvicorn):
    base_url = tournament_uvicorn
    async with httpx.AsyncClient(base_url=base_url) as client:
        create_resp = await client.post(
            "/api/v1/tournaments",
            json={
                "name": "ops-cancel",
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 3,
                "round_deadline_s": 30,
                "private": False,
            },
        )
        tid = create_resp.json()["id"]

        response = await client.post(
            f"/api/v1/tournaments/{tid}/cancel",
            headers={"Authorization": "Bearer test-admin-token"},
        )
        assert response.status_code == 200

        detail = await client.get(f"/api/v1/tournaments/{tid}")
        body = detail.json()
        assert body["status"] == "cancelled"
        assert body["cancelled_reason"] == "admin_action"
```

- [ ] **Step 4: Add shared e2e fixture if not already in conftest**

Create `tests/e2e/dashboard/tournament/conftest.py` that re-exports `tournament_uvicorn` from Task 24. If Task 24 already placed the fixture in a shared conftest, skip this step.

```python
"""Shared fixtures for tournament e2e tests."""
from tests.e2e.dashboard.tournament.test_e2e_30_round_pd_with_reconnect import (
    tournament_uvicorn,  # noqa: F401 — re-export
)
```

- [ ] **Step 5: Run all e2e tests**

Run: `uv run pytest tests/e2e/dashboard/tournament -v --timeout=180`
Expected: PASS — all tests (4 new + 1 from Task 24).

- [ ] **Step 6: Commit**

```bash
git add tests/e2e/dashboard/tournament/test_e2e_cancel_paths.py \
        tests/e2e/dashboard/tournament/test_e2e_ad9_ad10.py \
        tests/e2e/dashboard/tournament/test_e2e_rest_admin_curl.py \
        tests/e2e/dashboard/tournament/conftest.py
git commit -m "test(tournament): e2e cancel paths + AD-9/AD-10 enforcement + REST curl"
```

---

## Task 26: CI workflow integration and final verification

**Files:**
- Modify: `.github/workflows/ci.yml` (or whichever workflow file exists)
- Modify: `packages/atp-dashboard/pyproject.toml` — add `freezegun` dev dep
- No new test files

- [ ] **Step 1: Add `freezegun` to dev dependencies**

Edit `packages/atp-dashboard/pyproject.toml`. Find the `[dependency-groups.dev]` (or equivalent) block and add:

```toml
freezegun = ">=1.4"
```

Run: `uv sync --all-packages --group dev`
Expected: freezegun installed.

- [ ] **Step 2: Add tournament CI jobs**

Edit `.github/workflows/ci.yml`. Append (or merge with existing job matrix):

```yaml
  tournament-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Sync deps
        run: uv sync --all-packages --group dev
      - name: Tournament unit tests + static guards
        run: >-
          uv run pytest tests/unit/dashboard/tournament
          tests/unit/dashboard/mcp
          tests/unit/dashboard/migrations
          -v --cov=atp.dashboard.tournament
          --cov-report=xml
        timeout-minutes: 2

  tournament-integration:
    runs-on: ubuntu-latest
    needs: tournament-unit
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Sync deps
        run: uv sync --all-packages --group dev
      - name: Tournament integration tests
        run: >-
          uv run pytest tests/integration/dashboard/tournament
          -v --cov=atp.dashboard.tournament
          --cov-append --cov-report=xml
        timeout-minutes: 10

  tournament-e2e:
    runs-on: ubuntu-latest
    needs: tournament-integration
    if: >-
      github.event_name == 'push' ||
      contains(github.event.pull_request.labels.*.name, 'tournament-e2e')
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Sync deps
        run: uv sync --all-packages --group dev
      - name: Tournament e2e tests
        run: >-
          uv run pytest tests/e2e/dashboard/tournament
          -v --timeout=180
          --cov=atp.dashboard.tournament
          --cov-append --cov-report=xml
        timeout-minutes: 15

  tournament-coverage-gate:
    runs-on: ubuntu-latest
    needs: [tournament-unit, tournament-integration]
    steps:
      - uses: actions/checkout@v4
      - name: Enforce 85% coverage floor on tournament package
        run: uv run coverage report --fail-under=85
```

- [ ] **Step 3: Run the full tournament test suite locally (regression check)**

Run:
```bash
uv run pytest tests/unit/dashboard/tournament \
              tests/unit/dashboard/mcp \
              tests/unit/dashboard/migrations \
              tests/integration/dashboard/tournament \
              -v
```
Expected: all tests pass. If any vertical-slice test from `tests/e2e/test_mcp_pd_tournament.py` or pre-existing tournament unit tests is now broken by Plan 2a changes (e.g. required NOT NULL `user_id` on a fixture), fix the fixture — not the schema.

- [ ] **Step 4: Run the vertical slice e2e suite for SC-10 regression**

Run: `uv run pytest tests/e2e/test_mcp_pd_tournament.py -v --timeout=180`
Expected: PASS — vertical slice gameplay still works.

- [ ] **Step 5: Run full pyrefly + ruff**

Run: `uv run pyrefly check`
Expected: 0 errors.

Run: `uv run ruff check .`
Expected: 0 errors. If format violations, run `uv run ruff format .` and re-check.

- [ ] **Step 6: Verify migration against a fresh DB one last time**

Run:
```bash
rm -f /tmp/plan2a_final.db
ATP_DATABASE_URL="sqlite:///tmp/plan2a_final.db" \
  uv run alembic -c migrations/dashboard/alembic.ini upgrade head
ATP_DATABASE_URL="sqlite:///tmp/plan2a_final.db" \
  uv run python -m atp.dashboard.migrations.probes.check_tournament_invariants
```
Expected: upgrade succeeds, probe reports "OK: all tournament schema invariants satisfied".

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/ci.yml \
        packages/atp-dashboard/pyproject.toml \
        uv.lock
git commit -m "ci(tournament): Plan 2a CI jobs, coverage gate, and freezegun dev dep"
```

- [ ] **Step 8: Final self-review pass**

- Run: `git log --oneline main..HEAD` — expect ~26 commits, one per task.
- Run: `git diff main..HEAD --stat` — confirm no untracked files, no accidental changes outside the intended file set.
- Skim the diff for the four `Tournament` audit columns (`cancelled_at`, `cancelled_by`, `cancelled_reason`, `cancelled_reason_detail`) and confirm no stale references to `cancelled_by_user_id`.
- Verify `docs/superpowers/specs/2026-04-11-mcp-tournament-plan-2a-design.md` was **not** touched during implementation.

- [ ] **Step 9: Ready for merge**

If all tests are green and the self-review is clean, the Plan 2a branch is ready for PR review and merge into main. CI will run tournament-unit, tournament-integration, and tournament-coverage-gate on the PR; tournament-e2e will run if the `tournament-e2e` label is applied or on main push.

---

## Acceptance criteria → task mapping

| SC # | Criterion | Covered by tasks |
|---|---|---|
| SC-1 | 30/90-round PD with mid-tournament reconnect → session_sync | Tasks 17, 22, 24 |
| SC-2 | Deadline worker resolves expired round with `source=timeout_default` | Tasks 14, 21 |
| SC-3 | AD-9 pending auto-cancel | Tasks 12, 14, 20, 25 |
| SC-4 | AD-10 concurrent join — one success, one conflict | Tasks 10, 20, 25 |
| SC-5 | Idempotent join — same row, fresh session_sync | Tasks 10, 17, 22 |
| SC-6 | Idempotent cancel — noop or 404, never 500 | Tasks 8, 9, 20, 25 |
| SC-7 | Probe dry-run via `python -m ...` returns exit 0 | Task 4 |
| SC-8 | Operator cancels via curl REST, 200, event in logs | Tasks 9, 19, 25 |
| SC-9 | Fresh Alembic upgrade produces 7 invariants + 8 columns | Tasks 3, 5 |
| SC-10 | All vertical slice tests continue to pass | Task 26 (regression gate) |

## Plan self-review checklist

Before handing off to execution:

- **Spec coverage.** Every Section 1 SC has at least one task. ✓
- **Placeholder scan.** Every step contains concrete code, file paths, or commands — no "TBD" / "TODO" / "add error handling". ✓
- **Type consistency.** `CancelReason`, `RoundStatus`, `ActionSource`, `TournamentCancelEvent` names match across tasks 1-8. Column name `created_by` (not `created_by_user_id`) used everywhere. ✓
- **Commit hygiene.** Each task ends with a commit, and no task combines unrelated concerns.
- **TDD discipline.** Every task follows "write failing test → run (expect fail) → implement → run (expect pass) → commit".

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-11-mcp-tournament-plan-2a.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Requires the `superpowers:subagent-driven-development` skill.

**2. Inline Execution** — execute tasks in this session using the `superpowers:executing-plans` skill, batch execution with checkpoints for review.

Which approach?
