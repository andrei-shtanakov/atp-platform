# Deprecate Ownerless Agents (LABS-54) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the ownerless `Agent` class-of-bugs by decoupling `SuiteExecution` from `Agent` (Option 1.C), deprecating the legacy `POST /api/agents` endpoint (Option 3.a), and flipping `Agent.owner_id` to `NOT NULL` at the DB level.

**Architecture:** Three sequential phases, each shipping as its own PR:
1. Decouple `SuiteExecution` from `Agent` via a denormalized `agent_name` string (CLI stops writing to `agents` table).
2. Replace legacy `POST /api/agents` with `410 Gone`; migrate tests to `/api/v1/agents`.
3. Refactor 43 ownerless test fixtures, delete dead `AgentService.create_agent`, flip `owner_id NOT NULL`, drop the partial unique index from LABS-15.

**Tech Stack:** Python 3.12, SQLAlchemy async, Alembic migrations (SQLite/Postgres), pytest with anyio, FastAPI.

**Source of truth for current state:**
- Alembic head revision: `f1a2b3c4d5e6` (action_reasoning). New migrations chain from here.
- `SuiteExecution.agent_id` currently `nullable=False` — relaxed in Phase 1.
- `Agent.owner_id` currently `nullable=True` — tightened in Phase 3.
- 43 ownerless `Agent(...)` call sites in `tests/` (as of 2026-04-17, verified via `grep -rn "Agent(" tests/ | grep -v owner_id | wc -l`).
- Partial unique index `uq_agent_ownerless_tenant_name_version` added by migration `e1b2c3d4f5a6`.

---

## Phase 1 — Decouple `SuiteExecution` from `Agent` (Option 1.C)

**PR scope:** One merged PR titled `feat(agents): decouple suite_executions from agents via agent_name [LABS-54 Phase 1]`. This is the largest phase. After merge, the CLI `atp test` flow no longer writes to the `agents` table.

**Files in this phase:**
- Create: `migrations/dashboard/versions/a7b8c9d0e1f2_suite_execution_agent_name.py`
- Modify: `packages/atp-dashboard/atp/dashboard/models.py` (`SuiteExecution` class)
- Modify: `packages/atp-dashboard/atp/dashboard/storage.py` (`create_suite_execution`, `submit_report`, `submit_result`, remove CLI dependency on `get_or_create_agent`)
- Modify: `atp/cli/main.py` (line ~968 — `atp test` flow)
- Modify: `packages/atp-dashboard/atp/dashboard/v2/services/test_service.py` (5 places)
- Modify: `packages/atp-dashboard/atp/dashboard/v2/services/comparison_service.py` (5 joins)
- Modify: `packages/atp-dashboard/atp/dashboard/v2/services/export_service.py` (2 places)
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/home.py:101`
- Modify: `packages/atp-dashboard/atp/dashboard/optimized_queries.py`
- Test: `tests/integration/dashboard/test_migration_suite_execution_agent_name.py` (new)
- Test: `tests/unit/dashboard/test_storage.py` (existing — update `get_or_create_agent` tests to not assume CLI path)

---

### Task 1: Create worktree and branch

**Files:** none (infra)

- [ ] **Step 1.1: Start from latest main**

Run:
```bash
cd /Users/Andrei_Shtanakov/labs/all_ai_orchestrators/atp-platform
git fetch github main
git worktree add .worktrees/labs-54-phase1 -b feat/labs-54-decouple-suite-execution github/main
cd .worktrees/labs-54-phase1
uv sync --group dev
```

Expected: Worktree created; `uv sync` installs dev deps; no errors.

- [ ] **Step 1.2: Verify starting state**

Run:
```bash
uv run pytest tests/unit/dashboard/test_storage.py tests/integration/dashboard/tournament -q -m "not slow"
```

Expected: All green. Record pass count as baseline.

---

### Task 2: Write failing migration test

**Files:**
- Create: `tests/integration/dashboard/test_migration_suite_execution_agent_name.py`

- [ ] **Step 2.1: Write the failing integration test**

Create `tests/integration/dashboard/test_migration_suite_execution_agent_name.py` with:

```python
"""Regression test for the agent_name denormalization migration (LABS-54 Phase 1).

Asserts that the migration:
  1. Adds agent_name column with backfill from agents.name.
  2. Relaxes agent_id to nullable.
  3. Preserves all existing data (success_rate, tenant_id, timestamps).
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.integration.dashboard.migration_helpers import (
    run_downgrade,
    run_upgrade,
    stamp_revision,
)


@pytest.mark.anyio
async def test_migration_adds_agent_name_and_backfills(
    async_engine: AsyncEngine,
) -> None:
    """Agent_name is populated from agents.name during upgrade."""
    await stamp_revision(async_engine, "f1a2b3c4d5e6")
    async with async_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO agents (id, tenant_id, name, agent_type, "
                "owner_id, version) VALUES "
                "(1, 'default', 'fixture-agent', 'http', NULL, 'latest')"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO suite_executions "
                "(id, tenant_id, suite_name, agent_id, started_at, status) "
                "VALUES (1, 'default', 'suite-a', 1, :started, 'completed')"
            ),
            {"started": datetime(2026, 1, 1)},
        )

    await run_upgrade(async_engine, "a7b8c9d0e1f2")

    async with async_engine.connect() as conn:
        result = await conn.execute(
            text("SELECT agent_name FROM suite_executions WHERE id = 1")
        )
        row = result.one()
        assert row[0] == "fixture-agent"


@pytest.mark.anyio
async def test_migration_agent_id_becomes_nullable(
    async_engine: AsyncEngine,
) -> None:
    """After upgrade, agent_id can be NULL for newly-inserted rows."""
    await stamp_revision(async_engine, "f1a2b3c4d5e6")
    await run_upgrade(async_engine, "a7b8c9d0e1f2")

    async with async_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO suite_executions "
                "(tenant_id, suite_name, agent_id, agent_name, "
                "started_at, status) VALUES "
                "('default', 'cli-run', NULL, 'cli-agent', :started, 'running')"
            ),
            {"started": datetime(2026, 1, 1)},
        )
        result = await conn.execute(
            text(
                "SELECT agent_id, agent_name FROM suite_executions "
                "WHERE suite_name = 'cli-run'"
            )
        )
        row = result.one()
        assert row[0] is None
        assert row[1] == "cli-agent"


@pytest.mark.anyio
async def test_migration_downgrade_removes_agent_name(
    async_engine: AsyncEngine,
) -> None:
    """Downgrade drops agent_name and restores agent_id NOT NULL."""
    await stamp_revision(async_engine, "f1a2b3c4d5e6")
    await run_upgrade(async_engine, "a7b8c9d0e1f2")
    await run_downgrade(async_engine, "f1a2b3c4d5e6")

    async with async_engine.connect() as conn:
        result = await conn.execute(text("PRAGMA table_info(suite_executions)"))
        columns = {row[1] for row in result.all()}
        assert "agent_name" not in columns
```

- [ ] **Step 2.2: Verify test fails (migration doesn't exist yet)**

Run:
```bash
uv run pytest tests/integration/dashboard/test_migration_suite_execution_agent_name.py -v
```

Expected: ImportError or FAIL — `a7b8c9d0e1f2` is not a valid revision.

Note: If `migration_helpers` doesn't exist, check `tests/integration/dashboard/` for the helper used by `test_migration_plan_2a.py` and reuse the same import pattern.

---

### Task 3: Create the migration

**Files:**
- Create: `migrations/dashboard/versions/a7b8c9d0e1f2_suite_execution_agent_name.py`

- [ ] **Step 3.1: Write the migration**

Create `migrations/dashboard/versions/a7b8c9d0e1f2_suite_execution_agent_name.py`:

```python
"""Decouple suite_executions from agents via denormalized agent_name.

Adds suite_executions.agent_name (populated by backfill from agents.name)
and relaxes suite_executions.agent_id to nullable. This lets the CLI
write suite executions without requiring an Agent row, breaking the
ownerless-agent creation path through the upload flow (LABS-54 Phase 1).

Revision ID: a7b8c9d0e1f2
Revises: f1a2b3c4d5e6
Create Date: 2026-04-17 12:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("suite_executions") as batch_op:
        batch_op.add_column(
            sa.Column("agent_name", sa.String(length=100), nullable=True)
        )

    op.execute(
        sa.text(
            "UPDATE suite_executions "
            "SET agent_name = (SELECT name FROM agents "
            "WHERE agents.id = suite_executions.agent_id) "
            "WHERE agent_name IS NULL"
        )
    )

    null_name_count = (
        op.get_bind()
        .execute(
            sa.text("SELECT COUNT(*) FROM suite_executions WHERE agent_name IS NULL")
        )
        .scalar_one()
    )
    if null_name_count:
        raise RuntimeError(
            f"Backfill left {null_name_count} suite_executions rows with NULL "
            "agent_name. Every row must have a resolvable agent.name."
        )

    with op.batch_alter_table("suite_executions") as batch_op:
        batch_op.alter_column(
            "agent_name",
            existing_type=sa.String(length=100),
            nullable=False,
        )
        batch_op.alter_column(
            "agent_id",
            existing_type=sa.Integer(),
            nullable=True,
        )
        batch_op.create_index(
            "idx_suite_exec_agent_name",
            ["agent_name"],
        )


def downgrade() -> None:
    with op.batch_alter_table("suite_executions") as batch_op:
        batch_op.drop_index("idx_suite_exec_agent_name")
        batch_op.alter_column(
            "agent_id",
            existing_type=sa.Integer(),
            nullable=False,
        )
        batch_op.drop_column("agent_name")
```

- [ ] **Step 3.2: Run the migration test**

Run:
```bash
uv run pytest tests/integration/dashboard/test_migration_suite_execution_agent_name.py -v
```

Expected: All three tests PASS.

- [ ] **Step 3.3: Commit**

```bash
git add migrations/dashboard/versions/a7b8c9d0e1f2_suite_execution_agent_name.py \
        tests/integration/dashboard/test_migration_suite_execution_agent_name.py
git commit -m "feat(migration): add suite_executions.agent_name, relax agent_id to nullable"
```

---

### Task 4: Update `SuiteExecution` ORM model

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/models.py`

- [ ] **Step 4.1: Write the failing unit test**

Add to `tests/unit/dashboard/test_model_columns.py` (or create if it doesn't have a SuiteExecution section):

```python
def test_suite_execution_has_agent_name_column() -> None:
    from atp.dashboard.models import SuiteExecution

    cols = {c.name: c for c in SuiteExecution.__table__.columns}
    assert "agent_name" in cols
    assert cols["agent_name"].nullable is False
    assert cols["agent_id"].nullable is True
```

- [ ] **Step 4.2: Verify test fails**

Run:
```bash
uv run pytest tests/unit/dashboard/test_model_columns.py::test_suite_execution_has_agent_name_column -v
```

Expected: FAIL — `agent_name` not in columns.

- [ ] **Step 4.3: Update the model**

In `packages/atp-dashboard/atp/dashboard/models.py`, modify the `SuiteExecution` class (currently lines 134-186). Change:

```python
    agent_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("agents.id"), nullable=False
    )
```

to:

```python
    # Nullable: CLI-produced executions write only agent_name; ownership is
    # out of scope here (LABS-54 Phase 1).
    agent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("agents.id"), nullable=True
    )
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
```

Update the relationship to reflect the nullable agent:

```python
    # Relationships
    agent: Mapped["Agent | None"] = relationship(back_populates="suite_executions")
```

Update indexes in `__table_args__` — add:

```python
        Index("idx_suite_exec_agent_name", "agent_name"),
```

- [ ] **Step 4.4: Update `__repr__`**

In the same class, replace:

```python
    def __repr__(self) -> str:
        return (
            f"SuiteExecution(id={self.id}, suite={self.suite_name!r}, "
            f"agent_id={self.agent_id})"
        )
```

with:

```python
    def __repr__(self) -> str:
        return (
            f"SuiteExecution(id={self.id}, suite={self.suite_name!r}, "
            f"agent={self.agent_name!r})"
        )
```

- [ ] **Step 4.5: Verify the model test passes**

Run:
```bash
uv run pytest tests/unit/dashboard/test_model_columns.py::test_suite_execution_has_agent_name_column -v
```

Expected: PASS.

- [ ] **Step 4.6: Run type checker**

Run:
```bash
uv run pyrefly check packages/atp-dashboard/atp/dashboard/models.py
```

Expected: No new errors.

- [ ] **Step 4.7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/models.py \
        tests/unit/dashboard/test_model_columns.py
git commit -m "feat(models): add SuiteExecution.agent_name, relax agent_id nullable"
```

---

### Task 5: Update `storage.py` — `create_suite_execution` accepts name string

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/storage.py`

- [ ] **Step 5.1: Write failing test for name-based API**

Add to `tests/unit/dashboard/test_storage.py`:

```python
@pytest.mark.anyio
async def test_create_suite_execution_by_name_no_agent_row(self) -> None:
    """CLI flow: create SuiteExecution with agent_name but no Agent row."""
    from atp.dashboard.storage import ResultStorage
    from unittest.mock import AsyncMock

    mock_session = AsyncMock()
    storage = ResultStorage(mock_session)

    execution = await storage.create_suite_execution_by_name(
        suite_name="test-suite",
        agent_name="cli-http-agent",
        runs_per_test=1,
    )

    assert execution.agent_name == "cli-http-agent"
    assert execution.agent_id is None
    mock_session.add.assert_called_once()
```

- [ ] **Step 5.2: Verify test fails**

Run:
```bash
uv run pytest tests/unit/dashboard/test_storage.py::TestResultStorage::test_create_suite_execution_by_name_no_agent_row -v
```

Expected: FAIL — `create_suite_execution_by_name` not defined.

- [ ] **Step 5.3: Add new `create_suite_execution_by_name` method**

In `packages/atp-dashboard/atp/dashboard/storage.py`, right after the existing `create_suite_execution` method (around line 127), add:

```python
    async def create_suite_execution_by_name(
        self,
        suite_name: str,
        agent_name: str,
        runs_per_test: int = 1,
        started_at: datetime | None = None,
    ) -> SuiteExecution:
        """Create a SuiteExecution with only agent_name (no Agent row).

        Used by the CLI upload path to record test executions without
        touching the agents table. This is the decoupled contract: a
        suite execution is a log of what ran, not a side effect of
        registering an agent (LABS-54).
        """
        execution = SuiteExecution(
            suite_name=suite_name,
            agent_id=None,
            agent_name=agent_name,
            runs_per_test=runs_per_test,
            started_at=started_at or datetime.now(tz=UTC),
            status="running",
        )
        self._session.add(execution)
        await self._session.flush()
        return execution
```

- [ ] **Step 5.4: Update the old `create_suite_execution` to copy the agent name**

In the same file, find `create_suite_execution` (around line 100). Add one line inside the `SuiteExecution(...)` construction so it populates `agent_name` from the Agent:

```python
        execution = SuiteExecution(
            suite_name=suite_name,
            agent_id=agent.id,
            agent_name=agent.name,
            runs_per_test=runs_per_test,
            started_at=started_at or datetime.now(tz=UTC),
            status="running",
        )
```

- [ ] **Step 5.5: Verify test passes**

Run:
```bash
uv run pytest tests/unit/dashboard/test_storage.py -v
```

Expected: All pass, including new test.

- [ ] **Step 5.6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/storage.py \
        tests/unit/dashboard/test_storage.py
git commit -m "feat(storage): add create_suite_execution_by_name for CLI decoupling"
```

---

### Task 6: Migrate `submit_report` and `submit_result` in `storage.py` to avoid `get_or_create_agent`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/storage.py` (lines 514, 645)

- [ ] **Step 6.1: Locate `submit_report`**

Find the method at line ~510 of `storage.py`. It currently does:

```python
        # Get or create agent
        agent = await self.get_or_create_agent(
            name=result.agent_name,
            agent_type=agent_type,
            config=adapter_config,
        )
        execution = await self.create_suite_execution(
            suite_name=result.suite_name,
            agent=agent,
            ...
        )
```

- [ ] **Step 6.2: Replace with name-based call**

Change to:

```python
        execution = await self.create_suite_execution_by_name(
            suite_name=result.suite_name,
            agent_name=result.agent_name,
            runs_per_test=runs_per_test,
            started_at=started_at,
        )
```

Remove the `agent_type` and `adapter_config` arguments flowing into `get_or_create_agent` — they are no longer needed at this call site. Keep them as locals only if other lines in `submit_report` still use them.

- [ ] **Step 6.3: Do the same for `submit_result` (line ~645)**

Apply the identical transformation: delete the `get_or_create_agent` call, replace `create_suite_execution` with `create_suite_execution_by_name`.

- [ ] **Step 6.4: Run storage + benchmark tests**

Run:
```bash
uv run pytest tests/unit/dashboard/test_storage.py tests/integration/dashboard -q -m "not slow"
```

Expected: All PASS. If any test asserts that an `Agent` row is created via `submit_report`/`submit_result`, update it to assert `SuiteExecution.agent_name` instead.

- [ ] **Step 6.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/storage.py tests/
git commit -m "refactor(storage): submit_report/submit_result use agent_name (no Agent row)"
```

---

### Task 7: Update CLI `atp test` flow to not create Agent

**Files:**
- Modify: `atp/cli/main.py` (line ~968)

- [ ] **Step 7.1: Find the CLI call site**

In `atp/cli/main.py` around line 971, locate:

```python
            # Get or create agent
            agent = await storage.get_or_create_agent(
                name=agent_name,
                agent_type=adapter_type,
                ...
            )
```

- [ ] **Step 7.2: Remove the Agent creation**

Delete the `get_or_create_agent` call entirely. Then locate the subsequent `create_suite_execution` call that took `agent=agent` and replace it with:

```python
            execution = await storage.create_suite_execution_by_name(
                suite_name=suite_name,
                agent_name=agent_name,
                runs_per_test=runs_per_test,
                started_at=started_at,
            )
```

Drop the unused `adapter_config` variable if it was only passed to `get_or_create_agent`.

- [ ] **Step 7.3: Run CLI tests**

Run:
```bash
uv run pytest tests/ -q -m "not slow" -k "cli"
```

Expected: CLI tests pass. If any test asserts `agent = get_or_create_agent(...)`, update it to use `agent_name` only.

- [ ] **Step 7.4: Manual smoke**

Run:
```bash
uv run atp test demo/test_suites/hello_world.yaml --adapter=cli
```

Expected: Exits with a test-result summary. No DB errors about missing `agent_id` or unique constraint.

- [ ] **Step 7.5: Commit**

```bash
git add atp/cli/main.py
git commit -m "refactor(cli): atp test no longer creates Agent row (LABS-54 Phase 1)"
```

---

### Task 8: Update `test_service.py` leaderboard/listing queries

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/services/test_service.py`

There are 5 places (lines 115, 165, 216, 329, 345, 400, 414, 423 per grep) that read `exec.agent.name` or join `Agent`.

- [ ] **Step 8.1: Replace `exec.agent.name` with `exec.agent_name`**

For each occurrence of `exec.agent.name if exec.agent else None` in `test_service.py`, simplify to `exec.agent_name` (non-nullable on the column now). For `exec.suite_execution.agent.name` use `exec.suite_execution.agent_name`.

Example change (line ~115):

```python
# Before
summary.agent_name = exec.agent.name if exec.agent else None
# After
summary.agent_name = exec.agent_name
```

- [ ] **Step 8.2: Replace `.join(Agent)` queries with filter on `SuiteExecution.agent_name`**

Lines 329 and 400 use a JOIN pattern like:

```python
            .join(Agent)
            .where(
                Agent.name == agent_name,
                ...
            )
```

Replace with:

```python
            .where(
                SuiteExecution.agent_name == agent_name,
                ...
            )
```

Remove the corresponding `.join(Agent)` call. Remove the `Agent` import from the top of the file if no longer used.

- [ ] **Step 8.3: Run test_service tests**

Run:
```bash
uv run pytest tests/unit/dashboard/test_services.py tests/unit/dashboard -q -m "not slow"
```

Expected: PASS. If assertions reference `exec.agent.name`, update them to `exec.agent_name`.

- [ ] **Step 8.4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/services/test_service.py tests/
git commit -m "refactor(test_service): read agent_name from SuiteExecution directly"
```

---

### Task 9: Update `comparison_service.py` joins

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/services/comparison_service.py`

Five join sites: lines 71-74, 209-213, 304-308, 392-396.

- [ ] **Step 9.1: Apply the same JOIN → column transformation to all five sites**

For each occurrence of:

```python
                .join(Agent)
                .where(
                    Agent.name == agent_name,
                    ...
                )
```

Replace with:

```python
                .where(
                    SuiteExecution.agent_name == agent_name,
                    ...
                )
```

Remove `.join(Agent)` and the `Agent` import at the top of the file if no other reference remains.

- [ ] **Step 9.2: Run comparison service tests**

Run:
```bash
uv run pytest tests/ -q -m "not slow" -k "comparison"
```

Expected: PASS.

- [ ] **Step 9.3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/services/comparison_service.py
git commit -m "refactor(comparison_service): drop Agent join, use agent_name column"
```

---

### Task 10: Update `export_service.py`, `home.py`, `optimized_queries.py`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/services/export_service.py` (lines 204, 340)
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/home.py` (line 101)
- Modify: `packages/atp-dashboard/atp/dashboard/optimized_queries.py`

- [ ] **Step 10.1: `export_service.py` — replace `exec.agent.name`**

Line 204:

```python
# Before
            agent_name = exec.agent.name if exec.agent else ""
# After
            agent_name = exec.agent_name
```

Line 340 — inside a dict construction:

```python
# Before
                    "name": exec.agent.name,
# After
                    "name": exec.agent_name,
```

- [ ] **Step 10.2: `home.py:101` — same transformation**

```python
# Before
        summary.agent_name = exec.agent.name if exec.agent else None
# After
        summary.agent_name = exec.agent_name
```

- [ ] **Step 10.3: `optimized_queries.py` — audit remaining references**

Run:
```bash
grep -n "Agent\." packages/atp-dashboard/atp/dashboard/optimized_queries.py
```

Lines 35 and 37 use `select(Agent).where(Agent.name.in_(...))` — this is a pure Agent listing (not a SuiteExecution join), so **keep it as-is**. It does not need to change; the Agent table still exists.

- [ ] **Step 10.4: Run full dashboard suite**

Run:
```bash
uv run pytest tests/unit/dashboard tests/integration/dashboard -q -m "not slow"
```

Expected: All PASS.

- [ ] **Step 10.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/services/export_service.py \
        packages/atp-dashboard/atp/dashboard/v2/routes/home.py
git commit -m "refactor(dashboard): read agent_name from SuiteExecution in home/export"
```

---

### Task 11: Full suite verification and PR

- [ ] **Step 11.1: Run the entire fast test suite**

Run:
```bash
uv run pytest tests/ -q -m "not slow"
```

Expected: All PASS with the same count as the baseline from Task 1.2 (or higher — we added three new tests).

- [ ] **Step 11.2: Run ruff format + check**

Run:
```bash
uv run ruff format .
uv run ruff check . --fix
rm -rf .ruff_cache
uv run ruff check .
```

Expected: "All checks passed!" after the final run.

- [ ] **Step 11.3: Run pyrefly**

Run:
```bash
uv run pyrefly check
```

Expected: No new errors (record existing baseline; new ones = fix before proceeding).

- [ ] **Step 11.4: Push branch and open PR**

Run:
```bash
git push -u origin feat/labs-54-decouple-suite-execution
gh pr create --title "feat(agents): decouple suite_executions from agents via agent_name [LABS-54 Phase 1]" \
  --body "$(cat <<'EOF'
## Summary

Phase 1 of LABS-54. Decouple `SuiteExecution` from `Agent`:

- New `SuiteExecution.agent_name` column (denormalized string).
- `SuiteExecution.agent_id` relaxed to nullable.
- CLI `atp test` flow no longer writes to the `agents` table.
- All dashboard queries read `agent_name` directly (no more `Agent` JOINs for listings).

## Why

`Agent` was overloaded: both a user-registered asset AND a side-effect of CLI runs. The CLI path created ownerless rows, forcing a partial-unique-index workaround (LABS-15). After this phase, CLI executions don't touch `agents` at all.

## Test plan

- [x] Migration upgrade/downgrade regression tests (`test_migration_suite_execution_agent_name.py`)
- [x] Storage test for `create_suite_execution_by_name`
- [x] Model test for new column + nullability
- [x] Full unit + integration suite green
- [x] Manual smoke: `atp test demo/test_suites/hello_world.yaml --adapter=cli`

## Follow-up PRs

- Phase 2: deprecate `POST /api/agents` (410 Gone)
- Phase 3: flip `Agent.owner_id NOT NULL`, drop partial index

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 11.5: Wait for CI and review**

Expected: CI green. Merge when approved.

---

## Phase 2 — Deprecate legacy `POST /api/agents` (Option 3.a)

**PR scope:** One small PR titled `feat(api): deprecate POST /api/agents, return 410 Gone [LABS-54 Phase 2]`. Merged only after Phase 1 is in main.

**Files in this phase:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/agents.py` (lines 55+)
- Modify: `tests/` — any test hitting `POST /api/agents` migrates to `POST /api/v1/agents`
- Test: `tests/integration/dashboard/test_legacy_agents_endpoint_gone.py` (new)

---

### Task 12: Add worktree from updated main

- [ ] **Step 12.1: Update main and create a fresh worktree**

Run:
```bash
cd /Users/Andrei_Shtanakov/labs/all_ai_orchestrators/atp-platform
git fetch github main
git worktree add .worktrees/labs-54-phase2 -b feat/labs-54-deprecate-legacy-agents github/main
cd .worktrees/labs-54-phase2
uv sync --group dev
```

Expected: Clean worktree on latest main (with Phase 1 merged).

---

### Task 13: Write the failing test for 410 Gone

**Files:**
- Create: `tests/integration/dashboard/test_legacy_agents_endpoint_gone.py`

- [ ] **Step 13.1: Write the test**

Create the file with:

```python
"""Regression test for LABS-54 Phase 2: legacy POST /api/agents returns 410."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_legacy_post_api_agents_returns_410(
    authenticated_client: AsyncClient,
) -> None:
    response = await authenticated_client.post(
        "/api/agents",
        json={"name": "test", "agent_type": "http"},
    )
    assert response.status_code == 410
    assert "Deprecation" in response.headers
    assert "Sunset" in response.headers
    assert "/api/v1/agents" in response.json()["detail"]
```

Note: Reuse `authenticated_client` from the existing dashboard conftest. If it does not exist, import it the same way other integration tests do (see `tests/integration/dashboard/conftest.py`).

- [ ] **Step 13.2: Run it and verify failure**

Run:
```bash
uv run pytest tests/integration/dashboard/test_legacy_agents_endpoint_gone.py -v
```

Expected: FAIL — endpoint currently returns 200/201.

---

### Task 14: Return 410 Gone from the legacy endpoint

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/agents.py`

- [ ] **Step 14.1: Replace the handler body**

Find `async def create_agent(` at line 55 of `packages/atp-dashboard/atp/dashboard/v2/routes/agents.py`. Replace its body (the entire function, keep the signature and decorator) with:

```python
async def create_agent(
    # keep existing parameters but mark them unused
    *args: object,
    **kwargs: object,
) -> None:
    """Deprecated. Use POST /api/v1/agents instead.

    Removed in LABS-54 Phase 2 because this endpoint had no ownership
    enforcement. The replacement at /api/v1/agents resolves the owner
    from the authenticated user's JWT.
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail=(
            "POST /api/agents is deprecated. Use POST /api/v1/agents "
            "(resolves owner from your JWT)."
        ),
        headers={
            "Deprecation": "true",
            "Sunset": "Wed, 17 Apr 2026 12:00:00 GMT",
            "Link": '</api/v1/agents>; rel="successor-version"',
        },
    )
```

Add `from fastapi import HTTPException, status` at the top if not already imported.

- [ ] **Step 14.2: Run the regression test**

Run:
```bash
uv run pytest tests/integration/dashboard/test_legacy_agents_endpoint_gone.py -v
```

Expected: PASS.

- [ ] **Step 14.3: Find and update any other test hitting the old endpoint**

Run:
```bash
grep -rn '"/api/agents"' tests/ | grep -v test_legacy_agents_endpoint_gone
```

For each hit, change `/api/agents` → `/api/v1/agents`. If the test was asserting ownerless-agent behavior, update the assertion to expect ownership (via JWT-derived `owner_id`). If the test is asserting the legacy path specifically, delete it.

- [ ] **Step 14.4: Run the full suite**

Run:
```bash
uv run pytest tests/ -q -m "not slow"
```

Expected: All PASS.

- [ ] **Step 14.5: Format and commit**

Run:
```bash
uv run ruff format .
rm -rf .ruff_cache
uv run ruff check .
uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/v2/routes/agents.py \
        tests/integration/dashboard/test_legacy_agents_endpoint_gone.py \
        tests/
git commit -m "feat(api): deprecate POST /api/agents, return 410 Gone"
```

- [ ] **Step 14.6: Push and open PR**

```bash
git push -u origin feat/labs-54-deprecate-legacy-agents
gh pr create --title "feat(api): deprecate POST /api/agents, return 410 Gone [LABS-54 Phase 2]" \
  --body "$(cat <<'EOF'
## Summary

Phase 2 of LABS-54. Legacy `POST /api/agents` returns 410 Gone with deprecation headers. All tests migrated to `POST /api/v1/agents`, which resolves ownership from JWT.

## Test plan

- [x] New regression test asserts 410 + Deprecation/Sunset headers
- [x] All legacy callers in tests migrated to `/api/v1/agents`
- [x] Full suite green

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 3 — Finalize: fixtures, `NOT NULL`, drop partial index

**PR scope:** One PR titled `feat(agents): Agent.owner_id NOT NULL, drop ownerless partial index [LABS-54 Phase 3]`. Merged after Phases 1 & 2 are in main.

**Files in this phase:**
- Delete: `AgentService.create_agent` method in `packages/atp-dashboard/atp/dashboard/v2/services/agent_service.py` (+ the DI binding if unused)
- Modify: `packages/atp-dashboard/atp/dashboard/models.py` — `Agent.owner_id` type + drop partial index
- Create: `migrations/dashboard/versions/b8c9d0e1f2a3_agent_owner_not_null.py`
- Refactor: 43 test fixtures that do `Agent(...)` without `owner_id`

---

### Task 15: Worktree from updated main

- [ ] **Step 15.1: Create fresh worktree**

```bash
cd /Users/Andrei_Shtanakov/labs/all_ai_orchestrators/atp-platform
git fetch github main
git worktree add .worktrees/labs-54-phase3 -b feat/labs-54-owner-not-null github/main
cd .worktrees/labs-54-phase3
uv sync --group dev
```

---

### Task 16: Delete `AgentService.create_agent`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/services/agent_service.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/dependencies.py` (if DI binding unused)
- Delete: tests in `tests/unit/dashboard/test_services.py` that test the deleted method

- [ ] **Step 16.1: Confirm zero production callers**

Run:
```bash
grep -rn "agent_service\.create_agent\|AgentService().*create_agent\|service\.create_agent" \
  packages/ atp/ | grep -v tests/
```

Expected: empty output. If any hit — STOP and escalate.

- [ ] **Step 16.2: Delete the method**

In `packages/atp-dashboard/atp/dashboard/v2/services/agent_service.py`, delete the `async def create_agent(self, agent_data: AgentCreate)` method (currently at line ~70).

- [ ] **Step 16.3: Check DI binding usage**

Run:
```bash
grep -rn "AgentServiceDep\|get_agent_service" packages/ atp/ | grep -v tests/
```

If no non-test callers remain, delete the `get_agent_service` function and `AgentServiceDep` alias in `packages/atp-dashboard/atp/dashboard/v2/dependencies.py`. If any route uses it, keep it.

- [ ] **Step 16.4: Delete the corresponding tests**

In `tests/unit/dashboard/test_services.py`, delete the tests under `class TestAgentService` that exercise `create_agent` (tests at lines ~167, 182, 213, 237, 263). Keep tests that exercise `list_agents` / `get_agent_by_name` if those methods still exist.

- [ ] **Step 16.5: Verify tests still pass**

Run:
```bash
uv run pytest tests/unit/dashboard/test_services.py -v
```

Expected: PASS (with deleted tests gone).

- [ ] **Step 16.6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/services/agent_service.py \
        packages/atp-dashboard/atp/dashboard/v2/dependencies.py \
        tests/unit/dashboard/test_services.py
git commit -m "refactor(services): delete dead AgentService.create_agent + DI binding"
```

---

### Task 17: Refactor 43 ownerless test fixtures

**Files:**
- Modify: many files under `tests/`

- [ ] **Step 17.1: Enumerate the current ownerless fixtures**

Run:
```bash
grep -rn "Agent(" tests/ | grep -v "owner_id" | grep -v ".pyc"
```

Expected: ~43 hits. Save the list to a scratch file for reference.

- [ ] **Step 17.2: Add a shared fixture for a default owner user**

If `tests/conftest.py` does not already have a `default_test_user` fixture, add one:

```python
@pytest.fixture
async def default_test_user(async_session) -> User:
    """Shared owner for Agent fixtures that don't otherwise care about ownership."""
    from atp.dashboard.models import User

    user = User(
        username="fixture-owner",
        email="fixture-owner@test.local",
        github_id=None,
        is_admin=False,
    )
    async_session.add(user)
    await async_session.flush()
    return user
```

- [ ] **Step 17.3: Apply the `owner_id=` addition to each hit**

For each hit identified in 17.1, edit the `Agent(...)` call to include `owner_id=default_test_user.id` (injecting the fixture as needed). Approximate grep-driven edit pattern per file:

```python
# Before
agent = Agent(name="x", agent_type="http")
# After
agent = Agent(name="x", agent_type="http", owner_id=default_test_user.id)
```

Where the test function doesn't take the fixture, add `default_test_user: User` to its signature.

- [ ] **Step 17.4: Re-run grep**

Run:
```bash
grep -rn "Agent(" tests/ | grep -v "owner_id" | grep -v ".pyc"
```

Expected: empty output.

- [ ] **Step 17.5: Run the full suite**

Run:
```bash
uv run pytest tests/ -q -m "not slow"
```

Expected: All PASS.

- [ ] **Step 17.6: Commit**

```bash
git add tests/
git commit -m "test(agents): add owner_id to all Agent fixtures"
```

---

### Task 18: Migration — Agent.owner_id NOT NULL + drop partial index

**Files:**
- Create: `migrations/dashboard/versions/b8c9d0e1f2a3_agent_owner_not_null.py`
- Test: `tests/integration/dashboard/test_migration_agent_owner_not_null.py` (new)

- [ ] **Step 18.1: Write the failing migration test**

Create `tests/integration/dashboard/test_migration_agent_owner_not_null.py`:

```python
"""Regression test for LABS-54 Phase 3: Agent.owner_id NOT NULL."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.integration.dashboard.migration_helpers import (
    run_upgrade,
    stamp_revision,
)


@pytest.mark.anyio
async def test_migration_backfills_ownerless_to_admin(
    async_engine: AsyncEngine,
) -> None:
    await stamp_revision(async_engine, "a7b8c9d0e1f2")

    async with async_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO users (id, username, email, is_admin) "
                "VALUES (42, 'admin', 'admin@test.local', 1)"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO agents (id, tenant_id, name, agent_type, "
                "owner_id, version) VALUES "
                "(1, 'default', 'orphan', 'http', NULL, 'latest')"
            )
        )

    await run_upgrade(async_engine, "b8c9d0e1f2a3")

    async with async_engine.connect() as conn:
        owner = (
            await conn.execute(text("SELECT owner_id FROM agents WHERE id = 1"))
        ).scalar_one()
        assert owner == 42


@pytest.mark.anyio
async def test_migration_drops_partial_unique_index(
    async_engine: AsyncEngine,
) -> None:
    await stamp_revision(async_engine, "a7b8c9d0e1f2")
    await run_upgrade(async_engine, "b8c9d0e1f2a3")

    async with async_engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name='uq_agent_ownerless_tenant_name_version'"
            )
        )
        assert result.first() is None
```

- [ ] **Step 18.2: Run the test — should fail**

Run:
```bash
uv run pytest tests/integration/dashboard/test_migration_agent_owner_not_null.py -v
```

Expected: FAIL — `b8c9d0e1f2a3` does not exist.

- [ ] **Step 18.3: Create the migration**

Create `migrations/dashboard/versions/b8c9d0e1f2a3_agent_owner_not_null.py`:

```python
"""Agent.owner_id NOT NULL + drop ownerless partial unique index.

Backfills any remaining NULL owner_id rows to the lowest-id admin user,
then flips the column to NOT NULL and drops the partial unique index
added by migration e1b2c3d4f5a6 (LABS-15 short-term guardrail).

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-04-17 13:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()

    null_count = bind.execute(
        sa.text("SELECT COUNT(*) FROM agents WHERE owner_id IS NULL")
    ).scalar_one()

    if null_count:
        admin_id = bind.execute(
            sa.text("SELECT id FROM users WHERE is_admin = 1 ORDER BY id LIMIT 1")
        ).scalar_one_or_none()
        if admin_id is None:
            raise RuntimeError(
                f"Cannot backfill agents.owner_id: {null_count} rows have NULL "
                "owner_id but no admin user exists. Create an admin first "
                "(set users.is_admin=1) and re-run."
            )
        bind.execute(
            sa.text("UPDATE agents SET owner_id = :uid WHERE owner_id IS NULL"),
            {"uid": admin_id},
        )

    op.drop_index("uq_agent_ownerless_tenant_name_version", "agents")

    with op.batch_alter_table("agents") as batch_op:
        batch_op.alter_column(
            "owner_id",
            existing_type=sa.Integer(),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("agents") as batch_op:
        batch_op.alter_column(
            "owner_id",
            existing_type=sa.Integer(),
            nullable=True,
        )

    op.create_index(
        "uq_agent_ownerless_tenant_name_version",
        "agents",
        ["tenant_id", "name", "version"],
        unique=True,
        sqlite_where=sa.text("owner_id IS NULL"),
        postgresql_where=sa.text("owner_id IS NULL"),
    )
```

- [ ] **Step 18.4: Run the migration test**

Run:
```bash
uv run pytest tests/integration/dashboard/test_migration_agent_owner_not_null.py -v
```

Expected: PASS.

---

### Task 19: Update `Agent` model — `owner_id` non-nullable + drop partial index

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/models.py`

- [ ] **Step 19.1: Update the type and column definition**

Change (around line 85-87):

```python
    owner_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
```

to:

```python
    owner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
```

- [ ] **Step 19.2: Remove the partial unique index from `__table_args__`**

Delete the entire `Index("uq_agent_ownerless_tenant_name_version", ...)` block (lines 116-124). Keep `UniqueConstraint("tenant_id", "owner_id", "name", "version", ...)` — it's now sufficient because `owner_id` can no longer be NULL.

- [ ] **Step 19.3: Run model + dashboard tests**

Run:
```bash
uv run pytest tests/unit/dashboard tests/integration/dashboard -q -m "not slow"
```

Expected: PASS.

- [ ] **Step 19.4: Commit migration + model together**

```bash
git add migrations/dashboard/versions/b8c9d0e1f2a3_agent_owner_not_null.py \
        packages/atp-dashboard/atp/dashboard/models.py \
        tests/integration/dashboard/test_migration_agent_owner_not_null.py
git commit -m "feat(migration): Agent.owner_id NOT NULL, drop ownerless partial index"
```

---

### Task 20: Full verification and PR

- [ ] **Step 20.1: Run the entire non-slow suite**

Run:
```bash
uv run pytest tests/ -q -m "not slow"
```

Expected: All PASS.

- [ ] **Step 20.2: Run slow suite (covers migrations end-to-end)**

Run:
```bash
uv run pytest tests/ -q -m "slow"
```

Expected: All PASS. Flakes in container-integration tests are acceptable if unrelated.

- [ ] **Step 20.3: Format + type-check**

Run:
```bash
uv run ruff format .
rm -rf .ruff_cache
uv run ruff check .
uv run pyrefly check
```

Expected: clean.

- [ ] **Step 20.4: Push and open PR**

Run:
```bash
git push -u origin feat/labs-54-owner-not-null
gh pr create --title "feat(agents): Agent.owner_id NOT NULL, drop ownerless partial index [LABS-54 Phase 3]" \
  --body "$(cat <<'EOF'
## Summary

Final phase of LABS-54. `Agent.owner_id` is now `NOT NULL` at the DB level. Partial unique index from LABS-15 is dropped (no longer needed — the primary `UniqueConstraint(tenant_id, owner_id, name, version)` is sufficient).

## Changes

- Delete dead `AgentService.create_agent` method + DI binding.
- Refactor 43 test fixtures to own their `Agent` rows.
- Backfill remaining ownerless rows to lowest-id admin during migration.
- Drop `uq_agent_ownerless_tenant_name_version`.

## Acceptance criteria (LABS-54)

- [x] No production code path creates Agent without owner_id
- [x] Agent.owner_id is NOT NULL at the DB level
- [x] All tests pass with owned agents
- [x] Partial unique index from LABS-15 removed
- [x] Deprecation of POST /api/agents complete (Phase 2)

## Test plan

- [x] Migration upgrade/downgrade regression tests
- [x] Full unit + integration suite
- [x] Slow suite green (migration end-to-end)

Closes LABS-54. Also resolves LABS-15 (the partial-index workaround).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 20.5: After merge: close Linear issues**

Add a comment to LABS-54 and LABS-15 pointing to the merged PR. Move both issues to Done.

---

## Self-review (built-in)

**Spec coverage:**
- ✅ Phase 1 (Easy cleanup §1 "Delete AgentService.create_agent") → Task 16
- ✅ Phase 1 (Easy cleanup §2 "Refactor 43 test fixtures") → Task 17
- ✅ Phase 2 (CLI auth redesign → 1.C decouple) → Tasks 2-11
- ✅ Phase 3 (POST /api/agents decision → 3.a 410 Gone) → Tasks 12-14
- ✅ Phase 4 (Schema migration NOT NULL + drop partial index) → Tasks 18-20
- ✅ All 4 acceptance criteria from LABS-54 covered in Phase 3 PR body

**Placeholder scan:** none. Every task has concrete file paths and complete code snippets.

**Type consistency:**
- `SuiteExecution.agent_name: str` (non-nullable) — used consistently in Tasks 4, 5, 8, 9, 10.
- `SuiteExecution.agent_id: int | None` — matches migration `a7b8c9d0e1f2`.
- `Agent.owner_id: int` (NOT NULL in Phase 3) — matches migration `b8c9d0e1f2a3`.
- Method names: `create_suite_execution_by_name` (Task 5) used consistently in Tasks 6 & 7.

**Known assumption to verify during execution:**
- `tests/integration/dashboard/migration_helpers.py` exists (pattern used by `test_migration_plan_2a.py`). If the helper is named differently, the import in Tasks 2 & 18 must be adjusted accordingly.
- `default_test_user` fixture name (Task 17) is new. If a fixture with this role already exists under a different name (e.g., `admin_user`, `test_user`), reuse it.
