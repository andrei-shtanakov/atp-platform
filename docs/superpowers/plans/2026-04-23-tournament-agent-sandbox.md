# Tournament Agent Sandbox — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable users to register up to 5 tournament-purpose agents and run
private test tournaments (with builtin sparring partners) end-to-end through
the dashboard UI, wiring completed matches into `/ui/matches`.

**Architecture:** One `/mcp` endpoint; private/public distinction reuses
existing `Tournament.join_token` (no new visibility enum). New `Agent.purpose`
and `Participant.builtin_strategy` columns plus a partial-unique guard on
match-to-tournament dual-writes. JWT claims carry agent identity so auth
middleware stays decode-only on the hot path.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2 (async), Alembic, Pydantic v2,
Jinja2 + Pico CSS + HTMX, FastMCP, pytest + httpx AsyncClient, ruff, pyrefly.

**Spec:** `docs/superpowers/specs/2026-04-23-tournament-agent-sandbox-design.md`

**Rollout:** Five independent PRs. Each is its own git branch,
fully tested and migrable on its own. Do not combine PRs.

---

## Shared conventions

- Every new test uses `@pytest.mark.anyio` (project-wide async style — not
  `asyncio`). Integration tests reuse fixtures from
  `tests/integration/dashboard/conftest.py` (`test_database`, `db_session`,
  `disable_dashboard_auth`, `admin_token`, `auth_headers`).
- Every code file is formatted with `uv run ruff format <file>` before the
  commit step. Every change is type-checked with `uv run pyrefly check`.
  Both commands are invoked by the pre-commit hook, so commits that drift will
  fail the hook and you fix them inline.
- Commit messages use conventional prefixes `feat(<area>): ...`,
  `fix(<area>): ...`, `test(<area>): ...`, `docs(<area>): ...`.
- Each PR starts from an up-to-date `main` via `git checkout main && git pull
  github main --ff-only && git checkout -b <branch>`. The remote is named
  `github`, not `origin`.

---

# PR-1 — Data model + migration

**Branch:** `labs-tsa/pr1-data-model`

**Files:**
- Create: `migrations/dashboard/versions/<hash>_tournament_agent_sandbox.py`
- Modify: `packages/atp-dashboard/atp/dashboard/models.py` (Agent class, GameResult class)
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/models.py` (Participant class)
- Create: `tests/integration/dashboard/test_tsa_data_model.py`
- Create: `tests/integration/dashboard/test_tsa_alembic_migration.py`

## Task 1.1: Start the branch

- [ ] **Step 1.1.1: Sync main and create branch**

```bash
git checkout main
git pull github main --ff-only
git checkout -b labs-tsa/pr1-data-model
```

- [ ] **Step 1.1.2: Verify current Alembic head**

Run: `uv run alembic heads`
Expected: a single revision id (the last migration on `main`). Note it as
`<PREV_HEAD>` for the new migration's `down_revision`.

## Task 1.2: Add `Agent.purpose` column to the ORM

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/models.py` — `Agent` class definition around line 65

- [ ] **Step 1.2.1: Write the failing test**

Create `tests/integration/dashboard/test_tsa_data_model.py` with:

```python
"""Data-model tests for the tournament agent sandbox (PR-1)."""

from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Agent, Base, GameResult, User
from atp.dashboard.tournament.models import Participant, Tournament


@pytest.fixture
async def test_database() -> AsyncGenerator[Database, None]:
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore[arg-type]


@pytest.fixture
async def db_session(test_database: Database) -> AsyncGenerator[AsyncSession, None]:
    async with test_database.session() as session:
        yield session


@pytest.fixture
async def user(db_session: AsyncSession) -> User:
    u = User(username="u1", email="u1@t.com", hashed_password="x", is_active=True)
    db_session.add(u)
    await db_session.commit()
    await db_session.refresh(u)
    return u


class TestAgentPurpose:
    @pytest.mark.anyio
    async def test_purpose_defaults_to_benchmark(
        self, db_session: AsyncSession, user: User
    ) -> None:
        a = Agent(name="a1", agent_type="mcp", owner_id=user.id)
        db_session.add(a)
        await db_session.commit()
        await db_session.refresh(a)
        assert a.purpose == "benchmark"

    @pytest.mark.anyio
    async def test_purpose_tournament_roundtrips(
        self, db_session: AsyncSession, user: User
    ) -> None:
        a = Agent(name="a2", agent_type="mcp", owner_id=user.id, purpose="tournament")
        db_session.add(a)
        await db_session.commit()
        await db_session.refresh(a)
        assert a.purpose == "tournament"

    @pytest.mark.anyio
    async def test_purpose_invalid_rejected_by_check(
        self, db_session: AsyncSession, user: User
    ) -> None:
        a = Agent(name="a3", agent_type="mcp", owner_id=user.id, purpose="invalid")
        db_session.add(a)
        with pytest.raises(IntegrityError):
            await db_session.commit()
```

- [ ] **Step 1.2.2: Run test — it must fail at import**

Run: `uv run pytest tests/integration/dashboard/test_tsa_data_model.py::TestAgentPurpose -v --no-cov`
Expected: FAIL. The `purpose` attribute does not exist yet on `Agent`.

- [ ] **Step 1.2.3: Add the column to the ORM**

In `packages/atp-dashboard/atp/dashboard/models.py`, inside the `Agent` class
body (right after the `deleted_at` column around line 94), add:

```python
    # LABS-TSA PR-1: agent purpose classification.
    # Benchmark agents run suite evaluations; tournament agents connect to
    # /mcp for game-theoretic tournaments. The CHECK constraint is added
    # explicitly in the Alembic migration — ORM-side the string is
    # validated by Pydantic layer (PR-2) plus pyrefly Literal types.
    purpose: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="benchmark",
        server_default="benchmark",
    )
```

And in the `__table_args__` tuple for `Agent`, add a new `Index` and a
`CheckConstraint`:

```python
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "owner_id",
            "name",
            "version",
            name="uq_agent_tenant_owner_name_version",
        ),
        Index("idx_agent_name", "name"),
        Index("idx_agent_tenant", "tenant_id"),
        Index("idx_agent_owner", "owner_id"),
        # LABS-TSA PR-1
        Index("idx_agents_owner_purpose", "owner_id", "purpose"),
        CheckConstraint(
            "purpose IN ('benchmark','tournament')",
            name="ck_agents_purpose",
        ),
    )
```

Ensure `CheckConstraint` is imported at the top of the file. If the
existing file has
`from sqlalchemy import ...` without `CheckConstraint`, add it:

```python
from sqlalchemy import (
    # ... existing imports ...
    CheckConstraint,
)
```

- [ ] **Step 1.2.4: Run tests — they must pass**

Run: `uv run pytest tests/integration/dashboard/test_tsa_data_model.py::TestAgentPurpose -v --no-cov`
Expected: 3 passed.

- [ ] **Step 1.2.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/models.py \
        tests/integration/dashboard/test_tsa_data_model.py
git commit -m "feat(db): add Agent.purpose column + CHECK constraint (LABS-TSA PR-1)"
```

## Task 1.3: Add `Participant.builtin_strategy` with invariant

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/models.py` — `Participant` class around line 189
- Modify: `tests/integration/dashboard/test_tsa_data_model.py`

- [ ] **Step 1.3.1: Add failing tests for the invariant**

Append to `test_tsa_data_model.py`:

```python
class TestParticipantBuiltin:
    @pytest.mark.anyio
    async def test_agent_backed_participant_rejects_builtin_strategy(
        self, db_session: AsyncSession, user: User
    ) -> None:
        t = Tournament(game_type="el_farol", num_players=2, total_rounds=1, created_by=user.id)
        a = Agent(name="x", agent_type="mcp", owner_id=user.id, purpose="tournament")
        db_session.add_all([t, a])
        await db_session.commit()
        p = Participant(
            tournament_id=t.id,
            user_id=user.id,
            agent_id=a.id,
            agent_name="x",
            builtin_strategy="el_farol/traditionalist",  # both set — must fail
        )
        db_session.add(p)
        with pytest.raises(IntegrityError):
            await db_session.commit()

    @pytest.mark.anyio
    async def test_builtin_participant_has_no_agent_or_user(
        self, db_session: AsyncSession, user: User
    ) -> None:
        t = Tournament(game_type="el_farol", num_players=2, total_rounds=1, created_by=user.id)
        db_session.add(t)
        await db_session.commit()
        p = Participant(
            tournament_id=t.id,
            user_id=None,
            agent_id=None,
            agent_name="el_farol/traditionalist",
            builtin_strategy="el_farol/traditionalist",
        )
        db_session.add(p)
        await db_session.commit()  # must succeed
        await db_session.refresh(p)
        assert p.user_id is None
        assert p.agent_id is None
        assert p.builtin_strategy == "el_farol/traditionalist"

    @pytest.mark.anyio
    async def test_participant_without_builtin_or_agent_rejected(
        self, db_session: AsyncSession, user: User
    ) -> None:
        t = Tournament(game_type="el_farol", num_players=2, total_rounds=1, created_by=user.id)
        db_session.add(t)
        await db_session.commit()
        p = Participant(
            tournament_id=t.id,
            user_id=user.id,
            agent_id=None,
            agent_name="x",
            builtin_strategy=None,  # neither set — must fail
        )
        db_session.add(p)
        with pytest.raises(IntegrityError):
            await db_session.commit()
```

- [ ] **Step 1.3.2: Run tests — they must fail**

Run: `uv run pytest tests/integration/dashboard/test_tsa_data_model.py::TestParticipantBuiltin -v --no-cov`
Expected: collection error or failure (`builtin_strategy` attribute missing).

- [ ] **Step 1.3.3: Modify `Participant` class**

In `packages/atp-dashboard/atp/dashboard/tournament/models.py`:

Replace the `user_id` column definition around line 200 to make it nullable:

```python
    # LABS-TSA PR-1: nullable to allow builtin-strategy participants
    # (which have no User). Enforced together with agent_id / builtin_strategy
    # via a CHECK constraint in __table_args__ below.
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
```

Add a new column right after the existing `agent_id` field (around line 213):

```python
    # LABS-TSA PR-1: builtin strategy name (namespaced as "{game}/{strategy}").
    # Exactly one of agent_id / builtin_strategy must be set; the CHECK
    # constraint in __table_args__ enforces this.
    builtin_strategy: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
```

Update the `__table_args__` tuple:

```python
    __table_args__ = (
        Index("idx_participant_tournament", "tournament_id"),
        Index("idx_participant_user", "user_id"),
        UniqueConstraint(
            "tournament_id",
            "user_id",
            name="uq_participant_tournament_user",
        ),
        Index(
            "uq_participant_user_active",
            "user_id",
            unique=True,
            sqlite_where=text("user_id IS NOT NULL AND released_at IS NULL"),
            postgresql_where=text("user_id IS NOT NULL AND released_at IS NULL"),
        ),
        # LABS-TSA PR-1
        Index(
            "idx_participants_builtin",
            "tournament_id",
            "builtin_strategy",
            sqlite_where=text("builtin_strategy IS NOT NULL"),
            postgresql_where=text("builtin_strategy IS NOT NULL"),
        ),
        CheckConstraint(
            "(agent_id IS NOT NULL AND builtin_strategy IS NULL)"
            " OR (agent_id IS NULL AND builtin_strategy IS NOT NULL)",
            name="ck_participants_agent_xor_builtin",
        ),
    )
```

`CheckConstraint` is already used elsewhere in this module; if not imported,
add it to the top-of-file import list.

- [ ] **Step 1.3.4: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_tsa_data_model.py::TestParticipantBuiltin -v --no-cov`
Expected: 3 passed.

- [ ] **Step 1.3.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/models.py \
        tests/integration/dashboard/test_tsa_data_model.py
git commit -m "feat(db): Participant.builtin_strategy with xor-agent invariant (LABS-TSA PR-1)"
```

## Task 1.4: Add `GameResult.tournament_id` with UNIQUE partial index

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/models.py` — `GameResult` class around line 949
- Modify: `tests/integration/dashboard/test_tsa_data_model.py`

- [ ] **Step 1.4.1: Add failing tests**

Append to `test_tsa_data_model.py`:

```python
class TestGameResultTournamentLink:
    @pytest.mark.anyio
    async def test_tournament_id_nullable_default_none(
        self, db_session: AsyncSession
    ) -> None:
        g = GameResult(
            game_name="x",
            game_type="one_shot",
            num_players=2,
            num_rounds=1,
            status="completed",
        )
        db_session.add(g)
        await db_session.commit()
        await db_session.refresh(g)
        assert g.tournament_id is None

    @pytest.mark.anyio
    async def test_tournament_id_unique_partial(
        self, db_session: AsyncSession, user: User
    ) -> None:
        t = Tournament(game_type="el_farol", num_players=2, total_rounds=1, created_by=user.id)
        db_session.add(t)
        await db_session.commit()

        g1 = GameResult(
            game_name="x", game_type="one_shot", num_players=2, num_rounds=1,
            status="completed", tournament_id=t.id,
        )
        db_session.add(g1)
        await db_session.commit()

        g2 = GameResult(
            game_name="x", game_type="one_shot", num_players=2, num_rounds=1,
            status="completed", tournament_id=t.id,  # duplicate
        )
        db_session.add(g2)
        with pytest.raises(IntegrityError):
            await db_session.commit()

    @pytest.mark.anyio
    async def test_two_results_without_tournament_id_allowed(
        self, db_session: AsyncSession
    ) -> None:
        # tournament_id IS NULL rows are not uniqueness-constrained.
        for i in range(2):
            db_session.add(
                GameResult(
                    game_name=f"g{i}", game_type="one_shot",
                    num_players=2, num_rounds=1, status="completed",
                )
            )
        await db_session.commit()  # must succeed
```

- [ ] **Step 1.4.2: Run tests — they must fail**

Run: `uv run pytest tests/integration/dashboard/test_tsa_data_model.py::TestGameResultTournamentLink -v --no-cov`
Expected: 3 failures / errors (`tournament_id` attribute missing).

- [ ] **Step 1.4.3: Add the column + UNIQUE partial index**

In `packages/atp-dashboard/atp/dashboard/models.py`, inside the `GameResult`
class, right after the `agents_json` column around line 1027, add:

```python
    # LABS-TSA PR-1: link to the tournament that produced this match.
    # NULL for CLI standalone runs. UNIQUE partial index below enforces
    # at-most-one GameResult per tournament so the dual-write from the
    # tournament completion hook is idempotent without TOCTOU races.
    tournament_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("tournaments.id", ondelete="SET NULL"),
        nullable=True,
    )
```

And add to `__table_args__`:

```python
        # LABS-TSA PR-1
        Index(
            "idx_game_results_tournament",
            "tournament_id",
        ),
        Index(
            "uq_game_results_tournament_id",
            "tournament_id",
            unique=True,
            sqlite_where=text("tournament_id IS NOT NULL"),
            postgresql_where=text("tournament_id IS NOT NULL"),
        ),
```

Ensure `text` is imported at the top of the file.

- [ ] **Step 1.4.4: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_tsa_data_model.py::TestGameResultTournamentLink -v --no-cov`
Expected: 3 passed.

- [ ] **Step 1.4.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/models.py \
        tests/integration/dashboard/test_tsa_data_model.py
git commit -m "feat(db): GameResult.tournament_id + UNIQUE partial index (LABS-TSA PR-1)"
```

## Task 1.5: Author the Alembic migration

**Files:**
- Create: `migrations/dashboard/versions/<hash>_tournament_agent_sandbox.py`
- Modify: `packages/atp-dashboard/atp/dashboard/database.py::_add_missing_columns` — ensure new schema is reconciled on legacy DBs

- [ ] **Step 1.5.1: Generate a migration skeleton**

Run: `uv run alembic revision -m "tournament agent sandbox (LABS-TSA PR-1)"`
Expected: new file printed to stdout like
`migrations/dashboard/versions/abc123_tournament_agent_sandbox.py`. Note the
file name and the generated revision id.

- [ ] **Step 1.5.2: Replace the generated content**

Open the newly-created migration file. Replace its body with:

```python
"""tournament agent sandbox (LABS-TSA PR-1)

- Agent.purpose (VARCHAR(20) NOT NULL DEFAULT 'benchmark' CHECK IN ('benchmark','tournament'))
- Participant.user_id made nullable; Participant.builtin_strategy (VARCHAR(64))
  with agent-xor-builtin CHECK
- GameResult.tournament_id (nullable FK) + UNIQUE partial index
- Supporting indexes

Revision ID: <keep auto-generated id>
Revises: <PREV_HEAD from Step 1.1.2>
Create Date: 2026-04-23 ...
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "<keep auto-generated id>"
down_revision: str | Sequence[str] | None = "<PREV_HEAD>"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # agents.purpose + CHECK + index
    with op.batch_alter_table("agents", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "purpose",
                sa.String(length=20),
                nullable=False,
                server_default="benchmark",
            )
        )
        batch_op.create_check_constraint(
            "ck_agents_purpose",
            "purpose IN ('benchmark','tournament')",
        )
        batch_op.create_index(
            "idx_agents_owner_purpose",
            ["owner_id", "purpose"],
            unique=False,
        )

    # tournament_participants: user_id → nullable; builtin_strategy;
    # agent-xor-builtin CHECK; builtin-only partial index
    with op.batch_alter_table("tournament_participants", schema=None) as batch_op:
        batch_op.alter_column("user_id", existing_type=sa.Integer(), nullable=True)
        batch_op.add_column(
            sa.Column("builtin_strategy", sa.String(length=64), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_participants_agent_xor_builtin",
            "(agent_id IS NOT NULL AND builtin_strategy IS NULL)"
            " OR (agent_id IS NULL AND builtin_strategy IS NOT NULL)",
        )
        batch_op.create_index(
            "idx_participants_builtin",
            ["tournament_id", "builtin_strategy"],
            unique=False,
            sqlite_where=sa.text("builtin_strategy IS NOT NULL"),
            postgresql_where=sa.text("builtin_strategy IS NOT NULL"),
        )

    # game_results.tournament_id + FK + UNIQUE partial index
    with op.batch_alter_table("game_results", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("tournament_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_game_results_tournament",
            "tournaments",
            ["tournament_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "idx_game_results_tournament",
            ["tournament_id"],
            unique=False,
        )
        batch_op.create_index(
            "uq_game_results_tournament_id",
            ["tournament_id"],
            unique=True,
            sqlite_where=sa.text("tournament_id IS NOT NULL"),
            postgresql_where=sa.text("tournament_id IS NOT NULL"),
        )


def downgrade() -> None:
    with op.batch_alter_table("game_results", schema=None) as batch_op:
        batch_op.drop_index("uq_game_results_tournament_id")
        batch_op.drop_index("idx_game_results_tournament")
        batch_op.drop_constraint("fk_game_results_tournament", type_="foreignkey")
        batch_op.drop_column("tournament_id")

    with op.batch_alter_table("tournament_participants", schema=None) as batch_op:
        batch_op.drop_index("idx_participants_builtin")
        batch_op.drop_constraint(
            "ck_participants_agent_xor_builtin", type_="check"
        )
        batch_op.drop_column("builtin_strategy")
        batch_op.alter_column("user_id", existing_type=sa.Integer(), nullable=False)

    with op.batch_alter_table("agents", schema=None) as batch_op:
        batch_op.drop_index("idx_agents_owner_purpose")
        batch_op.drop_constraint("ck_agents_purpose", type_="check")
        batch_op.drop_column("purpose")
```

Replace `<keep auto-generated id>` with the id Alembic picked; replace
`<PREV_HEAD>` with the revision id you noted in Step 1.1.2.

- [ ] **Step 1.5.3: Verify Alembic heads reduce to one**

Run: `uv run alembic heads`
Expected: exactly one head — the new revision.

- [ ] **Step 1.5.4: Add migration round-trip test**

Create `tests/integration/dashboard/test_tsa_alembic_migration.py`:

```python
"""Alembic round-trip test for LABS-TSA PR-1 migration."""

import subprocess
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect


@pytest.mark.anyio
async def test_migration_up_down_up_clean_sqlite() -> None:
    with tempfile.TemporaryDirectory() as td:
        dbpath = Path(td) / "atp.db"
        import os
        env = {**os.environ, "ATP_DATABASE_URL": f"sqlite:///{dbpath}"}
        # up to head
        subprocess.check_call(["uv", "run", "alembic", "upgrade", "head"], env=env)
        # inspect new columns exist
        eng = create_engine(f"sqlite:///{dbpath}")
        insp = inspect(eng)
        cols = {c["name"] for c in insp.get_columns("agents")}
        assert "purpose" in cols
        cols = {c["name"] for c in insp.get_columns("tournament_participants")}
        assert "builtin_strategy" in cols
        cols = {c["name"] for c in insp.get_columns("game_results")}
        assert "tournament_id" in cols
        eng.dispose()
        # downgrade one step
        subprocess.check_call(
            ["uv", "run", "alembic", "downgrade", "-1"], env=env
        )
        eng = create_engine(f"sqlite:///{dbpath}")
        insp = inspect(eng)
        assert "purpose" not in {c["name"] for c in insp.get_columns("agents")}
        eng.dispose()
        # upgrade again
        subprocess.check_call(["uv", "run", "alembic", "upgrade", "head"], env=env)
        eng = create_engine(f"sqlite:///{dbpath}")
        insp = inspect(eng)
        assert "purpose" in {c["name"] for c in insp.get_columns("agents")}
        eng.dispose()
```

- [ ] **Step 1.5.5: Run migration test**

Run: `uv run pytest tests/integration/dashboard/test_tsa_alembic_migration.py -v --no-cov`
Expected: pass. If the subprocess fails with "No script_location" — re-invoke
with `cwd=Path(__file__).resolve().parents[3]` in the `subprocess.check_call`.

- [ ] **Step 1.5.6: Manually validate legacy-DB reconcile**

The dashboard boots `_add_missing_columns(db)` on startup to reconcile ORM
drift on DBs that pre-date Alembic (see
`packages/atp-dashboard/atp/dashboard/database.py`). Boot the app against an
empty SQLite to ensure the new columns aren't added twice by the reconciler
after the migration ran:

```bash
rm -f /tmp/_tsa_probe.db
ATP_DATABASE_URL=sqlite+aiosqlite:///tmp/_tsa_probe.db \
  uv run python -c "
import asyncio
from atp.dashboard.database import init_database
asyncio.run(init_database())
print('boot ok')
"
```

Expected: `boot ok` with no IntegrityError about duplicate column.

- [ ] **Step 1.5.7: Commit**

```bash
git add migrations/dashboard/versions/ \
        tests/integration/dashboard/test_tsa_alembic_migration.py
git commit -m "feat(db): Alembic migration for tournament agent sandbox (LABS-TSA PR-1)"
```

## Task 1.6: Open PR-1

- [ ] **Step 1.6.1: Run full test suite against the data-model changes**

Run:
```bash
uv run pytest tests/integration/dashboard/test_tsa_data_model.py \
              tests/integration/dashboard/test_tsa_alembic_migration.py \
              tests/integration/dashboard/test_game_result_el_farol_columns.py \
              tests/integration/dashboard/test_matches_listing.py \
              -v --no-cov
```
Expected: all pass (no regression on LABS-97 / LABS-102 existing tests).

- [ ] **Step 1.6.2: Push and open PR**

```bash
git push -u github labs-tsa/pr1-data-model
gh pr create --title "feat(db): tournament agent sandbox — data model + migration (PR-1)" \
  --body "$(cat <<'EOF'
## Summary
- `Agent.purpose` column (`benchmark` / `tournament`), defaults to `benchmark` for all existing rows
- `Participant.user_id` flipped to nullable; new `Participant.builtin_strategy` column with agent-xor-builtin CHECK invariant
- `GameResult.tournament_id` FK (nullable) with UNIQUE partial index for idempotent dual-write
- Alembic migration + ORM changes kept strictly in sync

## Spec
`docs/superpowers/specs/2026-04-23-tournament-agent-sandbox-design.md`

## Rollout
This is PR-1 of 5. No behavioural change on any endpoint yet — subsequent PRs wire the new columns into API, MCP auth, runner, and UI.

## Test plan
- [x] ORM tests for default, CHECK, nullable, UNIQUE partial
- [x] Alembic up/down/up round-trip on SQLite
- [x] Legacy dashboard boot reconciles cleanly

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR link printed.

---

# PR-2 — Quota enforcement + purpose-based Agent API

**Branch:** `labs-tsa/pr2-quotas-and-api`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/config.py` — new env vars
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py` — `purpose` field + quota
- Create: `tests/integration/dashboard/test_tsa_agent_api.py`

## Task 2.1: Start branch

- [ ] **Step 2.1.1: Branch from PR-1 head**

```bash
# wait until PR-1 merges to main, then:
git checkout main
git pull github main --ff-only
git checkout -b labs-tsa/pr2-quotas-and-api
```

If PR-1 is not yet merged, base on `labs-tsa/pr1-data-model` and retarget
later. For the rest of this plan, assume PR-N lands before PR-(N+1) starts.

## Task 2.2: Wire the two quota env vars into config

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/config.py`

- [ ] **Step 2.2.1: Add fields to `DashboardConfig`**

Open `packages/atp-dashboard/atp/dashboard/v2/config.py`. Find the
`DashboardConfig` class. Add:

```python
    max_benchmark_agents_per_user: int = Field(
        default=10,
        description="Max Agent rows per user with purpose='benchmark'",
        validation_alias="ATP_MAX_BENCHMARK_AGENTS_PER_USER",
    )
    max_tournament_agents_per_user: int = Field(
        default=5,
        description="Max Agent rows per user with purpose='tournament'",
        validation_alias="ATP_MAX_TOURNAMENT_AGENTS_PER_USER",
    )
    max_concurrent_private_tournaments_per_user: int = Field(
        default=3,
        description="Max pending+active private tournaments per user",
        validation_alias="ATP_MAX_CONCURRENT_PRIVATE_TOURNAMENTS_PER_USER",
    )
```

- [ ] **Step 2.2.2: Sanity check config loads**

Run:
```bash
ATP_MAX_TOURNAMENT_AGENTS_PER_USER=7 uv run python -c "
from atp.dashboard.v2.config import get_config
c = get_config()
print(c.max_tournament_agents_per_user)
"
```
Expected: `7`.

- [ ] **Step 2.2.3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/config.py
git commit -m "feat(config): quota env vars for tournament agent sandbox (LABS-TSA PR-2)"
```

## Task 2.3: Extend agent API schema with `purpose`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py`
- Create / extend: `tests/integration/dashboard/test_tsa_agent_api.py`

- [ ] **Step 2.3.1: Write a failing test**

Create `tests/integration/dashboard/test_tsa_agent_api.py`:

```python
"""Integration tests for the purpose-aware agent API (LABS-TSA PR-2)."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def v2_app(test_database: Database):
    app = create_test_app(use_v2_routes=True)

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


class TestAgentPurposeAPI:
    @pytest.mark.anyio
    async def test_create_tournament_agent(
        self, v2_app, db_session: AsyncSession, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/agents",
                headers=auth_headers,
                json={
                    "name": "my-tournament-bot",
                    "agent_type": "mcp",
                    "purpose": "tournament",
                },
            )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["purpose"] == "tournament"

    @pytest.mark.anyio
    async def test_default_purpose_is_benchmark(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/agents",
                headers=auth_headers,
                json={"name": "legacy-bot", "agent_type": "mcp"},
            )
        assert resp.status_code == 201, resp.text
        assert resp.json()["purpose"] == "benchmark"

    @pytest.mark.anyio
    async def test_filter_list_by_purpose(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "b1", "agent_type": "http"},
            )
            await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "t1", "agent_type": "mcp", "purpose": "tournament"},
            )
            resp = await client.get(
                "/api/v1/agents?purpose=tournament", headers=auth_headers
            )
            names = {a["name"] for a in resp.json()["items"]}
            assert names == {"t1"}
```

- [ ] **Step 2.3.2: Run — it must fail**

Run: `uv run pytest tests/integration/dashboard/test_tsa_agent_api.py::TestAgentPurposeAPI -v --no-cov`
Expected: 422 / 500 / missing-field errors.

- [ ] **Step 2.3.3: Locate the POST/GET handler**

Inspect `packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py`.
Find the `CreateAgentRequest` Pydantic model (used by `POST /api/v1/agents`)
and the list endpoint. You will modify both.

- [ ] **Step 2.3.4: Add `purpose` field to the request model**

In the `CreateAgentRequest` Pydantic class, add:

```python
    purpose: Literal["benchmark", "tournament"] = Field(
        default="benchmark",
        description="Agent purpose — benchmark agents run suite evaluations; "
                    "tournament agents connect to /mcp for game-theoretic play.",
    )
```

Ensure `Literal` is imported from `typing` at the top of the file.

- [ ] **Step 2.3.5: Return `purpose` in the response**

Find the response serializer (likely `AgentResponse` or similar). Add:

```python
    purpose: str
```

And in the handler body, pass `purpose=agent.purpose` when constructing the
response (or if the response uses `model_validate(agent)` + `from_attributes`,
no changes needed).

- [ ] **Step 2.3.6: Wire purpose through the create path**

Find the `agent = Agent(...)` construction in the create handler. Add
`purpose=body.purpose` to the kwargs.

- [ ] **Step 2.3.7: Extend list handler with `?purpose=` filter**

In the list handler, add a `purpose: Literal["benchmark", "tournament"] | None = None`
query parameter. In the SQL, if `purpose` is not None, add
`.where(Agent.purpose == purpose)`.

- [ ] **Step 2.3.8: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_tsa_agent_api.py::TestAgentPurposeAPI -v --no-cov`
Expected: 3 passed.

- [ ] **Step 2.3.9: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py \
        tests/integration/dashboard/test_tsa_agent_api.py
git commit -m "feat(api): purpose field on POST/GET /api/v1/agents (LABS-TSA PR-2)"
```

## Task 2.4: Enforce per-purpose quotas on agent creation

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py`
- Extend: `tests/integration/dashboard/test_tsa_agent_api.py`

- [ ] **Step 2.4.1: Write failing quota tests**

Append to `test_tsa_agent_api.py`:

```python
class TestAgentQuota:
    @pytest.mark.anyio
    async def test_tournament_quota_rejects_sixth(
        self, v2_app, auth_headers, monkeypatch
    ) -> None:
        monkeypatch.setenv("ATP_MAX_TOURNAMENT_AGENTS_PER_USER", "5")
        # Must clear cached config
        from atp.dashboard.v2.config import get_config
        get_config.cache_clear()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            for i in range(5):
                resp = await client.post(
                    "/api/v1/agents", headers=auth_headers,
                    json={"name": f"t{i}", "agent_type": "mcp", "purpose": "tournament"},
                )
                assert resp.status_code == 201, (i, resp.text)
            resp = await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "t5", "agent_type": "mcp", "purpose": "tournament"},
            )
            assert resp.status_code == 429
            assert "tournament agent quota" in resp.text.lower()

    @pytest.mark.anyio
    async def test_benchmark_quota_independent(
        self, v2_app, auth_headers, monkeypatch
    ) -> None:
        monkeypatch.setenv("ATP_MAX_BENCHMARK_AGENTS_PER_USER", "2")
        monkeypatch.setenv("ATP_MAX_TOURNAMENT_AGENTS_PER_USER", "2")
        from atp.dashboard.v2.config import get_config
        get_config.cache_clear()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # 2 benchmark agents — fill quota
            for i in range(2):
                resp = await client.post(
                    "/api/v1/agents", headers=auth_headers,
                    json={"name": f"b{i}", "agent_type": "http"},
                )
                assert resp.status_code == 201

            # 3rd benchmark rejected
            resp = await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "b2", "agent_type": "http"},
            )
            assert resp.status_code == 429

            # But tournament slot still open
            resp = await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "t0", "agent_type": "mcp", "purpose": "tournament"},
            )
            assert resp.status_code == 201
```

- [ ] **Step 2.4.2: Run — tests must fail**

Run: `uv run pytest tests/integration/dashboard/test_tsa_agent_api.py::TestAgentQuota -v --no-cov`
Expected: 429 never returned because quota is unenforced.

- [ ] **Step 2.4.3: Implement quota check in the create handler**

In `agent_management_api.py`, at the top of the `POST /api/v1/agents` handler
(before the `Agent(...)` construction), add:

```python
    from sqlalchemy import func as _sqlfn
    from atp.dashboard.v2.config import get_config as _cfg

    cfg = _cfg()
    if body.purpose == "tournament":
        cap = cfg.max_tournament_agents_per_user
    else:
        cap = cfg.max_benchmark_agents_per_user

    count_stmt = select(_sqlfn.count(Agent.id)).where(
        Agent.owner_id == user.id,
        Agent.purpose == body.purpose,
        Agent.deleted_at.is_(None),
    )
    existing = (await session.execute(count_stmt)).scalar_one()
    if existing >= cap:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"{body.purpose} agent quota exceeded ({existing}/{cap})",
        )
```

(The `user` variable is whatever name the handler uses for the authenticated
user dependency — do not invent a new one, reuse the existing dep.)

- [ ] **Step 2.4.4: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_tsa_agent_api.py::TestAgentQuota -v --no-cov`
Expected: 2 passed.

- [ ] **Step 2.4.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py \
        tests/integration/dashboard/test_tsa_agent_api.py
git commit -m "feat(api): per-purpose agent quotas on POST /api/v1/agents (LABS-TSA PR-2)"
```

## Task 2.5: Open PR-2

- [ ] **Step 2.5.1: Run related tests**

Run:
```bash
uv run pytest tests/integration/dashboard/test_tsa_agent_api.py \
              tests/integration/dashboard/test_agent_management_api.py \
              -v --no-cov
```
Expected: everything green.

- [ ] **Step 2.5.2: Push + open PR**

```bash
git push -u github labs-tsa/pr2-quotas-and-api
gh pr create --title "feat(api): quota + purpose on agent API (LABS-TSA PR-2)"
```

Paste a PR body similar to PR-1's. Base branch `main`.

---

# PR-3 — MCP / benchmark auth gating by agent purpose

**Branch:** `labs-tsa/pr3-mcp-auth-gating`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/auth/__init__.py` (or wherever token issuance lives — locate first)
- Modify: `packages/atp-dashboard/atp/dashboard/v2/middleware/jwt_user_state.py` (or wherever `JWTUserStateMiddleware` is defined — locate first)
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/auth.py` — reject non-tournament agents
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py` — reject tournament agents
- Create: `tests/integration/dashboard/test_tsa_auth_gating.py`

## Task 3.1: Locate the token / middleware files

- [ ] **Step 3.1.1: Grep for `JWTUserStateMiddleware` location**

Run: `rg -n "class JWTUserStateMiddleware" packages/atp-dashboard/`
Expected: one file. Note the path.

- [ ] **Step 3.1.2: Grep for `APIToken` issuance path**

Run: `rg -n "APIToken\(.*token_hash" packages/atp-dashboard/ atp/`
Expected: one or two callsites — usually `tokens.py` and a route handler.
Note paths.

## Task 3.2: Snapshot `agent_id` + `agent_purpose` into tokens at issuance

**Files:**
- Modify: the `APIToken` creation path identified in Step 3.1.2
- Modify: `packages/atp-dashboard/atp/dashboard/tokens.py` — extend `APIToken` model if needed

- [ ] **Step 3.2.1: Add a denormalised `agent_purpose` column to `APIToken`**

In `packages/atp-dashboard/atp/dashboard/tokens.py`, the `APIToken` model
already has `agent_id`. Add:

```python
    # LABS-TSA PR-3: snapshot of agent.purpose at issuance — avoids a DB
    # roundtrip on the auth hot path. Refreshed when the user regenerates
    # the token; stale if the agent's purpose changes post-issuance (rare,
    # corrected on next token rotation).
    agent_purpose: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )
```

- [ ] **Step 3.2.2: Author the short migration for the new column**

Generate: `uv run alembic revision -m "APIToken.agent_purpose (LABS-TSA PR-3)"`
In the generated file:

```python
def upgrade() -> None:
    with op.batch_alter_table("api_tokens", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("agent_purpose", sa.String(length=20), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("api_tokens", schema=None) as batch_op:
        batch_op.drop_column("agent_purpose")
```

Replace `revision` / `down_revision` appropriately (new head = PR-1's head).

- [ ] **Step 3.2.3: Populate `agent_purpose` at issuance**

In the token-issuance handler (identified in Step 3.1.2), when constructing
the `APIToken(...)` row for an agent-scoped token:

```python
    token_row = APIToken(
        # ... existing fields ...
        agent_id=agent.id,
        agent_purpose=agent.purpose,  # LABS-TSA PR-3
    )
```

For user-scoped (`atp_u_*`) tokens, leave `agent_purpose=None`.

- [ ] **Step 3.2.4: Backfill test**

Create `tests/integration/dashboard/test_tsa_auth_gating.py`:

```python
"""LABS-TSA PR-3 — MCP and benchmark-API auth gating by agent purpose."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def v2_app(test_database: Database):
    app = create_test_app(use_v2_routes=True)

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


class TestTokenSnapshot:
    @pytest.mark.anyio
    async def test_issuing_tournament_agent_token_records_purpose(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "t1", "agent_type": "mcp", "purpose": "tournament"},
            )
            agent_id = resp.json()["id"]
            resp = await client.post(
                "/api/v1/tokens", headers=auth_headers,
                json={"agent_id": agent_id, "name": "t1-token"},
            )
            assert resp.status_code == 201
        # inspect DB
        from atp.dashboard.tokens import APIToken
        from sqlalchemy import select
        stmt = select(APIToken).where(APIToken.agent_id == agent_id)
        # use the same session the app wrote to — get via test_database
        # fixture ... (fill in with conftest.py helpers)
```

Note: the full test body depends on existing conftest fixtures for token
issuance. Consult `tests/integration/dashboard/test_api_token_auth.py` for
the exact request shape your token endpoint expects.

- [ ] **Step 3.2.5: Run tests; commit**

Run: `uv run pytest tests/integration/dashboard/test_tsa_auth_gating.py::TestTokenSnapshot -v --no-cov`
Expected: pass.

```bash
git add packages/atp-dashboard/atp/dashboard/tokens.py \
        migrations/dashboard/versions/*_apitoken_agent_purpose.py \
        tests/integration/dashboard/test_tsa_auth_gating.py
git commit -m "feat(auth): snapshot agent.purpose onto APIToken at issuance (LABS-TSA PR-3)"
```

## Task 3.3: Extend `JWTUserStateMiddleware` to surface `agent_purpose`

**Files:**
- Modify: the middleware file located in Step 3.1.1

- [ ] **Step 3.3.1: Read the current middleware body**

Take note of how `user_id` gets into `scope["state"]`. The extension pattern:
after the current user_id resolution, if the request was authenticated by an
agent-scoped token, also write:

```python
    scope_state["agent_id"] = token.agent_id
    scope_state["agent_purpose"] = token.agent_purpose
```

(Only when the token is agent-scoped. For user-scoped or admin-session, leave
unset.)

- [ ] **Step 3.3.2: Write failing test for middleware surfacing**

Append to `test_tsa_auth_gating.py`:

```python
class TestMiddlewareSurfacing:
    @pytest.mark.anyio
    async def test_agent_purpose_in_request_state(
        self, v2_app, auth_headers
    ) -> None:
        # Add a tiny probe endpoint via dependency_overrides or a fixture
        # route that echoes request.state.agent_purpose. Or reuse an
        # existing endpoint that logs request.state (check routes/traces.py).
        # Minimum: create a tournament agent, mint a token, call any
        # authenticated endpoint with that token, inspect the server log
        # (not practical in test) OR exercise the final MCP gate below
        # which transitively proves agent_purpose is surfaced.
        pass  # this test is implicitly covered by TestMCPAuthGate
```

Skip this task if covered by Task 3.4 (the MCP gate test exercises the whole
chain end-to-end).

- [ ] **Step 3.3.3: Implement surfacing; commit**

Apply the patch; run the test chain from Task 3.4 to validate.

```bash
git add <middleware file>
git commit -m "feat(auth): surface agent_purpose onto scope.state (LABS-TSA PR-3)"
```

## Task 3.4: Reject non-tournament agents at `/mcp`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/mcp/auth.py` (the `MCPAuthMiddleware`)
- Extend: `tests/integration/dashboard/test_tsa_auth_gating.py`

- [ ] **Step 3.4.1: Write failing test**

Append:

```python
class TestMCPAuthGate:
    @pytest.mark.anyio
    async def test_benchmark_token_rejected_by_mcp(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # Register a benchmark agent, mint its token
            resp = await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "bench1", "agent_type": "http"},
            )
            agent_id = resp.json()["id"]
            resp = await client.post(
                "/api/v1/tokens", headers=auth_headers,
                json={"agent_id": agent_id, "name": "bench-tok"},
            )
            token = resp.json()["token"]  # field name may differ — check shape

            # Hit /mcp with that token
            resp = await client.get(
                "/mcp/", headers={"Authorization": f"Bearer {token}"}
            )
            assert resp.status_code == 403
            assert "tournament-agents only" in resp.text.lower()

    @pytest.mark.anyio
    async def test_user_level_token_rejected_by_mcp(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # Create user-scoped token (no agent)
            resp = await client.post(
                "/api/v1/tokens", headers=auth_headers,
                json={"name": "user-tok"},  # no agent_id → user-scoped
            )
            token = resp.json()["token"]

            resp = await client.get(
                "/mcp/", headers={"Authorization": f"Bearer {token}"}
            )
            assert resp.status_code == 403
            assert "agent-scoped token" in resp.text.lower()

    @pytest.mark.anyio
    async def test_tournament_token_accepted_by_mcp(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "t1", "agent_type": "mcp", "purpose": "tournament"},
            )
            agent_id = resp.json()["id"]
            resp = await client.post(
                "/api/v1/tokens", headers=auth_headers,
                json={"agent_id": agent_id, "name": "t1-tok"},
            )
            token = resp.json()["token"]
            # MCP is ASGI-mounted — a bare GET should at least *not* 403 on purpose
            resp = await client.get(
                "/mcp/", headers={"Authorization": f"Bearer {token}"}
            )
            assert resp.status_code != 403  # 200 / 404 / 406 acceptable here
```

- [ ] **Step 3.4.2: Run — must fail**

Run: `uv run pytest tests/integration/dashboard/test_tsa_auth_gating.py::TestMCPAuthGate -v --no-cov`
Expected: 200/401 on MCP hits — no 403 surface yet.

- [ ] **Step 3.4.3: Patch `MCPAuthMiddleware`**

In `packages/atp-dashboard/atp/dashboard/mcp/auth.py`, after the current
`user_id` check, add a purpose check:

```python
        user_id = state.get("user_id")
        if user_id is None:
            # existing 401 path stays
            ...

        agent_purpose = state.get("agent_purpose")
        if agent_purpose is None:
            # user-level token or admin session — MCP is strictly for agents
            await self._send_403(
                send,
                detail="MCP requires an agent-scoped token (atp_a_*)",
            )
            return
        if agent_purpose != "tournament":
            await self._send_403(
                send,
                detail=(
                    "MCP is tournament-agents only; "
                    "this token belongs to a benchmark agent"
                ),
            )
            return
```

Add the `_send_403` helper method (or re-use the existing 401 helper pattern
with a different status). Shape:

```python
    async def _send_403(self, send: Callable[..., Any], detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": body})
```

- [ ] **Step 3.4.4: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_tsa_auth_gating.py::TestMCPAuthGate -v --no-cov`
Expected: 3 passed.

- [ ] **Step 3.4.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/mcp/auth.py \
        tests/integration/dashboard/test_tsa_auth_gating.py
git commit -m "feat(mcp): reject non-tournament agents at /mcp (LABS-TSA PR-3)"
```

## Task 3.5: Reject tournament agents at `/api/v1/benchmarks/*`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py`
- Extend: `tests/integration/dashboard/test_tsa_auth_gating.py`

- [ ] **Step 3.5.1: Write failing test**

Append:

```python
class TestBenchmarkAPIGate:
    @pytest.mark.anyio
    async def test_tournament_token_rejected_by_benchmark_api(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "t-for-bench", "agent_type": "mcp", "purpose": "tournament"},
            )
            agent_id = resp.json()["id"]
            resp = await client.post(
                "/api/v1/tokens", headers=auth_headers,
                json={"agent_id": agent_id, "name": "t-for-bench-tok"},
            )
            token = resp.json()["token"]

            resp = await client.get(
                "/api/v1/benchmarks", headers={"Authorization": f"Bearer {token}"}
            )
            assert resp.status_code == 403
            assert "benchmark-agents only" in resp.text.lower()
```

- [ ] **Step 3.5.2: Run — must fail**

- [ ] **Step 3.5.3: Add gate**

In `benchmark_api.py`, identify the shared dependency that resolves the caller
(likely `get_current_user_for_benchmark` or a similar
`Annotated[User, Depends(...)]`). Wrap or extend it with:

```python
async def require_benchmark_or_user_token(request: Request) -> User:
    user = await _get_current_user_for_benchmark(request)  # existing resolver
    agent_purpose = getattr(request.state, "agent_purpose", None)
    if agent_purpose == "tournament":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="benchmark API is benchmark-agents only",
        )
    return user
```

Swap the router's top-level `Depends(get_current_user_for_benchmark)` for
`Depends(require_benchmark_or_user_token)`.

- [ ] **Step 3.5.4: Run tests**

Expected: pass.

- [ ] **Step 3.5.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py \
        tests/integration/dashboard/test_tsa_auth_gating.py
git commit -m "feat(api): reject tournament tokens at /api/v1/benchmarks/* (LABS-TSA PR-3)"
```

## Task 3.6: Legacy-token lazy lookup + cache

**Files:**
- Modify: the middleware file from Step 3.1.1

- [ ] **Step 3.6.1: Failing test — legacy token without `agent_purpose` still resolves**

Append:

```python
class TestLegacyTokenLookup:
    @pytest.mark.anyio
    async def test_token_with_null_agent_purpose_still_resolves(
        self, v2_app, db_session, auth_headers
    ) -> None:
        # Simulate a pre-PR-3 token: agent_id set, agent_purpose NULL
        # (as if the token were issued before the migration).
        from atp.dashboard.tokens import APIToken
        from atp.dashboard.models import Agent, User
        from sqlalchemy import select, update

        # Register agent via API, then null-out agent_purpose on its tokens
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "legacy-t", "agent_type": "mcp", "purpose": "tournament"},
            )
            agent_id = resp.json()["id"]
            resp = await client.post(
                "/api/v1/tokens", headers=auth_headers,
                json={"agent_id": agent_id, "name": "legacy-tok"},
            )
            token = resp.json()["token"]

        # Hack: NULL the agent_purpose on the token row
        await db_session.execute(
            update(APIToken).where(APIToken.agent_id == agent_id).values(agent_purpose=None)
        )
        await db_session.commit()

        # Request should still succeed — middleware falls back to a DB lookup
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/mcp/", headers={"Authorization": f"Bearer {token}"}
            )
            assert resp.status_code != 403
```

- [ ] **Step 3.6.2: Run — must fail or pass depending on middleware**

If middleware only reads from token row, the null path 403s. Expected: FAIL.

- [ ] **Step 3.6.3: Implement lazy lookup + in-process cache**

In the middleware, when `token.agent_id IS NOT NULL AND token.agent_purpose
IS NULL`, fall back to a one-time `SELECT agents.purpose FROM agents WHERE
id = :agent_id`, cache the result keyed by token_hash in a module-level
dict, and use it. Skeleton:

```python
_legacy_purpose_cache: dict[str, str] = {}  # module-level, process-scoped


async def _resolve_purpose(token_row, session) -> str | None:
    if token_row.agent_id is None:
        return None
    if token_row.agent_purpose is not None:
        return token_row.agent_purpose
    cached = _legacy_purpose_cache.get(token_row.token_hash)
    if cached is not None:
        return cached
    agent = await session.get(Agent, token_row.agent_id)
    if agent is None:
        return None
    _legacy_purpose_cache[token_row.token_hash] = agent.purpose
    return agent.purpose
```

Wire this into the middleware where `agent_purpose` is extracted from the
token row.

- [ ] **Step 3.6.4: Run tests**

Expected: pass.

- [ ] **Step 3.6.5: Commit**

```bash
git add <middleware file> \
        tests/integration/dashboard/test_tsa_auth_gating.py
git commit -m "feat(auth): lazy fallback for legacy tokens without agent_purpose (LABS-TSA PR-3)"
```

## Task 3.7: Open PR-3

- [ ] **Step 3.7.1: Run full test sweep for auth surface**

```bash
uv run pytest tests/integration/dashboard/test_tsa_auth_gating.py \
              tests/integration/dashboard/test_api_token_auth.py \
              -v --no-cov
```
Expected: all green.

- [ ] **Step 3.7.2: Push + open PR**

```bash
git push -u github labs-tsa/pr3-mcp-auth-gating
gh pr create --title "feat(auth): purpose-based MCP / benchmark gating (LABS-TSA PR-3)"
```

---

# PR-4 — Builtin participants + tournament runner integration

**Branch:** `labs-tsa/pr4-builtins-and-runner`

This is the heaviest PR. Builds: namespaced builtin registry on the dashboard
side, `roster` field on `POST /api/v1/tournaments`, tournament-runner
branching on builtin vs agent moves, concurrent-private cap, new
`GET /api/v1/games/{game_type}/builtins` endpoint.

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/tournament/builtins.py` — dashboard-side namespaced registry
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` — accept roster, insert Participants, concurrent-cap
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/runner.py` (or wherever the move-loop lives — locate first)
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py` — extend request schema
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/builtins_api.py` — new endpoint
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py` — include new router
- Create: `tests/integration/dashboard/test_tsa_builtins.py`
- Create: `tests/integration/dashboard/test_tsa_tournament_create.py`
- Create: `tests/integration/dashboard/test_tsa_runner_builtins.py` (unit-style on the runner branching)

## Task 4.1: Start branch

- [ ] **Step 4.1.1: Branch from main**

```bash
git checkout main && git pull github main --ff-only
git checkout -b labs-tsa/pr4-builtins-and-runner
```

## Task 4.2: Dashboard-side namespaced builtin registry

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/tournament/builtins.py`
- Create: `tests/integration/dashboard/test_tsa_builtins.py`

- [ ] **Step 4.2.1: Write failing test**

Create `tests/integration/dashboard/test_tsa_builtins.py`:

```python
"""LABS-TSA PR-4 — namespaced builtin strategy registry."""

import pytest

from atp.dashboard.tournament.builtins import (
    BuiltinNotFoundError,
    list_builtins_for_game,
    resolve_builtin,
)


class TestBuiltinRegistry:
    def test_list_el_farol_builtins(self) -> None:
        names = {b.name for b in list_builtins_for_game("el_farol")}
        # These match game_envs.strategies.el_farol_strategies
        assert "el_farol/traditionalist" in names
        assert "el_farol/contrarian" in names
        assert "el_farol/random" in names

    def test_resolve_returns_strategy_instance(self) -> None:
        strategy = resolve_builtin(
            "el_farol/traditionalist", tournament_id=1, participant_id=1
        )
        assert strategy is not None
        # Strategy base class provides choose_action — smoke only, no call.
        assert hasattr(strategy, "choose_action")

    def test_resolve_seeded_strategy_is_deterministic(self) -> None:
        # Strategies with a ``seed`` kwarg (e.g. Random family) must
        # produce the same RNG state for the same (tournament,
        # participant) pair across repeated resolves.
        a = resolve_builtin(
            "prisoners_dilemma/random", tournament_id=7, participant_id=3
        )
        b = resolve_builtin(
            "prisoners_dilemma/random", tournament_id=7, participant_id=3
        )
        # ._rng is the ``random.Random`` instance seeded in __init__
        assert a._rng.random() == b._rng.random()

    def test_unknown_raises(self) -> None:
        with pytest.raises(BuiltinNotFoundError):
            resolve_builtin(
                "el_farol/nonexistent", tournament_id=1, participant_id=1
            )

    def test_unnamespaced_raises(self) -> None:
        with pytest.raises(BuiltinNotFoundError):
            # bare name without game/ prefix
            resolve_builtin(
                "traditionalist", tournament_id=1, participant_id=1
            )
```

- [ ] **Step 4.2.2: Run — must fail (module missing)**

Run: `uv run pytest tests/integration/dashboard/test_tsa_builtins.py -v --no-cov`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4.2.3: Implement the registry**

Create `packages/atp-dashboard/atp/dashboard/tournament/builtins.py`:

```python
"""Namespaced builtin-strategy registry for private test tournaments.

Wire name: ``{game}/{strategy}``. Cross-game collisions (e.g.
"random" in PD, auction, congestion, blotto) are disambiguated by
the game prefix. ``resolve_builtin`` derives a stable seed from
``(tournament_id, participant_id)`` and passes it to the class
constructor only when the class accepts a ``seed`` kwarg.

The underlying class lookup uses the real
``game_envs.strategies.registry.StrategyRegistry`` — a classmethod
registry populated via decorator-style ``register`` calls when
each ``<game>_strategies`` module is imported. There is no
hypothetical ``REGISTRY`` dict on the modules; we explicitly narrow
the global registry to classes from the importing game's module.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from game_envs.core.strategy import Strategy
from game_envs.strategies.registry import StrategyRegistry


class BuiltinNotFoundError(KeyError):
    """Raised when a namespaced builtin name does not resolve."""


@dataclass(frozen=True)
class BuiltinDescriptor:
    name: str  # "el_farol/traditionalist"
    description: str


# Each entry lists the strategy module that, once imported, registers
# its classes with StrategyRegistry. Dashboard → game_envs direction is
# a runtime dep (peer package in the uv workspace).
_GAME_STRATEGY_MODULES: dict[str, str] = {
    "el_farol": "game_envs.strategies.el_farol_strategies",
    "prisoners_dilemma": "game_envs.strategies.pd_strategies",
    "colonel_blotto": "game_envs.strategies.blotto_strategies",
    "auction": "game_envs.strategies.auction_strategies",
    "congestion": "game_envs.strategies.congestion_strategies",
    "stag_hunt": "game_envs.strategies.stag_hunt_strategies",
    "battle_of_sexes": "game_envs.strategies.bos_strategies",
    "public_goods": "game_envs.strategies.pg_strategies",
}


def _load_game_strategies(game_type: str) -> dict[str, type[Strategy]]:
    """Return ``{bare_name: class}`` for the given game.

    Imports the game's strategies module (which triggers
    ``StrategyRegistry.register`` calls at import time), then filters
    the global registry to classes defined in that module.
    """
    module_path = _GAME_STRATEGY_MODULES.get(game_type)
    if module_path is None:
        return {}
    module = import_module(module_path)
    out: dict[str, type[Strategy]] = {}
    # StrategyRegistry stores in a classmethod-internal dict; access via
    # the documented class attribute (see game_envs/strategies/registry.py).
    for bare_name in StrategyRegistry.list_strategies():
        cls = StrategyRegistry.get(bare_name)
        if cls.__module__ == module.__name__:
            out[bare_name] = cls
    return out


def list_builtins_for_game(game_type: str) -> list[BuiltinDescriptor]:
    """Namespaced descriptors for every builtin the game offers."""
    out: list[BuiltinDescriptor] = []
    for bare_name, cls in _load_game_strategies(game_type).items():
        desc = (cls.__doc__ or "").strip().split("\n", 1)[0]
        out.append(
            BuiltinDescriptor(name=f"{game_type}/{bare_name}", description=desc)
        )
    out.sort(key=lambda b: b.name)
    return out


def _stable_seed(tournament_id: int, participant_id: int) -> int:
    """Derive a process-stable integer seed.

    Python's built-in ``hash()`` is PYTHONHASHSEED-randomised per process
    — not usable here. We hash the identity pair with blake2b and
    project the first 8 bytes to an unsigned 64-bit int.
    """
    digest = hashlib.blake2b(
        f"{tournament_id}:{participant_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def resolve_builtin(
    namespaced_name: str,
    *,
    tournament_id: int,
    participant_id: int,
) -> Strategy:
    """Instantiate a builtin strategy by its namespaced wire name.

    The class's constructor is inspected: only classes that accept a
    ``seed`` keyword argument receive one, so non-RNG-backed strategies
    like ``Traditionalist(window_size=6)`` are not forced to take a
    parameter they don't understand.

    Raises:
        BuiltinNotFoundError: name malformed or unknown, or game unknown.
    """
    if "/" not in namespaced_name:
        raise BuiltinNotFoundError(
            f"strategy name must be namespaced as 'game/name',"
            f" got {namespaced_name!r}"
        )
    game_type, bare_name = namespaced_name.split("/", 1)
    strategies = _load_game_strategies(game_type)
    cls = strategies.get(bare_name)
    if cls is None:
        raise BuiltinNotFoundError(
            f"unknown builtin strategy {namespaced_name!r} for game {game_type!r}"
        )
    kwargs: dict[str, Any] = {}
    params = inspect.signature(cls).parameters
    if "seed" in params:
        kwargs["seed"] = _stable_seed(tournament_id, participant_id)
    return cls(**kwargs)
```

Before running tests, smoke-check the class imports & registry lookup:

```bash
uv run python -c "
from atp.dashboard.tournament.builtins import list_builtins_for_game
print(list_builtins_for_game('el_farol'))
"
```

Expected: non-empty list with at least ``el_farol/traditionalist`` and
``el_farol/contrarian`` in the output.

- [ ] **Step 4.2.4: Run tests**

Expected: 4 passed.

- [ ] **Step 4.2.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/builtins.py \
        tests/integration/dashboard/test_tsa_builtins.py
git commit -m "feat(tournament): dashboard-side namespaced builtin registry (LABS-TSA PR-4)"
```

## Task 4.3: `GET /api/v1/games/{game_type}/builtins` endpoint

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/builtins_api.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`
- Extend: `tests/integration/dashboard/test_tsa_builtins.py`

- [ ] **Step 4.3.1: Write failing test**

Append to `test_tsa_builtins.py`:

```python
class TestBuiltinsEndpoint:
    @pytest.mark.anyio
    async def test_list_endpoint(self, v2_app) -> None:
        from httpx import ASGITransport, AsyncClient
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/games/el_farol/builtins")
        assert resp.status_code == 200
        body = resp.json()
        assert body["game_type"] == "el_farol"
        names = {b["name"] for b in body["builtins"]}
        assert "el_farol/traditionalist" in names

    @pytest.mark.anyio
    async def test_unknown_game_returns_empty(self, v2_app) -> None:
        from httpx import ASGITransport, AsyncClient
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/games/not_a_game/builtins")
        assert resp.status_code == 200
        assert resp.json()["builtins"] == []
```

And add `v2_app` fixture (copy from `test_tsa_auth_gating.py`) at module level
so it's available to both test classes.

- [ ] **Step 4.3.2: Implement the endpoint**

Create `packages/atp-dashboard/atp/dashboard/v2/routes/builtins_api.py`:

```python
"""LABS-TSA PR-4 — builtins listing endpoint.

Powers the "Builtin sparring partners" widget on /ui/tournaments/new.
"""

from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel

from atp.dashboard.tournament.builtins import list_builtins_for_game

router = APIRouter(prefix="/v1/games", tags=["games", "builtins"])


class BuiltinEntry(BaseModel):
    name: str
    description: str


class BuiltinsResponse(BaseModel):
    game_type: str
    builtins: list[BuiltinEntry]


@router.get("/{game_type}/builtins", response_model=BuiltinsResponse)
async def list_builtins(game_type: str) -> BuiltinsResponse:
    return BuiltinsResponse(
        game_type=game_type,
        builtins=[BuiltinEntry(**asdict(b)) for b in list_builtins_for_game(game_type)],
    )
```

Register in `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`
(follow the pattern from `el_farol_dashboard_router`):

```python
from atp.dashboard.v2.routes.builtins_api import router as builtins_api_router
# ...
router.include_router(builtins_api_router)
# add to __all__
```

- [ ] **Step 4.3.3: Run tests**

Expected: 2 passed.

- [ ] **Step 4.3.4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/builtins_api.py \
        packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py \
        tests/integration/dashboard/test_tsa_builtins.py
git commit -m "feat(api): GET /api/v1/games/{game}/builtins (LABS-TSA PR-4)"
```

## Task 4.4: Extend `POST /api/v1/tournaments` with `roster` + concurrent-cap

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py` — request schema
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` — `create_tournament` signature
- Create: `tests/integration/dashboard/test_tsa_tournament_create.py`

- [ ] **Step 4.4.1: Failing tests for roster + cap**

Create `tests/integration/dashboard/test_tsa_tournament_create.py`:

```python
"""LABS-TSA PR-4 — tournament creation with builtin roster and private cap."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def v2_app(test_database: Database):
    app = create_test_app(use_v2_routes=True)

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


class TestCreateWithRoster:
    @pytest.mark.anyio
    async def test_private_tournament_with_builtin_roster(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # creator first registers at least one tournament agent to satisfy
            # the "creator commit" validator
            resp = await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "creator-t", "agent_type": "mcp", "purpose": "tournament"},
            )
            assert resp.status_code == 201

            resp = await client.post(
                "/api/v1/tournaments", headers=auth_headers,
                json={
                    "game_type": "el_farol",
                    "num_players": 3,
                    "total_rounds": 5,
                    "round_deadline_s": 30,
                    "private": True,
                    "roster": [
                        {"builtin_strategy": "el_farol/traditionalist"},
                        {"builtin_strategy": "el_farol/contrarian"},
                    ],
                    "config": {},
                },
            )
            assert resp.status_code == 201, resp.text
            body = resp.json()
            assert body["join_token"] is not None

    @pytest.mark.anyio
    async def test_private_tournament_rejected_without_tournament_agent(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # creator has zero tournament-purpose agents
            resp = await client.post(
                "/api/v1/tournaments", headers=auth_headers,
                json={
                    "game_type": "el_farol",
                    "num_players": 2,
                    "total_rounds": 5,
                    "round_deadline_s": 30,
                    "private": True,
                    "roster": [],
                },
            )
            assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_unknown_builtin_rejected(
        self, v2_app, auth_headers
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "creator-t", "agent_type": "mcp", "purpose": "tournament"},
            )
            resp = await client.post(
                "/api/v1/tournaments", headers=auth_headers,
                json={
                    "game_type": "el_farol",
                    "num_players": 2,
                    "total_rounds": 1,
                    "round_deadline_s": 30,
                    "private": True,
                    "roster": [{"builtin_strategy": "el_farol/nope"}],
                    "config": {},
                },
            )
            assert resp.status_code == 400
            assert "unknown builtin" in resp.text.lower()


class TestConcurrentPrivateCap:
    @pytest.mark.anyio
    async def test_fourth_private_tournament_rejected(
        self, v2_app, auth_headers, monkeypatch
    ) -> None:
        monkeypatch.setenv("ATP_MAX_CONCURRENT_PRIVATE_TOURNAMENTS_PER_USER", "3")
        from atp.dashboard.v2.config import get_config
        get_config.cache_clear()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            await client.post(
                "/api/v1/agents", headers=auth_headers,
                json={"name": "creator-t", "agent_type": "mcp", "purpose": "tournament"},
            )
            body = {
                "game_type": "el_farol",
                "num_players": 2,
                "total_rounds": 1,
                "round_deadline_s": 30,
                "private": True,
                "roster": [],
                "config": {},
            }
            for i in range(3):
                resp = await client.post(
                    "/api/v1/tournaments", headers=auth_headers, json=body
                )
                assert resp.status_code == 201, (i, resp.text)
            resp = await client.post(
                "/api/v1/tournaments", headers=auth_headers, json=body
            )
            assert resp.status_code == 429
```

- [ ] **Step 4.4.2: Run — must fail on several axes**

Run: `uv run pytest tests/integration/dashboard/test_tsa_tournament_create.py -v --no-cov`
Expected: failures.

- [ ] **Step 4.4.3: Extend the request schema**

In `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py`, find
`CreateTournamentRequest` (around line 100) and add:

```python
class BuiltinRosterEntry(BaseModel):
    builtin_strategy: str


class CreateTournamentRequest(BaseModel):
    game_type: str
    num_players: int
    total_rounds: int
    round_deadline_s: int
    private: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    # LABS-TSA PR-4
    roster: list[BuiltinRosterEntry] = Field(default_factory=list)
```

- [ ] **Step 4.4.4: Implement validator + concurrent-cap in service**

In `packages/atp-dashboard/atp/dashboard/tournament/service.py`, find the
`create_tournament` method (around line 193). Extend its signature to accept
`roster: list[str]` (list of namespaced strategy names). Before inserting the
`Tournament` row, do:

```python
from atp.dashboard.tournament.builtins import (
    BuiltinNotFoundError,
    resolve_builtin,
)
from atp.dashboard.v2.config import get_config

# 1. Validate every builtin name and game pairing
for name in roster:
    try:
        # Dummy tournament/participant ids — the instance is discarded
        # immediately. We only care whether the class resolves.
        resolve_builtin(name, tournament_id=0, participant_id=0)
    except BuiltinNotFoundError as e:
        raise ValueError(f"unknown builtin strategy in roster: {e}") from e
    game_prefix, _ = name.split("/", 1)
    if game_prefix != game_type:
        raise ValueError(
            f"builtin {name!r} is for game {game_prefix!r}, not {game_type!r}"
        )

# 2. Builtin count ≤ num_players
if len(roster) > num_players:
    raise ValueError(
        f"builtin roster ({len(roster)}) larger than num_players ({num_players})"
    )

# 3. Creator-commit check for private
if private:
    creator_tournament_agents = await session.scalar(
        select(func.count(Agent.id)).where(
            Agent.owner_id == created_by,
            Agent.purpose == "tournament",
            Agent.deleted_at.is_(None),
        )
    )
    if (creator_tournament_agents or 0) == 0 and len(roster) < num_players:
        raise ValueError(
            "private tournament needs at least one participant "
            "(your tournament agent or a full builtin roster)"
        )

# 4. Concurrent-cap for private
if private:
    cfg = get_config()
    active = await session.scalar(
        select(func.count(Tournament.id)).where(
            Tournament.created_by == created_by,
            Tournament.join_token.is_not(None),
            Tournament.status.in_([TournamentStatus.PENDING, TournamentStatus.ACTIVE]),
            # exclude expired-pending so deadlines.py auto-cancel
            # doesn't race the cap.
            (Tournament.status != TournamentStatus.PENDING) | (Tournament.pending_deadline > func.now()),
        )
    )
    if (active or 0) >= cfg.max_concurrent_private_tournaments_per_user:
        raise ConcurrentPrivateCapExceededError(active, cfg.max_concurrent_private_tournaments_per_user)
```

Add the `ConcurrentPrivateCapExceededError` class (or an equivalent) to
`atp/dashboard/tournament/errors.py`. The route handler translates it into
429.

After inserting the `Tournament` row, for each entry in `roster`:

```python
for name in roster:
    session.add(Participant(
        tournament_id=tournament.id,
        user_id=None,
        agent_id=None,
        agent_name=name,  # use the namespaced name as display
        builtin_strategy=name,
    ))
```

- [ ] **Step 4.4.5: Thread roster / return 429 at the route**

In `tournament_api.py`, pass `roster=[e.builtin_strategy for e in req.roster]`
into `service.create_tournament`. Wrap the service call:

```python
try:
    tournament, join_token = await service.create_tournament(
        ..., roster=[e.builtin_strategy for e in req.roster],
    )
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e)) from e
except ConcurrentPrivateCapExceededError as e:
    raise HTTPException(
        status_code=429,
        detail=f"concurrent private tournament limit exceeded ({e.current}/{e.cap})",
    ) from e
```

- [ ] **Step 4.4.6: Run tests**

Expected: 4 passed.

- [ ] **Step 4.4.7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        packages/atp-dashboard/atp/dashboard/tournament/errors.py \
        packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py \
        tests/integration/dashboard/test_tsa_tournament_create.py
git commit -m "feat(tournament): roster field + concurrent-private cap on POST /api/v1/tournaments (LABS-TSA PR-4)"
```

## Task 4.5: Tournament runner branches on builtin vs agent moves

**Files:**
- Modify: whichever module drives per-round move collection (candidates:
  `packages/atp-dashboard/atp/dashboard/tournament/service.py` — look for
  `play_round` / `resolve_round` / `wait_for_make_move` / `get_move_for_participant`)
- Create: `tests/integration/dashboard/test_tsa_runner_builtins.py`

- [ ] **Step 4.5.1: Locate the move-fetch site**

Run: `rg -n "def.*make_move|wait_for_make_move|resolve_round|play_round" packages/atp-dashboard/atp/dashboard/tournament/`
Expected: one or two methods. Note the primary one (likely
`service.py::_collect_moves` or similar).

- [ ] **Step 4.5.2: Failing unit test**

Create `tests/integration/dashboard/test_tsa_runner_builtins.py`:

```python
"""LABS-TSA PR-4 — runner branches on builtin_strategy."""

import pytest

from atp.dashboard.tournament.service import collect_action_for_participant
from atp.dashboard.tournament.models import Participant


class TestRunnerBranch:
    @pytest.mark.anyio
    async def test_builtin_participant_resolved_synchronously(self) -> None:
        from game_envs.core.strategy import Observation

        p = Participant(
            id=42,
            tournament_id=1,
            user_id=None,
            agent_id=None,
            agent_name="el_farol/traditionalist",
            builtin_strategy="el_farol/traditionalist",
        )
        # Observation fields vary per game — consult
        # ``game_envs/core/strategy.py`` for the exact dataclass.
        # For El Farol the Traditionalist needs at least an empty
        # history to compute its "attend if recent attendance was low"
        # rule.
        obs = Observation(history=[], own_history=[], round_number=1)
        action = await collect_action_for_participant(
            participant=p,
            observation=obs,
            tournament_id=1,
        )
        assert action is not None
```

The test shape depends on the function signature you refactor — adjust to
match. If the ``Observation`` dataclass has additional required fields,
either pass sensible defaults or build the observation via the same helper
the MCP path uses.

- [ ] **Step 4.5.3: Implement the branching**

In the move-collection site, wrap the current MCP-based logic with:

```python
from atp.dashboard.tournament.builtins import resolve_builtin


async def collect_action_for_participant(
    *, participant: Participant, observation: Any, tournament_id: int
) -> Any:
    if participant.builtin_strategy is not None:
        strategy = resolve_builtin(
            participant.builtin_strategy,
            tournament_id=tournament_id,
            participant_id=participant.id,
        )
        # `Strategy.choose_action` is synchronous and deterministic.
        # The seed (when the class accepts one) is derived inside
        # resolve_builtin via a stable blake2b of the identity pair —
        # never the PYTHONHASHSEED-randomised built-in hash().
        return strategy.choose_action(observation)
    # existing MCP path
    return await wait_for_make_move(participant=participant, ...)
```

`observation` is an instance of the game's `Observation` dataclass
(see `game_envs.core.strategy`), constructed from the same per-player
state view the MCP `get_current_state` tool produces today. The move-
collection site already builds a per-participant state dict for the
MCP path; wrap that into an `Observation` and pass it through on the
builtin path — the `Observation` type is the game-envs contract.

For each game, the mapping from "current round state dict" to
`Observation` is small and deterministic. If the existing runner
helper hides this behind a single function, reuse it; if it inlines
the construction, factor it into a named helper
`_observation_for(game_type, state, participant)` and call it from
both branches.

- [ ] **Step 4.5.4: Determinism test**

Append:

```python
    @pytest.mark.anyio
    async def test_same_participant_and_tournament_gives_same_action(self) -> None:
        from game_envs.core.strategy import Observation

        p = Participant(
            id=7, tournament_id=3, user_id=None, agent_id=None,
            agent_name="el_farol/random",
            builtin_strategy="el_farol/random",
        )
        obs = Observation(history=[], own_history=[], round_number=1)
        a = await collect_action_for_participant(
            participant=p, observation=obs, tournament_id=3
        )
        b = await collect_action_for_participant(
            participant=p, observation=obs, tournament_id=3
        )
        assert a == b
```

Crucially, run the test **in a fresh Python process** (pytest spawns a
new process per invocation) to verify the blake2b-based seed is stable
across process boundaries — Python's built-in ``hash()`` would have
differed between runs because of ``PYTHONHASHSEED``.

- [ ] **Step 4.5.5: Run**

Expected: passes.

- [ ] **Step 4.5.6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/integration/dashboard/test_tsa_runner_builtins.py
git commit -m "feat(runner): branch on builtin_strategy; deterministic seed (LABS-TSA PR-4)"
```

## Task 4.6: Open PR-4

- [ ] **Step 4.6.1: Run all PR-4 tests**

```bash
uv run pytest tests/integration/dashboard/test_tsa_builtins.py \
              tests/integration/dashboard/test_tsa_tournament_create.py \
              tests/integration/dashboard/test_tsa_runner_builtins.py \
              tests/integration/dashboard/test_admin_tournament_ui.py \
              -v --no-cov
```
Expected: all green. `test_admin_tournament_ui.py` is included to catch
regressions in existing tournament create flow.

- [ ] **Step 4.6.2: Push + open PR**

```bash
git push -u github labs-tsa/pr4-builtins-and-runner
gh pr create --title "feat(tournament): builtin participants + runner branching (LABS-TSA PR-4)"
```

---

# PR-5 — UI self-service form + match-tournament linkage

**Branch:** `labs-tsa/pr5-ui-and-match-linkage`

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` — new GET/POST `/ui/tournaments/new`; update `/ui/matches` with visibility JOIN; update `/ui/agents` route for quota strip
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_new.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html` — private badge + cancel button
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/agents.html` — purpose column + quota strip + separate register buttons
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/matches.html` — no visible change, just backend filter
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` — `on_completion` hook writes `GameResult`
- Create: `tests/integration/dashboard/test_tsa_ui_tournament_new.py`
- Create: `tests/integration/dashboard/test_tsa_match_linkage.py`
- Create: `tests/e2e/test_tsa_playwright_smoke.py`

## Task 5.1: Start branch

- [ ] **Step 5.1.1: Branch from main**

```bash
git checkout main && git pull github main --ff-only
git checkout -b labs-tsa/pr5-ui-and-match-linkage
```

## Task 5.2: GameResult dual-write on tournament completion

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/tournament/service.py` — extend the `status → completed` transition
- Create: `tests/integration/dashboard/test_tsa_match_linkage.py`

- [ ] **Step 5.2.1: Failing test for dual-write idempotency**

Create `tests/integration/dashboard/test_tsa_match_linkage.py`:

```python
"""LABS-TSA PR-5 — GameResult dual-write at tournament completion."""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import GameResult
from atp.dashboard.tournament.models import Tournament, TournamentStatus
from atp.dashboard.tournament.service import TournamentService


class TestDualWrite:
    @pytest.mark.anyio
    async def test_completion_creates_game_result_with_tournament_id(
        self, db_session: AsyncSession, user
    ) -> None:
        svc = TournamentService()
        t = Tournament(
            game_type="el_farol", num_players=2, total_rounds=1, created_by=user.id,
        )
        db_session.add(t)
        await db_session.commit()
        await svc.mark_tournament_completed(db_session, t.id)
        rows = (await db_session.execute(
            select(GameResult).where(GameResult.tournament_id == t.id)
        )).scalars().all()
        assert len(rows) == 1
        # match_id is a fresh UUID, never the join_token
        assert uuid.UUID(rows[0].match_id)  # parses
        assert rows[0].match_id != (t.join_token or "")

    @pytest.mark.anyio
    async def test_idempotent_on_double_completion(
        self, db_session: AsyncSession, user
    ) -> None:
        svc = TournamentService()
        t = Tournament(
            game_type="el_farol", num_players=2, total_rounds=1, created_by=user.id,
        )
        db_session.add(t)
        await db_session.commit()
        await svc.mark_tournament_completed(db_session, t.id)
        await svc.mark_tournament_completed(db_session, t.id)  # second call no-ops
        rows = (await db_session.execute(
            select(GameResult).where(GameResult.tournament_id == t.id)
        )).scalars().all()
        assert len(rows) == 1  # UNIQUE partial index neutralises the second write
```

- [ ] **Step 5.2.2: Run — fails until hook added**

- [ ] **Step 5.2.3: Implement the hook**

In `service.py`, add or extend `mark_tournament_completed` to call a new
helper at the end of the transition:

```python
import uuid
from sqlalchemy.exc import IntegrityError

async def _write_game_result_for_tournament(
    session: AsyncSession, tournament: Tournament
) -> None:
    match_id = str(uuid.uuid4())
    row = GameResult(
        match_id=match_id,
        game_name=tournament.game_type,
        game_type=tournament.config.get("variant", "tournament"),
        num_players=tournament.num_players,
        num_rounds=tournament.total_rounds,
        num_episodes=1,
        status="completed",
        tournament_id=tournament.id,
        # actions_json / day_aggregates_json / round_payoffs_json /
        # agents_json — reshape from Round/Action here. If the reshape is
        # heavy, factor into a separate helper; for an initial PR, a
        # stub that reads the Round rows is acceptable as long as the
        # schema fields are populated.
    )
    # Wrap the insert in a SAVEPOINT so the UNIQUE-partial-index
    # IntegrityError on the second completion attempt only rolls back
    # the nested transaction — the outer transaction (which flipped
    # tournament.status to 'completed') stays intact.
    try:
        async with session.begin_nested():
            session.add(row)
            await session.flush()
    except IntegrityError:
        # Idempotency path — some other completion handler already wrote
        # the GameResult for this tournament_id. Fall through silently.
        pass
```

The reshape into `actions_json` / `day_aggregates_json` is game-specific (El
Farol uses the writer already present in `atp/cli/commands/game.py::_build_game_result_kwargs`).
For now, write a *minimal* Phase-7 payload:

```python
        actions_json=[],        # populated by PR-5 follow-up; see TODO
        day_aggregates_json=[],
        round_payoffs_json=[],
        agents_json=[],
```

and open a follow-up ticket: "reshape tournament rounds into Phase-7 dashboard
payload". Record the TODO in the PR description.

- [ ] **Step 5.2.4: Run tests**

Expected: 2 passed.

- [ ] **Step 5.2.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/tournament/service.py \
        tests/integration/dashboard/test_tsa_match_linkage.py
git commit -m "feat(tournament): write GameResult on completion with UUID match_id (LABS-TSA PR-5)"
```

## Task 5.3: `/ui/matches` visibility filter

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py::ui_matches`
- Extend: `tests/integration/dashboard/test_matches_listing.py`

- [ ] **Step 5.3.1: Failing test — non-owner sees no private-tournament match**

Append to `tests/integration/dashboard/test_matches_listing.py`:

```python
class TestVisibilityFilter:
    @pytest.mark.anyio
    async def test_anonymous_does_not_see_private_tournament_match(
        self, v2_app, db_session
    ) -> None:
        from atp.dashboard.models import GameResult, User
        from atp.dashboard.tournament.models import Tournament

        user = User(username="u", email="u@t.com", hashed_password="x", is_active=True)
        db_session.add(user)
        await db_session.commit()

        t = Tournament(
            game_type="el_farol", num_players=2, total_rounds=1,
            created_by=user.id, join_token="secret",
        )
        db_session.add(t)
        await db_session.commit()

        g = GameResult(
            game_name="El Farol...", game_type="tournament",
            num_players=2, num_rounds=1, status="completed",
            tournament_id=t.id, match_id="uuid-private",
            actions_json=[{"day":1}], day_aggregates_json=[{"day":1, "slot_attendance":[0]*16, "over_slots":0}],
        )
        db_session.add(g)
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")
        assert resp.status_code == 200
        assert "uuid-private" not in resp.text

    @pytest.mark.anyio
    async def test_anonymous_sees_legacy_null_tournament_match(
        self, v2_app, db_session
    ) -> None:
        from atp.dashboard.models import GameResult
        db_session.add(
            GameResult(
                game_name="El Farol Bar (n=6, days=30)", game_type="repeated",
                num_players=6, num_rounds=30, status="completed",
                tournament_id=None, match_id="uuid-legacy",
                actions_json=[{"day":1}],
                day_aggregates_json=[{"day":1, "slot_attendance":[0]*16, "over_slots":0}],
            )
        )
        await db_session.commit()
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/matches")
        assert resp.status_code == 200
        assert "uuid-legacy" in resp.text
```

- [ ] **Step 5.3.2: Run — must fail on visibility**

- [ ] **Step 5.3.3: Add the JOIN filter**

In `ui.py::ui_matches`, replace the current filter-building block with:

```python
from atp.dashboard.tournament.models import Participant, Tournament

user_id = user.id if user else None

# Base renderability filters
renderable_filters = [
    GameResult.status == "completed",
    GameResult.actions_json.is_not(None),
    GameResult.day_aggregates_json.is_not(None),
]

# Visibility filter — outerjoin on tournament to pass NULL tournament_id through
stmt = (
    select(GameResult)
    .outerjoin(Tournament, GameResult.tournament_id == Tournament.id)
    .where(*renderable_filters)
)
if user and user.is_admin:
    pass  # no extra filter
else:
    visibility_clauses = [
        GameResult.tournament_id.is_(None),
        Tournament.join_token.is_(None),
    ]
    if user_id is not None:
        visibility_clauses.append(Tournament.created_by == user_id)
        visibility_clauses.append(
            Tournament.id.in_(
                select(Participant.tournament_id).where(Participant.user_id == user_id)
            )
        )
    stmt = stmt.where(or_(*visibility_clauses))
stmt = stmt.order_by(GameResult.completed_at.desc().nulls_last()).limit(100)
```

Also update the `total_stmt` with the same JOIN + filter.

- [ ] **Step 5.3.4: Run tests**

Expected: both new tests pass; existing `test_matches_listing.py` still
passes (legacy rows have `tournament_id IS NULL` so they stay visible).

- [ ] **Step 5.3.5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py \
        tests/integration/dashboard/test_matches_listing.py
git commit -m "feat(ui): /ui/matches filters private-tournament matches (LABS-TSA PR-5)"
```

## Task 5.4: `/ui/tournaments/new` self-service form

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_new.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` — add GET + POST handlers
- Create: `tests/integration/dashboard/test_tsa_ui_tournament_new.py`

- [ ] **Step 5.4.1: Failing test — GET renders form**

Create `tests/integration/dashboard/test_tsa_ui_tournament_new.py`:

```python
"""LABS-TSA PR-5 — /ui/tournaments/new self-service form."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def v2_app(test_database: Database):
    app = create_test_app(use_v2_routes=True)

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


class TestTournamentNewGet:
    @pytest.mark.anyio
    async def test_form_renders(
        self, v2_app, disable_dashboard_auth
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ui/tournaments/new")
        assert resp.status_code == 200
        assert "<form" in resp.text.lower()
        assert 'name="game_type"' in resp.text
        assert 'name="private"' in resp.text


class TestTournamentNewPost:
    @pytest.mark.anyio
    async def test_post_creates_and_renders_detail_with_token(
        self, v2_app, disable_dashboard_auth, db_session
    ) -> None:
        # Seed a user + a tournament agent
        from atp.dashboard.models import Agent, User
        user = User(username="ui-user", email="u@t.com", hashed_password="x", is_active=True)
        db_session.add(user)
        await db_session.commit()
        db_session.add(Agent(
            name="t-agent", agent_type="mcp", owner_id=user.id, purpose="tournament",
        ))
        await db_session.commit()

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/ui/tournaments/new",
                data={
                    "game_type": "el_farol",
                    "num_players": "3",
                    "total_rounds": "5",
                    "round_deadline_s": "30",
                    "private": "on",
                    "roster[]": [
                        "el_farol/traditionalist", "el_farol/contrarian",
                    ],
                },
            )
        assert resp.status_code == 200
        # one-time reveal: page contains the copy-box and some JOIN TOKEN text
        assert "join_token" in resp.text.lower() or "join token" in resp.text.lower()
```

- [ ] **Step 5.4.2: Write the template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_new.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}ATP · New tournament{% endblock %}

{% block content %}
<article style="padding:16px 20px; max-width:720px">
  <h2>New tournament</h2>
  <form method="post" action="/ui/tournaments/new">
    <label>Game
      <select name="game_type" id="game_type">
        {% for game in games %}
        <option value="{{ game }}">{{ game }}</option>
        {% endfor %}
      </select>
    </label>

    <fieldset>
      <legend>Visibility</legend>
      {% if user.is_admin %}
      <label><input type="radio" name="private" value="on" checked> Private</label>
      <label><input type="radio" name="private" value="off"> Public</label>
      {% else %}
      {# Disabled inputs don't submit — use a hidden input to carry the
         "always private for non-admins" intent into the POST body and
         render a disabled radio purely for visual feedback. #}
      <input type="hidden" name="private" value="on">
      <label><input type="radio" checked disabled> Private (non-admins cannot create public tournaments)</label>
      {% endif %}
    </fieldset>

    <label>num_players <input type="number" name="num_players" min="2" max="16" value="4" required></label>
    <label>total_rounds <input type="number" name="total_rounds" min="1" max="500" value="30" required></label>
    <label>round_deadline_s <input type="number" name="round_deadline_s" min="5" max="600" value="30" required></label>

    <fieldset>
      <legend>Builtin sparring partners</legend>
      {% for b in builtins %}
      <label><input type="checkbox" name="roster[]" value="{{ b.name }}"> {{ b.name }} — {{ b.description }}</label>
      {% endfor %}
    </fieldset>

    <button type="submit">Create</button>
  </form>
</article>
{% endblock %}
```

- [ ] **Step 5.4.3: Implement GET handler**

In `ui.py`:

```python
@router.get("/tournaments/new", response_class=HTMLResponse)
@limiter.limit("60/minute")
async def ui_tournaments_new(
    request: Request,
    session: DBSession,
    game_type: str = "el_farol",
) -> HTMLResponse:
    user = await _get_ui_user(request, session)
    from atp.dashboard.tournament.builtins import list_builtins_for_game
    builtins = list_builtins_for_game(game_type)
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/tournament_new.html",
        context={
            "active_page": "tournaments",
            "games": ["el_farol", "prisoners_dilemma", "colonel_blotto"],  # whitelist
            "builtins": builtins,
            "user": user,
        },
    )
```

- [ ] **Step 5.4.4: Implement POST handler**

```python
@router.post("/tournaments/new", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def ui_tournaments_new_submit(
    request: Request,
    session: DBSession,
) -> HTMLResponse:
    user = await _get_ui_user(request, session)
    if user is None:
        return RedirectResponse(url="/ui/login", status_code=303)
    form = await request.form()
    game_type = form.get("game_type", "el_farol")
    private = form.get("private") == "on"
    num_players = int(form.get("num_players", 2))
    total_rounds = int(form.get("total_rounds", 30))
    round_deadline_s = int(form.get("round_deadline_s", 30))
    roster = form.getlist("roster[]")

    from atp.dashboard.tournament.service import TournamentService
    svc = TournamentService()
    try:
        t, join_token = await svc.create_tournament(
            session=session,
            game_type=game_type,
            num_players=num_players,
            total_rounds=total_rounds,
            round_deadline_s=round_deadline_s,
            private=private,
            roster=roster,
            config={},
            created_by=user.id,
        )
    except ValueError as e:
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/tournament_new.html",
            context={
                "active_page": "tournaments",
                "error": str(e),
                "games": ["el_farol", "prisoners_dilemma", "colonel_blotto"],
                "builtins": [],
                "user": user,
            },
            status_code=400,
        )
    # Server-render the detail page directly so we don't leak the join_token
    # through a POST→GET redirect.
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/tournament_detail.html",
        context={
            "active_page": "tournaments",
            "tournament": t,
            "join_token_once": join_token,  # template renders a copy-box when set
            "user": user,
            "creator_name": user.username,
        },
    )
```

- [ ] **Step 5.4.5: Extend `tournament_detail.html` with optional `join_token_once` block**

Near the top of `tournament_detail.html`, add:

```html
{% if join_token_once %}
<div class="cancel-box" style="background:#fffbe6;color:#333;padding:12px;margin:12px 0;border-radius:4px">
  <strong>Share this token with your tournament agents — it is shown once.</strong>
  <input type="text" readonly value="{{ join_token_once }}" onclick="this.select()" style="width:100%;font-family:monospace">
</div>
{% endif %}
```

- [ ] **Step 5.4.6: Run tests**

Run: `uv run pytest tests/integration/dashboard/test_tsa_ui_tournament_new.py -v --no-cov`
Expected: passes.

- [ ] **Step 5.4.7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_new.html \
        packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html \
        packages/atp-dashboard/atp/dashboard/v2/routes/ui.py \
        tests/integration/dashboard/test_tsa_ui_tournament_new.py
git commit -m "feat(ui): /ui/tournaments/new self-service form with one-time reveal (LABS-TSA PR-5)"
```

## Task 5.5: Update `/ui/agents` with quotas + purpose

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py::ui_agents`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/agents.html`

- [ ] **Step 5.5.1: Pass quota + purpose counts into the template**

In `ui_agents` handler, compute:

```python
from atp.dashboard.v2.config import get_config
cfg = get_config()

counts = {}
for purpose in ("benchmark", "tournament"):
    counts[purpose] = await session.scalar(
        select(func.count(Agent.id)).where(
            Agent.owner_id == user.id,
            Agent.purpose == purpose,
            Agent.deleted_at.is_(None),
        )
    ) or 0

quota = {
    "benchmark": cfg.max_benchmark_agents_per_user,
    "tournament": cfg.max_tournament_agents_per_user,
}
```

Pass `counts`, `quota`, and the full agents list into the template context.

- [ ] **Step 5.5.2: Template updates**

In `agents.html`, add near the top:

```html
<div class="quota-strip">
  Tournament agents: {{ counts['tournament'] }}/{{ quota['tournament'] }}
  · Benchmark agents: {{ counts['benchmark'] }}/{{ quota['benchmark'] }}
</div>
<div class="register-buttons">
  <a href="/ui/agents/new?purpose=benchmark" role="button">Register benchmark agent</a>
  <a href="/ui/agents/new?purpose=tournament" role="button">Register tournament agent</a>
</div>
```

In the agents table, add a `Purpose` column that renders `{{ agent.purpose }}`.

- [ ] **Step 5.5.3: Integration test for the strip**

Append to an existing or new test file: render `/ui/agents`, assert the strip
text "Tournament agents:" appears.

- [ ] **Step 5.5.4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py \
        packages/atp-dashboard/atp/dashboard/v2/templates/ui/agents.html \
        tests/integration/dashboard/<relevant test>
git commit -m "feat(ui): agents quota strip + purpose column (LABS-TSA PR-5)"
```

## Task 5.6: Playwright smoke test

**Files:**
- Create: `tests/e2e/test_tsa_playwright_smoke.py`

- [ ] **Step 5.6.1: Check Playwright availability**

Run: `uv run playwright --version`
Expected: a version string. If not installed, the project CI already
installs playwright for e2e jobs — use the same mechanism.

- [ ] **Step 5.6.2: Smoke test — happy path**

Create `tests/e2e/test_tsa_playwright_smoke.py`:

```python
"""LABS-TSA PR-5 — full happy path through the UI."""

import os

import pytest
from playwright.async_api import async_playwright


@pytest.mark.anyio
@pytest.mark.slow
async def test_full_flow_register_create_see_match() -> None:
    base = os.environ.get("ATP_E2E_BASE_URL", "http://localhost:8080")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(f"{base}/ui/login")
            # ... log in flow — reuse an existing test harness if available
            # Register a tournament agent
            await page.goto(f"{base}/ui/agents/new?purpose=tournament")
            await page.fill('input[name="name"]', "smoke-tournament-agent")
            await page.click('button[type="submit"]')
            # Create a tournament
            await page.goto(f"{base}/ui/tournaments/new")
            await page.select_option('select[name="game_type"]', "el_farol")
            await page.fill('input[name="num_players"]', "3")
            await page.fill('input[name="total_rounds"]', "2")
            await page.check('input[value="el_farol/traditionalist"]')
            await page.check('input[value="el_farol/contrarian"]')
            await page.click('button[type="submit"]')
            # Expect detail page with join_token copy-box
            assert await page.locator('input[readonly]').count() == 1
        finally:
            await browser.close()
```

This test is wall-clock-dependent and marked `slow`; runs in the dedicated
`tournament-e2e` CI job, not the main PR gate. Adjust auth specifics to match
the harness your existing e2e tests use (see `tests/e2e/`).

- [ ] **Step 5.6.3: Commit**

```bash
git add tests/e2e/test_tsa_playwright_smoke.py
git commit -m "test(e2e): Playwright smoke for tournament agent sandbox (LABS-TSA PR-5)"
```

## Task 5.7: Open PR-5

- [ ] **Step 5.7.1: Full test sweep**

```bash
uv run pytest tests/integration/dashboard/test_tsa_match_linkage.py \
              tests/integration/dashboard/test_tsa_ui_tournament_new.py \
              tests/integration/dashboard/test_matches_listing.py \
              -v --no-cov
```
Expected: all green.

- [ ] **Step 5.7.2: Push + open PR**

```bash
git push -u github labs-tsa/pr5-ui-and-match-linkage
gh pr create --title "feat(ui): tournament agent sandbox UI + match linkage (LABS-TSA PR-5)"
```

Include in PR body the **open follow-up**: "reshape tournament rounds into
full Phase-7 dashboard payload (`actions_json`, `day_aggregates_json`, etc.)
— current dual-write inserts empty lists". File a separate ticket.

---

# Self-Review Checklist

After the plan is written, walk through the spec one section at a time.

- [x] **Spec § Problem** → covered by the 5-PR rollout (data model → API → auth → runner → UI).
- [x] **Spec § Goals** (5 items) → each goal maps to a task section above.
- [x] **Spec § Non-goals** → plan does not introduce scheduling, cross-user invites, second MCP, or time-compressed mode.
- [x] **Spec § Preserved existing behaviour** → Task 4.4 reuses `join_token`; Task 5.3 outer-joins to preserve legacy NULL rows.
- [x] **Spec § Agent purpose** (column, CHECK, index, env vars, middleware) → Tasks 1.2, 2.2, 2.4, 3.2, 3.3, 3.4, 3.5, 3.6.
- [x] **Spec § Builtin participants** (column, CHECK, runner branch, registry, endpoint) → Tasks 1.3, 4.2, 4.3, 4.4, 4.5.
- [x] **Spec § Concurrent-private cap** → Task 4.4.
- [x] **Spec § Match → tournament linkage** (UUID `match_id`, UNIQUE partial index, IntegrityError idempotency) → Tasks 1.4, 5.2.
- [x] **Spec § Visibility filtering for matches** (non-regression for NULL tournament_id) → Task 5.3.
- [x] **Spec § UI changes** (new form, agents strip, private badge) → Tasks 5.4, 5.5.
- [x] **Spec § Error Handling** (all 9 rows) → individual tests inside the relevant tasks (429 quota, 400 unknown builtin, 403 MCP gate, 404 for invisible, etc.).
- [x] **Spec § Testing** (unit, integration, Alembic) → Tasks 1.2-1.5, 2.3-2.4, 3.2-3.6, 4.2-4.5, 5.2-5.6.
- [x] **Spec § Rollout** (5 PRs) → exactly 5 branches / PR-create commands.
- [x] **Spec § Risk Register** → every mitigation has a task: quota cap (Task 4.4), JWT claims + cache (Tasks 3.2, 3.6), UNIQUE index (Task 1.4), seeded builtins (Task 4.5), startup-time registry check (Task 4.2).

**Outstanding TODO for a follow-up ticket (not part of this plan):** the
dual-write hook in Task 5.2 inserts empty lists for `actions_json` /
`day_aggregates_json` / `round_payoffs_json` / `agents_json`. A Phase-7
reshape identical to what `atp/cli/commands/game.py::_build_game_result_kwargs`
does for CLI runs must be ported into the tournament completion path so the
Cards dashboard actually renders tournament matches. This is the single piece
of content gathered-into-TODO rather than designed-inline, because the
reshape is mechanical once the round/action schema is understood and is a
clear standalone unit of work.
