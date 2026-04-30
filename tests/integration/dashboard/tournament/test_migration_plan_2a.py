"""Integration tests for the Plan 2a Alembic migration."""

import os
import subprocess

import pytest
from sqlalchemy import create_engine, inspect, text


@pytest.fixture
def fresh_db(tmp_path):
    db_path = tmp_path / "plan2a_fresh.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
    engine.dispose()

    env = {**os.environ, "ATP_DASHBOARD_DATABASE_URL": f"sqlite:///{db_path}"}
    result = subprocess.run(
        ["uv", "run", "alembic", "-n", "dashboard", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"alembic upgrade failed: {result.stderr}"

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
    columns = {
        c["name"] for c in inspect(engine).get_columns("tournament_participants")
    }
    assert "released_at" in columns
    user_col = next(
        c
        for c in inspect(engine).get_columns("tournament_participants")
        if c["name"] == "user_id"
    )
    # Plan 2a originally shipped user_id as NOT NULL. LABS-TSA PR-1
    # flipped it to nullable so builtin-strategy participants (which
    # have no User backing them) can live in the same table. This
    # fresh_db fixture runs every migration up to head, so the final
    # state is nullable.
    assert user_col["nullable"] is True
    engine.dispose()


def test_upgrade_creates_action_source(fresh_db):
    engine = create_engine(fresh_db)
    columns = {c["name"] for c in inspect(engine).get_columns("tournament_actions")}
    assert "source" in columns
    engine.dispose()


def test_upgrade_creates_six_base_invariants(fresh_db):
    """Smoke-test base invariants after migrations up to head.

    LABS-TSA PR-6 replaced the two user-keyed Participant uniqueness
    entries with agent-keyed partial unique indexes, so the
    ``uq_participant_tournament_user`` UniqueConstraint no longer appears
    and ``uq_participant_user_active`` is replaced by
    ``uq_participant_tournament_agent`` and ``uq_participant_agent_active``.
    """
    engine = create_engine(fresh_db)
    inspector = inspect(engine)

    ac_unique = {
        u["name"] for u in inspector.get_unique_constraints("tournament_actions")
    }
    assert "uq_action_round_participant" in ac_unique

    r_unique = {
        u["name"] for u in inspector.get_unique_constraints("tournament_rounds")
    }
    assert "uq_round_tournament_number" in r_unique

    r_indexes = {i["name"] for i in inspector.get_indexes("tournament_rounds")}
    assert "idx_round_status_deadline" in r_indexes

    p_indexes = {i["name"] for i in inspector.get_indexes("tournament_participants")}
    assert "uq_participant_tournament_agent" in p_indexes
    assert "uq_participant_agent_active" in p_indexes
    # Old user-keyed indexes gone after PR-6
    assert "uq_participant_user_active" not in p_indexes

    pc_unique = {
        u["name"] for u in inspector.get_unique_constraints("tournament_participants")
    }
    assert "uq_participant_tournament_user" not in pc_unique

    engine.dispose()


def test_upgrade_creates_check_constraint(fresh_db):
    engine = create_engine(fresh_db)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='tournaments'"
            )
        ).scalar()
    assert "ck_tournament_cancel_consistency" in result
    engine.dispose()


def test_downgrade_then_upgrade_idempotent(fresh_db):
    """Smoke-test downgrade and re-upgrade produce a clean schema."""
    env = {**os.environ, "ATP_DASHBOARD_DATABASE_URL": fresh_db}
    result = subprocess.run(
        ["uv", "run", "alembic", "-n", "dashboard", "downgrade", "-1"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"downgrade failed: {result.stderr}"

    result = subprocess.run(
        ["uv", "run", "alembic", "-n", "dashboard", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"re-upgrade failed: {result.stderr}"
