"""Regression test for the agent_name denormalization migration (LABS-54 Phase 1).

Asserts that migration a7b8c9d0e1f2:
  1. Adds agent_name column with backfill from agents.name.
  2. Relaxes agent_id to nullable.
  3. Downgrade drops agent_name and restores agent_id NOT NULL.
"""

import os
import subprocess
from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect, text


def _alembic(env: dict[str, str], *args: str) -> None:
    result = subprocess.run(
        ["uv", "run", "alembic", "-n", "dashboard", *args],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"alembic {' '.join(args)} failed: {result.stderr}"


@pytest.fixture
def db_at_pre_migration(tmp_path):
    """SQLite DB upgraded to f1a2b3c4d5e6 (the revision before ours)."""
    db_path = tmp_path / "labs54_phase1.db"
    env = {**os.environ, "ATP_DASHBOARD_DATABASE_URL": f"sqlite:///{db_path}"}
    _alembic(env, "upgrade", "f1a2b3c4d5e6")
    return f"sqlite:///{db_path}", env


def test_migration_adds_agent_name_and_backfills(db_at_pre_migration):
    """Agent_name is populated from agents.name during upgrade."""
    url, env = db_at_pre_migration

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO agents "
                "(id, tenant_id, name, agent_type, config, "
                "created_at, updated_at, owner_id, version) VALUES "
                "(1, 'default', 'fixture-agent', 'http', '{}', "
                ":now, :now, NULL, 'latest')"
            ),
            {"now": datetime(2026, 1, 1)},
        )
        conn.execute(
            text(
                "INSERT INTO suite_executions "
                "(id, tenant_id, suite_name, agent_id, started_at, "
                "runs_per_test, total_tests, passed_tests, failed_tests, "
                "success_rate, status) VALUES "
                "(1, 'default', 'suite-a', 1, :started, "
                "1, 0, 0, 0, 0.0, 'completed')"
            ),
            {"started": datetime(2026, 1, 1)},
        )
    engine.dispose()

    _alembic(env, "upgrade", "a7b8c9d0e1f2")

    engine = create_engine(url)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT agent_name FROM suite_executions WHERE id = 1")
        ).one()
        assert row[0] == "fixture-agent"
    engine.dispose()


def test_migration_agent_id_becomes_nullable(db_at_pre_migration):
    """After upgrade, agent_id can be NULL for newly-inserted rows."""
    url, env = db_at_pre_migration
    _alembic(env, "upgrade", "a7b8c9d0e1f2")

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO suite_executions "
                "(tenant_id, suite_name, agent_id, agent_name, started_at, "
                "runs_per_test, total_tests, passed_tests, failed_tests, "
                "success_rate, status) VALUES "
                "('default', 'cli-run', NULL, 'cli-agent', :started, "
                "1, 0, 0, 0, 0.0, 'running')"
            ),
            {"started": datetime(2026, 1, 1)},
        )
        row = conn.execute(
            text(
                "SELECT agent_id, agent_name FROM suite_executions "
                "WHERE suite_name = 'cli-run'"
            )
        ).one()
        assert row[0] is None
        assert row[1] == "cli-agent"
    engine.dispose()


def test_migration_downgrade_removes_agent_name(db_at_pre_migration):
    """Downgrade drops agent_name and restores agent_id NOT NULL."""
    url, env = db_at_pre_migration
    _alembic(env, "upgrade", "a7b8c9d0e1f2")
    _alembic(env, "downgrade", "f1a2b3c4d5e6")

    engine = create_engine(url)
    columns = inspect(engine).get_columns("suite_executions")
    names = {c["name"] for c in columns}
    assert "agent_name" not in names

    agent_id_col = next(c for c in columns if c["name"] == "agent_id")
    assert agent_id_col["nullable"] is False
    engine.dispose()
