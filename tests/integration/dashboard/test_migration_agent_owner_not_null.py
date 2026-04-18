"""Regression test for LABS-54 Phase 3 migration (b8c9d0e1f2a3).

Asserts that:
  1. Ownerless Agent rows get backfilled to the lowest-id admin user.
  2. The partial unique index uq_agent_ownerless_tenant_name_version is dropped.
  3. Agent.owner_id becomes NOT NULL.
  4. Upgrade refuses to run if ownerless rows exist but no admin user does.
  5. Downgrade restores nullability and re-creates the partial index.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect, text


def _alembic(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "alembic", "-n", "dashboard", *args],
        env=env,
        capture_output=True,
        text=True,
    )


def _alembic_ok(env: dict[str, str], *args: str) -> None:
    result = _alembic(env, *args)
    assert result.returncode == 0, f"alembic {' '.join(args)} failed: {result.stderr}"


@pytest.fixture
def db_at_phase2(tmp_path):
    """SQLite DB upgraded to a7b8c9d0e1f2 (the revision before ours)."""
    db_path = tmp_path / "labs54_phase3.db"
    env = {**os.environ, "ATP_DASHBOARD_DATABASE_URL": f"sqlite:///{db_path}"}
    _alembic_ok(env, "upgrade", "a7b8c9d0e1f2")
    return f"sqlite:///{db_path}", env


def _insert_user(conn, user_id: int, username: str, *, is_admin: bool) -> None:
    now = datetime(2026, 4, 18)
    conn.execute(
        text(
            "INSERT INTO users "
            "(id, tenant_id, username, email, hashed_password, is_admin, "
            "is_active, created_at, updated_at) VALUES "
            "(:id, 'default', :u, :e, 'x', :admin, 1, :now, :now)"
        ),
        {
            "id": user_id,
            "u": username,
            "e": f"{username}@t.local",
            "admin": 1 if is_admin else 0,
            "now": now,
        },
    )


def _insert_agent(
    conn, *, agent_id: int, name: str, owner_id: int | None, version: str = "latest"
) -> None:
    now = datetime(2026, 4, 18)
    conn.execute(
        text(
            "INSERT INTO agents "
            "(id, tenant_id, name, agent_type, config, "
            "created_at, updated_at, owner_id, version) VALUES "
            "(:id, 'default', :name, 'http', '{}', "
            ":now, :now, :owner, :ver)"
        ),
        {"id": agent_id, "name": name, "owner": owner_id, "ver": version, "now": now},
    )


def test_migration_backfills_ownerless_to_admin(db_at_phase2):
    """Ownerless rows get backfilled to the lowest-id admin user."""
    url, env = db_at_phase2

    engine = create_engine(url)
    with engine.begin() as conn:
        _insert_user(conn, 42, "admin", is_admin=True)
        _insert_user(conn, 99, "later-admin", is_admin=True)
        _insert_agent(conn, agent_id=1, name="orphan", owner_id=None)
    engine.dispose()

    _alembic_ok(env, "upgrade", "b8c9d0e1f2a3")

    engine = create_engine(url)
    with engine.connect() as conn:
        owner = conn.execute(
            text("SELECT owner_id FROM agents WHERE id = 1")
        ).scalar_one()
        assert owner == 42
    engine.dispose()


def test_migration_drops_partial_unique_index(db_at_phase2):
    """After upgrade, the partial unique index must be gone."""
    url, env = db_at_phase2
    _alembic_ok(env, "upgrade", "b8c9d0e1f2a3")

    engine = create_engine(url)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' "
                "AND name='uq_agent_ownerless_tenant_name_version'"
            )
        ).first()
        assert row is None
    engine.dispose()


def test_migration_makes_owner_id_not_null(db_at_phase2):
    """After upgrade, inserting a row with NULL owner_id must fail."""
    url, env = db_at_phase2
    _alembic_ok(env, "upgrade", "b8c9d0e1f2a3")

    engine = create_engine(url)
    agent_id_col = next(
        c for c in inspect(engine).get_columns("agents") if c["name"] == "owner_id"
    )
    assert agent_id_col["nullable"] is False
    engine.dispose()


def test_migration_aborts_when_no_admin_exists(db_at_phase2):
    """Upgrade must refuse when ownerless rows exist but no admin does."""
    url, env = db_at_phase2

    engine = create_engine(url)
    with engine.begin() as conn:
        _insert_user(conn, 1, "nobody", is_admin=False)
        _insert_agent(conn, agent_id=1, name="orphan", owner_id=None)
    engine.dispose()

    result = _alembic(env, "upgrade", "b8c9d0e1f2a3")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "no admin user" in combined.lower() or "Cannot backfill" in combined


def test_migration_downgrade_restores_partial_index(db_at_phase2):
    """Downgrade must re-create the ownerless partial unique index."""
    url, env = db_at_phase2
    _alembic_ok(env, "upgrade", "b8c9d0e1f2a3")
    _alembic_ok(env, "downgrade", "a7b8c9d0e1f2")

    engine = create_engine(url)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' "
                "AND name='uq_agent_ownerless_tenant_name_version'"
            )
        ).first()
        assert row is not None
    owner_col = next(
        c for c in inspect(engine).get_columns("agents") if c["name"] == "owner_id"
    )
    assert owner_col["nullable"] is True
    engine.dispose()
