"""Tests for scripts/ops/reconcile_prod_schema.py.

The reconcile script operates on raw sqlite3 so we test it the same
way: build a pretend-prod DB by hand, call the fix_* helpers, assert
the resulting schema matches HEAD.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "ops" / "reconcile_prod_schema.py"
)


@pytest.fixture(scope="module")
def reconcile_module():
    """Load reconcile_prod_schema.py as a module for direct function access."""
    spec = importlib.util.spec_from_file_location("reconcile_prod_schema", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reconcile_prod_schema"] = mod
    spec.loader.exec_module(mod)
    return mod


def _seed_legacy_agents(db: sqlite3.Connection) -> None:
    """Recreate the pre-d7f3a2b1c4e5 agents schema: `uq_agent_tenant_name`."""
    db.execute(
        """
        CREATE TABLE agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
            name VARCHAR(100) NOT NULL,
            agent_type VARCHAR(50) NOT NULL,
            config JSON,
            description TEXT,
            created_at DATETIME,
            updated_at DATETIME,
            owner_id INTEGER NOT NULL,
            version VARCHAR(50) NOT NULL DEFAULT 'latest',
            deleted_at DATETIME,
            purpose VARCHAR(20) NOT NULL DEFAULT 'benchmark',
            CONSTRAINT uq_agent_tenant_name UNIQUE (tenant_id, name)
        )
        """
    )
    db.execute(
        "INSERT INTO agents (tenant_id, name, agent_type, owner_id, version, purpose) "
        "VALUES ('default', 'bot-a', 'mcp', 1, 'latest', 'tournament')"
    )
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    db.execute("INSERT INTO users (id) VALUES (1)")


def _seed_legacy_participants(db: sqlite3.Connection) -> None:
    """Recreate pre-853688412c5b participants schema:
    user_id NOT NULL, no builtin_strategy, no agent_id."""
    db.execute("CREATE TABLE tournaments (id INTEGER PRIMARY KEY)")
    db.execute("INSERT INTO tournaments (id) VALUES (100)")
    db.execute(
        """
        CREATE TABLE tournament_participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tournament_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            agent_name VARCHAR(100) NOT NULL,
            joined_at DATETIME NOT NULL,
            total_score FLOAT,
            released_at DATETIME,
            FOREIGN KEY(tournament_id) REFERENCES tournaments(id)
        )
        """
    )
    db.execute(
        "INSERT INTO tournament_participants "
        "(tournament_id, user_id, agent_name, joined_at) "
        "VALUES (100, 1, 'legacy-bot', '2026-01-01 00:00:00')"
    )


def test_fix_agents_rebuild_drops_legacy_unique(tmp_path, reconcile_module) -> None:
    db = sqlite3.connect(tmp_path / "t.db")
    _seed_legacy_agents(db)

    changed = reconcile_module.fix_agents_legacy_unique(db)
    assert changed is True

    # Legacy constraint is gone from the new CREATE TABLE SQL.
    sql = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='agents'"
    ).fetchone()[0]
    assert "uq_agent_tenant_name" not in sql
    assert "uq_agent_tenant_owner_name_version" in sql

    # Data survived.
    rows = db.execute(
        "SELECT name, owner_id, purpose FROM agents ORDER BY id"
    ).fetchall()
    assert rows == [("bot-a", 1, "tournament")]

    # Running again is a no-op.
    changed_again = reconcile_module.fix_agents_legacy_unique(db)
    assert changed_again is False


def test_fix_participants_rebuild_makes_user_id_nullable(
    tmp_path, reconcile_module
) -> None:
    db = sqlite3.connect(tmp_path / "t.db")
    _seed_legacy_participants(db)

    changed = reconcile_module.fix_participants_user_id_nullable(db)
    assert changed is True

    # user_id is nullable now (notnull flag at index 3 == 0).
    info = {
        row[1]: row for row in db.execute("PRAGMA table_info(tournament_participants)")
    }
    assert info["user_id"][3] == 0
    assert "builtin_strategy" in info
    assert "agent_id" in info

    # We can now insert a builtin-style participant with user_id=NULL.
    db.execute(
        "INSERT INTO tournament_participants "
        "(tournament_id, user_id, agent_name, joined_at, builtin_strategy) "
        "VALUES (100, NULL, 'el_farol/contrarian', "
        "'2026-01-01 00:00:00', 'el_farol/contrarian')"
    )

    # Running again is a no-op.
    changed_again = reconcile_module.fix_participants_user_id_nullable(db)
    assert changed_again is False


def test_stamp_alembic_head_idempotent(tmp_path, reconcile_module) -> None:
    db = sqlite3.connect(tmp_path / "t.db")

    first = reconcile_module.stamp_alembic_head(db, "d2e5a1c7f3b8")
    assert first is True

    version = db.execute("SELECT version_num FROM alembic_version").fetchall()
    assert version == [("d2e5a1c7f3b8",)]

    second = reconcile_module.stamp_alembic_head(db, "d2e5a1c7f3b8")
    assert second is False  # Already at head, nothing to do.


def test_full_reconcile_is_idempotent(tmp_path, reconcile_module) -> None:
    """Run every fix helper in sequence; re-run; expect no further changes."""
    db = sqlite3.connect(tmp_path / "t.db")
    _seed_legacy_agents(db)
    _seed_legacy_participants(db)

    assert reconcile_module.fix_agents_legacy_unique(db) is True
    assert reconcile_module.fix_participants_user_id_nullable(db) is True
    assert reconcile_module.stamp_alembic_head(db, "d2e5a1c7f3b8") is True

    # Second pass.
    assert reconcile_module.fix_agents_legacy_unique(db) is False
    assert reconcile_module.fix_participants_user_id_nullable(db) is False
    assert reconcile_module.stamp_alembic_head(db, "d2e5a1c7f3b8") is False
