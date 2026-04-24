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
    db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY)")
    db.execute("INSERT OR IGNORE INTO users (id) VALUES (1)")


def _seed_legacy_participants(db: sqlite3.Connection) -> None:
    """Recreate pre-853688412c5b participants schema:
    user_id NOT NULL, no builtin_strategy, no agent_id. Includes a
    matching ``agents`` row so the legacy participant can be
    backfilled through the (user_id, agent_name) lookup."""
    db.execute("CREATE TABLE tournaments (id INTEGER PRIMARY KEY)")
    db.execute("INSERT INTO tournaments (id) VALUES (100)")
    db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY)")
    db.execute("INSERT OR IGNORE INTO users (id) VALUES (1)")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL,
            name VARCHAR(100) NOT NULL
        )
        """
    )
    db.execute(
        "INSERT OR IGNORE INTO agents (id, owner_id, name) VALUES (42, 1, 'legacy-bot')"
    )
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

    # Rebuilt schema matches Alembic HEAD: legacy constraint gone, new
    # 4-tuple unique present, purpose CHECK present, NOT NULL columns
    # enforced.
    sql = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='agents'"
    ).fetchone()[0]
    assert "uq_agent_tenant_name " not in sql  # legacy (note trailing space)
    assert "uq_agent_tenant_owner_name_version" in sql
    assert "ck_agents_purpose" in sql
    # NOT NULL declarations for the three columns the old script left nullable.
    info = {row[1]: row for row in db.execute("PRAGMA table_info(agents)")}
    assert info["config"][3] == 1
    assert info["created_at"][3] == 1
    assert info["updated_at"][3] == 1

    # HEAD must NOT carry uq_agent_ownerless_tenant_name_version (dropped
    # by migration b8c9d0e1f2a3 when owner_id became NOT NULL).
    index_names = {row[1] for row in db.execute("PRAGMA index_list(agents)")}
    assert "uq_agent_ownerless_tenant_name_version" not in index_names
    assert "idx_agents_owner_purpose" in index_names

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

    # Rebuilt schema matches Alembic HEAD.
    info = {
        row[1]: row for row in db.execute("PRAGMA table_info(tournament_participants)")
    }
    assert info["user_id"][3] == 0  # nullable
    assert info["agent_name"][2] == "VARCHAR(200)"  # HEAD length, not 100
    assert "builtin_strategy" in info
    assert "agent_id" in info

    table_sql = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' "
        "AND name='tournament_participants'"
    ).fetchone()[0]
    # Strict XOR check — not the permissive both-NULL variant.
    assert "ck_participants_agent_xor_builtin" in table_sql
    assert "agent_id IS NULL AND builtin_strategy IS NULL" not in table_sql

    # Partial indexes declared by Alembic HEAD, with matching names.
    index_names = {
        row[1] for row in db.execute("PRAGMA index_list(tournament_participants)")
    }
    assert "uq_participant_tournament_agent" in index_names
    assert "uq_participant_agent_active" in index_names
    assert "idx_participants_builtin" in index_names

    for idx_name, expected_where in (
        ("uq_participant_tournament_agent", "agent_id IS NOT NULL"),
        ("uq_participant_agent_active", "agent_id IS NOT NULL AND released_at IS NULL"),
        ("idx_participants_builtin", "builtin_strategy IS NOT NULL"),
    ):
        idx_sql = db.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
            (idx_name,),
        ).fetchone()[0]
        assert expected_where in idx_sql, f"{idx_name}: {idx_sql!r}"

    # We can now insert a builtin-style participant with user_id=NULL.
    db.execute(
        "INSERT INTO tournament_participants "
        "(tournament_id, user_id, agent_name, joined_at, builtin_strategy) "
        "VALUES (100, NULL, 'el_farol/contrarian', "
        "'2026-01-01 00:00:00', 'el_farol/contrarian')"
    )

    # Strict XOR rejects a row with neither agent_id nor builtin_strategy.
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO tournament_participants "
            "(tournament_id, user_id, agent_name, joined_at) "
            "VALUES (100, 1, 'orphan-bot', '2026-01-01 00:00:00')"
        )

    # Running again is a no-op.
    changed_again = reconcile_module.fix_participants_user_id_nullable(db)
    assert changed_again is False


def test_fix_participants_drops_orphan_rows_before_xor_check(
    tmp_path, reconcile_module
) -> None:
    """Pre-PR-4 participant rows (agent_id IS NULL AND builtin_strategy IS
    NULL) must be either backfilled via (user_id, agent_name) lookup or
    deleted — otherwise the XOR CHECK would fail the rebuild."""
    db = sqlite3.connect(tmp_path / "t.db")

    # Seed a participant schema that already has agent_id + builtin_strategy
    # columns (so needs_agent_id / needs_builtin are False and the backfill
    # branch actually runs) but still has user_id as NOT NULL.
    db.execute("CREATE TABLE tournaments (id INTEGER PRIMARY KEY)")
    db.execute("INSERT INTO tournaments (id) VALUES (100)")
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    db.execute("INSERT INTO users (id) VALUES (7)")
    db.execute(
        """
        CREATE TABLE agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL,
            name VARCHAR(100) NOT NULL,
            version VARCHAR(50) NOT NULL DEFAULT 'latest'
        )
        """
    )
    db.execute("INSERT INTO agents (id, owner_id, name) VALUES (42, 7, 'real-bot')")
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
            agent_id INTEGER,
            builtin_strategy VARCHAR(64)
        )
        """
    )
    # Row 1: resolvable via (user_id=7, name='real-bot') → agent_id=42
    db.execute(
        "INSERT INTO tournament_participants "
        "(tournament_id, user_id, agent_name, joined_at) "
        "VALUES (100, 7, 'real-bot', '2026-01-01 00:00:00')"
    )
    # Row 2: orphan — no matching agent, will be deleted
    db.execute(
        "INSERT INTO tournament_participants "
        "(tournament_id, user_id, agent_name, joined_at) "
        "VALUES (100, 7, 'ghost-bot', '2026-01-01 00:00:00')"
    )

    changed = reconcile_module.fix_participants_user_id_nullable(db)
    assert changed is True

    rows = db.execute(
        "SELECT agent_name, agent_id, builtin_strategy FROM tournament_participants"
    ).fetchall()
    assert rows == [("real-bot", 42, None)]


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
