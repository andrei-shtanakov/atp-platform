"""One-off reconcile for prod DBs that pre-date Alembic-on-deploy.

History: ATP boots with ``create_all()`` + ``_add_missing_columns()`` and
does NOT apply Alembic migrations. As the model evolved, prod drifted
from HEAD in ways that neither of those helpers can repair:

- Drops of old unique constraints (e.g. ``uq_agent_tenant_name``)
- NOT NULL → NULL alterations (e.g. ``tournament_participants.user_id``)
- New CHECK constraints (e.g. ``ck_agents_purpose``)
- Partial unique indexes (e.g. ``uq_agent_ownerless_tenant_name_version``)

This script rebuilds the drifted tables to match HEAD and then stamps
Alembic at HEAD so the new ``alembic upgrade head`` entrypoint has
a valid starting point. Safe to re-run — each step is idempotent.

Usage (inside the platform container):

    uv run --no-sync python /app/scripts/ops/reconcile_prod_schema.py

Requires ``ATP_DATABASE_URL`` to point at the SQLite file (default in
docker-compose.yml: ``sqlite+aiosqlite:////data/atp.db``).
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path


def _db_path_from_env() -> Path:
    url = os.environ.get("ATP_DATABASE_URL", "")
    # sqlite+aiosqlite:////data/atp.db  or  sqlite:////data/atp.db
    if "sqlite" not in url:
        sys.exit(
            "ATP_DATABASE_URL is not sqlite — this reconcile script is "
            "sqlite-only. For postgres, run Alembic directly."
        )
    path = url.split("://", 1)[1].lstrip("/")
    return Path("/" + path)


def _col_info(db: sqlite3.Connection, table: str) -> dict[str, tuple]:
    return {row[1]: row for row in db.execute(f"PRAGMA table_info({table})")}


def _index_list(db: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in db.execute(f"PRAGMA index_list({table})")]


def _table_sql(db: sqlite3.Connection, table: str) -> str:
    row = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row[0] if row else ""


def fix_agents_legacy_unique(db: sqlite3.Connection) -> bool:
    """Drop ``uq_agent_tenant_name`` unique constraint.

    Migration ``d7f3a2b1c4e5`` replaced it with
    ``uq_agent_tenant_owner_name_version``. If both are present, or
    only the legacy one is present, rebuild the table with just the
    new 4-tuple unique constraint.
    """
    sql = _table_sql(db, "agents")
    if not sql:
        return False
    if "uq_agent_tenant_name" not in sql:
        print("  [agents] legacy uq_agent_tenant_name already gone — skip")
        return False

    print("  [agents] rebuilding: dropping legacy uq_agent_tenant_name")
    db.execute(
        """
        CREATE TABLE agents_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
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
            FOREIGN KEY(owner_id) REFERENCES users(id),
            CONSTRAINT uq_agent_tenant_owner_name_version
                UNIQUE (tenant_id, owner_id, name, version),
            CONSTRAINT ck_agents_purpose
                CHECK (purpose IN ('benchmark','tournament'))
        )
        """
    )

    live_cols = _col_info(db, "agents")
    # Carry over every column that exists on the live table, defaulting
    # values we need to preserve; the CREATE TABLE above ships the
    # current model's columns. Anything missing on live is absent from
    # the SELECT — SQLite uses the DEFAULT on the new table.
    carry = [
        c
        for c in (
            "id",
            "tenant_id",
            "name",
            "agent_type",
            "config",
            "description",
            "created_at",
            "updated_at",
            "owner_id",
            "version",
            "deleted_at",
            "purpose",
        )
        if c in live_cols
    ]
    col_list = ",".join(carry)
    db.execute(f"INSERT INTO agents_new ({col_list}) SELECT {col_list} FROM agents")

    db.execute("DROP TABLE agents")
    db.execute("ALTER TABLE agents_new RENAME TO agents")

    # Recreate indexes the model declares.
    for name, cols in (
        ("idx_agent_name", ["name"]),
        ("idx_agent_tenant", ["tenant_id"]),
        ("idx_agent_owner", ["owner_id"]),
        ("idx_agents_owner_purpose", ["owner_id", "purpose"]),
        ("ix_agents_tenant_id", ["tenant_id"]),
    ):
        db.execute(f"CREATE INDEX IF NOT EXISTS {name} ON agents ({', '.join(cols)})")
    # Partial unique index for ownerless rows (migration e1b2c3d4f5a6).
    db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_ownerless_tenant_name_version "
        "ON agents (tenant_id, name, version) WHERE owner_id IS NULL"
    )
    return True


def fix_participants_user_id_nullable(db: sqlite3.Connection) -> bool:
    """Make ``tournament_participants.user_id`` nullable + add
    ``builtin_strategy`` if missing.

    Migration ``853688412c5b`` relaxed ``user_id`` to allow builtin
    sparring partners, which carry no user. SQLite can't ALTER COLUMN
    so we rebuild the table.
    """
    cols = _col_info(db, "tournament_participants")
    if not cols:
        return False

    user_id_col = cols.get("user_id")
    # notnull flag at index 3 (0-indexed): 1 => NOT NULL, 0 => NULL
    needs_nullable = user_id_col is not None and user_id_col[3] == 1
    needs_builtin = "builtin_strategy" not in cols
    needs_agent_id = "agent_id" not in cols

    if not (needs_nullable or needs_builtin or needs_agent_id):
        print("  [tournament_participants] already migrated — skip")
        return False

    print(
        f"  [tournament_participants] rebuilding "
        f"(user_id_nullable={needs_nullable}, "
        f"add_builtin_strategy={needs_builtin}, add_agent_id={needs_agent_id})"
    )

    db.execute(
        """
        CREATE TABLE tournament_participants_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            tournament_id INTEGER NOT NULL,
            user_id INTEGER,
            agent_name VARCHAR(100) NOT NULL,
            joined_at DATETIME NOT NULL,
            total_score FLOAT,
            released_at DATETIME,
            agent_id INTEGER,
            builtin_strategy VARCHAR(64),
            FOREIGN KEY(tournament_id) REFERENCES tournaments(id),
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(agent_id) REFERENCES agents(id),
            CONSTRAINT ck_participants_agent_xor_builtin CHECK (
                (agent_id IS NOT NULL AND builtin_strategy IS NULL) OR
                (agent_id IS NULL AND builtin_strategy IS NOT NULL) OR
                (agent_id IS NULL AND builtin_strategy IS NULL)
            )
        )
        """
    )

    carry = [
        c
        for c in (
            "id",
            "tournament_id",
            "user_id",
            "agent_name",
            "joined_at",
            "total_score",
            "released_at",
            "agent_id",
            "builtin_strategy",
        )
        if c in cols
    ]
    col_list = ",".join(carry)
    db.execute(
        f"INSERT INTO tournament_participants_new ({col_list}) "
        f"SELECT {col_list} FROM tournament_participants"
    )

    db.execute("DROP TABLE tournament_participants")
    db.execute(
        "ALTER TABLE tournament_participants_new RENAME TO tournament_participants"
    )

    for name, cols_ in (
        ("idx_participant_tournament", ["tournament_id"]),
        ("idx_participant_user", ["user_id"]),
        ("idx_participant_agent", ["agent_id"]),
    ):
        db.execute(
            f"CREATE INDEX IF NOT EXISTS {name} "
            f"ON tournament_participants ({', '.join(cols_)})"
        )

    # Partial unique index from migration d2e5a1c7f3b8 (PR-6): an agent
    # can participate in only one active tournament at a time.
    db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS "
        "uq_tournament_participant_active_agent "
        "ON tournament_participants (agent_id) "
        "WHERE agent_id IS NOT NULL AND released_at IS NULL"
    )
    return True


def stamp_alembic_head(db: sqlite3.Connection, head: str) -> bool:
    """Write or update ``alembic_version`` so the next
    ``alembic upgrade head`` starts from the right revision.

    After this, future migrations added on top of HEAD will run
    cleanly. Runs without Alembic as a dependency so this script is
    safe to exec in a minimal python container.
    """
    existing = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
    ).fetchone()
    if not existing:
        db.execute(
            "CREATE TABLE alembic_version ("
            "version_num VARCHAR(32) NOT NULL, "
            "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
        )
    current = db.execute("SELECT version_num FROM alembic_version").fetchall()
    if any(row[0] == head for row in current):
        print(f"  [alembic_version] already at {head} — skip")
        return False

    db.execute("DELETE FROM alembic_version")
    db.execute("INSERT INTO alembic_version (version_num) VALUES (?)", (head,))
    print(f"  [alembic_version] stamped at {head}")
    return True


def main() -> int:
    # HEAD must match ``uv run alembic -c alembic.ini heads`` at the time
    # the reconcile script was last touched. Update this constant when
    # adding new drift fixes.
    ALEMBIC_HEAD = "d2e5a1c7f3b8"

    db_path = _db_path_from_env()
    if not db_path.exists():
        print(f"No DB at {db_path} — nothing to reconcile")
        return 0

    print(f"Reconciling {db_path} against Alembic HEAD {ALEMBIC_HEAD}")
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA foreign_keys = OFF")
    try:
        db.execute("BEGIN")
        fix_agents_legacy_unique(db)
        fix_participants_user_id_nullable(db)
        stamp_alembic_head(db, ALEMBIC_HEAD)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.execute("PRAGMA foreign_keys = ON")
        db.close()

    print("Reconcile complete. Safe to re-run if needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
