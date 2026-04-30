"""Regression tests for additive-column schema reconciliation in init_database.

These tests drive the implementation of `_reconcile_additive_columns(db)` in
`packages/atp-dashboard/atp/dashboard/database.py`. The reconciler must bring a
legacy `game_results` table schema in line with the current ORM model by:

- Adding missing additive (nullable or defaulted) columns via ALTER TABLE.
- Creating missing ORM-defined indexes by name.
- Being idempotent on repeated invocations.
- Playing nicely with fresh installs where create_all already provisioned
  everything.

A temporary *file-based* SQLite DB is used (not :memory:) so that we can
pre-seed a legacy schema via raw sqlite3 *before* `init_database()` constructs
its own engine against the same URL.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from atp.dashboard.database import (
    init_database,
    set_database,
)
from atp.dashboard.models import Base, GameResult

# --- Legacy schema (pre-migration 7a1c3d9e4b02) ----------------------------

LEGACY_GAME_RESULTS_DDL = """
CREATE TABLE game_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    game_name VARCHAR(255) NOT NULL,
    game_type VARCHAR(50) NOT NULL DEFAULT 'normal_form',
    num_players INTEGER DEFAULT 2,
    num_rounds INTEGER DEFAULT 1,
    num_episodes INTEGER DEFAULT 1,
    status VARCHAR(20) DEFAULT 'running',
    created_at DATETIME,
    completed_at DATETIME,
    players_json JSON,
    payoff_matrix_json JSON,
    strategy_timeline_json JSON,
    cooperation_dynamics_json JSON,
    episodes_json JSON,
    metadata_json JSON
);
"""

NEW_EL_FAROL_COLUMNS = (
    "match_id",
    "game_version",
    "num_days",
    "num_slots",
    "max_intervals",
    "max_total_slots",
    "capacity_ratio",
    "capacity_threshold",
    "actions_json",
    "day_aggregates_json",
    "round_payoffs_json",
    "agents_json",
)


# --- Helpers ----------------------------------------------------------------


def _seed_legacy_game_results(db_path: Path) -> None:
    """Create the pre-migration game_results table with raw sqlite3."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(LEGACY_GAME_RESULTS_DDL)
        conn.commit()
    finally:
        conn.close()


async def _inspect_columns(url: str, table: str) -> list[dict]:
    """Return a list of column dicts for `table` via a short-lived engine."""
    engine = create_async_engine(url, connect_args={"check_same_thread": False})
    try:
        async with engine.connect() as conn:

            def _gather(sync_conn):
                from sqlalchemy import inspect

                insp = inspect(sync_conn)
                return list(insp.get_columns(table))

            cols = await conn.run_sync(_gather)
        return cols
    finally:
        await engine.dispose()


async def _inspect_indexes(url: str, table: str) -> list[dict]:
    """Return a list of index dicts for `table` via a short-lived engine."""
    engine = create_async_engine(url, connect_args={"check_same_thread": False})
    try:
        async with engine.connect() as conn:

            def _gather(sync_conn):
                from sqlalchemy import inspect

                insp = inspect(sync_conn)
                return list(insp.get_indexes(table))

            idx = await conn.run_sync(_gather)
        return idx
    finally:
        await engine.dispose()


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def sqlite_file(tmp_path: Path) -> Path:
    """Path to a fresh temp SQLite file (not yet created)."""
    return tmp_path / "dashboard.db"


@pytest.fixture
def sqlite_url(sqlite_file: Path) -> str:
    """Async SQLite URL pointing at the temp file."""
    return f"sqlite+aiosqlite:///{sqlite_file}"


@pytest.fixture
async def reset_global_db() -> AsyncGenerator[None, None]:
    """Ensure module-level `_database` is cleared after each test.

    The engine is disposed by the test body (via `close()`) before this
    fixture's teardown runs, so the temp file can be removed cleanly.
    """
    yield
    # Teardown: flush module global, regardless of test outcome.
    set_database(None)  # type: ignore[arg-type]


# --- Tests ------------------------------------------------------------------


@pytest.mark.anyio
async def test_init_database_adds_missing_game_result_columns_on_legacy_schema(
    sqlite_file: Path,
    sqlite_url: str,
    reset_global_db: None,
) -> None:
    """GIVEN a legacy `game_results` schema missing the Phase 7 El Farol columns
    WHEN init_database(url) is awaited
    THEN inspect(engine).get_columns('game_results') contains every new column.
    """
    # GIVEN — pre-seed legacy schema
    _seed_legacy_game_results(sqlite_file)

    # sanity: the legacy schema does NOT contain the new columns yet
    pre_cols = await _inspect_columns(sqlite_url, "game_results")
    pre_names = {c["name"] for c in pre_cols}
    for col in NEW_EL_FAROL_COLUMNS:
        assert col not in pre_names, f"setup error: {col} already present"

    # WHEN
    db = await init_database(url=sqlite_url)
    try:
        # THEN
        post_cols = await _inspect_columns(sqlite_url, "game_results")
        post_names = {c["name"] for c in post_cols}
        missing = [c for c in NEW_EL_FAROL_COLUMNS if c not in post_names]
        assert not missing, (
            f"init_database failed to reconcile additive columns; "
            f"still missing: {missing}"
        )
    finally:
        await db.close()


@pytest.mark.anyio
async def test_init_database_enables_select_after_legacy_schema_upgrade(
    sqlite_file: Path,
    sqlite_url: str,
    reset_global_db: None,
) -> None:
    """GIVEN legacy schema pre-seeded
    WHEN init_database runs and a session issues select(GameResult)
    THEN no OperationalError is raised; result set is empty.
    """
    # GIVEN
    _seed_legacy_game_results(sqlite_file)

    # WHEN
    db = await init_database(url=sqlite_url)
    try:
        async with db.session() as session:
            result = await session.execute(select(GameResult))
            rows = result.scalars().all()

        # THEN
        assert rows == []
    finally:
        await db.close()


@pytest.mark.anyio
async def test_init_database_insert_and_readback_after_reconcile(
    sqlite_file: Path,
    sqlite_url: str,
    reset_global_db: None,
) -> None:
    """GIVEN legacy schema pre-seeded
    WHEN init_database runs, then a GameResult row with new-column values is
         inserted and re-read
    THEN all new fields are preserved intact.
    """
    # GIVEN
    _seed_legacy_game_results(sqlite_file)

    # WHEN
    db = await init_database(url=sqlite_url)
    try:
        async with db.session() as session:
            game = GameResult(
                game_name="El Farol",
                game_type="el_farol",
                num_players=2,
                num_rounds=10,
                num_episodes=1,
                status="completed",
                created_at=datetime.now(),
                completed_at=datetime.now(),
                match_id="m-run-1",
                actions_json=[{"day": 1, "agent_id": "p0"}],
                capacity_threshold=6,
                agents_json=[
                    {
                        "agent_id": "p0",
                        "display_name": "tft",
                        "user_id": "u1",
                    }
                ],
            )
            session.add(game)
            await session.commit()
            await session.refresh(game)
            row_id = game.id

        async with db.session() as session:
            result = await session.execute(
                select(GameResult).where(GameResult.id == row_id)
            )
            gr = result.scalar_one()

        # THEN
        assert gr.match_id == "m-run-1"
        assert gr.actions_json == [{"day": 1, "agent_id": "p0"}]
        assert gr.capacity_threshold == 6
        assert gr.agents_json == [
            {
                "agent_id": "p0",
                "display_name": "tft",
                "user_id": "u1",
            }
        ]
    finally:
        await db.close()


@pytest.mark.anyio
async def test_init_database_creates_missing_index_on_legacy_game_results(
    sqlite_file: Path,
    sqlite_url: str,
    reset_global_db: None,
) -> None:
    """GIVEN legacy schema (no idx_game_result_match / idx_game_result_game_completed)
    WHEN init_database runs
    THEN inspect(engine).get_indexes('game_results') lists both new index names.
    """
    # GIVEN
    _seed_legacy_game_results(sqlite_file)

    # sanity: the legacy schema has no ORM-defined extra indexes
    pre_indexes = await _inspect_indexes(sqlite_url, "game_results")
    pre_names = {i["name"] for i in pre_indexes}
    assert "idx_game_result_match" not in pre_names
    assert "idx_game_result_game_completed" not in pre_names

    # WHEN
    db = await init_database(url=sqlite_url)
    try:
        # THEN
        post_indexes = await _inspect_indexes(sqlite_url, "game_results")
        post_names = {i["name"] for i in post_indexes}
        assert "idx_game_result_match" in post_names, (
            f"missing idx_game_result_match; indexes={post_names}"
        )
        assert "idx_game_result_game_completed" in post_names, (
            f"missing idx_game_result_game_completed; indexes={post_names}"
        )
    finally:
        await db.close()


@pytest.mark.anyio
async def test_init_database_is_idempotent_on_already_current_schema(
    sqlite_url: str,
    reset_global_db: None,
) -> None:
    """GIVEN a fresh DB file where init_database has already run once
    WHEN init_database is awaited a second time against the same URL
    THEN no error is raised, and game_results has exactly the ORM column set.
    """
    # GIVEN — first init on a fresh file
    db1 = await init_database(url=sqlite_url)
    await db1.close()
    set_database(None)  # type: ignore[arg-type]

    # WHEN — second init
    db2 = await init_database(url=sqlite_url)
    try:
        # THEN
        cols = await _inspect_columns(sqlite_url, "game_results")
        names = {c["name"] for c in cols}
        expected_n = len(GameResult.__table__.columns)
        assert len(names) == expected_n, (
            f"expected {expected_n} unique columns, got {len(names)}: {names}"
        )
    finally:
        await db2.close()


@pytest.mark.anyio
async def test_init_database_noop_on_legacy_db_when_called_twice(
    sqlite_file: Path,
    sqlite_url: str,
    reset_global_db: None,
) -> None:
    """GIVEN legacy schema pre-seeded and init_database already reconciled once
    WHEN init_database runs a second time
    THEN no error (no duplicate ALTER TABLE) and new columns remain present.
    """
    # GIVEN
    _seed_legacy_game_results(sqlite_file)
    db1 = await init_database(url=sqlite_url)
    await db1.close()
    set_database(None)  # type: ignore[arg-type]

    # WHEN
    db2 = await init_database(url=sqlite_url)
    try:
        # THEN
        cols = await _inspect_columns(sqlite_url, "game_results")
        names = {c["name"] for c in cols}
        for col in NEW_EL_FAROL_COLUMNS:
            assert col in names, f"column {col} dropped on second init"
    finally:
        await db2.close()


@pytest.mark.anyio
async def test_init_database_reconcile_skips_non_existent_tables(
    sqlite_file: Path,
    sqlite_url: str,
    reset_global_db: None,
) -> None:
    """GIVEN a completely fresh file (no tables) — just touch it
    WHEN init_database runs
    THEN all ORM tables are created and reconcile does not raise; in
         particular, game_results exists.
    """
    # GIVEN — create the empty file so its path is valid but has no tables
    sqlite_file.touch()

    # WHEN
    db = await init_database(url=sqlite_url)
    try:
        # THEN — game_results table exists after create_all + reconcile
        cols = await _inspect_columns(sqlite_url, "game_results")
        assert len(cols) > 0, (
            "expected game_results table to exist after init_database; got no columns"
        )
        # And the reconciler didn't strip ORM columns
        names = {c["name"] for c in cols}
        orm_names = {c.name for c in GameResult.__table__.columns}
        missing = orm_names - names
        assert not missing, f"missing ORM columns after fresh init: {missing}"
    finally:
        await db.close()


# Silence unused-import lint for Base (kept in imports per spec).
_ = Base
