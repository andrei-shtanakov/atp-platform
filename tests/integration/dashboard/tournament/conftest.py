"""Shared integration fixtures for tournament tests."""

import os
import subprocess

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# LABS-TSA PR-6 (post-review): TournamentService.join now calls
# ``get_config()`` from the by-name auto-provision path to enforce the
# per-user tournament-agent quota. ``DashboardConfig`` rejects empty
# ATP_SECRET_KEY in non-debug mode — set a benign test secret here the
# same way the unit-test conftest does so these integration tests can
# run in a fresh shell.
os.environ.setdefault("ATP_SECRET_KEY", "integration-test-secret-key")
os.environ.setdefault("ATP_DEBUG", "true")


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

    env = {**os.environ, "ATP_DASHBOARD_DATABASE_URL": f"sqlite:///{db_path}"}
    result = subprocess.run(
        ["uv", "run", "alembic", "-n", "dashboard", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"alembic upgrade failed: {result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(tournament_db):
    return async_sessionmaker(tournament_db, expire_on_commit=False)
