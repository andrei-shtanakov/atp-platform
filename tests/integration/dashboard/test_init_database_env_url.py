"""Regression test: init_database() must respect ATP_DATABASE_URL.

Dashboard CLI writers (``atp game run`` → ``_store_game_result``) call
``init_database()`` with no URL so they can share whatever DB the
dashboard server is configured against. Before this fix they silently
defaulted to ``~/.atp/dashboard.db`` — which in the production Docker
image is a read-only path and crashed with ``Permission denied``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from atp.dashboard.database import init_database, set_database


@pytest.mark.anyio
async def test_init_database_reads_env_url(tmp_path: Path) -> None:
    """GIVEN ATP_DATABASE_URL set but no url arg
    WHEN init_database() runs
    THEN the engine connects to the env-specified DB, not the default.
    """
    db_path = tmp_path / "custom.db"
    url = f"sqlite+aiosqlite:///{db_path}"

    prev = os.environ.get("ATP_DATABASE_URL")
    os.environ["ATP_DATABASE_URL"] = url
    try:
        db = await init_database()
        try:
            # The engine's URL should match what we set.
            assert str(db.engine.url) == url
            # And the DB file gets created.
            assert db_path.exists()
        finally:
            await db.close()
            set_database(None)  # type: ignore[arg-type]
    finally:
        if prev is None:
            os.environ.pop("ATP_DATABASE_URL", None)
        else:
            os.environ["ATP_DATABASE_URL"] = prev


@pytest.mark.anyio
async def test_init_database_explicit_url_wins_over_env(tmp_path: Path) -> None:
    """Explicit ``url=`` takes precedence over the env var."""
    env_path = tmp_path / "env.db"
    explicit_path = tmp_path / "explicit.db"
    env_url = f"sqlite+aiosqlite:///{env_path}"
    explicit_url = f"sqlite+aiosqlite:///{explicit_path}"

    prev = os.environ.get("ATP_DATABASE_URL")
    os.environ["ATP_DATABASE_URL"] = env_url
    try:
        db = await init_database(url=explicit_url)
        try:
            assert str(db.engine.url) == explicit_url
            assert explicit_path.exists()
            assert not env_path.exists()
        finally:
            await db.close()
            set_database(None)  # type: ignore[arg-type]
    finally:
        if prev is None:
            os.environ.pop("ATP_DATABASE_URL", None)
        else:
            os.environ["ATP_DATABASE_URL"] = prev
