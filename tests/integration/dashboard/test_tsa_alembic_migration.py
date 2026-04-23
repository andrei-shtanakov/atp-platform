"""Alembic round-trip test for LABS-TSA PR-1 migration."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect


@pytest.mark.anyio
async def test_migration_up_down_up_clean_sqlite() -> None:
    with tempfile.TemporaryDirectory() as td:
        dbpath = Path(td) / "atp.db"
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
        subprocess.check_call(["uv", "run", "alembic", "downgrade", "-1"], env=env)
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
