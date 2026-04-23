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
        # downgrade one step — PR-3 (agent_purpose column)
        subprocess.check_call(["uv", "run", "alembic", "downgrade", "-1"], env=env)
        eng = create_engine(f"sqlite:///{dbpath}")
        insp = inspect(eng)
        # PR-3 column gone, but PR-1 columns should remain
        assert "agent_purpose" not in {
            c["name"] for c in insp.get_columns("api_tokens")
        }
        assert "purpose" in {c["name"] for c in insp.get_columns("agents")}
        eng.dispose()
        # upgrade again
        subprocess.check_call(["uv", "run", "alembic", "upgrade", "head"], env=env)
        eng = create_engine(f"sqlite:///{dbpath}")
        insp = inspect(eng)
        assert "purpose" in {c["name"] for c in insp.get_columns("agents")}
        assert "agent_purpose" in {c["name"] for c in insp.get_columns("api_tokens")}
        eng.dispose()


@pytest.mark.anyio
async def test_pr3_migration_adds_agent_purpose_column() -> None:
    """LABS-TSA PR-3: api_tokens.agent_purpose nullable column round-trip."""
    with tempfile.TemporaryDirectory() as td:
        dbpath = Path(td) / "atp.db"
        env = {**os.environ, "ATP_DATABASE_URL": f"sqlite:///{dbpath}"}
        subprocess.check_call(["uv", "run", "alembic", "upgrade", "head"], env=env)
        eng = create_engine(f"sqlite:///{dbpath}")
        insp = inspect(eng)
        cols = {c["name"]: c for c in insp.get_columns("api_tokens")}
        assert "agent_purpose" in cols
        assert cols["agent_purpose"]["nullable"] is True
        eng.dispose()
