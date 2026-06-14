"""Integration tests for the /ui/eval-leaderboard dashboard page (SP-3)."""

import os
from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, SuiteExecution
from atp.dashboard.v2.config import DashboardConfig, get_config
from atp.dashboard.v2.factory import create_app


@pytest.fixture
async def fresh_app():
    # Save env vars so they can be restored — leaking ATP_DISABLE_AUTH=true
    # into the process env breaks other tests that assert 401 behaviour.
    saved_env = {
        k: os.environ.get(k)
        for k in ("ATP_SECRET_KEY", "ATP_DISABLE_AUTH", "ATP_RATE_LIMIT_ENABLED")
    }

    os.environ["ATP_SECRET_KEY"] = "test-secret"
    os.environ["ATP_DISABLE_AUTH"] = "true"
    os.environ["ATP_RATE_LIMIT_ENABLED"] = "false"
    get_config.cache_clear()

    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)

    config = DashboardConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        debug=True,
        secret_key="test-secret",
        disable_auth=True,
        rate_limit_enabled=False,
    )
    app = create_app(config=config)

    yield app, db

    await db.close()
    set_database(None)  # type: ignore[arg-type]
    for key, old_val in saved_env.items():
        if old_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_val
    get_config.cache_clear()


@pytest.mark.anyio
async def test_eval_leaderboard_ranks_by_critical_pass_rate(fresh_app: tuple) -> None:
    """Agents are ranked by critical_pass_rate descending for the suite."""
    app, db = fresh_app
    now = datetime.now()

    async with db.session() as session:
        session.add_all(
            [
                SuiteExecution(
                    suite_name="code-review",
                    agent_name="claude_code",
                    started_at=now - timedelta(minutes=2),
                    status="completed",
                    critical_pass_rate=0.8,
                ),
                SuiteExecution(
                    suite_name="code-review",
                    agent_name="anthropic_api",
                    started_at=now - timedelta(minutes=1),
                    status="completed",
                    critical_pass_rate=0.6,
                ),
            ]
        )
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-leaderboard?suite_name=code-review")

    assert resp.status_code == 200
    body = resp.text
    assert "claude_code" in body
    assert "anthropic_api" in body
    # 0.8 ranked above 0.6.
    assert body.index("claude_code") < body.index("anthropic_api")


@pytest.mark.anyio
async def test_eval_leaderboard_empty_db(fresh_app: tuple) -> None:
    """Empty store renders the page (selector + hint) without crashing."""
    app, _ = fresh_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-leaderboard")
    assert resp.status_code == 200
