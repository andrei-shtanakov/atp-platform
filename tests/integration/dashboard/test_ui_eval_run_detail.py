"""Integration tests for the /ui/eval-run/{suite}/{agent} drill-down page (Task 5)."""

import os
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base
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
async def test_eval_run_detail_renders_axis_sweep(fresh_app: tuple) -> None:
    from atp.dashboard.storage import ResultStorage

    app, db = fresh_app
    async with db.session() as session:
        storage = ResultStorage(session)
        ex = await storage.create_suite_execution_by_name(
            suite_name="code-review",
            agent_name="claude_code",
            started_at=datetime.now(),
            adapter="pipe-check",
        )
        await storage.update_suite_execution(ex, status="completed")
        for axis, passed in [("clean", True), ("severe", False)]:
            te = await storage.create_test_execution(
                suite_execution=ex,
                test_id=f"case-{axis}",
                test_name=f"case-{axis}",
                dimensions={"axis_level": axis, "critical_pass": passed},
            )
            await storage.update_test_execution(te, status="completed", success=passed)
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-run/code-review/claude_code")
    assert resp.status_code == 200
    assert "case-clean" in resp.text
    assert "severe" in resp.text


@pytest.mark.anyio
async def test_eval_run_detail_shows_notice_when_no_cases(fresh_app: tuple) -> None:
    from atp.dashboard.storage import ResultStorage

    app, db = fresh_app
    async with db.session() as session:
        storage = ResultStorage(session)
        ex = await storage.create_suite_execution_by_name(
            suite_name="code-review",
            agent_name="anthropic_api",
            started_at=datetime.now(),
            adapter="pipe-check",
        )
        await storage.update_suite_execution(ex, status="completed")
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-run/code-review/anthropic_api")
    assert resp.status_code == 200
    assert "no per-case detail" in resp.text.lower()


@pytest.mark.anyio
async def test_eval_run_detail_no_completed_run(fresh_app: tuple) -> None:
    """An unknown (suite, agent) pair gets a distinct 'no completed run' notice,
    not the misleading aggregate-only message."""
    app, _db = fresh_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-run/code-review/ghost-agent")
    assert resp.status_code == 200
    assert "no completed run found" in resp.text.lower()
    assert "aggregate-only" not in resp.text.lower()


@pytest.mark.anyio
async def test_eval_run_detail_resolves_at_model_agent_id(fresh_app: tuple) -> None:
    from atp.dashboard.storage import ResultStorage

    app, db = fresh_app
    async with db.session() as session:
        storage = ResultStorage(session)
        ex = await storage.create_suite_execution_by_name(
            suite_name="code-review",
            agent_name="ollama@qwen2.5:14b",
            started_at=datetime.now(),
            adapter="pipe-check",
        )
        await storage.update_suite_execution(ex, status="completed")
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-run/code-review/ollama@qwen2.5:14b")
    assert resp.status_code == 200
    assert "ollama@qwen2.5:14b" in resp.text
