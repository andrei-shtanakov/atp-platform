"""A submission scored end to end by the real evaluator registry.

The unit tests stub the resolver, so they prove the semantics table is
implemented; they cannot prove the route reaches the resolver at all. These
tests go through HTTP against an app composed exactly as production composes
it (`atp.server.build_server_resolver`), submit a real response, and read the
published label back off `GET /runs/{id}/status`.

Every suite here asserts something the server policy actually permits. That is
deliberate: with a suite the policy withholds, a run legitimately takes the
completion-only path, so a route that never called the resolver would pass —
green for a reason unrelated to the thing under test.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import get_current_active_user, get_current_admin_user
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_dashboard_app
from atp.evaluation import DETERMINISTIC_ALLOWLIST
from atp.server import build_server_resolver

pytestmark = pytest.mark.anyio


def suite_with(assertions: list[dict[str, Any]]) -> dict[str, Any]:
    """A one-test suite carrying the given assertions."""
    return {
        "test_suite": "evaluated-benchmark",
        "version": "1.0",
        "tests": [
            {
                "id": "test-1",
                "name": "Test One",
                "task": {"description": "Write a greeting"},
                "assertions": assertions,
            }
        ],
    }


#: `contains` is served by the artifact evaluator, which inspects the response
#: and nothing else — permitted on this plane, and it really runs.
CONTAINS_HELLO = [{"type": "contains", "config": {"pattern": "hello"}}]


def response_with(text: str) -> dict[str, Any]:
    return {
        "task_id": "task-1",
        "status": "completed",
        "artifacts": [
            {"type": "file", "path": "out.txt", "content": text},
        ],
    }


@pytest.fixture(autouse=True)
def _server_env(monkeypatch: pytest.MonkeyPatch):
    """Middleware re-reads config per request; without a key it 500s."""
    from atp.dashboard.v2.config import get_config

    monkeypatch.setenv("ATP_SECRET_KEY", "test-secret-for-benchmark-evaluation")
    monkeypatch.setenv("ATP_RATE_LIMIT_ENABLED", "false")
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
async def test_database() -> AsyncGenerator[Database, None]:
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with db.session_factory() as session:
        session.add(
            User(
                id=1,
                username="alice-admin",
                email="alice@example.com",
                hashed_password="x",
                is_active=True,
                is_admin=True,
            )
        )
        await session.commit()
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore[arg-type]


def _app_with(evaluating: bool, database: Database, user: User) -> Any:
    from atp.dashboard.v2.config import DashboardConfig

    config = DashboardConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        secret_key="test-secret-key",
        debug=True,
        disable_auth=True,
        rate_limit_enabled=False,
    )
    app = create_dashboard_app(
        evaluator_resolver=build_server_resolver() if evaluating else None,
        evaluation_mode=DETERMINISTIC_ALLOWLIST if evaluating else "completion_only",
        config=config,
    )

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def override_user() -> User:
        return user

    app.dependency_overrides[get_db_session] = override_get_session
    app.dependency_overrides[get_current_active_user] = override_user
    app.dependency_overrides[get_current_admin_user] = override_user
    return app


@pytest.fixture
async def admin(test_database: Database) -> User:
    async with test_database.session_factory() as session:
        user = await session.get(User, 1)
        assert user is not None
        return user


@pytest.fixture
async def evaluating_client(
    test_database: Database, admin: User
) -> AsyncGenerator[AsyncClient, None]:
    """A server composed the way `atp.server` composes production."""
    transport = ASGITransport(app=_app_with(True, test_database, admin))
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def completion_only_client(
    test_database: Database, admin: User
) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=_app_with(False, test_database, admin))
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def run_one(
    client: AsyncClient, suite: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    """Create a benchmark, run it once with `response`, return the run status."""
    created = await client.post(
        "/api/v1/benchmarks", json={"name": suite["test_suite"], "suite": suite}
    )
    assert created.status_code == 201, created.text
    benchmark_id = created.json()["id"]

    started = await client.post(f"/api/v1/benchmarks/{benchmark_id}/start")
    assert started.status_code in (200, 201), started.text
    run_id = started.json()["id"]

    await client.get(f"/api/v1/runs/{run_id}/next-task")
    submitted = await client.post(
        f"/api/v1/runs/{run_id}/submit",
        json={"task_index": 0, "response": response, "events": []},
    )
    assert submitted.status_code == 200, submitted.text

    status = await client.get(f"/api/v1/runs/{run_id}/status")
    assert status.status_code == 200, status.text
    return status.json()


class TestEvaluatedRun:
    async def test_a_passing_assertion_is_labelled_as_evaluation(
        self, evaluating_client: AsyncClient
    ) -> None:
        body = await run_one(
            evaluating_client, suite_with(CONTAINS_HELLO), response_with("hello there")
        )
        assert body["score_semantics"]["kind"] == "aggregated_evaluation"
        assert body["score_semantics"]["quality_signal"] is True
        assert "contains" in body["score_components"]

    async def test_a_failing_assertion_moves_the_number(
        self, evaluating_client: AsyncClient
    ) -> None:
        """The whole point: the score stops being 100-for-completed.

        Without this, every other assertion in this file is satisfied by a
        route that scores completion and labels it confidently.
        """
        body = await run_one(
            evaluating_client, suite_with(CONTAINS_HELLO), response_with("goodbye")
        )
        assert body["total_score"] < 100.0
        assert body["score_semantics"]["quality_signal"] is True
        assert body["score_components"]["contains"] == 0.0

    async def test_the_per_task_evidence_is_readable(
        self, evaluating_client: AsyncClient
    ) -> None:
        body = await run_one(
            evaluating_client, suite_with(CONTAINS_HELLO), response_with("hello there")
        )
        records = body["completed_tasks"][0]["eval_results"]
        assert records is not None
        assert records[0]["assertion_type"] == "contains"
        assert records[0]["status"] == "applied"
        assert records[0]["passed"] is True


class TestNotEvaluated:
    async def test_completion_only_server_says_completion(
        self, completion_only_client: AsyncClient
    ) -> None:
        """Same suite, same submission, no resolver: the old behaviour intact."""
        body = await run_one(
            completion_only_client,
            suite_with(CONTAINS_HELLO),
            response_with("goodbye"),
        )
        assert body["total_score"] == 100.0
        assert body["score_semantics"]["kind"] == "completion_rate"
        assert body["score_semantics"]["quality_signal"] is False
        assert body["score_components"] == {}

    async def test_a_withheld_evaluator_is_coverage_not_a_zero(
        self, evaluating_client: AsyncClient
    ) -> None:
        """`pytest` executes submission-derived code; the policy refuses it."""
        body = await run_one(
            evaluating_client,
            suite_with([{"type": "pytest", "config": {}}]),
            response_with("hello"),
        )
        assert body["score_semantics"]["kind"] == "completion_rate"
        assert body["score_semantics"]["quality_signal"] is False
        assert body["score_components"] == {}
        skipped = body["score_semantics"]["coverage"]["assertions_skipped"]
        assert skipped == [
            {
                "assertion_type": "pytest",
                "reason": "not_allowed_by_policy",
                "count": 1,
            }
        ]

    async def test_an_incomplete_submission_is_not_evaluated(
        self, evaluating_client: AsyncClient
    ) -> None:
        body = await run_one(
            evaluating_client,
            suite_with(CONTAINS_HELLO),
            {"task_id": "task-1", "status": "failed", "error": "boom"},
        )
        assert body["total_score"] == 0.0
        assert body["score_semantics"]["kind"] == "completion_rate"
        assert body["completed_tasks"][0]["eval_results"] is None
