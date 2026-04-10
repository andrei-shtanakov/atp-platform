"""Tests for Benchmark API routes."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.auth import (
    get_current_active_user,
    get_current_admin_user,
)
from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base, User
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app

SAMPLE_SUITE: dict = {
    "test_suite": "sample-benchmark",
    "version": "1.0",
    "tests": [
        {
            "id": "test-1",
            "name": "Test One",
            "task": {"description": "Do something"},
        },
        {
            "id": "test-2",
            "name": "Test Two",
            "task": {"description": "Do another thing"},
        },
    ],
}


@pytest.fixture
async def test_database() -> AsyncGenerator[Database, None]:
    """Create and configure a test database with a seeded admin user."""
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Seed admin (id=1) and regular user (id=2) used by auth-override
    # fixtures and ownership tests.
    async with db.session_factory() as session:
        session.add_all(
            [
                User(
                    id=1,
                    username="alice-admin",
                    email="alice@example.com",
                    hashed_password="x",
                    is_active=True,
                    is_admin=True,
                ),
                User(
                    id=2,
                    username="bob",
                    email="bob@example.com",
                    hashed_password="x",
                    is_active=True,
                    is_admin=False,
                ),
            ]
        )
        await session.commit()
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore[arg-type]


@pytest.fixture
async def admin_user(test_database: Database) -> User:
    """Return the seeded admin user (id=1)."""
    async with test_database.session_factory() as session:
        user = await session.get(User, 1)
        assert user is not None
        return user


@pytest.fixture
async def regular_user(test_database: Database) -> User:
    """Return the seeded non-admin user (id=2)."""
    async with test_database.session_factory() as session:
        user = await session.get(User, 2)
        assert user is not None
        return user


@pytest.fixture
def v2_app(test_database: Database, admin_user: User):
    """Create a test app with v2 routes, authenticated as the admin user.

    Auth dependencies are overridden to return the seeded admin, which
    also satisfies AdminUser (required by ``create_benchmark``).  Tests
    that need to assert ownership violations should use ``v2_app_as_bob``
    instead.
    """
    app = create_test_app()

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def override_current_user() -> User:
        return admin_user

    async def override_admin_user() -> User:
        return admin_user

    app.dependency_overrides[get_db_session] = override_get_session
    app.dependency_overrides[get_current_active_user] = override_current_user
    app.dependency_overrides[get_current_admin_user] = override_admin_user
    return app


@pytest.fixture
def v2_app_as_bob(test_database: Database, regular_user: User):
    """Test app authenticated as the non-admin user bob (id=2).

    Used by ownership-violation tests — bob tries to touch runs owned
    by alice (the admin) and should always get 404.
    """
    app = create_test_app()

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def override_current_user() -> User:
        return regular_user

    app.dependency_overrides[get_db_session] = override_get_session
    app.dependency_overrides[get_current_active_user] = override_current_user
    return app


@pytest.fixture
async def bob_client(v2_app_as_bob) -> AsyncGenerator[AsyncClient, None]:
    """httpx client authenticated as bob (non-admin, non-owner)."""
    transport = ASGITransport(app=v2_app_as_bob)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def client(
    v2_app,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an httpx AsyncClient bound to the test app."""
    transport = ASGITransport(app=v2_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
class TestListBenchmarks:
    """Tests for GET /api/v1/benchmarks."""

    async def test_empty_list(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/benchmarks")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_after_create(self, client: AsyncClient) -> None:
        await client.post(
            "/api/v1/benchmarks",
            json={"name": "B1", "suite": SAMPLE_SUITE},
        )
        resp = await client.get("/api/v1/benchmarks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "B1"
        assert data[0]["tasks_count"] == 2


@pytest.mark.anyio
class TestCreateBenchmark:
    """Tests for POST /api/v1/benchmarks."""

    async def test_create_returns_201(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/benchmarks",
            json={"name": "NewBench", "suite": SAMPLE_SUITE},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "NewBench"
        assert data["tasks_count"] == 2
        assert data["id"] is not None


@pytest.mark.anyio
class TestGetBenchmark:
    """Tests for GET /api/v1/benchmarks/{id}."""

    async def test_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/benchmarks/999")
        assert resp.status_code == 404

    async def test_get_existing(self, client: AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/benchmarks",
            json={"name": "FindMe", "suite": SAMPLE_SUITE},
        )
        bm_id = create_resp.json()["id"]
        resp = await client.get(f"/api/v1/benchmarks/{bm_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "FindMe"


@pytest.mark.anyio
class TestRunLifecycle:
    """Tests for run start, next-task, submit, cancel."""

    async def test_start_run(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/v1/benchmarks",
            json={"name": "RunBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]

        resp = await client.post(
            f"/api/v1/benchmarks/{bm_id}/start?agent_name=my-agent",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "IN_PROGRESS"
        assert data["agent_name"] == "my-agent"

    async def test_next_task_returns_atp_request(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/v1/benchmarks",
            json={"name": "NTBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(
            f"/api/v1/benchmarks/{bm_id}/start",
        )
        run_id = run.json()["id"]

        resp = await client.get(f"/api/v1/runs/{run_id}/next-task")
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["task"]["description"] == "Do something"
        assert data["metadata"]["task_index"] == 0

    async def test_204_when_tasks_exhausted(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/v1/benchmarks",
            json={"name": "ExBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(
            f"/api/v1/benchmarks/{bm_id}/start",
        )
        run_id = run.json()["id"]

        # Consume both tasks
        await client.get(f"/api/v1/runs/{run_id}/next-task")
        await client.get(f"/api/v1/runs/{run_id}/next-task")

        resp = await client.get(f"/api/v1/runs/{run_id}/next-task")
        assert resp.status_code == 204

    async def test_submit_result(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/v1/benchmarks",
            json={"name": "SubBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(
            f"/api/v1/benchmarks/{bm_id}/start",
        )
        run_id = run.json()["id"]

        # Get first task
        await client.get(f"/api/v1/runs/{run_id}/next-task")

        # Submit result
        resp = await client.post(
            f"/api/v1/runs/{run_id}/submit",
            json={
                "response": {
                    "task_id": "test-1",
                    "status": "completed",
                },
                "task_index": 0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_index"] == 0
        assert data["score"] == 100.0

    async def test_cancel_run(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/v1/benchmarks",
            json={"name": "CancelBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(
            f"/api/v1/benchmarks/{bm_id}/start",
        )
        run_id = run.json()["id"]

        resp = await client.post(f"/api/v1/runs/{run_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

        # Verify via status endpoint
        status_resp = await client.get(f"/api/v1/runs/{run_id}/status")
        assert status_resp.json()["status"] == "CANCELLED"

    async def test_next_task_batch_returns_list(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/v1/benchmarks",
            json={"name": "BatchBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(f"/api/v1/benchmarks/{bm_id}/start")
        run_id = run.json()["id"]

        resp = await client.get(f"/api/v1/runs/{run_id}/next-task?batch=2")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["metadata"]["task_index"] == 0
        assert data[1]["metadata"]["task_index"] == 1

    async def test_run_status_with_completed_tasks(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/v1/benchmarks",
            json={"name": "StatusBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(
            f"/api/v1/benchmarks/{bm_id}/start",
        )
        run_id = run.json()["id"]

        # Submit one task
        await client.get(f"/api/v1/runs/{run_id}/next-task")
        await client.post(
            f"/api/v1/runs/{run_id}/submit",
            json={
                "response": {
                    "task_id": "test-1",
                    "status": "completed",
                },
                "task_index": 0,
            },
        )

        resp = await client.get(f"/api/v1/runs/{run_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks_count"] == 2
        assert len(data["completed_tasks"]) == 1
        assert data["completed_tasks"][0]["score"] == 100.0


@pytest.mark.anyio
class TestNextTaskBatch:
    """Tests for GET /api/v1/runs/{run_id}/next-task?batch=N."""

    async def _create_run(self, client: AsyncClient, name: str) -> int:
        """Helper: create a benchmark + run, return run_id."""
        bm = await client.post(
            "/api/v1/benchmarks",
            json={"name": name, "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(f"/api/v1/benchmarks/{bm_id}/start")
        return run.json()["id"]

    async def test_batch_returns_multiple_tasks(self, client: AsyncClient) -> None:
        """batch=2 returns a list of 2 task dicts."""
        run_id = await self._create_run(client, "BatchMulti")

        resp = await client.get(f"/api/v1/runs/{run_id}/next-task?batch=2")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["metadata"]["task_index"] == 0
        assert data[1]["metadata"]["task_index"] == 1

    async def test_batch_default_returns_single_dict(self, client: AsyncClient) -> None:
        """No batch param (default=1) returns a single dict, not a list."""
        run_id = await self._create_run(client, "BatchDefault")

        resp = await client.get(f"/api/v1/runs/{run_id}/next-task")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "task_id" in data
        assert data["metadata"]["task_index"] == 0

    async def test_batch_partial_at_end(self, client: AsyncClient) -> None:
        """batch=10 with only 1 task left returns a list of 1."""
        run_id = await self._create_run(client, "BatchPartial")

        # Consume first task individually
        await client.get(f"/api/v1/runs/{run_id}/next-task")

        # Request a large batch — only 1 task remains
        resp = await client.get(f"/api/v1/runs/{run_id}/next-task?batch=10")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["metadata"]["task_index"] == 1

    async def test_batch_204_when_exhausted(self, client: AsyncClient) -> None:
        """After all tasks consumed, batch=5 returns 204."""
        run_id = await self._create_run(client, "BatchExhausted")

        # Consume all tasks (suite has 2)
        await client.get(f"/api/v1/runs/{run_id}/next-task?batch=2")

        resp = await client.get(f"/api/v1/runs/{run_id}/next-task?batch=5")
        assert resp.status_code == 204


@pytest.mark.anyio
class TestRunOwnership:
    """Regression tests for the IDOR fix (2026-04-10).

    Alice (admin) creates a benchmark and starts a run.  Bob (regular
    user) then tries every lifecycle endpoint on that run and must
    receive 404 — not 403, to avoid leaking run_id existence.
    """

    async def _alice_create_run(self, client: AsyncClient) -> tuple[int, int]:
        """Create benchmark + start a run as Alice. Returns (bm_id, run_id)."""
        bm = await client.post(
            "/api/v1/benchmarks",
            json={"name": "OwnedByAlice", "suite": SAMPLE_SUITE},
        )
        assert bm.status_code == 201, bm.text
        bm_id = bm.json()["id"]
        run = await client.post(f"/api/v1/benchmarks/{bm_id}/start")
        assert run.status_code == 200
        return bm_id, run.json()["id"]

    async def test_start_run_assigns_user_id_to_current_user(
        self, client: AsyncClient
    ) -> None:
        """start_run must stamp Run.user_id with the authenticated user."""
        from atp.dashboard.benchmark.models import Run
        from atp.dashboard.database import get_database

        _, run_id = await self._alice_create_run(client)

        db = get_database()
        async with db.session_factory() as session:
            run = await session.get(Run, run_id)
            assert run is not None
            # Alice is the admin (id=1) seeded by test_database fixture
            assert run.user_id == 1
            assert run.tenant_id  # default tenant

    async def test_next_task_rejects_other_user(
        self, client: AsyncClient, bob_client: AsyncClient
    ) -> None:
        _, run_id = await self._alice_create_run(client)
        resp = await bob_client.get(f"/api/v1/runs/{run_id}/next-task")
        assert resp.status_code == 404

    async def test_submit_rejects_other_user(
        self, client: AsyncClient, bob_client: AsyncClient
    ) -> None:
        _, run_id = await self._alice_create_run(client)
        # Alice pulls the task first so there's something to submit against
        await client.get(f"/api/v1/runs/{run_id}/next-task")
        resp = await bob_client.post(
            f"/api/v1/runs/{run_id}/submit",
            json={
                "response": {"task_id": "test-1", "status": "completed"},
                "task_index": 0,
            },
        )
        assert resp.status_code == 404
        # And Alice's run should have zero submitted TaskResults from bob
        from sqlalchemy import select

        from atp.dashboard.benchmark.models import TaskResult
        from atp.dashboard.database import get_database

        db = get_database()
        async with db.session_factory() as session:
            result = await session.execute(
                select(TaskResult).where(TaskResult.run_id == run_id)
            )
            assert result.scalars().all() == []

    async def test_status_rejects_other_user(
        self, client: AsyncClient, bob_client: AsyncClient
    ) -> None:
        _, run_id = await self._alice_create_run(client)
        resp = await bob_client.get(f"/api/v1/runs/{run_id}/status")
        assert resp.status_code == 404

    async def test_cancel_rejects_other_user(
        self, client: AsyncClient, bob_client: AsyncClient
    ) -> None:
        _, run_id = await self._alice_create_run(client)
        resp = await bob_client.post(f"/api/v1/runs/{run_id}/cancel")
        assert resp.status_code == 404

        # Verify the run is still IN_PROGRESS (not cancelled by bob)
        status_resp = await client.get(f"/api/v1/runs/{run_id}/status")
        assert status_resp.json()["status"] == "IN_PROGRESS"

    async def test_events_rejects_other_user(
        self, client: AsyncClient, bob_client: AsyncClient
    ) -> None:
        _, run_id = await self._alice_create_run(client)
        resp = await bob_client.post(
            f"/api/v1/runs/{run_id}/events",
            json={"events": [{"type": "test", "data": "x"}]},
        )
        assert resp.status_code == 404

    async def test_bob_cannot_create_benchmark(self, bob_client: AsyncClient) -> None:
        """create_benchmark is AdminUser-only — non-admin gets 403."""
        resp = await bob_client.post(
            "/api/v1/benchmarks",
            json={"name": "BobBench", "suite": SAMPLE_SUITE},
        )
        assert resp.status_code == 403
