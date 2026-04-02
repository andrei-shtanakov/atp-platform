"""Tests for Benchmark API routes."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database, set_database
from atp.dashboard.models import Base
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
    """Create and configure a test database."""
    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    set_database(db)
    yield db
    await db.close()
    set_database(None)  # type: ignore[arg-type]


@pytest.fixture
def v2_app(test_database: Database):
    """Create a test app with v2 routes."""
    app = create_test_app()

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


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
    """Tests for GET /api/api/v1/benchmarks."""

    async def test_empty_list(self, client: AsyncClient) -> None:
        resp = await client.get("/api/api/v1/benchmarks")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_after_create(self, client: AsyncClient) -> None:
        await client.post(
            "/api/api/v1/benchmarks",
            json={"name": "B1", "suite": SAMPLE_SUITE},
        )
        resp = await client.get("/api/api/v1/benchmarks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "B1"
        assert data[0]["tasks_count"] == 2


@pytest.mark.anyio
class TestCreateBenchmark:
    """Tests for POST /api/api/v1/benchmarks."""

    async def test_create_returns_201(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/api/v1/benchmarks",
            json={"name": "NewBench", "suite": SAMPLE_SUITE},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "NewBench"
        assert data["tasks_count"] == 2
        assert data["id"] is not None


@pytest.mark.anyio
class TestGetBenchmark:
    """Tests for GET /api/api/v1/benchmarks/{id}."""

    async def test_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/api/v1/benchmarks/999")
        assert resp.status_code == 404

    async def test_get_existing(self, client: AsyncClient) -> None:
        create_resp = await client.post(
            "/api/api/v1/benchmarks",
            json={"name": "FindMe", "suite": SAMPLE_SUITE},
        )
        bm_id = create_resp.json()["id"]
        resp = await client.get(f"/api/api/v1/benchmarks/{bm_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "FindMe"


@pytest.mark.anyio
class TestRunLifecycle:
    """Tests for run start, next-task, submit, cancel."""

    async def test_start_run(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/api/v1/benchmarks",
            json={"name": "RunBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]

        resp = await client.post(
            f"/api/api/v1/benchmarks/{bm_id}/start?agent_name=my-agent",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "IN_PROGRESS"
        assert data["agent_name"] == "my-agent"

    async def test_next_task_returns_atp_request(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/api/v1/benchmarks",
            json={"name": "NTBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(
            f"/api/api/v1/benchmarks/{bm_id}/start",
        )
        run_id = run.json()["id"]

        resp = await client.get(f"/api/api/v1/runs/{run_id}/next-task")
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["task"]["description"] == "Do something"
        assert data["metadata"]["task_index"] == 0

    async def test_204_when_tasks_exhausted(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/api/v1/benchmarks",
            json={"name": "ExBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(
            f"/api/api/v1/benchmarks/{bm_id}/start",
        )
        run_id = run.json()["id"]

        # Consume both tasks
        await client.get(f"/api/api/v1/runs/{run_id}/next-task")
        await client.get(f"/api/api/v1/runs/{run_id}/next-task")

        resp = await client.get(f"/api/api/v1/runs/{run_id}/next-task")
        assert resp.status_code == 204

    async def test_submit_result(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/api/v1/benchmarks",
            json={"name": "SubBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(
            f"/api/api/v1/benchmarks/{bm_id}/start",
        )
        run_id = run.json()["id"]

        # Get first task
        await client.get(f"/api/api/v1/runs/{run_id}/next-task")

        # Submit result
        resp = await client.post(
            f"/api/api/v1/runs/{run_id}/submit",
            json={
                "response": {
                    "task_id": "test-1",
                    "status": "completed",
                }
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_index"] == 0
        assert data["score"] == 100.0

    async def test_cancel_run(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/api/v1/benchmarks",
            json={"name": "CancelBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(
            f"/api/api/v1/benchmarks/{bm_id}/start",
        )
        run_id = run.json()["id"]

        resp = await client.post(f"/api/api/v1/runs/{run_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

        # Verify via status endpoint
        status_resp = await client.get(f"/api/api/v1/runs/{run_id}/status")
        assert status_resp.json()["status"] == "CANCELLED"

    async def test_run_status_with_completed_tasks(self, client: AsyncClient) -> None:
        bm = await client.post(
            "/api/api/v1/benchmarks",
            json={"name": "StatusBench", "suite": SAMPLE_SUITE},
        )
        bm_id = bm.json()["id"]
        run = await client.post(
            f"/api/api/v1/benchmarks/{bm_id}/start",
        )
        run_id = run.json()["id"]

        # Submit one task
        await client.get(f"/api/api/v1/runs/{run_id}/next-task")
        await client.post(
            f"/api/api/v1/runs/{run_id}/submit",
            json={
                "response": {
                    "task_id": "test-1",
                    "status": "completed",
                }
            },
        )

        resp = await client.get(f"/api/api/v1/runs/{run_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks_count"] == 2
        assert len(data["completed_tasks"]) == 1
        assert data["completed_tasks"][0]["score"] == 100.0
