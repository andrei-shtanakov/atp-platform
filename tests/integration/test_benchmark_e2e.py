"""End-to-end integration test for the benchmark API flow.

Exercises the full benchmark lifecycle:
1. Create benchmark
2. Start run
3. Pull all tasks
4. Submit results
5. Verify completion
6. Check leaderboard
"""

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.auth import get_current_active_user
from atp.dashboard.database import init_database
from atp.dashboard.models import User
from atp.dashboard.v2.factory import create_test_app

# The factory mounts api_router at "/api" and benchmark_api_router
# has prefix="/api/v1", so full prefix is "/api/v1".
BASE = "/api/v1"

DB_URL = "sqlite+aiosqlite:///:memory:"


def _mock_user() -> User:
    """Synthetic admin user for benchmark endpoints that require auth."""
    user = User(
        username="benchmark_e2e",
        email="bench@test.com",
        hashed_password="x",
        is_active=True,
        is_admin=True,
    )
    user.id = 1
    return user


@pytest.fixture()
async def client() -> Any:
    """Create test app, initialize DB, and yield async client.

    Benchmark endpoints (POST /benchmarks, /runs/{id}/* etc.) require an
    authenticated user since the IDOR fix. We override get_current_active_user
    rather than minting a real JWT — this test cares about the benchmark
    state machine, not the auth path.
    """
    app = create_test_app(database_url=DB_URL)
    app.dependency_overrides[get_current_active_user] = _mock_user
    # Manually trigger database initialization (lifespan is not
    # called by httpx ASGITransport).
    await init_database(url=DB_URL, echo=False)

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def _make_suite() -> dict[str, Any]:
    """Build a minimal test suite payload with two tests."""
    return {
        "test_suite": "integration",
        "version": "1.0",
        "defaults": {"constraints": {}, "scoring": {}},
        "tests": [
            {
                "id": "t-001",
                "name": "Echo",
                "task": {"description": "Echo hello"},
                "assertions": [],
            },
            {
                "id": "t-002",
                "name": "Math",
                "task": {"description": "Compute 2+2"},
                "assertions": [],
            },
        ],
    }


@pytest.mark.anyio
async def test_full_benchmark_flow(client: AsyncClient) -> None:
    """Run the complete benchmark lifecycle end-to-end."""
    # 1. Create benchmark with 2 tests
    suite = _make_suite()
    resp = await client.post(
        f"{BASE}/benchmarks",
        json={
            "name": "e2e-v1",
            "description": "E2E",
            "suite": suite,
            "family_tag": "e2e",
        },
    )
    assert resp.status_code == 201, resp.text
    bm = resp.json()
    bm_id = bm["id"]
    assert bm["tasks_count"] == 2

    # 2. Start run
    resp = await client.post(
        f"{BASE}/benchmarks/{bm_id}/start",
        params={"agent_name": "test-agent"},
    )
    assert resp.status_code == 200, resp.text
    run = resp.json()
    run_id = run["id"]
    assert run["status"] == "IN_PROGRESS"

    # 3. Pull and submit all tasks
    for _ in range(2):
        task_resp = await client.get(
            f"{BASE}/runs/{run_id}/next-task",
        )
        assert task_resp.status_code == 200, task_resp.text
        task = task_resp.json()
        assert "task_id" in task

        submit_resp = await client.post(
            f"{BASE}/runs/{run_id}/submit",
            json={
                "response": {
                    "version": "1.0",
                    "task_id": task["task_id"],
                    "status": "completed",
                    "artifacts": [],
                },
                "task_index": task["metadata"]["task_index"],
            },
        )
        assert submit_resp.status_code == 200, submit_resp.text

    # 4. Verify no more tasks
    resp = await client.get(f"{BASE}/runs/{run_id}/next-task")
    assert resp.status_code == 204

    # 5. Check run status
    resp = await client.get(f"{BASE}/runs/{run_id}/status")
    assert resp.status_code == 200, resp.text
    run_status = resp.json()
    assert run_status["status"] == "COMPLETED"
    assert run_status["total_score"] is not None
    assert run_status["total_score"] == 100.0
    assert len(run_status["completed_tasks"]) == 2

    # 6. Check leaderboard
    resp = await client.get(
        f"{BASE}/benchmarks/{bm_id}/leaderboard",
    )
    assert resp.status_code == 200, resp.text
    lb = resp.json()
    assert len(lb) >= 1
    assert lb[0]["agent_name"] == "test-agent"
    assert lb[0]["best_score"] == 100.0


@pytest.mark.anyio
async def test_next_task_returns_204_when_all_consumed(
    client: AsyncClient,
) -> None:
    """After all tasks are pulled, next-task returns 204."""
    suite = {
        "test_suite": "single",
        "version": "1.0",
        "defaults": {"constraints": {}, "scoring": {}},
        "tests": [
            {
                "id": "t-only",
                "name": "Only",
                "task": {"description": "Single task"},
                "assertions": [],
            },
        ],
    }
    resp = await client.post(
        f"{BASE}/benchmarks",
        json={
            "name": "single-bench",
            "description": "one test",
            "suite": suite,
        },
    )
    assert resp.status_code == 201
    bm_id = resp.json()["id"]

    resp = await client.post(
        f"{BASE}/benchmarks/{bm_id}/start",
        params={"agent_name": "agent-x"},
    )
    assert resp.status_code == 200
    run_id = resp.json()["id"]

    # Consume the only task
    resp = await client.get(f"{BASE}/runs/{run_id}/next-task")
    assert resp.status_code == 200

    # Now should be 204
    resp = await client.get(f"{BASE}/runs/{run_id}/next-task")
    assert resp.status_code == 204


@pytest.mark.anyio
async def test_cancel_run(client: AsyncClient) -> None:
    """Cancelling a run sets status to cancelled."""
    suite = _make_suite()
    resp = await client.post(
        f"{BASE}/benchmarks",
        json={"name": "cancel-test", "suite": suite},
    )
    assert resp.status_code == 201
    bm_id = resp.json()["id"]

    resp = await client.post(
        f"{BASE}/benchmarks/{bm_id}/start",
        params={"agent_name": "cancel-agent"},
    )
    assert resp.status_code == 200
    run_id = resp.json()["id"]

    resp = await client.post(f"{BASE}/runs/{run_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    resp = await client.get(f"{BASE}/runs/{run_id}/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "CANCELLED"
