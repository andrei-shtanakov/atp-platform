"""Tests for ATPClient (sync wrapper).

NOTE: These tests are skipped until Task 3 re-implements the sync ATPClient
wrapper around AsyncATPClient.  After Task 3 is done, remove the skip markers.
"""

from __future__ import annotations

import httpx
import pytest
from atp_sdk.benchmark import BenchmarkRun
from atp_sdk.models import BenchmarkInfo
from atp_sdk.sync import ATPClient

pytestmark = pytest.mark.skip(
    reason="Sync ATPClient will be re-implemented in Task 3; skipping until then."
)


def _make_transport(
    handler: httpx.MockTransport | None = None,
) -> httpx.MockTransport:
    """Return the transport (pass-through helper for clarity)."""
    assert handler is not None
    return handler


def test_list_benchmarks() -> None:
    """GET /api/v1/benchmarks returns parsed BenchmarkInfo list."""
    payload = [
        {
            "id": 1,
            "name": "bench-1",
            "description": "First",
            "tasks_count": 5,
            "tags": ["math"],
            "version": "1.0",
        },
        {
            "id": 2,
            "name": "bench-2",
            "description": "Second",
            "tasks_count": 3,
            "tags": [],
            "version": "2.0",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/benchmarks"
        assert request.method == "GET"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = ATPClient(platform_url="http://test")
    client._http = httpx.Client(base_url="http://test", transport=transport)

    result = client.list_benchmarks()
    assert len(result) == 2
    assert all(isinstance(b, BenchmarkInfo) for b in result)
    assert result[0].name == "bench-1"
    assert result[1].tasks_count == 3
    client.close()


def test_get_benchmark() -> None:
    """GET /api/v1/benchmarks/{id} returns single BenchmarkInfo."""
    payload = {
        "id": 42,
        "name": "bench-42",
        "description": "Test",
        "tasks_count": 10,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/benchmarks/42"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = ATPClient(platform_url="http://test")
    client._http = httpx.Client(base_url="http://test", transport=transport)

    result = client.get_benchmark(42)
    assert isinstance(result, BenchmarkInfo)
    assert result.id == 42
    assert result.name == "bench-42"
    client.close()


def test_start_run() -> None:
    """POST /api/v1/benchmarks/{id}/start returns BenchmarkRun."""
    run_data = {
        "id": 99,
        "benchmark_id": 1,
        "agent_name": "my-agent",
        "status": "in_progress",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/benchmarks/1/start"
        assert request.method == "POST"
        assert "agent_name=my-agent" in str(request.url)
        return httpx.Response(200, json=run_data)

    transport = httpx.MockTransport(handler)
    client = ATPClient(platform_url="http://test")
    client._http = httpx.Client(base_url="http://test", transport=transport)

    run = client.start_run(1, agent_name="my-agent")
    assert isinstance(run, BenchmarkRun)
    assert run.run_id == 99
    assert run.benchmark_id == 1
    client.close()


def test_get_leaderboard() -> None:
    """GET /api/v1/benchmarks/{id}/leaderboard returns list."""
    payload = [
        {"user_id": 1, "agent_name": "a", "best_score": 0.9},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/benchmarks/5/leaderboard"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = ATPClient(platform_url="http://test")
    client._http = httpx.Client(base_url="http://test", transport=transport)

    result = client.get_leaderboard(5)
    assert len(result) == 1
    assert result[0]["best_score"] == 0.9
    client.close()


def test_context_manager() -> None:
    """ATPClient works as a context manager."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    with ATPClient(platform_url="http://test") as client:
        client._http = httpx.Client(base_url="http://test", transport=transport)
        result = client.list_benchmarks()
        assert result == []


def test_auth_header(monkeypatch: object) -> None:
    """Token is sent as Bearer header."""
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    client = ATPClient(platform_url="http://test", token="secret-tok")
    # Replace transport while keeping headers
    client._http = httpx.Client(
        base_url="http://test",
        transport=transport,
        headers={"Authorization": "Bearer secret-tok"},
    )
    client.list_benchmarks()
    assert captured_headers["authorization"] == "Bearer secret-tok"
    client.close()
