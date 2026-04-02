"""Tests for BenchmarkRun."""

from __future__ import annotations

from typing import Any

import httpx
from atp_sdk.benchmark import BenchmarkRun


def _make_run(
    transport: httpx.MockTransport,
    run_id: int = 1,
    benchmark_id: int = 10,
) -> BenchmarkRun:
    """Create a BenchmarkRun with a mock transport."""
    http = httpx.Client(base_url="http://test", transport=transport)
    return BenchmarkRun(http=http, run_id=run_id, benchmark_id=benchmark_id)


def test_benchmark_run_iterates_tasks() -> None:
    """Iterator yields tasks until 204 No Content."""
    tasks = [
        {"task": "solve x+1=2", "id": 1},
        {"task": "solve 2x=4", "id": 2},
    ]
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        assert request.url.path == "/api/v1/runs/1/next-task"
        if call_count < len(tasks):
            task = tasks[call_count]
            call_count += 1
            return httpx.Response(200, json=task)
        return httpx.Response(204)

    run = _make_run(httpx.MockTransport(handler))
    collected = list(run)

    assert len(collected) == 2
    assert collected[0]["id"] == 1
    assert collected[1]["id"] == 2


def test_benchmark_run_empty() -> None:
    """Iterator stops immediately on 204."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    run = _make_run(httpx.MockTransport(handler))
    assert list(run) == []


def test_benchmark_run_submit() -> None:
    """Submit posts response and returns score."""
    score_data: dict[str, Any] = {
        "task_id": 1,
        "score": 0.95,
        "status": "scored",
    }
    captured_body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/runs/1/submit"
        assert request.method == "POST"
        import json

        captured_body.update(json.loads(request.content))
        return httpx.Response(200, json=score_data)

    run = _make_run(httpx.MockTransport(handler))
    result = run.submit(
        {"answer": "x=1"},
        task_index=0,
        events=[{"type": "tool_call", "name": "calc"}],
    )

    assert result["score"] == 0.95
    assert captured_body["response"] == {"answer": "x=1"}
    assert captured_body["task_index"] == 0
    assert len(captured_body["events"]) == 1


def test_benchmark_run_submit_no_events() -> None:
    """Submit without events omits the events key."""
    captured_body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured_body.update(json.loads(request.content))
        return httpx.Response(200, json={"score": 1.0})

    run = _make_run(httpx.MockTransport(handler))
    run.submit({"answer": "42"}, task_index=0)
    assert "events" not in captured_body
    assert captured_body["task_index"] == 0


def test_benchmark_run_status() -> None:
    """Status returns run info dict."""
    status_data = {
        "status": "in_progress",
        "current_task_index": 3,
        "total_score": 2.5,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/runs/1/status"
        return httpx.Response(200, json=status_data)

    run = _make_run(httpx.MockTransport(handler))
    result = run.status()
    assert result["status"] == "in_progress"
    assert result["current_task_index"] == 3


def test_benchmark_run_cancel() -> None:
    """Cancel posts to the cancel endpoint."""
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        assert request.url.path == "/api/v1/runs/1/cancel"
        assert request.method == "POST"
        called = True
        return httpx.Response(200)

    run = _make_run(httpx.MockTransport(handler))
    run.cancel()
    assert called


def test_benchmark_run_leaderboard() -> None:
    """Leaderboard fetches from the benchmark endpoint."""
    lb_data = [
        {"user_id": 1, "agent_name": "bot", "best_score": 0.99},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/benchmarks/10/leaderboard"
        return httpx.Response(200, json=lb_data)

    run = _make_run(httpx.MockTransport(handler), benchmark_id=10)
    result = run.leaderboard()
    assert len(result) == 1
    assert result[0]["best_score"] == 0.99
