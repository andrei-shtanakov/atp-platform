"""Tests for BenchmarkRun with batch support and async iteration (Task 4)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from atp_sdk.benchmark import BenchmarkRun

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(responses: list[httpx.Response]) -> MagicMock:
    """Create a mock AsyncATPClient whose _request yields given responses."""
    client = MagicMock()
    client._request = AsyncMock(side_effect=responses)
    return client


def _task_response(task: dict[str, Any], status: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = task
    return resp


def _tasks_response(tasks: list[dict[str, Any]], status: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = tasks
    return resp


def _no_content() -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 204
    return resp


def _make_run(
    client: MagicMock,
    run_id: int = 1,
    benchmark_id: int = 10,
    batch_size: int = 1,
) -> BenchmarkRun:
    return BenchmarkRun(
        client=client, run_id=run_id, benchmark_id=benchmark_id, batch_size=batch_size
    )


# ---------------------------------------------------------------------------
# Async iteration – single task per request
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_iter_single_task() -> None:
    """Async iteration yields one task then stops on 204."""
    task = {"id": 1, "task": "solve x+1=2"}
    client = _make_client([_task_response(task), _no_content()])

    run = _make_run(client)
    results: list[dict[str, Any]] = []
    async for t in run:
        results.append(t)

    assert results == [task]


@pytest.mark.anyio
async def test_async_iter_multiple_tasks() -> None:
    """Async iteration yields all tasks across multiple requests."""
    tasks = [{"id": i, "task": f"task {i}"} for i in range(3)]
    responses = [_task_response(t) for t in tasks] + [_no_content()]
    client = _make_client(responses)

    run = _make_run(client)
    results = [t async for t in run]

    assert results == tasks


@pytest.mark.anyio
async def test_async_iter_empty_immediately() -> None:
    """Async iteration stops immediately when 204 is the first response."""
    client = _make_client([_no_content()])
    run = _make_run(client)

    results = [t async for t in run]
    assert results == []


# ---------------------------------------------------------------------------
# Async iteration – batch_size > 1
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_iter_batch_size_3() -> None:
    """With batch_size=3, server returns list of tasks per request."""
    batch1 = [{"id": i} for i in range(3)]
    batch2 = [{"id": i} for i in range(3, 5)]
    client = _make_client(
        [_tasks_response(batch1), _tasks_response(batch2), _no_content()]
    )

    run = _make_run(client, batch_size=3)
    results = [t async for t in run]

    assert results == batch1 + batch2


@pytest.mark.anyio
async def test_async_iter_partial_batch() -> None:
    """A partial batch (fewer than requested) is handled correctly."""
    tasks = [{"id": 1}, {"id": 2}]
    # Server returns 2 tasks even though batch_size=5, then 204
    client = _make_client([_tasks_response(tasks), _no_content()])

    run = _make_run(client, batch_size=5)
    results = [t async for t in run]

    assert results == tasks


@pytest.mark.anyio
async def test_async_iter_single_dict_response() -> None:
    """Server can return single dict (batch_size=1 compat); handled as list of 1."""
    task = {"id": 42, "task": "do something"}
    client = _make_client([_task_response(task), _no_content()])

    # batch_size=1 returns dict, not list
    run = _make_run(client, batch_size=1)
    results = [t async for t in run]

    assert results == [task]


# ---------------------------------------------------------------------------
# next_batch
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_next_batch_returns_list() -> None:
    """next_batch(n) returns list of n task dicts."""
    tasks = [{"id": i} for i in range(3)]
    client = _make_client([_tasks_response(tasks)])

    run = _make_run(client, batch_size=3)
    result = await run.next_batch(3)

    assert result == tasks


@pytest.mark.anyio
async def test_next_batch_empty_on_204() -> None:
    """next_batch returns [] when server returns 204."""
    client = _make_client([_no_content()])
    run = _make_run(client)

    result = await run.next_batch(1)
    assert result == []


@pytest.mark.anyio
async def test_next_batch_partial() -> None:
    """next_batch returns fewer items when server gives partial batch."""
    tasks = [{"id": 1}]
    client = _make_client([_tasks_response(tasks)])

    run = _make_run(client, batch_size=5)
    result = await run.next_batch(5)

    assert result == tasks


@pytest.mark.anyio
async def test_next_batch_never_raises_stop_iteration() -> None:
    """next_batch returns [] on exhaustion, never raises StopIteration."""
    client = _make_client([_no_content()])
    run = _make_run(client)

    # Call next_batch twice — second should return [] too (already exhausted)
    result1 = await run.next_batch(1)
    result2 = await run.next_batch(1)

    assert result1 == []
    assert result2 == []


# ---------------------------------------------------------------------------
# submit, status, cancel, leaderboard
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_submit() -> None:
    """submit posts response and events, returns score dict."""
    score_data: dict[str, Any] = {"score": 0.95, "task_id": 1}
    client = MagicMock()
    client._request = AsyncMock(return_value=_task_response(score_data))

    run = _make_run(client)
    result = await run.submit(
        {"answer": "x=1"},
        task_index=0,
        events=[{"type": "tool_call"}],
    )

    assert result == score_data
    client._request.assert_called_once()
    call_kwargs = client._request.call_args
    assert call_kwargs[0][0] == "POST"
    assert "/submit" in call_kwargs[0][1]


@pytest.mark.anyio
async def test_submit_no_events() -> None:
    """submit without events omits events from payload."""
    client = MagicMock()
    client._request = AsyncMock(return_value=_task_response({"score": 1.0}))

    run = _make_run(client)
    result = await run.submit({"answer": "42"}, task_index=0)

    assert result["score"] == 1.0


@pytest.mark.anyio
async def test_status() -> None:
    """status() returns run status dict."""
    status_data = {"status": "in_progress", "current_task_index": 3}
    client = MagicMock()
    client._request = AsyncMock(return_value=_task_response(status_data))

    run = _make_run(client)
    result = await run.status()

    assert result == status_data
    client._request.assert_called_once()
    call_args = client._request.call_args[0]
    assert call_args[0] == "GET"
    assert "/status" in call_args[1]


@pytest.mark.anyio
async def test_cancel() -> None:
    """cancel() posts to cancel endpoint."""
    client = MagicMock()
    client._request = AsyncMock(return_value=_no_content())

    run = _make_run(client)
    await run.cancel()

    client._request.assert_called_once()
    call_args = client._request.call_args[0]
    assert call_args[0] == "POST"
    assert "/cancel" in call_args[1]


@pytest.mark.anyio
async def test_leaderboard() -> None:
    """leaderboard() returns list of leaderboard entries."""
    lb_data = [{"user_id": 1, "agent_name": "bot", "best_score": 0.99}]
    client = MagicMock()
    client._request = AsyncMock(return_value=_tasks_response(lb_data))

    run = _make_run(client, benchmark_id=10)
    result = await run.leaderboard()

    assert result == lb_data
    call_args = client._request.call_args[0]
    assert "benchmarks/10/leaderboard" in call_args[1]


# ---------------------------------------------------------------------------
# Sync iteration guard
# ---------------------------------------------------------------------------


def test_sync_iter_outside_async_raises_runtime_error() -> None:
    """__iter__ raises RuntimeError when an event loop is running."""
    # We need to simulate the case from outside an async context:
    # create a run, call iter() — no running loop, so asyncio.run is used.
    # Since we can't easily run a real asyncio.run in pytest-anyio without
    # interference, we verify the error path when a loop IS running.
    client = MagicMock()
    client._request = AsyncMock(return_value=httpx.Response(204))
    run = _make_run(client)

    # Verify that trying to use sync iter raises RuntimeError
    # when called from inside a running event loop
    async def _inner() -> None:
        with pytest.raises(RuntimeError, match="async"):
            list(run)

    asyncio.run(_inner())


def test_sync_iter_without_loop_collects_all() -> None:
    """__iter__ works in a plain sync context (no running loop)."""
    tasks = [{"id": 1}, {"id": 2}]
    responses = [_tasks_response(tasks), _no_content()]

    call_idx = 0

    async def fake_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        nonlocal call_idx
        resp = responses[call_idx]
        call_idx += 1
        return resp

    client = MagicMock()
    client._request = fake_request
    run = BenchmarkRun(client=client, run_id=1, benchmark_id=10, batch_size=2)

    result = list(run)
    assert result == tasks
