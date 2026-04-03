"""Tests for ATPClient sync wrapper (Task 3)."""

from __future__ import annotations

import concurrent.futures
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from atp_sdk.benchmark import BenchmarkRun
from atp_sdk.models import BenchmarkInfo
from atp_sdk.sync import ATPClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    token: str = "test-tok",
    platform_url: str = "http://test",
    **kwargs: Any,
) -> ATPClient:
    """Create ATPClient with mocked async internals."""
    # Providing an explicit token bypasses load_token entirely
    client = ATPClient(platform_url=platform_url, token=token, **kwargs)
    return client


def _patch_async(client: ATPClient, method: str, return_value: Any) -> AsyncMock:
    """Patch a method on the underlying async client."""
    mock = AsyncMock(return_value=return_value)
    setattr(client._async_client, method, mock)
    return mock


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_open_close() -> None:
    """ATPClient works as a sync context manager."""
    with _make_client() as client:
        assert client is not None
        assert client.platform_url == "http://test"
    # After __exit__, thread should be stopped


def test_context_manager_calls_close() -> None:
    """__exit__ calls close() on the client."""
    client = _make_client()
    client.close = MagicMock()  # type: ignore[method-assign]
    client.__exit__(None, None, None)
    client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_token_property() -> None:
    """token property delegates to async client."""
    client = _make_client(token="my-token")
    assert client.token == "my-token"


def test_token_property_none_when_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """token is None when none provided and none in env/config."""
    monkeypatch.delenv("ATP_TOKEN", raising=False)
    with patch("atp_sdk.client.load_token", return_value=None):
        client = ATPClient(platform_url="http://test")
    assert client.token is None
    client.close()


def test_platform_url_property() -> None:
    """platform_url property returns the configured URL."""
    client = _make_client(platform_url="http://myplatform.example.com")
    assert client.platform_url == "http://myplatform.example.com"
    client.close()


# ---------------------------------------------------------------------------
# list_benchmarks
# ---------------------------------------------------------------------------


def test_list_benchmarks_works_synchronously() -> None:
    """list_benchmarks returns list of BenchmarkInfo synchronously."""
    benchmarks = [
        BenchmarkInfo(id=1, name="b1", description="first", tasks_count=5),
        BenchmarkInfo(id=2, name="b2", description="second", tasks_count=3),
    ]
    client = _make_client()
    _patch_async(client, "list_benchmarks", benchmarks)

    result = client.list_benchmarks()

    assert len(result) == 2
    assert all(isinstance(b, BenchmarkInfo) for b in result)
    assert result[0].name == "b1"
    client.close()


# ---------------------------------------------------------------------------
# get_benchmark
# ---------------------------------------------------------------------------


def test_get_benchmark() -> None:
    """get_benchmark returns BenchmarkInfo for the given id."""
    benchmark = BenchmarkInfo(id=42, name="b42", description="Test", tasks_count=10)
    client = _make_client()
    _patch_async(client, "get_benchmark", benchmark)

    result = client.get_benchmark(42)

    assert isinstance(result, BenchmarkInfo)
    assert result.id == 42
    client.close()


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------


def test_start_run_returns_benchmark_run() -> None:
    """start_run returns a BenchmarkRun."""
    mock_run = MagicMock(spec=BenchmarkRun)
    mock_run.run_id = 99
    mock_run.benchmark_id = 1

    client = _make_client()
    _patch_async(client, "start_run", mock_run)

    result = client.start_run(1, agent_name="my-agent")

    assert result is mock_run
    client.close()


def test_start_run_passes_batch_size() -> None:
    """start_run forwards batch_size to async client."""
    mock_run = MagicMock(spec=BenchmarkRun)
    client = _make_client()
    mock = _patch_async(client, "start_run", mock_run)

    client.start_run(2, agent_name="bot", batch_size=5)

    mock.assert_called_once_with(2, agent_name="bot", timeout=3600, batch_size=5)
    client.close()


# ---------------------------------------------------------------------------
# get_leaderboard
# ---------------------------------------------------------------------------


def test_get_leaderboard() -> None:
    """get_leaderboard returns list of dicts."""
    lb_data: list[dict[str, Any]] = [{"user_id": 1, "best_score": 0.9}]
    client = _make_client()
    _patch_async(client, "get_leaderboard", lb_data)

    result = client.get_leaderboard(5)

    assert result == lb_data
    client.close()


# ---------------------------------------------------------------------------
# explicit close
# ---------------------------------------------------------------------------


def test_explicit_close() -> None:
    """close() can be called explicitly without error."""
    client = _make_client()
    client.close()
    # Second close should not raise
    client.close()


# ---------------------------------------------------------------------------
# retry params forwarded
# ---------------------------------------------------------------------------


def test_retry_params_forwarded() -> None:
    """Retry params are passed through to the underlying AsyncATPClient."""
    client = ATPClient(
        platform_url="http://test",
        token="tok",
        max_retries=5,
        retry_backoff=2.0,
        max_retry_delay=60.0,
        retry_on_timeout=False,
    )
    rc = client._async_client._retry_config
    assert rc.max_retries == 5
    assert rc.retry_backoff == 2.0
    assert rc.max_retry_delay == 60.0
    assert rc.retry_on_timeout is False
    client.close()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_thread_safety_concurrent_list_benchmarks() -> None:
    """Multiple threads can call list_benchmarks concurrently."""
    benchmarks = [BenchmarkInfo(id=1, name="b1", description="d", tasks_count=2)]
    client = _make_client()
    _patch_async(client, "list_benchmarks", benchmarks)

    results: list[list[BenchmarkInfo]] = []

    def call() -> list[BenchmarkInfo]:
        return client.list_benchmarks()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(call) for _ in range(8)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    assert len(results) == 8
    for r in results:
        assert r == benchmarks

    client.close()


# ---------------------------------------------------------------------------
# login (uses sync auth directly)
# ---------------------------------------------------------------------------


def test_login_calls_sync_auth() -> None:
    """login() delegates to the sync auth.login() function."""
    client = _make_client()

    with patch("atp_sdk.sync.auth_login", return_value="new-token") as mock_login:
        token = client.login(open_browser=False)

    mock_login.assert_called_once_with(platform_url="http://test", open_browser=False)
    assert token == "new-token"
    client.close()
