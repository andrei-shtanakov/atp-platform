"""Tests for AsyncATPClient (TDD)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from atp_sdk.client import AsyncATPClient
from atp_sdk.models import BenchmarkInfo
from atp_sdk.retry import RetryConfig

# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_context_manager_opens_and_closes() -> None:
    """AsyncATPClient works as an async context manager."""
    async with AsyncATPClient(platform_url="http://test", token="tok") as client:
        assert client is not None
        assert client.platform_url == "http://test"
    # After exit, _http should be closed (aclose called)
    # We can't assert on the closed state easily, but no exception means success


@pytest.mark.anyio
async def test_context_manager_calls_close() -> None:
    """__aexit__ calls close() which calls _http.aclose()."""
    client = AsyncATPClient(platform_url="http://test", token="tok")
    client._http = MagicMock()
    client._http.aclose = AsyncMock()

    await client.__aenter__()
    await client.__aexit__(None, None, None)

    client._http.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# Explicit close
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_explicit_close() -> None:
    """close() calls _http.aclose()."""
    client = AsyncATPClient(platform_url="http://test", token="tok")
    client._http = MagicMock()
    client._http.aclose = AsyncMock()

    await client.close()

    client._http.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def test_explicit_token_takes_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit token overrides env var and saved config."""
    monkeypatch.setenv("ATP_TOKEN", "env-token")
    with patch("atp_sdk.client.load_token", return_value="saved-token"):
        client = AsyncATPClient(platform_url="http://test", token="explicit-token")
    assert client.token == "explicit-token"


def test_token_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token falls back to ATP_TOKEN env var when no explicit token."""
    monkeypatch.setenv("ATP_TOKEN", "env-token")
    with patch("atp_sdk.client.load_token", return_value=None):
        client = AsyncATPClient(platform_url="http://test")
    assert client.token == "env-token"


def test_token_from_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token falls back to saved config when no explicit token or env var."""
    monkeypatch.delenv("ATP_TOKEN", raising=False)
    with patch("atp_sdk.client.load_token", return_value="saved-token"):
        client = AsyncATPClient(platform_url="http://test")
    assert client.token == "saved-token"


def test_no_token_when_none_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token is None when no source provides one."""
    monkeypatch.delenv("ATP_TOKEN", raising=False)
    with patch("atp_sdk.client.load_token", return_value=None):
        client = AsyncATPClient(platform_url="http://test")
    assert client.token is None


# ---------------------------------------------------------------------------
# Retry config forwarded correctly
# ---------------------------------------------------------------------------


def test_retry_config_params_forwarded() -> None:
    """RetryConfig is built from constructor params."""
    client = AsyncATPClient(
        platform_url="http://test",
        token="tok",
        max_retries=5,
        retry_backoff=2.0,
        max_retry_delay=60.0,
        retry_on_timeout=False,
    )
    assert client._retry_config.max_retries == 5
    assert client._retry_config.retry_backoff == 2.0
    assert client._retry_config.max_retry_delay == 60.0
    assert client._retry_config.retry_on_timeout is False


def test_retry_config_defaults() -> None:
    """Default retry config matches RetryConfig defaults."""
    client = AsyncATPClient(platform_url="http://test", token="tok")
    assert client._retry_config == RetryConfig()


# ---------------------------------------------------------------------------
# list_benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_benchmarks_returns_benchmark_info_list() -> None:
    """list_benchmarks returns list of BenchmarkInfo parsed from response."""
    payload: list[dict[str, Any]] = [
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

    client = AsyncATPClient(platform_url="http://test", token="tok")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = payload
    client._request = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

    result = await client.list_benchmarks()

    assert len(result) == 2
    assert all(isinstance(b, BenchmarkInfo) for b in result)
    assert result[0].name == "bench-1"
    assert result[1].tasks_count == 3
    client._request.assert_called_once_with("GET", "/api/v1/benchmarks")


# ---------------------------------------------------------------------------
# get_benchmark
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_benchmark_returns_benchmark_info() -> None:
    """get_benchmark returns a single BenchmarkInfo parsed from response."""
    payload: dict[str, Any] = {
        "id": 42,
        "name": "bench-42",
        "description": "Test",
        "tasks_count": 10,
        "tags": [],
        "version": "1.0",
    }

    client = AsyncATPClient(platform_url="http://test", token="tok")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = payload
    client._request = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

    result = await client.get_benchmark(42)

    assert isinstance(result, BenchmarkInfo)
    assert result.id == 42
    assert result.name == "bench-42"
    client._request.assert_called_once_with("GET", "/api/v1/benchmarks/42")


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_start_run_returns_benchmark_run_with_correct_run_id() -> None:
    """start_run returns BenchmarkRun with run_id from server response."""
    run_data: dict[str, Any] = {
        "id": 99,
        "benchmark_id": 1,
        "agent_name": "my-agent",
        "status": "in_progress",
    }

    client = AsyncATPClient(platform_url="http://test", token="tok")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = run_data

    with patch("atp_sdk.client.BenchmarkRun") as mock_benchmark_run_cls:
        mock_run = MagicMock()
        mock_run.run_id = 99
        mock_benchmark_run_cls.return_value = mock_run

        client._request = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

        result = await client.start_run(1, agent_name="my-agent")

        mock_benchmark_run_cls.assert_called_once_with(
            client=client,
            run_id=99,
            benchmark_id=1,
            batch_size=1,
        )
        assert result.run_id == 99


@pytest.mark.anyio
async def test_start_run_passes_batch_size() -> None:
    """start_run passes batch_size to BenchmarkRun constructor."""
    run_data: dict[str, Any] = {"id": 7, "benchmark_id": 2, "status": "in_progress"}

    client = AsyncATPClient(platform_url="http://test", token="tok")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = run_data

    with patch("atp_sdk.client.BenchmarkRun") as mock_benchmark_run_cls:
        client._request = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]
        await client.start_run(2, batch_size=5)

        mock_benchmark_run_cls.assert_called_once_with(
            client=client,
            run_id=7,
            benchmark_id=2,
            batch_size=5,
        )


# ---------------------------------------------------------------------------
# get_leaderboard
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_leaderboard_returns_list() -> None:
    """get_leaderboard returns raw list of dicts from response."""
    payload: list[dict[str, Any]] = [
        {"user_id": 1, "agent_name": "a", "best_score": 0.9},
    ]

    client = AsyncATPClient(platform_url="http://test", token="tok")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = payload
    client._request = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

    result = await client.get_leaderboard(5)

    assert len(result) == 1
    assert result[0]["best_score"] == 0.9
    client._request.assert_called_once_with("GET", "/api/v1/benchmarks/5/leaderboard")
