"""ATP SDK async client for interacting with the ATP benchmark platform."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable
from types import TracebackType
from typing import Any

import httpx

from atp_sdk.auth import load_token
from atp_sdk.benchmark import BenchmarkRun
from atp_sdk.models import BenchmarkInfo
from atp_sdk.retry import RetryConfig, retry_request

logger = logging.getLogger("atp_sdk")


class AsyncATPClient:
    """Async client for the ATP benchmark platform API.

    Token resolution order:
    1. Explicit ``token`` argument
    2. ``ATP_TOKEN`` environment variable
    3. Saved token from ``~/.atp/config.json``
    """

    def __init__(
        self,
        platform_url: str = "http://localhost:8000",
        token: str | None = None,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        max_retry_delay: float = 30.0,
        retry_on_timeout: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self.platform_url = platform_url.rstrip("/")
        self.token = (
            token
            or os.environ.get("ATP_TOKEN")
            or load_token(platform_url=self.platform_url)
        )
        self._retry_config = RetryConfig(
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            max_retry_delay=max_retry_delay,
            retry_on_timeout=retry_on_timeout,
        )
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._http = httpx.AsyncClient(
            base_url=self.platform_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Execute an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: URL path relative to platform_url.
            **kwargs: Additional arguments passed to httpx.AsyncClient.request.

        Returns:
            The httpx.Response.
        """

        def sender() -> Awaitable[httpx.Response]:
            return self._http.request(method, url, **kwargs)

        response = await retry_request(sender, self._retry_config)
        return response

    async def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List all available benchmarks."""
        resp = await self._request("GET", "/api/v1/benchmarks")
        resp.raise_for_status()
        return [BenchmarkInfo.model_validate(b) for b in resp.json()]

    async def get_benchmark(self, benchmark_id: str | int) -> BenchmarkInfo:
        """Get details of a specific benchmark.

        Args:
            benchmark_id: ID of the benchmark.
        """
        resp = await self._request("GET", f"/api/v1/benchmarks/{benchmark_id}")
        resp.raise_for_status()
        return BenchmarkInfo.model_validate(resp.json())

    async def start_run(
        self,
        benchmark_id: str | int,
        agent_name: str = "",
        timeout: int = 3600,
        batch_size: int = 1,
    ) -> BenchmarkRun:
        """Start a new benchmark run.

        Args:
            benchmark_id: ID of the benchmark to run.
            agent_name: Name of the agent participating.
            timeout: Run timeout in seconds.
            batch_size: Number of tasks to pull per batch.

        Returns:
            A BenchmarkRun for pulling and submitting tasks.
        """
        resp = await self._request(
            "POST",
            f"/api/v1/benchmarks/{benchmark_id}/start",
            params={"agent_name": agent_name, "timeout": timeout},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return BenchmarkRun(
            client=self,
            run_id=data["id"],
            benchmark_id=benchmark_id,
            batch_size=batch_size,
        )

    async def get_leaderboard(self, benchmark_id: str | int) -> list[dict[str, Any]]:
        """Get the leaderboard for a benchmark.

        Args:
            benchmark_id: ID of the benchmark.
        """
        resp = await self._request(
            "GET", f"/api/v1/benchmarks/{benchmark_id}/leaderboard"
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncATPClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
