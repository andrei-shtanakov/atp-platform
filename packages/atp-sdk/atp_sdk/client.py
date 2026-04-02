"""ATP SDK client for interacting with the ATP benchmark platform."""

from __future__ import annotations

import os
from types import TracebackType
from typing import Any

import httpx

from atp_sdk.benchmark import BenchmarkRun
from atp_sdk.models import BenchmarkInfo


class ATPClient:
    """Client for the ATP benchmark platform API."""

    def __init__(
        self,
        platform_url: str = "http://localhost:8000",
        token: str | None = None,
    ) -> None:
        self.platform_url = platform_url.rstrip("/")
        self.token = token or os.environ.get("ATP_TOKEN")
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._http = httpx.Client(
            base_url=self.platform_url,
            headers=headers,
        )

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List all available benchmarks."""
        resp = self._http.get("/api/v1/benchmarks")
        resp.raise_for_status()
        return [BenchmarkInfo.model_validate(b) for b in resp.json()]

    def get_benchmark(self, benchmark_id: str | int) -> BenchmarkInfo:
        """Get details of a specific benchmark."""
        resp = self._http.get(f"/api/v1/benchmarks/{benchmark_id}")
        resp.raise_for_status()
        return BenchmarkInfo.model_validate(resp.json())

    def start_run(
        self,
        benchmark_id: str | int,
        agent_name: str = "",
        timeout: int = 3600,
    ) -> BenchmarkRun:
        """Start a new benchmark run.

        Returns a BenchmarkRun iterator for pulling tasks.
        """
        resp = self._http.post(
            f"/api/v1/benchmarks/{benchmark_id}/start",
            params={"agent_name": agent_name, "timeout": timeout},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return BenchmarkRun(
            http=self._http,
            run_id=data["id"],
            benchmark_id=benchmark_id,
        )

    def get_leaderboard(self, benchmark_id: str | int) -> list[dict[str, Any]]:
        """Get the leaderboard for a benchmark."""
        resp = self._http.get(f"/api/v1/benchmarks/{benchmark_id}/leaderboard")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> ATPClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
