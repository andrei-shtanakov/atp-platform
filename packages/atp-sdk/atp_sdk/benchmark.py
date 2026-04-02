"""BenchmarkRun iterator for pulling tasks from the ATP platform."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx


class BenchmarkRun:
    """Iterator that pulls tasks from the platform API.

    Yields ATPRequest dicts from the next-task endpoint and provides
    methods to submit results, check status, cancel, and view the
    leaderboard.
    """

    def __init__(
        self,
        http: httpx.Client,
        run_id: int,
        benchmark_id: str | int,
    ) -> None:
        self._http = http
        self.run_id = run_id
        self.benchmark_id = benchmark_id

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield ATPRequest dicts until the server returns 204."""
        while True:
            resp = self._http.get(f"/api/v1/runs/{self.run_id}/next-task")
            if resp.status_code == 204:
                return
            resp.raise_for_status()
            yield resp.json()

    def submit(
        self,
        response: dict[str, Any],
        task_index: int,
        events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Submit a task response and optional events.

        Args:
            response: ATPResponse as a dict.
            task_index: Task index from ATPRequest.metadata.task_index.
            events: Optional list of ATPEvent dicts.
        """
        payload: dict[str, Any] = {
            "response": response,
            "task_index": task_index,
        }
        if events is not None:
            payload["events"] = events
        resp = self._http.post(
            f"/api/v1/runs/{self.run_id}/submit",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def status(self) -> dict[str, Any]:
        """Get the current run status."""
        resp = self._http.get(f"/api/v1/runs/{self.run_id}/status")
        resp.raise_for_status()
        return resp.json()

    def cancel(self) -> None:
        """Cancel the benchmark run."""
        resp = self._http.post(f"/api/v1/runs/{self.run_id}/cancel")
        resp.raise_for_status()

    def leaderboard(self) -> list[dict[str, Any]]:
        """Get the leaderboard for this benchmark."""
        resp = self._http.get(f"/api/v1/benchmarks/{self.benchmark_id}/leaderboard")
        resp.raise_for_status()
        return resp.json()
