"""BenchmarkRun iterator for pulling tasks from the ATP platform."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atp_sdk.client import AsyncATPClient

logger = logging.getLogger("atp_sdk")


class BenchmarkRun:
    """Async iterator that pulls tasks from the platform API.

    Yields ATPRequest dicts from the next-task endpoint and provides
    methods to submit results, check status, cancel, and view the
    leaderboard.

    Supports both async iteration (``async for task in run``) and batch
    pulls via ``next_batch(n)``.  Sync iteration via ``for task in run``
    is supported only outside a running event loop; inside an async
    context it raises ``RuntimeError``.
    """

    def __init__(
        self,
        client: AsyncATPClient,
        run_id: int,
        benchmark_id: str | int,
        batch_size: int = 1,
    ) -> None:
        self._client = client
        self.run_id = run_id
        self.benchmark_id = benchmark_id
        self.batch_size = batch_size
        self._exhausted: bool = False
        self._buffer: deque[dict[str, Any]] = deque()

    async def _fetch_batch(self, batch: int | None = None) -> list[dict[str, Any]]:
        """Fetch up to *batch* tasks from the server.

        Args:
            batch: Number of tasks to request. None / 1 → no query param.

        Returns:
            List of task dicts. Empty list when server returns 204.
        """
        if self._exhausted:
            return []

        n = batch if batch is not None else self.batch_size
        params: dict[str, Any] = {}
        if n > 1:
            params["batch"] = n

        resp = await self._client._request(
            "GET",
            f"/api/v1/runs/{self.run_id}/next-task",
            params=params if params else None,
        )

        if resp.status_code == 204:
            self._exhausted = True
            logger.debug("run %d exhausted (204)", self.run_id)
            return []

        resp.raise_for_status()
        data: Any = resp.json()

        # Server may return a single dict (batch=1 compat) or a list
        if isinstance(data, dict):
            data = [data]

        logger.debug("run %d fetched %d tasks", self.run_id, len(data))
        return data

    async def next_batch(self, n: int) -> list[dict[str, Any]]:
        """Pull up to *n* tasks from the server.

        Returns an empty list when the run is exhausted. Never raises
        ``StopIteration``.

        Args:
            n: Maximum number of tasks to retrieve.

        Returns:
            List of task dicts (may be shorter than *n* or empty).
        """
        return await self._fetch_batch(n)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self

    async def __anext__(self) -> dict[str, Any]:
        # Drain the local buffer first
        if self._buffer:
            return self._buffer.popleft()

        if self._exhausted:
            raise StopAsyncIteration

        tasks = await self._fetch_batch(self.batch_size)
        if not tasks:
            raise StopAsyncIteration

        # Queue all but the first
        for task in tasks[1:]:
            self._buffer.append(task)
        return tasks[0]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Sync iteration — only valid outside a running event loop.

        Raises:
            RuntimeError: If called from inside a running event loop.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot use sync iteration inside async context. "
                "Use 'async for task in run:' instead."
            )
        except RuntimeError as exc:
            if "async context" in str(exc):
                raise
            # No running loop — use client's background loop if available

        async def _collect_all() -> list[dict[str, Any]]:
            return [task async for task in self]

        # Use the sync wrapper's background loop if available
        # (set by ATPClient.start_run via _sync_loop attribute)
        sync_loop = getattr(self, "_sync_loop", None)
        if sync_loop is not None and sync_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_collect_all(), sync_loop)
            return iter(future.result())

        return iter(asyncio.run(_collect_all()))

    # ------------------------------------------------------------------
    # Internal helper for sync dispatch
    # ------------------------------------------------------------------

    def _run_sync(self, coro: Any) -> Any:
        """Dispatch *coro* to the background loop set by ATPClient.

        Raises:
            RuntimeError: If no ``_sync_loop`` is attached (i.e. the run
                was created via ``AsyncATPClient``, not ``ATPClient``).
        """
        sync_loop = getattr(self, "_sync_loop", None)
        if sync_loop is None or not sync_loop.is_running():
            # Close the coroutine to avoid "was never awaited" warning
            coro.close()
            raise RuntimeError(
                "Sync methods require a BenchmarkRun obtained from "
                "ATPClient.start_run(). Use the async methods with "
                "AsyncATPClient instead."
            )
        future = asyncio.run_coroutine_threadsafe(coro, sync_loop)
        return future.result()

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def submit(
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

        Returns:
            Score dict from the server.
        """
        payload: dict[str, Any] = {
            "response": response,
            "task_index": task_index,
        }
        if events is not None:
            payload["events"] = events

        resp = await self._client._request(
            "POST",
            f"/api/v1/runs/{self.run_id}/submit",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def status(self) -> dict[str, Any]:
        """Get the current run status."""
        resp = await self._client._request(
            "GET",
            f"/api/v1/runs/{self.run_id}/status",
        )
        resp.raise_for_status()
        return resp.json()

    async def cancel(self) -> None:
        """Cancel the benchmark run."""
        resp = await self._client._request(
            "POST",
            f"/api/v1/runs/{self.run_id}/cancel",
        )
        resp.raise_for_status()

    async def leaderboard(self) -> list[dict[str, Any]]:
        """Get the leaderboard for this benchmark."""
        resp = await self._client._request(
            "GET",
            f"/api/v1/benchmarks/{self.benchmark_id}/leaderboard",
        )
        resp.raise_for_status()
        return resp.json()

    async def emit(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Send events to the server during run execution.

        Events are appended to the run's event log. Maximum 1000
        events per run.

        Args:
            events: List of event dicts with event_type, data,
                timestamp.

        Returns:
            Dict with accepted count and total events.
        """
        resp = await self._client._request(
            "POST",
            f"/api/v1/runs/{self.run_id}/events",
            json={"events": events},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Sync wrappers (require _sync_loop from ATPClient.start_run)
    # ------------------------------------------------------------------

    def submit_sync(
        self,
        response: dict[str, Any],
        task_index: int,
        events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Sync version of :meth:`submit`."""
        return self._run_sync(self.submit(response, task_index, events))

    def status_sync(self) -> dict[str, Any]:
        """Sync version of :meth:`status`."""
        return self._run_sync(self.status())

    def cancel_sync(self) -> None:
        """Sync version of :meth:`cancel`."""
        self._run_sync(self.cancel())

    def leaderboard_sync(self) -> list[dict[str, Any]]:
        """Sync version of :meth:`leaderboard`."""
        return self._run_sync(self.leaderboard())

    def emit_sync(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Sync version of :meth:`emit`."""
        return self._run_sync(self.emit(events))

    def next_batch_sync(self, n: int) -> list[dict[str, Any]]:
        """Sync version of :meth:`next_batch`."""
        return self._run_sync(self.next_batch(n))
