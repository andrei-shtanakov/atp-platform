"""Sync wrapper over AsyncATPClient via a background thread + event loop.

Thread-safe: one background thread owns the event loop; all public methods
dispatch coroutines via ``asyncio.run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any

from atp_sdk.auth import login as auth_login
from atp_sdk.benchmark import BenchmarkRun
from atp_sdk.client import AsyncATPClient
from atp_sdk.models import BenchmarkInfo

logger = logging.getLogger("atp_sdk")

_THREAD_NAME = "atp-sdk-sync"
_JOIN_TIMEOUT = 5.0


class ATPClient:
    """Sync wrapper over AsyncATPClient via background thread + event loop.

    Thread-safe: can be shared across multiple threads.

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
        on_token_expired: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name=_THREAD_NAME,
            daemon=True,
        )
        self._thread.start()
        logger.debug("ATPClient background thread started (%s)", _THREAD_NAME)

        self._async_client = AsyncATPClient(
            platform_url=platform_url,
            token=token,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            max_retry_delay=max_retry_delay,
            retry_on_timeout=retry_on_timeout,
            timeout=timeout,
            on_token_expired=on_token_expired,
        )
        self._closed = False

    def _run(self, coro: Any) -> Any:
        """Dispatch *coro* to the background loop and block until done.

        Thread-safe: may be called from any thread.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def token(self) -> str | None:
        """JWT access token used for API authentication."""
        return self._async_client.token

    @property
    def platform_url(self) -> str:
        """Base URL of the ATP platform."""
        return self._async_client.platform_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def login(self, open_browser: bool = True) -> str:
        """Authenticate via Device Flow and save the token.

        Uses the sync auth module directly (webbrowser + polling is
        inherently synchronous).

        Args:
            open_browser: Whether to open the browser automatically.

        Returns:
            JWT access token.
        """
        token = auth_login(platform_url=self.platform_url, open_browser=open_browser)
        # Update the async client's token for subsequent requests
        self._async_client.token = token
        return token

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List all available benchmarks."""
        return self._run(self._async_client.list_benchmarks())

    def get_benchmark(self, benchmark_id: str | int) -> BenchmarkInfo:
        """Get details of a specific benchmark.

        Args:
            benchmark_id: ID of the benchmark.
        """
        return self._run(self._async_client.get_benchmark(benchmark_id))

    def start_run(
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
        run = self._run(
            self._async_client.start_run(
                benchmark_id,
                agent_name=agent_name,
                timeout=timeout,
                batch_size=batch_size,
            )
        )
        # Attach background loop so sync __iter__ can use it
        run._sync_loop = self._loop
        return run

    def get_leaderboard(self, benchmark_id: str | int) -> list[dict[str, Any]]:
        """Get the leaderboard for a benchmark.

        Args:
            benchmark_id: ID of the benchmark.
        """
        return self._run(self._async_client.get_leaderboard(benchmark_id))

    def close(self) -> None:
        """Close the async client, stop the background loop, and join the thread."""
        if self._closed:
            return
        self._closed = True
        try:
            self._run(self._async_client.close())
        except Exception:
            logger.debug("Error closing async client", exc_info=True)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=_JOIN_TIMEOUT)
        logger.debug("ATPClient background thread stopped")

    def __enter__(self) -> ATPClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
