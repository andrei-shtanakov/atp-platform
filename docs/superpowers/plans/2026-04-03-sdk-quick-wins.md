# SDK Quick Wins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add batch task fetching, async-first client with sync wrapper, and retry with exponential backoff to `atp-platform-sdk`.

**Architecture:** Rewrite SDK client as async-first (`AsyncATPClient`), wrap with sync `ATPClient` via background thread. Add `?batch=N` query param to server's next-task endpoint. Internal `_request()` method handles retries with full jitter backoff.

**Tech Stack:** httpx (async), asyncio, pydantic, FastAPI, SQLAlchemy, pytest + anyio

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `packages/atp-sdk/atp_sdk/retry.py` | Retry logic with exponential backoff + jitter |
| Rewrite | `packages/atp-sdk/atp_sdk/client.py` | `AsyncATPClient` — async-first client |
| Create | `packages/atp-sdk/atp_sdk/sync.py` | `ATPClient` — sync wrapper via background thread |
| Rewrite | `packages/atp-sdk/atp_sdk/benchmark.py` | `BenchmarkRun` with batch support + `__aiter__` |
| Modify | `packages/atp-sdk/atp_sdk/__init__.py` | Re-export `AsyncATPClient`, update `__all__` |
| Modify | `packages/atp-sdk/pyproject.toml` | Bump version to 2.0.0 |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py` | Add `?batch=N` to next-task endpoint |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/config.py` | Add `batch_max_size` setting |
| Create | `tests/unit/sdk/test_retry.py` | Tests for retry logic |
| Create | `tests/unit/sdk/test_async_client.py` | Tests for AsyncATPClient |
| Create | `tests/unit/sdk/test_sync_client.py` | Tests for sync ATPClient wrapper |
| Create | `tests/unit/sdk/test_benchmark_batch.py` | Tests for batch iteration |
| Modify | `tests/unit/benchmark/test_benchmark_api.py` | Tests for `?batch=N` server endpoint |

---

### Task 1: Implement retry module

**Files:**
- Create: `packages/atp-sdk/atp_sdk/retry.py`
- Create: `tests/unit/sdk/test_retry.py`

- [ ] **Step 1: Write tests for retry logic**

Create `tests/unit/sdk/test_retry.py`:

```python
"""Tests for retry logic with exponential backoff."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from atp_sdk.retry import RetryConfig, retry_request


class TestRetryConfig:
    """Tests for RetryConfig defaults and validation."""

    def test_defaults(self) -> None:
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.retry_backoff == 1.0
        assert cfg.max_retry_delay == 30.0
        assert cfg.retry_on_timeout is True

    def test_disabled(self) -> None:
        cfg = RetryConfig(max_retries=0)
        assert cfg.max_retries == 0


class TestRetryRequest:
    """Tests for the retry_request wrapper."""

    @pytest.mark.anyio
    async def test_no_retry_on_success(self) -> None:
        """Successful request should not retry."""
        response = httpx.Response(200, json={"ok": True})
        sender = AsyncMock(return_value=response)

        result = await retry_request(sender, RetryConfig())
        assert result.status_code == 200
        assert sender.call_count == 1

    @pytest.mark.anyio
    async def test_retry_on_502(self) -> None:
        """502 should trigger retry."""
        fail = httpx.Response(502)
        ok = httpx.Response(200, json={"ok": True})
        sender = AsyncMock(side_effect=[fail, ok])

        result = await retry_request(
            sender, RetryConfig(retry_backoff=0.01)
        )
        assert result.status_code == 200
        assert sender.call_count == 2

    @pytest.mark.anyio
    async def test_retry_on_503(self) -> None:
        """503 should trigger retry."""
        fail = httpx.Response(503)
        ok = httpx.Response(200, json={"ok": True})
        sender = AsyncMock(side_effect=[fail, ok])

        result = await retry_request(
            sender, RetryConfig(retry_backoff=0.01)
        )
        assert result.status_code == 200
        assert sender.call_count == 2

    @pytest.mark.anyio
    async def test_retry_on_transport_error(self) -> None:
        """Network error should trigger retry."""
        ok = httpx.Response(200, json={"ok": True})
        sender = AsyncMock(
            side_effect=[httpx.ConnectError("conn refused"), ok]
        )

        result = await retry_request(
            sender, RetryConfig(retry_backoff=0.01)
        )
        assert result.status_code == 200
        assert sender.call_count == 2

    @pytest.mark.anyio
    async def test_no_retry_on_400(self) -> None:
        """400 should not trigger retry."""
        response = httpx.Response(400, json={"error": "bad"})
        sender = AsyncMock(return_value=response)

        result = await retry_request(sender, RetryConfig())
        assert result.status_code == 400
        assert sender.call_count == 1

    @pytest.mark.anyio
    async def test_retry_exhausted_raises(self) -> None:
        """All retries exhausted should raise last error."""
        sender = AsyncMock(
            side_effect=httpx.ConnectError("conn refused")
        )

        with pytest.raises(httpx.ConnectError):
            await retry_request(
                sender,
                RetryConfig(max_retries=2, retry_backoff=0.01),
            )
        assert sender.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.anyio
    async def test_retry_exhausted_returns_last_response(self) -> None:
        """All retries with bad status should return last response."""
        fail = httpx.Response(503)
        sender = AsyncMock(return_value=fail)

        result = await retry_request(
            sender,
            RetryConfig(max_retries=2, retry_backoff=0.01),
        )
        assert result.status_code == 503
        assert sender.call_count == 3

    @pytest.mark.anyio
    async def test_retry_on_429_with_retry_after(self) -> None:
        """429 with Retry-After header should wait."""
        fail = httpx.Response(
            429, headers={"Retry-After": "0"}
        )
        ok = httpx.Response(200, json={"ok": True})
        sender = AsyncMock(side_effect=[fail, ok])

        result = await retry_request(
            sender, RetryConfig(retry_backoff=0.01)
        )
        assert result.status_code == 200
        assert sender.call_count == 2

    @pytest.mark.anyio
    async def test_no_retry_on_timeout_when_disabled(self) -> None:
        """Timeout should not retry when retry_on_timeout=False."""
        sender = AsyncMock(
            side_effect=httpx.ReadTimeout("read timeout")
        )

        with pytest.raises(httpx.ReadTimeout):
            await retry_request(
                sender,
                RetryConfig(
                    max_retries=2,
                    retry_on_timeout=False,
                    retry_backoff=0.01,
                ),
            )
        assert sender.call_count == 1

    @pytest.mark.anyio
    async def test_max_retry_delay_caps_backoff(self) -> None:
        """Backoff delay should not exceed max_retry_delay."""
        delays: list[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            delays.append(delay)

        fail = httpx.Response(503)
        ok = httpx.Response(200)
        sender = AsyncMock(side_effect=[fail, fail, fail, ok])

        with patch("atp_sdk.retry.asyncio.sleep", mock_sleep):
            await retry_request(
                sender,
                RetryConfig(
                    max_retries=3,
                    retry_backoff=1.0,
                    max_retry_delay=2.0,
                ),
            )
        # All delays should be <= max_retry_delay
        for d in delays:
            assert d <= 2.0

    @pytest.mark.anyio
    async def test_disabled_retries(self) -> None:
        """max_retries=0 should not retry."""
        sender = AsyncMock(
            side_effect=httpx.ConnectError("conn refused")
        )

        with pytest.raises(httpx.ConnectError):
            await retry_request(
                sender, RetryConfig(max_retries=0)
            )
        assert sender.call_count == 1
```

- [ ] **Step 2: Run tests to see them fail**

Run: `cd /Users/Andrei_Shtanakov/labs/all_ai_orchestrators/atp-platform && uv run python -m pytest tests/unit/sdk/test_retry.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement retry module**

Create `packages/atp-sdk/atp_sdk/retry.py`:

```python
"""Retry logic with exponential backoff and full jitter.

Full jitter: delay = random(0, min(max_delay, base * 2^attempt))
See: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

logger = logging.getLogger("atp_sdk")

RETRYABLE_STATUS_CODES = {502, 503, 504}


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    retry_backoff: float = 1.0
    max_retry_delay: float = 30.0
    retry_on_timeout: bool = True


def _is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES or status_code == 429


async def retry_request(
    sender: Callable[[], Awaitable[httpx.Response]],
    config: RetryConfig,
) -> httpx.Response:
    """Execute an HTTP request with retry logic.

    Args:
        sender: Async callable that performs the HTTP request.
        config: Retry configuration.

    Returns:
        The HTTP response.

    Raises:
        httpx.TransportError: If all retries are exhausted and
            the last attempt raised a transport error.
    """
    last_exception: httpx.TransportError | None = None
    last_response: httpx.Response | None = None

    for attempt in range(1 + config.max_retries):
        try:
            response = await sender()
        except httpx.TimeoutException as exc:
            if not config.retry_on_timeout:
                raise
            last_exception = exc
            last_response = None
        except httpx.TransportError as exc:
            last_exception = exc
            last_response = None
        else:
            last_exception = None
            last_response = response

            if not _is_retryable_status(response.status_code):
                return response

            # 429 with Retry-After
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    delay = float(retry_after)
                    if attempt < config.max_retries:
                        logger.warning(
                            "Rate limited, retry %d/%d, "
                            "Retry-After=%.1fs",
                            attempt + 1,
                            config.max_retries,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

        if attempt >= config.max_retries:
            break

        # Full jitter backoff
        max_delay = min(
            config.max_retry_delay,
            config.retry_backoff * (2**attempt),
        )
        delay = random.uniform(0, max_delay)

        status_info = ""
        if last_response is not None:
            status_info = f" ({last_response.status_code})"
        elif last_exception is not None:
            status_info = f" ({type(last_exception).__name__})"

        logger.warning(
            "Retry %d/%d%s, delay=%.2fs",
            attempt + 1,
            config.max_retries,
            status_info,
            delay,
        )
        await asyncio.sleep(delay)

    # All retries exhausted
    if last_exception is not None:
        raise last_exception
    assert last_response is not None
    return last_response
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/sdk/test_retry.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff + pyrefly**

Run: `uv run ruff format packages/atp-sdk/ tests/unit/sdk/ && uv run ruff check packages/atp-sdk/ tests/unit/sdk/ --fix && uv run pyrefly check`

- [ ] **Step 6: Commit**

```bash
git add packages/atp-sdk/atp_sdk/retry.py tests/unit/sdk/test_retry.py
git commit -m "feat(sdk): add retry module with exponential backoff and full jitter"
```

---

### Task 2: Implement AsyncATPClient

**Files:**
- Rewrite: `packages/atp-sdk/atp_sdk/client.py`
- Create: `tests/unit/sdk/test_async_client.py`

- [ ] **Step 1: Write tests for AsyncATPClient**

Create `tests/unit/sdk/test_async_client.py`:

```python
"""Tests for AsyncATPClient."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from atp_sdk.client import AsyncATPClient


class TestAsyncATPClient:
    """Tests for the async-first ATP client."""

    @pytest.mark.anyio
    async def test_context_manager(self) -> None:
        """Test async context manager opens and closes properly."""
        async with AsyncATPClient(
            platform_url="http://test:8000"
        ) as client:
            assert client._http is not None
        assert client._http.is_closed

    @pytest.mark.anyio
    async def test_list_benchmarks(self) -> None:
        """Test listing benchmarks."""
        mock_response = httpx.Response(
            200,
            json=[
                {
                    "id": 1,
                    "name": "bench-1",
                    "description": "desc",
                    "tasks_count": 5,
                }
            ],
        )

        async with AsyncATPClient(
            platform_url="http://test:8000"
        ) as client:
            with patch.object(
                client,
                "_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                result = await client.list_benchmarks()
                assert len(result) == 1
                assert result[0].name == "bench-1"

    @pytest.mark.anyio
    async def test_start_run(self) -> None:
        """Test starting a benchmark run."""
        mock_response = httpx.Response(
            200,
            json={
                "id": 42,
                "benchmark_id": 1,
                "agent_name": "test",
                "status": "in_progress",
                "adapter_type": "sdk",
                "current_task_index": 0,
            },
        )

        async with AsyncATPClient(
            platform_url="http://test:8000"
        ) as client:
            with patch.object(
                client,
                "_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                run = await client.start_run(
                    benchmark_id=1, agent_name="test"
                )
                assert run.run_id == 42

    @pytest.mark.anyio
    async def test_retry_config_passed(self) -> None:
        """Test that retry config is respected."""
        async with AsyncATPClient(
            platform_url="http://test:8000",
            max_retries=5,
            retry_backoff=2.0,
            max_retry_delay=60.0,
            timeout=15.0,
        ) as client:
            assert client._retry_config.max_retries == 5
            assert client._retry_config.retry_backoff == 2.0
            assert client._retry_config.max_retry_delay == 60.0

    @pytest.mark.anyio
    async def test_token_from_env(self) -> None:
        """Test token resolution from environment."""
        with patch.dict(
            "os.environ", {"ATP_TOKEN": "env-token"}
        ):
            async with AsyncATPClient(
                platform_url="http://test:8000"
            ) as client:
                assert client.token == "env-token"

    @pytest.mark.anyio
    async def test_explicit_token(self) -> None:
        """Test explicit token takes priority."""
        with patch.dict(
            "os.environ", {"ATP_TOKEN": "env-token"}
        ):
            async with AsyncATPClient(
                platform_url="http://test:8000",
                token="explicit-token",
            ) as client:
                assert client.token == "explicit-token"

    @pytest.mark.anyio
    async def test_close(self) -> None:
        """Test explicit close."""
        client = AsyncATPClient(
            platform_url="http://test:8000"
        )
        await client.close()
        assert client._http.is_closed

    @pytest.mark.anyio
    async def test_logging_on_request(self) -> None:
        """Test that requests are logged."""
        mock_response = httpx.Response(200, json=[])

        async with AsyncATPClient(
            platform_url="http://test:8000"
        ) as client:
            with patch.object(
                client._http,
                "request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                await client._request("GET", "/api/v1/benchmarks")
```

- [ ] **Step 2: Run tests to see them fail**

Run: `uv run python -m pytest tests/unit/sdk/test_async_client.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite client.py as AsyncATPClient**

Rewrite `packages/atp-sdk/atp_sdk/client.py`:

```python
"""Async-first ATP SDK client.

AsyncATPClient is the primary client for interacting with the ATP
benchmark platform. For sync usage, use ATPClient from atp_sdk.sync.
"""

from __future__ import annotations

import logging
import os
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

    Args:
        platform_url: Base URL of the ATP platform.
        token: Optional explicit auth token.
        max_retries: Max retry attempts (0 to disable).
        retry_backoff: Base delay for exponential backoff.
        max_retry_delay: Maximum delay between retries.
        retry_on_timeout: Whether to retry on timeout errors.
        timeout: HTTP timeout in seconds.
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
        logger.debug(
            "AsyncATPClient connected to %s", self.platform_url
        )

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send an HTTP request with retry logic.

        All public methods go through this method.
        """

        async def sender() -> httpx.Response:
            return await self._http.request(method, url, **kwargs)

        return await retry_request(sender, self._retry_config)

    async def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List all available benchmarks."""
        resp = await self._request("GET", "/api/v1/benchmarks")
        resp.raise_for_status()
        return [BenchmarkInfo.model_validate(b) for b in resp.json()]

    async def get_benchmark(
        self, benchmark_id: str | int
    ) -> BenchmarkInfo:
        """Get details of a specific benchmark."""
        resp = await self._request(
            "GET", f"/api/v1/benchmarks/{benchmark_id}"
        )
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
            benchmark_id: ID of the benchmark.
            agent_name: Name of the agent.
            timeout: Run timeout in seconds.
            batch_size: Number of tasks to prefetch per request.

        Returns:
            BenchmarkRun iterator for pulling tasks.
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

    async def get_leaderboard(
        self, benchmark_id: str | int
    ) -> list[dict[str, Any]]:
        """Get the leaderboard for a benchmark."""
        resp = await self._request(
            "GET",
            f"/api/v1/benchmarks/{benchmark_id}/leaderboard",
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()
        logger.debug("AsyncATPClient closed")

    async def __aenter__(self) -> AsyncATPClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/sdk/test_async_client.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff + pyrefly**

Run: `uv run ruff format packages/atp-sdk/ tests/unit/sdk/ && uv run ruff check packages/atp-sdk/ tests/unit/sdk/ --fix && uv run pyrefly check`

- [ ] **Step 6: Commit**

```bash
git add packages/atp-sdk/atp_sdk/client.py tests/unit/sdk/test_async_client.py
git commit -m "feat(sdk): rewrite client as async-first AsyncATPClient with retry"
```

---

### Task 3: Implement sync wrapper (ATPClient)

**Files:**
- Create: `packages/atp-sdk/atp_sdk/sync.py`
- Create: `tests/unit/sdk/test_sync_client.py`

- [ ] **Step 1: Write tests for sync ATPClient**

Create `tests/unit/sdk/test_sync_client.py`:

```python
"""Tests for sync ATPClient wrapper."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from atp_sdk.sync import ATPClient


class TestSyncATPClient:
    """Tests for the sync wrapper over AsyncATPClient."""

    def test_context_manager(self) -> None:
        """Test sync context manager."""
        with ATPClient(
            platform_url="http://test:8000"
        ) as client:
            assert client._async_client is not None
        # Loop should be stopped
        assert not client._loop.is_running()

    def test_list_benchmarks(self) -> None:
        """Test sync list_benchmarks."""
        mock_response = httpx.Response(
            200,
            json=[
                {
                    "id": 1,
                    "name": "bench-1",
                    "description": "desc",
                    "tasks_count": 5,
                }
            ],
        )

        with ATPClient(
            platform_url="http://test:8000"
        ) as client:
            with patch.object(
                client._async_client,
                "_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                result = client.list_benchmarks()
                assert len(result) == 1
                assert result[0].name == "bench-1"

    def test_start_run(self) -> None:
        """Test sync start_run."""
        mock_response = httpx.Response(
            200,
            json={
                "id": 42,
                "benchmark_id": 1,
                "agent_name": "test",
                "status": "in_progress",
                "adapter_type": "sdk",
                "current_task_index": 0,
            },
        )

        with ATPClient(
            platform_url="http://test:8000"
        ) as client:
            with patch.object(
                client._async_client,
                "_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                run = client.start_run(
                    benchmark_id=1, agent_name="test"
                )
                assert run.run_id == 42

    def test_explicit_close(self) -> None:
        """Test explicit close without context manager."""
        client = ATPClient(platform_url="http://test:8000")
        client.close()
        assert not client._loop.is_running()

    def test_retry_params_forwarded(self) -> None:
        """Test that retry params are forwarded to async client."""
        with ATPClient(
            platform_url="http://test:8000",
            max_retries=5,
            retry_backoff=2.0,
        ) as client:
            cfg = client._async_client._retry_config
            assert cfg.max_retries == 5
            assert cfg.retry_backoff == 2.0

    def test_thread_safety(self) -> None:
        """Test that multiple threads can use client safely."""
        import concurrent.futures

        results = []

        mock_response = httpx.Response(200, json=[])

        with ATPClient(
            platform_url="http://test:8000"
        ) as client:
            with patch.object(
                client._async_client,
                "_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):

                def call_list() -> list:
                    return client.list_benchmarks()

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=4
                ) as pool:
                    futures = [
                        pool.submit(call_list) for _ in range(8)
                    ]
                    for f in concurrent.futures.as_completed(futures):
                        results.append(f.result())

        assert len(results) == 8
```

- [ ] **Step 2: Run tests to see them fail**

Run: `uv run python -m pytest tests/unit/sdk/test_sync_client.py -v`
Expected: FAIL

- [ ] **Step 3: Implement sync wrapper**

Create `packages/atp-sdk/atp_sdk/sync.py`:

```python
"""Synchronous ATP client wrapper.

ATPClient wraps AsyncATPClient using a dedicated background thread
with its own event loop. This avoids RuntimeError when used inside
an already-running event loop (Jupyter, FastAPI tests, etc.).

Thread-safe: a single ATPClient instance can be shared across
multiple threads.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from types import TracebackType
from typing import Any

from atp_sdk.benchmark import BenchmarkRun
from atp_sdk.client import AsyncATPClient
from atp_sdk.models import BenchmarkInfo

logger = logging.getLogger("atp_sdk")


class ATPClient:
    """Synchronous client for the ATP benchmark platform API.

    Wraps AsyncATPClient via a background thread + event loop.
    All constructor arguments are forwarded to AsyncATPClient.
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
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="atp-sdk-sync",
        )
        self._thread.start()

        self._async_client = AsyncATPClient(
            platform_url=platform_url,
            token=token,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            max_retry_delay=max_retry_delay,
            retry_on_timeout=retry_on_timeout,
            timeout=timeout,
        )
        logger.debug("ATPClient (sync) started background loop")

    def _run(self, coro: Any) -> Any:
        """Run a coroutine on the background loop and wait."""
        future = asyncio.run_coroutine_threadsafe(
            coro, self._loop
        )
        return future.result()

    @property
    def token(self) -> str | None:
        """Current auth token."""
        return self._async_client.token

    @property
    def platform_url(self) -> str:
        """Platform base URL."""
        return self._async_client.platform_url

    def login(self, open_browser: bool = True) -> str:
        """Perform Device Flow login.

        Note: This is inherently sync (uses webbrowser + polling)
        so it calls the sync auth module directly.
        """
        from atp_sdk.auth import login

        self._async_client.token = login(
            platform_url=self.platform_url,
            open_browser=open_browser,
        )
        # Update the async client's auth header
        self._async_client._http.headers[
            "Authorization"
        ] = f"Bearer {self._async_client.token}"
        return self._async_client.token

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        """List all available benchmarks."""
        return self._run(
            self._async_client.list_benchmarks()
        )

    def get_benchmark(
        self, benchmark_id: str | int
    ) -> BenchmarkInfo:
        """Get details of a specific benchmark."""
        return self._run(
            self._async_client.get_benchmark(benchmark_id)
        )

    def start_run(
        self,
        benchmark_id: str | int,
        agent_name: str = "",
        timeout: int = 3600,
        batch_size: int = 1,
    ) -> BenchmarkRun:
        """Start a new benchmark run."""
        return self._run(
            self._async_client.start_run(
                benchmark_id=benchmark_id,
                agent_name=agent_name,
                timeout=timeout,
                batch_size=batch_size,
            )
        )

    def get_leaderboard(
        self, benchmark_id: str | int
    ) -> list[dict[str, Any]]:
        """Get the leaderboard for a benchmark."""
        return self._run(
            self._async_client.get_leaderboard(benchmark_id)
        )

    def close(self) -> None:
        """Close the client and stop the background loop."""
        self._run(self._async_client.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        logger.debug("ATPClient (sync) closed")

    def __enter__(self) -> ATPClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/sdk/test_sync_client.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff + pyrefly**

Run: `uv run ruff format packages/atp-sdk/ tests/unit/sdk/ && uv run ruff check packages/atp-sdk/ tests/unit/sdk/ --fix && uv run pyrefly check`

- [ ] **Step 6: Commit**

```bash
git add packages/atp-sdk/atp_sdk/sync.py tests/unit/sdk/test_sync_client.py
git commit -m "feat(sdk): add sync ATPClient wrapper via background thread"
```

---

### Task 4: Rewrite BenchmarkRun with batch support + async iteration

**Files:**
- Rewrite: `packages/atp-sdk/atp_sdk/benchmark.py`
- Create: `tests/unit/sdk/test_benchmark_batch.py`

- [ ] **Step 1: Write tests for batch BenchmarkRun**

Create `tests/unit/sdk/test_benchmark_batch.py`:

```python
"""Tests for BenchmarkRun with batch support and async iteration."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from atp_sdk.benchmark import BenchmarkRun


def _make_run(
    batch_size: int = 1,
) -> tuple[BenchmarkRun, AsyncMock]:
    """Create a BenchmarkRun with a mocked client."""
    mock_client = MagicMock()
    mock_client._request = AsyncMock()
    run = BenchmarkRun(
        client=mock_client,
        run_id=1,
        benchmark_id=1,
        batch_size=batch_size,
    )
    return run, mock_client._request


class TestBenchmarkRunAsync:
    """Tests for async iteration."""

    @pytest.mark.anyio
    async def test_aiter_single_task(self) -> None:
        """Test async iteration with single tasks."""
        run, mock_req = _make_run()
        task_resp = httpx.Response(
            200, json=[{"task_id": "t1", "task": {"description": "do"}, "metadata": {"task_index": 0}}]
        )
        empty_resp = httpx.Response(204)
        mock_req.side_effect = [task_resp, empty_resp]

        tasks = []
        async for task in run:
            tasks.append(task)

        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "t1"

    @pytest.mark.anyio
    async def test_aiter_batch(self) -> None:
        """Test async iteration with batch_size=3."""
        run, mock_req = _make_run(batch_size=3)
        batch_resp = httpx.Response(
            200,
            json=[
                {"task_id": "t1", "task": {"description": "a"}, "metadata": {"task_index": 0}},
                {"task_id": "t2", "task": {"description": "b"}, "metadata": {"task_index": 1}},
                {"task_id": "t3", "task": {"description": "c"}, "metadata": {"task_index": 2}},
            ],
        )
        empty_resp = httpx.Response(204)
        mock_req.side_effect = [batch_resp, empty_resp]

        tasks = []
        async for task in run:
            tasks.append(task)

        assert len(tasks) == 3

    @pytest.mark.anyio
    async def test_aiter_partial_batch(self) -> None:
        """Test batch that returns fewer than requested."""
        run, mock_req = _make_run(batch_size=5)
        partial_resp = httpx.Response(
            200,
            json=[
                {"task_id": "t1", "task": {"description": "a"}, "metadata": {"task_index": 0}},
                {"task_id": "t2", "task": {"description": "b"}, "metadata": {"task_index": 1}},
            ],
        )
        empty_resp = httpx.Response(204)
        mock_req.side_effect = [partial_resp, empty_resp]

        tasks = []
        async for task in run:
            tasks.append(task)

        assert len(tasks) == 2

    @pytest.mark.anyio
    async def test_aiter_empty(self) -> None:
        """Test async iteration with no tasks."""
        run, mock_req = _make_run()
        mock_req.return_value = httpx.Response(204)

        tasks = []
        async for task in run:
            tasks.append(task)

        assert len(tasks) == 0


class TestBenchmarkRunNextBatch:
    """Tests for next_batch() method."""

    @pytest.mark.anyio
    async def test_next_batch_returns_list(self) -> None:
        """next_batch returns a list, not an iterator."""
        run, mock_req = _make_run()
        batch_resp = httpx.Response(
            200,
            json=[
                {"task_id": "t1", "task": {"description": "a"}, "metadata": {"task_index": 0}},
                {"task_id": "t2", "task": {"description": "b"}, "metadata": {"task_index": 1}},
            ],
        )
        mock_req.return_value = batch_resp

        tasks = await run.next_batch(2)
        assert isinstance(tasks, list)
        assert len(tasks) == 2

    @pytest.mark.anyio
    async def test_next_batch_empty_on_204(self) -> None:
        """next_batch returns empty list on 204."""
        run, mock_req = _make_run()
        mock_req.return_value = httpx.Response(204)

        tasks = await run.next_batch(5)
        assert tasks == []

    @pytest.mark.anyio
    async def test_next_batch_partial(self) -> None:
        """next_batch returns fewer than requested."""
        run, mock_req = _make_run()
        partial_resp = httpx.Response(
            200,
            json=[{"task_id": "t1", "task": {"description": "a"}, "metadata": {"task_index": 0}}],
        )
        mock_req.return_value = partial_resp

        tasks = await run.next_batch(5)
        assert len(tasks) == 1


class TestBenchmarkRunSubmit:
    """Tests for submit and other methods."""

    @pytest.mark.anyio
    async def test_submit(self) -> None:
        """Test async submit."""
        run, mock_req = _make_run()
        mock_req.return_value = httpx.Response(
            200, json={"task_index": 0, "score": 100.0}
        )

        result = await run.submit(
            response={"status": "completed"},
            task_index=0,
        )
        assert result["score"] == 100.0

    @pytest.mark.anyio
    async def test_status(self) -> None:
        """Test async status."""
        run, mock_req = _make_run()
        mock_req.return_value = httpx.Response(
            200,
            json={"id": 1, "status": "in_progress"},
        )

        result = await run.status()
        assert result["status"] == "in_progress"

    @pytest.mark.anyio
    async def test_cancel(self) -> None:
        """Test async cancel."""
        run, mock_req = _make_run()
        mock_req.return_value = httpx.Response(
            200, json={"status": "cancelled"}
        )

        await run.cancel()
        mock_req.assert_called_once()
```

- [ ] **Step 2: Run tests to see them fail**

Run: `uv run python -m pytest tests/unit/sdk/test_benchmark_batch.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite benchmark.py with batch support + async**

Rewrite `packages/atp-sdk/atp_sdk/benchmark.py`:

```python
"""BenchmarkRun iterator for pulling tasks from the ATP platform.

Supports both sync iteration (__iter__) and async iteration (__aiter__),
with optional batch prefetching via ?batch=N.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atp_sdk.client import AsyncATPClient

logger = logging.getLogger("atp_sdk")


class BenchmarkRun:
    """Iterator that pulls tasks from the platform API.

    Supports batch prefetching and both sync/async iteration.

    Args:
        client: AsyncATPClient instance.
        run_id: Run ID from the platform.
        benchmark_id: Benchmark ID.
        batch_size: Number of tasks to fetch per request.
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
        self._batch_size = batch_size
        self._buffer: deque[dict[str, Any]] = deque()
        self._exhausted = False

    async def _fetch_batch(
        self, batch: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch a batch of tasks from the server.

        Returns:
            List of task dicts. Empty list if run is exhausted.
        """
        n = batch or self._batch_size
        params: dict[str, int] = {}
        if n > 1:
            params["batch"] = n
        resp = await self._client._request(
            "GET",
            f"/api/v1/runs/{self.run_id}/next-task",
            params=params,
        )
        if resp.status_code == 204:
            self._exhausted = True
            return []
        resp.raise_for_status()
        data = resp.json()
        # Server returns a list for batch, single dict for batch=1
        if isinstance(data, dict):
            data = [data]
        logger.debug(
            "Fetched %d/%d tasks for run %d",
            len(data),
            n,
            self.run_id,
        )
        return data

    async def next_batch(self, n: int) -> list[dict[str, Any]]:
        """Fetch up to n tasks directly.

        Returns:
            List of 0..n task dicts. Empty list signals run exhaustion.
            Never raises StopIteration.
        """
        return await self._fetch_batch(n)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._buffer:
            return self._buffer.popleft()
        if self._exhausted:
            raise StopAsyncIteration
        tasks = await self._fetch_batch()
        if not tasks:
            raise StopAsyncIteration
        self._buffer.extend(tasks)
        return self._buffer.popleft()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Sync iteration — requires a running event loop via ATPClient."""
        import asyncio

        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot use sync iteration inside an async context. "
                "Use 'async for task in run:' instead."
            )

        async def _collect() -> list[dict[str, Any]]:
            return [task async for task in self]

        tasks = asyncio.run(_collect())
        yield from tasks

    async def submit(
        self,
        response: dict[str, Any],
        task_index: int,
        events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Submit a task response and optional events."""
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
            "GET", f"/api/v1/runs/{self.run_id}/status"
        )
        resp.raise_for_status()
        return resp.json()

    async def cancel(self) -> None:
        """Cancel the benchmark run."""
        resp = await self._client._request(
            "POST", f"/api/v1/runs/{self.run_id}/cancel"
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/sdk/test_benchmark_batch.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff + pyrefly**

Run: `uv run ruff format packages/atp-sdk/ tests/unit/sdk/ && uv run ruff check packages/atp-sdk/ tests/unit/sdk/ --fix && uv run pyrefly check`

- [ ] **Step 6: Commit**

```bash
git add packages/atp-sdk/atp_sdk/benchmark.py tests/unit/sdk/test_benchmark_batch.py
git commit -m "feat(sdk): rewrite BenchmarkRun with batch prefetch and async iteration"
```

---

### Task 5: Add `?batch=N` to server next-task endpoint

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py:170-244`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/config.py:69-77`
- Modify: `tests/unit/benchmark/test_benchmark_api.py`

- [ ] **Step 1: Add `batch_max_size` to DashboardConfig**

In `packages/atp-dashboard/atp/dashboard/v2/config.py`, add after `github_client_secret`:

```python
    # Batch settings
    batch_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum batch size for next-task endpoint",
    )
```

And in `to_dict()`:
```python
            "batch_max_size": self.batch_max_size,
```

- [ ] **Step 2: Write tests for batch next-task**

Add to `tests/unit/benchmark/test_benchmark_api.py`:

```python
class TestNextTaskBatch:
    """Tests for ?batch=N on next-task endpoint."""

    @pytest.mark.anyio
    async def test_batch_returns_multiple_tasks(
        self, client: AsyncClient, benchmark_id: int, run_id: int
    ) -> None:
        """batch=2 should return 2 tasks."""
        resp = await client.get(
            f"/api/v1/runs/{run_id}/next-task?batch=2"
        )
        assert resp.status_code == 200
        tasks = resp.json()
        assert isinstance(tasks, list)
        assert len(tasks) == 2

    @pytest.mark.anyio
    async def test_batch_default_returns_single(
        self, client: AsyncClient, benchmark_id: int, run_id: int
    ) -> None:
        """No batch param should return single task in array."""
        resp = await client.get(
            f"/api/v1/runs/{run_id}/next-task"
        )
        assert resp.status_code == 200
        data = resp.json()
        # Single task (backward compat — still dict for batch=1)
        assert isinstance(data, dict)

    @pytest.mark.anyio
    async def test_batch_partial_at_end(
        self, client: AsyncClient, benchmark_id: int, run_id: int
    ) -> None:
        """batch=10 with only 2 tasks left returns 2."""
        # Consume first task
        await client.get(f"/api/v1/runs/{run_id}/next-task")
        # Now only 1 task left, request batch=10
        resp = await client.get(
            f"/api/v1/runs/{run_id}/next-task?batch=10"
        )
        assert resp.status_code == 200
        tasks = resp.json()
        assert isinstance(tasks, list)
        assert len(tasks) == 1

    @pytest.mark.anyio
    async def test_batch_204_when_exhausted(
        self, client: AsyncClient, benchmark_id: int, run_id: int
    ) -> None:
        """batch=N returns 204 when no tasks left."""
        # Consume all 2 tasks
        await client.get(f"/api/v1/runs/{run_id}/next-task?batch=2")
        # Now empty
        resp = await client.get(
            f"/api/v1/runs/{run_id}/next-task?batch=5"
        )
        assert resp.status_code == 204
```

- [ ] **Step 3: Modify next_task endpoint to support batch**

Replace the `next_task` function in `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py`:

```python
@router.get("/runs/{run_id}/next-task")
async def next_task(
    run_id: int,
    session: DBSession,
    batch: int = Query(default=1, ge=1),
) -> Any:
    """Get the next task(s) for a run as ATPRequest dict(s).

    Args:
        batch: Number of tasks to fetch (default 1, max from config).

    Returns 204 No Content when all tasks have been consumed.
    When batch=1, returns a single dict (backward compat).
    When batch>1, returns a JSON array.
    """
    from atp.dashboard.v2.config import get_config

    config = get_config()
    batch = min(batch, config.batch_max_size)

    # Atomic increment by batch size
    stmt = (
        update(Run)
        .where(Run.id == run_id, Run.status == RunStatus.IN_PROGRESS)
        .values(current_task_index=Run.current_task_index + batch)
        .returning(Run.current_task_index, Run.benchmark_id)
    )
    result = await session.execute(stmt)
    row = result.one_or_none()
    if row is None:
        run = await session.get(Run, run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run {run_id} not found",
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    new_index, benchmark_id = row
    start_idx = new_index - batch

    bm = await session.get(Benchmark, benchmark_id)
    if bm is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark not found",
        )

    suite = TestSuite.model_validate(bm.suite)

    tasks: list[dict[str, Any]] = []
    for idx in range(start_idx, new_index):
        if idx >= len(suite.tests):
            break
        test_def = suite.tests[idx]

        constraints: dict[str, Any] = {}
        if test_def.constraints.max_steps is not None:
            constraints["max_steps"] = test_def.constraints.max_steps
        if test_def.constraints.max_tokens is not None:
            constraints["max_tokens"] = test_def.constraints.max_tokens
        if test_def.constraints.timeout_seconds is not None:
            constraints["timeout_seconds"] = test_def.constraints.timeout_seconds
        if test_def.constraints.allowed_tools is not None:
            constraints["allowed_tools"] = test_def.constraints.allowed_tools
        if test_def.constraints.budget_usd is not None:
            constraints["budget_usd"] = test_def.constraints.budget_usd

        request = ATPRequest(
            task_id=str(uuid.uuid4()),
            task=Task(
                description=test_def.task.description,
                input_data=test_def.task.input_data,
                expected_artifacts=test_def.task.expected_artifacts,
            ),
            constraints=constraints,
            metadata={
                "test_id": test_def.id,
                "test_name": test_def.name,
                "task_index": idx,
                "run_id": run_id,
            },
        )
        tasks.append(request.model_dump())

    await session.flush()

    if not tasks:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # Backward compat: batch=1 returns single dict
    if batch == 1:
        return tasks[0]
    return tasks
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/benchmark/test_benchmark_api.py -v`
Expected: All PASS (existing + new)

- [ ] **Step 5: Run ruff + pyrefly**

Run: `uv run ruff format packages/atp-dashboard/ tests/unit/benchmark/ && uv run ruff check packages/atp-dashboard/ tests/unit/benchmark/ --fix && uv run pyrefly check`

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py packages/atp-dashboard/atp/dashboard/v2/config.py tests/unit/benchmark/test_benchmark_api.py
git commit -m "feat(api): add ?batch=N support to next-task endpoint"
```

---

### Task 6: Update __init__.py, pyproject.toml, and run full test suite

**Files:**
- Modify: `packages/atp-sdk/atp_sdk/__init__.py`
- Modify: `packages/atp-sdk/pyproject.toml`

- [ ] **Step 1: Update __init__.py**

Rewrite `packages/atp-sdk/atp_sdk/__init__.py`:

```python
from atp_sdk.auth import load_token, login, save_token
from atp_sdk.benchmark import BenchmarkRun
from atp_sdk.client import AsyncATPClient
from atp_sdk.models import (
    BenchmarkInfo,
    LeaderboardEntry,
    RunInfo,
    RunStatus,
)
from atp_sdk.sync import ATPClient

__all__ = [
    "ATPClient",
    "AsyncATPClient",
    "BenchmarkInfo",
    "BenchmarkRun",
    "LeaderboardEntry",
    "RunInfo",
    "RunStatus",
    "load_token",
    "login",
    "save_token",
]
```

- [ ] **Step 2: Bump version to 2.0.0**

In `packages/atp-sdk/pyproject.toml`, change:
```toml
version = "2.0.0"
```

Also add `pytest-mock` to dev deps if not present:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-anyio",
]
```

- [ ] **Step 3: Run the full SDK test suite**

Run: `uv run python -m pytest tests/unit/sdk/ -v`
Expected: All PASS

- [ ] **Step 4: Run the import test**

Run: `uv run python -c "from atp_sdk import ATPClient, AsyncATPClient; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Run ruff + pyrefly on entire project**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add packages/atp-sdk/atp_sdk/__init__.py packages/atp-sdk/pyproject.toml
git commit -m "feat(sdk): bump to v2.0.0, export AsyncATPClient"
```

---

### Task 7: Integration verification

**Files:** All files from Tasks 1-6

- [ ] **Step 1: Run the full project test suite**

Run: `uv run python -m pytest tests/ -v -x -q`
Expected: No regressions

- [ ] **Step 2: Verify SDK build**

Run: `cd packages/atp-sdk && uv build --out-dir dist`
Expected: `atp_platform_sdk-2.0.0` wheel built

- [ ] **Step 3: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`
Expected: Clean

- [ ] **Step 4: Commit any formatting changes**

```bash
git add -u
git diff --cached --stat
# Only commit if there are changes
git commit -m "style: format and lint fixes for SDK v2.0.0"
```
