# SDK Quick Wins: Batch API, Async Client, Retry

**Date:** 2026-04-03
**Status:** Approved

## Overview

Three improvements to `atp-platform-sdk` (packages/atp-sdk/):
1. Batch task fetching (`?batch=N`)
2. Async-first client with sync wrapper
3. Retry with exponential backoff

## 1. Batch API

### Server (benchmark_api.py)

`GET /api/v1/runs/{run_id}/next-task` gains optional query param `?batch=N`:
- Default: 1, max: 10
- Atomically increments `current_task_index` by N
- Returns JSON array of tasks
- If fewer than N tasks remain, returns what's available
- If 0 tasks remain, returns 204 No Content

### SDK (benchmark.py)

- `BenchmarkRun` accepts `batch_size` parameter (set at run creation)
- `__iter__()` and `__aiter__()` use internal prefetch buffer: fetch batch, yield one at a time
- `next_batch(n)` method returns list of tasks directly for parallel processing

## 2. Async-first Client

### Architecture

Rewrite `ATPClient` as async-first using `httpx.AsyncClient`. Provide thin sync wrapper.

### Files

| File | Content |
|------|---------|
| `client.py` | `ATPClient` — async-first, all methods `async def` |
| `sync.py` | `SyncATPClient` — sync wrapper using `asyncio.run()` |
| `benchmark.py` | `BenchmarkRun` supports both `__iter__` and `__aiter__` |

### Backward Compatibility

- `from atp_sdk import ATPClient` continues to work (now async)
- `from atp_sdk import SyncATPClient` for sync usage
- Re-export both in `__init__.py`

## 3. Retry with Exponential Backoff

### Implementation

Internal `_request()` method in `ATPClient` wraps all HTTP calls.

**Retry triggers:**
- `httpx.TransportError` (network errors)
- HTTP 502, 503, 504
- HTTP 429 (rate limit) — respects `Retry-After` header

**Not retried:**
- Other 4xx responses
- Task-level timeouts (business logic)

**Configuration:**
```python
ATPClient(
    max_retries=3,        # default 3
    retry_backoff=1.0,    # base delay in seconds
    retry=True,           # set False to disable
)
```

**Backoff schedule:** 1s → 2s → 4s with ±20% jitter.

## Scope

- No new dependencies (httpx already supports async)
- asyncio.run() for sync wrapper (no anyio dependency in SDK)
- Tests use pytest + anyio (dev dependency only)
