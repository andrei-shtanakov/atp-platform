# SDK Quick Wins: Batch API, Async Client, Retry

**Date:** 2026-04-03
**Status:** Approved (rev.3 — final)

## Overview

Three improvements to `atp-platform-sdk` (packages/atp-sdk/):
1. Batch task fetching (`?batch=N`)
2. Async-first client with sync wrapper
3. Retry with exponential backoff

## 1. Batch API

### Server (benchmark_api.py)

`GET /api/v1/runs/{run_id}/next-task` gains optional query param `?batch=N`:
- Default: 1
- Max: configurable via `ATP_BATCH_MAX_SIZE` env var (default: 10)
- Atomically increments `current_task_index` by N
- Returns JSON array of tasks
- If fewer than N tasks remain, returns what's available
- If 0 tasks remain, returns 204 No Content

### SDK (benchmark.py)

- `BenchmarkRun` accepts `batch_size` parameter (set at run creation)
- `__iter__()` and `__aiter__()` use internal prefetch buffer: fetch batch, yield one at a time
- `next_batch(n)` returns list of tasks directly for parallel processing:
  - Returns `list[Task]` with 0..n items. Empty list signals run exhaustion.
  - Never raises `StopIteration` — caller checks `len()`.
- **204 handling:** `__iter__`/`__aiter__` raise `StopIteration`/`StopAsyncIteration` on 204 — clean loop exit
- **Lost tasks:** No ack mechanism. If client crashes after fetching a batch, those tasks are marked as issued but unsubmitted. The existing server-side timeout (status=partial) covers this — tasks left unsubmitted after run timeout are reported as partial. This is intentional: ack adds complexity without proportional benefit at current scale.

## 2. Async-first Client

### Architecture

Rewrite `ATPClient` as async-first using `httpx.AsyncClient`. Provide sync wrapper using a dedicated background thread with its own event loop.

### Files

| File | Content |
|------|---------|
| `client.py` | `AsyncATPClient` — async-first, all methods `async def` |
| `sync.py` | `ATPClient` — sync wrapper via background thread + event loop |
| `benchmark.py` | `BenchmarkRun` supports both `__iter__` and `__aiter__` |

### Sync Wrapper Strategy

`asyncio.run()` fails inside an already-running event loop (Jupyter, FastAPI tests, some CI). Instead:

- `ATPClient` (sync) spawns a daemon thread with a dedicated `asyncio.EventLoop`
- Methods dispatch coroutines to the background loop via `asyncio.run_coroutine_threadsafe().result()`
- Thread and loop are cleaned up on `client.close()` / `__exit__`
- No runtime dependency on anyio or nest_asyncio
- **Thread safety:** `ATPClient` is thread-safe. `asyncio.run_coroutine_threadsafe()` is inherently thread-safe, so a single client instance can be shared across multiple threads.

### Context Managers

Both clients support context manager protocol:

```python
# Sync
with ATPClient(server_url="...") as client:
    ...

# Async
async with AsyncATPClient(server_url="...") as client:
    ...
```

`AsyncATPClient` implements `__aenter__`/`__aexit__` to properly close `httpx.AsyncClient` and avoid resource warnings.

### Backward Compatibility — Version 2.0.0

This is a **major version bump** reflecting internal restructuring:

- `ATPClient` remains the sync entrypoint (same name, same sync API) — **no breaking change for sync users**
- `AsyncATPClient` is the new async class
- `from atp_sdk import ATPClient` — sync, works as before
- `from atp_sdk import AsyncATPClient` — async, new

## 3. Retry with Exponential Backoff

### Implementation

Internal `_request()` method in `AsyncATPClient` wraps all HTTP calls.

**Retry triggers:**
- `httpx.TransportError` (network errors)
- HTTP 502, 503, 504 (server errors)
- HTTP 429 (rate limit) — respects `Retry-After` header

**Not retried:**
- Other 4xx responses
- Task-level timeouts (business logic)

**Configuration:**
```python
AsyncATPClient(
    max_retries=3,            # default 3, set 0 to disable
    retry_backoff=1.0,        # base delay in seconds
    max_retry_delay=30.0,     # cap on backoff delay
    retry_on_timeout=True,    # retry on timeout errors (default True)
    timeout=30.0,             # httpx timeout for all requests
)
```

`retry_on_timeout=False` disables retry on `httpx.TimeoutException` specifically — useful when slow responses indicate a real problem rather than a transient failure.

**Backoff:** Full jitter — `delay = random(0, min(max_retry_delay, retry_backoff * 2^attempt))`. Full jitter provides better distribution under concurrent clients than equal jitter (see AWS architecture blog).

### Timeouts

`httpx.AsyncClient` is created with explicit `httpx.Timeout(timeout)`:
- Applies to connect, read, write, pool
- Default: 30s
- Configurable per-client

## 4. Observability

Standard library logging via `logging.getLogger("atp_sdk")`. No external dependencies.

| Event | Level | Example |
|-------|-------|---------|
| Retry attempt | WARNING | `Retry 2/3 for GET /runs/42/next-task (502), delay=1.8s` |
| Batch fetch | DEBUG | `Fetched 3/5 tasks for run 42` |
| Client open/close | DEBUG | `AsyncATPClient connected to https://atp.example.com` |
| Auth token refresh | DEBUG | `Token loaded from ~/.atp/config.json` |
| Request completed | DEBUG | `GET /benchmarks 200 in 0.15s` |

Users configure logging as usual: `logging.basicConfig(level=logging.DEBUG)` to see SDK internals.

## Migration: 1.x → 2.0

```python
# Sync users: no changes needed
client = ATPClient(server_url="https://atp.example.com")
benchmarks = client.list_benchmarks()  # works as before

# Async users: use AsyncATPClient
async with AsyncATPClient(server_url="https://atp.example.com") as client:
    benchmarks = await client.list_benchmarks()

# Retry: was retry=True/False, now max_retries=N
# Old: ATPClient(retry=False)
# New: ATPClient(max_retries=0)
```

## Scope

- No new runtime dependencies
- Sync wrapper uses stdlib `threading` + `asyncio`
- Tests use pytest + anyio (dev dependency only)
- Version bump: 1.0.0 → 2.0.0
