# Webhooks Design

## Summary

Webhook notifications when benchmark runs complete or fail. URL configured per-benchmark, delivered with 3 retries and SSRF protection.

## Context

- Benchmark API at `/api/v1/benchmarks` and `/api/v1/runs`
- Run completion logic in `benchmark_api.py` — updates DB status, no events
- `WebhookAlertChannel` pattern exists in `budgets.py` (reference, not reused)
- Production behind nginx, single worker

## Design

### Data Model

Add to `Benchmark` SQLAlchemy model (`benchmark/models.py`):

```python
webhook_url: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
```

Accept in `POST /v1/benchmarks` create request body. Validate with SSRF deny-list.

### SSRF Protection

Pydantic validator on webhook_url rejects:
- Private IP ranges: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
- Link-local: `169.254.0.0/16`
- Loopback: `127.0.0.0/8`, `[::1]`
- Non-HTTP schemes (only `https://` and `http://` allowed)
- Hostnames resolving to private IPs (resolve before accepting)

Implemented as a standalone function `validate_webhook_url(url)` that raises `ValueError` on blocked URLs. Called both at benchmark creation and before delivery.

### Webhook Payload

```json
{
  "event": "run.completed",
  "timestamp": "2026-04-04T12:00:00Z",
  "delivery_id": "uuid-here",
  "benchmark": {
    "id": "bench-123",
    "name": "Code Quality v2"
  },
  "run": {
    "id": "run-456",
    "status": "completed",
    "total_score": 87.5,
    "tasks_completed": 10,
    "tasks_total": 10,
    "started_at": "2026-04-04T11:55:00Z",
    "finished_at": "2026-04-04T12:00:00Z"
  }
}
```

Headers: `Content-Type: application/json`, `X-ATP-Event: run.completed`, `X-ATP-Delivery: <uuid>`

### Delivery

Module: `packages/atp-dashboard/atp/dashboard/webhook.py`

- `async def deliver_webhook(url: str, payload: dict) -> bool`
- 3 attempts with backoff: 1s, 5s, 15s
- Timeout: 10s per attempt
- Logs success/failure per attempt
- Re-validates URL before delivery (SSRF check)
- Returns True if any attempt succeeds

### Background Task Safety

Use a module-level `set[asyncio.Task]` to prevent GC of in-flight tasks:

```python
_background_tasks: set[asyncio.Task] = set()

def schedule_webhook(url: str, payload: dict) -> None:
    task = asyncio.create_task(deliver_webhook(url, payload))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
```

This ensures tasks survive until completion even during shutdown.

### Hook Point

In `benchmark_api.py`, after run finalization (both completed and failed):

```python
if bm.webhook_url:
    event = "run.completed" if run.status == RunStatus.COMPLETED else "run.failed"
    payload = build_webhook_payload(event, bm, run)
    schedule_webhook(bm.webhook_url, payload)
```

### Testing

- **Unit**: SSRF validation (block private IPs, allow public), payload format, retry backoff with mock httpx
- **Integration**: webhook fires on run completion via TestClient
- **Edge cases**: webhook_url=None (no-op), unreachable URL (3 retries then drop), SSRF blocked URL

### Files

| File | Action |
|------|--------|
| `packages/atp-dashboard/atp/dashboard/benchmark/models.py` | Add `webhook_url` column |
| `packages/atp-dashboard/atp/dashboard/webhook.py` | New: SSRF validation, delivery with retry, schedule_webhook |
| `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py` | Accept webhook_url, trigger on completion |
| `tests/unit/dashboard/test_webhook.py` | New: SSRF, delivery, retry, payload tests |
