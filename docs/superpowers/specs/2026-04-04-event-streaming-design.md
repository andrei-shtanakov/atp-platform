# Event Streaming Design

## Summary

Allow SDK participants to emit events during benchmark run execution (before submit). Events stored in Run.events JSON column with a cap of 1000 per run.

## Design

### SDK: `BenchmarkRun.emit()` / `emit_sync()`

```python
async def emit(self, events: list[dict[str, Any]]) -> None:
    """Send events to the server during run execution."""
    await self._client._request(
        "POST", f"/api/v1/runs/{self.run_id}/events",
        json={"events": events},
    )
```

Plus `emit_sync()` wrapper.

### Server: `POST /api/v1/runs/{run_id}/events`

- Accepts `{"events": [{"event_type": "...", "data": {...}, "timestamp": "..."}]}`
- Appends to `Run.events` JSON column
- Validates: run must be IN_PROGRESS, max 1000 total events (422 if exceeded)
- Rate limited: 120/min (same as benchmark API)

### Model: `Run.events`

New column: `events: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True, default=list)`

Max 1000 events per run. Server rejects with 422 when limit reached.

### Files

| File | Action |
|------|--------|
| `packages/atp-dashboard/atp/dashboard/benchmark/models.py` | Add `events` column to Run |
| `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py` | New POST /runs/{id}/events endpoint |
| `packages/atp-sdk/atp_sdk/benchmark.py` | Add `emit()` + `emit_sync()` |
| `tests/unit/dashboard/test_benchmark_events.py` | Server tests |
