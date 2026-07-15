# ADR-ECO-003e M0 — UsageCapture seam + Action №0 exposure probe

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the minimal observe-only `UsageCapture` slice of ADR-ECO-003e (the
single mandatory usage-capture seam at the runner⟷adapter boundary) plus the probe
tooling needed to run Action №0 (runtime acceptance gate: prove the old cost wiring
is unreached, measure which adapter paths actually carry token usage).

**Architecture:** A new dependency-light contract module
`packages/atp-core/atp/cost/capture.py` defines `UsageRecord` / `UsageCapture` /
`JsonlUsageCapture` (reusing 003d's `PerClassUsage`). The orchestrator records one
`UsageRecord` per adapter call at the single point where every execution path
(success / timeout / failure, with or without event streaming) converges
(`atp/runner/orchestrator.py::_execute_run`). A probe-report tool aggregates the
JSONL into the Action №0 evidence table. `track_response_cost` gets a
DeprecationWarning (removal in M1).

**Tech Stack:** Python 3.12, pydantic (existing models untouched), stdlib
dataclasses/json, pytest + anyio, ruff, pyrefly.

## Global Constraints

- Package management: ONLY `uv` (`uv run pytest`, `uv run ruff`, `uv run pyrefly check`) — never pip.
- Type hints required everywhere; run `uv run pyrefly check` after every change and fix errors.
- Line length: 88 chars. Format with `uv run ruff format .`; lint `uv run ruff check .`.
- Async tests use `anyio`, not `asyncio` (M0 tests are sync — capture is sync by design).
- Git: work on branch `feat/003e-m0-usage-capture-probe`; direct commits to `main` are forbidden; finish via `gh pr create`.
- `capture.py` must stay dependency-light (stdlib + `PerClassUsage` import only): downstream repos (Maestro / spec-runner / robin-runtime) will vendor a pinned copy per 003e D2.
- Capture must NEVER break a test run: all sink IO errors are caught and logged, never raised.
- M0 is observe-only: no budget enforcement, no model plumbing into adapters (that is M1/M3 — see Roadmap at the end).

## Design decisions locked in this plan (with reasons)

1. **Seam location = orchestrator `_execute_run`, not `AgentAdapter`.** The ADR says
   "adapter boundary"; in this codebase the event-streaming path
   (`adapter.stream_events`) bypasses `execute()` entirely, and
   `execute_with_tracing` has zero callers. The one place every runtime path
   converges with a finalized `ATPResponse` is `_execute_run` (after its
   try/except/finally, `orchestrator.py:678`). One call site = the mandatory seam.
   Re-homing into a base-adapter template method can be revisited in M1 if
   non-orchestrator callers appear.
2. **`model`/`provider` are recorded as `None` in M0.** No adapter currently knows
   its model at the seam (CLI agents self-report metrics without a model id). An
   honest `None` (not the string `"unknown"`) is itself the Action №0 evidence;
   per-adapter model plumbing is M1, ordered by the probe's $-exposure result.
3. **Sink = JSONL file via `ATP_USAGE_CAPTURE_PATH` env var; `NullUsageCapture`
   when unset.** The seam is always called; only the sink varies. No new config
   surface in `ATPSettings` for M0 (ATPSettings has cwd-walking gotchas; env var
   is enough for a probe).
4. **`PerClassUsage` is imported from `atp.cost.cloud_pricer`.** It is a plain
   frozen dataclass; litellm is imported lazily inside functions, so this import
   is safe without the `[pricing]` extra.
5. **`track_response_cost` is deprecated in M0, removed in M1.** It is exported
   API surface (`tests/unit/test_api_surface.py:47`); M0 stays additive.

---

### Task 1: Capture contract module (`atp/cost/capture.py`)

**Files:**
- Create: `packages/atp-core/atp/cost/capture.py`
- Modify: `packages/atp-core/atp/cost/__init__.py` (add exports)
- Test: `tests/unit/cost/test_capture.py`

**Interfaces:**
- Consumes: `PerClassUsage` from `atp.cost.cloud_pricer` (frozen dataclass:
  `input_tokens: int, output_tokens: int, cache_creation_tokens: int,
  cache_read_tokens: int, usage_source: str | None`).
- Produces (used by Tasks 2 and 4):
  - `UsageRecord` frozen dataclass (fields below),
  - `UsageCapture` Protocol with `record_usage(record: UsageRecord) -> None`,
  - `NullUsageCapture`, `JsonlUsageCapture(path: Path)`,
  - `usage_from_metrics(metrics) -> PerClassUsage | None`,
  - `capture_from_env() -> UsageCapture`,
  - constant `CAPTURE_PATH_ENV = "ATP_USAGE_CAPTURE_PATH"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cost/test_capture.py`:

```python
"""Tests for the UsageCapture seam (ADR-ECO-003e M0)."""

import json
from pathlib import Path

from atp.cost.capture import (
    CAPTURE_PATH_ENV,
    JsonlUsageCapture,
    NullUsageCapture,
    UsageRecord,
    capture_from_env,
    usage_from_metrics,
)
from atp.cost.cloud_pricer import PerClassUsage
from atp.protocol import Metrics


def make_record(call_id: str = "call-1") -> UsageRecord:
    return UsageRecord(
        call_id=call_id,
        timestamp="2026-07-15T00:00:00+00:00",
        adapter_type="cli",
        status="completed",
        model=None,
        provider=None,
        usage=PerClassUsage(
            input_tokens=10,
            output_tokens=20,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            usage_source="measured",
        ),
        reported_cost_usd=None,
        test_id="test-1",
    )


class TestUsageFromMetrics:
    def test_none_metrics_gives_none(self) -> None:
        assert usage_from_metrics(None) is None

    def test_all_token_fields_absent_gives_none(self) -> None:
        assert usage_from_metrics(Metrics(wall_time_seconds=1.0)) is None

    def test_partial_fields_fill_zero(self) -> None:
        usage = usage_from_metrics(Metrics(input_tokens=5))
        assert usage == PerClassUsage(
            input_tokens=5,
            output_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            usage_source="measured",
        )

    def test_all_four_classes_pass_through(self) -> None:
        usage = usage_from_metrics(
            Metrics(
                input_tokens=1,
                output_tokens=2,
                cache_creation_tokens=3,
                cache_read_tokens=4,
            )
        )
        assert usage is not None
        assert (
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_creation_tokens,
            usage.cache_read_tokens,
        ) == (1, 2, 3, 4)


class TestJsonlUsageCapture:
    def test_appends_one_json_line_per_record(self, tmp_path: Path) -> None:
        sink = JsonlUsageCapture(tmp_path / "usage.jsonl")
        sink.record_usage(make_record("a"))
        sink.record_usage(make_record("b"))
        lines = (tmp_path / "usage.jsonl").read_text().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["call_id"] == "a"
        assert first["usage"]["input_tokens"] == 10

    def test_idempotent_on_call_id(self, tmp_path: Path) -> None:
        sink = JsonlUsageCapture(tmp_path / "usage.jsonl")
        sink.record_usage(make_record("a"))
        sink.record_usage(make_record("a"))
        lines = (tmp_path / "usage.jsonl").read_text().splitlines()
        assert len(lines) == 1

    def test_none_usage_serializes_as_null(self, tmp_path: Path) -> None:
        sink = JsonlUsageCapture(tmp_path / "usage.jsonl")
        record = UsageRecord(
            call_id="c",
            timestamp="2026-07-15T00:00:00+00:00",
            adapter_type="http",
            status="failed",
            model=None,
            provider=None,
            usage=None,
            reported_cost_usd=None,
            test_id=None,
        )
        sink.record_usage(record)
        row = json.loads((tmp_path / "usage.jsonl").read_text())
        assert row["usage"] is None

    def test_io_error_is_swallowed(self, tmp_path: Path) -> None:
        # Point the sink at a path whose parent is a *file* -> open() fails.
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        sink = JsonlUsageCapture(blocker / "usage.jsonl")
        sink.record_usage(make_record("a"))  # must not raise


class TestCaptureFromEnv:
    def test_unset_env_gives_null_capture(
        self, monkeypatch: object
    ) -> None:
        monkeypatch.delenv(CAPTURE_PATH_ENV, raising=False)  # type: ignore[attr-defined]
        assert isinstance(capture_from_env(), NullUsageCapture)

    def test_set_env_gives_jsonl_capture(
        self, monkeypatch: object, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(  # type: ignore[attr-defined]
            CAPTURE_PATH_ENV, str(tmp_path / "u.jsonl")
        )
        assert isinstance(capture_from_env(), JsonlUsageCapture)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/cost/test_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atp.cost.capture'`

- [ ] **Step 3: Implement the module**

Create `packages/atp-core/atp/cost/capture.py`:

```python
"""Runtime usage capture — the mandatory UsageCapture seam (ADR-ECO-003e D2).

M0 is the observe-only slice: it records per-call token usage (or its
absence) at the runner⟷adapter boundary. BudgetControl (reserve/settle,
003e D1/D3) builds on the same records later.

This module must stay dependency-light (stdlib + PerClassUsage): downstream
repos vendor a pinned copy per the 003e D2 vendoring rule.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from atp.cost.cloud_pricer import PerClassUsage

if TYPE_CHECKING:
    from atp.protocol import Metrics

logger = logging.getLogger(__name__)

CAPTURE_PATH_ENV = "ATP_USAGE_CAPTURE_PATH"


@dataclass(frozen=True)
class UsageRecord:
    """One captured adapter call, with or without token usage.

    ``usage is None`` means the adapter path produced no token accounting —
    that absence is itself the Action №0 signal, so it is recorded, not
    skipped. ``model``/``provider`` are None until adapters plumb real ids
    (003e adoption, M1).
    """

    call_id: str
    timestamp: str
    adapter_type: str
    status: str
    model: str | None
    provider: str | None
    usage: PerClassUsage | None
    reported_cost_usd: float | None
    test_id: str | None = None


class UsageCapture(Protocol):
    """The mandatory usage-capture seam (ADR-ECO-003e D2)."""

    def record_usage(self, record: UsageRecord) -> None:
        """Record one adapter call. Must never raise."""
        ...


class NullUsageCapture:
    """Sink used when no capture path is configured."""

    def record_usage(self, record: UsageRecord) -> None:
        """Discard the record."""


class JsonlUsageCapture:
    """Append-only JSONL sink, idempotent on call_id within the process."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._seen: set[str] = set()

    def record_usage(self, record: UsageRecord) -> None:
        """Append the record as one JSON line; swallow and log IO errors."""
        with self._lock:
            if record.call_id in self._seen:
                return
            self._seen.add(record.call_id)
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(record)) + "\n")
            except OSError as e:
                logger.warning("Usage capture write failed: %s", e)


def usage_from_metrics(metrics: Metrics | None) -> PerClassUsage | None:
    """Convert response Metrics into PerClassUsage.

    Returns None when no token class is reported at all (the "path does
    not capture usage" case); fills absent classes with 0 otherwise.
    """
    if metrics is None:
        return None
    fields = (
        metrics.input_tokens,
        metrics.output_tokens,
        metrics.cache_creation_tokens,
        metrics.cache_read_tokens,
    )
    if all(f is None for f in fields):
        return None
    return PerClassUsage(
        input_tokens=metrics.input_tokens or 0,
        output_tokens=metrics.output_tokens or 0,
        cache_creation_tokens=metrics.cache_creation_tokens or 0,
        cache_read_tokens=metrics.cache_read_tokens or 0,
        usage_source="measured",
    )


def capture_from_env() -> UsageCapture:
    """Build the process-wide capture sink from ATP_USAGE_CAPTURE_PATH."""
    path = os.environ.get(CAPTURE_PATH_ENV)
    if path:
        return JsonlUsageCapture(Path(path))
    return NullUsageCapture()
```

- [ ] **Step 4: Export from `atp.cost`**

In `packages/atp-core/atp/cost/__init__.py` add to the existing imports/`__all__`:

```python
from atp.cost.capture import (
    CAPTURE_PATH_ENV,
    JsonlUsageCapture,
    NullUsageCapture,
    UsageCapture,
    UsageRecord,
    capture_from_env,
    usage_from_metrics,
)
```

and append the same seven names to `__all__`.

- [ ] **Step 5: Run tests + quality gates**

Run: `uv run pytest tests/unit/cost/test_capture.py -v`
Expected: all PASS.
Run: `uv run ruff format . && uv run ruff check . && uv run pyrefly check`
Expected: clean (fix anything reported).

- [ ] **Step 6: Commit**

```bash
git add packages/atp-core/atp/cost/capture.py packages/atp-core/atp/cost/__init__.py tests/unit/cost/test_capture.py
git commit -m "feat(cost): UsageCapture contract + JSONL sink (ADR-ECO-003e M0)"
```

---

### Task 2: Wire the seam into the orchestrator

**Files:**
- Modify: `atp/runner/orchestrator.py` (imports ~line 13–43; `__init__` at :57–95; `_execute_run` — record after the `finally` block, near :678 `end_time = datetime.now(tz=UTC)`)
- Test: `tests/unit/runner/test_orchestrator_usage_capture.py`

**Interfaces:**
- Consumes from Task 1: `UsageCapture`, `UsageRecord`, `capture_from_env`, `usage_from_metrics`.
- Produces: `TestOrchestrator(usage_capture=...)` keyword arg (default `None` → `capture_from_env()`); every `_execute_run` invocation emits exactly one `UsageRecord` with `call_id=request.task_id`, `adapter_type=self.adapter.adapter_type`, `status=response.status.value`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/runner/test_orchestrator_usage_capture.py`.
Reuse the stub-adapter style of the existing orchestrator tests
(`tests/unit/runner/` has stub adapters returning canned `ATPResponse`; if the
exact fixture module differs, copy the minimal stub below as-is):

```python
"""Orchestrator emits one UsageRecord per adapter call (003e M0 seam)."""

from collections.abc import AsyncIterator

import pytest

from atp.adapters.base import AgentAdapter
from atp.cost.capture import UsageRecord
from atp.loader.models import TestDefinition
from atp.protocol import ATPEvent, ATPRequest, ATPResponse, Metrics, ResponseStatus
from atp.runner.orchestrator import TestOrchestrator


class RecordingCapture:
    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    def record_usage(self, record: UsageRecord) -> None:
        self.records.append(record)


class StubAdapter(AgentAdapter):
    @property
    def adapter_type(self) -> str:
        return "stub"

    async def execute(self, request: ATPRequest) -> ATPResponse:
        return ATPResponse(
            task_id=request.task_id,
            status=ResponseStatus.COMPLETED,
            metrics=Metrics(input_tokens=11, output_tokens=7),
        )

    def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        async def _gen() -> AsyncIterator[ATPEvent | ATPResponse]:
            yield await self.execute(request)

        return _gen()


def make_test() -> TestDefinition:
    return TestDefinition.model_validate(
        {
            "id": "t-usage-capture",
            "name": "usage capture smoke",
            "task": {"description": "say hi"},
        }
    )


@pytest.mark.anyio
async def test_execute_run_records_usage() -> None:
    capture = RecordingCapture()
    orch = TestOrchestrator(adapter=StubAdapter(), usage_capture=capture)
    result = await orch.run_single_test(make_test())
    assert result is not None
    assert len(capture.records) == 1
    rec = capture.records[0]
    assert rec.adapter_type == "stub"
    assert rec.status == "completed"
    assert rec.usage is not None
    assert rec.usage.input_tokens == 11
    assert rec.model is None
    assert rec.reported_cost_usd is None
    assert rec.test_id == "t-usage-capture"
```

Note: if `TestDefinition` requires more fields than shown, mirror the minimal
definition used by neighboring tests in `tests/unit/runner/` — the assertion
block is the contract, the fixture shape is not.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/runner/test_orchestrator_usage_capture.py -v`
Expected: FAIL — `TypeError: TestOrchestrator.__init__() got an unexpected keyword argument 'usage_capture'`

- [ ] **Step 3: Implement the seam**

In `atp/runner/orchestrator.py`:

(a) add to the imports block:

```python
from atp.cost.capture import (
    UsageCapture,
    UsageRecord,
    capture_from_env,
    usage_from_metrics,
)
```

(b) extend `__init__` signature (after `max_parallel_tests: int = 5,`):

```python
        usage_capture: UsageCapture | None = None,
```

and in the body (after `self._tests_semaphore = ...`):

```python
        self._usage_capture = usage_capture or capture_from_env()
```

Document the new arg in the `__init__` docstring:

```python
            usage_capture: Usage-capture sink (ADR-ECO-003e seam).
                Defaults to capture_from_env().
```

(c) in `_execute_run`, immediately after `end_time = datetime.now(tz=UTC)`
(currently orchestrator.py:678) and before `run_result = RunResult(...)`:

```python
            self._usage_capture.record_usage(
                UsageRecord(
                    call_id=request.task_id,
                    timestamp=end_time.isoformat(),
                    adapter_type=self.adapter.adapter_type,
                    status=response.status.value,
                    model=None,  # M1 plumbs real model ids per adapter
                    provider=None,
                    usage=usage_from_metrics(response.metrics),
                    reported_cost_usd=(
                        response.metrics.cost_usd if response.metrics else None
                    ),
                    test_id=test.id,
                )
            )
```

This point is reached on success, timeout, and failure alike (all exception
branches assign `response` before the `finally`), so one record per call is
guaranteed.

- [ ] **Step 4: Run tests + quality gates**

Run: `uv run pytest tests/unit/runner/test_orchestrator_usage_capture.py tests/unit/runner -v`
Expected: new test PASS, no regressions in the runner suite.
Run: `uv run ruff format . && uv run ruff check . && uv run pyrefly check`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add atp/runner/orchestrator.py tests/unit/runner/test_orchestrator_usage_capture.py
git commit -m "feat(runner): record UsageRecord per adapter call — 003e mandatory seam"
```

---

### Task 3: Deprecate `track_response_cost`

**Files:**
- Modify: `packages/atp-adapters/atp/adapters/base.py:312` (function body top + docstring)
- Test: `tests/unit/adapters/test_track_response_cost_deprecation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: calling `track_response_cost(...)` emits `DeprecationWarning`. Export
  stays in `atp.adapters.__all__` (API-surface test untouched); removal is M1.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/adapters/test_track_response_cost_deprecation.py`:

```python
"""track_response_cost is deprecated by the 003e UsageCapture seam."""

import pytest

from atp.adapters.base import track_response_cost
from atp.protocol import ATPResponse, ResponseStatus


@pytest.mark.anyio
async def test_track_response_cost_warns_deprecation() -> None:
    response = ATPResponse(task_id="t", status=ResponseStatus.COMPLETED)
    with pytest.warns(DeprecationWarning, match="UsageCapture"):
        await track_response_cost(response, provider="anthropic")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/adapters/test_track_response_cost_deprecation.py -v`
Expected: FAIL — `DID NOT WARN`

- [ ] **Step 3: Implement**

In `packages/atp-adapters/atp/adapters/base.py`, at the top of
`track_response_cost` (before the `if response.metrics is None:` check) add:

```python
    warnings.warn(
        "track_response_cost is deprecated and was never wired into the "
        "runtime; usage flows through the UsageCapture seam "
        "(atp.cost.capture, ADR-ECO-003e). Scheduled for removal.",
        DeprecationWarning,
        stacklevel=2,
    )
```

Add `import warnings` to the module imports. Append to the docstring first
line: `"Deprecated: superseded by the ADR-ECO-003e UsageCapture seam."`

- [ ] **Step 4: Run tests + quality gates**

Run: `uv run pytest tests/unit/adapters -v -m "not slow"` and
`uv run pytest tests/unit/test_api_surface.py -v`
Expected: PASS (export unchanged, surface test still green).
Run: `uv run ruff format . && uv run ruff check . && uv run pyrefly check`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-adapters/atp/adapters/base.py tests/unit/adapters/test_track_response_cost_deprecation.py
git commit -m "chore(adapters): deprecate unreached track_response_cost (003e)"
```

---

### Task 4: Probe report tool (`python -m atp.cost.probe_report`)

**Files:**
- Create: `packages/atp-core/atp/cost/probe_report.py`
- Test: `tests/unit/cost/test_probe_report.py`

**Interfaces:**
- Consumes: the JSONL rows written by `JsonlUsageCapture` (shape =
  `asdict(UsageRecord)`).
- Produces: `build_report(rows: list[dict]) -> str` (markdown) and a `main()`
  entry point: `uv run python -m atp.cost.probe_report <usage.jsonl>`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cost/test_probe_report.py`:

```python
"""Aggregation for the Action №0 exposure probe (ADR-ECO-003e)."""

import json
from pathlib import Path

from atp.cost.probe_report import build_report, load_rows


def row(
    adapter: str,
    usage: dict | None,
    model: str | None = None,
    cost: float | None = None,
    call_id: str = "c1",
) -> dict:
    return {
        "call_id": call_id,
        "timestamp": "2026-07-15T00:00:00+00:00",
        "adapter_type": adapter,
        "status": "completed",
        "model": model,
        "provider": None,
        "usage": usage,
        "reported_cost_usd": cost,
        "test_id": "t",
    }


USAGE = {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_creation_tokens": 0,
    "cache_read_tokens": 25,
    "usage_source": "measured",
}


def test_load_rows_reads_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "u.jsonl"
    p.write_text(
        json.dumps(row("cli", USAGE)) + "\n" + json.dumps(row("http", None)) + "\n"
    )
    assert len(load_rows(p)) == 2


def test_report_groups_by_adapter_and_counts_coverage() -> None:
    rows = [
        row("cli", USAGE, call_id="a"),
        row("cli", None, call_id="b"),
        row("http", None, call_id="c"),
    ]
    report = build_report(rows)
    # cli: 2 calls, 1 with usage; http: 1 call, 0 with usage
    assert "| cli | 2 | 1 |" in report
    assert "| http | 1 | 0 |" in report


def test_report_flags_cost_usd_and_model_coverage() -> None:
    rows = [row("cli", USAGE, model=None, cost=None, call_id="a")]
    report = build_report(rows)
    assert "cost_usd populated: 0/1" in report
    assert "model known: 0/1" in report


def test_report_sums_tokens() -> None:
    # USAGE per record: input 100 + output 50 + cache_read 25 = 175 total.
    rows = [row("cli", USAGE, call_id="a"), row("cli", USAGE, call_id="b")]
    report = build_report(rows)
    assert "| cli | 2 | 2 | 350 | 200 | 100 |" in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/cost/test_probe_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atp.cost.probe_report'`

- [ ] **Step 3: Implement**

Create `packages/atp-core/atp/cost/probe_report.py`:

```python
"""Action №0 exposure-probe report (ADR-ECO-003e).

Aggregates the JSONL written by JsonlUsageCapture into the runtime
acceptance-gate evidence: which adapter paths carry token usage, whether
model ids and cost_usd are ever populated, and token volume per path.

Usage: uv run python -m atp.cost.probe_report <usage.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _AdapterAgg:
    calls: int = 0
    with_usage: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    statuses: dict[str, int] = field(default_factory=dict)


def load_rows(path: Path) -> list[dict]:
    """Read one UsageRecord dict per JSONL line."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_report(rows: list[dict]) -> str:
    """Render the markdown evidence table for Action №0."""
    aggs: dict[str, _AdapterAgg] = defaultdict(_AdapterAgg)
    model_known = 0
    cost_populated = 0
    for r in rows:
        agg = aggs[r["adapter_type"]]
        agg.calls += 1
        status = r.get("status", "?")
        agg.statuses[status] = agg.statuses.get(status, 0) + 1
        usage = r.get("usage")
        if usage is not None:
            agg.with_usage += 1
            agg.input_tokens += usage["input_tokens"]
            agg.output_tokens += usage["output_tokens"]
            agg.cache_read_tokens += usage["cache_read_tokens"]
            agg.cache_creation_tokens += usage["cache_creation_tokens"]
        if r.get("model") is not None:
            model_known += 1
        if r.get("reported_cost_usd") is not None:
            cost_populated += 1

    total = len(rows)
    lines = [
        "# 003e Action №0 — usage-capture exposure report",
        "",
        f"- records: {total}",
        f"- model known: {model_known}/{total}",
        f"- cost_usd populated: {cost_populated}/{total}",
        "",
        "| adapter | calls | with_usage | tokens_total | input | output |",
        "|---|---|---|---|---|---|",
    ]
    for adapter in sorted(aggs):
        a = aggs[adapter]
        tokens_total = (
            a.input_tokens
            + a.output_tokens
            + a.cache_read_tokens
            + a.cache_creation_tokens
        )
        lines.append(
            f"| {adapter} | {a.calls} | {a.with_usage} "
            f"| {tokens_total} | {a.input_tokens} | {a.output_tokens} |"
        )
    lines.append("")
    for adapter in sorted(aggs):
        statuses = ", ".join(
            f"{k}={v}" for k, v in sorted(aggs[adapter].statuses.items())
        )
        lines.append(f"- {adapter} statuses: {statuses}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m atp.cost.probe_report <usage.jsonl>")
        return 2
    print(build_report(load_rows(Path(args[0]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests + quality gates**

Run: `uv run pytest tests/unit/cost/test_probe_report.py -v`
Expected: PASS.
Run: `uv run ruff format . && uv run ruff check . && uv run pyrefly check`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-core/atp/cost/probe_report.py tests/unit/cost/test_probe_report.py
git commit -m "feat(cost): Action №0 probe report over captured usage (003e)"
```

---

### Task 5: Action №0 runbook + PR

**Files:**
- Create: `docs/cost/003e-action0-probe-runbook.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the operator instructions for the bounded paid probe (run by
  Andrei; the paid runs themselves are NOT part of this plan's execution).

- [ ] **Step 1: Write the runbook**

Create `docs/cost/003e-action0-probe-runbook.md`:

```markdown
# ADR-ECO-003e Action №0 — bounded exposure probe (runbook)

Goal: runtime-confirm the static audit (usage wiring absent on flagged
paths) and measure which adapter paths carry token usage. 1–2 pre-limited
runs; no claim of full statistics. This is the acceptance gate for the
003e implementation in atp-platform.

## Steps

1. Pick 1–2 cheap suites that exercise different adapter paths, e.g.:
   - CLI path: `method/cases/code-review` with ONE routable agent
     (`method/run_pipe_check.py` limited to a single agent), or any small
     suite via `--adapter=cli`.
   - HTTP path: an `examples/` suite against a local demo agent, if wired.
2. Export the capture sink before the run:

   ```bash
   export ATP_USAGE_CAPTURE_PATH=_bench_output/003e-probe/usage.jsonl
   ```

3. Run the suite(s) with a hard external limit (small case count, one
   agent, one run each — the "bounded" in bounded probe).
4. Build the evidence report:

   ```bash
   uv run python -m atp.cost.probe_report _bench_output/003e-probe/usage.jsonl
   ```

5. Read the report against the acceptance gate:
   - `cost_usd populated: 0/N` → confirms cost_usd is never set (audit
     finding holds at runtime).
   - `model known: 0/N` → confirms no model id reaches the seam (the
     `model="unknown"` finding).
   - per-adapter `with_usage` → which paths actually carry tokens; paths
     with usage but no price identity are the $-exposure to fix first.
6. Paste the report into the 003e thread in
   `../prograph-vault/authored/decisions/` review notes (or the PR), and
   order M1 adapter adoption by the token volume column, per the ADR
   ("fix by money, not by which gap looks scariest").

## Rollback

Unset `ATP_USAGE_CAPTURE_PATH` — the seam degrades to NullUsageCapture;
no other behavior changes.
```

- [ ] **Step 2: Full test suite + gates**

Run: `uv run pytest tests/ -v -m "not slow"`
Expected: PASS, no regressions.
Run: `uv run ruff format . && uv run ruff check . && uv run pyrefly check`
Expected: clean.

- [ ] **Step 3: Commit + PR**

```bash
git add docs/cost/003e-action0-probe-runbook.md
git commit -m "docs(cost): Action №0 bounded-probe runbook (003e)"
git push -u origin feat/003e-m0-usage-capture-probe
gh pr create --title "feat: ADR-ECO-003e M0 — UsageCapture seam + Action №0 probe" --body "$(cat <<'EOF'
Observe-only M0 slice of ADR-ECO-003e (runtime cost control):

- `atp.cost.capture`: `UsageRecord`/`UsageCapture` contract + JSONL sink,
  the single mandatory usage-capture seam (003e D2).
- Orchestrator records one `UsageRecord` per adapter call (success,
  timeout, and failure paths) — enabled only when
  `ATP_USAGE_CAPTURE_PATH` is set; otherwise a no-op sink. No behavior
  change by default, no enforcement in this slice.
- `python -m atp.cost.probe_report`: aggregates captured usage into the
  Action №0 acceptance-gate evidence (per-adapter usage coverage,
  model-known and cost_usd-populated counts).
- `track_response_cost` (defined but never reached at runtime) now emits
  a DeprecationWarning; removal lands with M1 adapter adoption.
- Runbook for the bounded paid probe: docs/cost/003e-action0-probe-runbook.md.

ADR: prograph-vault/authored/decisions/2026-07-15-adr-eco-003e-runtime-cost-control.md
(sibling vault repo, referenced by name).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Then follow the repo Copilot-review loop; merge is done by the user.

---

## Roadmap — follow-up plans (NOT in this plan)

Each is a separate plan document, written after the M0 probe evidence lands:

- **M1 — adapter adoption + model identity.** Plumb real `model`/`provider`
  into `UsageRecord` per adapter (order by the probe's token-volume column:
  likely cli first, then http/bedrock/container/autogen/crewai/langgraph/mcp/
  sdk); remove `track_response_cost` and the dead `enable_cost_tracking` flag;
  decide the final seam home (orchestrator vs base-adapter template) once
  non-orchestrator callers are enumerated.
- **M2 — price snapshot (003e D7).** Snapshot generator: canonical catalog
  (`method/agents-catalog.toml`) + `method/price_overrides.toml` + litellm map
  → versioned snapshot artifact stamped `price_map_version`; a synchronous
  snapshot pricer with tri-state `pricing_status ∈ {known, ceiling, unknown}`
  (the "silent zero = unknown" rule, D6.3); deprecate & remove the System-A
  table in `atp/cost/models.py` and migrate `CostTracker` off it.
- **M3 — BudgetControl (003e D1/D3/D4/D5).** `estimate/reserve/settle` with an
  atomic reservation store (SQLite single-writer, idempotent on
  call_id/reservation_id, settle-timeout reaper); scope taxonomy
  attempt⊂task⊂run⊂day; per-scope policy (`enforcement_mode`,
  `unknown_price_policy`, `deny_outcome`); wire deny into the orchestrator
  (note: `budget_usd` already flows into `ATPRequest.constraints` at
  `orchestrator.py:135` but nothing enforces it — that's the natural first
  enforcement point).
- **M4 — ecosystem handoff.** When the contract module stabilizes, write the
  vendoring handoff note for Maestro / spec-runner / robin-runtime into
  `../prograph-vault/authored/notes/` (their repos are read-only from here);
  arbiter alignment of its advisory budget invariant.
```
