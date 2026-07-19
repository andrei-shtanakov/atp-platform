# Runtime Observability & Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing (but unused) structlog infrastructure into the ATP CLI runtime, add run_id correlation to test runs, handle SIGINT/SIGTERM cooperatively, and make interrupted `atp test` runs resumable via suite checkpoints.

**Architecture:** All four gaps close inside two seams that already exist: the click group `cli()` in `atp/cli/main.py` (process-wide setup: logging, signals) and `TestOrchestrator` in `atp/runner/orchestrator.py` (per-suite execution: correlation, cooperative shutdown, checkpoints). No new services, no schema migrations. The checkpoint is a single atomic JSON file per (suite, agent) pair; resume = seed `SuiteResult.tests` from it and run only the remaining tests.

**Tech Stack:** Python 3.11+, click, structlog (already a dep via `packages/atp-core`), pydantic (protocol models), dataclasses (result models), pytest + anyio.

## Global Constraints

- Package management: **only `uv`** (`uv run pytest`, `uv run ruff ...`). Never pip.
- Line length: **88 chars**. Type hints required. Public APIs get docstrings.
- Async tests: `@pytest.mark.anyio` (project convention, see `tests/unit/runner/test_orchestrator.py:165`).
- After every change: `uv run ruff format . && uv run ruff check . --fix`, then `uv run pyrefly check` (run `pyrefly init` first if not initialized) and fix errors.
- Git: work on branch `feat/runtime-observability-recovery`; changes land **only via PR**; merge is done by a human (polyrepo umbrella policy).
- Existing exit-code contract in `atp/cli/main.py:42-44`: 0=pass, 1=fail, 2=error. This plan adds 130=interrupted.
- Do not touch the dashboard's own logging (`packages/atp-dashboard/.../v2/logging_config.py`) — it configures itself in `create_app()` and stays as is.

## File Structure

| File | Role |
|---|---|
| `atp/cli/main.py` | Modify: wire `configure_logging` into `cli()`; correlation + signal install + `--resume` flag in `test_cmd`/`run`; `EXIT_INTERRUPTED` |
| `atp/core/results.py` | Modify: add `run_id` field to `SuiteResult` |
| `atp/runner/orchestrator.py` | Modify: capture correlation id; `request_shutdown()`; checkpoint skip/record |
| `atp/runner/checkpoint.py` | **Create**: `SuiteCheckpoint` (serialize/rehydrate `TestResult`, atomic save) |
| `tests/unit/cli/test_logging_wiring.py` | **Create**: Task 1 tests |
| `tests/unit/cli/test_signal_handlers.py` | **Create**: Task 3 CLI-helper tests |
| `tests/unit/runner/test_orchestrator.py` | Modify: run_id + shutdown + checkpoint-integration tests |
| `tests/unit/runner/test_checkpoint.py` | **Create**: Task 4 round-trip tests |

---

### Task 1: Wire structlog into the CLI entry point

`atp/core/logging.py:454` has a complete `configure_logging()` that nothing calls. Every `atp` invocation should pass through it so all ~84 stdlib `logging.getLogger(__name__)` call sites get structured output (the `ProcessorFormatter.foreign_pre_chain` bridges stdlib records through the structlog processors, including redaction and correlation).

**Files:**
- Modify: `atp/cli/main.py` (the `cli()` group, `main.py:283-318`)
- Test: `tests/unit/cli/test_logging_wiring.py` (create)

**Interfaces:**
- Consumes: `configure_logging(level, json_output, log_file, module_levels)` from `atp/core/logging.py:454`; `get_cached_settings()` from `atp/core/settings.py` (returns `ATPSettings` with `.logging: LoggingSettings` — fields `level`, `json_output`, `file`).
- Produces: `_setup_cli_logging(verbose: bool) -> None` in `atp/cli/main.py` (Tasks 2-4 assume logging is already configured process-wide).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_logging_wiring.py`:

```python
"""Tests that the CLI entry point configures structured logging."""

import logging

import pytest
import structlog
from click.testing import CliRunner

from atp.cli.main import cli


@pytest.fixture(autouse=True)
def restore_root_logger():
    """Save and restore root logger handlers/level around each test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_cli_installs_structlog_formatter() -> None:
    result = CliRunner().invoke(cli, ["version"])
    assert result.exit_code == 0
    root = logging.getLogger()
    assert any(
        isinstance(h.formatter, structlog.stdlib.ProcessorFormatter)
        for h in root.handlers
    ), "cli() must route stdlib logging through structlog ProcessorFormatter"


def test_cli_verbose_sets_debug_level() -> None:
    result = CliRunner().invoke(cli, ["--verbose", "version"])
    assert result.exit_code == 0
    assert logging.getLogger().level == logging.DEBUG
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/cli/test_logging_wiring.py -v`
Expected: both FAIL (no ProcessorFormatter on root handlers; level not DEBUG).

- [ ] **Step 3: Implement `_setup_cli_logging` and call it from `cli()`**

In `atp/cli/main.py`, add above the `cli()` group definition:

```python
def _setup_cli_logging(verbose: bool) -> None:
    """Configure structured logging for the CLI process.

    Routes stdlib logging through structlog (JSON when not a TTY or when
    settings request it; pretty console otherwise). --verbose forces DEBUG.
    """
    from atp.core.logging import configure_logging
    from atp.core.settings import get_cached_settings

    settings = get_cached_settings()
    configure_logging(
        level="DEBUG" if verbose else settings.logging.level,
        # False means "not explicitly requested" -> let configure_logging
        # auto-detect by TTY; True forces JSON.
        json_output=True if settings.logging.json_output else None,
        log_file=settings.logging.file,
    )
```

Then inside `def cli(...)` (`main.py:283`), right after `config_ctx.verbose = verbose`:

```python
    _setup_cli_logging(verbose)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/cli/test_logging_wiring.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Guard against regressions in the full CLI test module**

Run: `uv run pytest tests/unit/cli -x -q`
Expected: PASS. If any test asserts on exact stdout text that now includes log lines, note that `configure_logging` writes to **stdout** (`logging.py:536`) while click output also goes to stdout — if this collides, change the console handler target in `_setup_cli_logging` is NOT allowed (that's core code); instead raise it in the PR description and set `level` default to `WARNING` for non-verbose runs.

- [ ] **Step 6: Format, lint, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add atp/cli/main.py tests/unit/cli/test_logging_wiring.py
git commit -m "feat(cli): wire structlog configure_logging into CLI entry point"
```

---

### Task 2: run_id correlation for test runs

Give every `atp test` invocation a run_id that (a) appears on every log record via the existing `add_correlation_id` processor, and (b) is recorded on the `SuiteResult`.

**Files:**
- Modify: `atp/core/results.py` (add field to `SuiteResult`, `results.py:279-287`)
- Modify: `atp/runner/orchestrator.py` (`run_suite`, `orchestrator.py:760-784`)
- Modify: `atp/cli/main.py` (`test_cmd` around `main.py:587`, `run` around its `asyncio.run(_run_suite(...))` call — the alias duplicates the body)
- Test: `tests/unit/runner/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: `correlation_context` (ctx manager, `__enter__` returns the id str) and `get_correlation_id() -> str | None` from `atp/core/logging.py:82-138`.
- Produces: `SuiteResult.run_id: str | None` (field, default `None`) — Task 4's checkpoint and any reporter may read it.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/runner/test_orchestrator.py` (uses the module's existing `MockAdapter` class and `test_suite` fixture):

```python
class TestRunIdCorrelation:
    """run_suite must capture the active correlation id as run_id."""

    @pytest.mark.anyio
    async def test_run_suite_captures_correlation_id(
        self, test_suite: TestSuite
    ) -> None:
        from atp.core.logging import correlation_context

        orchestrator = TestOrchestrator(adapter=MockAdapter())
        with correlation_context("run-abc-123"):
            result = await orchestrator.run_suite(
                test_suite, agent_name="agent-x"
            )
        assert result.run_id == "run-abc-123"

    @pytest.mark.anyio
    async def test_run_suite_without_context_leaves_run_id_none(
        self, test_suite: TestSuite
    ) -> None:
        orchestrator = TestOrchestrator(adapter=MockAdapter())
        result = await orchestrator.run_suite(test_suite, agent_name="agent-x")
        assert result.run_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/runner/test_orchestrator.py -k RunIdCorrelation -v`
Expected: FAIL — `SuiteResult` has no attribute/field `run_id` (TypeError or AttributeError).

- [ ] **Step 3: Add the field and capture it**

In `atp/core/results.py`, inside `@dataclass class SuiteResult` (`results.py:279`), after `error: str | None = None` add:

```python
    run_id: str | None = None
```

In `atp/runner/orchestrator.py` add to the imports block:

```python
from atp.core.logging import get_correlation_id
```

and in `run_suite` (`orchestrator.py:780`) change the `SuiteResult(...)` construction to:

```python
        result = SuiteResult(
            suite_name=suite.test_suite,
            agent_name=agent_name,
            start_time=datetime.now(tz=UTC),
            run_id=get_correlation_id(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/runner/test_orchestrator.py -k RunIdCorrelation -v`
Expected: 2 PASS. Also run the whole file: `uv run pytest tests/unit/runner/test_orchestrator.py -q` — PASS.

- [ ] **Step 5: Establish the context in the CLI**

In `atp/cli/main.py`, in `test_cmd`, wrap the suite execution (`main.py:587`, the `result = asyncio.run(_run_suite(...))` statement) as:

```python
        from atp.core.logging import correlation_context

        with correlation_context() as run_id:
            if verbose:
                click.echo(f"Run ID: {run_id}")
            result = asyncio.run(
                _run_suite(
                    suite=suite,
                    adapter_type=adapter,
                    adapter_config=config_dict,
                    agent_name=agent_name,
                    model=model,
                    parallel=parallel,
                    runs_per_test=runs,
                    fail_fast=fail_fast,
                    sandbox_enabled=sandbox,
                    verbose=verbose,
                    output_format=output,
                    summary_format=summary_format,
                    output_file=output_file,
                    save_to_db=not no_save,
                    save_results_dir=save_results,
                    live=live,
                    enable_tracing=trace,
                )
            )
```

Apply the identical wrapper in the `run` alias command (`@cli.command(name="run")`, `main.py:618`; its body contains the same `asyncio.run(_run_suite(...))` call — search for it inside `def run(`). `ContextVar` state set before `asyncio.run` is visible inside the loop, so the orchestrator's `get_correlation_id()` sees it.

- [ ] **Step 6: Format, lint, typecheck, commit**

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add atp/core/results.py atp/runner/orchestrator.py atp/cli/main.py \
  tests/unit/runner/test_orchestrator.py
git commit -m "feat(runner): correlate test runs with run_id via correlation_context"
```

---

### Task 3: Cooperative SIGINT/SIGTERM shutdown

Today a Ctrl-C mid-suite raises `KeyboardInterrupt` somewhere inside `asyncio.run` and loses everything. Add a cooperative flag: signal → finish the in-flight test, skip the rest, report partial results, exit 130.

**Files:**
- Modify: `atp/runner/orchestrator.py` (`__init__`, sequential loop `orchestrator.py:837`, parallel wrapper `orchestrator.py:543`)
- Modify: `atp/cli/main.py` (`EXIT_INTERRUPTED`, `_install_signal_handlers`, wiring in `_run_suite` around `main.py:878-891`)
- Test: `tests/unit/runner/test_orchestrator.py` (append), `tests/unit/cli/test_signal_handlers.py` (create)

**Interfaces:**
- Consumes: `TestOrchestrator` internals from Task 2's state of the file.
- Produces: `TestOrchestrator.request_shutdown() -> None`, `TestOrchestrator.shutdown_requested: bool` (property); `_install_signal_handlers(orchestrator) -> Callable[[], None]` in `main.py`; `EXIT_INTERRUPTED = 130`. Task 4 relies on `result.error` being set to `"interrupted: shutdown requested"` on interrupt (checkpoint is kept when `error` is not None).

- [ ] **Step 1: Write the failing orchestrator tests**

Append to `tests/unit/runner/test_orchestrator.py`:

```python
class TestCooperativeShutdown:
    """request_shutdown() stops the suite between tests."""

    @pytest.mark.anyio
    async def test_shutdown_before_start_runs_nothing(
        self, test_suite: TestSuite
    ) -> None:
        orchestrator = TestOrchestrator(adapter=MockAdapter())
        orchestrator.request_shutdown()
        result = await orchestrator.run_suite(test_suite, agent_name="a")
        assert result.tests == []
        assert result.error == "interrupted: shutdown requested"

    @pytest.mark.anyio
    async def test_shutdown_after_first_test_skips_rest(
        self, test_suite: TestSuite
    ) -> None:
        orchestrator = TestOrchestrator(adapter=MockAdapter())

        def stop_after_first(event: ProgressEvent) -> None:
            if event.event_type == ProgressEventType.TEST_COMPLETED:
                orchestrator.request_shutdown()

        orchestrator.progress_callback = stop_after_first
        result = await orchestrator.run_suite(test_suite, agent_name="a")
        assert len(result.tests) == 1  # suite has 2 tests
        assert result.error == "interrupted: shutdown requested"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/runner/test_orchestrator.py -k CooperativeShutdown -v`
Expected: FAIL with `AttributeError: ... has no attribute 'request_shutdown'`.

- [ ] **Step 3: Implement the flag in the orchestrator**

In `TestOrchestrator.__init__` (`orchestrator.py:105`, after `self._usage_capture = ...`):

```python
        self._shutdown_requested = False
```

Add methods after `_emit_progress` (`orchestrator.py:113`):

```python
    def request_shutdown(self) -> None:
        """Request cooperative shutdown: finish the current test, skip the rest."""
        logger.warning("Shutdown requested; will stop after the current test")
        self._shutdown_requested = True

    @property
    def shutdown_requested(self) -> bool:
        """Whether a cooperative shutdown has been requested."""
        return self._shutdown_requested
```

In `run_suite`, at the top of the sequential loop body (`orchestrator.py:837`, before `logger.info("Running test %d/%d...")`):

```python
                    for idx, test in enumerate(suite.tests):
                        if self._shutdown_requested:
                            result.error = "interrupted: shutdown requested"
                            break
```

Guard the parallel entry the same way — in `run_suite`, wrap the parallel branch (`orchestrator.py:819`): before calling `_execute_tests_parallel`, add:

```python
                if self._shutdown_requested:
                    result.error = "interrupted: shutdown requested"
```

(as an `if/else` so `_execute_tests_parallel` only runs when not shut down), and in `run_test_with_semaphore` (`orchestrator.py:543`) return early after acquiring the semaphore:

```python
        async def run_test_with_semaphore(test: TestDefinition) -> TestResult:
            async with self._tests_semaphore:
                if self._shutdown_requested:
                    skipped = TestResult(test=test)
                    skipped.error = "skipped: shutdown requested"
                    skipped.end_time = skipped.start_time
                    return skipped
```

After the `gather` in `_execute_tests_parallel` returns and results are collected, no extra handling is needed — skipped entries carry their error. In `run_suite`, after the parallel branch completes, set the suite error if any test was skipped:

```python
                    if any(
                        t.error == "skipped: shutdown requested"
                        for t in result.tests
                    ):
                        result.error = "interrupted: shutdown requested"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/runner/test_orchestrator.py -q`
Expected: PASS (new + existing).

- [ ] **Step 5: Write the failing CLI signal-handler test**

Create `tests/unit/cli/test_signal_handlers.py`:

```python
"""Tests for cooperative signal handling in the CLI runner."""

import asyncio
import os
import signal

import pytest

from atp.cli.main import _install_signal_handlers


class FakeOrchestrator:
    def __init__(self) -> None:
        self.stopped = False

    def request_shutdown(self) -> None:
        self.stopped = True


@pytest.mark.anyio
async def test_sigint_triggers_request_shutdown() -> None:
    orchestrator = FakeOrchestrator()
    remove = _install_signal_handlers(orchestrator)
    try:
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(0.05)  # let the loop dispatch the handler
        assert orchestrator.stopped is True
    finally:
        remove()
```

Run: `uv run pytest tests/unit/cli/test_signal_handlers.py -v`
Expected: FAIL with ImportError (`_install_signal_handlers` not defined).

- [ ] **Step 6: Implement signal wiring in the CLI**

In `atp/cli/main.py` add next to the exit codes (`main.py:44`):

```python
EXIT_INTERRUPTED = 130  # Interrupted by SIGINT/SIGTERM (128 + SIGINT)
```

Add near `_run_suite`:

```python
def _install_signal_handlers(orchestrator: Any) -> Callable[[], None]:
    """Install SIGINT/SIGTERM handlers that request cooperative shutdown.

    Returns a zero-arg callable that removes the installed handlers.
    """
    import signal

    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, orchestrator.request_shutdown)
            installed.append(sig)
        except (NotImplementedError, RuntimeError):
            pass  # e.g. Windows or non-main thread

    def _remove() -> None:
        for sig in installed:
            loop.remove_signal_handler(sig)

    return _remove
```

(`Callable` comes from `collections.abc` — add `from collections.abc import Callable` to the imports if absent.)

In `_run_suite`, wrap the orchestrator run (`main.py:878-891`):

```python
        async with TestOrchestrator(
            adapter=adapter,
            sandbox_config=sandbox_config,
            progress_callback=progress_callback,
            runs_per_test=runs_per_test,
            fail_fast=fail_fast,
            parallel_tests=use_parallel,
            max_parallel_tests=parallel,
        ) as orchestrator:
            remove_handlers = _install_signal_handlers(orchestrator)
            try:
                result = await orchestrator.run_suite(
                    suite=suite,
                    agent_name=agent_name,
                    runs_per_test=runs_per_test,
                )
            finally:
                remove_handlers()
```

At the **end** of `_run_suite`, right before its final `return` (so partial results are still evaluated/reported/saved first):

```python
    if orchestrator.shutdown_requested:
        click.echo(
            "Run interrupted; partial results were reported. "
            "Re-run with --resume to continue.",
            err=True,
        )
        sys.exit(EXIT_INTERRUPTED)
```

- [ ] **Step 7: Run tests, then commit**

```bash
uv run pytest tests/unit/cli/test_signal_handlers.py tests/unit/runner -q
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add atp/runner/orchestrator.py atp/cli/main.py \
  tests/unit/runner/test_orchestrator.py tests/unit/cli/test_signal_handlers.py
git commit -m "feat(runner): cooperative SIGINT/SIGTERM shutdown with exit code 130"
```

---

### Task 4: Suite checkpoint + `atp test --resume`

Persist each completed `TestResult` to an atomic JSON file; on `--resume`, skip completed tests and seed their saved results. Checkpoint is deleted only when the suite runs to completion.

**Files:**
- Create: `atp/runner/checkpoint.py`
- Modify: `atp/runner/orchestrator.py` (`__init__`, `run_suite`)
- Modify: `atp/cli/main.py` (`--resume` flag on `test`/`run`, `_run_suite` signature + checkpoint creation)
- Test: `tests/unit/runner/test_checkpoint.py` (create), `tests/unit/runner/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: `TestResult`/`RunResult` dataclasses (`atp/core/results.py:177-209`; `RunResult.response: ATPResponse`, `.events: list[ATPEvent]` are pydantic — use `.model_dump(mode="json")` / `.model_validate(...)`); `TestSuite.tests: list[TestDefinition]` with `.id`; Task 3's `result.error` interrupt convention.
- Produces:
  - `SuiteCheckpoint(path: Path)`; `SuiteCheckpoint.default_path(suite_name: str, agent_name: str, base_dir: Path | None = None) -> Path`; `.completed_ids() -> set[str]`; `.record(result: TestResult) -> None`; `.load_results(suite: TestSuite) -> list[TestResult]`; `.delete() -> None`
  - `TestOrchestrator(..., checkpoint: SuiteCheckpoint | None = None)`
  - CLI flag `--resume` on `atp test` / `atp run`

- [ ] **Step 1: Write the failing checkpoint unit tests**

Create `tests/unit/runner/test_checkpoint.py`:

```python
"""Tests for suite checkpoint persistence."""

from datetime import UTC, datetime
from pathlib import Path

from atp.core.results import RunResult, TestResult
from atp.loader.models import (
    Constraints,
    TaskDefinition,
    TestDefaults,
    TestDefinition,
    TestSuite,
)
from atp.protocol import ATPResponse, ResponseStatus
from atp.runner.checkpoint import SuiteCheckpoint


def make_test(test_id: str) -> TestDefinition:
    return TestDefinition(
        id=test_id,
        name=f"Test {test_id}",
        task=TaskDefinition(description="do a thing"),
        constraints=Constraints(timeout_seconds=10),
    )


def make_result(test: TestDefinition) -> TestResult:
    run = RunResult(
        test_id=test.id,
        run_number=1,
        response=ATPResponse(
            task_id=test.id, status=ResponseStatus.COMPLETED
        ),
        end_time=datetime.now(tz=UTC),
    )
    return TestResult(test=test, runs=[run], end_time=datetime.now(tz=UTC))


def make_suite(*tests: TestDefinition) -> TestSuite:
    return TestSuite(
        test_suite="cp-suite",
        tests=list(tests),
        defaults=TestDefaults(runs_per_test=1),
    )


def test_record_and_reload_round_trip(tmp_path: Path) -> None:
    test = make_test("t-1")
    cp = SuiteCheckpoint(tmp_path / "cp.json")
    cp.record(make_result(test))

    reloaded = SuiteCheckpoint(tmp_path / "cp.json")
    assert reloaded.completed_ids() == {"t-1"}
    results = reloaded.load_results(make_suite(test))
    assert len(results) == 1
    assert results[0].test.id == "t-1"
    assert results[0].success is True
    assert results[0].runs[0].response.status == ResponseStatus.COMPLETED


def test_load_ignores_tests_missing_from_suite(tmp_path: Path) -> None:
    stale = make_test("gone")
    cp = SuiteCheckpoint(tmp_path / "cp.json")
    cp.record(make_result(stale))
    reloaded = SuiteCheckpoint(tmp_path / "cp.json")
    assert reloaded.load_results(make_suite(make_test("t-2"))) == []


def test_delete_removes_file(tmp_path: Path) -> None:
    cp = SuiteCheckpoint(tmp_path / "cp.json")
    cp.record(make_result(make_test("t-1")))
    assert (tmp_path / "cp.json").exists()
    cp.delete()
    assert not (tmp_path / "cp.json").exists()
    cp.delete()  # idempotent


def test_default_path_slugifies(tmp_path: Path) -> None:
    p = SuiteCheckpoint.default_path("My Suite!", "agent/x", base_dir=tmp_path)
    assert p.parent == tmp_path
    assert p.name == "My-Suite--agent-x.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/runner/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atp.runner.checkpoint'`.

- [ ] **Step 3: Implement `atp/runner/checkpoint.py`**

```python
"""Suite-level checkpoint persistence for crash-safe ``atp test`` runs.

Each completed :class:`TestResult` is appended to a single JSON file with an
atomic temp-file + ``os.replace`` write, so a crash or SIGTERM mid-run never
corrupts the checkpoint. ``atp test --resume`` seeds completed results from
the file and executes only the remaining tests.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from atp.core.results import RunResult, TestResult
from atp.loader.models import TestDefinition, TestSuite
from atp.protocol import ATPEvent, ATPResponse

CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_DIR = Path(".atp-runs") / "checkpoints"


def _slug(name: str) -> str:
    """Make a name safe for use as a filename component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "unnamed"


def _dt_to_str(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _dt_from_str(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _run_to_dict(run: RunResult) -> dict[str, Any]:
    return {
        "test_id": run.test_id,
        "run_number": run.run_number,
        "response": run.response.model_dump(mode="json"),
        "events": [event.model_dump(mode="json") for event in run.events],
        "start_time": _dt_to_str(run.start_time),
        "end_time": _dt_to_str(run.end_time),
        "error": run.error,
    }


def _run_from_dict(payload: dict[str, Any]) -> RunResult:
    start_time = _dt_from_str(payload["start_time"])
    if start_time is None:
        raise ValueError("checkpoint run entry missing start_time")
    return RunResult(
        test_id=payload["test_id"],
        run_number=payload["run_number"],
        response=ATPResponse.model_validate(payload["response"]),
        events=[ATPEvent.model_validate(e) for e in payload["events"]],
        start_time=start_time,
        end_time=_dt_from_str(payload["end_time"]),
        error=payload["error"],
    )


def _test_result_to_dict(result: TestResult) -> dict[str, Any]:
    return {
        "runs": [_run_to_dict(run) for run in result.runs],
        "start_time": _dt_to_str(result.start_time),
        "end_time": _dt_to_str(result.end_time),
        "error": result.error,
    }


def _test_result_from_dict(
    test: TestDefinition, payload: dict[str, Any]
) -> TestResult:
    start_time = _dt_from_str(payload["start_time"])
    if start_time is None:
        raise ValueError("checkpoint test entry missing start_time")
    return TestResult(
        test=test,
        runs=[_run_from_dict(run) for run in payload["runs"]],
        start_time=start_time,
        end_time=_dt_from_str(payload["end_time"]),
        error=payload["error"],
    )


class SuiteCheckpoint:
    """Persist completed test results so an interrupted run can resume."""

    def __init__(self, path: Path) -> None:
        """Load an existing checkpoint file if present, else start empty."""
        self.path = path
        self._tests: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return  # corrupt/unreadable checkpoint -> start fresh
            if data.get("version") == CHECKPOINT_VERSION:
                self._tests = data.get("tests", {})

    @staticmethod
    def default_path(
        suite_name: str,
        agent_name: str,
        base_dir: Path | None = None,
    ) -> Path:
        """Conventional checkpoint location for a (suite, agent) pair."""
        base = base_dir if base_dir is not None else DEFAULT_CHECKPOINT_DIR
        return base / f"{_slug(suite_name)}--{_slug(agent_name)}.json"

    def completed_ids(self) -> set[str]:
        """IDs of tests already recorded in this checkpoint."""
        return set(self._tests)

    def record(self, result: TestResult) -> None:
        """Record a completed test result and persist atomically."""
        self._tests[result.test.id] = _test_result_to_dict(result)
        self._save()

    def load_results(self, suite: TestSuite) -> list[TestResult]:
        """Rehydrate recorded results, matching tests by id in the suite."""
        by_id = {test.id: test for test in suite.tests}
        results: list[TestResult] = []
        for test_id, payload in self._tests.items():
            test = by_id.get(test_id)
            if test is None:
                continue  # suite changed since the checkpoint was written
            results.append(_test_result_from_dict(test, payload))
        return results

    def delete(self) -> None:
        """Remove the checkpoint file (idempotent)."""
        self.path.unlink(missing_ok=True)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"version": CHECKPOINT_VERSION, "tests": self._tests})
        )
        os.replace(tmp, self.path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/runner/test_checkpoint.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Write the failing orchestrator-integration test**

Append to `tests/unit/runner/test_orchestrator.py`:

```python
class TestCheckpointIntegration:
    """run_suite skips checkpointed tests and records new completions."""

    @pytest.mark.anyio
    async def test_resume_skips_completed_and_records_rest(
        self, test_suite: TestSuite, tmp_path
    ) -> None:
        from atp.runner.checkpoint import SuiteCheckpoint

        cp_path = tmp_path / "cp.json"

        # First run: interrupt after the first test completes.
        orch1 = TestOrchestrator(
            adapter=MockAdapter(), checkpoint=SuiteCheckpoint(cp_path)
        )

        def stop_after_first(event: ProgressEvent) -> None:
            if event.event_type == ProgressEventType.TEST_COMPLETED:
                orch1.request_shutdown()

        orch1.progress_callback = stop_after_first
        first = await orch1.run_suite(test_suite, agent_name="a")
        assert len(first.tests) == 1
        assert SuiteCheckpoint(cp_path).completed_ids() == {
            first.tests[0].test.id
        }

        # Second run: resume; completed test must not re-execute.
        executed: list[str] = []

        class CountingAdapter(MockAdapter):
            async def execute(self, request):  # type: ignore[override]
                executed.append(request.task_id)
                return await super().execute(request)

            async def stream_events(self, request):  # type: ignore[override]
                executed.append(request.task_id)
                async for item in super().stream_events(request):
                    yield item

        orch2 = TestOrchestrator(
            adapter=CountingAdapter(), checkpoint=SuiteCheckpoint(cp_path)
        )
        second = await orch2.run_suite(test_suite, agent_name="a")
        assert len(second.tests) == 2
        done_id = first.tests[0].test.id
        assert all(done_id not in task_id for task_id in executed)
        # Suite completed -> checkpoint removed.
        assert not cp_path.exists()
```

Run: `uv run pytest tests/unit/runner/test_orchestrator.py -k CheckpointIntegration -v`
Expected: FAIL — `TestOrchestrator.__init__` has no `checkpoint` parameter.

- [ ] **Step 6: Integrate the checkpoint into `TestOrchestrator`**

In `orchestrator.py` imports:

```python
from atp.runner.checkpoint import SuiteCheckpoint
```

`__init__` (`orchestrator.py:63`): add parameter `checkpoint: SuiteCheckpoint | None = None,` (after `usage_capture`), docstring line for it, and:

```python
        self.checkpoint = checkpoint
```

In `run_suite`, right after `suite.apply_defaults()` (`orchestrator.py:816`), compute pending tests and seed saved results:

```python
            pending_tests = suite.tests
            if self.checkpoint is not None:
                done = self.checkpoint.completed_ids()
                if done:
                    restored = self.checkpoint.load_results(suite)
                    result.tests.extend(restored)
                    pending_tests = [
                        t for t in suite.tests if t.id not in done
                    ]
                    logger.info(
                        "Resuming suite: %d restored, %d pending",
                        len(restored),
                        len(pending_tests),
                    )
```

Replace both iteration sources with `pending_tests`: the parallel call becomes `self._execute_tests_parallel(tests=pending_tests, ...)` and the sequential loop becomes `for idx, test in enumerate(pending_tests):` (also use `len(pending_tests)` in that loop's log line).

Record completions — sequential path, right after `result.tests.append(test_result)` (`orchestrator.py:852`):

```python
                        if self.checkpoint is not None:
                            self.checkpoint.record(test_result)
```

Parallel path, in `run_test_with_semaphore` (Task 3's version), record before returning the real result:

```python
                test_result = await self.run_single_test(
                    test, runs=num_runs, suite_name=suite_name
                )
                if self.checkpoint is not None:
                    self.checkpoint.record(test_result)
                return test_result
```

Delete on completion — in `run_suite`, immediately after `result.end_time = datetime.now(tz=UTC)` (`orchestrator.py:872`):

```python
            ran_to_completion = (
                result.error is None
                and len(result.tests) == len(suite.tests)
            )
            if self.checkpoint is not None and ran_to_completion:
                self.checkpoint.delete()
```

(Interrupted runs set `result.error` in Task 3, so their checkpoint survives; `fail_fast` breaks leave `len(result.tests) < len(suite.tests)`, so theirs survives too and `--resume` picks up after the failure.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/runner -q`
Expected: PASS (checkpoint integration + all existing runner tests).

- [ ] **Step 8: Wire `--resume` through the CLI**

In `atp/cli/main.py`:

1. Add the option to **both** `test` (`main.py:334` block) and `run` (`main.py:618` block) commands, next to `--no-save`:

```python
@click.option(
    "--resume",
    is_flag=True,
    help=(
        "Resume an interrupted run: skip tests already recorded in the "
        "suite checkpoint (.atp-runs/checkpoints/) and run the rest."
    ),
)
```

2. Add `resume: bool,` to both command function signatures and pass `resume=resume` in both `_run_suite(...)` calls.

3. In `_run_suite`: add parameter `resume: bool = False,` to the signature and docstring; before the `TestOrchestrator` construction (`main.py:878`):

```python
    from atp.runner.checkpoint import SuiteCheckpoint

    checkpoint_path = SuiteCheckpoint.default_path(
        suite.test_suite, agent_name
    )
    if not resume:
        checkpoint_path.unlink(missing_ok=True)  # stale checkpoint: start fresh
    checkpoint = SuiteCheckpoint(checkpoint_path)
```

4. Pass `checkpoint=checkpoint,` in the `TestOrchestrator(...)` construction.

- [ ] **Step 9: End-to-end smoke check, format, typecheck, commit**

```bash
uv run pytest tests/unit/runner tests/unit/cli -q
uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check
git add atp/runner/checkpoint.py atp/runner/orchestrator.py atp/cli/main.py \
  tests/unit/runner/test_checkpoint.py tests/unit/runner/test_orchestrator.py
git commit -m "feat(runner): suite checkpoints and atp test --resume"
```

---

### Task 5: Full verification and PR

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest tests/unit -q`
Expected: PASS. If unrelated tests were already failing on `main`, record them in the PR description; do not fix them here.

- [ ] **Step 2: Manual smoke test** (uses any small suite, e.g. `my-test-suite.yaml` in the repo root)

```bash
uv run atp test my-test-suite.yaml --list-only
uv run atp --verbose version
```

Expected: list output unchanged; verbose run shows structured (colored console) log lines and a `Run ID:` echo on test runs.

- [ ] **Step 3: Push branch and open PR**

```bash
git push -u origin feat/runtime-observability-recovery
gh pr create --title "Runtime observability & recovery: structlog wiring, run_id, signals, checkpoints" \
  --body "$(cat <<'EOF'
Closes the audit gaps from docs/superpowers/plans/2026-07-19-runtime-observability-recovery.md:

- Wire the existing atp.core.logging configure_logging() into the CLI entry point (structured output + secret redaction for all stdlib loggers).
- run_id correlation: every `atp test` run gets a correlation id on all log records and on SuiteResult.run_id.
- Cooperative SIGINT/SIGTERM shutdown: finish the in-flight test, report partial results, exit 130.
- Suite checkpoints + `atp test --resume`: interrupted runs skip already-completed tests.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Then track GitHub Copilot review comments and address them; merge is done by a human (umbrella git policy).

---

## Self-Review (completed at plan time)

- Spec coverage: (1) structlog wiring → Task 1; (2) run_id correlation → Task 2; (3) SIGTERM/SIGINT → Task 3; (4) checkpoints/resume for `atp test` → Task 4. Out of scope by decision: dashboard logging unification, `method/` print()-layer, JSON log schema alignment with the ecosystem obs-v1 contract (candidates for a follow-up plan).
- Known risk (Task 1, flagged in Step 5): `configure_logging` writes logs to stdout, which `atp test --output=json` also uses. Non-verbose default level stays INFO; if CLI JSON output tests break, the documented fallback is raising default to WARNING and flagging the stdout/stderr split in the PR.
- Type consistency: `SuiteCheckpoint` API used identically in Tasks 4 steps 5-8; `request_shutdown`/`shutdown_requested` names consistent between orchestrator (Task 3 step 3) and CLI (`_install_signal_handlers`, `_run_suite` tail).
