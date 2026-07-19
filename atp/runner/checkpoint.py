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


def _test_result_from_dict(test: TestDefinition, payload: dict[str, Any]) -> TestResult:
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
        """Rehydrate recorded results in suite order.

        The checkpoint dict preserves completion order, which is
        nondeterministic after a parallel run; iterating the suite keeps
        restored results stable. Recorded tests no longer present in the
        suite are skipped.
        """
        results: list[TestResult] = []
        for test in suite.tests:
            payload = self._tests.get(test.id)
            if payload is None:
                continue
            results.append(_test_result_from_dict(test, payload))
        return results

    def delete(self) -> None:
        """Remove the checkpoint file (idempotent)."""
        self.path.unlink(missing_ok=True)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Per-process temp name: concurrent atp processes on the same suite
        # must not interleave writes into a shared temp file. Within one
        # process this method is synchronous, so asyncio tasks cannot race.
        tmp = self.path.with_suffix(f".json.tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps({"version": CHECKPOINT_VERSION, "tests": self._tests})
        )
        os.replace(tmp, self.path)
