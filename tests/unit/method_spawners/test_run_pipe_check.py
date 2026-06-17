"""Tests for the pipe-check harness CLI error paths (Phase A-2)."""

import os
import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[3] / "method" / "run_pipe_check.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(HARNESS), *args],
        capture_output=True,
        env=os.environ.copy(),
        timeout=60,
    )


def test_unknown_task_type_exits_2_with_stderr() -> None:
    # Fail fast on an unknown --task-type: one-line stderr + exit 2 (not a
    # traceback), before any agent runs.
    proc = _run(["--agents", "claude_code", "--task-type", "bogus", "--dry-run"])
    assert proc.returncode == 2
    err = proc.stderr.decode()
    assert "unknown task_type" in err
    assert "Traceback" not in err


def test_unknown_agent_exits_2() -> None:
    proc = _run(["--agents", "nope", "--task-type", "review", "--dry-run"])
    assert proc.returncode == 2
    assert "Unknown agent" in proc.stderr.decode()


def _completed_test_result() -> object:
    """Build a minimal completed TestResult for a real code-review case."""
    from datetime import UTC, datetime

    from atp_method.loader import load_case

    from atp.core.results import RunResult, TestResult
    from atp.protocol import ArtifactFile, ATPResponse, ResponseStatus

    case_path = (
        Path(__file__).resolve().parents[3]
        / "method"
        / "cases"
        / "code-review"
        / "case-code-review-sqli-moderate-001.yaml"
    )
    test_def = load_case(case_path)
    response = ATPResponse(
        task_id=test_def.id,
        status=ResponseStatus.COMPLETED,
        artifacts=[ArtifactFile(path="review.md", content="[]")],
    )
    run = RunResult(
        test_id=test_def.id,
        run_number=1,
        response=response,
        events=[],
        end_time=datetime.now(tz=UTC),
    )
    return TestResult(test=test_def, runs=[run])


def test_grade_case_surfaces_continuous_metrics() -> None:
    import anyio

    from atp.evaluators.base import EvalCheck, EvalResult
    from method.run_pipe_check import _grade_case

    class _StubEval:
        async def evaluate(self, test_def, response, events, assertion):  # type: ignore[no-untyped-def]
            return EvalResult(
                evaluator="stub",
                checks=[
                    EvalCheck(
                        name="critical_check",
                        passed=True,
                        score=1.0,
                        details={
                            "malformed": False,
                            "recall": 0.5,
                            "precision": 0.75,
                            "fp_count": 1,
                        },
                    )
                ],
            )

    tr = _completed_test_result()
    base = anyio.run(_grade_case, _StubEval(), tr, "moderate", False)
    assert base["recall"] == 0.5
    assert base["precision"] == 0.75
    assert base["fp_count"] == 1


def test_write_case_details_one_line_per_case(tmp_path: Path) -> None:
    import json

    from method.run_pipe_check import _write_case_details

    case_results = [
        {"case_id": "a", "recall": 0.5, "precision": 0.75, "fp_count": 1},
        {"case_id": "b", "recall": 1.0, "precision": 1.0, "fp_count": 0},
    ]
    out = tmp_path / "case_details_stub.jsonl"
    _write_case_details(out, case_results)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    for line, expected in zip(lines, case_results, strict=True):
        obj = json.loads(line)
        assert obj == expected
        assert "recall" in obj
        assert "precision" in obj
        assert "fp_count" in obj
