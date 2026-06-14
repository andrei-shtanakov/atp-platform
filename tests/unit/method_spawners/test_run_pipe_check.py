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
