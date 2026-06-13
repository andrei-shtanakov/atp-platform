"""Tests for the claude_code spawner shim (offline, via a fake claude)."""

import json
import os
import subprocess
import sys
from pathlib import Path

SHIM = (
    Path(__file__).resolve().parents[3] / "method" / "spawners" / "claude_code_shim.py"
)
FAKE = Path(__file__).resolve().parent / "fixtures" / "fake_claude.py"


def _run_shim(request: dict) -> dict:
    env = {**os.environ, "CLAUDE_BIN": f"{sys.executable} {FAKE}"}
    proc = subprocess.run(
        [sys.executable, str(SHIM)],
        input=json.dumps(request).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return json.loads(proc.stdout.decode())


def test_shim_emits_valid_atp_response_with_review_artifact() -> None:
    request = {
        "version": "1.0",
        "task_id": "t1",
        "task": {"description": "Review the diff against the rules."},
        "constraints": {},
    }
    resp = _run_shim(request)
    assert resp["task_id"] == "t1"
    assert resp["status"] == "completed"
    arts = resp["artifacts"]
    assert len(arts) == 1
    assert "SEC-011" in arts[0]["content"]
    assert resp["metrics"]["total_tokens"] == 920
    assert resp["metrics"]["cost_usd"] == 0.0123
