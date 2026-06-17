"""Tests for the codex_cli spawner shim (offline, via a fake codex binary)."""

import json
import os
import subprocess
import sys
from pathlib import Path

SHIM = Path(__file__).resolve().parents[3] / "method" / "spawners" / "codex_cli_shim.py"
FAKE = Path(__file__).resolve().parent / "fixtures" / "fake_codex.py"


def _run_shim(request: dict, extra_env: dict | None = None) -> dict:
    env = {**os.environ, "CODEX_BIN": f"{sys.executable} {FAKE}"}
    if extra_env:
        env.update(extra_env)
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
    assert arts[0]["path"] == "review.md"
    findings = json.loads(arts[0]["content"])
    assert findings[0]["rule_id"] == "sql-injection"
    assert "SELECT" in findings[0]["anchor"]
    # Token usage is parsed from codex --json events; cost stays unknown (null).
    assert resp["metrics"]["total_tokens"] == 1500
    assert resp["metrics"]["input_tokens"] == 1100
    assert resp["metrics"]["output_tokens"] == 400
    assert resp["metrics"]["cost_usd"] is None


def test_shim_captures_tokens_from_json_events() -> None:
    resp = _run_shim({"task_id": "t1", "task": {"description": "review"}})
    assert resp["status"] == "completed"
    assert resp["metrics"]["input_tokens"] == 1100
    assert resp["metrics"]["output_tokens"] == 400
    assert resp["metrics"]["total_tokens"] == 1500


def test_shim_surfaces_token_breakdown_without_inflating_total() -> None:
    resp = _run_shim({"task_id": "t1", "task": {"description": "review"}})
    m = resp["metrics"]
    # Breakdowns are surfaced for transparency...
    assert m["cached_input_tokens"] == 500
    assert m["reasoning_output_tokens"] == 0
    # ...but total stays input+output (cached is a subset of input;
    # output already includes reasoning per OpenAI convention).
    assert m["total_tokens"] == m["input_tokens"] + m["output_tokens"]
    assert m["total_tokens"] == 1500


def test_shim_failure_path_emits_failed() -> None:
    request = {
        "version": "1.0",
        "task_id": "t2",
        "task": {"description": "Review the diff."},
    }
    resp = _run_shim(request, extra_env={"FAKE_CODEX_FAIL": "1"})
    assert resp["task_id"] == "t2"
    assert resp["status"] == "failed"
    assert resp["artifacts"] == []
    assert "codex exec failed" in resp["error"]
