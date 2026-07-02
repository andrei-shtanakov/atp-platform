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
    import json as _json

    findings = _json.loads(arts[0]["content"])
    assert findings[0]["rule_id"] == "sql-injection"
    assert "SELECT" in findings[0]["anchor"]
    # total_tokens MUST include cache classes — Claude Code caches the system
    # prompt/tools/context, so raw input_tokens is only the non-cached delta.
    # 800 (input) + 1000 (cache_creation) + 5000 (cache_read) + 120 (output).
    metrics = resp["metrics"]
    assert metrics["total_tokens"] == 6920
    assert metrics["input_tokens"] == 800
    assert metrics["output_tokens"] == 120
    assert metrics["cache_creation_tokens"] == 1000
    assert metrics["cache_read_tokens"] == 5000
    assert metrics["cost_usd"] == 0.0123

    # The cache fields must survive the protocol boundary (Metrics has no
    # extra="forbid", but it also drops unknown keys — so they must be real
    # fields, not silently discarded when the adapter parses the response).
    from atp.protocol.models import Metrics

    parsed = Metrics.model_validate(metrics)
    assert parsed.total_tokens == 6920
    assert parsed.cache_creation_tokens == 1000
    assert parsed.cache_read_tokens == 5000
