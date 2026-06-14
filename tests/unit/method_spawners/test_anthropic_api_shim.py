"""Tests for the anthropic_api spawner shim (offline, via a fake anthropic SDK)."""

import json
import os
import subprocess
import sys
from pathlib import Path

SHIM = (
    Path(__file__).resolve().parents[3]
    / "method"
    / "spawners"
    / "anthropic_api_shim.py"
)

# A stand-in `anthropic` module dropped on PYTHONPATH so the shim's lazy
# `import anthropic` resolves to this instead of the real SDK (no network/key).
_FAKE_ANTHROPIC = """
import json as _json

_FINDINGS = _json.dumps(
    [{"rule_id": "sql-injection", "anchor": 'f"SELECT', "severity": "critical"}]
)


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Usage:
    input_tokens = 800
    output_tokens = 120


class _Msg:
    content = [_Block(_FINDINGS)]
    usage = _Usage()


class _Messages:
    def create(self, *a, **k):
        return _Msg()


class Anthropic:
    def __init__(self, *a, **k):
        self.messages = _Messages()
"""


def _run_shim(request: dict, env: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, str(SHIM)],
        input=json.dumps(request).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return json.loads(proc.stdout.decode())


def test_missing_key_emits_failed() -> None:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    resp = _run_shim(
        {"version": "1.0", "task_id": "t1", "task": {"description": "x"}}, env
    )
    assert resp["status"] == "failed"
    assert "ANTHROPIC_API_KEY" in resp["error"]


def test_success_with_fake_sdk(tmp_path: Path) -> None:
    (tmp_path / "anthropic.py").write_text(_FAKE_ANTHROPIC)
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "test-key",
        "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    request = {
        "version": "1.0",
        "task_id": "t2",
        "task": {"description": "Review the diff against the rules."},
        "context": {"artifacts": []},
    }
    resp = _run_shim(request, env)
    assert resp["task_id"] == "t2"
    assert resp["status"] == "completed"
    arts = resp["artifacts"]
    assert len(arts) == 1
    findings = json.loads(arts[0]["content"])
    assert findings[0]["rule_id"] == "sql-injection"
    assert "SELECT" in findings[0]["anchor"]
    assert resp["metrics"]["total_tokens"] == 920
    # raw API response carries no cost field; the baseline leaves it null
    assert resp["metrics"]["cost_usd"] is None
