"""Tests for the Robin spawner shim (offline, via a fake robin CLI)."""

import json
import os
import subprocess
import sys
from pathlib import Path

SHIM = Path(__file__).resolve().parents[3] / "method" / "spawners" / "robin_shim.py"
FAKE = Path(__file__).resolve().parent / "fixtures" / "fake_robin.py"


def _run_shim(
    request: dict | str, tmp_path: Path, extra_env: dict | None = None
) -> dict:
    env = {
        **os.environ,
        "ROBIN_BIN": f"{sys.executable} {FAKE}",
        "ROBIN_DIR": str(tmp_path),  # any existing dir: the fake ignores cwd
    }
    if extra_env:
        env.update(extra_env)
    payload = request if isinstance(request, str) else json.dumps(request)
    proc = subprocess.run(
        [sys.executable, str(SHIM)],
        input=payload.encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return json.loads(proc.stdout.decode())


def _request(question: str = "Which repo owns the agents-catalog SSOT?") -> dict:
    return {
        "version": "1.0",
        "task_id": "robin-t1",
        "task": {"description": question},
        "constraints": {},
    }


def test_shim_emits_completed_response_with_answer_artifact(tmp_path: Path) -> None:
    resp = _run_shim(_request(), tmp_path)
    assert resp["task_id"] == "robin-t1"
    assert resp["status"] == "completed"
    (artifact,) = resp["artifacts"]
    assert artifact["path"] == "answer.md"
    assert "agents-catalog.toml" in artifact["content"]
    assert "Grounding sources (2):" in artifact["content"]
    assert "authored/decisions/" in artifact["content"]
    # Robin prints dollars, not tokens.
    assert resp["metrics"]["cost_usd"] == 0.0123
    assert resp["metrics"]["total_tokens"] is None


def test_shim_passes_retrieve_only_output_through(tmp_path: Path) -> None:
    resp = _run_shim(_request(), tmp_path, {"FAKE_ROBIN_RETRIEVE": "1"})
    assert resp["status"] == "completed"
    content = resp["artifacts"][0]["content"]
    assert "Grounding sources (0):" in content  # zero-sources marker for assertions
    assert resp["metrics"]["cost_usd"] is None


def test_shim_fails_on_robin_error(tmp_path: Path) -> None:
    resp = _run_shim(_request(), tmp_path, {"FAKE_ROBIN_FAIL": "1"})
    assert resp["status"] == "failed"
    assert "rc=3" in resp["error"]
    assert "no vault mounted" in resp["error"]


def test_shim_fails_on_empty_question(tmp_path: Path) -> None:
    resp = _run_shim(_request(question="  "), tmp_path)
    assert resp["status"] == "failed"
    assert "task.description" in resp["error"]


def test_shim_fails_on_invalid_stdin(tmp_path: Path) -> None:
    resp = _run_shim("{not json", tmp_path)
    assert resp["status"] == "failed"
    assert "invalid ATPRequest" in resp["error"]


def test_shim_fails_on_missing_robin_dir(tmp_path: Path) -> None:
    resp = _run_shim(_request(), tmp_path, {"ROBIN_DIR": str(tmp_path / "nope")})
    assert resp["status"] == "failed"
    assert "ROBIN_DIR" in resp["error"]
