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


def _corpus_request(workspace: str) -> dict:
    return {
        "version": "1.0",
        "task_id": "corpus-1",
        "task": {
            "description": "Extract requirements with citations.",
            "input_data": {
                "run_mode": "read_only_corpus",
                "artifact_corpus": {
                    "id": "c1",
                    "files": ["policy-current.md", "archive/policy-2023.md"],
                },
            },
        },
        "context": {"workspace_path": workspace},
        "constraints": {},
    }


def test_corpus_run_sets_cwd_to_workspace(tmp_path) -> None:
    workspace = tmp_path / "corpus"
    workspace.mkdir()
    (workspace / "policy-current.md").write_text("deadline: 2026-08-01\n")
    log_path = tmp_path / "invocation.json"

    resp = _run_shim(
        _corpus_request(str(workspace)),
        extra_env={"ATP_FAKE_CODEX_LOG": str(log_path)},
    )

    invocation = json.loads(log_path.read_text())
    assert invocation["cwd"] == str(workspace)
    # The read-only sandbox (already passed on every run) is the
    # write/exec confinement; cwd is the directory surface.
    assert "--sandbox" in invocation["argv"]
    sandbox_mode = invocation["argv"][invocation["argv"].index("--sandbox") + 1]
    assert sandbox_mode == "read-only"
    assert resp["status"] == "completed"


def test_corpus_run_normalizes_absolute_citation_paths(tmp_path) -> None:
    workspace = tmp_path / "corpus"
    workspace.mkdir()
    log_path = tmp_path / "invocation.json"

    resp = _run_shim(
        _corpus_request(str(workspace)),
        extra_env={"ATP_FAKE_CODEX_LOG": str(log_path)},
    )

    content = resp["artifacts"][0]["content"]
    # fake_codex cites <workspace>/policy-current.md absolutely; the shim
    # must strip the prefix so the grader sees a corpus-relative path.
    assert str(workspace) not in content
    assert "policy-current.md" in content


def test_non_corpus_run_keeps_default_cwd(tmp_path) -> None:
    log_path = tmp_path / "invocation.json"
    request = {
        "version": "1.0",
        "task_id": "t9",
        "task": {"description": "Review the diff against the rules."},
        "constraints": {},
    }
    resp = _run_shim(request, extra_env={"ATP_FAKE_CODEX_LOG": str(log_path)})
    invocation = json.loads(log_path.read_text())
    # No cwd override on non-corpus runs: the fake must inherit the test
    # process's working directory (any other value = corpus branch leaked).
    assert invocation["cwd"] == os.getcwd()
    assert resp["status"] == "completed"
