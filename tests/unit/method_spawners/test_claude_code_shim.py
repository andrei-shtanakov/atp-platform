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


def _run_shim(request: dict, extra_env: dict[str, str] | None = None) -> dict:
    env = {**os.environ, "CLAUDE_BIN": f"{sys.executable} {FAKE}", **(extra_env or {})}
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


def test_corpus_run_sets_cwd_and_confinement_flags(tmp_path) -> None:
    workspace = tmp_path / "corpus"
    workspace.mkdir()
    (workspace / "policy-current.md").write_text("deadline: 2026-08-01\n")
    log_path = tmp_path / "invocation.json"

    resp = _run_shim(
        _corpus_request(str(workspace)),
        extra_env={"ATP_FAKE_CLAUDE_LOG": str(log_path)},
    )

    invocation = json.loads(log_path.read_text())
    assert invocation["cwd"] == str(workspace)
    argv = invocation["argv"]
    assert "--allowed-tools" in argv
    assert argv[argv.index("--allowed-tools") + 1] == "Read,Glob,Grep"
    assert resp["status"] == "completed"


def test_corpus_run_normalizes_absolute_citation_paths(tmp_path) -> None:
    workspace = tmp_path / "corpus"
    workspace.mkdir()
    log_path = tmp_path / "invocation.json"

    resp = _run_shim(
        _corpus_request(str(workspace)),
        extra_env={"ATP_FAKE_CLAUDE_LOG": str(log_path)},
    )

    content = resp["artifacts"][0]["content"]
    # fake_claude cites <workspace>/policy-current.md absolutely; the shim
    # must strip the prefix so the grader sees a corpus-relative path.
    assert str(workspace) not in content
    assert "policy-current.md" in content


def test_non_corpus_run_has_no_confinement_flags(tmp_path) -> None:
    log_path = tmp_path / "invocation.json"
    request = {
        "version": "1.0",
        "task_id": "t2",
        "task": {"description": "Review the diff against the rules."},
        "constraints": {},
    }
    resp = _run_shim(request, extra_env={"ATP_FAKE_CLAUDE_LOG": str(log_path)})
    invocation = json.loads(log_path.read_text())
    assert "--allowed-tools" not in invocation["argv"]
    assert resp["status"] == "completed"
