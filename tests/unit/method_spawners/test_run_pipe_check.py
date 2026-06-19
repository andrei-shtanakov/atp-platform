"""Tests for the pipe-check harness CLI error paths (Phase A-2)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

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
    proc = _run(
        ["--agents", "claude_code@claude-opus-4-8", "--task-type", "bogus", "--dry-run"]
    )
    assert proc.returncode == 2
    err = proc.stderr.decode()
    assert "unknown task_type" in err
    assert "Traceback" not in err


def test_unknown_agent_exits_2() -> None:
    proc = _run(["--agents", "nope", "--task-type", "review", "--dry-run"])
    assert proc.returncode == 2
    assert "Unknown agent" in proc.stderr.decode()


def test_dashboard_replace_without_to_dashboard_exits_2() -> None:
    # --dashboard-replace alone is a misconfiguration: fail fast (exit 2),
    # not silently no-op.
    proc = _run(["--dashboard-replace", "--agents", "claude_code", "--dry-run"])
    assert proc.returncode == 2
    err = proc.stderr.decode()
    assert "--dashboard-replace requires --to-dashboard" in err
    assert "Traceback" not in err


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


def test_export_to_dashboard_imports_written_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--to-dashboard wiring reuses the bridge to land reports in the store."""
    import json

    import anyio

    from method.run_pipe_check import _export_to_dashboard

    # Point the bridge's default DB at a throwaway sqlite (init_database reads
    # ATP_DATABASE_URL when no url is passed) so the dev DB is untouched.
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    monkeypatch.setenv("ATP_DATABASE_URL", db_url)
    (tmp_path / "report_benchmark_claude_code.json").write_text(
        json.dumps(
            {
                "payload_version": "1.0.0",
                "run_id": "rpc-run-1",
                "benchmark_id": "code-review",
                "agent_id": "claude_code",
                "ts": "2026-06-19T10:00:00+00:00",
                "score": 0.9,
                "score_components": {"critical_pass_rate": 0.9},
                "duration_seconds": 1.0,
            }
        )
    )

    path_before = list(sys.path)
    anyio.run(_export_to_dashboard, tmp_path, False)
    assert sys.path == path_before  # helper must not mutate global sys.path

    from sqlalchemy import select

    from atp.dashboard import init_database
    from atp.dashboard.models import SuiteExecution

    async def _check() -> list[str]:
        db = await init_database(url=db_url)
        async with db.session() as session:
            return list(
                (await session.execute(select(SuiteExecution.run_uuid))).scalars().all()
            )

    assert anyio.run(_check) == ["rpc-run-1"]


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


def test_agents_registry_builds_harness_at_model_ids() -> None:
    from method.run_pipe_check import AGENTS

    assert "claude_code@claude-opus-4-8" in AGENTS
    assert "anthropic_api@claude-opus-4-8" in AGENTS
    assert "deepseek@deepseek-chat" in AGENTS
    assert "ollama@qwen2.5:14b" in AGENTS
    # codex is NOT in the default matrix (no pinned model)
    assert not any(a.startswith("codex_cli@") for a in AGENTS)
    spec = AGENTS["ollama@qwen2.5:14b"]
    assert spec["model"] == "qwen2.5:14b"
    assert spec["model_env"] == "OLLAMA_MODEL"
    assert spec["harness"] == "ollama"
    assert spec["shim"].endswith("ollama_shim.py")


def test_safe_agent_id_renders_filesystem_safe() -> None:
    from method.run_pipe_check import safe_agent_id

    assert safe_agent_id("ollama@qwen2.5:14b") == "ollama_qwen2_5_14b"
    assert safe_agent_id("claude_code@claude-opus-4-8") == "claude_code_claude-opus-4-8"


def test_legacy_bare_harness_id_is_unknown() -> None:
    # The old harness-only id no longer resolves; must exit 2.
    proc = _run(["--agents", "claude_code", "--task-type", "review", "--dry-run"])
    assert proc.returncode == 2
    assert "Unknown agent" in proc.stderr.decode()
