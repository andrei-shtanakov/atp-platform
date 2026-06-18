"""Unit tests for method/import_pipecheck_to_dashboard.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "method" / "import_pipecheck_to_dashboard.py"
)
_spec = importlib.util.spec_from_file_location("import_pipecheck", _MODULE_PATH)
assert _spec and _spec.loader
imp = importlib.util.module_from_spec(_spec)
# Register before exec: ``from __future__ import annotations`` makes @dataclass
# resolve string annotations via ``sys.modules[cls.__module__]``; without this
# the module is absent and dataclass creation raises AttributeError.
sys.modules[_spec.name] = imp
_spec.loader.exec_module(imp)


def _write_report(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "payload_version": "1.0.0",
        "run_id": "run-abc",
        "benchmark_id": "code-review",
        "agent_id": "claude_code",
        "ts": "2026-06-17T15:30:48.100372+00:00",
        "score": 0.9,
        "score_components": {
            "critical_pass_rate": 0.9,
            "mean_rubric": 0.0,
            "malformed_rate": 0.05,
        },
        "duration_seconds": 42.0,
        "breakpoint_axis_level": "severe",
        "per_task": [{"task_index": 0, "task_type": "review"}],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))
    return path


def test_parse_report_extracts_store_fields(tmp_path: Path) -> None:
    r = imp.parse_report(_write_report(tmp_path / "report_benchmark_claude_code.json"))
    assert r is not None
    assert r.run_uuid == "run-abc"
    assert r.suite_name == "code-review"
    assert r.agent_name == "claude_code"
    assert r.critical_pass_rate == 0.9
    assert r.malformed_rate == 0.05
    assert r.task_type == "review"
    assert r.breakpoint_axis_level == "severe"
    assert isinstance(r.started_at, datetime)


def test_breakpoint_axis_level_absent_is_none(tmp_path: Path) -> None:
    r = imp.parse_report(
        _write_report(tmp_path / "report_benchmark_x.json", breakpoint_axis_level=None)
    )
    assert r is not None
    assert r.breakpoint_axis_level is None


def test_task_type_falls_back_to_benchmark_map(tmp_path: Path) -> None:
    r = imp.parse_report(
        _write_report(
            tmp_path / "report_benchmark_x.json",
            benchmark_id="req-extraction",
            per_task=[],
        )
    )
    assert r is not None
    assert r.task_type == "req-extraction"


def test_parse_report_rejects_incomplete_payload(tmp_path: Path) -> None:
    p = tmp_path / "report_benchmark_bad.json"
    p.write_text(json.dumps({"benchmark_id": "code-review"}))  # no run_id/agent/ts
    assert imp.parse_report(p) is None


def test_parse_report_rejects_non_json(tmp_path: Path) -> None:
    p = tmp_path / "report_benchmark_broken.json"
    p.write_text("not json {")
    assert imp.parse_report(p) is None


def test_discover_reports_is_recursive_and_sorted(tmp_path: Path) -> None:
    _write_report(tmp_path / "report_benchmark_a.json", run_id="a")
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_report(sub / "report_benchmark_b.json", run_id="b")
    (tmp_path / "not_a_report.json").write_text("{}")
    found = imp.discover_reports(tmp_path)
    assert [r.run_uuid for r in found] == ["a", "b"]


def test_main_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_report(tmp_path / "report_benchmark_claude_code.json")
    rc = imp.main(["--results-dir", str(tmp_path), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Found 1 report" in out
    assert "nothing written" in out


def test_main_missing_dir_returns_1(tmp_path: Path) -> None:
    assert imp.main(["--results-dir", str(tmp_path / "nope")]) == 1


def test_main_empty_dir_returns_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert imp.main(["--results-dir", str(tmp_path)]) == 0
    assert "No report_benchmark" in capsys.readouterr().out


@pytest.mark.anyio
async def test_import_reports_idempotent(tmp_path: Path) -> None:
    """Importing twice writes rows once; second pass skips by run_uuid."""
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'dash.db'}"
    _write_report(tmp_path / "report_benchmark_claude_code.json", run_id="run-1")
    _write_report(
        tmp_path / "report_benchmark_anthropic_api.json",
        run_id="run-2",
        agent_id="anthropic_api",
        score_components={
            "critical_pass_rate": 0.7,
            "malformed_rate": 0.0,
            "mean_rubric": 0.0,
        },
    )
    reports = imp.discover_reports(tmp_path)

    imported, skipped = await imp.import_reports(reports, db_url=db_url)
    assert (imported, skipped) == (2, 0)

    imported2, skipped2 = await imp.import_reports(reports, db_url=db_url)
    assert (imported2, skipped2) == (0, 2)

    # Verify the rows render-ready: completed + scored, leaderboard ranks them.
    from atp.dashboard import ResultStorage, init_database

    db = await init_database(url=db_url)
    async with db.session() as session:
        storage = ResultStorage(session)
        board = await storage.suite_leaderboard("code-review")
        agents = {e["agent_name"]: e for e in board}
        assert agents["claude_code"]["critical_pass_rate"] == 0.9
        assert agents["claude_code"]["breakpoint_axis_level"] == "severe"
        assert agents["anthropic_api"]["critical_pass_rate"] == 0.7
        # ranked desc by critical_pass_rate
        assert [e["agent_name"] for e in board] == ["claude_code", "anthropic_api"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
