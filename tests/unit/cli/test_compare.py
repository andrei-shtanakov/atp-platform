"""Tests for the compare CLI command."""

import json
from pathlib import Path

from click.testing import CliRunner

from atp.cli.commands.compare import (
    _build_comparison_json,
    _compute_summary,
    _extract_test_scores,
    _resolve_agent_dirs,
    compare_command,
)


def _make_result_file(tmp_path: Path, agent: str, suite: str) -> Path:
    """Create a mock JSON result file."""
    agent_dir = tmp_path / agent
    agent_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "tests": [
            {
                "test_id": "t1",
                "test_name": "basic test",
                "success": True,
                "score": 85.0,
                "duration_seconds": 1.5,
                "evaluations": [{"evaluator": "artifact", "passed": True}],
                "score_breakdown": {
                    "components": {
                        "quality": {"normalized": 0.9},
                        "completeness": {"normalized": 0.8},
                        "efficiency": {"normalized": 0.7},
                        "cost": {"normalized": 0.6},
                    }
                },
            },
            {
                "test_id": "t2",
                "test_name": "harder test",
                "success": False,
                "score": 40.0,
                "duration_seconds": 3.0,
                "evaluations": [{"evaluator": "behavior", "passed": False}],
                "score_breakdown": {"components": {}},
            },
        ]
    }
    path = agent_dir / f"{suite}.json"
    path.write_text(json.dumps(data))
    return agent_dir


class TestResolveAgentDirs:
    def test_direct_agent_dirs(self, tmp_path: Path) -> None:
        d1 = _make_result_file(tmp_path, "agent_a", "suite1")
        d2 = _make_result_file(tmp_path, "agent_b", "suite1")
        result = _resolve_agent_dirs((d1, d2))
        assert len(result) == 2

    def test_parent_dir_scan(self, tmp_path: Path) -> None:
        _make_result_file(tmp_path, "agent_a", "suite1")
        _make_result_file(tmp_path, "agent_b", "suite1")
        result = _resolve_agent_dirs((tmp_path,))
        assert len(result) == 2

    def test_skips_files(self, tmp_path: Path) -> None:
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        result = _resolve_agent_dirs((f,))
        assert len(result) == 0

    def test_empty_dirs_skipped(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_agent"
        empty.mkdir()
        result = _resolve_agent_dirs((tmp_path,))
        assert len(result) == 0


class TestExtractTestScores:
    def test_extracts_scores(self) -> None:
        raw = {
            "suite1": {
                "tests": [
                    {
                        "test_id": "t1",
                        "test_name": "test one",
                        "success": True,
                        "score": 90.0,
                        "duration_seconds": 2.0,
                        "evaluations": [{"evaluator": "artifact", "passed": True}],
                        "score_breakdown": {"components": {}},
                    }
                ]
            }
        }
        tests = _extract_test_scores(raw)
        assert "t1" in tests
        assert tests["t1"]["score"] == 90.0
        assert tests["t1"]["passed"] is True
        assert tests["t1"]["evaluations"] == {"artifact": "PASS"}


class TestComputeSummary:
    def test_computes_metrics(self) -> None:
        tests = {
            "t1": {
                "score": 80.0,
                "passed": True,
                "duration": 1.0,
                "quality": 0.9,
                "completeness": 0.8,
                "efficiency": 0.7,
                "cost": 0.6,
            },
            "t2": {
                "score": 60.0,
                "passed": False,
                "duration": 2.0,
                "quality": 0.5,
                "completeness": None,
                "efficiency": None,
                "cost": None,
            },
        }
        s = _compute_summary(tests)
        assert s["total_tests"] == 2
        assert s["passed"] == 1
        assert s["pass_rate"] == 0.5
        assert s["avg_score"] == 70.0
        assert s["total_duration"] == 3.0


class TestBuildComparisonJson:
    def test_builds_structure(self) -> None:
        agents = {
            "a": {"t1": {"score": 80, "passed": True}},
            "b": {"t1": {"score": 60, "passed": False}},
        }
        summaries = {
            "a": {"avg_score": 80},
            "b": {"avg_score": 60},
        }
        result = _build_comparison_json(agents, summaries, ["a", "b"])
        assert result["agents"] == ["a", "b"]
        assert "t1" in result["per_test"]
        assert result["per_test"]["t1"]["a"]["score"] == 80


class TestCompareCommandCLI:
    def test_results_mode_console(self, tmp_path: Path) -> None:
        _make_result_file(tmp_path, "agent_a", "suite1")
        _make_result_file(tmp_path, "agent_b", "suite1")

        runner = CliRunner()
        result = runner.invoke(
            compare_command,
            [str(tmp_path / "agent_a"), str(tmp_path / "agent_b")],
        )
        assert result.exit_code == 0
        assert "Summary" in result.output
        assert "agent_a" in result.output
        assert "agent_b" in result.output

    def test_results_mode_json(self, tmp_path: Path) -> None:
        _make_result_file(tmp_path, "agent_a", "suite1")
        out_file = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            compare_command,
            [
                str(tmp_path / "agent_a"),
                "--output=json",
                f"--output-file={out_file}",
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "agents" in data
        assert "agent_a" in data["agents"]

    def test_parent_dir_mode(self, tmp_path: Path) -> None:
        _make_result_file(tmp_path, "agent_a", "suite1")
        _make_result_file(tmp_path, "agent_b", "suite1")

        runner = CliRunner()
        result = runner.invoke(compare_command, [str(tmp_path)])
        assert result.exit_code == 0
        assert "Summary" in result.output

    def test_no_results_dirs(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        runner = CliRunner()
        result = runner.invoke(compare_command, [str(empty)])
        assert result.exit_code != 0
