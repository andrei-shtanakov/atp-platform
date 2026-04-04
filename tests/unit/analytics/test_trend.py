"""Tests for cross-run trend analysis."""

from __future__ import annotations

import json
from pathlib import Path

from atp.analytics.trend import analyze_trend


def _write_report(
    path: Path,
    success_rate: float,
    suite: str = "test-suite",
    agent: str = "test-agent",
) -> None:
    """Write a minimal JSON report file."""
    data = {
        "version": "1.0",
        "generated_at": "2026-01-01T00:00:00",
        "summary": {
            "suite_name": suite,
            "agent_name": agent,
            "total_tests": 10,
            "passed_tests": int(success_rate * 10),
            "success_rate": success_rate,
        },
    }
    path.write_text(json.dumps(data))


class TestAnalyzeTrend:
    """Tests for analyze_trend function."""

    def test_degrading_trend(self, tmp_path: Path) -> None:
        """Detects degrading success_rate slope."""
        rates = [0.95, 0.90, 0.85, 0.80, 0.75]
        paths = []
        for i, rate in enumerate(rates):
            p = tmp_path / f"run-{i:03d}.json"
            _write_report(p, rate)
            paths.append(p)

        report = analyze_trend(paths)
        assert report.direction == "degrading"
        assert report.is_regression is True
        assert report.slope < -0.01

    def test_improving_trend(self, tmp_path: Path) -> None:
        """Detects improving success_rate slope."""
        rates = [0.70, 0.75, 0.80, 0.85, 0.90]
        paths = []
        for i, rate in enumerate(rates):
            p = tmp_path / f"run-{i:03d}.json"
            _write_report(p, rate)
            paths.append(p)

        report = analyze_trend(paths)
        assert report.direction == "improving"
        assert report.is_regression is False
        assert report.slope > 0.01

    def test_stable_trend(self, tmp_path: Path) -> None:
        """Detects stable success_rate (no drift)."""
        rates = [0.85, 0.86, 0.85, 0.84, 0.85]
        paths = []
        for i, rate in enumerate(rates):
            p = tmp_path / f"run-{i:03d}.json"
            _write_report(p, rate)
            paths.append(p)

        report = analyze_trend(paths)
        assert report.direction == "stable"
        assert report.is_regression is False

    def test_single_report(self, tmp_path: Path) -> None:
        """Single report returns stable with no regression."""
        p = tmp_path / "run-000.json"
        _write_report(p, 0.90)

        report = analyze_trend([p])
        assert report.direction == "stable"
        assert report.slope == 0.0
        assert report.is_regression is False

    def test_empty_input(self) -> None:
        """Empty input returns stable."""
        report = analyze_trend([])
        assert report.direction == "stable"
        assert len(report.points) == 0

    def test_window_limits_reports(self, tmp_path: Path) -> None:
        """Window parameter limits the number of reports analyzed."""
        paths = []
        for i in range(20):
            p = tmp_path / f"run-{i:03d}.json"
            _write_report(p, 0.90 - i * 0.01)
            paths.append(p)

        report = analyze_trend(paths, window=5)
        assert len(report.points) == 5

    def test_invalid_report_skipped(self, tmp_path: Path) -> None:
        """Invalid JSON files are skipped gracefully."""
        p1 = tmp_path / "run-000.json"
        _write_report(p1, 0.90)

        p2 = tmp_path / "run-001.txt"
        p2.write_text("not json")

        p3 = tmp_path / "run-002.json"
        _write_report(p3, 0.80)

        report = analyze_trend([p1, p2, p3])
        assert len(report.points) == 2

    def test_custom_threshold(self, tmp_path: Path) -> None:
        """Custom regression threshold changes sensitivity."""
        rates = [0.90, 0.89, 0.88, 0.87, 0.86]
        paths = []
        for i, rate in enumerate(rates):
            p = tmp_path / f"run-{i:03d}.json"
            _write_report(p, rate)
            paths.append(p)

        # With default threshold (0.01), this is degrading
        report = analyze_trend(paths, regression_threshold=0.01)
        assert report.is_regression is True

        # With high threshold, this is stable
        report = analyze_trend(paths, regression_threshold=0.05)
        assert report.is_regression is False

    def test_report_extracts_suite_and_agent(self, tmp_path: Path) -> None:
        """Suite and agent names extracted from first report."""
        p = tmp_path / "run-000.json"
        _write_report(p, 0.90, suite="my-suite", agent="my-agent")
        p2 = tmp_path / "run-001.json"
        _write_report(p2, 0.85)

        report = analyze_trend([p, p2])
        assert report.suite_name == "my-suite"
        assert report.agent_name == "my-agent"

    def test_summary_property(self, tmp_path: Path) -> None:
        """TrendReport.summary returns human-readable string."""
        rates = [0.95, 0.90, 0.85]
        paths = []
        for i, rate in enumerate(rates):
            p = tmp_path / f"run-{i:03d}.json"
            _write_report(p, rate)
            paths.append(p)

        report = analyze_trend(paths)
        assert "DEGRADING" in report.summary
        assert "slope=" in report.summary
