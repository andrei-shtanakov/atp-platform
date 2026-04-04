"""Cross-run trend analysis for detecting gradual behavioral drift.

Reads sequential JSON report files and computes OLS slope over
success_rate to detect regressions invisible to single-run statistics.

Complements within-run Welch's t-test (point-in-time variance) by
catching directional drift across runs (e.g. 0.92 → 0.85 → 0.78).
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SuiteRunPoint:
    """A single data point from a suite run JSON report."""

    run_index: int
    success_rate: float
    passed_tests: int
    total_tests: int
    generated_at: str


@dataclass
class TrendReport:
    """Result of cross-run trend analysis."""

    suite_name: str
    agent_name: str
    window: int
    points: list[SuiteRunPoint]
    slope: float
    direction: str  # "improving" | "degrading" | "stable"
    is_regression: bool

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        if len(self.points) < 2:
            return "Insufficient data (need at least 2 runs)"
        rates = [p.success_rate for p in self.points]
        return (
            f"{self.direction.upper()}: slope={self.slope:+.4f}/run "
            f"over {len(self.points)} runs "
            f"(range: {min(rates):.2f}–{max(rates):.2f})"
        )


def _load_report(path: Path) -> dict | None:
    """Load and validate a JSON report file."""
    try:
        data = json.loads(path.read_text())
        summary = data.get("summary", {})
        if "success_rate" not in summary:
            logger.warning("No success_rate in %s, skipping", path)
            return None
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def analyze_trend(
    report_paths: list[Path],
    window: int = 10,
    regression_threshold: float = 0.01,
) -> TrendReport:
    """Analyze success_rate trend across sequential JSON reports.

    Args:
        report_paths: Paths to JSON report files, in chronological order.
        window: Maximum number of most recent reports to consider.
        regression_threshold: Minimum negative slope to flag as regression.

    Returns:
        TrendReport with slope, direction, and regression flag.
    """
    paths = sorted(report_paths)[-window:]

    points: list[SuiteRunPoint] = []
    suite_name = ""
    agent_name = ""

    for i, path in enumerate(paths):
        data = _load_report(path)
        if data is None:
            continue
        summary = data["summary"]
        if not suite_name:
            suite_name = summary.get("suite_name", "")
        if not agent_name:
            agent_name = summary.get("agent_name", "")
        points.append(
            SuiteRunPoint(
                run_index=i,
                success_rate=summary["success_rate"],
                passed_tests=summary.get("passed_tests", 0),
                total_tests=summary.get("total_tests", 0),
                generated_at=data.get("generated_at", ""),
            )
        )

    if len(points) < 2:
        return TrendReport(
            suite_name=suite_name,
            agent_name=agent_name,
            window=window,
            points=points,
            slope=0.0,
            direction="stable",
            is_regression=False,
        )

    xs = [p.run_index for p in points]
    ys = [p.success_rate for p in points]
    slope, _intercept = statistics.linear_regression(xs, ys)

    if slope < -regression_threshold:
        direction = "degrading"
    elif slope > regression_threshold:
        direction = "improving"
    else:
        direction = "stable"

    return TrendReport(
        suite_name=suite_name,
        agent_name=agent_name,
        window=window,
        points=points,
        slope=slope,
        direction=direction,
        is_regression=slope < -regression_threshold,
    )
