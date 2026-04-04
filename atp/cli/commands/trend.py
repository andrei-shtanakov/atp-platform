"""CLI command for cross-run trend analysis."""

from __future__ import annotations

from pathlib import Path

import click


@click.command(name="trend")
@click.argument(
    "reports",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--window",
    type=int,
    default=10,
    help="Number of most recent reports to analyze.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.01,
    help="Minimum slope magnitude to flag as regression.",
)
@click.option(
    "--exit-on-regression",
    is_flag=True,
    default=False,
    help="Exit with code 1 if regression detected (for CI gates).",
)
@click.option(
    "--output",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format.",
)
def trend_command(
    reports: tuple[Path, ...],
    window: int,
    threshold: float,
    exit_on_regression: bool,
    output: str,
) -> None:
    """Analyze success_rate trend across sequential JSON reports.

    Detects gradual behavioral drift invisible to single-run statistics.

    Examples:

        atp trend reports/run-*.json

        atp trend reports/*.json --window 20 --exit-on-regression
    """
    from atp.analytics.trend import analyze_trend

    report = analyze_trend(
        report_paths=list(reports),
        window=window,
        regression_threshold=threshold,
    )

    if output == "json":
        import json

        data = {
            "suite_name": report.suite_name,
            "agent_name": report.agent_name,
            "window": report.window,
            "slope": report.slope,
            "direction": report.direction,
            "is_regression": report.is_regression,
            "points": [
                {
                    "run_index": p.run_index,
                    "success_rate": p.success_rate,
                    "passed_tests": p.passed_tests,
                    "total_tests": p.total_tests,
                    "generated_at": p.generated_at,
                }
                for p in report.points
            ],
        }
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"Suite: {report.suite_name or '(unknown)'}")
        click.echo(f"Agent: {report.agent_name or '(unknown)'}")
        click.echo(f"Trend: {report.summary}")
        if report.points:
            click.echo(f"Runs analyzed: {len(report.points)}")
            for p in report.points:
                marker = ""
                if p == report.points[-1] and report.is_regression:
                    marker = " ← regression"
                click.echo(
                    f"  [{p.run_index}] success_rate={p.success_rate:.4f} "
                    f"({p.passed_tests}/{p.total_tests}){marker}"
                )

    if exit_on_regression and report.is_regression:
        raise SystemExit(1)
