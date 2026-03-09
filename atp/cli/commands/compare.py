"""CLI command for multi-model comparison.

Supports two modes:
1. Live comparison: run a suite against multiple model configs
   atp compare suite.yaml model_a.yaml model_b.yaml

2. Results comparison: compare pre-saved JSON result directories
   atp compare results/openai results/anthropic
   atp compare results/  --output=json --output-file=comparison.json
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml


@click.command(name="compare")
@click.argument(
    "paths",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--runs",
    type=int,
    default=1,
    help="Number of runs per test (live mode, default: 1)",
)
@click.option(
    "--tags",
    type=str,
    help="Filter tests by tags (comma-separated, use ! to exclude)",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on first failure per model (live mode)",
)
@click.option(
    "--output",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format (console or json)",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    help="Output file path",
)
@click.option(
    "--sort-by",
    type=click.Choice(["score", "name", "cost", "duration"]),
    default="name",
    help="Sort agents by metric (results mode)",
)
def compare_command(
    paths: tuple[Path, ...],
    runs: int,
    tags: str | None,
    fail_fast: bool,
    output: str,
    output_file: Path | None,
    sort_by: str,
) -> None:
    """Compare test results across multiple agents.

    PATHS can be either:

    \b
    1. Result directories (from --save-results):
       atp compare results/openai results/anthropic
       atp compare results/   (all subdirectories)

    \b
    2. A suite file + model config files (live comparison):
       atp compare suite.yaml model_a.yaml model_b.yaml

    Examples:

    \b
      # Compare saved results
      atp compare results/gpt-4o-mini results/claude-haiku

    \b
      # Compare all agents in a directory
      atp compare results/ --output=json --output-file=comparison.json

    \b
      # Live comparison of two models
      atp compare tests/suite.yaml model_a.yaml model_b.yaml --runs=3
    """
    try:
        # Detect mode: if first path is a YAML file, it's live mode
        if paths[0].is_file() and paths[0].suffix in (".yaml", ".yml"):
            _run_live_mode(paths, runs, tags, fail_fast, output, output_file)
        else:
            _run_results_mode(paths, output, output_file, sort_by)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)


# ── Results comparison mode ──────────────────────────────────────


def _run_results_mode(
    paths: tuple[Path, ...],
    output_format: str,
    output_file: Path | None,
    sort_by: str,
) -> None:
    """Compare pre-saved JSON result directories."""
    agent_dirs = _resolve_agent_dirs(paths)
    if not agent_dirs:
        click.echo("No result directories found.", err=True)
        sys.exit(1)

    agents: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, float | int]] = {}

    for d in agent_dirs:
        name = d.name
        raw = _load_agent_results(d)
        tests = _extract_test_scores(raw)
        agents[name] = tests
        summaries[name] = _compute_summary(tests)

    # Sort agents
    agent_names = sorted(
        agents.keys(),
        key=lambda n: _sort_key(summaries[n], sort_by),
        reverse=sort_by in ("score", "cost"),
    )

    if output_format == "json":
        data = _build_comparison_json(agents, summaries, agent_names)
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json_str)
            click.echo(f"Results written to {output_file}")
        else:
            click.echo(json_str)
    else:
        _print_comparison_table(agents, summaries, agent_names)
        if output_file:
            data = _build_comparison_json(agents, summaries, agent_names)
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json_str)
            click.echo(f"\nJSON saved to {output_file}")


def _resolve_agent_dirs(paths: tuple[Path, ...]) -> list[Path]:
    """Resolve paths to agent result directories."""
    dirs: list[Path] = []
    for p in paths:
        if p.is_file():
            continue
        # Check if this is an agent dir (has JSON result files)
        if any(p.glob("*.json")):
            dirs.append(p)
        else:
            # Scan subdirectories
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and any(sub.glob("*.json")):
                    dirs.append(sub)
    return dirs


def _load_agent_results(agent_dir: Path) -> dict[str, Any]:
    """Load all JSON result files from an agent directory."""
    results: dict[str, Any] = {}
    for f in sorted(agent_dir.glob("*.json")):
        if f.name in ("metadata.json", "summary.json"):
            continue
        with open(f) as fh:
            results[f.stem] = json.load(fh)
    return results


def _extract_test_scores(
    agent_results: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract per-test scores from agent results."""
    tests: dict[str, dict[str, Any]] = {}
    for suite_name, suite_data in agent_results.items():
        for test in suite_data.get("tests", []):
            test_id = test["test_id"]
            evals = test.get("evaluations", [])
            eval_detail = {}
            for e in evals:
                status = "PASS" if e["passed"] else "FAIL"
                eval_detail[e["evaluator"]] = status

            breakdown = test.get("score_breakdown", {})
            components = breakdown.get("components", {})

            tests[test_id] = {
                "suite": suite_name,
                "name": test["test_name"],
                "score": test.get("score"),
                "passed": test["success"],
                "duration": test.get("duration_seconds", 0),
                "evaluations": eval_detail,
                "quality": components.get("quality", {}).get("normalized"),
                "completeness": components.get("completeness", {}).get("normalized"),
                "efficiency": components.get("efficiency", {}).get("normalized"),
                "cost": components.get("cost", {}).get("normalized"),
            }
    return tests


def _compute_summary(
    tests: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    """Compute aggregate metrics."""
    scores = [t["score"] for t in tests.values() if t["score"] is not None]
    passed = sum(1 for t in tests.values() if t["passed"])
    total_duration = sum(t["duration"] for t in tests.values())

    dims = ["quality", "completeness", "efficiency", "cost"]
    dim_avgs: dict[str, float] = {}
    for dim in dims:
        vals = [t[dim] for t in tests.values() if t[dim] is not None]
        dim_avgs[dim] = round(sum(vals) / len(vals), 3) if vals else 0

    return {
        "total_tests": len(tests),
        "passed": passed,
        "pass_rate": round(passed / len(tests), 3) if tests else 0,
        "avg_score": (round(sum(scores) / len(scores), 1) if scores else 0),
        "min_score": round(min(scores), 1) if scores else 0,
        "max_score": round(max(scores), 1) if scores else 0,
        "total_duration": round(total_duration, 1),
        **dim_avgs,
    }


def _sort_key(summary: dict[str, float | int], sort_by: str) -> float | int | str:
    """Get sort key for an agent summary."""
    if sort_by == "score":
        return summary.get("avg_score", 0)
    if sort_by == "cost":
        return summary.get("cost", 0)
    if sort_by == "duration":
        return -summary.get("total_duration", 0)
    return 0


def _print_comparison_table(
    agents: dict[str, dict[str, dict[str, Any]]],
    summaries: dict[str, dict[str, float | int]],
    agent_names: list[str],
) -> None:
    """Print comparison table to stdout."""
    # Summary table
    click.echo("\n## Summary\n")
    header = "| Metric |"
    separator = "|--------|"
    for name in agent_names:
        header += f" {name} |"
        separator += "--------|"
    click.echo(header)
    click.echo(separator)

    metrics = [
        ("Tests passed", "passed", "total_tests"),
        ("Pass rate", "pass_rate", None),
        ("Avg score", "avg_score", None),
        ("Min score", "min_score", None),
        ("Max score", "max_score", None),
        ("Duration (s)", "total_duration", None),
        ("Quality", "quality", None),
        ("Completeness", "completeness", None),
        ("Efficiency", "efficiency", None),
        ("Cost", "cost", None),
    ]

    for label, key, total_key in metrics:
        row = f"| {label} |"
        for name in agent_names:
            s = summaries[name]
            if total_key:
                row += f" {s[key]}/{s[total_key]} |"
            elif isinstance(s[key], float) and s[key] <= 1.0:
                row += f" {s[key]:.3f} |"
            else:
                row += f" {s[key]} |"
        click.echo(row)

    # Per-test table
    all_test_ids = sorted({tid for a in agents.values() for tid in a})

    click.echo("\n## Per-test scores\n")
    header = "| Test |"
    separator = "|------|"
    for name in agent_names:
        header += f" {name} |"
        separator += "--------|"
    click.echo(header)
    click.echo(separator)

    for tid in all_test_ids:
        test_name = ""
        for a in agents.values():
            if tid in a:
                test_name = a[tid]["name"][:30]
                break

        row = f"| {tid} {test_name} |"
        for name in agent_names:
            if tid in agents[name]:
                t = agents[name][tid]
                score = t["score"]
                mark = "+" if t["passed"] else "x"
                score_str = f"{score:.0f}" if score is not None else "N/A"
                row += f" {mark} {score_str} |"
            else:
                row += " — |"
        click.echo(row)


def _build_comparison_json(
    agents: dict[str, dict[str, dict[str, Any]]],
    summaries: dict[str, dict[str, float | int]],
    agent_names: list[str],
) -> dict[str, Any]:
    """Build comparison JSON structure."""
    comparison: dict[str, Any] = {
        "agents": agent_names,
        "summaries": {n: summaries[n] for n in agent_names},
        "per_test": {},
    }
    all_ids = sorted({tid for a in agents.values() for tid in a})
    for tid in all_ids:
        comparison["per_test"][tid] = {}
        for name in agent_names:
            if tid in agents[name]:
                comparison["per_test"][tid][name] = agents[name][tid]
    return comparison


# ── Live comparison mode ─────────────────────────────────────────


def _run_live_mode(
    paths: tuple[Path, ...],
    runs: int,
    tags: str | None,
    fail_fast: bool,
    output_format: str,
    output_file: Path | None,
) -> None:
    """Run live comparison with model config files."""
    suite_file = paths[0]
    config_files = paths[1:]
    if not config_files:
        raise click.ClickException(
            "Live mode requires at least one model config file after the suite file."
        )

    configs = _load_model_configs(config_files)
    result = asyncio.run(
        _run_comparison(
            suite_file=suite_file,
            configs=configs,
            runs_per_test=runs,
            tag_filter=tags,
            fail_fast=fail_fast,
        )
    )
    _output_live_results(result, output_format, output_file)
    sys.exit(0)


def _load_model_configs(
    config_files: tuple[Path, ...],
) -> list[dict[str, Any]]:
    """Load model configurations from YAML files."""
    configs: list[dict[str, Any]] = []
    for path in config_files:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise click.ClickException(
                f"Invalid model config in {path}: expected a YAML mapping"
            )
        if "name" not in data:
            data["name"] = path.stem
        if "adapter" not in data:
            raise click.ClickException(f"Missing 'adapter' field in {path}")
        configs.append(data)
    return configs


async def _run_comparison(
    suite_file: Path,
    configs: list[dict[str, Any]],
    runs_per_test: int,
    tag_filter: str | None,
    fail_fast: bool,
) -> Any:
    """Run the comparison."""
    from atp.sdk.compare import ModelConfig, acompare

    model_configs = [ModelConfig(**c) for c in configs]
    return await acompare(
        suite=suite_file,
        configs=model_configs,
        runs_per_test=runs_per_test,
        tag_filter=tag_filter,
        fail_fast=fail_fast,
    )


def _output_live_results(
    result: Any,
    output_format: str,
    output_file: Path | None,
) -> None:
    """Output live comparison results."""
    from atp.sdk.compare import (
        format_comparison_json,
        format_comparison_table,
    )

    if output_format == "json":
        data = format_comparison_json(result)
        json_str = json.dumps(data, indent=2, default=str)
        if output_file:
            output_file.write_text(json_str)
            click.echo(f"Results written to {output_file}")
        else:
            click.echo(json_str)
    else:
        table = format_comparison_table(result)
        click.echo(table)
        if output_file:
            output_file.write_text(table)
            click.echo(f"\nResults also written to {output_file}")
