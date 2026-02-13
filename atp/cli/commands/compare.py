"""CLI command for multi-model comparison."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml


@click.command(name="compare")
@click.argument(
    "suite_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "config_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--runs",
    type=int,
    default=1,
    help="Number of runs per test (default: 1)",
)
@click.option(
    "--tags",
    type=str,
    help="Filter tests by tags (comma-separated, use ! to exclude)",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on first failure per model",
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
    help="Output file path (for json output)",
)
def compare_command(
    suite_file: Path,
    config_files: tuple[Path, ...],
    runs: int,
    tags: str | None,
    fail_fast: bool,
    output: str,
    output_file: Path | None,
) -> None:
    """Compare a test suite across multiple model configurations.

    SUITE_FILE is the path to a YAML test suite definition.

    CONFIG_FILES are one or more YAML files, each defining a model
    configuration with fields: name, adapter, config.

    Example config file (model_a.yaml):

    \b
        name: gpt-4
        adapter: http
        config:
          base_url: http://localhost:8000

    Examples:

    \b
      # Compare two models
      atp compare tests/suite.yaml model_a.yaml model_b.yaml

    \b
      # Compare with JSON output
      atp compare tests/suite.yaml model_a.yaml model_b.yaml --output=json

    \b
      # Compare with multiple runs
      atp compare tests/suite.yaml model_a.yaml model_b.yaml --runs=3
    """
    try:
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

        _output_results(result, output, output_file)
        sys.exit(0)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)


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


def _output_results(
    result: Any,
    output_format: str,
    output_file: Path | None,
) -> None:
    """Output comparison results."""
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
