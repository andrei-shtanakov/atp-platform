"""CLI command for pre-run cost estimation."""

import json
import sys
from decimal import Decimal
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from atp.analytics.estimator import CostEstimate

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2


def _create_estimate_table(title: str = "Cost Estimate") -> Table:
    """Create a Rich table for estimate display.

    Args:
        title: Table title.

    Returns:
        Configured Rich Table instance.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Test ID", style="green", no_wrap=True)
    table.add_column("Test Name", style="dim")
    table.add_column("Input Tokens", style="blue", justify="right")
    table.add_column("Output Tokens (min-max)", style="magenta", justify="right")
    table.add_column("Runs", style="dim", justify="right")
    table.add_column("Cost Range (USD)", style="yellow", justify="right")
    return table


@click.command(name="estimate")
@click.argument("suite_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Model name for pricing (e.g., gpt-4o, claude-3-5-sonnet-20241022)",
)
@click.option(
    "--provider",
    "-p",
    type=str,
    default="",
    help="Provider name for fallback pricing (anthropic, openai, google)",
)
@click.option(
    "--runs",
    "-r",
    type=int,
    default=None,
    help="Number of runs per test (default: from suite defaults)",
)
@click.option(
    "--tags",
    type=str,
    help="Filter tests by tags (comma-separated, use ! to exclude)",
)
@click.option(
    "--pricing-config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom pricing YAML configuration",
)
@click.option(
    "--budget-check",
    type=float,
    default=None,
    help="Budget limit in USD; abort with exit code 1 if estimate exceeds it",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
def estimate_command(
    suite_file: Path,
    model: str,
    provider: str,
    runs: int | None,
    tags: str | None,
    pricing_config: Path | None,
    budget_check: float | None,
    output: str,
) -> None:
    """Estimate LLM API costs before running a test suite.

    Analyzes test definitions to estimate token counts and applies
    model pricing to produce a cost range (min/max).

    SUITE_FILE is the path to a YAML test suite definition.

    Examples:

      # Estimate costs for a suite with GPT-4o
      atp estimate tests/suite.yaml --model=gpt-4o

      # Estimate with custom runs
      atp estimate tests/suite.yaml --model=claude-3-5-sonnet-20241022 --runs=5

      # Check against a budget
      atp estimate tests/suite.yaml --model=gpt-4o --budget-check=10.00

      # Output as JSON
      atp estimate tests/suite.yaml --model=gpt-4o --output=json

    Exit Codes:

      0 - Estimation successful (and within budget if --budget-check)
      1 - Estimate exceeds budget
      2 - Error occurred
    """
    console = Console()

    try:
        from atp.analytics.cost import PricingConfig
        from atp.analytics.estimator import CostEstimator
        from atp.loader import TestLoader

        # Load pricing config
        if pricing_config:
            pricing = PricingConfig.from_yaml(pricing_config)
        else:
            pricing = PricingConfig.default()

        # Load and optionally filter suite
        loader = TestLoader()
        suite = loader.load_file(suite_file)
        if tags:
            suite = suite.filter_by_tags(tags)

        if not suite.tests:
            click.echo("No tests match the specified criteria.", err=True)
            sys.exit(EXIT_ERROR)

        # Run estimation
        estimator = CostEstimator(pricing_config=pricing)
        estimate = estimator.estimate_suite(
            suite,
            model=model,
            provider=provider,
            runs_per_test=runs,
        )

        # Output results
        if output == "json":
            _output_json(estimate)
        else:
            _output_console(estimate, console)

        # Budget check
        if budget_check is not None:
            budget_limit = Decimal(str(budget_check))
            if estimate.total_max > budget_limit:
                console.print(
                    f"\n[bold red]BUDGET CHECK FAILED[/bold red]: "
                    f"max estimate ${estimate.total_max:.4f} "
                    f"exceeds budget ${budget_limit:.2f}"
                )
                sys.exit(EXIT_FAILURE)
            else:
                console.print(
                    f"\n[green]Budget OK[/green]: "
                    f"max estimate ${estimate.total_max:.4f} "
                    f"within budget ${budget_limit:.2f}"
                )

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


def _output_console(
    estimate: CostEstimate,
    console: Console,
) -> None:
    """Output estimate to console with Rich formatting.

    Args:
        estimate: Cost estimate to display.
        console: Rich console instance.
    """
    console.print(f"\n[bold]Cost Estimate: {estimate.suite_name}[/bold]")
    console.print(f"Model: {estimate.model}")
    if estimate.pricing:
        console.print(
            f"Pricing: ${estimate.pricing.input_per_1k}/1k input, "
            f"${estimate.pricing.output_per_1k}/1k output"
        )
    console.print()

    table = _create_estimate_table()

    for test in estimate.tests:
        table.add_row(
            test.test_id,
            _truncate(test.test_name, 30),
            str(test.input_tokens),
            f"{test.output_tokens_min}-{test.output_tokens_max}",
            str(test.runs),
            f"${test.cost_min:.4f}-${test.cost_max:.4f}",
        )

    console.print(table)

    # Summary
    console.print(f"\nTotal tests: {estimate.total_tests}")
    console.print(f"Total runs: {estimate.total_runs}")
    console.print(f"Total input tokens: {estimate.total_input_tokens:,}")
    console.print(
        f"Total output tokens: "
        f"{estimate.total_output_tokens_min:,}"
        f"-{estimate.total_output_tokens_max:,}"
    )
    console.print(
        f"\n[bold]Estimated cost range: "
        f"${estimate.total_min:.4f} - "
        f"${estimate.total_max:.4f} USD[/bold]"
    )


def _output_json(
    estimate: CostEstimate,
) -> None:
    """Output estimate as JSON.

    Args:
        estimate: Cost estimate to display.
    """
    data = {
        "suite_name": estimate.suite_name,
        "model": estimate.model,
        "provider": estimate.provider,
        "total_tests": estimate.total_tests,
        "total_runs": estimate.total_runs,
        "total_input_tokens": estimate.total_input_tokens,
        "total_output_tokens_min": estimate.total_output_tokens_min,
        "total_output_tokens_max": estimate.total_output_tokens_max,
        "total_min_usd": str(estimate.total_min),
        "total_max_usd": str(estimate.total_max),
        "tests": [
            {
                "test_id": t.test_id,
                "test_name": t.test_name,
                "input_tokens": t.input_tokens,
                "output_tokens_min": t.output_tokens_min,
                "output_tokens_max": t.output_tokens_max,
                "cost_min_usd": str(t.cost_min),
                "cost_max_usd": str(t.cost_max),
                "runs": t.runs,
            }
            for t in estimate.tests
        ],
    }
    click.echo(json.dumps(data, indent=2))


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length with ellipsis.

    Args:
        text: Text to truncate.
        max_len: Maximum length.

    Returns:
        Truncated text.
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
