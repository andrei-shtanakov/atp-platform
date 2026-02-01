"""CLI commands for running ATP benchmarks."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from atp.benchmarks import (
    BenchmarkCategoryNotFoundError,
    BenchmarkNotFoundError,
    BenchmarkResult,
    BenchmarkSuiteResult,
    get_registry,
)

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2


def _create_suites_table(title: str = "Benchmark Suites") -> Table:
    """Create a Rich table for benchmark suite display.

    Args:
        title: Table title.

    Returns:
        Configured Rich Table instance.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("Tests", style="yellow", justify="right")
    table.add_column("Version", style="blue")
    table.add_column("Description", style="dim")
    return table


def _create_results_table(title: str = "Benchmark Results") -> Table:
    """Create a Rich table for benchmark results display.

    Args:
        title: Table title.

    Returns:
        Configured Rich Table instance.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Test ID", style="green", no_wrap=True)
    table.add_column("Score", style="yellow", justify="right")
    table.add_column("Status", style="magenta")
    table.add_column("Time (s)", style="blue", justify="right")
    table.add_column("Baseline Delta", style="dim", justify="right")
    return table


def _format_score(score: float) -> str:
    """Format a score for display.

    Args:
        score: Normalized score (0-100).

    Returns:
        Formatted score string.
    """
    return f"{score:.1f}"


def _format_delta(delta: float | None) -> str:
    """Format a baseline delta for display.

    Args:
        delta: Score delta from baseline.

    Returns:
        Formatted delta string with +/- prefix.
    """
    if delta is None:
        return "-"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}"


def _format_status(passed: bool) -> str:
    """Format pass/fail status for display.

    Args:
        passed: Whether the test passed.

    Returns:
        Status string.
    """
    return "[green]PASS[/green]" if passed else "[red]FAIL[/red]"


@click.group(name="benchmark")
def benchmark_command() -> None:
    """Run and manage ATP benchmark suites.

    Benchmarks are curated test suites for evaluating agent performance
    across common tasks like coding, research, reasoning, and data processing.

    Examples:

      # List all available benchmarks
      atp benchmark list

      # List benchmarks in a specific category
      atp benchmark list --category=coding

      # Run all benchmarks in a category
      atp benchmark run coding

      # Run all benchmarks
      atp benchmark run --all

      # Run benchmarks with a specific agent
      atp benchmark run coding --agent=my-agent

      # Output results as JSON
      atp benchmark run coding --output=json --output-file=results.json
    """
    pass


@benchmark_command.command(name="list")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["coding", "research", "reasoning", "data_processing"]),
    help="Filter by benchmark category",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed information about each benchmark",
)
def list_benchmarks(category: str | None, verbose: bool) -> None:
    """List all available benchmark suites.

    Shows registered benchmark suites with their categories, test counts,
    and descriptions.

    Examples:

      # List all benchmarks
      atp benchmark list

      # List only coding benchmarks
      atp benchmark list --category=coding

      # Show detailed information
      atp benchmark list --verbose

    Exit Codes:

      0 - Success
      2 - Error occurred
    """
    console = Console()

    try:
        registry = get_registry()

        if category:
            try:
                suites = registry.get_by_category(category)
            except BenchmarkCategoryNotFoundError:
                click.echo(f"Unknown category: {category}", err=True)
                sys.exit(EXIT_ERROR)
        else:
            suites = [registry.get(name) for name in registry.list_suites()]

        if not suites:
            cat_str = f" in category '{category}'" if category else ""
            click.echo(f"No benchmark suites found{cat_str}.")
            sys.exit(EXIT_SUCCESS)

        # Create and populate table
        title = f"{category.title()} Benchmarks" if category else "All Benchmarks"
        table = _create_suites_table(title=title)

        for suite in sorted(suites, key=lambda s: (s.category.value, s.name)):
            info = suite.get_info()
            desc = suite.description.strip().split("\n")[0]
            truncated_desc = desc[:50] + "..." if len(desc) > 50 else desc
            table.add_row(
                info.name,
                info.category.value,
                str(info.test_count),
                info.version,
                truncated_desc,
            )

        console.print(table)

        if verbose:
            console.print(f"\nTotal: {len(suites)} suite(s)")
            categories = registry.list_categories()
            console.print(f"Categories: {', '.join(categories)}")

            # Show difficulty distribution
            for suite in suites:
                info = suite.get_info()
                if info.difficulty_distribution:
                    console.print(f"\n[bold]{info.name}[/bold]:")
                    console.print(f"  Description: {suite.description.strip()}")
                    console.print(f"  Difficulty: {info.difficulty_distribution}")
                    if info.average_baseline_score is not None:
                        console.print(
                            f"  Average baseline: {info.average_baseline_score:.1f}"
                        )

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error listing benchmarks: {e}", err=True)
        sys.exit(EXIT_ERROR)


@benchmark_command.command(name="run")
@click.argument("category", required=False)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    help="Run all benchmark suites across all categories",
)
@click.option(
    "--suite",
    "-s",
    "suite_name",
    type=str,
    help="Run a specific benchmark suite by name",
)
@click.option(
    "--agent",
    "-a",
    "agent_name",
    type=str,
    default="test-agent",
    help="Name of the agent to benchmark",
)
@click.option(
    "--adapter",
    type=str,
    default="http",
    help="Adapter type to use (http, cli, container, etc.)",
)
@click.option(
    "--adapter-config",
    type=str,
    multiple=True,
    help="Adapter configuration as key=value pairs",
)
@click.option(
    "--output",
    "-o",
    "output_format",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format for results",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    help="Output file path (for json output)",
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help="Maximum number of tests to run in parallel",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on first test failure",
)
def run_benchmarks(
    category: str | None,
    run_all: bool,
    suite_name: str | None,
    agent_name: str,
    adapter: str,
    adapter_config: tuple[str, ...],
    output_format: str,
    output_file: Path | None,
    parallel: int,
    verbose: bool,
    fail_fast: bool,
) -> None:
    """Run benchmark suites against an agent.

    CATEGORY is the benchmark category to run (coding, research, reasoning,
    data_processing). Use --all to run all categories, or --suite to run
    a specific suite.

    Examples:

      # Run all coding benchmarks
      atp benchmark run coding

      # Run all benchmarks
      atp benchmark run --all

      # Run a specific suite
      atp benchmark run --suite=coding

      # Run with a specific agent
      atp benchmark run coding --agent=my-agent

      # Output results as JSON
      atp benchmark run coding --output=json --output-file=results.json

      # Run benchmarks in parallel
      atp benchmark run coding --parallel=4

    Exit Codes:

      0 - All benchmarks passed
      1 - One or more benchmarks failed
      2 - Error occurred
    """
    # Validate arguments
    if not category and not run_all and not suite_name:
        click.echo(
            "Error: Specify a CATEGORY, --all, or --suite to run benchmarks.",
            err=True,
        )
        sys.exit(EXIT_ERROR)

    if parallel < 1:
        click.echo("Error: --parallel must be at least 1", err=True)
        sys.exit(EXIT_ERROR)

    try:
        result = asyncio.run(
            _run_benchmarks(
                category=category,
                run_all=run_all,
                suite_name=suite_name,
                agent_name=agent_name,
                adapter_type=adapter,
                adapter_config=adapter_config,
                output_format=output_format,
                output_file=output_file,
                parallel=parallel,
                verbose=verbose,
                fail_fast=fail_fast,
            )
        )

        sys.exit(EXIT_SUCCESS if result else EXIT_FAILURE)

    except BenchmarkNotFoundError as e:
        click.echo(f"Benchmark not found: {e.name}", err=True)
        sys.exit(EXIT_ERROR)
    except BenchmarkCategoryNotFoundError as e:
        click.echo(f"Category not found: {e.category}", err=True)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        click.echo(f"Error running benchmarks: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _run_benchmarks(
    category: str | None,
    run_all: bool,
    suite_name: str | None,
    agent_name: str,
    adapter_type: str,
    adapter_config: tuple[str, ...],
    output_format: str,
    output_file: Path | None,
    parallel: int,
    verbose: bool,
    fail_fast: bool,
) -> bool:
    """Run benchmarks asynchronously.

    Args:
        category: Benchmark category to run.
        run_all: Whether to run all benchmarks.
        suite_name: Specific suite name to run.
        agent_name: Name of the agent to benchmark.
        adapter_type: Type of adapter to use.
        adapter_config: Adapter configuration.
        output_format: Output format (console or json).
        output_file: Output file path.
        parallel: Maximum parallel tests.
        verbose: Verbose output.
        fail_fast: Stop on first failure.

    Returns:
        True if all benchmarks passed, False otherwise.
    """
    from atp.adapters import create_adapter
    from atp.runner import SandboxConfig, TestOrchestrator, create_progress_callback

    console = Console()
    registry = get_registry()

    # Get suites to run
    suites_to_run = []
    if suite_name:
        suites_to_run = [registry.get(suite_name)]
    elif run_all:
        suites_to_run = [registry.get(name) for name in registry.list_suites()]
    elif category:
        suites_to_run = registry.get_by_category(category)

    if not suites_to_run:
        click.echo("No benchmark suites found to run.", err=True)
        return False

    # Parse adapter config
    config_dict: dict[str, Any] = {}
    for item in adapter_config:
        if "=" in item:
            key, value = item.split("=", 1)
            try:
                config_dict[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                config_dict[key] = value

    # Create adapter
    adapter = create_adapter(adapter_type, config_dict)

    # Create sandbox config
    sandbox_config = SandboxConfig(enabled=False)

    # Create progress callback
    progress_callback = create_progress_callback(
        max_parallel=parallel,
        verbose=verbose,
        use_colors=True,
    )

    all_results: list[BenchmarkSuiteResult] = []
    overall_success = True

    for benchmark_suite in suites_to_run:
        click.echo(f"\nRunning benchmark suite: {benchmark_suite.name}")
        click.echo(f"Category: {benchmark_suite.category.value}")
        click.echo(f"Tests: {len(benchmark_suite.tests)}")
        click.echo("-" * 50)

        # Convert benchmark tests to test suite format
        test_suite = _convert_to_test_suite(benchmark_suite)

        # Run tests
        async with TestOrchestrator(
            adapter=adapter,
            sandbox_config=sandbox_config,
            progress_callback=progress_callback,
            runs_per_test=1,
            fail_fast=fail_fast,
            parallel_tests=parallel > 1,
            max_parallel_tests=parallel,
        ) as orchestrator:
            suite_result = await orchestrator.run_suite(
                suite=test_suite,
                agent_name=agent_name,
                runs_per_test=1,
            )

        # Convert to benchmark results
        benchmark_results: list[BenchmarkResult] = []
        for test_result in suite_result.tests:
            # Calculate score from test result
            raw_score = 1.0 if test_result.success else 0.0
            normalized_score = registry.normalize_score(raw_score)

            execution_time = 0.0
            tokens_used = None
            cost_usd = None

            if test_result.runs:
                run = test_result.runs[0]
                if run.response and run.response.metrics:
                    metrics = run.response.metrics
                    if metrics.wall_time_seconds:
                        execution_time = metrics.wall_time_seconds
                    if metrics.total_tokens:
                        tokens_used = metrics.total_tokens
                    if metrics.cost_usd:
                        cost_usd = metrics.cost_usd

            result = BenchmarkResult(
                test_id=test_result.test.id,
                raw_score=raw_score,
                normalized_score=normalized_score,
                passed=test_result.success,
                execution_time_seconds=execution_time,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                error=None if test_result.success else "Test failed",
            )
            benchmark_results.append(result)

        # Create suite result
        suite_benchmark_result = registry.create_suite_result(
            suite_name=benchmark_suite.name,
            agent_name=agent_name,
            results=benchmark_results,
        )
        all_results.append(suite_benchmark_result)

        if not suite_result.success:
            overall_success = False

        if fail_fast and not suite_result.success:
            break

    # Output results
    _output_results(
        results=all_results,
        output_format=output_format,
        output_file=output_file,
        verbose=verbose,
        console=console,
    )

    return overall_success


def _convert_to_test_suite(benchmark_suite: Any) -> Any:
    """Convert a benchmark suite to a TestSuite format.

    Args:
        benchmark_suite: BenchmarkSuite to convert.

    Returns:
        TestSuite compatible with the runner.
    """
    from atp.loader.models import (
        Assertion,
        Constraints,
        TaskDefinition,
        TestDefinition,
        TestSuite,
    )

    tests = []
    for benchmark_test in benchmark_suite.tests:
        # Create task definition
        task = TaskDefinition(
            description=benchmark_test.task_description,
            expected_artifacts=benchmark_test.expected_artifacts,
        )

        # Create constraints
        constraints = Constraints(
            max_steps=benchmark_test.max_steps,
            timeout_seconds=benchmark_test.timeout_seconds,
        )

        # Convert assertions
        assertions = [
            Assertion(type=a.get("type", ""), config=a.get("config", {}))
            if isinstance(a, dict)
            else Assertion(type=a.type, config=a.config)
            for a in benchmark_test.assertions
        ]

        test = TestDefinition(
            id=benchmark_test.id,
            name=benchmark_test.name,
            description=benchmark_test.description,
            task=task,
            constraints=constraints,
            assertions=assertions,
            tags=benchmark_test.tags,
        )
        tests.append(test)

    return TestSuite(
        test_suite=benchmark_suite.name,
        version=benchmark_suite.version,
        description=benchmark_suite.description,
        tests=tests,
    )


def _output_results(
    results: list[BenchmarkSuiteResult],
    output_format: str,
    output_file: Path | None,
    verbose: bool,
    console: Console,
) -> None:
    """Output benchmark results.

    Args:
        results: List of benchmark suite results.
        output_format: Output format (console or json).
        output_file: Output file path.
        verbose: Verbose output.
        console: Rich console instance.
    """
    if output_format == "json":
        _output_json(results, output_file)
    else:
        _output_console(results, verbose, console)


def _output_console(
    results: list[BenchmarkSuiteResult],
    verbose: bool,
    console: Console,
) -> None:
    """Output results to console.

    Args:
        results: List of benchmark suite results.
        verbose: Verbose output.
        console: Rich console instance.
    """
    console.print("\n" + "=" * 60)
    console.print("[bold]Benchmark Results Summary[/bold]")
    console.print("=" * 60)

    total_tests = 0
    total_passed = 0
    total_failed = 0
    overall_score = 0.0

    for suite_result in results:
        console.print(f"\n[bold cyan]{suite_result.suite_name}[/bold cyan]")
        console.print(f"Category: {suite_result.category.value}")
        console.print(f"Agent: {suite_result.agent_name}")
        console.print("-" * 40)

        # Create results table
        table = _create_results_table(title="")

        for result in suite_result.results:
            # Get baseline delta if available
            delta = None
            if suite_result.baseline_comparison:
                # Use average baseline delta for display
                deltas = list(suite_result.baseline_comparison.values())
                if deltas:
                    delta = sum(deltas) / len(deltas)

            table.add_row(
                result.test_id,
                _format_score(result.normalized_score),
                _format_status(result.passed),
                f"{result.execution_time_seconds:.2f}",
                _format_delta(delta) if result.passed else "-",
            )

        console.print(table)

        # Suite summary
        console.print(
            f"\nPassed: {suite_result.passed_tests}/{suite_result.total_tests}"
        )
        console.print(f"Pass Rate: {suite_result.pass_rate:.1f}%")
        console.print(f"Average Score: {suite_result.average_normalized_score:.1f}/100")
        console.print(f"Total Time: {suite_result.total_execution_time_seconds:.2f}s")

        # Baseline comparison
        if suite_result.baseline_comparison:
            console.print("\n[bold]Baseline Comparison:[/bold]")
            for model_name, delta in suite_result.baseline_comparison.items():
                delta_str = _format_delta(delta)
                color = "green" if delta >= 0 else "red"
                console.print(f"  vs {model_name}: [{color}]{delta_str}[/{color}]")

        total_tests += suite_result.total_tests
        total_passed += suite_result.passed_tests
        total_failed += suite_result.failed_tests
        overall_score += suite_result.average_normalized_score

    # Overall summary
    if len(results) > 1:
        console.print("\n" + "=" * 60)
        console.print("[bold]Overall Summary[/bold]")
        console.print("=" * 60)
        console.print(f"Total Suites: {len(results)}")
        console.print(f"Total Tests: {total_tests}")
        console.print(f"Passed: {total_passed}")
        console.print(f"Failed: {total_failed}")
        if total_tests > 0:
            console.print(
                f"Overall Pass Rate: {(total_passed / total_tests) * 100:.1f}%"
            )
        if results:
            console.print(f"Average Score: {overall_score / len(results):.1f}/100")


def _output_json(
    results: list[BenchmarkSuiteResult],
    output_file: Path | None,
) -> None:
    """Output results as JSON.

    Args:
        results: List of benchmark suite results.
        output_file: Output file path.
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "suites": [],
        "summary": {
            "total_suites": len(results),
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
            "overall_pass_rate": 0.0,
            "average_score": 0.0,
        },
    }

    total_tests = 0
    total_passed = 0
    overall_score = 0.0

    for suite_result in results:
        suite_data = {
            "name": suite_result.suite_name,
            "category": suite_result.category.value,
            "agent": suite_result.agent_name,
            "total_tests": suite_result.total_tests,
            "passed_tests": suite_result.passed_tests,
            "failed_tests": suite_result.failed_tests,
            "pass_rate": suite_result.pass_rate,
            "average_score": suite_result.average_normalized_score,
            "total_execution_time_seconds": suite_result.total_execution_time_seconds,
            "baseline_comparison": suite_result.baseline_comparison,
            "results": [
                {
                    "test_id": r.test_id,
                    "raw_score": r.raw_score,
                    "normalized_score": r.normalized_score,
                    "passed": r.passed,
                    "execution_time_seconds": r.execution_time_seconds,
                    "tokens_used": r.tokens_used,
                    "cost_usd": r.cost_usd,
                    "error": r.error,
                }
                for r in suite_result.results
            ],
        }
        output_data["suites"].append(suite_data)

        total_tests += suite_result.total_tests
        total_passed += suite_result.passed_tests
        overall_score += suite_result.average_normalized_score

    output_data["summary"]["total_tests"] = total_tests
    output_data["summary"]["total_passed"] = total_passed
    output_data["summary"]["total_failed"] = total_tests - total_passed
    if total_tests > 0:
        output_data["summary"]["overall_pass_rate"] = (total_passed / total_tests) * 100
    if results:
        output_data["summary"]["average_score"] = overall_score / len(results)

    json_output = json.dumps(output_data, indent=2)

    if output_file:
        output_file.write_text(json_output)
        click.echo(f"Results written to {output_file}")
    else:
        click.echo(json_output)


@benchmark_command.command(name="info")
@click.argument("name")
def benchmark_info(name: str) -> None:
    """Show detailed information about a benchmark suite.

    NAME is the benchmark suite name to get info about.

    Examples:

      # Get info about the coding benchmark
      atp benchmark info coding

    Exit Codes:

      0 - Success
      1 - Benchmark not found
      2 - Error occurred
    """
    console = Console()

    try:
        registry = get_registry()
        suite = registry.get(name)
        info = suite.get_info()

        # Create info table
        table = Table(title=f"Benchmark: {info.name}", show_header=False, box=None)
        table.add_column("Property", style="bold cyan", width=25)
        table.add_column("Value", style="white")

        table.add_row("Name", info.name)
        table.add_row("Category", info.category.value)
        table.add_row("Version", info.version)
        table.add_row("Tests", str(info.test_count))
        table.add_row("Description", suite.description.strip())

        if info.difficulty_distribution:
            dist_str = ", ".join(
                f"{k}: {v}" for k, v in info.difficulty_distribution.items()
            )
            table.add_row("Difficulty Distribution", dist_str)

        if info.average_baseline_score is not None:
            table.add_row(
                "Average Baseline Score", f"{info.average_baseline_score:.1f}/100"
            )

        table.add_row("Default Timeout", f"{suite.default_timeout_seconds}s")
        if suite.default_max_steps:
            table.add_row("Default Max Steps", str(suite.default_max_steps))

        console.print(table)

        # List tests
        console.print(f"\n[bold]Tests ({len(suite.tests)}):[/bold]")

        tests_table = Table(show_header=True, header_style="bold")
        tests_table.add_column("ID", style="cyan")
        tests_table.add_column("Name", style="white")
        tests_table.add_column("Difficulty", style="yellow")
        tests_table.add_column("Est. Time", style="blue")

        for test in suite.tests:
            tests_table.add_row(
                test.id,
                test.name,
                test.metadata.difficulty.value,
                f"{test.metadata.estimated_time_seconds}s",
            )

        console.print(tests_table)

        # Show baseline scores if available
        baselines = registry.get_baseline_scores(name)
        has_baselines = any(scores for scores in baselines.values())

        if has_baselines:
            console.print("\n[bold]Baseline Scores:[/bold]")
            for test_id, scores in baselines.items():
                if scores:
                    console.print(f"  {test_id}:")
                    for baseline in scores:
                        console.print(
                            f"    {baseline.model_name}: "
                            f"{baseline.score:.1f} ({baseline.date})"
                        )

        sys.exit(EXIT_SUCCESS)

    except BenchmarkNotFoundError:
        click.echo(f"Benchmark suite not found: {name}", err=True)
        sys.exit(EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error getting benchmark info: {e}", err=True)
        sys.exit(EXIT_ERROR)


@benchmark_command.command(name="categories")
def list_categories() -> None:
    """List available benchmark categories.

    Shows all registered benchmark categories that can be used with
    the 'atp benchmark run' command.

    Examples:

      atp benchmark categories

    Exit Codes:

      0 - Success
    """
    console = Console()
    registry = get_registry()
    categories = registry.list_categories()

    console.print("[bold]Available Benchmark Categories:[/bold]\n")

    category_descriptions = {
        "coding": "Code generation, review, and bug fixing tasks",
        "research": "Web research and document summarization",
        "reasoning": "Logical puzzles and mathematical problems",
        "data_processing": "Data transformation and analysis tasks",
    }

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Category", style="green")
    table.add_column("Description", style="white")
    table.add_column("Suites", style="yellow", justify="right")

    for category in sorted(categories):
        desc = category_descriptions.get(category, "")
        try:
            suites = registry.get_by_category(category)
            count = len(suites)
        except BenchmarkCategoryNotFoundError:
            count = 0
        table.add_row(category, desc, str(count))

    console.print(table)
    sys.exit(EXIT_SUCCESS)
