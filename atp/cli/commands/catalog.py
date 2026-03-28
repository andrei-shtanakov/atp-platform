"""CLI commands for the ATP Test Catalog."""

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from atp.adapters import create_adapter
from atp.catalog.comparison import format_comparison_table
from atp.catalog.repository import CatalogRepository
from atp.catalog.sync import parse_catalog_yaml, sync_builtin_catalog
from atp.dashboard.database import init_database
from atp.loader.loader import TestLoader
from atp.runner.orchestrator import TestOrchestrator

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2


def _parse_path(path: str) -> tuple[str, str]:
    """Parse a 'category/suite' path into its components.

    Args:
        path: Path string in 'category/suite' format.

    Returns:
        Tuple of (category_slug, suite_slug).

    Raises:
        click.BadParameter: If path format is invalid.
    """
    parts = path.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise click.BadParameter(
            f"Expected 'category/suite' format, got: {path!r}",
            param_hint="PATH",
        )
    return parts[0], parts[1]


async def _do_sync() -> None:
    """Sync builtin catalog into the database."""
    db = await init_database()
    async with db.session() as session:
        await sync_builtin_catalog(session)


async def _do_list(category: str | None) -> None:
    """List catalog categories or suites.

    Args:
        category: Optional category slug to filter by.
    """
    console = Console()
    db = await init_database()

    async with db.session() as session:
        repo = CatalogRepository(session)

        if category is None:
            # List categories with suite/test counts
            suites = await repo.list_suites()

            # Auto-sync if DB is empty
            if not suites:
                click.echo("No catalog data found. Auto-syncing builtin catalog...")
                await sync_builtin_catalog(session)
                suites = await repo.list_suites()

            # Group by category
            category_data: dict[str, dict] = {}
            for suite in suites:
                cat_slug = suite.category.slug
                if cat_slug not in category_data:
                    category_data[cat_slug] = {
                        "name": suite.category.name,
                        "suites": 0,
                        "tests": 0,
                    }
                category_data[cat_slug]["suites"] += 1
                category_data[cat_slug]["tests"] += len(suite.tests)

            if not category_data:
                click.echo("No categories found.")
                return

            table = Table(
                title="Catalog Categories",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Slug", style="green", no_wrap=True)
            table.add_column("Name", style="white")
            table.add_column("Suites", style="yellow", justify="right")
            table.add_column("Tests", style="magenta", justify="right")

            for slug, data in sorted(category_data.items()):
                table.add_row(
                    slug,
                    data["name"],
                    str(data["suites"]),
                    str(data["tests"]),
                )

            console.print(table)

        else:
            # List suites in a specific category
            suites = await repo.list_suites(category_slug=category)

            if not suites:
                click.echo(f"No suites found in category: {category!r}")
                return

            table = Table(
                title=f"Suites in '{category}'",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Slug", style="green", no_wrap=True)
            table.add_column("Name", style="white")
            table.add_column("Difficulty", style="yellow")
            table.add_column("Tests", style="magenta", justify="right")
            table.add_column("Avg Score", style="blue", justify="right")
            table.add_column("Submissions", style="dim", justify="right")

            for suite in suites:
                total_submissions = sum(t.total_submissions for t in suite.tests)
                avg_scores = [
                    t.avg_score for t in suite.tests if t.avg_score is not None
                ]
                avg_score_str = (
                    f"{sum(avg_scores) / len(avg_scores):.1f}" if avg_scores else "N/A"
                )
                table.add_row(
                    suite.slug,
                    suite.name,
                    suite.difficulty or "-",
                    str(len(suite.tests)),
                    avg_score_str,
                    str(total_submissions),
                )

            console.print(table)


async def _do_info(path: str) -> None:
    """Show details for a specific catalog suite.

    Args:
        path: 'category/suite' path string.
    """
    console = Console()
    category_slug, suite_slug = _parse_path(path)

    db = await init_database()
    async with db.session() as session:
        repo = CatalogRepository(session)
        suite = await repo.get_suite_by_path(category_slug, suite_slug)

        if suite is None:
            click.echo(f"Suite not found: {path!r}", err=True)
            sys.exit(EXIT_FAILURE)

        # Suite info table
        info_table = Table(title=f"Suite: {suite.name}", show_header=False, box=None)
        info_table.add_column("Property", style="bold cyan", width=20)
        info_table.add_column("Value", style="white")

        info_table.add_row("Path", f"{category_slug}/{suite_slug}")
        info_table.add_row("Name", suite.name)
        info_table.add_row("Author", suite.author)
        info_table.add_row("Source", suite.source)
        info_table.add_row("Difficulty", suite.difficulty or "-")
        info_table.add_row("Version", suite.version)
        info_table.add_row(
            "Est. Minutes",
            str(suite.estimated_minutes) if suite.estimated_minutes else "-",
        )
        if suite.description:
            info_table.add_row("Description", suite.description)

        console.print(info_table)
        console.print()

        if not suite.tests:
            click.echo("No tests in this suite.")
            return

        # Tests table
        tests_table = Table(
            title="Tests",
            show_header=True,
            header_style="bold cyan",
        )
        tests_table.add_column("Slug", style="green", no_wrap=True)
        tests_table.add_column("Name", style="white")
        tests_table.add_column("Avg Score", style="blue", justify="right")
        tests_table.add_column("Best Score", style="magenta", justify="right")
        tests_table.add_column("Submissions", style="dim", justify="right")

        for test in suite.tests:
            tests_table.add_row(
                test.slug,
                test.name,
                f"{test.avg_score:.1f}" if test.avg_score is not None else "N/A",
                f"{test.best_score:.1f}" if test.best_score is not None else "N/A",
                str(test.total_submissions),
            )

        console.print(tests_table)


def _parse_adapter_config(config_str: str) -> dict[str, Any]:
    """Parse adapter config string 'key=value,key2=value2' into a dict.

    Args:
        config_str: Comma-separated key=value pairs.

    Returns:
        Parsed config dict, empty dict for empty/blank input.
    """
    if not config_str or not config_str.strip():
        return {}
    result: dict[str, Any] = {}
    for pair in config_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" in pair:
            key, _, value = pair.partition("=")
            result[key.strip()] = value.strip()
        else:
            result[pair] = True
    return result


async def _execute_catalog_run(
    session: Any,
    category_slug: str,
    suite_slug: str,
    adapter_type: str,
    adapter_config: dict[str, Any],
    agent_name: str,
    runs_per_test: int,
) -> None:
    """Run a catalog suite via TestOrchestrator and record submissions.

    Args:
        session: SQLAlchemy async session.
        category_slug: Category slug.
        suite_slug: Suite slug.
        adapter_type: Adapter type string.
        adapter_config: Adapter configuration dict.
        agent_name: Name of the agent being tested.
        runs_per_test: Number of runs per test.
    """
    repo = CatalogRepository(session)
    catalog_suite = await repo.get_suite_by_path(category_slug, suite_slug)

    if catalog_suite is None:
        click.echo(f"Suite not found: {category_slug}/{suite_slug!r}", err=True)
        sys.exit(EXIT_FAILURE)

    # Load ATP test suite from stored YAML
    loader = TestLoader()
    suite = loader.load_string(catalog_suite.suite_yaml)

    # Create adapter
    adapter = create_adapter(adapter_type, adapter_config)

    # Run via orchestrator
    click.echo(f"Running suite: {catalog_suite.name} ({len(suite.tests)} tests)...")
    async with TestOrchestrator(
        adapter=adapter,
        runs_per_test=runs_per_test,
    ) as orchestrator:
        suite_result = await orchestrator.run_suite(
            suite=suite,
            agent_name=agent_name,
            runs_per_test=runs_per_test,
        )

    # Build slug → CatalogTest mapping for submission creation
    test_by_slug = {ct.slug: ct for ct in catalog_suite.tests}

    # Create submissions and update stats
    comparison_rows: list[dict] = []
    for test_result in suite_result.tests:
        test_slug = test_result.test.id
        catalog_test = test_by_slug.get(test_slug)
        if catalog_test is None:
            continue

        score = (
            100.0 * test_result.successful_runs / test_result.total_runs
            if test_result.total_runs > 0
            else 0.0
        )

        # Extract metrics from the first run if available
        total_tokens: int | None = None
        cost_usd: float | None = None
        duration: float | None = test_result.duration_seconds

        if test_result.runs:
            first_run = test_result.runs[0]
            if first_run.response and first_run.response.metrics:
                m = first_run.response.metrics
                if m.total_tokens is not None:
                    total_tokens = m.total_tokens
                elif m.input_tokens is not None or m.output_tokens is not None:
                    total_tokens = (m.input_tokens or 0) + (m.output_tokens or 0)
                cost_usd = m.cost_usd

        await repo.create_submission(
            test_id=catalog_test.id,
            agent_name=agent_name,
            agent_type=adapter_type,
            score=score,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            duration_seconds=duration,
        )
        await repo.update_test_stats(catalog_test.id)

        comparison_rows.append(
            {
                "name": catalog_test.name or test_slug,
                "score": score,
                "avg": catalog_test.avg_score,
                "best": catalog_test.best_score,
            }
        )

    await session.commit()

    # Collect top-3 agents across all tests for comparison table
    top3_scores: dict[str, list[float]] = {}
    for catalog_test in catalog_suite.tests:
        submissions = await repo.get_top_submissions(catalog_test.id, limit=5)
        for sub in submissions:
            top3_scores.setdefault(sub.agent_name, []).append(sub.score)

    top3 = sorted(
        [
            {"name": name, "score": sum(scores) / len(scores)}
            for name, scores in top3_scores.items()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )[:3]

    table_str = format_comparison_table(comparison_rows, top3)
    click.echo(table_str)
    click.echo(
        f"\nCompleted: {suite_result.passed_tests}/{suite_result.total_tests} passed"
    )


async def _do_run(
    path: str,
    adapter: str,
    adapter_config: str,
    agent_name: str,
    runs: int,
) -> None:
    """Run a catalog suite against an agent via TestOrchestrator.

    Args:
        path: 'category/suite' path string.
        adapter: Adapter type.
        adapter_config: Adapter configuration string (key=value pairs).
        agent_name: Agent name.
        runs: Number of runs per test.
    """
    category_slug, suite_slug = _parse_path(path)
    config_dict = _parse_adapter_config(adapter_config)

    db = await init_database()
    async with db.session() as session:
        await _execute_catalog_run(
            session=session,
            category_slug=category_slug,
            suite_slug=suite_slug,
            adapter_type=adapter,
            adapter_config=config_dict,
            agent_name=agent_name,
            runs_per_test=runs,
        )


async def _do_publish(file_path: Path) -> None:
    """Publish a catalog YAML file to the database.

    Args:
        file_path: Path to the YAML file.
    """
    raw_content = file_path.read_text(encoding="utf-8")
    catalog_meta, tests = parse_catalog_yaml(raw_content)

    category_slug: str = catalog_meta.get("category", "")
    if not category_slug:
        # Try to derive from slug prefix or directory
        suite_slug_full: str = catalog_meta.get("slug", file_path.stem)
        parts = suite_slug_full.split("/", 1)
        if len(parts) == 2:
            category_slug, suite_slug_value = parts[0], parts[1]
        else:
            category_slug = file_path.parent.name
            suite_slug_value = suite_slug_full
    else:
        suite_slug_value = catalog_meta.get("slug", file_path.stem)

    category_name = category_slug.replace("-", " ").title()
    tags_list: list[str] = catalog_meta.get("tags", [])
    tags_payload: dict | None = {"tags": tags_list} if tags_list else None

    db = await init_database()
    async with db.session() as session:
        repo = CatalogRepository(session)

        category = await repo.upsert_category(
            slug=category_slug,
            name=category_name,
        )

        suite = await repo.upsert_suite(
            category_id=category.id,
            slug=suite_slug_value,
            name=catalog_meta.get("name") or suite_slug_value,
            author=catalog_meta.get("author", "community"),
            source=catalog_meta.get("source", "community"),
            suite_yaml=raw_content,
            description=catalog_meta.get("description"),
            difficulty=catalog_meta.get("difficulty"),
            estimated_minutes=catalog_meta.get("estimated_minutes"),
            tags=tags_payload,
            version=str(catalog_meta.get("version", "1.0")),
        )

        for test_data in tests:
            test_slug: str = test_data.get("id", "")
            test_tags_list: list[str] = test_data.get("tags", [])
            test_tags: dict | None = (
                {"tags": test_tags_list} if test_tags_list else None
            )
            task_block: dict = test_data.get("task", {})
            task_description: str = task_block.get("description", "")

            await repo.upsert_test(
                suite_id=suite.id,
                slug=test_slug,
                name=test_data.get("name", test_slug),
                task_description=task_description,
                difficulty=catalog_meta.get("difficulty"),
                tags=test_tags,
            )

        await session.commit()

    click.echo(
        f"Published: {category_slug}/{suite_slug_value} "
        f"({len(tests)} tests) from {file_path}"
    )


async def _do_results(path: str) -> None:
    """Show top submissions for a catalog suite.

    Args:
        path: 'category/suite' path string.
    """
    console = Console()
    category_slug, suite_slug = _parse_path(path)

    db = await init_database()
    async with db.session() as session:
        repo = CatalogRepository(session)
        suite = await repo.get_suite_by_path(category_slug, suite_slug)

        if suite is None:
            click.echo(f"Suite not found: {path!r}", err=True)
            sys.exit(EXIT_FAILURE)

        if not suite.tests:
            click.echo("No tests in this suite.")
            return

        table = Table(
            title=f"Top Results: {suite.name}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Test", style="green", no_wrap=True)
        table.add_column("Rank", style="dim", justify="right")
        table.add_column("Agent", style="white")
        table.add_column("Score", style="magenta", justify="right")
        table.add_column("Tokens", style="blue", justify="right")
        table.add_column("Cost (USD)", style="yellow", justify="right")

        for test in suite.tests:
            submissions = await repo.get_top_submissions(test.id, limit=5)
            for rank, sub in enumerate(submissions, start=1):
                tokens_str = str(sub.total_tokens) if sub.total_tokens else "-"
                cost_str = f"${sub.cost_usd:.4f}" if sub.cost_usd is not None else "-"
                table.add_row(
                    test.slug if rank == 1 else "",
                    str(rank),
                    sub.agent_name,
                    f"{sub.score:.1f}",
                    tokens_str,
                    cost_str,
                )

        console.print(table)


@click.group(name="catalog")
def catalog_command() -> None:
    """Browse and manage the ATP Test Catalog.

    The catalog provides curated test suites organized by category.
    Use subcommands to browse, run, and publish test suites.

    Examples:

      # Sync builtin catalog to DB
      atp catalog sync

      # List all categories
      atp catalog list

      # List suites in a category
      atp catalog list coding

      # Show suite details
      atp catalog info coding/file-operations

      # Run a catalog suite
      atp catalog run coding/file-operations --adapter=http

      # Publish a custom suite
      atp catalog publish my-suite.yaml

      # View results for a suite
      atp catalog results coding/file-operations
    """
    pass


@catalog_command.command(name="sync")
def sync_cmd() -> None:
    """Sync builtin catalog suites into the database.

    Scans the builtin YAML files and upserts all categories, suites,
    and tests into the ATP database.

    Examples:

      atp catalog sync

    Exit Codes:

      0 - Success
      2 - Error occurred
    """
    try:
        asyncio.run(_do_sync())
        click.echo("Builtin catalog synced successfully.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


@catalog_command.command(name="list")
@click.argument("category", required=False, default=None)
def list_cmd(category: str | None) -> None:
    """List catalog categories or suites within a category.

    Without CATEGORY: shows all categories with suite and test counts.
    With CATEGORY: shows all suites in that category.

    Examples:

      # List all categories
      atp catalog list

      # List suites in 'coding' category
      atp catalog list coding

    Exit Codes:

      0 - Success
      2 - Error occurred
    """
    try:
        asyncio.run(_do_list(category))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


@catalog_command.command(name="info")
@click.argument("path")
def info_cmd(path: str) -> None:
    """Show detailed information about a catalog suite.

    PATH must be in 'category/suite' format (e.g., 'coding/file-operations').

    Examples:

      atp catalog info coding/file-operations

    Exit Codes:

      0 - Success
      1 - Suite not found
      2 - Error occurred
    """
    try:
        asyncio.run(_do_info(path))
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


@catalog_command.command(name="run")
@click.argument("path")
@click.option(
    "--adapter",
    default="http",
    help="Adapter type (http, cli, container, etc.)",
)
@click.option(
    "--adapter-config",
    default="",
    help="Adapter configuration as key=value pairs (e.g. url=http://localhost:8080)",
)
@click.option(
    "--agent-name",
    default="my-agent",
    help="Name of the agent being tested",
)
@click.option(
    "--runs",
    default=1,
    show_default=True,
    help="Number of runs per test",
)
def run_cmd(
    path: str, adapter: str, adapter_config: str, agent_name: str, runs: int
) -> None:
    """Run a catalog suite against an agent via TestOrchestrator.

    PATH must be in 'category/suite' format (e.g., 'coding/file-operations').

    Examples:

      atp catalog run coding/file-operations --adapter=http
      atp catalog run coding/file-operations --adapter=http --runs=3

    Exit Codes:

      0 - Success
      1 - Suite not found
      2 - Error occurred
    """
    try:
        asyncio.run(_do_run(path, adapter, adapter_config, agent_name, runs))
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


@catalog_command.command(name="publish")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def publish_cmd(file: Path) -> None:
    """Publish a catalog YAML file to the database.

    FILE is the path to a YAML file containing a catalog suite definition.
    The file must include a 'catalog:' section with metadata.

    Examples:

      atp catalog publish my-suite.yaml

    Exit Codes:

      0 - Success
      2 - Error occurred
    """
    try:
        asyncio.run(_do_publish(file))
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


@catalog_command.command(name="results")
@click.argument("path")
def results_cmd(path: str) -> None:
    """Show top agent results for a catalog suite.

    PATH must be in 'category/suite' format (e.g., 'coding/file-operations').

    Examples:

      atp catalog results coding/file-operations

    Exit Codes:

      0 - Success
      1 - Suite not found
      2 - Error occurred
    """
    try:
        asyncio.run(_do_results(path))
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)
