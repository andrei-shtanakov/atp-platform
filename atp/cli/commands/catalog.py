"""CLI commands for the ATP Test Catalog."""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from atp.catalog.repository import CatalogRepository
from atp.catalog.sync import parse_catalog_yaml, sync_builtin_catalog
from atp.dashboard.database import init_database

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


async def _do_run(
    path: str, adapter: str, adapter_config: str, agent_name: str
) -> None:
    """Show run instructions for a catalog suite.

    Args:
        path: 'category/suite' path string.
        adapter: Adapter type.
        adapter_config: Adapter configuration string.
        agent_name: Agent name.
    """
    category_slug, suite_slug = _parse_path(path)

    db = await init_database()
    async with db.session() as session:
        repo = CatalogRepository(session)
        suite = await repo.get_suite_by_path(category_slug, suite_slug)

        if suite is None:
            click.echo(f"Suite not found: {path!r}", err=True)
            sys.exit(EXIT_FAILURE)

    click.echo(f"Suite: {suite.name}")
    click.echo(f"  Category:   {category_slug}")
    click.echo(f"  Difficulty: {suite.difficulty or 'N/A'}")
    click.echo(f"  Tests:      {len(suite.tests)}")
    click.echo()
    click.echo(
        "Run functionality requires a running agent. Save the suite YAML and use:"
    )
    config_flag = f" --adapter-config {adapter_config}" if adapter_config else ""
    click.echo(
        f"  atp test <suite.yaml> --adapter={adapter}"
        f"{config_flag} --agent-name={agent_name}"
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
    help="Adapter configuration as key=value pairs",
)
@click.option(
    "--agent-name",
    default="my-agent",
    help="Name of the agent being tested",
)
def run_cmd(path: str, adapter: str, adapter_config: str, agent_name: str) -> None:
    """Show how to run a catalog suite against an agent.

    PATH must be in 'category/suite' format (e.g., 'coding/file-operations').

    Note: Full run integration is a work-in-progress. This command
    shows the equivalent 'atp test' invocation to run the suite.

    Examples:

      atp catalog run coding/file-operations --adapter=http

    Exit Codes:

      0 - Success
      1 - Suite not found
      2 - Error occurred
    """
    try:
        asyncio.run(_do_run(path, adapter, adapter_config, agent_name))
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
