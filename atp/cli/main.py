"""Main CLI entry point for ATP."""

import sys
from pathlib import Path

import click

from atp.loader import TestLoader


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """ATP - Agent Test Platform CLI."""
    pass


@cli.command()
@click.argument("suite_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tags",
    type=str,
    help=(
        "Filter tests by tags. "
        "Use comma-separated values for multiple tags (OR logic). "
        "Prefix with '!' to exclude. "
        "Examples: --tags=smoke, --tags=smoke,core, --tags=!slow, "
        "--tags=smoke,!slow"
    ),
)
@click.option(
    "--list-only",
    is_flag=True,
    help="List matching tests without running them",
)
def run(suite_file: Path, tags: str | None, list_only: bool) -> None:
    """Run tests from a test suite file.

    Args:
        suite_file: Path to the test suite YAML file
        tags: Tag filter expression (e.g., "smoke,!slow")
        list_only: If True, only list tests without running them
    """
    try:
        # Load test suite
        loader = TestLoader()
        suite = loader.load_file(suite_file)

        # Apply tag filtering
        if tags:
            suite = suite.filter_by_tags(tags)

        # Check if any tests match
        if not suite.tests:
            click.echo("No tests match the specified criteria.", err=True)
            sys.exit(1)

        # List tests
        if list_only:
            click.echo(f"Test Suite: {suite.test_suite}")
            click.echo(f"Tests ({len(suite.tests)}):")
            for test in suite.tests:
                tags_str = ", ".join(test.tags) if test.tags else "no tags"
                click.echo(f"  - {test.id}: {test.name} [{tags_str}]")
            return

        # TODO: Implement actual test execution
        click.echo(f"Running {len(suite.tests)} test(s) from {suite.test_suite}")
        for test in suite.tests:
            click.echo(f"  Running: {test.id} - {test.name}")

        click.echo("Test execution not yet implemented.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("suite_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tags",
    type=str,
    help="Filter tests by tags (comma-separated, use ! to exclude)",
)
def list(suite_file: Path, tags: str | None) -> None:
    """List tests in a test suite.

    Args:
        suite_file: Path to the test suite YAML file
        tags: Tag filter expression
    """
    try:
        loader = TestLoader()
        suite = loader.load_file(suite_file)

        # Apply tag filtering
        if tags:
            suite = suite.filter_by_tags(tags)

        click.echo(f"Test Suite: {suite.test_suite}")
        click.echo(f"Version: {suite.version}")
        if suite.description:
            click.echo(f"Description: {suite.description}")

        click.echo(f"\nTests ({len(suite.tests)}):")
        for test in suite.tests:
            tags_str = ", ".join(test.tags) if test.tags else "no tags"
            click.echo(f"  {test.id}: {test.name}")
            click.echo(f"    Tags: {tags_str}")
            if test.description:
                click.echo(f"    Description: {test.description}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
