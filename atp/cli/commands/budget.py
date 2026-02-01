"""CLI commands for managing cost budgets and alerts."""

import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2


def _create_budgets_table(title: str = "Budgets") -> Table:
    """Create a Rich table for budget display.

    Args:
        title: Table title.

    Returns:
        Configured Rich Table instance.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Period", style="magenta")
    table.add_column("Limit (USD)", style="yellow", justify="right")
    table.add_column("Threshold", style="blue", justify="right")
    table.add_column("Active", style="dim")
    table.add_column("Description", style="dim")
    return table


def _create_status_table(title: str = "Budget Status") -> Table:
    """Create a Rich table for budget status display.

    Args:
        title: Table title.

    Returns:
        Configured Rich Table instance.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Period", style="magenta")
    table.add_column("Spent", style="yellow", justify="right")
    table.add_column("Limit", style="blue", justify="right")
    table.add_column("Remaining", style="cyan", justify="right")
    table.add_column("Usage", justify="right")
    table.add_column("Status")
    return table


def _format_usage(percentage: float, is_over_limit: bool) -> str:
    """Format usage percentage with color coding.

    Args:
        percentage: Usage percentage.
        is_over_limit: Whether budget is exceeded.

    Returns:
        Formatted percentage string with color.
    """
    if is_over_limit:
        return f"[bold red]{percentage:.1f}%[/bold red]"
    elif percentage >= 80:
        return f"[yellow]{percentage:.1f}%[/yellow]"
    else:
        return f"[green]{percentage:.1f}%[/green]"


def _format_status(is_over_limit: bool, is_over_threshold: bool) -> str:
    """Format status indicator.

    Args:
        is_over_limit: Whether budget is exceeded.
        is_over_threshold: Whether alert threshold is exceeded.

    Returns:
        Status string.
    """
    if is_over_limit:
        return "[bold red]EXCEEDED[/bold red]"
    elif is_over_threshold:
        return "[yellow]WARNING[/yellow]"
    else:
        return "[green]OK[/green]"


@click.group(name="budget")
def budget_command() -> None:
    """Manage cost budgets and alerts.

    Budgets allow you to set spending limits and receive alerts
    when approaching or exceeding those limits.

    Examples:

      # List all budgets
      atp budget list

      # Show current budget status
      atp budget status

      # Create a new daily budget
      atp budget create --name=daily-limit --period=daily --limit=100

      # Set budget thresholds from config file
      atp budget set-thresholds --config=cost.yaml
    """
    pass


@budget_command.command(name="list")
@click.option(
    "--period",
    "-p",
    type=click.Choice(["daily", "weekly", "monthly"]),
    help="Filter by budget period",
)
@click.option(
    "--active/--all",
    default=True,
    help="Show only active budgets (default) or all budgets",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
def list_budgets(
    period: str | None,
    active: bool,
    output: str,
) -> None:
    """List all configured budgets.

    Shows all budgets with their periods, limits, and thresholds.

    Examples:

      # List all active budgets
      atp budget list

      # List only daily budgets
      atp budget list --period=daily

      # List all budgets including inactive
      atp budget list --all

      # Output as JSON
      atp budget list --output=json

    Exit Codes:

      0 - Success
      2 - Error occurred
    """
    console = Console()

    try:
        result = asyncio.run(
            _list_budgets_async(
                period=period,
                is_active=active if active else None,
            )
        )

        if output == "json":
            _output_budgets_json(result)
        else:
            _output_budgets_console(result, console)

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error listing budgets: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _list_budgets_async(
    period: str | None,
    is_active: bool | None,
) -> list[dict[str, Any]]:
    """List budgets asynchronously.

    Args:
        period: Optional period filter.
        is_active: Optional active status filter.

    Returns:
        List of budget dictionaries.
    """
    from atp.analytics.budgets import BudgetPeriod, get_budget_manager

    manager = await get_budget_manager()
    budgets = await manager.list_budgets(
        period=BudgetPeriod(period) if period else None,
        is_active=is_active,
    )

    return [
        {
            "id": b.id,
            "name": b.name,
            "period": b.period,
            "limit_usd": str(b.limit_usd),
            "alert_threshold": b.alert_threshold,
            "is_active": b.is_active,
            "description": b.description,
            "alert_channels": b.alert_channels,
            "scope": b.scope,
        }
        for b in budgets
    ]


def _output_budgets_console(
    budgets: list[dict[str, Any]],
    console: Console,
) -> None:
    """Output budgets to console.

    Args:
        budgets: List of budget dictionaries.
        console: Rich console instance.
    """
    if not budgets:
        console.print("No budgets found.")
        return

    table = _create_budgets_table()

    for budget in budgets:
        active_str = "[green]Yes[/green]" if budget["is_active"] else "[dim]No[/dim]"
        desc = budget["description"] or ""
        if len(desc) > 40:
            desc = desc[:37] + "..."

        table.add_row(
            budget["name"],
            budget["period"],
            f"${Decimal(budget['limit_usd']):,.2f}",
            f"{budget['alert_threshold'] * 100:.0f}%",
            active_str,
            desc,
        )

    console.print(table)
    console.print(f"\nTotal: {len(budgets)} budget(s)")


def _output_budgets_json(budgets: list[dict[str, Any]]) -> None:
    """Output budgets as JSON.

    Args:
        budgets: List of budget dictionaries.
    """
    click.echo(json.dumps(budgets, indent=2))


@budget_command.command(name="status")
@click.option(
    "--name",
    "-n",
    type=str,
    help="Show status for a specific budget",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
@click.option(
    "--alert/--no-alert",
    default=False,
    help="Send alerts for threshold violations",
)
def budget_status(
    name: str | None,
    output: str,
    alert: bool,
) -> None:
    """Show current budget status.

    Displays spending against budget limits for all active budgets.

    Examples:

      # Show status of all budgets
      atp budget status

      # Show status of a specific budget
      atp budget status --name=daily-limit

      # Show status and send alerts for violations
      atp budget status --alert

      # Output as JSON
      atp budget status --output=json

    Exit Codes:

      0 - All budgets within limits
      1 - One or more budgets exceeded
      2 - Error occurred
    """
    console = Console()

    try:
        result = asyncio.run(_check_budgets_async(name, alert))

        if output == "json":
            _output_status_json(result)
        else:
            _output_status_console(result, console)

        # Exit with failure if any budget exceeded
        if result["has_exceeded"]:
            sys.exit(EXIT_FAILURE)
        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error checking budget status: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _check_budgets_async(
    name: str | None,
    send_alerts: bool,
) -> dict[str, Any]:
    """Check budget status asynchronously.

    Args:
        name: Optional budget name filter.
        send_alerts: Whether to send alerts.

    Returns:
        Budget status dictionary.
    """
    from atp.analytics.budgets import get_budget_manager

    manager = await get_budget_manager()

    if send_alerts:
        result = await manager.check_and_alert()
    else:
        result = await manager.check_budgets()

    statuses = [
        {
            "budget_id": s.budget_id,
            "budget_name": s.budget_name,
            "period": s.period.value,
            "period_start": s.period_start.isoformat(),
            "limit_usd": str(s.limit),
            "spent_usd": str(s.spent),
            "remaining_usd": str(s.remaining),
            "percentage": s.percentage,
            "is_over_threshold": s.is_over_threshold,
            "is_over_limit": s.is_over_limit,
            "triggered_alerts": s.triggered_alerts,
        }
        for s in result.statuses
        if name is None or s.budget_name == name
    ]

    return {
        "timestamp": result.timestamp.isoformat(),
        "has_alerts": result.has_alerts,
        "has_exceeded": result.has_exceeded,
        "statuses": statuses,
    }


def _output_status_console(
    result: dict[str, Any],
    console: Console,
) -> None:
    """Output budget status to console.

    Args:
        result: Budget status dictionary.
        console: Rich console instance.
    """
    statuses = result["statuses"]

    if not statuses:
        console.print("No budget status available.")
        return

    table = _create_status_table()

    for status in statuses:
        percentage = status["percentage"]
        is_over_limit = status["is_over_limit"]
        is_over_threshold = status["is_over_threshold"]

        table.add_row(
            status["budget_name"],
            status["period"],
            f"${Decimal(status['spent_usd']):,.2f}",
            f"${Decimal(status['limit_usd']):,.2f}",
            f"${Decimal(status['remaining_usd']):,.2f}",
            _format_usage(percentage, is_over_limit),
            _format_status(is_over_limit, is_over_threshold),
        )

    console.print(table)
    console.print(f"\nChecked at: {result['timestamp']}")

    if result["has_exceeded"]:
        console.print("\n[bold red]WARNING: One or more budgets exceeded![/bold red]")
    elif result["has_alerts"]:
        console.print(
            "\n[yellow]NOTICE: One or more budgets approaching limit.[/yellow]"
        )


def _output_status_json(result: dict[str, Any]) -> None:
    """Output budget status as JSON.

    Args:
        result: Budget status dictionary.
    """
    click.echo(json.dumps(result, indent=2))


@budget_command.command(name="create")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Budget name (must be unique)",
)
@click.option(
    "--period",
    "-p",
    type=click.Choice(["daily", "weekly", "monthly"]),
    required=True,
    help="Budget period",
)
@click.option(
    "--limit",
    "-l",
    type=float,
    required=True,
    help="Budget limit in USD",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.8,
    help="Alert threshold (0.0-1.0, default 0.8)",
)
@click.option(
    "--channels",
    "-c",
    type=str,
    multiple=True,
    help="Alert channels (log, webhook, email)",
)
@click.option(
    "--description",
    "-d",
    type=str,
    help="Budget description",
)
@click.option(
    "--scope",
    type=str,
    multiple=True,
    help="Scope filters as key=value pairs (provider=anthropic, model=claude-3)",
)
def create_budget(
    name: str,
    period: str,
    limit: float,
    threshold: float,
    channels: tuple[str, ...],
    description: str | None,
    scope: tuple[str, ...],
) -> None:
    """Create a new budget.

    Creates a new cost budget with the specified parameters.

    Examples:

      # Create a daily budget of $100
      atp budget create --name=daily-limit --period=daily --limit=100

      # Create a monthly budget with custom threshold
      atp budget create --name=monthly-cap --period=monthly --limit=2000 --threshold=0.9

      # Create a budget with alert channels
      atp budget create --name=team-budget --period=weekly --limit=500 \\
          --channels=log --channels=webhook

      # Create a scoped budget for a specific provider
      atp budget create --name=anthropic-daily --period=daily --limit=50 \\
          --scope=provider=anthropic

    Exit Codes:

      0 - Budget created successfully
      2 - Error occurred
    """
    console = Console()

    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        click.echo("Error: --threshold must be between 0.0 and 1.0", err=True)
        sys.exit(EXIT_ERROR)

    # Parse scope
    scope_dict: dict[str, str] = {}
    for item in scope:
        if "=" in item:
            key, value = item.split("=", 1)
            scope_dict[key] = value

    try:
        asyncio.run(
            _create_budget_async(
                name=name,
                period=period,
                limit=Decimal(str(limit)),
                threshold=threshold,
                channels=list(channels) if channels else None,
                description=description,
                scope=scope_dict if scope_dict else None,
            )
        )

        console.print(f"[green]Budget '{name}' created successfully.[/green]")
        console.print(f"  Period: {period}")
        console.print(f"  Limit: ${limit:,.2f}")
        console.print(f"  Alert threshold: {threshold * 100:.0f}%")
        if channels:
            console.print(f"  Alert channels: {', '.join(channels)}")
        if scope_dict:
            console.print(f"  Scope: {scope_dict}")

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error creating budget: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _create_budget_async(
    name: str,
    period: str,
    limit: Decimal,
    threshold: float,
    channels: list[str] | None,
    description: str | None,
    scope: dict[str, str] | None,
) -> dict[str, Any]:
    """Create a budget asynchronously.

    Args:
        name: Budget name.
        period: Budget period.
        limit: Budget limit.
        threshold: Alert threshold.
        channels: Alert channels.
        description: Budget description.
        scope: Scope filters.

    Returns:
        Created budget info.
    """
    from atp.analytics.budgets import BudgetPeriod, get_budget_manager

    manager = await get_budget_manager()
    budget = await manager.create_budget(
        name=name,
        period=BudgetPeriod(period),
        limit_usd=limit,
        alert_threshold=threshold,
        scope=scope,
        alert_channels=channels,
        description=description,
    )

    return {
        "id": budget.id,
        "name": budget.name,
        "period": budget.period,
        "limit_usd": str(budget.limit_usd),
    }


@budget_command.command(name="update")
@click.argument("name")
@click.option(
    "--limit",
    "-l",
    type=float,
    help="New budget limit in USD",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    help="New alert threshold (0.0-1.0)",
)
@click.option(
    "--channels",
    "-c",
    type=str,
    multiple=True,
    help="New alert channels (replaces existing)",
)
@click.option(
    "--activate/--deactivate",
    default=None,
    help="Activate or deactivate the budget",
)
def update_budget(
    name: str,
    limit: float | None,
    threshold: float | None,
    channels: tuple[str, ...],
    activate: bool | None,
) -> None:
    """Update an existing budget.

    NAME is the budget name to update.

    Examples:

      # Update budget limit
      atp budget update daily-limit --limit=150

      # Update alert threshold
      atp budget update daily-limit --threshold=0.9

      # Deactivate a budget
      atp budget update old-budget --deactivate

      # Update multiple settings
      atp budget update team-budget --limit=1000 --channels=log --channels=email

    Exit Codes:

      0 - Budget updated successfully
      1 - Budget not found
      2 - Error occurred
    """
    console = Console()

    # Validate threshold if provided
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        click.echo("Error: --threshold must be between 0.0 and 1.0", err=True)
        sys.exit(EXIT_ERROR)

    try:
        result = asyncio.run(
            _update_budget_async(
                name=name,
                limit=Decimal(str(limit)) if limit is not None else None,
                threshold=threshold,
                channels=list(channels) if channels else None,
                is_active=activate,
            )
        )

        if result is None:
            click.echo(f"Budget not found: {name}", err=True)
            sys.exit(EXIT_FAILURE)

        console.print(f"[green]Budget '{name}' updated successfully.[/green]")
        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error updating budget: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _update_budget_async(
    name: str,
    limit: Decimal | None,
    threshold: float | None,
    channels: list[str] | None,
    is_active: bool | None,
) -> dict[str, Any] | None:
    """Update a budget asynchronously.

    Args:
        name: Budget name.
        limit: New limit.
        threshold: New threshold.
        channels: New alert channels.
        is_active: New active status.

    Returns:
        Updated budget info or None if not found.
    """
    from atp.analytics.budgets import get_budget_manager

    manager = await get_budget_manager()
    budget = await manager.update_budget(
        name=name,
        limit_usd=limit,
        alert_threshold=threshold,
        alert_channels=channels,
        is_active=is_active,
    )

    if budget is None:
        return None

    return {
        "id": budget.id,
        "name": budget.name,
    }


@budget_command.command(name="delete")
@click.argument("name")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Delete without confirmation",
)
def delete_budget(name: str, force: bool) -> None:
    """Delete a budget.

    NAME is the budget name to delete.

    Examples:

      # Delete a budget (with confirmation)
      atp budget delete old-budget

      # Delete without confirmation
      atp budget delete old-budget --force

    Exit Codes:

      0 - Budget deleted successfully
      1 - Budget not found or operation cancelled
      2 - Error occurred
    """
    # Confirm deletion
    if not force:
        if not click.confirm(f"Delete budget '{name}'?"):
            click.echo("Operation cancelled.")
            sys.exit(EXIT_FAILURE)

    try:
        success = asyncio.run(_delete_budget_async(name))

        if not success:
            click.echo(f"Budget not found: {name}", err=True)
            sys.exit(EXIT_FAILURE)

        click.echo(f"Budget '{name}' deleted successfully.")
        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error deleting budget: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _delete_budget_async(name: str) -> bool:
    """Delete a budget asynchronously.

    Args:
        name: Budget name.

    Returns:
        True if deleted, False if not found.
    """
    from atp.analytics.budgets import get_budget_manager

    manager = await get_budget_manager()
    return await manager.delete_budget(name)


@budget_command.command(name="set-thresholds")
@click.option(
    "--config",
    "-c",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to budget configuration YAML file",
)
def set_thresholds(config_file: Path) -> None:
    """Configure budgets from a YAML file.

    Reads budget configuration from a YAML file and creates or updates
    budgets accordingly.

    Configuration file format:

      cost:
        budgets:
          daily: 100.00
          monthly: 2000.00
        alerts:
          - threshold: 0.8
            channels: ["log", "webhook"]
          - threshold: 1.0
            channels: ["log", "email"]

    Examples:

      # Apply budget configuration
      atp budget set-thresholds --config=cost.yaml

    Exit Codes:

      0 - Configuration applied successfully
      2 - Error occurred
    """
    console = Console()

    try:
        result = asyncio.run(_set_thresholds_async(config_file))

        console.print(f"[green]Budget configuration applied from {config_file}[/green]")
        if result.get("daily"):
            console.print(f"  Daily budget: ${result['daily']:,.2f}")
        if result.get("weekly"):
            console.print(f"  Weekly budget: ${result['weekly']:,.2f}")
        if result.get("monthly"):
            console.print(f"  Monthly budget: ${result['monthly']:,.2f}")
        if result.get("alerts"):
            console.print(f"  Alert thresholds: {result['alerts']}")

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error applying configuration: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _set_thresholds_async(config_file: Path) -> dict[str, Any]:
    """Set thresholds from config file asynchronously.

    Args:
        config_file: Path to configuration file.

    Returns:
        Applied configuration summary.
    """
    from atp.analytics.budgets import BudgetConfig, get_budget_manager

    config = BudgetConfig.from_yaml(config_file)
    manager = await get_budget_manager()
    await manager.initialize(config)

    return {
        "daily": float(config.daily) if config.daily else None,
        "weekly": float(config.weekly) if config.weekly else None,
        "monthly": float(config.monthly) if config.monthly else None,
        "alerts": len(config.alerts),
    }


@budget_command.command(name="check")
@click.option(
    "--fail-on-exceeded",
    is_flag=True,
    help="Exit with non-zero status if any budget is exceeded",
)
@click.option(
    "--fail-on-warning",
    is_flag=True,
    help="Exit with non-zero status if any budget threshold is reached",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Only output if there are issues",
)
def check_budgets(
    fail_on_exceeded: bool,
    fail_on_warning: bool,
    quiet: bool,
) -> None:
    """Check budgets for CI/CD integration.

    Checks all active budgets and exits with appropriate status code
    based on budget status. Useful for CI/CD pipelines.

    Examples:

      # Check budgets and fail if any exceeded
      atp budget check --fail-on-exceeded

      # Check budgets and fail if any threshold reached
      atp budget check --fail-on-warning

      # Quiet mode - only output if issues
      atp budget check --fail-on-exceeded --quiet

    Exit Codes:

      0 - All budgets OK
      1 - Budget issue detected (based on flags)
      2 - Error occurred
    """
    console = Console()

    try:
        result = asyncio.run(_check_budgets_async(None, send_alerts=False))

        if result["has_exceeded"]:
            if not quiet:
                console.print("[bold red]BUDGET EXCEEDED[/bold red]")
                for status in result["statuses"]:
                    if status["is_over_limit"]:
                        console.print(
                            f"  {status['budget_name']}: "
                            f"${Decimal(status['spent_usd']):,.2f} / "
                            f"${Decimal(status['limit_usd']):,.2f} "
                            f"({status['percentage']:.1f}%)"
                        )
            if fail_on_exceeded:
                sys.exit(EXIT_FAILURE)

        elif result["has_alerts"]:
            if not quiet:
                console.print("[yellow]BUDGET WARNING[/yellow]")
                for status in result["statuses"]:
                    if status["is_over_threshold"]:
                        console.print(
                            f"  {status['budget_name']}: "
                            f"${Decimal(status['spent_usd']):,.2f} / "
                            f"${Decimal(status['limit_usd']):,.2f} "
                            f"({status['percentage']:.1f}%)"
                        )
            if fail_on_warning:
                sys.exit(EXIT_FAILURE)

        elif not quiet:
            console.print("[green]All budgets OK[/green]")

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error checking budgets: {e}", err=True)
        sys.exit(EXIT_ERROR)
