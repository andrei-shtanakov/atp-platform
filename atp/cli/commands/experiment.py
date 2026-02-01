"""CLI commands for managing A/B testing experiments."""

import asyncio
import json
import sys
from typing import Any

import click
from rich.console import Console
from rich.table import Table

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2


def _create_experiments_table(title: str = "Experiments") -> Table:
    """Create a Rich table for experiment display.

    Args:
        title: Table title.

    Returns:
        Configured Rich Table instance.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Suite", style="blue")
    table.add_column("Status", style="magenta")
    table.add_column("Control", style="dim")
    table.add_column("Treatment", style="dim")
    table.add_column("Samples", justify="right")
    table.add_column("Winner", style="yellow")
    return table


def _create_results_table(title: str = "Statistical Results") -> Table:
    """Create a Rich table for experiment results.

    Args:
        title: Table title.

    Returns:
        Configured Rich Table instance.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="green")
    table.add_column("Control Mean", justify="right")
    table.add_column("Treatment Mean", justify="right")
    table.add_column("Change %", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Significant", justify="center")
    table.add_column("Winner", style="yellow")
    return table


def _format_status(status: str) -> str:
    """Format experiment status with color.

    Args:
        status: Status string.

    Returns:
        Formatted status with Rich markup.
    """
    status_colors = {
        "draft": "[dim]DRAFT[/dim]",
        "running": "[green]RUNNING[/green]",
        "paused": "[yellow]PAUSED[/yellow]",
        "concluded": "[blue]CONCLUDED[/blue]",
        "rolled_back": "[bold red]ROLLED BACK[/bold red]",
    }
    return status_colors.get(status, status.upper())


def _format_winner(winner: str | None) -> str:
    """Format winner with color.

    Args:
        winner: Winner string.

    Returns:
        Formatted winner with Rich markup.
    """
    if winner is None:
        return "[dim]-[/dim]"
    winner_colors = {
        "control": "[blue]Control[/blue]",
        "treatment": "[green]Treatment[/green]",
        "tie": "[yellow]Tie[/yellow]",
        "inconclusive": "[dim]Inconclusive[/dim]",
    }
    return winner_colors.get(winner, winner)


@click.group(name="experiment")
def experiment_command() -> None:
    """Manage A/B testing experiments.

    A/B experiments allow you to compare agent versions with statistical rigor.
    Define traffic splits, track metrics, and determine winners with confidence.

    Examples:

      # List all experiments
      atp experiment list

      # Create a new experiment
      atp experiment create --name=test-v2 --suite=tests.yaml \\
          --control=agent-v1 --treatment=agent-v2

      # Start an experiment
      atp experiment start 1

      # Get experiment status
      atp experiment status 1

      # Conclude an experiment
      atp experiment conclude 1
    """
    pass


@experiment_command.command(name="list")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["draft", "running", "paused", "concluded", "rolled_back"]),
    help="Filter by experiment status",
)
@click.option(
    "--suite",
    type=str,
    help="Filter by test suite name",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
def list_experiments(
    status: str | None,
    suite: str | None,
    output: str,
) -> None:
    """List all A/B experiments.

    Shows all experiments with their status, variants, and results.

    Examples:

      # List all experiments
      atp experiment list

      # List only running experiments
      atp experiment list --status=running

      # List experiments for a specific suite
      atp experiment list --suite=tests.yaml

      # Output as JSON
      atp experiment list --output=json

    Exit Codes:

      0 - Success
      2 - Error occurred
    """
    console = Console()

    try:
        result = asyncio.run(_list_experiments_async(status, suite))

        if output == "json":
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            _output_experiments_console(result, console)

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error listing experiments: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _list_experiments_async(
    status: str | None,
    suite_name: str | None,
) -> list[dict[str, Any]]:
    """List experiments asynchronously.

    Args:
        status: Optional status filter.
        suite_name: Optional suite name filter.

    Returns:
        List of experiment dictionaries.
    """
    from atp.analytics.ab_testing import ExperimentStatus, get_experiment_manager

    manager = get_experiment_manager()

    status_enum = ExperimentStatus(status) if status else None
    experiments = manager.list_experiments(status=status_enum, suite_name=suite_name)

    return [
        {
            "id": e.id,
            "name": e.config.name,
            "suite_name": e.config.suite_name,
            "status": e.status.value,
            "control": e.config.control_variant.name,
            "treatment": e.config.treatment_variant.name,
            "control_samples": e.control_sample_size,
            "treatment_samples": e.treatment_sample_size,
            "winner": e.winner.value if e.winner else None,
            "created_at": e.created_at.isoformat() if e.created_at else None,
            "started_at": e.started_at.isoformat() if e.started_at else None,
            "concluded_at": e.concluded_at.isoformat() if e.concluded_at else None,
        }
        for e in experiments
    ]


def _output_experiments_console(
    experiments: list[dict[str, Any]],
    console: Console,
) -> None:
    """Output experiments to console.

    Args:
        experiments: List of experiment dictionaries.
        console: Rich console instance.
    """
    if not experiments:
        console.print("No experiments found.")
        return

    table = _create_experiments_table()

    for exp in experiments:
        samples = f"{exp['control_samples']}/{exp['treatment_samples']}"
        table.add_row(
            str(exp["id"]),
            exp["name"],
            exp["suite_name"],
            _format_status(exp["status"]),
            exp["control"],
            exp["treatment"],
            samples,
            _format_winner(exp["winner"]),
        )

    console.print(table)
    console.print(f"\nTotal: {len(experiments)} experiment(s)")


@experiment_command.command(name="create")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Experiment name (must be unique)",
)
@click.option(
    "--suite",
    "-s",
    type=str,
    required=True,
    help="Test suite to run the experiment on",
)
@click.option(
    "--control",
    "-c",
    type=str,
    required=True,
    help="Control variant agent name",
)
@click.option(
    "--treatment",
    "-t",
    type=str,
    required=True,
    help="Treatment variant agent name",
)
@click.option(
    "--traffic-split",
    type=str,
    default="50/50",
    help="Traffic split as control/treatment (e.g., 50/50, 80/20)",
)
@click.option(
    "--min-samples",
    type=int,
    default=30,
    help="Minimum samples per variant (default: 30)",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum total samples (auto-conclude when reached)",
)
@click.option(
    "--max-days",
    type=int,
    default=None,
    help="Maximum duration in days (auto-conclude when reached)",
)
@click.option(
    "--metric",
    type=click.Choice(["score", "success_rate", "duration", "cost"]),
    default="score",
    help="Primary metric to optimize (default: score)",
)
@click.option(
    "--rollback/--no-rollback",
    default=True,
    help="Enable automatic rollback on degradation",
)
@click.option(
    "--description",
    "-d",
    type=str,
    help="Experiment description",
)
def create_experiment(
    name: str,
    suite: str,
    control: str,
    treatment: str,
    traffic_split: str,
    min_samples: int,
    max_samples: int | None,
    max_days: int | None,
    metric: str,
    rollback: bool,
    description: str | None,
) -> None:
    """Create a new A/B experiment.

    Creates an experiment to compare two agent variants on a test suite.

    Examples:

      # Create a basic experiment
      atp experiment create --name=v2-test --suite=tests.yaml \\
          --control=agent-v1 --treatment=agent-v2

      # Create with custom traffic split
      atp experiment create --name=gradual-rollout --suite=tests.yaml \\
          --control=agent-v1 --treatment=agent-v2 --traffic-split=90/10

      # Create with auto-conclusion limits
      atp experiment create --name=quick-test --suite=tests.yaml \\
          --control=agent-v1 --treatment=agent-v2 \\
          --max-samples=1000 --max-days=7

    Exit Codes:

      0 - Experiment created successfully
      2 - Error occurred
    """
    console = Console()

    # Parse traffic split
    try:
        parts = traffic_split.split("/")
        control_weight = float(parts[0])
        treatment_weight = float(parts[1])
    except (ValueError, IndexError):
        click.echo(
            "Error: --traffic-split must be in format 'X/Y' (e.g., 50/50)",
            err=True,
        )
        sys.exit(EXIT_ERROR)

    try:
        result = asyncio.run(
            _create_experiment_async(
                name=name,
                suite_name=suite,
                control_name=control,
                treatment_name=treatment,
                control_weight=control_weight,
                treatment_weight=treatment_weight,
                min_samples=min_samples,
                max_samples=max_samples,
                max_days=max_days,
                metric=metric,
                rollback_enabled=rollback,
                description=description,
            )
        )

        console.print(f"[green]Experiment '{name}' created successfully.[/green]")
        console.print(f"  ID: {result['id']}")
        console.print(f"  Suite: {suite}")
        console.print(f"  Control: {control} ({control_weight:.0f}%)")
        console.print(f"  Treatment: {treatment} ({treatment_weight:.0f}%)")
        console.print(f"  Primary metric: {metric}")
        console.print(f"  Min samples per variant: {min_samples}")
        if max_samples:
            console.print(f"  Max samples: {max_samples}")
        if max_days:
            console.print(f"  Max duration: {max_days} days")
        console.print(f"  Rollback enabled: {rollback}")
        console.print("\nUse 'atp experiment start {id}' to begin the experiment.")

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error creating experiment: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _create_experiment_async(
    name: str,
    suite_name: str,
    control_name: str,
    treatment_name: str,
    control_weight: float,
    treatment_weight: float,
    min_samples: int,
    max_samples: int | None,
    max_days: int | None,
    metric: str,
    rollback_enabled: bool,
    description: str | None,
) -> dict[str, Any]:
    """Create an experiment asynchronously.

    Args:
        name: Experiment name.
        suite_name: Test suite name.
        control_name: Control agent name.
        treatment_name: Treatment agent name.
        control_weight: Control traffic weight.
        treatment_weight: Treatment traffic weight.
        min_samples: Minimum samples per variant.
        max_samples: Maximum samples.
        max_days: Maximum duration in days.
        metric: Primary metric.
        rollback_enabled: Whether rollback is enabled.
        description: Description.

    Returns:
        Created experiment info.
    """
    from atp.analytics.ab_testing import (
        ExperimentConfig,
        MetricConfig,
        MetricType,
        RollbackConfig,
        Variant,
        VariantType,
        get_experiment_manager,
    )

    manager = get_experiment_manager()

    config = ExperimentConfig(
        name=name,
        description=description,
        suite_name=suite_name,
        control_variant=Variant(
            name=control_name,
            variant_type=VariantType.CONTROL,
            agent_name=control_name,
            traffic_weight=control_weight,
        ),
        treatment_variant=Variant(
            name=treatment_name,
            variant_type=VariantType.TREATMENT,
            agent_name=treatment_name,
            traffic_weight=treatment_weight,
        ),
        metrics=[
            MetricConfig(
                metric_type=MetricType(metric),
                is_primary=True,
                minimize=(metric in ("duration", "cost")),
            )
        ],
        rollback=RollbackConfig(enabled=rollback_enabled),
        min_sample_size=min_samples,
        max_sample_size=max_samples,
        max_duration_days=max_days,
    )

    experiment = manager.create_experiment(config)

    return {
        "id": experiment.id,
        "name": experiment.config.name,
        "status": experiment.status.value,
    }


@experiment_command.command(name="start")
@click.argument("experiment_id", type=int)
def start_experiment(experiment_id: int) -> None:
    """Start an experiment (transition from draft to running).

    EXPERIMENT_ID is the ID of the experiment to start.

    Examples:

      atp experiment start 1

    Exit Codes:

      0 - Experiment started successfully
      1 - Experiment not found or cannot be started
      2 - Error occurred
    """
    console = Console()

    try:
        asyncio.run(_start_experiment_async(experiment_id))
        console.print(f"[green]Experiment {experiment_id} started.[/green]")
        console.print("Traffic is now being routed to both variants.")
        sys.exit(EXIT_SUCCESS)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error starting experiment: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _start_experiment_async(experiment_id: int) -> None:
    """Start an experiment asynchronously."""
    from atp.analytics.ab_testing import get_experiment_manager

    manager = get_experiment_manager()
    manager.start_experiment(experiment_id)


@experiment_command.command(name="pause")
@click.argument("experiment_id", type=int)
def pause_experiment(experiment_id: int) -> None:
    """Pause a running experiment.

    EXPERIMENT_ID is the ID of the experiment to pause.

    Examples:

      atp experiment pause 1

    Exit Codes:

      0 - Experiment paused successfully
      1 - Experiment not found or cannot be paused
      2 - Error occurred
    """
    console = Console()

    try:
        asyncio.run(_pause_experiment_async(experiment_id))
        console.print(f"[yellow]Experiment {experiment_id} paused.[/yellow]")
        console.print("Use 'atp experiment resume {id}' to resume.")
        sys.exit(EXIT_SUCCESS)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error pausing experiment: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _pause_experiment_async(experiment_id: int) -> None:
    """Pause an experiment asynchronously."""
    from atp.analytics.ab_testing import get_experiment_manager

    manager = get_experiment_manager()
    manager.pause_experiment(experiment_id)


@experiment_command.command(name="resume")
@click.argument("experiment_id", type=int)
def resume_experiment(experiment_id: int) -> None:
    """Resume a paused experiment.

    EXPERIMENT_ID is the ID of the experiment to resume.

    Examples:

      atp experiment resume 1

    Exit Codes:

      0 - Experiment resumed successfully
      1 - Experiment not found or cannot be resumed
      2 - Error occurred
    """
    console = Console()

    try:
        asyncio.run(_resume_experiment_async(experiment_id))
        console.print(f"[green]Experiment {experiment_id} resumed.[/green]")
        sys.exit(EXIT_SUCCESS)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error resuming experiment: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _resume_experiment_async(experiment_id: int) -> None:
    """Resume an experiment asynchronously."""
    from atp.analytics.ab_testing import get_experiment_manager

    manager = get_experiment_manager()
    manager.resume_experiment(experiment_id)


@experiment_command.command(name="conclude")
@click.argument("experiment_id", type=int)
@click.option(
    "--reason",
    "-r",
    type=str,
    default="Manual conclusion",
    help="Reason for concluding the experiment",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force conclusion even without minimum samples",
)
def conclude_experiment(
    experiment_id: int,
    reason: str,
    force: bool,
) -> None:
    """Conclude an experiment and determine the winner.

    EXPERIMENT_ID is the ID of the experiment to conclude.

    Examples:

      # Conclude with default reason
      atp experiment conclude 1

      # Conclude with custom reason
      atp experiment conclude 1 --reason="Reached statistical significance"

      # Force conclude without minimum samples
      atp experiment conclude 1 --force

    Exit Codes:

      0 - Experiment concluded successfully
      1 - Experiment not found or cannot be concluded
      2 - Error occurred
    """
    console = Console()

    try:
        result = asyncio.run(_conclude_experiment_async(experiment_id, reason, force))

        console.print(f"[blue]Experiment {experiment_id} concluded.[/blue]")
        console.print(f"  Reason: {reason}")
        console.print(f"  Winner: {_format_winner(result['winner'])}")

        if result["recommendation"]:
            console.print(f"\n{result['recommendation']}")

        sys.exit(EXIT_SUCCESS)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error concluding experiment: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _conclude_experiment_async(
    experiment_id: int,
    reason: str,
    force: bool,
) -> dict[str, Any]:
    """Conclude an experiment asynchronously."""
    from atp.analytics.ab_testing import get_experiment_manager

    manager = get_experiment_manager()
    experiment = manager.get_experiment(experiment_id)

    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found")

    if not force and not experiment.can_conclude:
        raise ValueError(
            f"Experiment needs at least {experiment.config.min_sample_size} "
            f"samples per variant. Current: control={experiment.control_sample_size}, "
            f"treatment={experiment.treatment_sample_size}. Use --force to override."
        )

    manager.conclude_experiment(experiment_id, reason)
    report = manager.generate_report(experiment_id)

    return {
        "winner": experiment.winner.value,
        "recommendation": report.recommendation,
    }


@experiment_command.command(name="status")
@click.argument("experiment_id", type=int)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
def experiment_status(
    experiment_id: int,
    output: str,
) -> None:
    """Show detailed status of an experiment.

    EXPERIMENT_ID is the ID of the experiment.

    Examples:

      # Show experiment status
      atp experiment status 1

      # Output as JSON
      atp experiment status 1 --output=json

    Exit Codes:

      0 - Success
      1 - Experiment not found
      2 - Error occurred
    """
    console = Console()

    try:
        result = asyncio.run(_get_experiment_status_async(experiment_id))

        if output == "json":
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            _output_status_console(result, console)

        sys.exit(EXIT_SUCCESS)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error getting experiment status: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _get_experiment_status_async(experiment_id: int) -> dict[str, Any]:
    """Get experiment status asynchronously."""
    from atp.analytics.ab_testing import get_experiment_manager

    manager = get_experiment_manager()
    report = manager.generate_report(experiment_id)

    results_data = []
    for r in report.statistical_results:
        results_data.append(
            {
                "metric": r.metric_type.value,
                "control_mean": r.control_metrics.mean,
                "control_std": r.control_metrics.std,
                "control_samples": r.control_metrics.sample_size,
                "treatment_mean": r.treatment_metrics.mean,
                "treatment_std": r.treatment_metrics.std,
                "treatment_samples": r.treatment_metrics.sample_size,
                "relative_change": r.relative_change,
                "p_value": r.p_value,
                "is_significant": r.is_significant,
                "effect_size": r.effect_size,
                "winner": r.winner.value,
            }
        )

    return {
        "experiment": report.summary,
        "results": results_data,
        "recommendation": report.recommendation,
    }


def _output_status_console(result: dict[str, Any], console: Console) -> None:
    """Output experiment status to console."""
    exp = result["experiment"]

    console.print(f"\n[bold]Experiment: {exp['experiment_name']}[/bold]")
    console.print(f"  Suite: {exp['suite_name']}")
    console.print(f"  Status: {_format_status(exp['status'])}")
    console.print(f"  Control: {exp['control_variant']}")
    console.print(f"  Treatment: {exp['treatment_variant']}")
    console.print(
        f"  Samples: {exp['control_samples']} (control) / "
        f"{exp['treatment_samples']} (treatment)"
    )
    console.print(f"  Winner: {_format_winner(exp['winner'])}")

    if exp.get("rollback_triggered"):
        console.print("[bold red]  ROLLBACK TRIGGERED[/bold red]")

    if exp.get("started_at"):
        console.print(f"  Started: {exp['started_at']}")
    if exp.get("concluded_at"):
        console.print(f"  Concluded: {exp['concluded_at']}")
    if exp.get("duration_days"):
        console.print(f"  Duration: {exp['duration_days']} days")

    # Show statistical results
    if result["results"]:
        console.print()
        table = _create_results_table()

        for r in result["results"]:
            change_str = f"{r['relative_change']:+.1f}%"
            if r["relative_change"] > 0:
                change_str = f"[green]{change_str}[/green]"
            elif r["relative_change"] < 0:
                change_str = f"[red]{change_str}[/red]"

            sig_str = "[green]Yes[/green]" if r["is_significant"] else "[dim]No[/dim]"

            table.add_row(
                r["metric"],
                f"{r['control_mean']:.4f}",
                f"{r['treatment_mean']:.4f}",
                change_str,
                f"{r['p_value']:.4f}",
                sig_str,
                _format_winner(r["winner"]),
            )

        console.print(table)

    # Show recommendation
    if result["recommendation"]:
        console.print(f"\n[bold]Recommendation:[/bold] {result['recommendation']}")


@experiment_command.command(name="delete")
@click.argument("experiment_id", type=int)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Delete without confirmation",
)
def delete_experiment(experiment_id: int, force: bool) -> None:
    """Delete an experiment.

    EXPERIMENT_ID is the ID of the experiment to delete.

    Note: Running experiments cannot be deleted. Pause or conclude them first.

    Examples:

      # Delete with confirmation
      atp experiment delete 1

      # Delete without confirmation
      atp experiment delete 1 --force

    Exit Codes:

      0 - Experiment deleted successfully
      1 - Experiment not found or operation cancelled
      2 - Error occurred
    """
    if not force:
        if not click.confirm(f"Delete experiment {experiment_id}?"):
            click.echo("Operation cancelled.")
            sys.exit(EXIT_FAILURE)

    try:
        asyncio.run(_delete_experiment_async(experiment_id))
        click.echo(f"Experiment {experiment_id} deleted successfully.")
        sys.exit(EXIT_SUCCESS)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error deleting experiment: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _delete_experiment_async(experiment_id: int) -> None:
    """Delete an experiment asynchronously."""
    from atp.analytics.ab_testing import ExperimentStatus, get_experiment_manager

    manager = get_experiment_manager()
    experiment = manager.get_experiment(experiment_id)

    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found")

    if experiment.status == ExperimentStatus.RUNNING:
        raise ValueError(
            "Cannot delete a running experiment. Pause or conclude it first."
        )

    # Remove from manager's internal dict
    if experiment_id in manager._experiments:
        del manager._experiments[experiment_id]


@experiment_command.command(name="report")
@click.argument("experiment_id", type=int)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
@click.option(
    "--output-file",
    type=click.Path(),
    help="Save report to file",
)
def experiment_report(
    experiment_id: int,
    output: str,
    output_file: str | None,
) -> None:
    """Generate a comprehensive experiment report.

    EXPERIMENT_ID is the ID of the experiment.

    Examples:

      # Show report in console
      atp experiment report 1

      # Save report as JSON
      atp experiment report 1 --output=json --output-file=report.json

    Exit Codes:

      0 - Success
      1 - Experiment not found
      2 - Error occurred
    """
    console = Console()

    try:
        result = asyncio.run(_get_experiment_status_async(experiment_id))

        if output == "json":
            json_output = json.dumps(result, indent=2, default=str)
            if output_file:
                with open(output_file, "w") as f:
                    f.write(json_output)
                console.print(f"Report saved to {output_file}")
            else:
                click.echo(json_output)
        else:
            _output_status_console(result, console)

        sys.exit(EXIT_SUCCESS)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error generating report: {e}", err=True)
        sys.exit(EXIT_ERROR)
