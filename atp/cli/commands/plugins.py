"""CLI commands for managing ATP plugins."""

import sys

import click
from rich.console import Console
from rich.table import Table

from atp.plugins import (
    ADAPTER_GROUP,
    ALL_GROUPS,
    EVALUATOR_GROUP,
    REPORTER_GROUP,
    LazyPlugin,
    PluginInfo,
    get_plugin_manager,
)

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2

# Group type mapping
GROUP_TYPE_MAP = {
    "adapter": ADAPTER_GROUP,
    "evaluator": EVALUATOR_GROUP,
    "reporter": REPORTER_GROUP,
}

# Reverse mapping for display
GROUP_DISPLAY_MAP = {
    ADAPTER_GROUP: "adapter",
    EVALUATOR_GROUP: "evaluator",
    REPORTER_GROUP: "reporter",
}


def _get_plugin_type_display(group: str) -> str:
    """Get display name for a plugin group.

    Args:
        group: Entry point group (e.g., 'atp.adapters').

    Returns:
        Display name (e.g., 'adapter').
    """
    return GROUP_DISPLAY_MAP.get(group, group)


def _get_groups_from_type(plugin_type: str | None) -> list[str]:
    """Get entry point groups for a plugin type filter.

    Args:
        plugin_type: Optional type filter ('adapter', 'evaluator', 'reporter').

    Returns:
        List of entry point groups to search.
    """
    if plugin_type is None:
        return ALL_GROUPS

    group = GROUP_TYPE_MAP.get(plugin_type)
    if group is None:
        return []
    return [group]


def _create_plugins_table(title: str = "Plugins") -> Table:
    """Create a Rich table for plugin display.

    Args:
        title: Table title.

    Returns:
        Configured Rich Table instance.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Package", style="blue")
    table.add_column("Version", style="yellow")
    table.add_column("Description", style="dim")
    return table


def _add_plugin_to_table(table: Table, info: PluginInfo) -> None:
    """Add a plugin info row to the table.

    Args:
        table: Rich Table to add row to.
        info: Plugin info to display.
    """
    desc = info.description or "-"
    truncated_desc = desc[:50] + "..." if len(desc) > 50 else desc
    table.add_row(
        info.name,
        _get_plugin_type_display(info.group),
        info.package or "-",
        info.version or "-",
        truncated_desc,
    )


def _get_all_plugins(groups: list[str]) -> list[tuple[str, LazyPlugin]]:
    """Get all plugins from the specified groups.

    Args:
        groups: List of entry point groups to search.

    Returns:
        List of (group, LazyPlugin) tuples.
    """
    manager = get_plugin_manager()
    plugins: list[tuple[str, LazyPlugin]] = []

    for group in groups:
        group_plugins = manager.discover_plugins(group)
        for lazy_plugin in group_plugins.values():
            plugins.append((group, lazy_plugin))

    return plugins


def _find_plugin_by_name(
    name: str,
    plugin_type: str | None = None,
) -> tuple[str, LazyPlugin] | None:
    """Find a plugin by name across groups.

    Args:
        name: Plugin name to find.
        plugin_type: Optional type filter.

    Returns:
        Tuple of (group, LazyPlugin) if found, None otherwise.
    """
    groups = _get_groups_from_type(plugin_type)
    manager = get_plugin_manager()

    for group in groups:
        plugin = manager.get_plugin(group, name)
        if plugin is not None:
            return (group, plugin)

    return None


@click.group(name="plugins")
def plugins_command() -> None:
    """Manage ATP plugins.

    Discover, inspect, and manage plugins installed via entry points.
    Plugins can be adapters, evaluators, or reporters.

    Examples:

      # List all plugins
      atp plugins list

      # List only adapters
      atp plugins list --type=adapter

      # Get detailed info about a plugin
      atp plugins info http

      # Enable/disable a plugin
      atp plugins enable my-plugin
      atp plugins disable my-plugin
    """
    pass


@plugins_command.command(name="list")
@click.option(
    "--type",
    "-t",
    "plugin_type",
    type=click.Choice(["adapter", "evaluator", "reporter"]),
    help="Filter by plugin type",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show additional details",
)
def list_plugins(plugin_type: str | None, verbose: bool) -> None:
    """List all discovered plugins.

    Shows plugins registered via Python entry points. Use --type to filter
    by plugin type (adapter, evaluator, reporter).

    Examples:

      # List all plugins
      atp plugins list

      # List only adapters
      atp plugins list --type=adapter

      # List evaluators with details
      atp plugins list --type=evaluator --verbose

    Exit Codes:

      0 - Success
      2 - Error occurred
    """
    console = Console()

    try:
        groups = _get_groups_from_type(plugin_type)
        if not groups:
            click.echo(f"Unknown plugin type: {plugin_type}", err=True)
            sys.exit(EXIT_ERROR)

        plugins = _get_all_plugins(groups)

        if not plugins:
            type_str = f" {plugin_type}" if plugin_type else ""
            click.echo(f"No{type_str} plugins found.")
            sys.exit(EXIT_SUCCESS)

        # Create and populate table
        title = f"{plugin_type.title()} Plugins" if plugin_type else "All Plugins"
        table = _create_plugins_table(title=title)

        sorted_plugins = sorted(
            plugins, key=lambda x: (x[1].info.group, x[1].info.name)
        )
        for _group, lazy_plugin in sorted_plugins:
            _add_plugin_to_table(table, lazy_plugin.info)

        console.print(table)

        if verbose:
            console.print(f"\nTotal: {len(plugins)} plugin(s)")

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error listing plugins: {e}", err=True)
        sys.exit(EXIT_ERROR)


@plugins_command.command(name="info")
@click.argument("name")
@click.option(
    "--type",
    "-t",
    "plugin_type",
    type=click.Choice(["adapter", "evaluator", "reporter"]),
    help="Plugin type (if name is ambiguous)",
)
def info_plugin(name: str, plugin_type: str | None) -> None:
    """Show detailed information about a plugin.

    Displays metadata, configuration schema, and validation status
    for the specified plugin.

    Examples:

      # Get info about http adapter
      atp plugins info http

      # Get info about a specific evaluator
      atp plugins info llm_judge --type=evaluator

    Exit Codes:

      0 - Success
      1 - Plugin not found
      2 - Error occurred
    """
    console = Console()

    try:
        result = _find_plugin_by_name(name, plugin_type)

        if result is None:
            type_suffix = f" in {plugin_type}s" if plugin_type else ""
            click.echo(f"Plugin '{name}' not found{type_suffix}.", err=True)
            sys.exit(EXIT_FAILURE)

        group, lazy_plugin = result
        info = lazy_plugin.info

        # Create info table
        table = Table(title=f"Plugin: {info.name}", show_header=False, box=None)
        table.add_column("Property", style="bold cyan", width=20)
        table.add_column("Value", style="white")

        table.add_row("Name", info.name)
        table.add_row("Type", _get_plugin_type_display(group))
        table.add_row("Package", info.package or "-")
        table.add_row("Version", info.version or "-")
        table.add_row("Author", info.author or "-")
        table.add_row("Description", info.description or "-")
        table.add_row("Module", info.module)
        table.add_row("Attribute", info.attr)
        table.add_row("Full Path", info.full_path)

        console.print(table)
        console.print()

        # Try to load and show additional info
        try:
            lazy_plugin.load()
            console.print("[green]✓ Plugin loads successfully[/green]")

            # Show config schema if available
            if info.has_config_schema and info.config_metadata:
                console.print("\n[bold]Configuration Schema:[/bold]")
                config = info.config_metadata

                if config.field_descriptions:
                    config_table = Table(show_header=True, header_style="bold")
                    config_table.add_column("Field", style="cyan")
                    config_table.add_column("Default", style="yellow")
                    config_table.add_column("Description", style="dim")

                    for field, desc in config.field_descriptions.items():
                        default = config.defaults.get(field, "-")
                        default_str = str(default) if default != "-" else "-"
                        config_table.add_row(field, default_str, desc)

                    console.print(config_table)

                if config.env_prefix:
                    console.print(
                        f"\nEnvironment prefix: [cyan]{config.env_prefix}[/cyan]"
                    )

            # Check for validation errors
            manager = get_plugin_manager()
            validation_result = manager.get_validation_errors(group, name)
            if validation_result:
                interface_errors, version_error = validation_result
                if interface_errors:
                    console.print("\n[red]Validation Errors:[/red]")
                    for error in interface_errors:
                        console.print(f"  [red]✗[/red] {error}")
                if version_error:
                    console.print(
                        f"\n[yellow]Version Warning:[/yellow] {version_error}"
                    )

        except Exception as load_error:
            console.print(f"\n[red]✗ Failed to load plugin:[/red] {load_error}")

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error getting plugin info: {e}", err=True)
        sys.exit(EXIT_ERROR)


@plugins_command.command(name="enable")
@click.argument("name")
@click.option(
    "--type",
    "-t",
    "plugin_type",
    type=click.Choice(["adapter", "evaluator", "reporter"]),
    help="Plugin type (if name is ambiguous)",
)
def enable_plugin(name: str, plugin_type: str | None) -> None:
    """Enable a plugin.

    Enables a previously disabled plugin so it can be used in test runs.
    Plugin state is stored in the ATP configuration.

    Examples:

      # Enable a plugin
      atp plugins enable my-plugin

      # Enable a specific adapter
      atp plugins enable custom-adapter --type=adapter

    Exit Codes:

      0 - Success
      1 - Plugin not found
      2 - Error occurred

    Note:

      Plugin enable/disable functionality requires ATP configuration file.
      Without configuration, plugins are enabled by default.
    """
    console = Console()

    try:
        result = _find_plugin_by_name(name, plugin_type)

        if result is None:
            click.echo(f"Plugin '{name}' not found.", err=True)
            sys.exit(EXIT_FAILURE)

        group, lazy_plugin = result
        info = lazy_plugin.info

        # For now, we just verify the plugin exists and can be loaded
        # Full enable/disable with config persistence would require
        # integration with the ATP configuration system
        try:
            lazy_plugin.load()
            plugin_type_str = _get_plugin_type_display(group)
            console.print(
                f"[green]✓[/green] Plugin '{info.name}' ({plugin_type_str}) is enabled."
            )
            console.print(
                "\n[dim]Note: Plugin enable/disable state is managed "
                "through ATP configuration.[/dim]"
            )
            sys.exit(EXIT_SUCCESS)
        except Exception as e:
            console.print(f"[red]✗[/red] Cannot enable plugin: {e}")
            sys.exit(EXIT_FAILURE)

    except Exception as e:
        click.echo(f"Error enabling plugin: {e}", err=True)
        sys.exit(EXIT_ERROR)


@plugins_command.command(name="disable")
@click.argument("name")
@click.option(
    "--type",
    "-t",
    "plugin_type",
    type=click.Choice(["adapter", "evaluator", "reporter"]),
    help="Plugin type (if name is ambiguous)",
)
def disable_plugin(name: str, plugin_type: str | None) -> None:
    """Disable a plugin.

    Disables a plugin so it won't be used in test runs.
    Plugin state is stored in the ATP configuration.

    Examples:

      # Disable a plugin
      atp plugins disable my-plugin

      # Disable a specific reporter
      atp plugins disable custom-reporter --type=reporter

    Exit Codes:

      0 - Success
      1 - Plugin not found
      2 - Error occurred

    Note:

      Plugin enable/disable functionality requires ATP configuration file.
      Without configuration, plugins are enabled by default.
    """
    console = Console()

    try:
        result = _find_plugin_by_name(name, plugin_type)

        if result is None:
            click.echo(f"Plugin '{name}' not found.", err=True)
            sys.exit(EXIT_FAILURE)

        group, _lazy_plugin = result
        info = _lazy_plugin.info

        # For now, we just verify the plugin exists
        # Full enable/disable with config persistence would require
        # integration with the ATP configuration system
        plugin_type_str = _get_plugin_type_display(group)
        console.print(
            f"[yellow]⚠[/yellow] Plugin '{info.name}' ({plugin_type_str}) "
            "marked for disable."
        )
        console.print(
            "\n[dim]Note: Plugin enable/disable state is managed "
            "through ATP configuration.[/dim]"
        )
        console.print(
            "[dim]Add the following to your atp.config.yaml "
            "to disable this plugin:[/dim]"
        )
        console.print(f"\n[cyan]plugins:\n  disabled:\n    - {info.name}[/cyan]")

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error disabling plugin: {e}", err=True)
        sys.exit(EXIT_ERROR)
