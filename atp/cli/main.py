"""Main CLI entry point for ATP."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml

from atp import __version__
from atp.cli.commands.benchmark import benchmark_command
from atp.cli.commands.budget import budget_command
from atp.cli.commands.experiment import experiment_command
from atp.cli.commands.generate import generate_command
from atp.cli.commands.init import init_command
from atp.cli.commands.plugins import plugins_command
from atp.loader import TestLoader

# Exit codes as per requirements
EXIT_SUCCESS = 0  # All tests passed
EXIT_FAILURE = 1  # Test failures detected
EXIT_ERROR = 2  # Error (invalid config, missing file, etc.)


class ConfigContext:
    """Context object to hold configuration state."""

    def __init__(self) -> None:
        """Initialize config context."""
        self.config: dict[str, Any] = {}
        self.config_file: Path | None = None
        self.verbose: bool = False

    def load_config(self, config_path: Path | None = None) -> None:
        """Load configuration from file.

        Args:
            config_path: Optional path to config file. If not provided,
                searches for atp.config.yaml in current directory and parents.
        """
        if config_path:
            if config_path.exists():
                self._load_file(config_path)
            else:
                raise click.ClickException(f"Config file not found: {config_path}")
        else:
            # Search for config file
            search_path = Path.cwd()
            for _ in range(10):  # Limit search depth
                config_file = search_path / "atp.config.yaml"
                if config_file.exists():
                    self._load_file(config_file)
                    break
                # Also check for .yaml extension
                config_file = search_path / "atp.config.yml"
                if config_file.exists():
                    self._load_file(config_file)
                    break
                parent = search_path.parent
                if parent == search_path:
                    break
                search_path = parent

    def _load_file(self, path: Path) -> None:
        """Load config from a specific file."""
        try:
            with open(path) as f:
                self.config = yaml.safe_load(f) or {}
            self.config_file = path
        except yaml.YAMLError as e:
            raise click.ClickException(f"Invalid YAML in config file: {e}")
        except OSError as e:
            raise click.ClickException(f"Cannot read config file: {e}")

    def get_default(self, key: str, default: Any = None) -> Any:
        """Get a default value from config."""
        defaults = self.config.get("defaults", {})
        return defaults.get(key, default)

    def get_agent_config(self, agent_name: str) -> dict[str, Any] | None:
        """Get agent configuration by name."""
        agents = self.config.get("agents", {})
        return agents.get(agent_name)


pass_config = click.make_pass_decorator(ConfigContext, ensure=True)


@click.group(invoke_without_command=True)
@click.option(
    "--config",
    "-c",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to atp.config.yaml configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.version_option(version=__version__, prog_name="atp")
@click.pass_context
def cli(ctx: click.Context, config_file: Path | None, verbose: bool) -> None:
    """ATP - Agent Test Platform CLI.

    A framework-agnostic platform for testing and evaluating AI agents.

    Examples:

      # Run tests from a suite file
      atp test suite.yaml --agent=my-agent

      # Run tests with specific adapter
      atp test suite.yaml --adapter=http --adapter-config endpoint=http://localhost:8000

      # List available tests
      atp test suite.yaml --list-only

      # Validate agent configuration
      atp validate --agent=my-agent

      # List available adapters
      atp list-agents

    For more information, visit: https://github.com/your-org/atp
    """
    ctx.ensure_object(ConfigContext)
    config_ctx = ctx.obj
    config_ctx.verbose = verbose

    try:
        config_ctx.load_config(config_file)
    except click.ClickException:
        if config_file:  # Only raise if explicitly specified
            raise

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command(name="version")
def version_cmd() -> None:
    """Show ATP version information.

    Examples:

      atp version
    """
    click.echo(f"ATP - Agent Test Platform v{__version__}")
    click.echo(f"Python: {sys.version.split()[0]}")
    click.echo(f"Platform: {sys.platform}")


@cli.command(name="test")
@click.argument("suite_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--agent",
    "--agent-name",
    "agent_name",
    type=str,
    default="test-agent",
    help="Name of the agent being tested (from config or identifier)",
)
@click.option(
    "--tags",
    type=str,
    help=(
        "Filter tests by tags. "
        "Use comma-separated values for multiple tags (OR logic). "
        "Prefix with '!' to exclude. "
        "Examples: --tags=smoke, --tags=smoke,core, --tags=!slow"
    ),
)
@click.option(
    "--list-only",
    is_flag=True,
    help="List matching tests without running them",
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help=(
        "Maximum number of tests to run in parallel. "
        "Default is 1 (sequential execution)."
    ),
)
@click.option(
    "--adapter",
    type=str,
    default="http",
    help="Adapter type to use (http, cli, container, langgraph, crewai, autogen, mcp)",
)
@click.option(
    "--adapter-config",
    type=str,
    multiple=True,
    help="Adapter configuration as key=value pairs",
)
@click.option(
    "--runs",
    type=int,
    default=1,
    help="Number of runs per test",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop execution on first failure",
)
@click.option(
    "--sandbox/--no-sandbox",
    default=False,
    help="Enable sandbox isolation for tests",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--output",
    type=click.Choice(["console", "json", "junit"]),
    default="console",
    help="Output format (console, json, or junit)",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    help="Output file path (for json output)",
)
@click.option(
    "--no-save",
    is_flag=True,
    help="Don't save results to dashboard database",
)
@pass_config
def test_cmd(
    config_ctx: ConfigContext,
    suite_file: Path,
    agent_name: str,
    tags: str | None,
    list_only: bool,
    parallel: int,
    adapter: str,
    adapter_config: tuple[str, ...],
    runs: int,
    fail_fast: bool,
    sandbox: bool,
    verbose: bool,
    output: str,
    output_file: Path | None,
    no_save: bool,
) -> None:
    """Run tests from a test suite file.

    SUITE_FILE is the path to a YAML test suite definition.

    Examples:

      # Run all tests in a suite
      atp test tests/suite.yaml

      # Run tests with a specific agent
      atp test tests/suite.yaml --agent=my-agent

      # Run only smoke tests
      atp test tests/suite.yaml --tags=smoke

      # Run tests in parallel
      atp test tests/suite.yaml --parallel=4

      # Run tests with multiple runs for statistics
      atp test tests/suite.yaml --runs=5

      # Output results as JSON
      atp test tests/suite.yaml --output=json --output-file=results.json

    Exit Codes:

      0 - All tests passed
      1 - One or more tests failed
      2 - Error occurred (invalid config, file not found, etc.)
    """
    # Apply config defaults
    verbose = verbose or config_ctx.verbose
    parallel = parallel or config_ctx.get_default("parallel_workers", 1)
    runs = runs or config_ctx.get_default("runs_per_test", 1)

    # Validate options
    if parallel < 1:
        click.echo("Error: --parallel must be at least 1", err=True)
        sys.exit(EXIT_ERROR)

    if runs < 1:
        click.echo("Error: --runs must be at least 1", err=True)
        sys.exit(EXIT_ERROR)

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
            sys.exit(EXIT_FAILURE)

        # List tests
        if list_only:
            click.echo(f"Test Suite: {suite.test_suite}")
            click.echo(f"Tests ({len(suite.tests)}):")
            for test in suite.tests:
                tags_str = ", ".join(test.tags) if test.tags else "no tags"
                click.echo(f"  - {test.id}: {test.name} [{tags_str}]")
            return

        # Merge adapter config from config file with CLI options
        config_dict: dict[str, Any] = {}

        # First, try to get agent config from config file
        agent_cfg = config_ctx.get_agent_config(agent_name)
        if agent_cfg:
            if "type" in agent_cfg:
                adapter = agent_cfg["type"]
            config_dict.update(
                {k: v for k, v in agent_cfg.items() if k not in ("type", "name")}
            )

        # Then apply CLI adapter-config options (override file config)
        for item in adapter_config:
            if "=" in item:
                key, value = item.split("=", 1)
                # Try to parse as JSON for complex values
                try:
                    config_dict[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config_dict[key] = value

        # Run tests
        result = asyncio.run(
            _run_suite(
                suite=suite,
                adapter_type=adapter,
                adapter_config=config_dict,
                agent_name=agent_name,
                parallel=parallel,
                runs_per_test=runs,
                fail_fast=fail_fast,
                sandbox_enabled=sandbox,
                verbose=verbose,
                output_format=output,
                output_file=output_file,
                save_to_db=not no_save,
            )
        )

        # Exit with appropriate code
        sys.exit(EXIT_SUCCESS if result else EXIT_FAILURE)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


# Alias 'run' command to 'test' for backward compatibility
@cli.command(name="run")
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
@click.option(
    "--parallel",
    type=int,
    default=1,
    help=(
        "Maximum number of tests to run in parallel. "
        "Default is 1 (sequential execution)."
    ),
)
@click.option(
    "--adapter",
    type=str,
    default="http",
    help="Adapter type to use (http, cli, container, langgraph, crewai, autogen, mcp)",
)
@click.option(
    "--adapter-config",
    type=str,
    multiple=True,
    help="Adapter configuration as key=value pairs",
)
@click.option(
    "--agent-name",
    type=str,
    default="test-agent",
    help="Name of the agent being tested",
)
@click.option(
    "--runs",
    type=int,
    default=1,
    help="Number of runs per test",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop execution on first failure",
)
@click.option(
    "--sandbox/--no-sandbox",
    default=False,
    help="Enable sandbox isolation for tests",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--output",
    type=click.Choice(["console", "json", "junit"]),
    default="console",
    help="Output format (console, json, or junit)",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    help="Output file path (for json output)",
)
@click.option(
    "--no-save",
    is_flag=True,
    help="Don't save results to dashboard database",
)
@pass_config
def run(
    config_ctx: ConfigContext,
    suite_file: Path,
    tags: str | None,
    list_only: bool,
    parallel: int,
    adapter: str,
    adapter_config: tuple[str, ...],
    agent_name: str,
    runs: int,
    fail_fast: bool,
    sandbox: bool,
    verbose: bool,
    output: str,
    output_file: Path | None,
    no_save: bool,
) -> None:
    """Run tests from a test suite file (alias for 'test' command).

    This command is provided for backward compatibility.
    Use 'atp test' for new scripts.
    """
    # Apply config defaults
    verbose = verbose or config_ctx.verbose
    parallel = parallel or config_ctx.get_default("parallel_workers", 1)
    runs = runs or config_ctx.get_default("runs_per_test", 1)

    # Validate parallel option
    if parallel < 1:
        click.echo("Error: --parallel must be at least 1", err=True)
        sys.exit(EXIT_FAILURE)

    if runs < 1:
        click.echo("Error: --runs must be at least 1", err=True)
        sys.exit(EXIT_FAILURE)

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
            sys.exit(EXIT_FAILURE)

        # List tests
        if list_only:
            click.echo(f"Test Suite: {suite.test_suite}")
            click.echo(f"Tests ({len(suite.tests)}):")
            for test in suite.tests:
                tags_str = ", ".join(test.tags) if test.tags else "no tags"
                click.echo(f"  - {test.id}: {test.name} [{tags_str}]")
            return

        # Parse adapter config
        config_dict: dict[str, Any] = {}

        # First, try to get agent config from config file
        agent_cfg = config_ctx.get_agent_config(agent_name)
        if agent_cfg:
            if "type" in agent_cfg:
                adapter = agent_cfg["type"]
            config_dict.update(
                {k: v for k, v in agent_cfg.items() if k not in ("type", "name")}
            )

        for item in adapter_config:
            if "=" in item:
                key, value = item.split("=", 1)
                # Try to parse as JSON for complex values
                try:
                    config_dict[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config_dict[key] = value

        # Run tests
        result = asyncio.run(
            _run_suite(
                suite=suite,
                adapter_type=adapter,
                adapter_config=config_dict,
                agent_name=agent_name,
                parallel=parallel,
                runs_per_test=runs,
                fail_fast=fail_fast,
                sandbox_enabled=sandbox,
                verbose=verbose,
                output_format=output,
                output_file=output_file,
                save_to_db=not no_save,
            )
        )

        # Exit with appropriate code
        sys.exit(EXIT_SUCCESS if result else EXIT_FAILURE)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_FAILURE)


async def _run_suite(
    suite: Any,
    adapter_type: str,
    adapter_config: dict[str, Any],
    agent_name: str,
    parallel: int,
    runs_per_test: int,
    fail_fast: bool,
    sandbox_enabled: bool,
    verbose: bool,
    output_format: str,
    output_file: Path | None,
    save_to_db: bool = True,
) -> bool:
    """Run a test suite asynchronously.

    Args:
        suite: Test suite to run
        adapter_type: Type of adapter to use
        adapter_config: Configuration for the adapter
        agent_name: Name of the agent being tested
        parallel: Maximum parallel tests
        runs_per_test: Number of runs per test
        fail_fast: Stop on first failure
        sandbox_enabled: Enable sandbox isolation
        verbose: Enable verbose output
        output_format: Output format (console or json)
        output_file: Output file path (for json output)
        save_to_db: Whether to save results to dashboard database

    Returns:
        True if all tests passed, False otherwise
    """

    from atp.adapters import create_adapter
    from atp.reporters import SuiteReport, create_reporter
    from atp.runner import (
        SandboxConfig,
        TestOrchestrator,
        create_progress_callback,
    )

    # Create adapter
    adapter = create_adapter(adapter_type, adapter_config)

    # Create sandbox config
    sandbox_config = SandboxConfig(enabled=sandbox_enabled)

    # Create progress callback
    progress_callback = create_progress_callback(
        max_parallel=parallel,
        verbose=verbose,
        use_colors=True,
    )

    # Determine if parallel execution should be used
    use_parallel = parallel > 1 and not fail_fast

    # Create and run orchestrator
    async with TestOrchestrator(
        adapter=adapter,
        sandbox_config=sandbox_config,
        progress_callback=progress_callback,
        runs_per_test=runs_per_test,
        fail_fast=fail_fast,
        parallel_tests=use_parallel,
        max_parallel_tests=parallel,
    ) as orchestrator:
        result = await orchestrator.run_suite(
            suite=suite,
            agent_name=agent_name,
            runs_per_test=runs_per_test,
        )

    # Generate report
    reporter_config: dict[str, Any] = {
        "verbose": verbose,
        "use_colors": True,
    }

    if output_file:
        reporter_config["output_file"] = output_file

    reporter = create_reporter(output_format, reporter_config)
    report = SuiteReport.from_suite_result(result)
    reporter.report(report)

    # Save results to dashboard database
    if save_to_db:
        await _save_results_to_db(
            result=result,
            suite_name=suite.test_suite,
            agent_name=agent_name,
            adapter_type=adapter_type,
            adapter_config=adapter_config,
            runs_per_test=runs_per_test,
        )

    return result.success


async def _save_results_to_db(
    result: Any,
    suite_name: str,
    agent_name: str,
    adapter_type: str,
    adapter_config: dict[str, Any],
    runs_per_test: int,
) -> None:
    """Save test results to the dashboard database.

    Args:
        result: SuiteResult from test execution
        suite_name: Name of the test suite
        agent_name: Name of the agent
        adapter_type: Type of adapter used
        adapter_config: Adapter configuration
        runs_per_test: Number of runs per test
    """
    from datetime import datetime

    from atp.dashboard import ResultStorage, init_database

    try:
        # Initialize database (creates tables if needed)
        db = await init_database()

        async with db.session() as session:
            storage = ResultStorage(session)

            # Get or create agent
            agent = await storage.get_or_create_agent(
                name=agent_name,
                agent_type=adapter_type,
                config=adapter_config,
            )

            # Create suite execution
            suite_exec = await storage.create_suite_execution(
                suite_name=suite_name,
                agent=agent,
                runs_per_test=runs_per_test,
                started_at=result.start_time,
            )

            # Save each test result
            for test_result in result.tests:
                test_exec = await storage.create_test_execution(
                    suite_execution=suite_exec,
                    test_id=test_result.test.id,
                    test_name=test_result.test.name,
                    tags=test_result.test.tags if test_result.test.tags else None,
                    total_runs=len(test_result.runs),
                )

                # Save run results
                for run_result in test_result.runs:
                    if run_result.response:
                        await storage.create_run_result(
                            test_execution=test_exec,
                            run_number=run_result.run_number,
                            response=run_result.response,
                            events=run_result.events,
                        )

                # Update test execution with results
                await storage.update_test_execution(
                    test_exec,
                    completed_at=datetime.now(),
                    successful_runs=sum(1 for r in test_result.runs if r.success),
                    success=test_result.success,
                    status="completed" if test_result.success else "failed",
                )

            # Update suite execution
            await storage.update_suite_execution(
                suite_exec,
                completed_at=datetime.now(),
                total_tests=len(result.tests),
                passed_tests=sum(1 for t in result.tests if t.success),
                failed_tests=sum(1 for t in result.tests if not t.success),
                success_rate=result.success_rate,
                status="completed" if result.success else "failed",
            )

    except Exception as e:
        # Don't fail the test run if saving fails
        click.echo(f"Warning: Failed to save results to database: {e}", err=True)


@cli.command(name="validate")
@click.option(
    "--agent",
    "agent_name",
    type=str,
    help="Name of the agent to validate (from config file)",
)
@click.option(
    "--adapter",
    type=str,
    help=(
        "Adapter type to validate "
        "(http, cli, container, langgraph, crewai, autogen, mcp)"
    ),
)
@click.option(
    "--adapter-config",
    type=str,
    multiple=True,
    help="Adapter configuration as key=value pairs",
)
@click.option(
    "--suite",
    "suite_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to test suite file to validate",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@pass_config
def validate_cmd(
    config_ctx: ConfigContext,
    agent_name: str | None,
    adapter: str | None,
    adapter_config: tuple[str, ...],
    suite_file: Path | None,
    verbose: bool,
) -> None:
    """Validate agent configuration or test suite.

    Validates that an agent is reachable and properly configured,
    or that a test suite file is valid.

    Examples:

      # Validate agent from config file
      atp validate --agent=my-agent

      # Validate HTTP adapter configuration
      atp validate --adapter=http --adapter-config endpoint=http://localhost:8000

      # Validate a test suite file
      atp validate --suite=tests/suite.yaml

      # Validate both agent and suite
      atp validate --agent=my-agent --suite=tests/suite.yaml

    Exit Codes:

      0 - Validation successful
      1 - Validation failed
      2 - Error occurred
    """
    verbose = verbose or config_ctx.verbose
    has_errors = False

    # Validate test suite if provided
    if suite_file:
        click.echo(f"Validating test suite: {suite_file}")
        try:
            loader = TestLoader()
            suite = loader.load_file(suite_file)
            click.echo(f"  ✓ Suite '{suite.test_suite}' is valid")
            click.echo(f"    Version: {suite.version}")
            click.echo(f"    Tests: {len(suite.tests)}")
            if verbose:
                for test in suite.tests:
                    tags_str = ", ".join(test.tags) if test.tags else "no tags"
                    click.echo(f"      - {test.id}: {test.name} [{tags_str}]")
        except Exception as e:
            click.echo(f"  ✗ Suite validation failed: {e}", err=True)
            has_errors = True

    # Validate agent if provided
    if agent_name or adapter:
        adapter_type = adapter or "http"
        config_dict: dict[str, Any] = {}

        # Get agent config from file
        if agent_name:
            agent_cfg = config_ctx.get_agent_config(agent_name)
            if agent_cfg:
                click.echo(f"Validating agent: {agent_name}")
                if "type" in agent_cfg:
                    adapter_type = agent_cfg["type"]
                config_dict.update(
                    {k: v for k, v in agent_cfg.items() if k not in ("type", "name")}
                )
            else:
                click.echo(f"  ✗ Agent '{agent_name}' not found in config", err=True)
                has_errors = True
                if not adapter:
                    sys.exit(EXIT_FAILURE)
        else:
            click.echo(f"Validating adapter: {adapter_type}")

        # Apply CLI adapter-config options
        for item in adapter_config:
            if "=" in item:
                key, value = item.split("=", 1)
                try:
                    config_dict[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config_dict[key] = value

        try:
            from atp.adapters import create_adapter, get_registry

            # Verify adapter type exists
            registry = get_registry()
            if not registry.is_registered(adapter_type):
                click.echo(
                    f"  ✗ Unknown adapter type: {adapter_type}",
                    err=True,
                )
                click.echo(
                    f"    Available types: {', '.join(registry.list_adapters())}",
                    err=True,
                )
                has_errors = True
            else:
                # Try to create adapter (validates config)
                adapter_instance = create_adapter(adapter_type, config_dict)
                click.echo("  ✓ Adapter configuration is valid")

                # Try health check if available
                if verbose or agent_name:
                    click.echo("  Checking agent health...")
                    is_healthy = asyncio.run(adapter_instance.health_check())
                    if is_healthy:
                        click.echo("  ✓ Agent is healthy and reachable")
                    else:
                        click.echo(
                            "  ⚠ Agent health check failed (may not be running)",
                            err=True,
                        )
                        # Don't fail on health check - agent might not be running yet
        except Exception as e:
            click.echo(f"  ✗ Validation failed: {e}", err=True)
            has_errors = True

    # If nothing to validate
    if not suite_file and not agent_name and not adapter:
        click.echo(
            "Nothing to validate. Specify --agent, --adapter, or --suite.",
            err=True,
        )
        sys.exit(EXIT_ERROR)

    sys.exit(EXIT_FAILURE if has_errors else EXIT_SUCCESS)


@cli.command(name="list-agents")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed adapter information",
)
@pass_config
def list_agents_cmd(config_ctx: ConfigContext, verbose: bool) -> None:
    """List available adapter types and configured agents.

    Shows all registered adapter types and any agents defined in the
    configuration file.

    Examples:

      # List all available adapters
      atp list-agents

      # Show detailed information
      atp list-agents --verbose
    """
    from atp.adapters import get_registry

    registry = get_registry()
    adapter_types = registry.list_adapters()

    click.echo("Available Adapter Types:")
    click.echo("-" * 40)

    adapter_descriptions = {
        "http": "HTTP/REST API adapter for agents with web endpoints",
        "container": "Docker container adapter for isolated agent execution",
        "cli": "Command-line adapter for CLI-based agents",
        "langgraph": "LangGraph framework adapter",
        "crewai": "CrewAI framework adapter",
        "autogen": "AutoGen framework adapter",
        "mcp": "MCP (Model Context Protocol) adapter for tool/resource access",
    }

    for adapter_type in sorted(adapter_types):
        description = adapter_descriptions.get(adapter_type, "Custom adapter")
        click.echo(f"  {adapter_type:<12} - {description}")

        if verbose:
            try:
                config_class = registry.get_config_class(adapter_type)
                # Show required fields from pydantic model
                required_fields = []
                optional_fields = []
                for name, field_info in config_class.model_fields.items():
                    if field_info.is_required():
                        required_fields.append(name)
                    else:
                        optional_fields.append(name)

                if required_fields:
                    click.echo(
                        f"                 Required: {', '.join(required_fields)}"
                    )
                if optional_fields[:5]:  # Show first 5 optional fields
                    fields_str = ", ".join(optional_fields[:5])
                    if len(optional_fields) > 5:
                        fields_str += f" (+{len(optional_fields) - 5} more)"
                    click.echo(f"                 Optional: {fields_str}")
            except Exception:
                pass

    # Show configured agents from config file
    if config_ctx.config:
        agents = config_ctx.config.get("agents", {})
        if agents:
            click.echo()
            click.echo("Configured Agents:")
            click.echo("-" * 40)
            for name, cfg in agents.items():
                agent_type = cfg.get("type", "unknown")
                click.echo(f"  {name:<16} (type: {agent_type})")
                if verbose and isinstance(cfg, dict):
                    for key, value in cfg.items():
                        if key not in ("type", "name"):
                            # Mask sensitive values
                            if "key" in key.lower() or "secret" in key.lower():
                                value = "***"
                            elif "password" in key.lower():
                                value = "***"
                            click.echo(f"                     {key}: {value}")


@cli.group()
def baseline() -> None:
    """Manage test baselines for regression detection.

    Baselines allow you to save test results and compare future runs
    against them to detect regressions or improvements.

    Examples:

      # Save a new baseline
      atp baseline save suite.yaml -o baseline.json --runs=5

      # Compare against a baseline
      atp baseline compare suite.yaml -b baseline.json
    """
    pass


@baseline.command(name="save")
@click.argument("suite_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file path for baseline JSON",
)
@click.option(
    "--tags",
    type=str,
    help="Filter tests by tags (comma-separated, use ! to exclude)",
)
@click.option(
    "--adapter",
    type=str,
    default="http",
    help="Adapter type to use (http, cli, container, langgraph, crewai, autogen, mcp)",
)
@click.option(
    "--adapter-config",
    type=str,
    multiple=True,
    help="Adapter configuration as key=value pairs",
)
@click.option(
    "--agent-name",
    type=str,
    default="test-agent",
    help="Name of the agent being tested",
)
@click.option(
    "--runs",
    type=int,
    default=5,
    help="Number of runs per test (default 5 for statistical significance)",
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
def baseline_save(
    suite_file: Path,
    output: Path,
    tags: str | None,
    adapter: str,
    adapter_config: tuple[str, ...],
    agent_name: str,
    runs: int,
    parallel: int,
    verbose: bool,
) -> None:
    """Run tests and save results as a baseline.

    Executes tests multiple times to gather statistical data,
    then saves the results as a baseline for future comparison.

    Examples:

      # Save baseline with default 5 runs
      atp baseline save tests/suite.yaml -o baseline.json

      # Save baseline with more runs for better statistics
      atp baseline save tests/suite.yaml -o baseline.json --runs=10

      # Save baseline for specific tests only
      atp baseline save tests/suite.yaml -o baseline.json --tags=smoke
    """
    if runs < 2:
        click.echo(
            "Warning: --runs should be at least 2 for statistical significance",
            err=True,
        )

    try:
        result = asyncio.run(
            _run_and_save_baseline(
                suite_file=suite_file,
                output=output,
                tags=tags,
                adapter_type=adapter,
                adapter_config=adapter_config,
                agent_name=agent_name,
                runs=runs,
                parallel=parallel,
                verbose=verbose,
            )
        )
        sys.exit(EXIT_SUCCESS if result else EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _run_and_save_baseline(
    suite_file: Path,
    output: Path,
    tags: str | None,
    adapter_type: str,
    adapter_config: tuple[str, ...],
    agent_name: str,
    runs: int,
    parallel: int,
    verbose: bool,
) -> bool:
    """Run tests and save baseline.

    Args:
        suite_file: Path to test suite.
        output: Output path for baseline file.
        tags: Tag filter.
        adapter_type: Adapter type.
        adapter_config: Adapter config.
        agent_name: Agent name.
        runs: Number of runs per test.
        parallel: Max parallel tests.
        verbose: Verbose output.

    Returns:
        True if successful.
    """
    from atp.adapters import create_adapter
    from atp.baseline import Baseline, TestBaseline, save_baseline
    from atp.loader import TestLoader
    from atp.runner import SandboxConfig, TestOrchestrator, create_progress_callback
    from atp.scoring import ScoreAggregator
    from atp.statistics import StatisticsCalculator

    # Load test suite
    loader = TestLoader()
    suite = loader.load_file(suite_file)

    if tags:
        suite = suite.filter_by_tags(tags)

    if not suite.tests:
        click.echo("No tests match the specified criteria.", err=True)
        return False

    click.echo(f"Running {len(suite.tests)} tests with {runs} runs each...")

    # Parse adapter config
    config_dict: dict[str, Any] = {}
    for item in adapter_config:
        if "=" in item:
            key, value = item.split("=", 1)
            try:
                config_dict[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                config_dict[key] = value

    # Create adapter and run tests
    adapter_instance = create_adapter(adapter_type, config_dict)
    sandbox_config = SandboxConfig(enabled=False)
    progress_callback = create_progress_callback(
        max_parallel=parallel,
        verbose=verbose,
        use_colors=True,
    )

    async with TestOrchestrator(
        adapter=adapter_instance,
        sandbox_config=sandbox_config,
        progress_callback=progress_callback,
        runs_per_test=runs,
        fail_fast=False,
        parallel_tests=parallel > 1,
        max_parallel_tests=parallel,
    ) as orchestrator:
        result = await orchestrator.run_suite(
            suite=suite,
            agent_name=agent_name,
            runs_per_test=runs,
        )

    # Calculate statistics and create baseline
    calc = StatisticsCalculator()
    aggregator = ScoreAggregator()
    tests_baseline: dict[str, TestBaseline] = {}

    for test_result in result.tests:
        test_id = test_result.test.id
        test_name = test_result.test.name

        # Get scores from runs
        scores: list[float] = []
        durations: list[float] = []
        tokens: list[int] = []
        costs: list[float] = []

        for run in test_result.runs:
            # Calculate score for this run
            breakdown = aggregator.aggregate(
                eval_results=[],
                response=run.response,
            )
            scores.append(breakdown.final_score)

            if run.response.metrics:
                if run.response.metrics.wall_time_seconds is not None:
                    durations.append(run.response.metrics.wall_time_seconds)
                if run.response.metrics.total_tokens is not None:
                    tokens.append(run.response.metrics.total_tokens)
                if run.response.metrics.cost_usd is not None:
                    costs.append(run.response.metrics.cost_usd)

        if not scores:
            continue

        # Calculate statistics
        stats = calc.compute(scores)

        tests_baseline[test_id] = TestBaseline(
            test_id=test_id,
            test_name=test_name,
            mean_score=stats.mean,
            std=stats.std,
            n_runs=stats.n_runs,
            ci_95=stats.confidence_interval,
            success_rate=test_result.successful_runs / test_result.total_runs,
            mean_duration=calc.calculate_mean(durations) if durations else None,
            mean_tokens=(
                calc.calculate_mean([float(t) for t in tokens]) if tokens else None
            ),
            mean_cost=calc.calculate_mean(costs) if costs else None,
        )

    # Create and save baseline
    baseline_obj = Baseline(
        suite_name=suite.test_suite,
        agent_name=agent_name,
        runs_per_test=runs,
        tests=tests_baseline,
    )

    save_baseline(baseline_obj, output)
    click.echo(f"Baseline saved to {output}")
    click.echo(f"  Tests: {len(tests_baseline)}")
    click.echo(f"  Runs per test: {runs}")

    return True


@baseline.command(name="compare")
@click.argument("suite_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--baseline",
    "-b",
    "baseline_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to baseline file to compare against",
)
@click.option(
    "--tags",
    type=str,
    help="Filter tests by tags (comma-separated, use ! to exclude)",
)
@click.option(
    "--adapter",
    type=str,
    default="http",
    help="Adapter type to use",
)
@click.option(
    "--adapter-config",
    type=str,
    multiple=True,
    help="Adapter configuration as key=value pairs",
)
@click.option(
    "--agent-name",
    type=str,
    default="test-agent",
    help="Name of the agent being tested",
)
@click.option(
    "--runs",
    type=int,
    default=5,
    help="Number of runs per test",
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help="Maximum number of tests to run in parallel",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    help="Output file path",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--fail-on-regression",
    is_flag=True,
    help="Exit with non-zero status if regressions detected",
)
def baseline_compare(
    suite_file: Path,
    baseline_file: Path,
    tags: str | None,
    adapter: str,
    adapter_config: tuple[str, ...],
    agent_name: str,
    runs: int,
    parallel: int,
    output: str,
    output_file: Path | None,
    verbose: bool,
    fail_on_regression: bool,
) -> None:
    """Run tests and compare results against a baseline.

    Detects regressions and improvements using statistical analysis.

    Examples:

      # Compare against baseline
      atp baseline compare tests/suite.yaml -b baseline.json

      # Fail CI if regressions detected
      atp baseline compare tests/suite.yaml -b baseline.json --fail-on-regression
    """
    try:
        has_regression = asyncio.run(
            _run_and_compare_baseline(
                suite_file=suite_file,
                baseline_file=baseline_file,
                tags=tags,
                adapter_type=adapter,
                adapter_config=adapter_config,
                agent_name=agent_name,
                runs=runs,
                parallel=parallel,
                output_format=output,
                output_file=output_file,
                verbose=verbose,
            )
        )

        if fail_on_regression and has_regression:
            sys.exit(EXIT_FAILURE)
        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _run_and_compare_baseline(
    suite_file: Path,
    baseline_file: Path,
    tags: str | None,
    adapter_type: str,
    adapter_config: tuple[str, ...],
    agent_name: str,
    runs: int,
    parallel: int,
    output_format: str,
    output_file: Path | None,
    verbose: bool,
) -> bool:
    """Run tests and compare against baseline.

    Args:
        suite_file: Path to test suite.
        baseline_file: Path to baseline file.
        tags: Tag filter.
        adapter_type: Adapter type.
        adapter_config: Adapter config.
        agent_name: Agent name.
        runs: Number of runs per test.
        parallel: Max parallel tests.
        output_format: Output format.
        output_file: Output file path.
        verbose: Verbose output.

    Returns:
        True if regressions detected.
    """
    from atp.adapters import create_adapter
    from atp.baseline import compare_results, load_baseline, print_comparison
    from atp.loader import TestLoader
    from atp.runner import SandboxConfig, TestOrchestrator, create_progress_callback
    from atp.scoring import ScoreAggregator

    # Load baseline
    baseline_obj = load_baseline(baseline_file)

    # Load test suite
    loader = TestLoader()
    suite = loader.load_file(suite_file)

    if tags:
        suite = suite.filter_by_tags(tags)

    if not suite.tests:
        click.echo("No tests match the specified criteria.", err=True)
        return False

    click.echo(f"Running {len(suite.tests)} tests with {runs} runs each...")

    # Parse adapter config
    config_dict: dict[str, Any] = {}
    for item in adapter_config:
        if "=" in item:
            key, value = item.split("=", 1)
            try:
                config_dict[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                config_dict[key] = value

    # Create adapter and run tests
    adapter_instance = create_adapter(adapter_type, config_dict)
    sandbox_config = SandboxConfig(enabled=False)
    progress_callback = create_progress_callback(
        max_parallel=parallel,
        verbose=verbose,
        use_colors=True,
    )

    async with TestOrchestrator(
        adapter=adapter_instance,
        sandbox_config=sandbox_config,
        progress_callback=progress_callback,
        runs_per_test=runs,
        fail_fast=False,
        parallel_tests=parallel > 1,
        max_parallel_tests=parallel,
    ) as orchestrator:
        result = await orchestrator.run_suite(
            suite=suite,
            agent_name=agent_name,
            runs_per_test=runs,
        )

    # Calculate scores for comparison
    aggregator = ScoreAggregator()
    current_scores: dict[str, list[float]] = {}
    test_names: dict[str, str] = {}

    for test_result in result.tests:
        test_id = test_result.test.id
        test_names[test_id] = test_result.test.name
        scores: list[float] = []

        for run in test_result.runs:
            breakdown = aggregator.aggregate(
                eval_results=[],
                response=run.response,
            )
            scores.append(breakdown.final_score)

        if scores:
            current_scores[test_id] = scores

    # Compare with baseline
    comparison = compare_results(
        current_scores=current_scores,
        baseline=baseline_obj,
        test_names=test_names,
    )

    # Output results
    print_comparison(
        result=comparison,
        output_format=output_format,
        output_file=output_file,
        use_colors=True,
        verbose=verbose,
    )

    return comparison.has_regressions


@cli.command(name="dashboard")
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to bind to (default: 0.0.0.0)",
)
@click.option(
    "--port",
    type=int,
    default=8080,
    help="Port to bind to (default: 8080)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
def dashboard_cmd(host: str, port: int, reload: bool) -> None:
    """Start the ATP web dashboard server.

    Launches a web interface for viewing test results, historical trends,
    and agent comparisons.

    Examples:

      # Start dashboard on default port
      atp dashboard

      # Start on custom port
      atp dashboard --port=3000

      # Start with auto-reload for development
      atp dashboard --reload

    Environment Variables:

      ATP_DATABASE_URL: Database connection URL
      ATP_SECRET_KEY: JWT secret key for authentication
      ATP_CORS_ORIGINS: Comma-separated list of allowed CORS origins
    """
    click.echo(f"Starting ATP Dashboard at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")

    from atp.dashboard import run_server

    run_server(host=host, port=port, reload=reload)


@cli.command(name="tui")
def tui_cmd() -> None:
    """Start the ATP Terminal User Interface.

    Launches an interactive terminal interface for viewing test results,
    managing test suites, and configuring agents.

    Examples:

      # Start the TUI
      atp tui

    Requirements:

      This command requires optional TUI dependencies. Install with:
      uv add atp-platform[tui]

    Keyboard Shortcuts:

      h - Home screen
      s - Suites screen
      r - Results screen
      a - Agents screen
      ? - Help screen
      q - Quit
    """
    try:
        from atp.tui import ATPTUI
    except ImportError as e:
        click.echo(
            "Error: TUI dependencies not installed.\n"
            "Install with: uv add atp-platform[tui]",
            err=True,
        )
        click.echo(f"Details: {e}", err=True)
        sys.exit(EXIT_ERROR)

    app = ATPTUI()
    app.run()


@cli.command(name="list")
@click.argument("suite_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tags",
    type=str,
    help="Filter tests by tags (comma-separated, use ! to exclude)",
)
def list_tests(suite_file: Path, tags: str | None) -> None:
    """List tests in a test suite.

    Examples:

      # List all tests
      atp list tests/suite.yaml

      # List only smoke tests
      atp list tests/suite.yaml --tags=smoke
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
        sys.exit(EXIT_ERROR)


# Register commands
cli.add_command(benchmark_command)
cli.add_command(budget_command)
cli.add_command(experiment_command)
cli.add_command(init_command)
cli.add_command(generate_command)
cli.add_command(plugins_command)


def main() -> None:
    """Main entry point for the CLI."""
    cli(auto_envvar_prefix="ATP")


if __name__ == "__main__":
    main()
