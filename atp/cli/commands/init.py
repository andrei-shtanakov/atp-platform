"""CLI init command for creating new ATP test suites."""

import sys
from pathlib import Path
from typing import Any

import click

from atp.generator.core import TestGenerator, TestSuiteData
from atp.generator.templates import TemplateRegistry, get_template_variables
from atp.loader.models import Assertion, Constraints

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 2

# Agent type configurations
AGENT_TYPES = {
    "http": {
        "description": "HTTP/REST API adapter for agents with web endpoints",
        "required_fields": ["endpoint"],
        "optional_fields": ["method", "headers", "timeout"],
        "prompts": {
            "endpoint": "Agent HTTP endpoint URL",
        },
    },
    "cli": {
        "description": "Command-line adapter for CLI-based agents",
        "required_fields": ["command"],
        "optional_fields": ["args", "env", "cwd"],
        "prompts": {
            "command": "Agent command to execute",
        },
    },
    "container": {
        "description": "Docker container adapter for isolated agent execution",
        "required_fields": ["image"],
        "optional_fields": ["command", "env", "volumes"],
        "prompts": {
            "image": "Docker image name",
        },
    },
}


def _prompt_suite_config(interactive: bool) -> dict[str, Any]:
    """Prompt for suite configuration.

    Args:
        interactive: Whether to use interactive prompts.

    Returns:
        Dictionary with suite configuration.
    """
    config: dict[str, Any] = {}

    if interactive:
        config["name"] = click.prompt(
            "Suite name",
            type=str,
            default="my-test-suite",
        )
        config["description"] = click.prompt(
            "Suite description (optional)",
            type=str,
            default="",
            show_default=False,
        )
        config["runs_per_test"] = click.prompt(
            "Runs per test",
            type=int,
            default=1,
        )
        config["timeout_seconds"] = click.prompt(
            "Default timeout (seconds)",
            type=int,
            default=300,
        )
    else:
        config["name"] = "my-test-suite"
        config["description"] = ""
        config["runs_per_test"] = 1
        config["timeout_seconds"] = 300

    return config


def _prompt_agent_config(interactive: bool) -> dict[str, Any] | None:
    """Prompt for agent configuration.

    Args:
        interactive: Whether to use interactive prompts.

    Returns:
        Dictionary with agent configuration, or None if skipped.
    """
    if not interactive:
        return None

    if not click.confirm("Add an agent configuration?", default=False):
        return None

    agent_config: dict[str, Any] = {}

    # Agent name
    agent_config["name"] = click.prompt(
        "Agent name",
        type=str,
        default="test-agent",
    )

    # Agent type selection
    click.echo("\nAvailable agent types:")
    for agent_type, info in AGENT_TYPES.items():
        click.echo(f"  {agent_type:<12} - {info['description']}")

    agent_type = click.prompt(
        "\nAgent type",
        type=click.Choice(list(AGENT_TYPES.keys())),
        default="http",
    )
    agent_config["type"] = agent_type

    # Type-specific configuration
    type_info = AGENT_TYPES[agent_type]
    config_values: dict[str, Any] = {}

    # Required fields
    for field in type_info["required_fields"]:
        prompt_text = type_info["prompts"].get(field, f"{field}")
        config_values[field] = click.prompt(prompt_text, type=str)

    # Optional fields
    if click.confirm("Configure optional settings?", default=False):
        for field in type_info["optional_fields"]:
            value = click.prompt(
                f"{field} (optional, press Enter to skip)",
                type=str,
                default="",
                show_default=False,
            )
            if value:
                config_values[field] = value

    agent_config["config"] = config_values
    return agent_config


def _prompt_test_config(
    interactive: bool,
    generator: TestGenerator,
    suite: TestSuiteData,
    first_test: bool = True,
) -> dict[str, Any] | None:
    """Prompt for test configuration.

    Args:
        interactive: Whether to use interactive prompts.
        generator: TestGenerator instance for template access.
        suite: Current suite data for ID generation.
        first_test: Whether this is the first test prompt (affects the question).

    Returns:
        Dictionary with test configuration, or None if skipped.
    """
    if not interactive:
        return None

    if first_test:
        if not click.confirm("Add a test?", default=True):
            return None
    # For subsequent tests, we've already asked "Add another test?" in the main loop

    # Choice between template and custom
    click.echo("\nTest creation options:")
    click.echo("  1. Use a template (pre-defined test patterns)")
    click.echo("  2. Create custom test")

    choice = click.prompt(
        "\nChoice",
        type=click.Choice(["1", "2"]),
        default="1",
    )

    if choice == "1":
        return _prompt_template_test(generator, suite)
    else:
        return _prompt_custom_test(generator, suite)


def _prompt_template_test(
    generator: TestGenerator,
    suite: TestSuiteData,
) -> dict[str, Any]:
    """Prompt for template-based test creation.

    Args:
        generator: TestGenerator instance.
        suite: Current suite data.

    Returns:
        Dictionary with test configuration.
    """
    registry = TemplateRegistry()
    templates = registry.list_templates()

    click.echo("\nAvailable templates:")
    for i, name in enumerate(templates, 1):
        template = registry.get(name)
        click.echo(f"  {i}. {name} - {template.description}")

    template_choice = click.prompt(
        "\nSelect template number",
        type=click.IntRange(1, len(templates)),
        default=1,
    )
    template_name = templates[template_choice - 1]
    template = registry.get(template_name)

    # Get template variables
    variables = get_template_variables(template)

    click.echo(f"\nTemplate '{template_name}' requires these values:")
    var_values: dict[str, str] = {}
    for var in sorted(variables):
        var_values[var] = click.prompt(f"  {var}", type=str)

    # Test details
    test_id = click.prompt(
        "Test ID",
        type=str,
        default=generator.generate_test_id(suite),
    )
    test_name = click.prompt(
        "Test name",
        type=str,
        default=f"Test {template_name}",
    )

    # Optional tags
    tags_input = click.prompt(
        "Tags (comma-separated, optional)",
        type=str,
        default="",
        show_default=False,
    )
    extra_tags = [t.strip() for t in tags_input.split(",") if t.strip()]

    return {
        "type": "template",
        "template_name": template_name,
        "test_id": test_id,
        "test_name": test_name,
        "variables": var_values,
        "extra_tags": extra_tags,
    }


def _prompt_custom_test(
    generator: TestGenerator,
    suite: TestSuiteData,
) -> dict[str, Any]:
    """Prompt for custom test creation.

    Args:
        generator: TestGenerator instance.
        suite: Current suite data.

    Returns:
        Dictionary with test configuration.
    """
    test_id = click.prompt(
        "Test ID",
        type=str,
        default=generator.generate_test_id(suite),
    )
    test_name = click.prompt("Test name", type=str)
    task_description = click.prompt("Task description", type=str)

    # Optional tags
    tags_input = click.prompt(
        "Tags (comma-separated, optional)",
        type=str,
        default="",
        show_default=False,
    )
    tags = [t.strip() for t in tags_input.split(",") if t.strip()]

    # Constraints
    constraints: dict[str, Any] = {}
    if click.confirm("Configure constraints?", default=False):
        max_steps = click.prompt(
            "Max steps (0 for unlimited)",
            type=int,
            default=0,
        )
        if max_steps > 0:
            constraints["max_steps"] = max_steps

        timeout = click.prompt(
            "Timeout seconds",
            type=int,
            default=300,
        )
        constraints["timeout_seconds"] = timeout

    # Assertions
    assertions: list[dict[str, Any]] = []
    if click.confirm("Add assertions?", default=False):
        while True:
            click.echo("\nAssertion types:")
            click.echo("  1. artifact_exists - Check if file was created")
            click.echo("  2. artifact_contains - Check file content")
            click.echo("  3. behavior_check - Check agent behavior")
            click.echo("  4. Done adding assertions")

            assertion_choice = click.prompt(
                "Choice",
                type=click.Choice(["1", "2", "3", "4"]),
                default="4",
            )

            if assertion_choice == "4":
                break
            elif assertion_choice == "1":
                path = click.prompt("File path to check", type=str)
                assertions.append({"type": "artifact_exists", "config": {"path": path}})
            elif assertion_choice == "2":
                path = click.prompt("File path", type=str)
                pattern = click.prompt("Content pattern (regex)", type=str)
                assertions.append(
                    {
                        "type": "artifact_contains",
                        "config": {"path": path, "pattern": pattern},
                    }
                )
            elif assertion_choice == "3":
                check_type = click.prompt("Behavior check type", type=str)
                assertions.append(
                    {
                        "type": "behavior_check",
                        "config": {"check": check_type},
                    }
                )

    return {
        "type": "custom",
        "test_id": test_id,
        "test_name": test_name,
        "task_description": task_description,
        "tags": tags,
        "constraints": constraints,
        "assertions": assertions,
    }


def _create_test_from_config(
    generator: TestGenerator,
    test_config: dict[str, Any],
) -> Any:
    """Create a test definition from configuration.

    Args:
        generator: TestGenerator instance.
        test_config: Test configuration dictionary.

    Returns:
        TestDefinition instance.
    """
    if test_config["type"] == "template":
        return generator.create_test_from_template(
            template_name=test_config["template_name"],
            test_id=test_config["test_id"],
            test_name=test_config["test_name"],
            variables=test_config["variables"],
            extra_tags=test_config.get("extra_tags"),
        )
    else:
        constraints = None
        if test_config.get("constraints"):
            constraints = Constraints(**test_config["constraints"])

        assertions = None
        if test_config.get("assertions"):
            assertions = [
                Assertion(type=a["type"], config=a["config"])
                for a in test_config["assertions"]
            ]

        return generator.create_custom_test(
            test_id=test_config["test_id"],
            test_name=test_config["test_name"],
            task_description=test_config["task_description"],
            constraints=constraints,
            assertions=assertions,
            tags=test_config.get("tags"),
        )


@click.command(name="init")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: suite-name.yaml)",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Run in interactive mode with prompts",
)
def init_command(output: Path | None, interactive: bool) -> None:
    """Initialize a new ATP test suite.

    Creates a new test suite YAML file through an interactive wizard
    or with default values in non-interactive mode.

    Examples:

      # Interactive wizard
      atp init

      # Non-interactive with defaults
      atp init --no-interactive -o my-suite.yaml

      # Interactive with custom output path
      atp init -o tests/my-suite.yaml

    Exit Codes:

      0 - Suite created successfully
      2 - Error occurred
    """
    try:
        generator = TestGenerator()

        # Get suite configuration
        suite_config = _prompt_suite_config(interactive)

        # Create suite
        suite = generator.create_suite(
            name=suite_config["name"],
            description=suite_config["description"] or None,
            version="1.0",
        )

        # Update defaults
        suite.defaults.runs_per_test = suite_config["runs_per_test"]
        suite.defaults.timeout_seconds = suite_config["timeout_seconds"]

        # Add agent if configured
        agent_config = _prompt_agent_config(interactive)
        if agent_config:
            suite = generator.add_agent(
                suite=suite,
                name=agent_config["name"],
                agent_type=agent_config["type"],
                config=agent_config["config"],
            )

        # Add tests
        tests_added = 0
        first_test = True
        while True:
            test_config = _prompt_test_config(
                interactive, generator, suite, first_test=first_test
            )
            if test_config is None:
                break

            test = _create_test_from_config(generator, test_config)
            suite = generator.add_test(suite, test)
            tests_added += 1
            click.echo(f"Test '{test.id}' added.")
            first_test = False

            if not interactive:
                break

            if not click.confirm("\nAdd another test?", default=False):
                break

        # If no tests added, create a sample test
        if tests_added == 0:
            if interactive:
                click.echo("\nNo tests added. Creating a sample test...")

            sample_test = generator.create_custom_test(
                test_id="test-001",
                test_name="Sample Test",
                task_description="Describe the task for the agent to perform.",
                tags=["sample"],
            )
            suite = generator.add_test(suite, sample_test)
            click.echo("Sample test 'test-001' added.")

        # Determine output path
        if output is None:
            safe_name = suite.name.replace(" ", "-").lower()
            output = Path(f"{safe_name}.yaml")

        # Save suite
        generator.save(suite, output)

        click.echo(f"\nTest suite created: {output}")
        click.echo(f"  Name: {suite.name}")
        click.echo(f"  Tests: {len(suite.tests)}")
        if suite.agents:
            click.echo(f"  Agents: {len(suite.agents)}")
        click.echo(f"\nRun tests with: atp test {output}")

        sys.exit(EXIT_SUCCESS)

    except click.Abort:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)
