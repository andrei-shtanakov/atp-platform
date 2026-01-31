"""CLI generate command for creating tests and suites."""

import sys
from pathlib import Path
from typing import Any

import click
import yaml

from atp.generator.core import TestGenerator, TestSuiteData
from atp.generator.templates import TemplateRegistry, get_template_variables
from atp.loader.models import Assertion, Constraints

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 2


def _load_existing_suite(file_path: Path) -> tuple[TestSuiteData, dict[str, Any]]:
    """Load an existing suite file and return mutable TestSuiteData.

    Args:
        file_path: Path to existing suite YAML file.

    Returns:
        Tuple of (TestSuiteData, original raw data dict).

    Raises:
        click.ClickException: If file cannot be loaded.
    """
    try:
        with open(file_path) as f:
            raw_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise click.ClickException(f"Suite file not found: {file_path}")
    except yaml.YAMLError as e:
        raise click.ClickException(f"Invalid YAML in suite file: {e}")

    # Convert to TestSuiteData for manipulation
    from atp.loader.models import (
        AgentConfig,
        TestDefaults,
        TestDefinition,
    )

    defaults = TestDefaults()
    if "defaults" in raw_data:
        defaults = TestDefaults(**raw_data["defaults"])

    agents = []
    if "agents" in raw_data:
        agents = [AgentConfig(**a) for a in raw_data["agents"]]

    tests = []
    if "tests" in raw_data:
        tests = [TestDefinition(**t) for t in raw_data["tests"]]

    suite_data = TestSuiteData(
        name=raw_data.get("test_suite", "unnamed"),
        version=raw_data.get("version", "1.0"),
        description=raw_data.get("description"),
        defaults=defaults,
        agents=agents,
        tests=tests,
    )

    return suite_data, raw_data


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


def _prompt_test_config(
    generator: TestGenerator,
    suite: TestSuiteData,
    template_name: str | None = None,
) -> dict[str, Any]:
    """Prompt for test configuration.

    Args:
        generator: TestGenerator instance for template access.
        suite: Current suite data for ID generation.
        template_name: Optional template name to use directly.

    Returns:
        Dictionary with test configuration.
    """
    if template_name:
        # Use specified template directly
        registry = TemplateRegistry()
        if not registry.has_template(template_name):
            raise click.ClickException(f"Template not found: {template_name}")

        template = registry.get(template_name)
        variables = get_template_variables(template)

        click.echo(f"\nTemplate '{template_name}' requires these values:")
        var_values: dict[str, str] = {}
        for var in sorted(variables):
            var_values[var] = click.prompt(f"  {var}", type=str)

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


@click.group(name="generate")
def generate_command() -> None:
    """Generate tests and test suites.

    The generate command provides subcommands for creating new tests
    and test suites, with support for templates and interactive prompts.

    Examples:

      # Generate a new test and add to existing suite
      atp generate test --suite=my-suite.yaml

      # Generate a test using a specific template
      atp generate test --suite=my-suite.yaml --template=file_creation

      # Generate a new test suite from scratch
      atp generate suite -o new-suite.yaml

      # Generate multiple tests in batch mode
      atp generate suite -o batch-suite.yaml --count=5
    """
    pass


@generate_command.command(name="test")
@click.option(
    "--suite",
    "-s",
    "suite_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to existing suite file to add test to",
)
@click.option(
    "--template",
    "-t",
    "template_name",
    type=str,
    default=None,
    help="Template name to use (e.g., file_creation, data_processing)",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: overwrite input suite file)",
)
@click.option(
    "--list-templates",
    is_flag=True,
    help="List available templates and exit",
)
def generate_test(
    suite_file: Path,
    template_name: str | None,
    output_file: Path | None,
    list_templates: bool,
) -> None:
    """Generate a new test and add it to an existing suite.

    Interactively creates a test using either a template or custom
    configuration, then adds it to the specified suite file.

    Examples:

      # Add a test to an existing suite interactively
      atp generate test --suite=tests/suite.yaml

      # Use a specific template
      atp generate test --suite=tests/suite.yaml --template=file_creation

      # Save to a different file (keep original unchanged)
      atp generate test --suite=tests/suite.yaml -o tests/new-suite.yaml

      # List available templates
      atp generate test --suite=tests/suite.yaml --list-templates

    Exit Codes:

      0 - Test generated successfully
      2 - Error occurred
    """
    # List templates if requested
    if list_templates:
        registry = TemplateRegistry()
        templates = registry.list_templates()
        click.echo("Available templates:")
        for name in templates:
            template = registry.get(name)
            click.echo(f"  {name:<20} - {template.description}")
            click.echo(f"    Category: {template.category}")
            click.echo(f"    Tags: {', '.join(template.tags)}")
        sys.exit(EXIT_SUCCESS)

    try:
        generator = TestGenerator()

        # Load existing suite
        if not suite_file.exists():
            raise click.ClickException(f"Suite file not found: {suite_file}")

        suite_data, _ = _load_existing_suite(suite_file)

        click.echo(
            f"Loaded suite '{suite_data.name}' with {len(suite_data.tests)} "
            f"existing tests"
        )

        # Create new test
        test_config = _prompt_test_config(generator, suite_data, template_name)
        test = _create_test_from_config(generator, test_config)

        # Add test to suite
        suite_data = generator.add_test(suite_data, test)
        click.echo(f"Test '{test.id}' added.")

        # Determine output path
        if output_file is None:
            output_file = suite_file

        # Save suite
        generator.save(suite_data, output_file)

        click.echo(f"\nSuite updated: {output_file}")
        click.echo(f"  Total tests: {len(suite_data.tests)}")
        click.echo(f"\nRun tests with: atp test {output_file}")

        sys.exit(EXIT_SUCCESS)

    except click.Abort:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(EXIT_ERROR)
    except click.ClickException:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


@generate_command.command(name="suite")
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: suite-name.yaml)",
)
@click.option(
    "--template",
    "-t",
    "template_name",
    type=str,
    default=None,
    help="Template name for generated tests",
)
@click.option(
    "--count",
    "-n",
    type=int,
    default=None,
    help="Number of tests to generate (batch mode)",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Run in interactive mode with prompts",
)
def generate_suite(
    output_file: Path | None,
    template_name: str | None,
    count: int | None,
    interactive: bool,
) -> None:
    """Generate a new test suite from scratch.

    Creates a complete test suite with configuration and tests.
    Can operate in interactive mode with prompts or batch mode
    for generating multiple tests at once.

    Examples:

      # Interactive suite creation
      atp generate suite

      # Non-interactive with defaults
      atp generate suite --no-interactive -o my-suite.yaml

      # Batch generate 5 tests using a template
      atp generate suite -o batch-suite.yaml --template=file_creation --count=5

      # Interactive with custom output path
      atp generate suite -o tests/my-suite.yaml

    Exit Codes:

      0 - Suite generated successfully
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

        # Batch mode: generate multiple tests with template
        if count is not None and count > 0:
            if not template_name:
                raise click.ClickException(
                    "Batch mode (--count) requires --template to be specified"
                )

            registry = TemplateRegistry()
            if not registry.has_template(template_name):
                raise click.ClickException(f"Template not found: {template_name}")

            template = registry.get(template_name)
            variables = get_template_variables(template)

            click.echo(f"\nGenerating {count} tests using template '{template_name}'")
            click.echo(f"Template requires variables: {', '.join(sorted(variables))}")

            for i in range(count):
                click.echo(f"\n--- Test {i + 1} of {count} ---")
                var_values: dict[str, str] = {}
                for var in sorted(variables):
                    var_values[var] = click.prompt(f"  {var}", type=str)

                test_id = generator.generate_test_id(suite)
                test_name = click.prompt(
                    "Test name",
                    type=str,
                    default=f"Test {template_name} {i + 1}",
                )

                test = generator.create_test_from_template(
                    template_name=template_name,
                    test_id=test_id,
                    test_name=test_name,
                    variables=var_values,
                )
                suite = generator.add_test(suite, test)
                click.echo(f"Test '{test_id}' added.")

        # Interactive mode: add tests one by one
        elif interactive:
            tests_added = 0
            first_test = True
            while True:
                if first_test:
                    if not click.confirm("Add a test?", default=True):
                        break
                else:
                    if not click.confirm("\nAdd another test?", default=False):
                        break

                test_config = _prompt_test_config(generator, suite, template_name)
                test = _create_test_from_config(generator, test_config)
                suite = generator.add_test(suite, test)
                tests_added += 1
                click.echo(f"Test '{test.id}' added.")
                first_test = False

            # If no tests added, create a sample test
            if tests_added == 0:
                click.echo("\nNo tests added. Creating a sample test...")
                sample_test = generator.create_custom_test(
                    test_id="test-001",
                    test_name="Sample Test",
                    task_description="Describe the task for the agent to perform.",
                    tags=["sample"],
                )
                suite = generator.add_test(suite, sample_test)
                click.echo("Sample test 'test-001' added.")

        # Non-interactive without count: create sample test
        else:
            sample_test = generator.create_custom_test(
                test_id="test-001",
                test_name="Sample Test",
                task_description="Describe the task for the agent to perform.",
                tags=["sample"],
            )
            suite = generator.add_test(suite, sample_test)
            click.echo("Sample test 'test-001' added.")

        # Determine output path
        if output_file is None:
            safe_name = suite.name.replace(" ", "-").lower()
            output_file = Path(f"{safe_name}.yaml")

        # Save suite
        generator.save(suite, output_file)

        click.echo(f"\nTest suite created: {output_file}")
        click.echo(f"  Name: {suite.name}")
        click.echo(f"  Tests: {len(suite.tests)}")
        click.echo(f"\nRun tests with: atp test {output_file}")

        sys.exit(EXIT_SUCCESS)

    except click.Abort:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(EXIT_ERROR)
    except click.ClickException:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)
