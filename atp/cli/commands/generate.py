"""CLI generate command for creating tests and suites."""

import sys
from pathlib import Path
from typing import Any

import click
import yaml

from atp.generator.core import TestGenerator, TestSuiteData
from atp.generator.regression import (
    AnonymizationLevel,
    RegressionTestGenerator,
    load_recordings_from_file,
)
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


@generate_command.command(name="regression")
@click.option(
    "--recordings",
    "-r",
    "recordings_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to recordings file (JSON format)",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file path for generated test suite",
)
@click.option(
    "--name",
    "-n",
    "suite_name",
    type=str,
    default=None,
    help="Name for the generated suite (default: derived from output filename)",
)
@click.option(
    "--description",
    "-d",
    "suite_description",
    type=str,
    default=None,
    help="Description for the generated suite",
)
@click.option(
    "--anonymize",
    type=click.Choice(["none", "basic", "strict"]),
    default="basic",
    help="Level of data anonymization: none, basic (default), or strict",
)
@click.option(
    "--no-dedupe",
    is_flag=True,
    help="Disable deduplication of similar recordings",
)
@click.option(
    "--similarity",
    type=float,
    default=0.8,
    help="Similarity threshold for deduplication (0.0-1.0, default: 0.8)",
)
@click.option(
    "--no-parameterize",
    is_flag=True,
    help="Disable automatic parameterization of recorded values",
)
@click.option(
    "--tags",
    type=str,
    default=None,
    help="Comma-separated tags to add to all generated tests",
)
@click.option(
    "--save-params",
    is_flag=True,
    help="Save extracted parameters to a separate file",
)
def generate_regression(
    recordings_file: Path,
    output_file: Path,
    suite_name: str | None,
    suite_description: str | None,
    anonymize: str,
    no_dedupe: bool,
    similarity: float,
    no_parameterize: bool,
    tags: str | None,
    save_params: bool,
) -> None:
    """Generate regression tests from recorded agent interactions.

    This command converts recorded agent interactions into reusable test
    cases. Recordings capture requests, responses, and events from real
    agent runs, which can then be used to create regression tests.

    Features:

      - Automatic anonymization of sensitive data (emails, API keys, etc.)
      - Deduplication of similar recordings
      - Parameterization of variable values for test reuse
      - Assertion generation based on actual responses

    Examples:

      # Generate tests from recordings with default settings
      atp generate regression -r recordings.json -o regression-tests.yaml

      # Strict anonymization with custom suite name
      atp generate regression -r logs.json -o tests.yaml --anonymize=strict \\
          --name="API Tests"

      # Disable deduplication and parameterization
      atp generate regression -r data.json -o tests.yaml --no-dedupe \\
          --no-parameterize

      # Add tags to all generated tests
      atp generate regression -r data.json -o tests.yaml --tags="regression,ci"

      # Save extracted parameters for later use
      atp generate regression -r data.json -o tests.yaml --save-params

    Recording Format:

      Recordings should be a JSON file containing an array of recording
      objects, or an object with a "recordings" key containing the array.
      Each recording includes request, response, and events data.

    Exit Codes:

      0 - Tests generated successfully
      2 - Error occurred
    """
    try:
        # Parse anonymization level
        anon_level = AnonymizationLevel(anonymize)

        # Parse tags
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

        # Derive suite name from output file if not provided
        if suite_name is None:
            suite_name = output_file.stem.replace("-", "_").replace(".", "_")

        click.echo(f"Loading recordings from: {recordings_file}")

        # Load recordings
        recordings = load_recordings_from_file(recordings_file)
        click.echo(f"  Found {len(recordings)} recording(s)")

        # Filter to completed recordings
        completed_count = sum(1 for r in recordings if r.status.value == "completed")
        click.echo(f"  Completed: {completed_count}")

        if completed_count == 0:
            raise click.ClickException(
                "No completed recordings found. "
                "Only completed recordings can be converted to tests."
            )

        # Create generator
        generator = RegressionTestGenerator(
            anonymization_level=anon_level,
            similarity_threshold=similarity,
            extract_parameters=not no_parameterize,
        )

        click.echo("\nGenerating tests...")
        click.echo(f"  Anonymization: {anonymize}")
        dedupe_status = "disabled" if no_dedupe else "enabled"
        click.echo(f"  Deduplication: {dedupe_status}")
        param_status = "disabled" if no_parameterize else "enabled"
        click.echo(f"  Parameterization: {param_status}")

        # Generate suite
        suite, parameters = generator.generate_from_recordings(
            recordings=recordings,
            suite_name=suite_name,
            suite_description=suite_description,
            deduplicate=not no_dedupe,
            tags=tag_list,
        )

        click.echo(f"\n  Generated {len(suite.tests)} test(s)")

        if no_dedupe:
            click.echo("  (deduplication disabled)")
        else:
            dedupe_removed = completed_count - len(suite.tests)
            if dedupe_removed > 0:
                click.echo(f"  Removed {dedupe_removed} duplicate(s)")

        # Save suite
        generator.save_suite(
            suite,
            output_file,
            parameters=parameters if save_params else None,
        )

        click.echo(f"\nTest suite saved: {output_file}")
        click.echo(f"  Name: {suite.name}")
        click.echo(f"  Tests: {len(suite.tests)}")

        if save_params and parameters:
            params_file = output_file.with_suffix(".params.yaml")
            click.echo(f"  Parameters: {params_file}")

        click.echo(f"\nRun tests with: atp test {output_file}")

        sys.exit(EXIT_SUCCESS)

    except click.Abort:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(EXIT_ERROR)
    except click.ClickException:
        raise
    except FileNotFoundError as e:
        click.echo(f"Error: File not found: {e}", err=True)
        sys.exit(EXIT_ERROR)
    except ValueError as e:
        click.echo(f"Error: Invalid data: {e}", err=True)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


@generate_command.command(name="from-traces")
@click.option(
    "--source",
    type=click.Choice(["langsmith", "otel"]),
    required=True,
    help="Trace source: langsmith or otel",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    required=True,
    help="Output YAML file path for generated test suite",
)
@click.option(
    "--project",
    type=str,
    default=None,
    help="LangSmith project name (for --source=langsmith)",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="API key (for --source=langsmith)",
)
@click.option(
    "--file",
    "trace_file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="OTLP JSON file path (for --source=otel)",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    help="Max number of traces to import (default: 50)",
)
@click.option(
    "--name",
    "suite_name",
    type=str,
    default=None,
    help="Suite name (default: derived from output filename)",
)
@click.option(
    "--no-dedupe",
    is_flag=True,
    help="Disable deduplication of similar traces",
)
@click.option(
    "--tags",
    type=str,
    default=None,
    help="Comma-separated tags to add to all generated tests",
)
def generate_from_traces(
    source: str,
    output_file: Path,
    project: str | None,
    api_key: str | None,
    trace_file: Path | None,
    limit: int,
    suite_name: str | None,
    no_dedupe: bool,
    tags: str | None,
) -> None:
    """Generate test suites from production traces.

    Import agent traces from LangSmith or OpenTelemetry and
    auto-generate regression test suites.

    Examples:

      # Import from LangSmith
      atp generate from-traces --source=langsmith \\
          --project=my-project --api-key=KEY \\
          --limit=50 --output=suite.yaml

      # Import from OpenTelemetry OTLP JSON
      atp generate from-traces --source=otel \\
          --file=traces.json --output=suite.yaml

    Exit Codes:

      0 - Tests generated successfully
      2 - Error occurred
    """
    import asyncio

    try:
        # Validate source-specific options
        if source == "langsmith":
            if not api_key:
                raise click.ClickException("--api-key is required for langsmith source")
        elif source == "otel":
            if not trace_file:
                raise click.ClickException("--file is required for otel source")

        # Derive suite name
        if suite_name is None:
            suite_name = output_file.stem.replace("-", "_").replace(".", "_")

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

        click.echo(f"Importing traces from: {source}")

        # Build importer and fetch
        from atp.generator.importers import get_importer

        fetch_kwargs: dict[str, object] = {}
        if source == "langsmith":
            importer = get_importer(
                "langsmith",
                api_key=api_key or "",
                project=project or "",
            )
        else:
            importer = get_importer(
                "otel",
                file_path=str(trace_file or ""),
            )

        records = asyncio.run(importer.fetch_traces(limit=limit, **fetch_kwargs))
        click.echo(f"  Fetched {len(records)} trace(s)")

        if not records:
            raise click.ClickException("No traces found")

        # Convert to test suite
        suite = importer.import_traces(
            records,
            suite_name=suite_name,
            deduplicate=not no_dedupe,
            tags=tag_list,
        )

        deduped = len(records) - len(suite.tests)
        if deduped > 0 and not no_dedupe:
            click.echo(f"  Deduplicated: removed {deduped}")

        click.echo(f"  Generated {len(suite.tests)} test(s)")

        # Save
        from atp.generator.writer import YAMLWriter

        writer = YAMLWriter()
        writer.save(suite, output_file)

        click.echo(f"\nTest suite saved: {output_file}")
        click.echo(f"  Name: {suite.name}")
        click.echo(f"  Tests: {len(suite.tests)}")
        click.echo(f"\nRun tests with: atp test {output_file}")

        sys.exit(EXIT_SUCCESS)

    except click.ClickException:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


@generate_command.command(name="from-description")
@click.argument("description", required=False)
@click.option(
    "--file",
    "-f",
    "desc_file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Read description from a text file",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Output YAML file path (default: stdout)",
)
@click.option(
    "--name",
    "-n",
    "suite_name",
    type=str,
    default=None,
    help="Suite name (default: derived from output filename)",
)
def generate_from_description(
    description: str | None,
    desc_file: Path | None,
    output_file: Path | None,
    suite_name: str | None,
) -> None:
    """Generate a test suite from a natural language description.

    Uses an LLM to convert a plain-text description into a valid
    ATP test suite YAML file.

    Provide the description as a positional argument or via --file.

    Environment Variables:

      ATP_LLM_API_KEY or OPENAI_API_KEY: LLM API key
      ATP_LLM_BASE_URL: API base URL (default: OpenAI)
      ATP_LLM_MODEL: Model to use (default: gpt-4o-mini)

    Examples:

      # Generate from inline description
      atp generate from-description "test that the agent can search
      the web and summarize results"

      # Generate from a requirements file
      atp generate from-description --file=requirements.txt
      --output=suite.yaml

      # Specify a suite name
      atp generate from-description -n my-suite
      "test file creation and deletion"

    Exit Codes:

      0 - Suite generated successfully
      2 - Error occurred
    """
    try:
        from atp.generator.nl_generator import NLTestGenerator

        # Determine description source
        if desc_file is not None:
            desc_text = desc_file.read_text(encoding="utf-8").strip()
            if not desc_text:
                raise click.ClickException(f"Description file is empty: {desc_file}")
        elif description is not None:
            desc_text = description.strip()
        else:
            raise click.ClickException("Provide a description argument or --file")

        if not desc_text:
            raise click.ClickException("Description cannot be empty")

        # Derive suite name
        if suite_name is None:
            if output_file is not None:
                suite_name = output_file.stem.replace("-", "_").replace(".", "_")
            else:
                suite_name = "generated-suite"

        click.echo("Generating test suite from description...")
        click.echo(f"  Suite name: {suite_name}")

        generator = NLTestGenerator()
        suite = generator.generate(
            description=desc_text,
            suite_name=suite_name,
        )

        click.echo(f"  Generated {len(suite.tests)} test(s)")

        # Output
        if output_file is not None:
            generator.save(suite, output_file)
            click.echo(f"\nTest suite saved: {output_file}")
            click.echo(f"\nRun tests with: atp test {output_file}")
        else:
            yaml_content = generator.to_yaml(suite)
            click.echo("\n" + yaml_content)

        sys.exit(EXIT_SUCCESS)

    except click.ClickException:
        raise
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)
