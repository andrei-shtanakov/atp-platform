"""Test templates for generating ATP test cases from predefined patterns."""

import re
from dataclasses import dataclass, field
from typing import Any

from atp.loader.models import Assertion, Constraints


@dataclass
class TestTemplate:
    """
    A reusable test template for generating ATP test cases.

    Templates define patterns for common testing scenarios with variable
    placeholders that can be substituted at test creation time.

    Attributes:
        name: Unique template identifier.
        description: Human-readable description of the template.
        category: Category for grouping templates (e.g., "file_operations").
        task_template: Task description with {variable} placeholders.
        default_constraints: Default execution constraints for tests.
        default_assertions: Default assertions with {variable} placeholders.
        tags: Default tags to apply to generated tests.

    Example:
        >>> template = TestTemplate(
        ...     name="file_creation",
        ...     description="Test file creation capabilities",
        ...     category="file_operations",
        ...     task_template="Create a file named {filename} with content: {content}",
        ...     default_assertions=[
        ...         Assertion(type="artifact_exists", config={"path": "{filename}"})
        ...     ],
        ...     tags=["file", "basic"],
        ... )
    """

    name: str
    description: str
    category: str
    task_template: str
    default_constraints: Constraints = field(default_factory=Constraints)
    default_assertions: list[Assertion] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class TemplateRegistry:
    """
    Registry for managing test templates.

    Provides methods to register, retrieve, and list templates.
    Includes built-in templates for common testing scenarios.

    Example:
        >>> registry = TemplateRegistry()
        >>> registry.register(my_template)
        >>> template = registry.get("my_template")
    """

    def __init__(self) -> None:
        """Initialize the template registry with built-in templates."""
        self._templates: dict[str, TestTemplate] = {}
        self._register_builtin_templates()

    def _register_builtin_templates(self) -> None:
        """Register the built-in templates."""
        builtin_templates = [
            TestTemplate(
                name="file_creation",
                description="Test agent's ability to create files with content",
                category="file_operations",
                task_template=(
                    "Create a file named '{filename}' with the following content:\n"
                    "{content}"
                ),
                default_constraints=Constraints(
                    max_steps=5,
                    timeout_seconds=60,
                ),
                default_assertions=[
                    Assertion(
                        type="artifact_exists",
                        config={"path": "{filename}"},
                    ),
                ],
                tags=["file", "basic", "creation"],
            ),
            TestTemplate(
                name="data_processing",
                description="Test agent's ability to process and transform data",
                category="data",
                task_template=(
                    "Read the input file '{input_file}' and process it according to:\n"
                    "{processing_instructions}\n"
                    "Save the result to '{output_file}'."
                ),
                default_constraints=Constraints(
                    max_steps=10,
                    timeout_seconds=120,
                ),
                default_assertions=[
                    Assertion(
                        type="artifact_exists",
                        config={"path": "{output_file}"},
                    ),
                ],
                tags=["data", "processing", "transformation"],
            ),
            TestTemplate(
                name="web_research",
                description="Test agent's ability to research and synthesize info",
                category="research",
                task_template=(
                    "Research the following topic:\n"
                    "{topic}\n\n"
                    "Requirements:\n"
                    "{requirements}\n\n"
                    "Save your findings to '{output_file}'."
                ),
                default_constraints=Constraints(
                    max_steps=20,
                    timeout_seconds=300,
                ),
                default_assertions=[
                    Assertion(
                        type="artifact_exists",
                        config={"path": "{output_file}"},
                    ),
                ],
                tags=["research", "web", "synthesis"],
            ),
            TestTemplate(
                name="code_generation",
                description="Test agent's ability to generate code",
                category="coding",
                task_template=(
                    "Write {language} code that implements the following:\n"
                    "{specification}\n\n"
                    "Save the code to '{output_file}'."
                ),
                default_constraints=Constraints(
                    max_steps=15,
                    timeout_seconds=180,
                ),
                default_assertions=[
                    Assertion(
                        type="artifact_exists",
                        config={"path": "{output_file}"},
                    ),
                ],
                tags=["code", "generation", "{language}"],
            ),
        ]

        for template in builtin_templates:
            self._templates[template.name] = template

    def register(self, template: TestTemplate) -> None:
        """
        Register a new template.

        Args:
            template: The template to register.

        Raises:
            ValueError: If a template with the same name already exists.
        """
        if template.name in self._templates:
            raise ValueError(f"Template already exists: {template.name}")
        self._templates[template.name] = template

    def get(self, name: str) -> TestTemplate:
        """
        Get a template by name.

        Args:
            name: The template name.

        Returns:
            The template.

        Raises:
            KeyError: If template not found.
        """
        if name not in self._templates:
            raise KeyError(f"Template not found: {name}")
        return self._templates[name]

    def list_templates(self) -> list[str]:
        """
        List all registered template names.

        Returns:
            List of template names.
        """
        return list(self._templates.keys())

    def list_by_category(self, category: str) -> list[TestTemplate]:
        """
        List templates by category.

        Args:
            category: The category to filter by.

        Returns:
            List of templates in the category.
        """
        return [t for t in self._templates.values() if t.category == category]

    def has_template(self, name: str) -> bool:
        """
        Check if a template exists.

        Args:
            name: The template name.

        Returns:
            True if template exists, False otherwise.
        """
        return name in self._templates


def substitute_variables(text: str, variables: dict[str, Any]) -> str:
    """
    Substitute variables in a template string.

    Uses {variable_name} syntax for placeholders. Variables not provided
    are left unchanged in the output.

    Args:
        text: The template string with {variable} placeholders.
        variables: Dictionary of variable names to values.

    Returns:
        The string with variables substituted.

    Example:
        >>> substitute_variables("Hello {name}!", {"name": "World"})
        'Hello World!'
    """
    result = text
    for key, value in variables.items():
        placeholder = "{" + key + "}"
        result = result.replace(placeholder, str(value))
    return result


def substitute_in_assertion(
    assertion: Assertion, variables: dict[str, Any]
) -> Assertion:
    """
    Substitute variables in an assertion's config.

    Recursively substitutes {variable} placeholders in all string values
    within the assertion config.

    Args:
        assertion: The assertion to process.
        variables: Dictionary of variable names to values.

    Returns:
        A new Assertion with substituted values.
    """

    def substitute_in_value(value: Any) -> Any:
        if isinstance(value, str):
            return substitute_variables(value, variables)
        if isinstance(value, dict):
            return {k: substitute_in_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [substitute_in_value(v) for v in value]
        return value

    new_config = substitute_in_value(assertion.config)
    return Assertion(type=assertion.type, config=new_config)


def extract_variables(text: str) -> list[str]:
    """
    Extract variable names from a template string.

    Args:
        text: The template string with {variable} placeholders.

    Returns:
        List of variable names found in the template.

    Example:
        >>> extract_variables("Hello {name}, you have {count} messages")
        ['name', 'count']
    """
    pattern = r"\{(\w+)\}"
    return re.findall(pattern, text)


def get_template_variables(template: TestTemplate) -> set[str]:
    """
    Get all variable names used in a template.

    Extracts variables from task_template, assertions, and tags.

    Args:
        template: The template to analyze.

    Returns:
        Set of variable names used in the template.
    """
    variables: set[str] = set()

    # Variables in task template
    variables.update(extract_variables(template.task_template))

    # Variables in assertions
    for assertion in template.default_assertions:
        for value in assertion.config.values():
            if isinstance(value, str):
                variables.update(extract_variables(value))

    # Variables in tags
    for tag in template.tags:
        variables.update(extract_variables(tag))

    return variables
