"""Core test generator engine for creating ATP test suites."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atp.generator.templates import (
    TemplateRegistry,
    TestTemplate,
    substitute_in_assertion,
    substitute_variables,
)
from atp.loader.models import (
    AgentConfig,
    Assertion,
    Constraints,
    TaskDefinition,
    TestDefaults,
    TestDefinition,
)

if TYPE_CHECKING:
    from atp.loader.models import TestSuite


@dataclass
class TestSuiteData:
    """
    Mutable test suite data for the generator.

    This is used internally during suite construction since the actual
    TestSuite model requires at least one test. Once construction is
    complete, use `to_test_suite()` to convert to the validated model.
    """

    name: str
    version: str = "1.0"
    description: str | None = None
    defaults: TestDefaults = field(default_factory=TestDefaults)
    agents: list[AgentConfig] = field(default_factory=list)
    tests: list[TestDefinition] = field(default_factory=list)

    def to_test_suite(self) -> "TestSuite":
        """Convert to validated TestSuite model.

        Raises:
            ValueError: If no tests have been added.
        """
        from atp.loader.models import TestSuite

        if not self.tests:
            raise ValueError("Cannot create TestSuite with no tests")

        return TestSuite(
            test_suite=self.name,
            version=self.version,
            description=self.description,
            defaults=self.defaults,
            agents=self.agents,
            tests=self.tests,
        )


class TestGenerator:
    """
    Core engine for generating ATP test suites.

    Provides methods for:
    - Creating test suites
    - Adding agents and tests
    - Generating unique test IDs
    - Serialization to YAML

    Example:
        >>> generator = TestGenerator()
        >>> suite = generator.create_suite("my_suite", "My test suite")
        >>> suite = generator.add_agent(suite, "agent1", "http", {"endpoint": "..."})
        >>> test = generator.create_custom_test("test-001", "Test", "Do something")
        >>> suite = generator.add_test(suite, test)
    """

    def __init__(self) -> None:
        """Initialize the test generator with template registry."""
        self._template_registry = TemplateRegistry()

    def create_suite(
        self,
        name: str,
        description: str | None = None,
        version: str = "1.0",
    ) -> TestSuiteData:
        """Create a new empty test suite.

        Args:
            name: Suite name.
            description: Optional suite description.
            version: Suite version (default "1.0").

        Returns:
            A new TestSuiteData instance ready for adding tests.
        """
        return TestSuiteData(
            name=name,
            version=version,
            description=description,
            defaults=TestDefaults(),
            agents=[],
            tests=[],
        )

    def add_agent(
        self,
        suite: TestSuiteData,
        name: str,
        agent_type: str,
        config: dict[str, Any],
    ) -> TestSuiteData:
        """Add an agent to the suite.

        Args:
            suite: The test suite to add the agent to.
            name: Agent name.
            agent_type: Agent type (e.g., "http", "cli", "container").
            config: Agent-specific configuration.

        Returns:
            The same suite with the agent added.
        """
        agent = AgentConfig(name=name, type=agent_type, config=config)
        suite.agents.append(agent)
        return suite

    def create_custom_test(
        self,
        test_id: str,
        test_name: str,
        task_description: str,
        constraints: Constraints | None = None,
        assertions: list[Assertion] | None = None,
        tags: list[str] | None = None,
        expected_artifacts: list[str] | None = None,
    ) -> TestDefinition:
        """Create a custom test without template.

        Args:
            test_id: Unique test identifier.
            test_name: Human-readable test name.
            task_description: Description of the task for the agent.
            constraints: Optional execution constraints.
            assertions: Optional list of assertions.
            tags: Optional list of tags.
            expected_artifacts: Optional list of expected artifact paths.

        Returns:
            A new TestDefinition instance.
        """
        return TestDefinition(
            id=test_id,
            name=test_name,
            tags=tags or [],
            task=TaskDefinition(
                description=task_description,
                expected_artifacts=expected_artifacts,
            ),
            constraints=constraints or Constraints(),
            assertions=assertions or [],
        )

    def add_test(self, suite: TestSuiteData, test: TestDefinition) -> TestSuiteData:
        """Add a test to the suite.

        Args:
            suite: The test suite to add the test to.
            test: The test definition to add.

        Returns:
            The same suite with the test added.

        Raises:
            ValueError: If a test with the same ID already exists.
        """
        existing_ids = {t.id for t in suite.tests}
        if test.id in existing_ids:
            raise ValueError(f"Duplicate test ID: {test.id}")

        suite.tests.append(test)
        return suite

    def generate_test_id(self, suite: TestSuiteData, prefix: str = "test") -> str:
        """Generate a unique test ID.

        Generates sequential IDs in the format "{prefix}-001", "{prefix}-002", etc.

        Args:
            suite: The test suite to check for existing IDs.
            prefix: Prefix for the generated ID (default "test").

        Returns:
            A unique test ID.

        Raises:
            ValueError: If unable to generate a unique ID (after 9999 attempts).
        """
        existing_ids = {t.id for t in suite.tests}
        for i in range(1, 10000):
            test_id = f"{prefix}-{i:04d}"
            if test_id not in existing_ids:
                return test_id
        raise ValueError("Cannot generate unique test ID after 9999 attempts")

    def register_template(self, template: TestTemplate) -> None:
        """
        Register a custom template for test generation.

        Args:
            template: The template to register.

        Raises:
            ValueError: If a template with the same name already exists.

        Example:
            >>> generator = TestGenerator()
            >>> template = TestTemplate(
            ...     name="api_test",
            ...     description="Test API endpoints",
            ...     category="api",
            ...     task_template="Test the {endpoint} endpoint with {method}",
            ... )
            >>> generator.register_template(template)
        """
        self._template_registry.register(template)

    def get_template(self, name: str) -> TestTemplate:
        """
        Get a registered template by name.

        Args:
            name: The template name.

        Returns:
            The template.

        Raises:
            KeyError: If template not found.
        """
        return self._template_registry.get(name)

    def list_templates(self) -> list[str]:
        """
        List all available template names.

        Returns:
            List of template names (both built-in and custom).
        """
        return self._template_registry.list_templates()

    def create_test_from_template(
        self,
        template_name: str,
        test_id: str,
        test_name: str,
        variables: dict[str, Any],
        constraints: Constraints | None = None,
        extra_assertions: list[Assertion] | None = None,
        extra_tags: list[str] | None = None,
        expected_artifacts: list[str] | None = None,
    ) -> TestDefinition:
        """
        Create a test from a registered template.

        Substitutes variables in the template's task_template, assertions,
        and tags. The template's default constraints can be overridden.

        Args:
            template_name: Name of the template to use.
            test_id: Unique test identifier.
            test_name: Human-readable test name.
            variables: Dictionary of variable values for substitution.
            constraints: Optional constraints override (uses template defaults if None).
            extra_assertions: Additional assertions to add to template defaults.
            extra_tags: Additional tags to add to template defaults.
            expected_artifacts: Optional list of expected artifact paths.

        Returns:
            A new TestDefinition instance.

        Raises:
            KeyError: If template not found.

        Example:
            >>> generator = TestGenerator()
            >>> test = generator.create_test_from_template(
            ...     template_name="file_creation",
            ...     test_id="test-001",
            ...     test_name="Create README",
            ...     variables={"filename": "README.md", "content": "# Hello"},
            ... )
        """
        template = self._template_registry.get(template_name)

        # Substitute variables in task template
        task_description = substitute_variables(template.task_template, variables)

        # Substitute variables in assertions
        assertions: list[Assertion] = []
        for assertion in template.default_assertions:
            assertions.append(substitute_in_assertion(assertion, variables))

        # Add extra assertions
        if extra_assertions:
            assertions.extend(extra_assertions)

        # Substitute variables in tags and add extra tags
        tags: list[str] = []
        for tag in template.tags:
            substituted_tag = substitute_variables(tag, variables)
            tags.append(substituted_tag)

        if extra_tags:
            tags.extend(extra_tags)

        # Use provided constraints or template defaults
        final_constraints = (
            constraints
            if constraints is not None
            else Constraints(
                max_steps=template.default_constraints.max_steps,
                max_tokens=template.default_constraints.max_tokens,
                timeout_seconds=template.default_constraints.timeout_seconds,
                allowed_tools=template.default_constraints.allowed_tools,
                budget_usd=template.default_constraints.budget_usd,
            )
        )

        return TestDefinition(
            id=test_id,
            name=test_name,
            tags=tags,
            task=TaskDefinition(
                description=task_description,
                expected_artifacts=expected_artifacts,
            ),
            constraints=final_constraints,
            assertions=assertions,
        )

    def to_yaml(self, suite: TestSuiteData) -> str:
        """Convert a test suite to YAML string.

        Args:
            suite: TestSuiteData to convert.

        Returns:
            YAML formatted string with proper indentation.

        Example:
            >>> generator = TestGenerator()
            >>> suite = generator.create_suite("my_suite", "My test suite")
            >>> test = generator.create_custom_test("test-001", "Test", "Do something")
            >>> suite = generator.add_test(suite, test)
            >>> yaml_content = generator.to_yaml(suite)
        """
        from atp.generator.writer import YAMLWriter

        writer = YAMLWriter()
        return writer.to_yaml(suite)

    def save(self, suite: TestSuiteData, file_path: str | Path) -> None:
        """Save a test suite to a YAML file.

        Args:
            suite: TestSuiteData to save.
            file_path: Path to output file.

        Example:
            >>> generator = TestGenerator()
            >>> suite = generator.create_suite("my_suite", "My test suite")
            >>> test = generator.create_custom_test("test-001", "Test", "Do something")
            >>> suite = generator.add_test(suite, test)
            >>> generator.save(suite, "output.yaml")
        """
        from atp.generator.writer import YAMLWriter

        writer = YAMLWriter()
        writer.save(suite, file_path)
