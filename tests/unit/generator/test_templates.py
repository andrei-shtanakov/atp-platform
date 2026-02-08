"""Unit tests for TestTemplate and TemplateRegistry classes."""

import pytest

from atp.generator.templates import (
    TemplateRegistry,
    TestTemplate,
    extract_variables,
    get_template_variables,
    substitute_in_assertion,
    substitute_variables,
)
from atp.loader.models import Assertion, Constraints


class TestTestTemplateDataclass:
    """Tests for TestTemplate dataclass."""

    def test_create_template_minimal(self) -> None:
        """Test creating template with minimal fields."""
        template = TestTemplate(
            name="test_template",
            description="A test template",
            category="testing",
            task_template="Do something with {item}",
        )

        assert template.name == "test_template"
        assert template.description == "A test template"
        assert template.category == "testing"
        assert template.task_template == "Do something with {item}"
        assert template.default_constraints.timeout_seconds is None
        assert template.default_assertions == []
        assert template.tags == []

    def test_create_template_with_all_fields(self) -> None:
        """Test creating template with all fields."""
        constraints = Constraints(max_steps=10, timeout_seconds=60)
        assertions = [
            Assertion(type="artifact_exists", config={"path": "{output}"}),
        ]

        template = TestTemplate(
            name="complete_template",
            description="Complete template",
            category="full",
            task_template="Process {input} to {output}",
            default_constraints=constraints,
            default_assertions=assertions,
            tags=["tag1", "tag2"],
        )

        assert template.name == "complete_template"
        assert template.default_constraints.max_steps == 10
        assert template.default_constraints.timeout_seconds == 60
        assert len(template.default_assertions) == 1
        assert template.tags == ["tag1", "tag2"]


class TestTemplateRegistry:
    """Tests for TemplateRegistry class."""

    def test_registry_has_builtin_templates(self) -> None:
        """Test that registry includes built-in templates."""
        registry = TemplateRegistry()

        templates = registry.list_templates()

        assert "file_creation" in templates
        assert "data_processing" in templates
        assert "web_research" in templates
        assert "code_generation" in templates

    def test_get_builtin_template(self) -> None:
        """Test getting a built-in template."""
        registry = TemplateRegistry()

        template = registry.get("file_creation")

        assert template.name == "file_creation"
        assert template.category == "file_operations"
        assert "{filename}" in template.task_template
        assert "{content}" in template.task_template

    def test_get_nonexistent_template_raises(self) -> None:
        """Test that getting nonexistent template raises KeyError."""
        registry = TemplateRegistry()

        with pytest.raises(KeyError, match="Template not found: nonexistent"):
            registry.get("nonexistent")

    def test_register_custom_template(self) -> None:
        """Test registering a custom template."""
        registry = TemplateRegistry()
        template = TestTemplate(
            name="custom_template",
            description="My custom template",
            category="custom",
            task_template="Do {action} on {target}",
        )

        registry.register(template)

        assert registry.has_template("custom_template")
        retrieved = registry.get("custom_template")
        assert retrieved.name == "custom_template"

    def test_register_duplicate_template_raises(self) -> None:
        """Test that registering duplicate template raises ValueError."""
        registry = TemplateRegistry()
        template = TestTemplate(
            name="file_creation",  # Already exists as built-in
            description="Duplicate",
            category="test",
            task_template="Test",
        )

        with pytest.raises(ValueError, match="Template already exists: file_creation"):
            registry.register(template)

    def test_list_templates(self) -> None:
        """Test listing all templates."""
        registry = TemplateRegistry()
        registry.register(
            TestTemplate(
                name="custom1",
                description="Custom 1",
                category="custom",
                task_template="Task 1",
            )
        )

        templates = registry.list_templates()

        assert len(templates) >= 5  # 4 built-in + 1 custom
        assert "custom1" in templates

    def test_list_by_category(self) -> None:
        """Test listing templates by category."""
        registry = TemplateRegistry()

        file_templates = registry.list_by_category("file_operations")

        assert len(file_templates) == 1
        assert file_templates[0].name == "file_creation"

    def test_list_by_nonexistent_category(self) -> None:
        """Test listing templates by nonexistent category."""
        registry = TemplateRegistry()

        templates = registry.list_by_category("nonexistent")

        assert templates == []

    def test_has_template(self) -> None:
        """Test checking if template exists."""
        registry = TemplateRegistry()

        assert registry.has_template("file_creation")
        assert not registry.has_template("nonexistent")


class TestSubstituteVariables:
    """Tests for substitute_variables function."""

    def test_substitute_single_variable(self) -> None:
        """Test substituting a single variable."""
        result = substitute_variables("Hello {name}!", {"name": "World"})

        assert result == "Hello World!"

    def test_substitute_multiple_variables(self) -> None:
        """Test substituting multiple variables."""
        result = substitute_variables(
            "Create {filename} with {content}",
            {"filename": "test.txt", "content": "Hello"},
        )

        assert result == "Create test.txt with Hello"

    def test_substitute_missing_variable_unchanged(self) -> None:
        """Test that missing variables are left unchanged."""
        result = substitute_variables(
            "Hello {name}, you have {count} messages",
            {"name": "User"},
        )

        assert result == "Hello User, you have {count} messages"

    def test_substitute_empty_variables(self) -> None:
        """Test substituting with empty dictionary."""
        result = substitute_variables("Hello {name}!", {})

        assert result == "Hello {name}!"

    def test_substitute_no_placeholders(self) -> None:
        """Test text without placeholders."""
        result = substitute_variables("Hello World!", {"name": "Test"})

        assert result == "Hello World!"

    def test_substitute_repeated_variable(self) -> None:
        """Test substituting repeated variable."""
        result = substitute_variables(
            "{name} says hi, {name} is here",
            {"name": "Alice"},
        )

        assert result == "Alice says hi, Alice is here"

    def test_substitute_numeric_value(self) -> None:
        """Test substituting numeric value."""
        result = substitute_variables(
            "Count: {count}",
            {"count": 42},
        )

        assert result == "Count: 42"

    def test_substitute_multiline_template(self) -> None:
        """Test substituting in multiline template."""
        template = """Create a file named '{filename}'
with the following content:
{content}"""
        result = substitute_variables(
            template,
            {"filename": "readme.md", "content": "# Hello"},
        )

        expected = """Create a file named 'readme.md'
with the following content:
# Hello"""
        assert result == expected


class TestSubstituteInAssertion:
    """Tests for substitute_in_assertion function."""

    def test_substitute_simple_string(self) -> None:
        """Test substituting in simple string config."""
        assertion = Assertion(
            type="artifact_exists",
            config={"path": "{filename}"},
        )

        result = substitute_in_assertion(assertion, {"filename": "output.txt"})

        assert result.type == "artifact_exists"
        assert result.config["path"] == "output.txt"

    def test_substitute_nested_dict(self) -> None:
        """Test substituting in nested dictionary config."""
        assertion = Assertion(
            type="complex",
            config={
                "level1": {
                    "level2": "{value}",
                },
            },
        )

        result = substitute_in_assertion(assertion, {"value": "nested_value"})

        assert result.config["level1"]["level2"] == "nested_value"

    def test_substitute_list_in_config(self) -> None:
        """Test substituting in list within config."""
        assertion = Assertion(
            type="multi_check",
            config={
                "paths": ["{file1}", "{file2}"],
            },
        )

        result = substitute_in_assertion(
            assertion, {"file1": "a.txt", "file2": "b.txt"}
        )

        assert result.config["paths"] == ["a.txt", "b.txt"]

    def test_substitute_preserves_non_string_values(self) -> None:
        """Test that non-string values are preserved."""
        assertion = Assertion(
            type="check",
            config={
                "path": "{file}",
                "count": 42,
                "enabled": True,
            },
        )

        result = substitute_in_assertion(assertion, {"file": "test.txt"})

        assert result.config["path"] == "test.txt"
        assert result.config["count"] == 42
        assert result.config["enabled"] is True

    def test_original_assertion_unchanged(self) -> None:
        """Test that original assertion is not modified."""
        assertion = Assertion(
            type="artifact_exists",
            config={"path": "{filename}"},
        )

        substitute_in_assertion(assertion, {"filename": "output.txt"})

        assert assertion.config["path"] == "{filename}"


class TestExtractVariables:
    """Tests for extract_variables function."""

    def test_extract_single_variable(self) -> None:
        """Test extracting single variable."""
        result = extract_variables("Hello {name}!")

        assert result == ["name"]

    def test_extract_multiple_variables(self) -> None:
        """Test extracting multiple variables."""
        result = extract_variables("Create {filename} with {content}")

        assert result == ["filename", "content"]

    def test_extract_no_variables(self) -> None:
        """Test extracting from text without variables."""
        result = extract_variables("Hello World!")

        assert result == []

    def test_extract_repeated_variable(self) -> None:
        """Test extracting repeated variable."""
        result = extract_variables("{name} says hi, {name} is here")

        assert result == ["name", "name"]

    def test_extract_underscore_variable(self) -> None:
        """Test extracting variable with underscore."""
        result = extract_variables("File: {file_name}")

        assert result == ["file_name"]


class TestGetTemplateVariables:
    """Tests for get_template_variables function."""

    def test_get_variables_from_task_template(self) -> None:
        """Test getting variables from task template."""
        template = TestTemplate(
            name="test",
            description="Test",
            category="test",
            task_template="Create {filename} with {content}",
        )

        variables = get_template_variables(template)

        assert "filename" in variables
        assert "content" in variables

    def test_get_variables_from_assertions(self) -> None:
        """Test getting variables from assertions."""
        template = TestTemplate(
            name="test",
            description="Test",
            category="test",
            task_template="Task",
            default_assertions=[
                Assertion(type="check", config={"path": "{output_file}"}),
            ],
        )

        variables = get_template_variables(template)

        assert "output_file" in variables

    def test_get_variables_from_tags(self) -> None:
        """Test getting variables from tags."""
        template = TestTemplate(
            name="test",
            description="Test",
            category="test",
            task_template="Task",
            tags=["language-{language}"],
        )

        variables = get_template_variables(template)

        assert "language" in variables

    def test_get_all_variables_combined(self) -> None:
        """Test getting all variables from all sources."""
        template = TestTemplate(
            name="test",
            description="Test",
            category="test",
            task_template="Process {input} to {output}",
            default_assertions=[
                Assertion(type="check", config={"path": "{output}"}),
            ],
            tags=["type-{type}"],
        )

        variables = get_template_variables(template)

        assert variables == {"input", "output", "type"}

    def test_get_variables_empty_template(self) -> None:
        """Test getting variables from template without placeholders."""
        template = TestTemplate(
            name="test",
            description="Test",
            category="test",
            task_template="Do something simple",
        )

        variables = get_template_variables(template)

        assert variables == set()


class TestBuiltinTemplates:
    """Tests for built-in templates."""

    def test_file_creation_template(self) -> None:
        """Test file_creation template structure."""
        registry = TemplateRegistry()
        template = registry.get("file_creation")

        assert template.name == "file_creation"
        assert template.category == "file_operations"
        assert template.default_constraints.max_steps == 5
        assert template.default_constraints.timeout_seconds == 60
        assert len(template.default_assertions) == 1
        assert template.default_assertions[0].type == "artifact_exists"
        assert "file" in template.tags

        variables = get_template_variables(template)
        assert "filename" in variables
        assert "content" in variables

    def test_data_processing_template(self) -> None:
        """Test data_processing template structure."""
        registry = TemplateRegistry()
        template = registry.get("data_processing")

        assert template.name == "data_processing"
        assert template.category == "data"
        assert template.default_constraints.max_steps == 10
        assert template.default_constraints.timeout_seconds == 120
        assert "data" in template.tags

        variables = get_template_variables(template)
        assert "input_file" in variables
        assert "output_file" in variables
        assert "processing_instructions" in variables

    def test_web_research_template(self) -> None:
        """Test web_research template structure."""
        registry = TemplateRegistry()
        template = registry.get("web_research")

        assert template.name == "web_research"
        assert template.category == "research"
        assert template.default_constraints.max_steps == 20
        assert template.default_constraints.timeout_seconds == 300
        assert "research" in template.tags

        variables = get_template_variables(template)
        assert "topic" in variables
        assert "requirements" in variables
        assert "output_file" in variables

    def test_code_generation_template(self) -> None:
        """Test code_generation template structure."""
        registry = TemplateRegistry()
        template = registry.get("code_generation")

        assert template.name == "code_generation"
        assert template.category == "coding"
        assert template.default_constraints.max_steps == 15
        assert template.default_constraints.timeout_seconds == 180
        assert "code" in template.tags

        variables = get_template_variables(template)
        assert "language" in variables
        assert "specification" in variables
        assert "output_file" in variables
