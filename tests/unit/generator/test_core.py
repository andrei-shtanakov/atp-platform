"""Unit tests for TestGenerator core class."""

import pytest

from atp.generator.core import TestGenerator, TestSuiteData
from atp.loader.models import (
    Assertion,
    Constraints,
    TaskDefinition,
    TestDefinition,
)


class TestTestSuiteData:
    """Tests for TestSuiteData class."""

    def test_create_empty_suite_data(self) -> None:
        """Test creating empty suite data."""
        suite = TestSuiteData(name="test_suite")

        assert suite.name == "test_suite"
        assert suite.version == "1.0"
        assert suite.description is None
        assert suite.agents == []
        assert suite.tests == []

    def test_create_suite_data_with_all_fields(self) -> None:
        """Test creating suite data with all fields."""
        suite = TestSuiteData(
            name="my_suite",
            version="2.0",
            description="A test suite",
        )

        assert suite.name == "my_suite"
        assert suite.version == "2.0"
        assert suite.description == "A test suite"

    def test_to_test_suite_with_tests(self) -> None:
        """Test converting suite data to TestSuite model."""
        suite = TestSuiteData(name="test_suite", description="Test")
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Do something"),
            )
        )

        test_suite = suite.to_test_suite()

        assert test_suite.test_suite == "test_suite"
        assert test_suite.description == "Test"
        assert len(test_suite.tests) == 1
        assert test_suite.tests[0].id == "test-001"

    def test_to_test_suite_without_tests_raises(self) -> None:
        """Test that converting empty suite data raises ValueError."""
        suite = TestSuiteData(name="empty_suite")

        with pytest.raises(ValueError, match="Cannot create TestSuite with no tests"):
            suite.to_test_suite()


class TestTestGeneratorCreateSuite:
    """Tests for TestGenerator.create_suite() method."""

    def test_create_suite_minimal(self) -> None:
        """Test creating suite with minimal arguments."""
        generator = TestGenerator()

        suite = generator.create_suite("my_suite")

        assert suite.name == "my_suite"
        assert suite.version == "1.0"
        assert suite.description is None
        assert suite.agents == []
        assert suite.tests == []

    def test_create_suite_with_description(self) -> None:
        """Test creating suite with description."""
        generator = TestGenerator()

        suite = generator.create_suite("my_suite", description="A test suite")

        assert suite.name == "my_suite"
        assert suite.description == "A test suite"

    def test_create_suite_with_version(self) -> None:
        """Test creating suite with custom version."""
        generator = TestGenerator()

        suite = generator.create_suite("my_suite", version="2.0")

        assert suite.version == "2.0"

    def test_create_suite_with_all_args(self) -> None:
        """Test creating suite with all arguments."""
        generator = TestGenerator()

        suite = generator.create_suite(
            "my_suite",
            description="Description",
            version="3.0",
        )

        assert suite.name == "my_suite"
        assert suite.description == "Description"
        assert suite.version == "3.0"


class TestTestGeneratorAddAgent:
    """Tests for TestGenerator.add_agent() method."""

    def test_add_agent_http(self) -> None:
        """Test adding an HTTP agent."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")

        result = generator.add_agent(
            suite,
            name="http-agent",
            agent_type="http",
            config={"endpoint": "http://localhost:8000"},
        )

        assert result is suite
        assert len(suite.agents) == 1
        assert suite.agents[0].name == "http-agent"
        assert suite.agents[0].type == "http"
        assert suite.agents[0].config == {"endpoint": "http://localhost:8000"}

    def test_add_agent_cli(self) -> None:
        """Test adding a CLI agent."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")

        generator.add_agent(
            suite,
            name="cli-agent",
            agent_type="cli",
            config={"command": "python", "args": ["agent.py"]},
        )

        assert len(suite.agents) == 1
        assert suite.agents[0].name == "cli-agent"
        assert suite.agents[0].type == "cli"
        assert suite.agents[0].config["command"] == "python"

    def test_add_multiple_agents(self) -> None:
        """Test adding multiple agents."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")

        generator.add_agent(suite, "agent1", "http", {"endpoint": "http://a:8000"})
        generator.add_agent(suite, "agent2", "cli", {"command": "python"})
        generator.add_agent(suite, "agent3", "container", {"image": "agent:latest"})

        assert len(suite.agents) == 3
        assert suite.agents[0].name == "agent1"
        assert suite.agents[1].name == "agent2"
        assert suite.agents[2].name == "agent3"

    def test_add_agent_returns_suite(self) -> None:
        """Test that add_agent returns the suite for chaining."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")

        result = generator.add_agent(suite, "agent", "http", {"endpoint": "http://x"})

        assert result is suite


class TestTestGeneratorCreateCustomTest:
    """Tests for TestGenerator.create_custom_test() method."""

    def test_create_custom_test_minimal(self) -> None:
        """Test creating custom test with minimal arguments."""
        generator = TestGenerator()

        test = generator.create_custom_test(
            test_id="test-001",
            test_name="My Test",
            task_description="Do something",
        )

        assert test.id == "test-001"
        assert test.name == "My Test"
        assert test.task.description == "Do something"
        assert test.tags == []
        assert test.assertions == []
        assert test.constraints.timeout_seconds is None

    def test_create_custom_test_with_constraints(self) -> None:
        """Test creating custom test with constraints."""
        generator = TestGenerator()
        constraints = Constraints(max_steps=10, timeout_seconds=60)

        test = generator.create_custom_test(
            test_id="test-001",
            test_name="My Test",
            task_description="Do something",
            constraints=constraints,
        )

        assert test.constraints.max_steps == 10
        assert test.constraints.timeout_seconds == 60

    def test_create_custom_test_with_assertions(self) -> None:
        """Test creating custom test with assertions."""
        generator = TestGenerator()
        assertions = [
            Assertion(type="artifact_exists", config={"path": "output.txt"}),
            Assertion(type="llm_eval", config={"criteria": "quality"}),
        ]

        test = generator.create_custom_test(
            test_id="test-001",
            test_name="My Test",
            task_description="Do something",
            assertions=assertions,
        )

        assert len(test.assertions) == 2
        assert test.assertions[0].type == "artifact_exists"
        assert test.assertions[1].type == "llm_eval"

    def test_create_custom_test_with_tags(self) -> None:
        """Test creating custom test with tags."""
        generator = TestGenerator()

        test = generator.create_custom_test(
            test_id="test-001",
            test_name="My Test",
            task_description="Do something",
            tags=["smoke", "basic"],
        )

        assert test.tags == ["smoke", "basic"]

    def test_create_custom_test_with_expected_artifacts(self) -> None:
        """Test creating custom test with expected artifacts."""
        generator = TestGenerator()

        test = generator.create_custom_test(
            test_id="test-001",
            test_name="My Test",
            task_description="Do something",
            expected_artifacts=["output.txt", "report.md"],
        )

        assert test.task.expected_artifacts == ["output.txt", "report.md"]

    def test_create_custom_test_with_all_args(self) -> None:
        """Test creating custom test with all arguments."""
        generator = TestGenerator()

        test = generator.create_custom_test(
            test_id="test-999",
            test_name="Complete Test",
            task_description="Complete task",
            constraints=Constraints(max_steps=20, timeout_seconds=120),
            assertions=[Assertion(type="artifact_exists", config={"path": "out.txt"})],
            tags=["integration", "slow"],
            expected_artifacts=["out.txt"],
        )

        assert test.id == "test-999"
        assert test.name == "Complete Test"
        assert test.task.description == "Complete task"
        assert test.constraints.max_steps == 20
        assert test.constraints.timeout_seconds == 120
        assert len(test.assertions) == 1
        assert test.tags == ["integration", "slow"]
        assert test.task.expected_artifacts == ["out.txt"]


class TestTestGeneratorAddTest:
    """Tests for TestGenerator.add_test() method."""

    def test_add_test_to_empty_suite(self) -> None:
        """Test adding a test to empty suite."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")
        test = generator.create_custom_test("test-001", "Test", "Do something")

        result = generator.add_test(suite, test)

        assert result is suite
        assert len(suite.tests) == 1
        assert suite.tests[0].id == "test-001"

    def test_add_multiple_tests(self) -> None:
        """Test adding multiple tests."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")

        generator.add_test(
            suite, generator.create_custom_test("test-001", "Test 1", "Task 1")
        )
        generator.add_test(
            suite, generator.create_custom_test("test-002", "Test 2", "Task 2")
        )
        generator.add_test(
            suite, generator.create_custom_test("test-003", "Test 3", "Task 3")
        )

        assert len(suite.tests) == 3
        assert suite.tests[0].id == "test-001"
        assert suite.tests[1].id == "test-002"
        assert suite.tests[2].id == "test-003"

    def test_add_test_duplicate_id_raises(self) -> None:
        """Test that adding test with duplicate ID raises ValueError."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")

        generator.add_test(
            suite, generator.create_custom_test("test-001", "Test 1", "Task 1")
        )

        with pytest.raises(ValueError, match="Duplicate test ID: test-001"):
            generator.add_test(
                suite, generator.create_custom_test("test-001", "Test 2", "Task 2")
            )

    def test_add_test_returns_suite(self) -> None:
        """Test that add_test returns the suite for chaining."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")
        test = generator.create_custom_test("test-001", "Test", "Task")

        result = generator.add_test(suite, test)

        assert result is suite


class TestTestGeneratorGenerateTestId:
    """Tests for TestGenerator.generate_test_id() method."""

    def test_generate_test_id_empty_suite(self) -> None:
        """Test generating ID for empty suite."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")

        test_id = generator.generate_test_id(suite)

        assert test_id == "test-0001"

    def test_generate_test_id_with_existing_tests(self) -> None:
        """Test generating ID with existing tests."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")
        generator.add_test(
            suite, generator.create_custom_test("test-0001", "Test 1", "Task 1")
        )

        test_id = generator.generate_test_id(suite)

        assert test_id == "test-0002"

    def test_generate_test_id_fills_gaps(self) -> None:
        """Test that generate_test_id fills gaps in sequence."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")
        # Add test-0001 and test-0003, leaving gap at test-0002
        generator.add_test(
            suite, generator.create_custom_test("test-0001", "Test 1", "Task 1")
        )
        generator.add_test(
            suite, generator.create_custom_test("test-0003", "Test 3", "Task 3")
        )

        test_id = generator.generate_test_id(suite)

        assert test_id == "test-0002"

    def test_generate_test_id_custom_prefix(self) -> None:
        """Test generating ID with custom prefix."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")

        test_id = generator.generate_test_id(suite, prefix="smoke")

        assert test_id == "smoke-0001"

    def test_generate_test_id_custom_prefix_with_existing(self) -> None:
        """Test generating ID with custom prefix and existing tests."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")
        # Add tests with different prefix
        generator.add_test(
            suite, generator.create_custom_test("smoke-0001", "Smoke 1", "Task 1")
        )
        generator.add_test(
            suite, generator.create_custom_test("smoke-0002", "Smoke 2", "Task 2")
        )

        test_id = generator.generate_test_id(suite, prefix="smoke")

        assert test_id == "smoke-0003"

    def test_generate_test_id_different_prefix_starts_at_0001(self) -> None:
        """Test that different prefix starts at 0001."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")
        # Add test with "test" prefix
        generator.add_test(
            suite, generator.create_custom_test("test-0001", "Test 1", "Task 1")
        )

        # Generate with "smoke" prefix
        test_id = generator.generate_test_id(suite, prefix="smoke")

        assert test_id == "smoke-0001"

    def test_generate_test_id_sequential(self) -> None:
        """Test generating multiple sequential IDs."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")

        # Generate and add tests sequentially
        for i in range(5):
            test_id = generator.generate_test_id(suite)
            test = generator.create_custom_test(
                test_id, f"Test {i + 1}", f"Task {i + 1}"
            )
            generator.add_test(suite, test)

        assert len(suite.tests) == 5
        assert suite.tests[0].id == "test-0001"
        assert suite.tests[4].id == "test-0005"

    def test_generate_test_id_exhausted_raises(self) -> None:
        """Test that exhausting all IDs raises ValueError."""
        generator = TestGenerator()
        suite = generator.create_suite("my_suite")

        # Pre-fill all possible IDs (test-0001 through test-9999)
        for i in range(1, 10000):
            test_id = f"test-{i:04d}"
            test = generator.create_custom_test(test_id, f"Test {i}", f"Task {i}")
            suite.tests.append(test)  # Bypass add_test to avoid validation overhead

        with pytest.raises(ValueError, match="Cannot generate unique test ID"):
            generator.generate_test_id(suite)


class TestTestGeneratorIntegration:
    """Integration tests for TestGenerator."""

    def test_full_suite_creation_workflow(self) -> None:
        """Test complete workflow of creating a test suite."""
        generator = TestGenerator()

        # Create suite
        suite = generator.create_suite(
            "integration_tests", description="Integration test suite", version="1.0"
        )

        # Add agents
        generator.add_agent(
            suite,
            name="http-agent",
            agent_type="http",
            config={"endpoint": "http://localhost:8000"},
        )
        generator.add_agent(
            suite,
            name="cli-agent",
            agent_type="cli",
            config={"command": "python", "args": ["agent.py"]},
        )

        # Add tests using generated IDs
        test1_id = generator.generate_test_id(suite)
        test1 = generator.create_custom_test(
            test_id=test1_id,
            test_name="File Creation Test",
            task_description="Create a file named output.txt",
            constraints=Constraints(max_steps=5, timeout_seconds=60),
            assertions=[
                Assertion(type="artifact_exists", config={"path": "output.txt"})
            ],
            tags=["smoke"],
            expected_artifacts=["output.txt"],
        )
        generator.add_test(suite, test1)

        test2_id = generator.generate_test_id(suite)
        test2 = generator.create_custom_test(
            test_id=test2_id,
            test_name="Data Processing Test",
            task_description="Process input.json and create output.json",
            constraints=Constraints(max_steps=10, timeout_seconds=120),
            tags=["regression"],
        )
        generator.add_test(suite, test2)

        # Verify suite structure
        assert suite.name == "integration_tests"
        assert suite.description == "Integration test suite"
        assert len(suite.agents) == 2
        assert len(suite.tests) == 2
        assert suite.tests[0].id == "test-0001"
        assert suite.tests[1].id == "test-0002"

        # Convert to TestSuite model
        test_suite = suite.to_test_suite()
        assert test_suite.test_suite == "integration_tests"
        assert len(test_suite.tests) == 2

    def test_chained_operations(self) -> None:
        """Test chaining operations using return values."""
        generator = TestGenerator()

        # Chain operations
        suite = generator.create_suite("chained_suite")
        suite = generator.add_agent(suite, "agent1", "http", {"endpoint": "http://a"})
        suite = generator.add_agent(suite, "agent2", "cli", {"command": "python"})

        test = generator.create_custom_test("test-001", "Test", "Do something")
        suite = generator.add_test(suite, test)

        assert len(suite.agents) == 2
        assert len(suite.tests) == 1


class TestTestGeneratorTemplates:
    """Tests for TestGenerator template methods."""

    def test_list_templates_includes_builtins(self) -> None:
        """Test that list_templates includes built-in templates."""
        generator = TestGenerator()

        templates = generator.list_templates()

        assert "file_creation" in templates
        assert "data_processing" in templates
        assert "web_research" in templates
        assert "code_generation" in templates

    def test_get_template(self) -> None:
        """Test getting a template by name."""
        generator = TestGenerator()

        template = generator.get_template("file_creation")

        assert template.name == "file_creation"
        assert template.category == "file_operations"

    def test_get_template_not_found_raises(self) -> None:
        """Test that getting nonexistent template raises KeyError."""
        generator = TestGenerator()

        with pytest.raises(KeyError, match="Template not found: nonexistent"):
            generator.get_template("nonexistent")

    def test_register_template(self) -> None:
        """Test registering a custom template."""
        from atp.generator.templates import TestTemplate

        generator = TestGenerator()
        template = TestTemplate(
            name="my_custom",
            description="Custom template",
            category="custom",
            task_template="Do {action}",
        )

        generator.register_template(template)

        assert "my_custom" in generator.list_templates()
        retrieved = generator.get_template("my_custom")
        assert retrieved.name == "my_custom"

    def test_register_duplicate_template_raises(self) -> None:
        """Test that registering duplicate template raises ValueError."""
        from atp.generator.templates import TestTemplate

        generator = TestGenerator()
        template = TestTemplate(
            name="file_creation",
            description="Duplicate",
            category="test",
            task_template="Test",
        )

        with pytest.raises(ValueError, match="Template already exists"):
            generator.register_template(template)

    def test_create_test_from_template_basic(self) -> None:
        """Test creating test from template with basic variables."""
        generator = TestGenerator()

        test = generator.create_test_from_template(
            template_name="file_creation",
            test_id="test-001",
            test_name="Create README",
            variables={"filename": "README.md", "content": "# Hello World"},
        )

        assert test.id == "test-001"
        assert test.name == "Create README"
        assert "README.md" in test.task.description
        assert "# Hello World" in test.task.description
        assert len(test.assertions) == 1
        assert test.assertions[0].config["path"] == "README.md"
        assert "file" in test.tags

    def test_create_test_from_template_with_constraints_override(self) -> None:
        """Test creating test from template with constraints override."""
        generator = TestGenerator()
        custom_constraints = Constraints(max_steps=3, timeout_seconds=30)

        test = generator.create_test_from_template(
            template_name="file_creation",
            test_id="test-001",
            test_name="Quick File",
            variables={"filename": "out.txt", "content": "test"},
            constraints=custom_constraints,
        )

        assert test.constraints.max_steps == 3
        assert test.constraints.timeout_seconds == 30

    def test_create_test_from_template_uses_default_constraints(self) -> None:
        """Test that template default constraints are used when not overridden."""
        generator = TestGenerator()

        test = generator.create_test_from_template(
            template_name="file_creation",
            test_id="test-001",
            test_name="File Test",
            variables={"filename": "test.txt", "content": "content"},
        )

        # file_creation template has max_steps=5, timeout=60
        assert test.constraints.max_steps == 5
        assert test.constraints.timeout_seconds == 60

    def test_create_test_from_template_with_extra_assertions(self) -> None:
        """Test creating test from template with extra assertions."""
        generator = TestGenerator()
        extra_assertions = [
            Assertion(type="llm_eval", config={"criteria": "quality"}),
        ]

        test = generator.create_test_from_template(
            template_name="file_creation",
            test_id="test-001",
            test_name="File Test",
            variables={"filename": "test.txt", "content": "content"},
            extra_assertions=extra_assertions,
        )

        assert len(test.assertions) == 2
        assert test.assertions[0].type == "artifact_exists"
        assert test.assertions[1].type == "llm_eval"

    def test_create_test_from_template_with_extra_tags(self) -> None:
        """Test creating test from template with extra tags."""
        generator = TestGenerator()

        test = generator.create_test_from_template(
            template_name="file_creation",
            test_id="test-001",
            test_name="File Test",
            variables={"filename": "test.txt", "content": "content"},
            extra_tags=["smoke", "critical"],
        )

        assert "file" in test.tags  # From template
        assert "smoke" in test.tags  # Extra
        assert "critical" in test.tags  # Extra

    def test_create_test_from_template_with_expected_artifacts(self) -> None:
        """Test creating test from template with expected artifacts."""
        generator = TestGenerator()

        test = generator.create_test_from_template(
            template_name="file_creation",
            test_id="test-001",
            test_name="File Test",
            variables={"filename": "test.txt", "content": "content"},
            expected_artifacts=["test.txt", "log.txt"],
        )

        assert test.task.expected_artifacts == ["test.txt", "log.txt"]

    def test_create_test_from_template_nonexistent_raises(self) -> None:
        """Test that creating test from nonexistent template raises KeyError."""
        generator = TestGenerator()

        with pytest.raises(KeyError, match="Template not found: nonexistent"):
            generator.create_test_from_template(
                template_name="nonexistent",
                test_id="test-001",
                test_name="Test",
                variables={},
            )

    def test_create_test_from_code_generation_template(self) -> None:
        """Test creating test from code_generation template."""
        generator = TestGenerator()

        test = generator.create_test_from_template(
            template_name="code_generation",
            test_id="code-001",
            test_name="Generate Python Function",
            variables={
                "language": "Python",
                "specification": "a function that calculates factorial",
                "output_file": "factorial.py",
            },
        )

        assert test.id == "code-001"
        assert "Python" in test.task.description
        assert "factorial" in test.task.description
        assert test.assertions[0].config["path"] == "factorial.py"
        # Check that language variable substitution in tags works
        assert "Python" in test.tags

    def test_create_test_from_data_processing_template(self) -> None:
        """Test creating test from data_processing template."""
        generator = TestGenerator()

        test = generator.create_test_from_template(
            template_name="data_processing",
            test_id="data-001",
            test_name="Process CSV",
            variables={
                "input_file": "data.csv",
                "processing_instructions": "Convert to JSON format",
                "output_file": "data.json",
            },
        )

        assert "data.csv" in test.task.description
        assert "Convert to JSON format" in test.task.description
        assert "data.json" in test.task.description
        assert test.assertions[0].config["path"] == "data.json"

    def test_create_test_from_web_research_template(self) -> None:
        """Test creating test from web_research template."""
        generator = TestGenerator()

        test = generator.create_test_from_template(
            template_name="web_research",
            test_id="research-001",
            test_name="Research AI Trends",
            variables={
                "topic": "Large Language Models in 2025",
                "requirements": "Include recent developments and future predictions",
                "output_file": "ai_research.md",
            },
        )

        assert "Large Language Models in 2025" in test.task.description
        assert "recent developments" in test.task.description
        assert test.assertions[0].config["path"] == "ai_research.md"
        assert test.constraints.max_steps == 20
        assert test.constraints.timeout_seconds == 300


class TestTestGeneratorTemplateIntegration:
    """Integration tests for template-based test generation."""

    def test_full_workflow_with_templates(self) -> None:
        """Test complete workflow using templates."""
        generator = TestGenerator()

        # Create suite
        suite = generator.create_suite("template_tests", description="Template tests")

        # Add agent
        generator.add_agent(
            suite,
            name="test-agent",
            agent_type="cli",
            config={"command": "python", "args": ["agent.py"]},
        )

        # Create tests from templates
        test1 = generator.create_test_from_template(
            template_name="file_creation",
            test_id=generator.generate_test_id(suite),
            test_name="Create Config File",
            variables={"filename": "config.json", "content": '{"key": "value"}'},
        )
        generator.add_test(suite, test1)

        test2 = generator.create_test_from_template(
            template_name="code_generation",
            test_id=generator.generate_test_id(suite),
            test_name="Generate Helper",
            variables={
                "language": "Python",
                "specification": "a helper function",
                "output_file": "helper.py",
            },
        )
        generator.add_test(suite, test2)

        # Verify
        assert len(suite.tests) == 2
        assert suite.tests[0].id == "test-0001"
        assert suite.tests[1].id == "test-0002"

        # Convert to TestSuite
        test_suite = suite.to_test_suite()
        assert len(test_suite.tests) == 2

    def test_custom_template_workflow(self) -> None:
        """Test workflow with custom registered template."""
        from atp.generator.templates import TestTemplate

        generator = TestGenerator()

        # Register custom template
        api_template = TestTemplate(
            name="api_test",
            description="Test API endpoint",
            category="api",
            task_template=(
                "Test the {method} endpoint at {endpoint}.\n"
                "Expected response: {expected_response}"
            ),
            default_constraints=Constraints(max_steps=5, timeout_seconds=30),
            default_assertions=[
                Assertion(type="response_check", config={"status": "{status_code}"}),
            ],
            tags=["api", "{method}"],
        )
        generator.register_template(api_template)

        # Create suite and test
        suite = generator.create_suite("api_tests")
        test = generator.create_test_from_template(
            template_name="api_test",
            test_id="api-001",
            test_name="Test GET Users",
            variables={
                "method": "GET",
                "endpoint": "/api/users",
                "expected_response": '{"users": []}',
                "status_code": "200",
            },
        )
        generator.add_test(suite, test)

        assert "GET" in test.task.description
        assert "/api/users" in test.task.description
        assert test.assertions[0].config["status"] == "200"
        assert "api" in test.tags
        assert "GET" in test.tags
