"""Tests for test suite definition API endpoints."""

from datetime import datetime

import pytest

from atp.dashboard.models import SuiteDefinition
from atp.dashboard.schemas import (
    AgentConfigCreate,
    AssertionCreate,
    ConstraintsCreate,
    ScoringWeightsCreate,
    SuiteCreateRequest,
    SuiteDefinitionResponse,
    TaskCreate,
    TemplateListResponse,
    TemplateResponse,
    TestCreateRequest,
    TestDefaultsCreate,
    TestResponse,
    YAMLExportResponse,
)


class TestSuiteCreateRequest:
    """Tests for SuiteCreateRequest schema."""

    def test_valid_minimal_request(self) -> None:
        """Test creating a minimal valid request."""
        request = SuiteCreateRequest(name="test-suite")
        assert request.name == "test-suite"
        assert request.version == "1.0"
        assert request.description is None
        assert request.tests == []
        assert request.agents == []

    def test_valid_full_request(self) -> None:
        """Test creating a full valid request."""
        request = SuiteCreateRequest(
            name="my-suite",
            version="2.0",
            description="A test suite",
            defaults=TestDefaultsCreate(
                runs_per_test=3,
                timeout_seconds=600,
                scoring=ScoringWeightsCreate(quality_weight=0.5),
            ),
            agents=[
                AgentConfigCreate(
                    name="agent1", type="http", config={"url": "http://localhost"}
                )
            ],
            tests=[
                TestCreateRequest(
                    id="test-001",
                    name="First Test",
                    task=TaskCreate(description="Do something"),
                )
            ],
        )
        assert request.name == "my-suite"
        assert request.version == "2.0"
        assert request.defaults.runs_per_test == 3
        assert len(request.agents) == 1
        assert len(request.tests) == 1

    def test_name_validation(self) -> None:
        """Test name field validation."""
        with pytest.raises(ValueError):
            SuiteCreateRequest(name="")

    def test_defaults_validation(self) -> None:
        """Test defaults field validation."""
        # runs_per_test minimum is 1
        with pytest.raises(ValueError):
            SuiteCreateRequest(
                name="test",
                defaults=TestDefaultsCreate(runs_per_test=0),
            )


class TestTestCreateRequest:
    """Tests for TestCreateRequest schema."""

    def test_valid_minimal_request(self) -> None:
        """Test creating a minimal valid request."""
        request = TestCreateRequest(
            id="test-001",
            name="Test One",
            task=TaskCreate(description="Do something"),
        )
        assert request.id == "test-001"
        assert request.name == "Test One"
        assert request.task.description == "Do something"
        assert request.tags == []
        assert request.assertions == []

    def test_valid_full_request(self) -> None:
        """Test creating a full valid request."""
        request = TestCreateRequest(
            id="test-002",
            name="Test Two",
            description="A more complex test",
            tags=["smoke", "api"],
            task=TaskCreate(
                description="Create a file",
                input_data={"key": "value"},
                expected_artifacts=["output.txt"],
            ),
            constraints=ConstraintsCreate(
                max_steps=10,
                timeout_seconds=120,
            ),
            assertions=[
                AssertionCreate(type="artifact_exists", config={"path": "output.txt"})
            ],
            scoring=ScoringWeightsCreate(quality_weight=0.6),
        )
        assert request.id == "test-002"
        assert request.description == "A more complex test"
        assert len(request.tags) == 2
        assert request.constraints.max_steps == 10
        assert len(request.assertions) == 1

    def test_id_validation(self) -> None:
        """Test id field validation."""
        with pytest.raises(ValueError):
            TestCreateRequest(
                id="",
                name="Test",
                task=TaskCreate(description="Do something"),
            )


class TestConstraintsCreate:
    """Tests for ConstraintsCreate schema."""

    def test_defaults(self) -> None:
        """Test default values."""
        constraints = ConstraintsCreate()
        assert constraints.max_steps is None
        assert constraints.max_tokens is None
        assert constraints.timeout_seconds == 300
        assert constraints.allowed_tools is None
        assert constraints.budget_usd is None

    def test_custom_values(self) -> None:
        """Test custom values."""
        constraints = ConstraintsCreate(
            max_steps=50,
            max_tokens=10000,
            timeout_seconds=600,
            allowed_tools=["tool1", "tool2"],
            budget_usd=1.5,
        )
        assert constraints.max_steps == 50
        assert constraints.max_tokens == 10000
        assert constraints.timeout_seconds == 600
        assert constraints.allowed_tools == ["tool1", "tool2"]
        assert constraints.budget_usd == 1.5

    def test_timeout_validation(self) -> None:
        """Test timeout_seconds minimum validation."""
        with pytest.raises(ValueError):
            ConstraintsCreate(timeout_seconds=0)

    def test_budget_validation(self) -> None:
        """Test budget_usd minimum validation."""
        with pytest.raises(ValueError):
            ConstraintsCreate(budget_usd=-1.0)


class TestScoringWeightsCreate:
    """Tests for ScoringWeightsCreate schema."""

    def test_defaults(self) -> None:
        """Test default weights."""
        weights = ScoringWeightsCreate()
        assert weights.quality_weight == 0.4
        assert weights.completeness_weight == 0.3
        assert weights.efficiency_weight == 0.2
        assert weights.cost_weight == 0.1

    def test_custom_weights(self) -> None:
        """Test custom weights."""
        weights = ScoringWeightsCreate(
            quality_weight=0.5,
            completeness_weight=0.25,
            efficiency_weight=0.15,
            cost_weight=0.1,
        )
        assert weights.quality_weight == 0.5
        assert weights.completeness_weight == 0.25

    def test_weight_validation(self) -> None:
        """Test weight range validation."""
        with pytest.raises(ValueError):
            ScoringWeightsCreate(quality_weight=1.5)
        with pytest.raises(ValueError):
            ScoringWeightsCreate(quality_weight=-0.1)


class TestTemplateResponse:
    """Tests for TemplateResponse schema."""

    def test_valid_response(self) -> None:
        """Test creating a valid response."""
        response = TemplateResponse(
            name="file_creation",
            description="Test file creation",
            category="file_operations",
            task_template="Create a file named {filename}",
            default_constraints=ConstraintsCreate(max_steps=5),
            default_assertions=[
                AssertionCreate(type="artifact_exists", config={"path": "{filename}"})
            ],
            tags=["file", "basic"],
            variables=["filename"],
        )
        assert response.name == "file_creation"
        assert response.category == "file_operations"
        assert len(response.variables) == 1
        assert "filename" in response.variables


class TestSuiteDefinitionResponse:
    """Tests for SuiteDefinitionResponse schema."""

    def test_valid_response(self) -> None:
        """Test creating a valid response."""
        now = datetime.now()
        response = SuiteDefinitionResponse(
            id=1,
            name="test-suite",
            version="1.0",
            description="A test suite",
            defaults=TestDefaultsCreate(),
            agents=[AgentConfigCreate(name="agent1")],
            tests=[
                TestResponse(
                    id="test-001",
                    name="Test One",
                    description=None,
                    tags=[],
                    task=TaskCreate(description="Do something"),
                    constraints=ConstraintsCreate(),
                    assertions=[],
                    scoring=None,
                )
            ],
            created_at=now,
            updated_at=now,
        )
        assert response.id == 1
        assert response.name == "test-suite"
        assert len(response.tests) == 1


class TestYAMLExportResponse:
    """Tests for YAMLExportResponse schema."""

    def test_valid_response(self) -> None:
        """Test creating a valid response."""
        response = YAMLExportResponse(
            yaml_content="test_suite: my-suite\n",
            suite_name="my-suite",
            test_count=5,
        )
        assert response.suite_name == "my-suite"
        assert response.test_count == 5
        assert "test_suite" in response.yaml_content


class TestSuiteDefinitionModel:
    """Tests for SuiteDefinition database model."""

    def test_model_creation(self) -> None:
        """Test creating a SuiteDefinition model."""
        suite = SuiteDefinition(
            name="test-suite",
            version="1.0",
            description="Test description",
            defaults_json={"runs_per_test": 1},
            agents_json=[{"name": "agent1"}],
            tests_json=[{"id": "test-001", "name": "Test"}],
        )
        assert suite.name == "test-suite"
        assert suite.test_count == 1
        assert suite.agent_count == 1

    def test_test_count_empty(self) -> None:
        """Test test_count with empty tests."""
        suite = SuiteDefinition(
            name="empty-suite",
            defaults_json={},
            agents_json=[],
            tests_json=[],
        )
        assert suite.test_count == 0

    def test_agent_count_empty(self) -> None:
        """Test agent_count with empty agents."""
        suite = SuiteDefinition(
            name="empty-suite",
            defaults_json={},
            agents_json=[],
            tests_json=[],
        )
        assert suite.agent_count == 0


class TestTemplateListResponse:
    """Tests for TemplateListResponse schema."""

    def test_valid_response(self) -> None:
        """Test creating a valid response."""
        response = TemplateListResponse(
            templates=[
                TemplateResponse(
                    name="file_creation",
                    description="Test file creation",
                    category="file_operations",
                    task_template="Create {filename}",
                    default_constraints=ConstraintsCreate(),
                    default_assertions=[],
                    tags=["file"],
                    variables=["filename"],
                )
            ],
            categories=["file_operations"],
            total=1,
        )
        assert response.total == 1
        assert len(response.templates) == 1
        assert "file_operations" in response.categories


class TestAgentConfigCreate:
    """Tests for AgentConfigCreate schema."""

    def test_minimal_config(self) -> None:
        """Test creating minimal agent config."""
        config = AgentConfigCreate(name="agent1")
        assert config.name == "agent1"
        assert config.type is None
        assert config.config == {}

    def test_full_config(self) -> None:
        """Test creating full agent config."""
        config = AgentConfigCreate(
            name="my-agent",
            type="http",
            config={"endpoint": "http://localhost:8000", "timeout": 30},
        )
        assert config.name == "my-agent"
        assert config.type == "http"
        assert config.config["endpoint"] == "http://localhost:8000"


class TestTaskCreate:
    """Tests for TaskCreate schema."""

    def test_minimal_task(self) -> None:
        """Test creating minimal task."""
        task = TaskCreate(description="Do something")
        assert task.description == "Do something"
        assert task.input_data is None
        assert task.expected_artifacts is None

    def test_full_task(self) -> None:
        """Test creating full task."""
        task = TaskCreate(
            description="Process data and save result",
            input_data={"key": "value", "count": 10},
            expected_artifacts=["output.json", "report.txt"],
        )
        assert task.description == "Process data and save result"
        assert task.input_data["key"] == "value"
        assert len(task.expected_artifacts) == 2

    def test_description_validation(self) -> None:
        """Test description field validation."""
        with pytest.raises(ValueError):
            TaskCreate(description="")


class TestAssertionCreate:
    """Tests for AssertionCreate schema."""

    def test_minimal_assertion(self) -> None:
        """Test creating minimal assertion."""
        assertion = AssertionCreate(type="artifact_exists")
        assert assertion.type == "artifact_exists"
        assert assertion.config == {}

    def test_full_assertion(self) -> None:
        """Test creating assertion with config."""
        assertion = AssertionCreate(
            type="artifact_exists",
            config={"path": "output.txt", "content_contains": "success"},
        )
        assert assertion.type == "artifact_exists"
        assert assertion.config["path"] == "output.txt"

    def test_type_validation(self) -> None:
        """Test type field validation."""
        with pytest.raises(ValueError):
            AssertionCreate(type="")
