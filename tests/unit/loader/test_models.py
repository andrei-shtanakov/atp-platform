"""Unit tests for loader models."""

import pytest
from pydantic import ValidationError

from atp.loader.models import (
    Assertion,
    Constraints,
    ScoringWeights,
    TaskDefinition,
    TestDefaults,
    TestDefinition,
    TestSuite,
)


class TestTaskDefinition:
    """Test TaskDefinition model."""

    def test_minimal_task(self):
        """Test creating minimal task with only description."""
        task = TaskDefinition(description="Do something")

        assert task.description == "Do something"
        assert task.input_data is None
        assert task.expected_artifacts is None

    def test_complete_task(self):
        """Test creating complete task with all fields."""
        task = TaskDefinition(
            description="Complete task",
            input_data={"key": "value"},
            expected_artifacts=["output.txt", "report.md"],
        )

        assert task.description == "Complete task"
        assert task.input_data == {"key": "value"}
        assert task.expected_artifacts == ["output.txt", "report.md"]

    def test_empty_description_raises_error(self):
        """Test that empty description raises validation error."""
        with pytest.raises(ValidationError):
            TaskDefinition(description="")


class TestConstraints:
    """Test Constraints model."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = Constraints()

        assert constraints.max_steps is None
        assert constraints.max_tokens is None
        assert constraints.timeout_seconds == 300
        assert constraints.allowed_tools is None
        assert constraints.budget_usd is None

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = Constraints(
            max_steps=50,
            max_tokens=10000,
            timeout_seconds=120,
            allowed_tools=["tool1", "tool2"],
            budget_usd=1.5,
        )

        assert constraints.max_steps == 50
        assert constraints.max_tokens == 10000
        assert constraints.timeout_seconds == 120
        assert constraints.allowed_tools == ["tool1", "tool2"]
        assert constraints.budget_usd == 1.5


class TestScoringWeights:
    """Test ScoringWeights model."""

    def test_default_weights(self):
        """Test default scoring weights."""
        weights = ScoringWeights()

        assert weights.quality_weight == 0.4
        assert weights.completeness_weight == 0.3
        assert weights.efficiency_weight == 0.2
        assert weights.cost_weight == 0.1

    def test_custom_weights(self):
        """Test custom scoring weights."""
        weights = ScoringWeights(
            quality_weight=0.5,
            completeness_weight=0.25,
            efficiency_weight=0.15,
            cost_weight=0.1,
        )

        assert weights.quality_weight == 0.5
        assert weights.completeness_weight == 0.25

    def test_weights_out_of_range(self):
        """Test that weights outside [0, 1] raise error."""
        with pytest.raises(ValidationError):
            ScoringWeights(quality_weight=1.5)

        with pytest.raises(ValidationError):
            ScoringWeights(cost_weight=-0.1)


class TestAssertion:
    """Test Assertion model."""

    def test_minimal_assertion(self):
        """Test assertion with only type."""
        assertion = Assertion(type="artifact_exists")

        assert assertion.type == "artifact_exists"
        assert assertion.config == {}

    def test_assertion_with_config(self):
        """Test assertion with configuration."""
        assertion = Assertion(
            type="llm_eval", config={"criteria": "accuracy", "threshold": 0.8}
        )

        assert assertion.type == "llm_eval"
        assert assertion.config["criteria"] == "accuracy"
        assert assertion.config["threshold"] == 0.8


class TestTestDefinition:
    """Test TestDefinition model."""

    def test_minimal_test(self):
        """Test minimal test definition."""
        test = TestDefinition(
            id="test-001", name="Basic test", task=TaskDefinition(description="Task")
        )

        assert test.id == "test-001"
        assert test.name == "Basic test"
        assert test.description is None
        assert test.tags == []
        assert test.constraints.timeout_seconds == 300
        assert test.assertions == []
        assert test.scoring is None

    def test_complete_test(self):
        """Test complete test definition with all fields."""
        test = TestDefinition(
            id="test-001",
            name="Complete test",
            description="A complete test",
            tags=["smoke", "regression"],
            task=TaskDefinition(description="Task"),
            constraints=Constraints(max_steps=10),
            assertions=[Assertion(type="artifact_exists")],
            scoring=ScoringWeights(quality_weight=0.5),
        )

        assert test.id == "test-001"
        assert test.tags == ["smoke", "regression"]
        assert test.constraints.max_steps == 10
        assert len(test.assertions) == 1
        assert test.scoring is not None
        assert test.scoring.quality_weight == 0.5


class TestTestSuite:
    """Test TestSuite model."""

    def test_minimal_suite(self):
        """Test minimal test suite."""
        suite = TestSuite(
            test_suite="sample",
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test",
                    task=TaskDefinition(description="Task"),
                )
            ],
        )

        assert suite.test_suite == "sample"
        assert suite.version == "1.0"
        assert len(suite.tests) == 1

    def test_suite_requires_at_least_one_test(self):
        """Test that suite requires at least one test."""
        with pytest.raises(ValidationError):
            TestSuite(test_suite="sample", tests=[])

    def test_apply_defaults_constraints(self):
        """Test that apply_defaults propagates constraint defaults."""
        suite = TestSuite(
            test_suite="sample",
            defaults=TestDefaults(
                constraints=Constraints(max_steps=50, timeout_seconds=120)
            ),
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test",
                    task=TaskDefinition(description="Task"),
                    constraints=Constraints(),
                )
            ],
        )

        suite.apply_defaults()

        # Default max_steps should be applied
        assert suite.tests[0].constraints.max_steps == 50
        # Default timeout should be applied
        assert suite.tests[0].constraints.timeout_seconds == 120

    def test_apply_defaults_preserves_explicit_values(self):
        """Test that explicit values are not overridden by defaults."""
        suite = TestSuite(
            test_suite="sample",
            defaults=TestDefaults(constraints=Constraints(max_steps=50)),
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test",
                    task=TaskDefinition(description="Task"),
                    constraints=Constraints(max_steps=100),
                )
            ],
        )

        suite.apply_defaults()

        # Explicit value should be preserved
        assert suite.tests[0].constraints.max_steps == 100

    def test_apply_defaults_scoring(self):
        """Test that apply_defaults propagates scoring defaults."""
        custom_weights = ScoringWeights(quality_weight=0.6, completeness_weight=0.4)
        suite = TestSuite(
            test_suite="sample",
            defaults=TestDefaults(scoring=custom_weights),
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test",
                    task=TaskDefinition(description="Task"),
                )
            ],
        )

        suite.apply_defaults()

        # Default scoring should be applied
        assert suite.tests[0].scoring is not None
        assert suite.tests[0].scoring.quality_weight == 0.6

    def test_apply_defaults_no_override_explicit_scoring(self):
        """Test that explicit scoring is not overridden."""
        suite = TestSuite(
            test_suite="sample",
            defaults=TestDefaults(
                scoring=ScoringWeights(quality_weight=0.6, completeness_weight=0.4)
            ),
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test",
                    task=TaskDefinition(description="Task"),
                    scoring=ScoringWeights(quality_weight=0.8, completeness_weight=0.2),
                )
            ],
        )

        suite.apply_defaults()

        # Explicit scoring should be preserved
        assert suite.tests[0].scoring.quality_weight == 0.8
