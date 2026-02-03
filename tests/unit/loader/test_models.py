"""Unit tests for loader models."""

import pytest
from pydantic import ValidationError

from atp.loader.models import (
    Assertion,
    CollaborationConfig,
    ComparisonConfig,
    Constraints,
    ContextAccumulationMode,
    HandoffConfig,
    HandoffTrigger,
    MultiAgentMode,
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
        assert constraints.timeout_seconds is None
        assert constraints.effective_timeout_seconds == 300
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
        assert test.constraints.timeout_seconds is None
        assert test.constraints.effective_timeout_seconds == 300
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

    def test_filter_by_tags_no_filter(self):
        """Test that filter_by_tags with None returns same suite."""
        suite = TestSuite(
            test_suite="sample",
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test 1",
                    tags=["smoke"],
                    task=TaskDefinition(description="Task"),
                ),
                TestDefinition(
                    id="test-002",
                    name="Test 2",
                    tags=["slow"],
                    task=TaskDefinition(description="Task"),
                ),
            ],
        )

        filtered = suite.filter_by_tags(None)
        assert len(filtered.tests) == 2

    def test_filter_by_tags_empty_string(self):
        """Test that filter_by_tags with empty string returns all tests."""
        suite = TestSuite(
            test_suite="sample",
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test 1",
                    tags=["smoke"],
                    task=TaskDefinition(description="Task"),
                ),
                TestDefinition(
                    id="test-002",
                    name="Test 2",
                    tags=["slow"],
                    task=TaskDefinition(description="Task"),
                ),
            ],
        )

        filtered = suite.filter_by_tags("")
        assert len(filtered.tests) == 2

    def test_filter_by_tags_include_single(self):
        """Test filtering with single include tag."""
        suite = TestSuite(
            test_suite="sample",
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test 1",
                    tags=["smoke"],
                    task=TaskDefinition(description="Task"),
                ),
                TestDefinition(
                    id="test-002",
                    name="Test 2",
                    tags=["slow"],
                    task=TaskDefinition(description="Task"),
                ),
                TestDefinition(
                    id="test-003",
                    name="Test 3",
                    tags=["smoke", "fast"],
                    task=TaskDefinition(description="Task"),
                ),
            ],
        )

        filtered = suite.filter_by_tags("smoke")
        assert len(filtered.tests) == 2
        assert filtered.tests[0].id == "test-001"
        assert filtered.tests[1].id == "test-003"

    def test_filter_by_tags_include_multiple(self):
        """Test filtering with multiple include tags (OR logic)."""
        suite = TestSuite(
            test_suite="sample",
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test 1",
                    tags=["smoke"],
                    task=TaskDefinition(description="Task"),
                ),
                TestDefinition(
                    id="test-002",
                    name="Test 2",
                    tags=["core"],
                    task=TaskDefinition(description="Task"),
                ),
                TestDefinition(
                    id="test-003",
                    name="Test 3",
                    tags=["slow"],
                    task=TaskDefinition(description="Task"),
                ),
            ],
        )

        filtered = suite.filter_by_tags("smoke,core")
        assert len(filtered.tests) == 2
        assert filtered.tests[0].id == "test-001"
        assert filtered.tests[1].id == "test-002"

    def test_filter_by_tags_exclude(self):
        """Test filtering with exclude tag."""
        suite = TestSuite(
            test_suite="sample",
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test 1",
                    tags=["smoke"],
                    task=TaskDefinition(description="Task"),
                ),
                TestDefinition(
                    id="test-002",
                    name="Test 2",
                    tags=["slow"],
                    task=TaskDefinition(description="Task"),
                ),
                TestDefinition(
                    id="test-003",
                    name="Test 3",
                    tags=["smoke", "fast"],
                    task=TaskDefinition(description="Task"),
                ),
            ],
        )

        filtered = suite.filter_by_tags("!slow")
        assert len(filtered.tests) == 2
        assert filtered.tests[0].id == "test-001"
        assert filtered.tests[1].id == "test-003"

    def test_filter_by_tags_combination(self):
        """Test filtering with combination of include and exclude."""
        suite = TestSuite(
            test_suite="sample",
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test 1",
                    tags=["smoke"],
                    task=TaskDefinition(description="Task"),
                ),
                TestDefinition(
                    id="test-002",
                    name="Test 2",
                    tags=["smoke", "slow"],
                    task=TaskDefinition(description="Task"),
                ),
                TestDefinition(
                    id="test-003",
                    name="Test 3",
                    tags=["core"],
                    task=TaskDefinition(description="Task"),
                ),
            ],
        )

        filtered = suite.filter_by_tags("smoke,!slow")
        assert len(filtered.tests) == 1
        assert filtered.tests[0].id == "test-001"

    def test_filter_by_tags_no_matches(self):
        """Test filtering that results in no matches."""
        suite = TestSuite(
            test_suite="sample",
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test 1",
                    tags=["slow"],
                    task=TaskDefinition(description="Task"),
                ),
            ],
        )

        filtered = suite.filter_by_tags("smoke")
        assert len(filtered.tests) == 0

    def test_filter_by_tags_preserves_suite_metadata(self):
        """Test that filtering preserves suite metadata."""
        suite = TestSuite(
            test_suite="sample",
            version="2.0",
            description="Test suite description",
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test 1",
                    tags=["smoke"],
                    task=TaskDefinition(description="Task"),
                ),
            ],
        )

        filtered = suite.filter_by_tags("smoke")
        assert filtered.test_suite == "sample"
        assert filtered.version == "2.0"
        assert filtered.description == "Test suite description"


class TestMultiAgentMode:
    """Test MultiAgentMode enum."""

    def test_comparison_mode(self):
        """Test comparison mode value."""
        assert MultiAgentMode.COMPARISON.value == "comparison"

    def test_collaboration_mode(self):
        """Test collaboration mode value."""
        assert MultiAgentMode.COLLABORATION.value == "collaboration"

    def test_handoff_mode(self):
        """Test handoff mode value."""
        assert MultiAgentMode.HANDOFF.value == "handoff"


class TestCollaborationConfig:
    """Test CollaborationConfig model."""

    def test_default_values(self):
        """Test default collaboration config values."""
        config = CollaborationConfig()

        assert config.max_turns == 10
        assert config.turn_timeout_seconds == 60.0
        assert config.require_consensus is False
        assert config.allow_parallel_turns is False
        assert config.coordinator_agent is None
        assert config.termination_condition == "all_complete"

    def test_custom_values(self):
        """Test custom collaboration config values."""
        config = CollaborationConfig(
            max_turns=5,
            turn_timeout_seconds=30.0,
            require_consensus=True,
            coordinator_agent="lead-agent",
            termination_condition="consensus",
        )

        assert config.max_turns == 5
        assert config.turn_timeout_seconds == 30.0
        assert config.require_consensus is True
        assert config.coordinator_agent == "lead-agent"
        assert config.termination_condition == "consensus"

    def test_invalid_max_turns(self):
        """Test that max_turns must be >= 1."""
        with pytest.raises(ValidationError):
            CollaborationConfig(max_turns=0)

    def test_invalid_timeout(self):
        """Test that turn_timeout_seconds must be > 0."""
        with pytest.raises(ValidationError):
            CollaborationConfig(turn_timeout_seconds=0)


class TestHandoffConfig:
    """Test HandoffConfig model."""

    def test_default_values(self):
        """Test default handoff config values."""
        config = HandoffConfig()

        assert config.handoff_trigger == HandoffTrigger.ALWAYS
        assert config.context_accumulation == ContextAccumulationMode.APPEND
        assert config.max_context_size is None
        assert config.allow_backtrack is False
        assert config.final_agent_decides is True
        assert config.agent_timeout_seconds == 120.0
        assert config.continue_on_failure is False

    def test_custom_values(self):
        """Test custom handoff config values."""
        config = HandoffConfig(
            handoff_trigger=HandoffTrigger.ON_SUCCESS,
            context_accumulation=ContextAccumulationMode.MERGE,
            max_context_size=10000,
            allow_backtrack=True,
            agent_timeout_seconds=60.0,
        )

        assert config.handoff_trigger == HandoffTrigger.ON_SUCCESS
        assert config.context_accumulation == ContextAccumulationMode.MERGE
        assert config.max_context_size == 10000
        assert config.allow_backtrack is True

    def test_handoff_triggers(self):
        """Test all handoff trigger values."""
        assert HandoffTrigger.ALWAYS.value == "always"
        assert HandoffTrigger.ON_SUCCESS.value == "on_success"
        assert HandoffTrigger.ON_FAILURE.value == "on_failure"
        assert HandoffTrigger.ON_PARTIAL.value == "on_partial"
        assert HandoffTrigger.EXPLICIT.value == "explicit"

    def test_context_accumulation_modes(self):
        """Test all context accumulation modes."""
        assert ContextAccumulationMode.APPEND.value == "append"
        assert ContextAccumulationMode.REPLACE.value == "replace"
        assert ContextAccumulationMode.MERGE.value == "merge"
        assert ContextAccumulationMode.SUMMARY.value == "summary"


class TestComparisonConfig:
    """Test ComparisonConfig model."""

    def test_default_values(self):
        """Test default comparison config values."""
        config = ComparisonConfig()

        assert config.metrics == ["quality", "speed", "cost"]
        assert config.determine_winner is True
        assert config.parallel_execution is True

    def test_custom_values(self):
        """Test custom comparison config values."""
        config = ComparisonConfig(
            metrics=["quality", "tokens"],
            determine_winner=False,
            parallel_execution=False,
        )

        assert config.metrics == ["quality", "tokens"]
        assert config.determine_winner is False
        assert config.parallel_execution is False


class TestMultiAgentTestDefinition:
    """Test multi-agent fields in TestDefinition."""

    def test_single_agent_no_mode(self):
        """Test that single agent doesn't require mode."""
        test = TestDefinition(
            id="test-001",
            name="Single agent test",
            task=TaskDefinition(description="Task"),
            agents=["agent-1"],
        )

        assert test.agents == ["agent-1"]
        assert test.mode is None
        assert test.is_multi_agent is False

    def test_comparison_mode(self):
        """Test test with comparison mode."""
        test = TestDefinition(
            id="test-001",
            name="Comparison test",
            task=TaskDefinition(description="Task"),
            agents=["agent-1", "agent-2"],
            mode=MultiAgentMode.COMPARISON,
        )

        assert test.mode == MultiAgentMode.COMPARISON
        assert test.is_multi_agent is True

    def test_collaboration_mode(self):
        """Test test with collaboration mode."""
        test = TestDefinition(
            id="test-001",
            name="Collaboration test",
            task=TaskDefinition(description="Task"),
            agents=["agent-1", "agent-2"],
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=CollaborationConfig(max_turns=5),
        )

        assert test.mode == MultiAgentMode.COLLABORATION
        assert test.collaboration_config is not None
        assert test.collaboration_config.max_turns == 5
        assert test.is_multi_agent is True

    def test_handoff_mode(self):
        """Test test with handoff mode."""
        test = TestDefinition(
            id="test-001",
            name="Handoff test",
            task=TaskDefinition(description="Task"),
            agents=["agent-1", "agent-2", "agent-3"],
            mode=MultiAgentMode.HANDOFF,
            handoff_config=HandoffConfig(handoff_trigger=HandoffTrigger.ON_SUCCESS),
        )

        assert test.mode == MultiAgentMode.HANDOFF
        assert test.handoff_config is not None
        assert test.handoff_config.handoff_trigger == HandoffTrigger.ON_SUCCESS
        assert test.is_multi_agent is True

    def test_mode_without_agents_raises_error(self):
        """Test that mode without agents raises validation error."""
        with pytest.raises(ValidationError, match="agents"):
            TestDefinition(
                id="test-001",
                name="Invalid test",
                task=TaskDefinition(description="Task"),
                mode=MultiAgentMode.COMPARISON,
            )

    def test_multiple_agents_without_mode_raises_error(self):
        """Test that multiple agents without mode raises validation error."""
        with pytest.raises(ValidationError, match="mode"):
            TestDefinition(
                id="test-001",
                name="Invalid test",
                task=TaskDefinition(description="Task"),
                agents=["agent-1", "agent-2"],
            )

    def test_comparison_mode_with_collaboration_config_raises_error(self):
        """Test that comparison mode with collaboration config raises error."""
        with pytest.raises(ValidationError, match="collaboration_config"):
            TestDefinition(
                id="test-001",
                name="Invalid test",
                task=TaskDefinition(description="Task"),
                agents=["agent-1", "agent-2"],
                mode=MultiAgentMode.COMPARISON,
                collaboration_config=CollaborationConfig(),
            )

    def test_comparison_mode_with_handoff_config_raises_error(self):
        """Test that comparison mode with handoff config raises error."""
        with pytest.raises(ValidationError, match="handoff_config"):
            TestDefinition(
                id="test-001",
                name="Invalid test",
                task=TaskDefinition(description="Task"),
                agents=["agent-1", "agent-2"],
                mode=MultiAgentMode.COMPARISON,
                handoff_config=HandoffConfig(),
            )

    def test_collaboration_mode_with_comparison_config_raises_error(self):
        """Test that collaboration mode with comparison config raises error."""
        with pytest.raises(ValidationError, match="comparison_config"):
            TestDefinition(
                id="test-001",
                name="Invalid test",
                task=TaskDefinition(description="Task"),
                agents=["agent-1", "agent-2"],
                mode=MultiAgentMode.COLLABORATION,
                comparison_config=ComparisonConfig(),
            )

    def test_handoff_mode_with_comparison_config_raises_error(self):
        """Test that handoff mode with comparison config raises error."""
        with pytest.raises(ValidationError, match="comparison_config"):
            TestDefinition(
                id="test-001",
                name="Invalid test",
                task=TaskDefinition(description="Task"),
                agents=["agent-1", "agent-2"],
                mode=MultiAgentMode.HANDOFF,
                comparison_config=ComparisonConfig(),
            )

    def test_coordinator_must_be_in_agents(self):
        """Test that coordinator_agent must be in agents list."""
        with pytest.raises(ValidationError, match="coordinator_agent"):
            TestDefinition(
                id="test-001",
                name="Invalid test",
                task=TaskDefinition(description="Task"),
                agents=["agent-1", "agent-2"],
                mode=MultiAgentMode.COLLABORATION,
                collaboration_config=CollaborationConfig(
                    coordinator_agent="unknown-agent"
                ),
            )

    def test_coordinator_in_agents_list_succeeds(self):
        """Test that valid coordinator_agent succeeds."""
        test = TestDefinition(
            id="test-001",
            name="Valid test",
            task=TaskDefinition(description="Task"),
            agents=["agent-1", "agent-2", "coordinator"],
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=CollaborationConfig(coordinator_agent="coordinator"),
        )

        assert test.collaboration_config.coordinator_agent == "coordinator"

    def test_is_multi_agent_property(self):
        """Test is_multi_agent property."""
        # Single agent without mode
        single = TestDefinition(
            id="test-001",
            name="Single",
            task=TaskDefinition(description="Task"),
        )
        assert single.is_multi_agent is False

        # With mode set
        with_mode = TestDefinition(
            id="test-002",
            name="With mode",
            task=TaskDefinition(description="Task"),
            agents=["agent-1", "agent-2"],
            mode=MultiAgentMode.COMPARISON,
        )
        assert with_mode.is_multi_agent is True
