"""Data models for test definitions and suites."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from atp.chaos import ChaosConfig, ChaosProfile


class MultiAgentMode(str, Enum):
    """Mode for multi-agent test execution."""

    COMPARISON = "comparison"
    COLLABORATION = "collaboration"
    HANDOFF = "handoff"


class CollaborationConfig(BaseModel):
    """Configuration for collaboration mode tests."""

    max_turns: int = Field(
        default=10, description="Maximum number of collaboration turns", ge=1
    )
    turn_timeout_seconds: float = Field(
        default=60.0, description="Timeout per turn in seconds", gt=0
    )
    require_consensus: bool = Field(
        default=False, description="Require all agents to agree on final result"
    )
    allow_parallel_turns: bool = Field(
        default=False, description="Allow agents to execute in parallel within a turn"
    )
    coordinator_agent: str | None = Field(
        None, description="Name of agent to act as coordinator (None for round-robin)"
    )
    termination_condition: str = Field(
        default="all_complete",
        description="Condition to end collaboration: 'all_complete', "
        "'any_complete', 'consensus', 'max_turns'",
    )


class HandoffTrigger(str, Enum):
    """Triggers for when to perform a handoff to the next agent."""

    ALWAYS = "always"
    ON_SUCCESS = "on_success"
    ON_FAILURE = "on_failure"
    ON_PARTIAL = "on_partial"
    EXPLICIT = "explicit"


class ContextAccumulationMode(str, Enum):
    """How context is accumulated across handoffs."""

    APPEND = "append"
    REPLACE = "replace"
    MERGE = "merge"
    SUMMARY = "summary"


class HandoffConfig(BaseModel):
    """Configuration for handoff mode tests."""

    handoff_trigger: HandoffTrigger = Field(
        default=HandoffTrigger.ALWAYS,
        description="When to trigger handoff to next agent",
    )
    context_accumulation: ContextAccumulationMode = Field(
        default=ContextAccumulationMode.APPEND,
        description="How to accumulate context across handoffs",
    )
    max_context_size: int | None = Field(
        None, description="Maximum context size in characters (None for unlimited)"
    )
    allow_backtrack: bool = Field(
        default=False, description="Allow previous agents to be re-invoked on failure"
    )
    final_agent_decides: bool = Field(
        default=True, description="Whether the final agent makes the overall decision"
    )
    agent_timeout_seconds: float = Field(
        default=120.0, description="Timeout per agent in seconds", gt=0
    )
    continue_on_failure: bool = Field(
        default=False, description="Continue to next agent even if current fails"
    )


class ComparisonConfig(BaseModel):
    """Configuration for comparison mode tests."""

    metrics: list[str] = Field(
        default_factory=lambda: ["quality", "speed", "cost"],
        description="Metrics to compare agents on",
    )
    determine_winner: bool = Field(
        default=True, description="Whether to determine a winner"
    )
    parallel_execution: bool = Field(
        default=True, description="Execute agents in parallel"
    )


class TaskDefinition(BaseModel):
    """Task specification for an agent."""

    description: str = Field(
        ..., description="Task description for the agent", min_length=1
    )
    input_data: dict[str, Any] | None = Field(None, description="Optional input data")
    expected_artifacts: list[str] | None = Field(
        None, description="Expected output artifacts"
    )


class Constraints(BaseModel):
    """Execution constraints for a test."""

    max_steps: int | None = Field(None, description="Maximum number of steps allowed")
    max_tokens: int | None = Field(None, description="Maximum tokens allowed")
    timeout_seconds: int | None = Field(None, description="Timeout in seconds")
    allowed_tools: list[str] | None = Field(
        None, description="List of allowed tools, None means all allowed"
    )
    budget_usd: float | None = Field(None, description="Budget limit in USD")

    @property
    def effective_timeout_seconds(self) -> int:
        """Get the effective timeout, defaulting to 300 if not set."""
        return self.timeout_seconds if self.timeout_seconds is not None else 300


class Assertion(BaseModel):
    """Single assertion for test evaluation."""

    type: str = Field(..., description="Assertion type (e.g., artifact_exists)")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Assertion configuration"
    )


class ScoringWeights(BaseModel):
    """Scoring weights for test evaluation."""

    quality_weight: float = Field(0.4, ge=0.0, le=1.0)
    completeness_weight: float = Field(0.3, ge=0.0, le=1.0)
    efficiency_weight: float = Field(0.2, ge=0.0, le=1.0)
    cost_weight: float = Field(0.1, ge=0.0, le=1.0)


class TestDefinition(BaseModel):
    """Complete test definition."""

    id: str = Field(..., description="Unique test identifier")
    name: str = Field(..., description="Human-readable test name")
    description: str | None = Field(None, description="Optional test description")
    tags: list[str] = Field(default_factory=list, description="Test tags")
    task: TaskDefinition = Field(..., description="Task specification")
    constraints: Constraints = Field(
        default_factory=Constraints, description="Execution constraints"
    )
    assertions: list[Assertion] = Field(
        default_factory=list, description="Test assertions"
    )
    scoring: ScoringWeights | None = Field(
        None, description="Optional scoring weights override"
    )

    # Multi-agent fields
    agents: list[str] | None = Field(
        None,
        description="List of agent names to run this test against "
        "(overrides suite-level agents)",
    )
    mode: MultiAgentMode | None = Field(
        None,
        description="Multi-agent execution mode: comparison, collaboration, or handoff",
    )
    comparison_config: ComparisonConfig | None = Field(
        None, description="Configuration for comparison mode"
    )
    collaboration_config: CollaborationConfig | None = Field(
        None, description="Configuration for collaboration mode"
    )
    handoff_config: HandoffConfig | None = Field(
        None, description="Configuration for handoff mode"
    )
    chaos: "ChaosSettings | None" = Field(
        None, description="Per-test chaos engineering settings (overrides suite-level)"
    )

    @model_validator(mode="after")
    def validate_multi_agent_config(self) -> "TestDefinition":
        """Validate multi-agent configuration consistency."""
        # If mode is set, agents should be specified
        if self.mode is not None and self.agents is None:
            raise ValueError("When 'mode' is specified, 'agents' must also be provided")

        # If agents is set with more than one agent, mode should be specified
        if self.agents is not None and len(self.agents) > 1 and self.mode is None:
            raise ValueError(
                "When multiple agents are specified, 'mode' is required "
                "(comparison, collaboration, or handoff)"
            )

        # Validate mode-specific config alignment
        if self.mode == MultiAgentMode.COMPARISON:
            # Comparison config is optional, uses defaults
            if self.collaboration_config is not None:
                raise ValueError(
                    "collaboration_config should not be set for comparison mode"
                )
            if self.handoff_config is not None:
                raise ValueError("handoff_config should not be set for comparison mode")

        elif self.mode == MultiAgentMode.COLLABORATION:
            # Collaboration config is optional, uses defaults
            if self.comparison_config is not None:
                raise ValueError(
                    "comparison_config should not be set for collaboration mode"
                )
            if self.handoff_config is not None:
                raise ValueError(
                    "handoff_config should not be set for collaboration mode"
                )

        elif self.mode == MultiAgentMode.HANDOFF:
            # Handoff config is optional, uses defaults
            if self.comparison_config is not None:
                raise ValueError("comparison_config should not be set for handoff mode")
            if self.collaboration_config is not None:
                raise ValueError(
                    "collaboration_config should not be set for handoff mode"
                )

        # Validate coordinator_agent exists in agents list
        if (
            self.collaboration_config is not None
            and self.collaboration_config.coordinator_agent is not None
            and self.agents is not None
            and self.collaboration_config.coordinator_agent not in self.agents
        ):
            raise ValueError(
                f"coordinator_agent '{self.collaboration_config.coordinator_agent}' "
                "must be in the agents list"
            )

        return self

    @property
    def is_multi_agent(self) -> bool:
        """Check if this is a multi-agent test."""
        return self.mode is not None or (
            self.agents is not None and len(self.agents) > 1
        )


class AgentConfig(BaseModel):
    """Agent configuration."""

    name: str = Field(..., description="Agent name")
    type: str | None = Field(None, description="Agent type (http, container, etc.)")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific configuration"
    )


class ChaosSettings(BaseModel):
    """Chaos engineering settings for test suite."""

    profile: ChaosProfile | str | None = Field(
        None, description="Predefined chaos profile name"
    )
    custom: ChaosConfig | None = Field(
        None, description="Custom chaos configuration (overrides profile)"
    )

    def get_config(self) -> ChaosConfig | None:
        """Get the effective chaos configuration.

        Returns:
            ChaosConfig from custom settings or profile, or None if disabled.
        """
        if self.custom is not None:
            return self.custom
        if self.profile is not None:
            from atp.chaos import get_profile

            return get_profile(self.profile)
        return None


class TestDefaults(BaseModel):
    """Default settings for all tests in a suite."""

    runs_per_test: int = Field(1, ge=1, description="Number of runs per test")
    timeout_seconds: int = Field(300, ge=1, description="Default timeout")
    scoring: ScoringWeights = Field(
        default_factory=ScoringWeights, description="Default scoring weights"
    )
    constraints: Constraints | None = Field(
        None, description="Default constraints for all tests"
    )
    chaos: ChaosSettings | None = Field(
        None, description="Default chaos engineering settings"
    )


class TestSuite(BaseModel):
    """Complete test suite with defaults and tests."""

    test_suite: str = Field(..., description="Suite name")
    version: str = Field("1.0", description="Suite version")
    description: str | None = Field(None, description="Optional suite description")
    defaults: TestDefaults = Field(
        default_factory=TestDefaults, description="Default settings"
    )
    agents: list[AgentConfig] = Field(
        default_factory=list, description="Agent configurations"
    )
    tests: list[TestDefinition] = Field(..., description="List of tests", min_length=1)
    chaos: ChaosSettings | None = Field(
        None, description="Suite-level chaos engineering settings"
    )

    def apply_defaults(self) -> None:
        """Apply defaults to all tests that don't have explicit values."""
        for test in self.tests:
            # Apply default constraints
            if self.defaults.constraints:
                if test.constraints.max_steps is None:
                    test.constraints.max_steps = self.defaults.constraints.max_steps
                if test.constraints.max_tokens is None:
                    test.constraints.max_tokens = self.defaults.constraints.max_tokens
                # Only apply default timeout if not explicitly set (None means not set)
                if test.constraints.timeout_seconds is None:
                    test.constraints.timeout_seconds = (
                        self.defaults.constraints.timeout_seconds
                    )
                if test.constraints.allowed_tools is None:
                    test.constraints.allowed_tools = (
                        self.defaults.constraints.allowed_tools
                    )
                if test.constraints.budget_usd is None:
                    test.constraints.budget_usd = self.defaults.constraints.budget_usd

            # Apply default scoring (deep copy to avoid sharing mutable object)
            if test.scoring is None:
                test.scoring = self.defaults.scoring.model_copy(deep=True)

    def filter_by_tags(self, tag_filter: str | None) -> "TestSuite":
        """Filter tests by tag expressions.

        Args:
            tag_filter: Comma-separated tag expressions (e.g., "smoke,!slow")
                       None means no filtering (return all tests)

        Returns:
            New TestSuite with filtered tests

        Example:
            >>> suite.filter_by_tags("smoke,!slow")
            # Returns suite with tests tagged 'smoke' but not 'slow'
        """
        if not tag_filter:
            return self

        from atp.loader.filters import TagFilter

        filter_obj = TagFilter.from_string(tag_filter)
        filtered_tests = [test for test in self.tests if filter_obj.matches(test.tags)]

        # Create new suite with filtered tests
        return self.model_copy(update={"tests": filtered_tests})

    def get_chaos_config(
        self, test: TestDefinition | None = None
    ) -> ChaosConfig | None:
        """Get the effective chaos configuration for a test.

        Args:
            test: Optional test definition to check for per-test overrides.

        Returns:
            ChaosConfig from test, suite, or defaults (in that order),
            or None if no chaos configuration is set.
        """
        # Per-test chaos settings take precedence
        if test is not None and test.chaos is not None:
            return test.chaos.get_config()

        # Suite-level chaos settings
        if self.chaos is not None:
            return self.chaos.get_config()

        # Default chaos settings
        if self.defaults.chaos is not None:
            return self.defaults.chaos.get_config()

        return None
