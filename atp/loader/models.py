"""Data models for test definitions and suites."""

from typing import Any

from pydantic import BaseModel, Field


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
    timeout_seconds: int = Field(300, description="Timeout in seconds")
    allowed_tools: list[str] | None = Field(
        None, description="List of allowed tools, None means all allowed"
    )
    budget_usd: float | None = Field(None, description="Budget limit in USD")


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


class AgentConfig(BaseModel):
    """Agent configuration."""

    name: str = Field(..., description="Agent name")
    type: str | None = Field(None, description="Agent type (http, container, etc.)")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific configuration"
    )


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

    def apply_defaults(self) -> None:
        """Apply defaults to all tests that don't have explicit values."""
        for test in self.tests:
            # Apply default constraints
            if self.defaults.constraints:
                if test.constraints.max_steps is None:
                    test.constraints.max_steps = self.defaults.constraints.max_steps
                if test.constraints.max_tokens is None:
                    test.constraints.max_tokens = self.defaults.constraints.max_tokens
                if test.constraints.timeout_seconds == 300:  # default value
                    test.constraints.timeout_seconds = (
                        self.defaults.constraints.timeout_seconds
                    )
                if test.constraints.allowed_tools is None:
                    test.constraints.allowed_tools = (
                        self.defaults.constraints.allowed_tools
                    )
                if test.constraints.budget_usd is None:
                    test.constraints.budget_usd = self.defaults.constraints.budget_usd

            # Apply default scoring weights if not specified
            if test.scoring is None:
                test.scoring = self.defaults.scoring
