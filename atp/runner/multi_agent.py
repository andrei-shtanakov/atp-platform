"""Multi-agent orchestrator for running tests across multiple agents."""

import asyncio
import copy
import logging
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any

from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from atp.adapters.base import AgentAdapter
from atp.core.metrics import get_metrics
from atp.core.telemetry import (
    add_span_event,
    get_tracer,
    set_span_attributes,
)
from atp.loader.models import TestDefinition, TestSuite
from atp.protocol import ATPResponse, ResponseStatus
from atp.runner.models import (
    ProgressCallback,
    ProgressEvent,
    ProgressEventType,
    SandboxConfig,
    SuiteResult,
    TestResult,
)
from atp.runner.orchestrator import TestOrchestrator

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class MultiAgentMode(str, Enum):
    """Mode for multi-agent test execution."""

    COMPARISON = "comparison"
    COLLABORATION = "collaboration"
    HANDOFF = "handoff"


class RankingMetric(str, Enum):
    """Metrics available for ranking agents."""

    QUALITY = "quality"
    SPEED = "speed"
    COST = "cost"
    SUCCESS_RATE = "success_rate"
    TOKENS = "tokens"
    STEPS = "steps"


class AgentConfig(BaseModel):
    """Configuration for a single agent in multi-agent execution."""

    name: str = Field(..., description="Agent name identifier")
    adapter: AgentAdapter = Field(..., description="Adapter instance for the agent")
    weight: float = Field(
        default=1.0, description="Weight for scoring (used in collaboration mode)"
    )

    model_config = {"arbitrary_types_allowed": True}


class AgentRanking(BaseModel):
    """Ranking result for a single agent."""

    agent_name: str = Field(..., description="Agent name")
    rank: int = Field(..., description="Rank (1-based, lower is better)")
    score: float = Field(..., description="Score for the ranking metric")
    metric: RankingMetric = Field(..., description="Metric used for ranking")


class ComparisonMetrics(BaseModel):
    """Aggregated metrics for comparison between agents."""

    agent_name: str = Field(..., description="Agent name")
    success_rate: float = Field(default=0.0, description="Success rate (0.0-1.0)")
    avg_duration_seconds: float | None = Field(
        None, description="Average duration in seconds"
    )
    avg_tokens: float | None = Field(None, description="Average tokens used")
    avg_steps: float | None = Field(None, description="Average steps taken")
    avg_cost_usd: float | None = Field(None, description="Average cost in USD")
    total_tests: int = Field(default=0, description="Total tests run")
    passed_tests: int = Field(default=0, description="Tests passed")
    failed_tests: int = Field(default=0, description="Tests failed")


class CollaborationMessageType(str, Enum):
    """Types of messages exchanged between agents in collaboration mode."""

    TASK_ASSIGNMENT = "task_assignment"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    HANDOFF = "handoff"
    STATUS_UPDATE = "status_update"


class CollaborationMessage(BaseModel):
    """Message exchanged between agents during collaboration."""

    message_id: str = Field(..., description="Unique message identifier")
    message_type: CollaborationMessageType = Field(..., description="Type of message")
    from_agent: str = Field(..., description="Sender agent name")
    to_agent: str | None = Field(
        None, description="Target agent name (None for broadcast)"
    )
    content: dict[str, Any] = Field(default_factory=dict, description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Message timestamp"
    )
    turn_number: int = Field(default=0, description="Turn number when message was sent")
    in_reply_to: str | None = Field(
        None, description="ID of message this is replying to"
    )


class SharedContext(BaseModel):
    """Shared context accessible by all agents in collaboration mode."""

    task_description: str = Field(..., description="Original task description")
    artifacts: dict[str, Any] = Field(
        default_factory=dict, description="Shared artifacts produced by agents"
    )
    variables: dict[str, Any] = Field(
        default_factory=dict, description="Shared variables for state management"
    )
    messages: list[CollaborationMessage] = Field(
        default_factory=list, description="Message history"
    )
    current_turn: int = Field(default=0, description="Current turn number")
    completed_subtasks: list[str] = Field(
        default_factory=list, description="IDs of completed subtasks"
    )

    def add_message(self, message: CollaborationMessage) -> None:
        """Add a message to the shared context."""
        self.messages.append(message)

    def get_messages_for_agent(self, agent_name: str) -> list[CollaborationMessage]:
        """Get all messages addressed to or from a specific agent."""
        return [
            m
            for m in self.messages
            if m.to_agent == agent_name
            or m.from_agent == agent_name
            or m.to_agent is None
        ]

    def get_messages_by_turn(self, turn: int) -> list[CollaborationMessage]:
        """Get all messages from a specific turn."""
        return [m for m in self.messages if m.turn_number == turn]

    def set_artifact(self, key: str, value: Any, agent_name: str) -> None:
        """Set a shared artifact with provenance tracking."""
        self.artifacts[key] = {
            "value": value,
            "set_by": agent_name,
            "turn": self.current_turn,
            "timestamp": datetime.now().isoformat(),
        }

    def get_artifact(self, key: str) -> Any | None:
        """Get a shared artifact value."""
        artifact = self.artifacts.get(key)
        if artifact:
            return artifact.get("value")
        return None

    def set_variable(self, key: str, value: Any) -> None:
        """Set a shared variable."""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a shared variable."""
        return self.variables.get(key, default)


class CollaborationConfig(BaseModel):
    """Configuration for collaboration mode."""

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


class CollaborationMetrics(BaseModel):
    """Metrics specific to collaboration mode."""

    total_turns: int = Field(default=0, description="Total number of turns taken")
    total_messages: int = Field(default=0, description="Total messages exchanged")
    messages_per_agent: dict[str, int] = Field(
        default_factory=dict, description="Message count per agent"
    )
    turns_per_agent: dict[str, int] = Field(
        default_factory=dict, description="Active turns per agent"
    )
    avg_turn_duration_seconds: float | None = Field(
        None, description="Average turn duration"
    )
    consensus_reached: bool = Field(
        default=False, description="Whether consensus was reached"
    )
    termination_reason: str | None = Field(None, description="Why collaboration ended")
    agent_contributions: dict[str, float] = Field(
        default_factory=dict,
        description="Contribution score per agent (0.0-1.0)",
    )
    total_tokens: int = Field(default=0, description="Total tokens used by all agents")
    total_cost_usd: float = Field(
        default=0.0, description="Total cost across all agents"
    )


# ===========================================================================
# Handoff Mode Models
# ===========================================================================


class HandoffTrigger(str, Enum):
    """Triggers for when to perform a handoff to the next agent."""

    ALWAYS = "always"  # Always handoff after each agent
    ON_SUCCESS = "on_success"  # Handoff only if current agent succeeded
    ON_FAILURE = "on_failure"  # Handoff only if current agent failed
    ON_PARTIAL = "on_partial"  # Handoff if partial completion
    EXPLICIT = "explicit"  # Agent must explicitly request handoff


class ContextAccumulationMode(str, Enum):
    """How context is accumulated across handoffs."""

    APPEND = "append"  # Append all previous outputs
    REPLACE = "replace"  # Only pass the most recent output
    MERGE = "merge"  # Merge artifacts, keeping latest values
    SUMMARY = "summary"  # Pass summarized context (for large outputs)


class HandoffConfig(BaseModel):
    """Configuration for handoff mode."""

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


class AgentHandoffOutput(BaseModel):
    """Output from a single agent in the handoff chain."""

    agent_name: str = Field(..., description="Name of the agent")
    sequence_position: int = Field(..., description="Position in handoff sequence")
    response: ATPResponse | None = Field(None, description="Agent's response")
    artifacts: list[Any] = Field(
        default_factory=list, description="Artifacts produced by this agent"
    )
    error: str | None = Field(None, description="Error message if agent failed")
    duration_seconds: float | None = Field(None, description="Time taken by this agent")


class HandoffContext(BaseModel):
    """Context passed between agents during handoff."""

    original_task: str = Field(..., description="Original task description")
    previous_outputs: list[AgentHandoffOutput] = Field(
        default_factory=list, description="Outputs from previous agents in sequence"
    )
    agent_sequence: list[str] = Field(
        default_factory=list, description="Order of agents in the handoff chain"
    )
    current_position: int = Field(default=0, description="Current position in sequence")
    accumulated_artifacts: dict[str, Any] = Field(
        default_factory=dict, description="Merged/accumulated artifacts"
    )
    handoff_notes: list[str] = Field(
        default_factory=list, description="Notes passed between agents"
    )

    def add_output(self, output: AgentHandoffOutput) -> None:
        """Add an output from an agent."""
        self.previous_outputs.append(output)
        # Update accumulated artifacts
        for idx, artifact in enumerate(output.artifacts):
            # Try to get name from attribute or dict key
            if hasattr(artifact, "name"):
                key = artifact.name
            elif isinstance(artifact, dict) and "name" in artifact:
                key = artifact["name"]
            else:
                key = f"artifact_{output.sequence_position}_{idx}"
            self.accumulated_artifacts[key] = artifact

    def get_last_output(self) -> AgentHandoffOutput | None:
        """Get the most recent agent output."""
        if self.previous_outputs:
            return self.previous_outputs[-1]
        return None

    def get_successful_outputs(self) -> list[AgentHandoffOutput]:
        """Get all successful agent outputs."""
        return [o for o in self.previous_outputs if o.error is None]

    def add_handoff_note(self, note: str) -> None:
        """Add a handoff note."""
        self.handoff_notes.append(note)


class HandoffMetrics(BaseModel):
    """Metrics specific to handoff mode."""

    total_agents: int = Field(default=0, description="Total agents in handoff chain")
    agents_executed: int = Field(
        default=0, description="Number of agents that executed"
    )
    successful_handoffs: int = Field(
        default=0, description="Number of successful handoffs"
    )
    failed_handoffs: int = Field(default=0, description="Number of failed handoffs")
    total_duration_seconds: float = Field(
        default=0.0, description="Total duration across all agents"
    )
    duration_per_agent: dict[str, float] = Field(
        default_factory=dict, description="Duration per agent"
    )
    agent_contributions: dict[str, float] = Field(
        default_factory=dict, description="Contribution score per agent (0.0-1.0)"
    )
    total_tokens: int = Field(default=0, description="Total tokens used by all agents")
    tokens_per_agent: dict[str, int] = Field(
        default_factory=dict, description="Tokens used per agent"
    )
    total_cost_usd: float = Field(
        default=0.0, description="Total cost across all agents"
    )
    cost_per_agent: dict[str, float] = Field(
        default_factory=dict, description="Cost per agent"
    )
    termination_reason: str | None = Field(None, description="Why handoff chain ended")
    final_agent: str | None = Field(
        None, description="Name of the agent that produced final output"
    )


@dataclass
class AgentTestResult:
    """Result of running a test with a specific agent."""

    agent_name: str
    test_result: TestResult
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    @property
    def success(self) -> bool:
        """Check if the test succeeded."""
        return self.test_result.success

    @property
    def duration_seconds(self) -> float | None:
        """Calculate duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class CollaborationTurnResult:
    """Result of a single turn in collaboration mode."""

    turn_number: int
    agent_name: str
    response: ATPResponse | None = None
    messages_sent: list[CollaborationMessage] = field(default_factory=list)
    messages_received: list[CollaborationMessage] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if turn completed successfully."""
        if self.error:
            return False
        if self.response:
            return self.response.status == ResponseStatus.COMPLETED
        return True

    @property
    def duration_seconds(self) -> float | None:
        """Calculate turn duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class CollaborationResult:
    """Result of a collaboration session."""

    test: TestDefinition
    shared_context: SharedContext
    turn_results: list[CollaborationTurnResult] = field(default_factory=list)
    collaboration_metrics: CollaborationMetrics = field(
        default_factory=CollaborationMetrics
    )
    final_response: ATPResponse | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if collaboration completed successfully."""
        if self.error:
            return False
        if self.final_response:
            return self.final_response.status == ResponseStatus.COMPLETED
        return all(tr.success for tr in self.turn_results)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total collaboration duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def total_turns(self) -> int:
        """Get total number of turns."""
        return len(self.turn_results)

    def get_turns_for_agent(self, agent_name: str) -> list[CollaborationTurnResult]:
        """Get all turns for a specific agent."""
        return [tr for tr in self.turn_results if tr.agent_name == agent_name]


@dataclass
class HandoffTurnResult:
    """Result of a single agent's execution in handoff mode."""

    agent_name: str
    sequence_position: int
    response: ATPResponse | None = None
    context_received: HandoffContext | None = None
    context_passed: HandoffContext | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    error: str | None = None
    handoff_triggered: bool = False

    @property
    def success(self) -> bool:
        """Check if agent execution succeeded."""
        if self.error:
            return False
        if self.response:
            return self.response.status == ResponseStatus.COMPLETED
        return False

    @property
    def duration_seconds(self) -> float | None:
        """Calculate execution duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class HandoffResult:
    """Result of a complete handoff chain execution."""

    test: TestDefinition
    handoff_context: HandoffContext
    turn_results: list[HandoffTurnResult] = field(default_factory=list)
    handoff_metrics: HandoffMetrics = field(default_factory=HandoffMetrics)
    final_response: ATPResponse | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if handoff chain completed successfully."""
        if self.error:
            return False
        if self.final_response:
            return self.final_response.status == ResponseStatus.COMPLETED
        return any(tr.success for tr in self.turn_results)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total handoff duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def total_agents_executed(self) -> int:
        """Get total number of agents that executed."""
        return len(self.turn_results)

    def get_result_for_agent(self, agent_name: str) -> HandoffTurnResult | None:
        """Get the result for a specific agent."""
        for result in self.turn_results:
            if result.agent_name == agent_name:
                return result
        return None

    def get_successful_agents(self) -> list[str]:
        """Get names of agents that succeeded."""
        return [tr.agent_name for tr in self.turn_results if tr.success]

    def get_failed_agents(self) -> list[str]:
        """Get names of agents that failed."""
        return [tr.agent_name for tr in self.turn_results if not tr.success]


@dataclass
class MultiAgentTestResult:
    """Result of running a single test across multiple agents."""

    test: TestDefinition
    agent_results: list[AgentTestResult] = field(default_factory=list)
    mode: MultiAgentMode = MultiAgentMode.COMPARISON
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    winner: str | None = None
    rankings: list[AgentRanking] = field(default_factory=list)
    # Collaboration-specific fields
    collaboration_result: CollaborationResult | None = None
    # Handoff-specific fields
    handoff_result: HandoffResult | None = None

    @property
    def all_succeeded(self) -> bool:
        """Check if all agents succeeded."""
        if self.mode == MultiAgentMode.COLLABORATION:
            return (
                self.collaboration_result is not None
                and self.collaboration_result.success
            )
        if self.mode == MultiAgentMode.HANDOFF:
            return self.handoff_result is not None and self.handoff_result.success
        if not self.agent_results:
            return False
        return all(r.success for r in self.agent_results)

    @property
    def any_succeeded(self) -> bool:
        """Check if any agent succeeded."""
        if self.mode == MultiAgentMode.COLLABORATION:
            return (
                self.collaboration_result is not None
                and self.collaboration_result.success
            )
        if self.mode == MultiAgentMode.HANDOFF:
            return self.handoff_result is not None and self.handoff_result.success
        return any(r.success for r in self.agent_results)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def agent_names(self) -> list[str]:
        """Get list of agent names."""
        if self.mode == MultiAgentMode.HANDOFF and self.handoff_result:
            return [tr.agent_name for tr in self.handoff_result.turn_results]
        return [r.agent_name for r in self.agent_results]

    def get_result_for_agent(self, agent_name: str) -> AgentTestResult | None:
        """Get result for a specific agent."""
        for result in self.agent_results:
            if result.agent_name == agent_name:
                return result
        return None


@dataclass
class MultiAgentSuiteResult:
    """Result of running a test suite across multiple agents."""

    suite_name: str
    mode: MultiAgentMode
    agents: list[str]
    test_results: list[MultiAgentTestResult] = field(default_factory=list)
    agent_suite_results: dict[str, SuiteResult] = field(default_factory=dict)
    comparison_metrics: list[ComparisonMetrics] = field(default_factory=list)
    overall_rankings: list[AgentRanking] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    error: str | None = None

    @property
    def total_tests(self) -> int:
        """Get total number of tests."""
        return len(self.test_results)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate total duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def best_agent(self) -> str | None:
        """Get the best performing agent based on overall rankings."""
        if not self.overall_rankings:
            return None
        return self.overall_rankings[0].agent_name

    def get_agent_suite_result(self, agent_name: str) -> SuiteResult | None:
        """Get suite result for a specific agent."""
        return self.agent_suite_results.get(agent_name)

    def get_comparison_for_agent(self, agent_name: str) -> ComparisonMetrics | None:
        """Get comparison metrics for a specific agent."""
        for metrics in self.comparison_metrics:
            if metrics.agent_name == agent_name:
                return metrics
        return None


class MultiAgentOrchestrator:
    """
    Orchestrates test execution across multiple agents.

    Supports comparison mode where the same test is run against multiple
    agents for performance comparison and ranking.

    Supports collaboration mode where agents work together on a task,
    exchanging messages and sharing context.

    Supports handoff mode where agents execute sequentially, passing
    context and results from one agent to the next.
    """

    def __init__(
        self,
        agents: list[AgentConfig],
        mode: MultiAgentMode = MultiAgentMode.COMPARISON,
        sandbox_config: SandboxConfig | None = None,
        progress_callback: ProgressCallback | None = None,
        runs_per_test: int = 1,
        parallel_agents: bool = True,
        max_parallel_agents: int = 5,
        ranking_metrics: list[RankingMetric] | None = None,
        determine_winner: bool = True,
        collaboration_config: CollaborationConfig | None = None,
        handoff_config: HandoffConfig | None = None,
    ) -> None:
        """
        Initialize the multi-agent orchestrator.

        Args:
            agents: List of agent configurations.
            mode: Multi-agent execution mode (comparison, collaboration, handoff).
            sandbox_config: Sandbox configuration for isolation.
            progress_callback: Optional callback for progress reporting.
            runs_per_test: Number of times to run each test per agent.
            parallel_agents: If True, run agents in parallel.
            max_parallel_agents: Maximum number of parallel agent executions.
            ranking_metrics: Metrics to use for ranking (default: success_rate).
            determine_winner: Whether to determine a winner for each test.
            collaboration_config: Configuration for collaboration mode.
            handoff_config: Configuration for handoff mode.
        """
        if not agents:
            raise ValueError("At least one agent is required")

        self.agents = agents
        self.mode = mode
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.progress_callback = progress_callback
        self.runs_per_test = runs_per_test
        self.parallel_agents = parallel_agents
        self.max_parallel_agents = max_parallel_agents
        self.ranking_metrics = ranking_metrics or [RankingMetric.SUCCESS_RATE]
        self.determine_winner = determine_winner
        self.collaboration_config = collaboration_config or CollaborationConfig()
        self.handoff_config = handoff_config or HandoffConfig()
        self._semaphore: asyncio.Semaphore | None = None
        self._orchestrators: dict[str, TestOrchestrator] = {}
        self._message_counter = 0

    def _emit_progress(self, event: ProgressEvent) -> None:
        """Emit a progress event if callback is registered."""
        if self.progress_callback:
            try:
                self.progress_callback(event)
            except Exception as e:
                logger.warning("Progress callback failed: %s", e)

    def _create_orchestrator(self, agent: AgentConfig) -> TestOrchestrator:
        """Create a test orchestrator for an agent."""
        return TestOrchestrator(
            adapter=agent.adapter,
            sandbox_config=self.sandbox_config,
            progress_callback=self.progress_callback,
            runs_per_test=self.runs_per_test,
        )

    def _calculate_metrics_for_agent(
        self, agent_name: str, suite_result: SuiteResult
    ) -> ComparisonMetrics:
        """Calculate comparison metrics for an agent's suite result."""
        durations: list[float] = []
        tokens: list[int] = []
        steps: list[int] = []
        costs: list[float] = []

        for test_result in suite_result.tests:
            durations.extend(test_result.get_run_durations())
            tokens.extend(test_result.get_run_tokens())
            steps.extend(test_result.get_run_steps())
            costs.extend(test_result.get_run_costs())

        return ComparisonMetrics(
            agent_name=agent_name,
            success_rate=suite_result.success_rate,
            avg_duration_seconds=sum(durations) / len(durations) if durations else None,
            avg_tokens=sum(tokens) / len(tokens) if tokens else None,
            avg_steps=sum(steps) / len(steps) if steps else None,
            avg_cost_usd=sum(costs) / len(costs) if costs else None,
            total_tests=suite_result.total_tests,
            passed_tests=suite_result.passed_tests,
            failed_tests=suite_result.failed_tests,
        )

    def _get_metric_value(
        self, metrics: ComparisonMetrics, metric: RankingMetric
    ) -> float:
        """Get a numeric value for a ranking metric."""
        if metric == RankingMetric.SUCCESS_RATE:
            return metrics.success_rate
        elif metric == RankingMetric.SPEED:
            # Lower is better, so invert
            return (
                -metrics.avg_duration_seconds
                if metrics.avg_duration_seconds is not None
                else float("-inf")
            )
        elif metric == RankingMetric.COST:
            # Lower is better, so invert
            return (
                -metrics.avg_cost_usd
                if metrics.avg_cost_usd is not None
                else float("-inf")
            )
        elif metric == RankingMetric.TOKENS:
            # Lower is better, so invert
            return (
                -metrics.avg_tokens if metrics.avg_tokens is not None else float("-inf")
            )
        elif metric == RankingMetric.STEPS:
            # Lower is better, so invert
            return (
                -metrics.avg_steps if metrics.avg_steps is not None else float("-inf")
            )
        else:
            return metrics.success_rate

    def _rank_agents(
        self,
        comparison_metrics: list[ComparisonMetrics],
        metric: RankingMetric,
    ) -> list[AgentRanking]:
        """Rank agents by a specific metric."""
        # Sort by metric value (higher is better)
        sorted_metrics = sorted(
            comparison_metrics,
            key=lambda m: self._get_metric_value(m, metric),
            reverse=True,
        )

        rankings = []
        for rank, metrics in enumerate(sorted_metrics, start=1):
            score = self._get_metric_value(metrics, metric)
            # Convert back to positive value for display
            if metric in (
                RankingMetric.SPEED,
                RankingMetric.COST,
                RankingMetric.TOKENS,
                RankingMetric.STEPS,
            ):
                score = -score if score != float("-inf") else 0.0
            rankings.append(
                AgentRanking(
                    agent_name=metrics.agent_name,
                    rank=rank,
                    score=score,
                    metric=metric,
                )
            )

        return rankings

    def _determine_test_winner(
        self, agent_results: list[AgentTestResult]
    ) -> tuple[str | None, list[AgentRanking]]:
        """Determine the winner for a single test."""
        if not agent_results:
            return None, []

        # Build metrics for each agent based on this single test
        test_metrics = []
        for result in agent_results:
            tr = result.test_result
            durations = tr.get_run_durations()
            tokens = tr.get_run_tokens()
            steps = tr.get_run_steps()
            costs = tr.get_run_costs()

            test_metrics.append(
                ComparisonMetrics(
                    agent_name=result.agent_name,
                    success_rate=1.0 if tr.success else 0.0,
                    avg_duration_seconds=(
                        sum(durations) / len(durations) if durations else None
                    ),
                    avg_tokens=sum(tokens) / len(tokens) if tokens else None,
                    avg_steps=sum(steps) / len(steps) if steps else None,
                    avg_cost_usd=sum(costs) / len(costs) if costs else None,
                    total_tests=1,
                    passed_tests=1 if tr.success else 0,
                    failed_tests=0 if tr.success else 1,
                )
            )

        # Use the first ranking metric to determine the winner
        primary_metric = (
            self.ranking_metrics[0]
            if self.ranking_metrics
            else RankingMetric.SUCCESS_RATE
        )
        rankings = self._rank_agents(test_metrics, primary_metric)

        winner = rankings[0].agent_name if rankings else None
        return winner, rankings

    async def run_single_test(
        self,
        test: TestDefinition,
        runs: int | None = None,
    ) -> MultiAgentTestResult:
        """
        Run a single test across all agents.

        Args:
            test: Test definition to execute.
            runs: Number of runs (overrides instance default).

        Returns:
            MultiAgentTestResult with results from all agents.
        """
        result = MultiAgentTestResult(
            test=test,
            mode=self.mode,
            start_time=datetime.now(),
        )

        with tracer.start_as_current_span(
            f"multi_agent_test:{test.name}",
            kind=SpanKind.INTERNAL,
            attributes={
                "atp.test.id": test.id,
                "atp.test.name": test.name,
                "atp.multi_agent.mode": self.mode.value,
                "atp.multi_agent.agent_count": len(self.agents),
                "atp.multi_agent.parallel": self.parallel_agents,
            },
        ) as span:
            add_span_event(
                "multi_agent_test_started",
                {"agents": [a.name for a in self.agents]},
            )

            # Emit progress event
            self._emit_progress(
                ProgressEvent(
                    event_type=ProgressEventType.TEST_STARTED,
                    test_id=test.id,
                    test_name=test.name,
                    details={
                        "mode": self.mode.value,
                        "agents": [a.name for a in self.agents],
                    },
                )
            )

            # Handle collaboration mode separately
            if self.mode == MultiAgentMode.COLLABORATION:
                collaboration_result = await self._run_collaboration(test)
                result.collaboration_result = collaboration_result
                result.end_time = datetime.now()

                # Set span attributes for collaboration
                set_span_attributes(
                    **{
                        "atp.multi_agent.all_succeeded": result.all_succeeded,
                        "atp.multi_agent.any_succeeded": result.any_succeeded,
                        "atp.multi_agent.duration_seconds": result.duration_seconds,
                        "atp.collaboration.total_turns": (
                            collaboration_result.total_turns
                        ),
                    }
                )

                if result.all_succeeded:
                    span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(
                        Status(StatusCode.ERROR, "Collaboration did not succeed")
                    )

                add_span_event(
                    "multi_agent_test_completed",
                    {
                        "all_succeeded": result.all_succeeded,
                        "mode": "collaboration",
                        "total_turns": collaboration_result.total_turns,
                    },
                )

                # Emit completion event
                self._emit_progress(
                    ProgressEvent(
                        event_type=(
                            ProgressEventType.TEST_COMPLETED
                            if result.any_succeeded
                            else ProgressEventType.TEST_FAILED
                        ),
                        test_id=test.id,
                        test_name=test.name,
                        success=result.all_succeeded,
                        details={
                            "mode": self.mode.value,
                            "total_turns": collaboration_result.total_turns,
                            "termination_reason": (
                                collaboration_result.collaboration_metrics.termination_reason
                            ),
                        },
                    )
                )

                return result

            # Handle handoff mode separately
            if self.mode == MultiAgentMode.HANDOFF:
                handoff_result = await self._run_handoff(test)
                result.handoff_result = handoff_result
                result.end_time = datetime.now()

                # Set span attributes for handoff
                set_span_attributes(
                    **{
                        "atp.multi_agent.all_succeeded": result.all_succeeded,
                        "atp.multi_agent.any_succeeded": result.any_succeeded,
                        "atp.multi_agent.duration_seconds": result.duration_seconds,
                        "atp.handoff.agents_executed": (
                            handoff_result.total_agents_executed
                        ),
                    }
                )

                if result.all_succeeded:
                    span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(Status(StatusCode.ERROR, "Handoff did not succeed"))

                add_span_event(
                    "multi_agent_test_completed",
                    {
                        "all_succeeded": result.all_succeeded,
                        "mode": "handoff",
                        "agents_executed": handoff_result.total_agents_executed,
                    },
                )

                # Emit completion event
                self._emit_progress(
                    ProgressEvent(
                        event_type=(
                            ProgressEventType.TEST_COMPLETED
                            if result.any_succeeded
                            else ProgressEventType.TEST_FAILED
                        ),
                        test_id=test.id,
                        test_name=test.name,
                        success=result.all_succeeded,
                        details={
                            "mode": self.mode.value,
                            "agents_executed": handoff_result.total_agents_executed,
                            "termination_reason": (
                                handoff_result.handoff_metrics.termination_reason
                            ),
                        },
                    )
                )

                return result

            # Comparison mode (original behavior)
            if self.parallel_agents:
                result.agent_results = await self._execute_agents_parallel(
                    test=test,
                    runs=runs,
                )
            else:
                result.agent_results = await self._execute_agents_sequential(
                    test=test,
                    runs=runs,
                )

            result.end_time = datetime.now()

            # Determine winner if in comparison mode
            if self.mode == MultiAgentMode.COMPARISON and self.determine_winner:
                winner, rankings = self._determine_test_winner(result.agent_results)
                result.winner = winner
                result.rankings = rankings

            # Set span attributes
            set_span_attributes(
                **{
                    "atp.multi_agent.all_succeeded": result.all_succeeded,
                    "atp.multi_agent.any_succeeded": result.any_succeeded,
                    "atp.multi_agent.winner": result.winner or "none",
                    "atp.multi_agent.duration_seconds": result.duration_seconds,
                }
            )

            if result.all_succeeded:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR, "Not all agents succeeded"))

            add_span_event(
                "multi_agent_test_completed",
                {
                    "all_succeeded": result.all_succeeded,
                    "winner": result.winner or "none",
                },
            )

            # Emit completion event
            self._emit_progress(
                ProgressEvent(
                    event_type=(
                        ProgressEventType.TEST_COMPLETED
                        if result.any_succeeded
                        else ProgressEventType.TEST_FAILED
                    ),
                    test_id=test.id,
                    test_name=test.name,
                    success=result.all_succeeded,
                    details={
                        "mode": self.mode.value,
                        "winner": result.winner,
                        "agent_results": {
                            r.agent_name: r.success for r in result.agent_results
                        },
                    },
                )
            )

        return result

    async def _execute_agents_parallel(
        self,
        test: TestDefinition,
        runs: int | None = None,
    ) -> list[AgentTestResult]:
        """Execute test across agents in parallel."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_parallel_agents)

        async def run_with_semaphore(agent: AgentConfig) -> AgentTestResult:
            async with self._semaphore:  # type: ignore[union-attr]
                return await self._execute_for_agent(
                    agent=agent,
                    test=test,
                    runs=runs,
                )

        tasks = [run_with_semaphore(agent) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        agent_results: list[AgentTestResult] = []
        for agent, result in zip(self.agents, results):
            if isinstance(result, BaseException):
                logger.error("Agent %s failed with exception: %s", agent.name, result)
                # Create a failed result for the exception

                agent_results.append(
                    AgentTestResult(
                        agent_name=agent.name,
                        test_result=TestResult(
                            test=test,
                            error=str(result),
                        ),
                        end_time=datetime.now(),
                    )
                )
            else:
                agent_results.append(result)

        return agent_results

    async def _execute_agents_sequential(
        self,
        test: TestDefinition,
        runs: int | None = None,
    ) -> list[AgentTestResult]:
        """Execute test across agents sequentially."""
        results: list[AgentTestResult] = []
        for agent in self.agents:
            try:
                result = await self._execute_for_agent(
                    agent=agent,
                    test=test,
                    runs=runs,
                )
                results.append(result)
            except Exception as e:
                logger.error("Agent %s failed with exception: %s", agent.name, e)
                results.append(
                    AgentTestResult(
                        agent_name=agent.name,
                        test_result=TestResult(
                            test=test,
                            error=str(e),
                        ),
                        end_time=datetime.now(),
                    )
                )
        return results

    async def _execute_for_agent(
        self,
        agent: AgentConfig,
        test: TestDefinition,
        runs: int | None = None,
    ) -> AgentTestResult:
        """Execute a test for a specific agent."""
        start_time = datetime.now()

        # Get or create orchestrator for this agent
        if agent.name not in self._orchestrators:
            self._orchestrators[agent.name] = self._create_orchestrator(agent)

        orchestrator = self._orchestrators[agent.name]
        test_result = await orchestrator.run_single_test(
            test=test,
            runs=runs,
        )

        return AgentTestResult(
            agent_name=agent.name,
            test_result=test_result,
            start_time=start_time,
            end_time=datetime.now(),
        )

    # =========================================================================
    # Collaboration Mode Methods
    # =========================================================================

    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        self._message_counter += 1
        return f"msg-{self._message_counter:06d}"

    def _create_shared_context(self, test: TestDefinition) -> SharedContext:
        """Create a new shared context for collaboration."""
        return SharedContext(
            task_description=test.task.description,
            artifacts={},
            variables={},
            messages=[],
            current_turn=0,
            completed_subtasks=[],
        )

    def _get_next_agent(
        self, current_turn: int, last_agent: str | None
    ) -> AgentConfig | None:
        """
        Get the next agent for a turn in collaboration mode.

        Uses round-robin scheduling unless a coordinator is specified.
        """
        if not self.agents:
            return None

        # If coordinator is specified, alternate between coordinator and others
        coordinator_name = self.collaboration_config.coordinator_agent
        if coordinator_name:
            coordinator = next(
                (a for a in self.agents if a.name == coordinator_name), None
            )
            if coordinator:
                # Coordinator goes first on odd turns (1, 3, 5...)
                # Other agents on even turns (2, 4, 6...)
                if current_turn % 2 == 1:
                    return coordinator
                else:
                    # Get non-coordinator agents
                    others = [a for a in self.agents if a.name != coordinator_name]
                    if others:
                        idx = (current_turn // 2 - 1) % len(others)
                        return others[idx]
                    return coordinator

        # Round-robin scheduling
        idx = (current_turn - 1) % len(self.agents)
        return self.agents[idx]

    def _create_collaboration_message(
        self,
        message_type: CollaborationMessageType,
        from_agent: str,
        content: dict[str, Any],
        to_agent: str | None = None,
        turn_number: int = 0,
        in_reply_to: str | None = None,
    ) -> CollaborationMessage:
        """Create a new collaboration message."""
        return CollaborationMessage(
            message_id=self._generate_message_id(),
            message_type=message_type,
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            timestamp=datetime.now(),
            turn_number=turn_number,
            in_reply_to=in_reply_to,
        )

    def _check_termination(
        self,
        shared_context: SharedContext,
        turn_results: list[CollaborationTurnResult],
    ) -> tuple[bool, str]:
        """
        Check if collaboration should terminate.

        Returns:
            Tuple of (should_terminate, reason)
        """
        config = self.collaboration_config
        current_turn = shared_context.current_turn

        # Check max turns
        if current_turn >= config.max_turns:
            return True, "max_turns_reached"

        # Check termination conditions
        if config.termination_condition == "max_turns":
            # Only terminate when max turns reached
            return False, ""

        # Check if any agent has completed
        completed_agents = set()
        for turn_result in turn_results:
            if turn_result.response and turn_result.success:
                completed_agents.add(turn_result.agent_name)

        if config.termination_condition == "any_complete":
            if completed_agents:
                return True, "agent_completed"

        if config.termination_condition == "all_complete":
            all_agent_names = {a.name for a in self.agents}
            if completed_agents >= all_agent_names:
                return True, "all_agents_completed"

        if config.termination_condition == "consensus":
            # Check if all recent turn results agree (simplified consensus)
            if len(turn_results) >= len(self.agents):
                recent_turns = turn_results[-len(self.agents) :]
                if all(tr.success for tr in recent_turns):
                    return True, "consensus_reached"

        return False, ""

    async def _execute_collaboration_turn(
        self,
        agent: AgentConfig,
        test: TestDefinition,
        shared_context: SharedContext,
        turn_number: int,
    ) -> CollaborationTurnResult:
        """Execute a single turn for an agent in collaboration mode."""
        turn_result = CollaborationTurnResult(
            turn_number=turn_number,
            agent_name=agent.name,
            start_time=datetime.now(),
        )

        # Get messages for this agent from the shared context
        agent_messages = shared_context.get_messages_for_agent(agent.name)
        turn_result.messages_received = [
            m for m in agent_messages if m.turn_number < turn_number
        ]

        try:
            # Create a modified test with collaboration context
            collab_test = self._create_collaboration_test(
                test=test,
                shared_context=shared_context,
                agent_name=agent.name,
                turn_number=turn_number,
            )

            # Execute the agent
            if agent.name not in self._orchestrators:
                self._orchestrators[agent.name] = self._create_orchestrator(agent)

            orchestrator = self._orchestrators[agent.name]
            test_result = await asyncio.wait_for(
                orchestrator.run_single_test(test=collab_test, runs=1),
                timeout=self.collaboration_config.turn_timeout_seconds,
            )

            # Check for test-level errors first
            if test_result.error:
                turn_result.error = test_result.error

            # Extract response from the first run
            if test_result.runs:
                turn_result.response = test_result.runs[0].response

                # Check for run-level errors
                run_error = test_result.runs[0].error
                if run_error and not turn_result.error:
                    turn_result.error = run_error

                # Extract any artifacts and add to shared context
                if turn_result.response and turn_result.response.artifacts:
                    for artifact in turn_result.response.artifacts:
                        artifact_key = getattr(artifact, "name", None) or getattr(
                            artifact, "path", f"artifact_{turn_number}"
                        )
                        shared_context.set_artifact(
                            artifact_key, artifact.model_dump(), agent.name
                        )

                # Create a result message
                result_msg = self._create_collaboration_message(
                    message_type=CollaborationMessageType.RESULT,
                    from_agent=agent.name,
                    content={
                        "status": (
                            turn_result.response.status.value
                            if turn_result.response
                            else "unknown"
                        ),
                        "artifacts": [
                            a.model_dump()
                            for a in (
                                turn_result.response.artifacts
                                if turn_result.response
                                else []
                            )
                        ],
                        "error": turn_result.error,
                    },
                    turn_number=turn_number,
                )
                shared_context.add_message(result_msg)
                turn_result.messages_sent.append(result_msg)

        except TimeoutError:
            turn_result.error = (
                f"Turn timeout after {self.collaboration_config.turn_timeout_seconds}s"
            )
            logger.warning("Agent %s timed out on turn %d", agent.name, turn_number)
        except Exception as e:
            turn_result.error = str(e)
            logger.error("Agent %s failed on turn %d: %s", agent.name, turn_number, e)

        turn_result.end_time = datetime.now()
        return turn_result

    def _create_collaboration_test(
        self,
        test: TestDefinition,
        shared_context: SharedContext,
        agent_name: str,
        turn_number: int,
    ) -> TestDefinition:
        """Create a modified test definition with collaboration context."""
        # Deep copy the test to avoid modifying the original
        collab_test = copy.deepcopy(test)

        # Build collaboration context string
        context_parts = [
            "=== COLLABORATION CONTEXT ===",
            f"Turn: {turn_number}",
            f"Your role: {agent_name}",
            "",
            f"Original task: {shared_context.task_description}",
            "",
        ]

        # Add recent messages
        recent_messages = shared_context.get_messages_for_agent(agent_name)[-10:]
        if recent_messages:
            context_parts.append("Recent messages:")
            for msg in recent_messages:
                context_parts.append(
                    f"  [{msg.from_agent} -> {msg.to_agent or 'all'}]: "
                    f"{msg.message_type.value} - {msg.content}"
                )
            context_parts.append("")

        # Add shared artifacts summary
        if shared_context.artifacts:
            context_parts.append("Shared artifacts:")
            for key, artifact_data in shared_context.artifacts.items():
                context_parts.append(
                    f"  - {key} (by {artifact_data.get('set_by', 'unknown')})"
                )
            context_parts.append("")

        # Add shared variables
        if shared_context.variables:
            context_parts.append("Shared state:")
            for key, value in shared_context.variables.items():
                context_parts.append(f"  - {key}: {value}")
            context_parts.append("")

        context_parts.append("=== END COLLABORATION CONTEXT ===")
        context_parts.append("")
        context_parts.append("Continue working on the task based on the above context.")

        # Prepend collaboration context to task description
        collab_test.task.description = (
            "\n".join(context_parts) + "\n\n" + collab_test.task.description
        )

        return collab_test

    def _calculate_collaboration_metrics(
        self,
        turn_results: list[CollaborationTurnResult],
        shared_context: SharedContext,
        termination_reason: str,
    ) -> CollaborationMetrics:
        """Calculate collaboration-specific metrics."""
        metrics = CollaborationMetrics(
            total_turns=len(turn_results),
            total_messages=len(shared_context.messages),
            termination_reason=termination_reason,
        )

        # Calculate per-agent metrics
        messages_per_agent: dict[str, int] = {}
        turns_per_agent: dict[str, int] = {}
        turn_durations: list[float] = []

        for turn_result in turn_results:
            agent = turn_result.agent_name
            turns_per_agent[agent] = turns_per_agent.get(agent, 0) + 1
            messages_per_agent[agent] = messages_per_agent.get(agent, 0) + len(
                turn_result.messages_sent
            )

            if turn_result.duration_seconds is not None:
                turn_durations.append(turn_result.duration_seconds)

            # Accumulate token and cost metrics
            if turn_result.response and turn_result.response.metrics:
                m = turn_result.response.metrics
                if m.total_tokens:
                    metrics.total_tokens += m.total_tokens
                if m.cost_usd:
                    metrics.total_cost_usd += m.cost_usd

        metrics.messages_per_agent = messages_per_agent
        metrics.turns_per_agent = turns_per_agent

        if turn_durations:
            metrics.avg_turn_duration_seconds = sum(turn_durations) / len(
                turn_durations
            )

        # Calculate contribution scores based on turns and messages
        total_activity = sum(turns_per_agent.values()) + sum(
            messages_per_agent.values()
        )
        if total_activity > 0:
            for agent in self.agents:
                agent_activity = turns_per_agent.get(
                    agent.name, 0
                ) + messages_per_agent.get(agent.name, 0)
                metrics.agent_contributions[agent.name] = (
                    agent_activity / total_activity
                )

        # Check if consensus was reached
        metrics.consensus_reached = termination_reason == "consensus_reached"

        return metrics

    def _create_final_collaboration_response(
        self,
        turn_results: list[CollaborationTurnResult],
        shared_context: SharedContext,
        test: TestDefinition,
    ) -> ATPResponse:
        """Create a final aggregated response from collaboration results."""
        # Aggregate all successful artifacts
        all_artifacts = []
        for artifact_data in shared_context.artifacts.values():
            artifact_value = artifact_data.get("value", {})
            if artifact_value:
                # Reconstruct artifact from stored data
                from atp.protocol import ArtifactStructured

                all_artifacts.append(
                    ArtifactStructured(
                        name=str(artifact_value.get("name", "collaboration_artifact")),
                        data=artifact_value,
                    )
                )

        # Determine overall status
        successful_turns = [tr for tr in turn_results if tr.success]
        if successful_turns:
            status = ResponseStatus.COMPLETED
        elif any(tr.error and "timeout" in tr.error.lower() for tr in turn_results):
            status = ResponseStatus.TIMEOUT
        else:
            status = ResponseStatus.FAILED

        # Aggregate metrics
        total_tokens = 0
        total_cost = 0.0
        total_steps = 0
        for tr in turn_results:
            if tr.response and tr.response.metrics:
                if tr.response.metrics.total_tokens:
                    total_tokens += tr.response.metrics.total_tokens
                if tr.response.metrics.cost_usd:
                    total_cost += tr.response.metrics.cost_usd
                if tr.response.metrics.total_steps:
                    total_steps += tr.response.metrics.total_steps

        from atp.protocol import Metrics

        metrics = Metrics(
            total_tokens=total_tokens if total_tokens > 0 else None,
            total_steps=total_steps if total_steps > 0 else None,
            cost_usd=total_cost if total_cost > 0 else None,
        )

        return ATPResponse(
            task_id=test.id,
            status=status,
            artifacts=all_artifacts,
            metrics=metrics,
        )

    async def _run_collaboration(
        self,
        test: TestDefinition,
    ) -> CollaborationResult:
        """
        Run a test in collaboration mode.

        Agents take turns working on the task, sharing context and messages.
        """
        shared_context = self._create_shared_context(test)
        turn_results: list[CollaborationTurnResult] = []
        collaboration_result = CollaborationResult(
            test=test,
            shared_context=shared_context,
            start_time=datetime.now(),
        )

        with tracer.start_as_current_span(
            f"collaboration:{test.name}",
            kind=SpanKind.INTERNAL,
            attributes={
                "atp.test.id": test.id,
                "atp.test.name": test.name,
                "atp.collaboration.max_turns": self.collaboration_config.max_turns,
                "atp.collaboration.agent_count": len(self.agents),
            },
        ) as span:
            add_span_event(
                "collaboration_started",
                {"agents": [a.name for a in self.agents]},
            )

            # Send initial task assignment messages to all agents
            for agent in self.agents:
                initial_msg = self._create_collaboration_message(
                    message_type=CollaborationMessageType.TASK_ASSIGNMENT,
                    from_agent="orchestrator",
                    to_agent=agent.name,
                    content={
                        "task": test.task.description,
                        "role": agent.name,
                        "total_agents": len(self.agents),
                    },
                    turn_number=0,
                )
                shared_context.add_message(initial_msg)

            last_agent: str | None = None
            termination_reason = ""

            try:
                while True:
                    shared_context.current_turn += 1
                    turn_number = shared_context.current_turn

                    # Emit progress
                    self._emit_progress(
                        ProgressEvent(
                            event_type=ProgressEventType.RUN_STARTED,
                            test_id=test.id,
                            test_name=test.name,
                            run_number=turn_number,
                            total_runs=self.collaboration_config.max_turns,
                            details={
                                "mode": "collaboration",
                                "turn": turn_number,
                            },
                        )
                    )

                    if self.collaboration_config.allow_parallel_turns:
                        # Execute all agents in parallel for this turn
                        turn_tasks = [
                            self._execute_collaboration_turn(
                                agent=agent,
                                test=test,
                                shared_context=shared_context,
                                turn_number=turn_number,
                            )
                            for agent in self.agents
                        ]
                        results = await asyncio.gather(*turn_tasks)
                        turn_results.extend(results)
                        last_agent = results[-1].agent_name if results else None
                    else:
                        # Get next agent for turn-based execution
                        agent = self._get_next_agent(turn_number, last_agent)
                        if not agent:
                            termination_reason = "no_agents_available"
                            break

                        turn_result = await self._execute_collaboration_turn(
                            agent=agent,
                            test=test,
                            shared_context=shared_context,
                            turn_number=turn_number,
                        )
                        turn_results.append(turn_result)
                        last_agent = agent.name

                    # Emit turn completion progress
                    self._emit_progress(
                        ProgressEvent(
                            event_type=ProgressEventType.RUN_COMPLETED,
                            test_id=test.id,
                            test_name=test.name,
                            run_number=turn_number,
                            total_runs=self.collaboration_config.max_turns,
                            success=turn_results[-1].success if turn_results else False,
                            details={
                                "mode": "collaboration",
                                "turn": turn_number,
                                "agent": last_agent,
                            },
                        )
                    )

                    # Check termination condition
                    should_terminate, reason = self._check_termination(
                        shared_context, turn_results
                    )
                    if should_terminate:
                        termination_reason = reason
                        break

            except Exception as e:
                collaboration_result.error = str(e)
                termination_reason = "error"
                logger.error("Collaboration failed: %s", e)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

            # Calculate collaboration metrics
            collaboration_result.collaboration_metrics = (
                self._calculate_collaboration_metrics(
                    turn_results, shared_context, termination_reason
                )
            )
            collaboration_result.turn_results = turn_results
            collaboration_result.shared_context = shared_context

            # Create final aggregated response
            collaboration_result.final_response = (
                self._create_final_collaboration_response(
                    turn_results, shared_context, test
                )
            )

            collaboration_result.end_time = datetime.now()

            # Set span attributes
            set_span_attributes(
                **{
                    "atp.collaboration.total_turns": len(turn_results),
                    "atp.collaboration.success": collaboration_result.success,
                    "atp.collaboration.termination_reason": termination_reason,
                }
            )

            if collaboration_result.success:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(
                    Status(StatusCode.ERROR, collaboration_result.error or "Failed")
                )

            add_span_event(
                "collaboration_completed",
                {
                    "total_turns": len(turn_results),
                    "success": collaboration_result.success,
                    "termination_reason": termination_reason,
                },
            )

        return collaboration_result

    # =========================================================================
    # Handoff Mode Methods
    # =========================================================================

    def _create_handoff_context(self, test: TestDefinition) -> HandoffContext:
        """Create a new handoff context for sequential agent execution."""
        return HandoffContext(
            original_task=test.task.description,
            previous_outputs=[],
            agent_sequence=[a.name for a in self.agents],
            current_position=0,
            accumulated_artifacts={},
            handoff_notes=[],
        )

    def _should_trigger_handoff(
        self, turn_result: HandoffTurnResult, position: int
    ) -> bool:
        """Determine if handoff should be triggered based on config and result."""
        trigger = self.handoff_config.handoff_trigger

        # Last agent never triggers handoff
        if position >= len(self.agents) - 1:
            return False

        if trigger == HandoffTrigger.ALWAYS:
            return True
        elif trigger == HandoffTrigger.ON_SUCCESS:
            return turn_result.success
        elif trigger == HandoffTrigger.ON_FAILURE:
            return not turn_result.success
        elif trigger == HandoffTrigger.ON_PARTIAL:
            if turn_result.response:
                return turn_result.response.status == ResponseStatus.PARTIAL
            return False
        elif trigger == HandoffTrigger.EXPLICIT:
            # Check if agent explicitly requested handoff (in response metadata)
            if turn_result.response and turn_result.response.artifacts:
                for artifact in turn_result.response.artifacts:
                    artifact_data = artifact.model_dump()
                    if artifact_data.get("name") == "handoff_request":
                        return True
            return False
        return True

    def _build_handoff_context_string(
        self, context: HandoffContext, agent_name: str, is_final: bool
    ) -> str:
        """Build a context string to prepend to the task for handoff mode."""
        config = self.handoff_config
        context_parts = [
            "=== HANDOFF CONTEXT ===",
            f"Agent: {agent_name}",
            f"Position: {context.current_position + 1} of "
            f"{len(context.agent_sequence)}",
            "",
            f"Original task: {context.original_task}",
            "",
        ]

        # Add previous outputs based on accumulation mode
        if context.previous_outputs:
            context_parts.append("Previous agent outputs:")

            if config.context_accumulation == ContextAccumulationMode.REPLACE:
                # Only show the last output
                last_output = context.previous_outputs[-1]
                context_parts.append(self._format_agent_output(last_output))
            elif config.context_accumulation == ContextAccumulationMode.SUMMARY:
                # Show summarized outputs
                context_parts.append(
                    f"  Total agents executed: {len(context.previous_outputs)}"
                )
                successful = [o for o in context.previous_outputs if o.error is None]
                failed = [o for o in context.previous_outputs if o.error is not None]
                context_parts.append(f"  Successful: {len(successful)}")
                context_parts.append(f"  Failed: {len(failed)}")
                if context.previous_outputs:
                    last = context.previous_outputs[-1]
                    context_parts.append(f"  Last agent: {last.agent_name}")
                    if last.error:
                        context_parts.append(f"    Status: FAILED - {last.error}")
                    else:
                        context_parts.append("    Status: SUCCESS")
            else:
                # APPEND or MERGE - show all outputs
                for output in context.previous_outputs:
                    context_parts.append(self._format_agent_output(output))

            context_parts.append("")

        # Add handoff notes
        if context.handoff_notes:
            context_parts.append("Handoff notes:")
            for note in context.handoff_notes:
                context_parts.append(f"  - {note}")
            context_parts.append("")

        # Add accumulated artifacts summary
        if context.accumulated_artifacts:
            context_parts.append("Accumulated artifacts:")
            for key in context.accumulated_artifacts.keys():
                context_parts.append(f"  - {key}")
            context_parts.append("")

        # Add role instruction for final agent
        if is_final and config.final_agent_decides:
            context_parts.append(
                "You are the FINAL agent in this handoff chain. "
                "Please provide the definitive answer/solution based on "
                "the work done by previous agents."
            )
            context_parts.append("")

        context_parts.append("=== END HANDOFF CONTEXT ===")
        context_parts.append("")
        context_parts.append("Continue working on the task based on the above context.")

        # Apply context size limit if configured
        context_str = "\n".join(context_parts)
        if config.max_context_size and len(context_str) > config.max_context_size:
            context_str = context_str[: config.max_context_size] + "\n... [truncated]"

        return context_str

    def _format_agent_output(self, output: AgentHandoffOutput) -> str:
        """Format a single agent output for context string."""
        lines = [f"  Agent: {output.agent_name} (position {output.sequence_position})"]
        if output.error:
            lines.append(f"    Status: FAILED - {output.error}")
        else:
            lines.append("    Status: SUCCESS")
            if output.artifacts:
                lines.append(f"    Artifacts: {len(output.artifacts)} produced")
        if output.duration_seconds is not None:
            lines.append(f"    Duration: {output.duration_seconds:.2f}s")
        return "\n".join(lines)

    def _create_handoff_test(
        self,
        test: TestDefinition,
        context: HandoffContext,
        agent_name: str,
        is_final: bool,
    ) -> TestDefinition:
        """Create a modified test definition with handoff context."""
        handoff_test = copy.deepcopy(test)

        # Build and prepend handoff context
        context_str = self._build_handoff_context_string(context, agent_name, is_final)
        handoff_test.task.description = (
            context_str + "\n\n" + handoff_test.task.description
        )

        return handoff_test

    async def _execute_handoff_turn(
        self,
        agent: AgentConfig,
        test: TestDefinition,
        context: HandoffContext,
        position: int,
    ) -> HandoffTurnResult:
        """Execute a single agent in the handoff chain."""
        is_final = position >= len(self.agents) - 1
        turn_result = HandoffTurnResult(
            agent_name=agent.name,
            sequence_position=position,
            context_received=copy.deepcopy(context),
            start_time=datetime.now(),
        )

        try:
            # Create modified test with handoff context
            handoff_test = self._create_handoff_test(
                test=test,
                context=context,
                agent_name=agent.name,
                is_final=is_final,
            )

            # Get or create orchestrator for this agent
            if agent.name not in self._orchestrators:
                self._orchestrators[agent.name] = self._create_orchestrator(agent)

            orchestrator = self._orchestrators[agent.name]

            # Execute with timeout
            test_result = await asyncio.wait_for(
                orchestrator.run_single_test(test=handoff_test, runs=1),
                timeout=self.handoff_config.agent_timeout_seconds,
            )

            # Check for test-level errors
            if test_result.error:
                turn_result.error = test_result.error

            # Extract response from the first run
            if test_result.runs:
                turn_result.response = test_result.runs[0].response

                # Check for run-level errors
                run_error = test_result.runs[0].error
                if run_error and not turn_result.error:
                    turn_result.error = run_error

            # Determine if handoff should be triggered
            turn_result.handoff_triggered = self._should_trigger_handoff(
                turn_result, position
            )

        except TimeoutError:
            turn_result.error = (
                f"Agent timeout after {self.handoff_config.agent_timeout_seconds}s"
            )
            logger.warning("Agent %s timed out at position %d", agent.name, position)
        except Exception as e:
            turn_result.error = str(e)
            logger.error("Agent %s failed at position %d: %s", agent.name, position, e)

        turn_result.end_time = datetime.now()

        # Update context with this agent's output
        agent_output = AgentHandoffOutput(
            agent_name=agent.name,
            sequence_position=position,
            response=turn_result.response,
            artifacts=(
                list(turn_result.response.artifacts)
                if turn_result.response and turn_result.response.artifacts
                else []
            ),
            error=turn_result.error,
            duration_seconds=turn_result.duration_seconds,
        )
        context.add_output(agent_output)
        context.current_position = position + 1

        # Store updated context
        turn_result.context_passed = copy.deepcopy(context)

        return turn_result

    def _calculate_handoff_metrics(
        self,
        turn_results: list[HandoffTurnResult],
        context: HandoffContext,
        termination_reason: str,
    ) -> HandoffMetrics:
        """Calculate handoff-specific metrics."""
        metrics = HandoffMetrics(
            total_agents=len(self.agents),
            agents_executed=len(turn_results),
            termination_reason=termination_reason,
        )

        total_duration = 0.0
        total_tokens = 0
        total_cost = 0.0

        for turn_result in turn_results:
            agent = turn_result.agent_name

            # Track duration
            if turn_result.duration_seconds is not None:
                metrics.duration_per_agent[agent] = turn_result.duration_seconds
                total_duration += turn_result.duration_seconds

            # Track tokens and cost
            if turn_result.response and turn_result.response.metrics:
                m = turn_result.response.metrics
                if m.total_tokens:
                    metrics.tokens_per_agent[agent] = m.total_tokens
                    total_tokens += m.total_tokens
                if m.cost_usd:
                    metrics.cost_per_agent[agent] = m.cost_usd
                    total_cost += m.cost_usd

            # Count handoffs
            if turn_result.handoff_triggered:
                if turn_result.success:
                    metrics.successful_handoffs += 1
                else:
                    metrics.failed_handoffs += 1

        metrics.total_duration_seconds = total_duration
        metrics.total_tokens = total_tokens
        metrics.total_cost_usd = total_cost

        # Calculate contribution scores based on success and position
        if turn_results:
            total_successful = sum(1 for tr in turn_results if tr.success)
            for turn_result in turn_results:
                agent = turn_result.agent_name
                # Base contribution on success and position weight
                # Later agents in a successful chain get more credit
                if turn_result.success:
                    position_weight = (turn_result.sequence_position + 1) / len(
                        turn_results
                    )
                    metrics.agent_contributions[agent] = position_weight / max(
                        total_successful, 1
                    )
                else:
                    metrics.agent_contributions[agent] = 0.0

        # Track final agent
        if turn_results:
            metrics.final_agent = turn_results[-1].agent_name

        return metrics

    def _create_final_handoff_response(
        self,
        turn_results: list[HandoffTurnResult],
        context: HandoffContext,
        test: TestDefinition,
    ) -> ATPResponse:
        """Create a final aggregated response from handoff results."""
        # Use the final agent's response if available and successful
        if turn_results:
            final_turn = turn_results[-1]
            if final_turn.response and final_turn.success:
                return final_turn.response

            # If final agent failed, find the last successful response
            for turn_result in reversed(turn_results):
                if turn_result.response and turn_result.success:
                    return turn_result.response

        # No successful responses - create a failed response
        all_artifacts = []
        for artifact_data in context.accumulated_artifacts.values():
            from atp.protocol import ArtifactStructured

            all_artifacts.append(
                ArtifactStructured(
                    name=str(artifact_data.get("name", "handoff_artifact"))
                    if isinstance(artifact_data, dict)
                    else "handoff_artifact",
                    data=artifact_data
                    if isinstance(artifact_data, dict)
                    else {"value": artifact_data},
                )
            )

        # Determine overall status
        if any(tr.success for tr in turn_results):
            status = ResponseStatus.PARTIAL
        elif any(tr.error and "timeout" in tr.error.lower() for tr in turn_results):
            status = ResponseStatus.TIMEOUT
        else:
            status = ResponseStatus.FAILED

        # Aggregate metrics
        total_tokens = 0
        total_cost = 0.0
        total_steps = 0
        for tr in turn_results:
            if tr.response and tr.response.metrics:
                if tr.response.metrics.total_tokens:
                    total_tokens += tr.response.metrics.total_tokens
                if tr.response.metrics.cost_usd:
                    total_cost += tr.response.metrics.cost_usd
                if tr.response.metrics.total_steps:
                    total_steps += tr.response.metrics.total_steps

        from atp.protocol import Metrics

        metrics = Metrics(
            total_tokens=total_tokens if total_tokens > 0 else None,
            total_steps=total_steps if total_steps > 0 else None,
            cost_usd=total_cost if total_cost > 0 else None,
        )

        # Collect errors
        errors = [tr.error for tr in turn_results if tr.error]
        error_msg = "; ".join(errors) if errors else None

        return ATPResponse(
            task_id=test.id,
            status=status,
            artifacts=all_artifacts,
            metrics=metrics,
            error=error_msg,
        )

    async def _run_handoff(self, test: TestDefinition) -> HandoffResult:
        """
        Run a test in handoff mode.

        Agents execute sequentially, each receiving context from previous agents.
        """
        context = self._create_handoff_context(test)
        turn_results: list[HandoffTurnResult] = []
        handoff_result = HandoffResult(
            test=test,
            handoff_context=context,
            start_time=datetime.now(),
        )

        with tracer.start_as_current_span(
            f"handoff:{test.name}",
            kind=SpanKind.INTERNAL,
            attributes={
                "atp.test.id": test.id,
                "atp.test.name": test.name,
                "atp.handoff.total_agents": len(self.agents),
                "atp.handoff.trigger": self.handoff_config.handoff_trigger.value,
            },
        ) as span:
            add_span_event(
                "handoff_started",
                {"agents": [a.name for a in self.agents]},
            )

            termination_reason = ""

            try:
                for position, agent in enumerate(self.agents):
                    # Emit progress event
                    self._emit_progress(
                        ProgressEvent(
                            event_type=ProgressEventType.RUN_STARTED,
                            test_id=test.id,
                            test_name=test.name,
                            run_number=position + 1,
                            total_runs=len(self.agents),
                            details={
                                "mode": "handoff",
                                "agent": agent.name,
                                "position": position,
                            },
                        )
                    )

                    turn_result = await self._execute_handoff_turn(
                        agent=agent,
                        test=test,
                        context=context,
                        position=position,
                    )
                    turn_results.append(turn_result)

                    # Emit turn completion progress
                    self._emit_progress(
                        ProgressEvent(
                            event_type=ProgressEventType.RUN_COMPLETED,
                            test_id=test.id,
                            test_name=test.name,
                            run_number=position + 1,
                            total_runs=len(self.agents),
                            success=turn_result.success,
                            details={
                                "mode": "handoff",
                                "agent": agent.name,
                                "position": position,
                                "handoff_triggered": turn_result.handoff_triggered,
                            },
                        )
                    )

                    # Check if we should continue
                    # Agent failed if there's an error OR the response status is FAILED
                    agent_failed = turn_result.error or (
                        turn_result.response
                        and turn_result.response.status == ResponseStatus.FAILED
                    )
                    if agent_failed and not self.handoff_config.continue_on_failure:
                        termination_reason = "agent_failed"
                        break

                    # Check if handoff was triggered
                    if not turn_result.handoff_triggered:
                        if position < len(self.agents) - 1:
                            termination_reason = "handoff_not_triggered"
                        else:
                            termination_reason = "chain_completed"
                        break

                if not termination_reason:
                    termination_reason = "chain_completed"

            except Exception as e:
                handoff_result.error = str(e)
                termination_reason = "error"
                logger.error("Handoff failed: %s", e)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

            # Calculate handoff metrics
            handoff_result.handoff_metrics = self._calculate_handoff_metrics(
                turn_results, context, termination_reason
            )
            handoff_result.turn_results = turn_results
            handoff_result.handoff_context = context

            # Create final aggregated response
            handoff_result.final_response = self._create_final_handoff_response(
                turn_results, context, test
            )

            handoff_result.end_time = datetime.now()

            # Set span attributes
            set_span_attributes(
                **{
                    "atp.handoff.agents_executed": len(turn_results),
                    "atp.handoff.success": handoff_result.success,
                    "atp.handoff.termination_reason": termination_reason,
                }
            )

            if handoff_result.success:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(
                    Status(StatusCode.ERROR, handoff_result.error or "Failed")
                )

            add_span_event(
                "handoff_completed",
                {
                    "agents_executed": len(turn_results),
                    "success": handoff_result.success,
                    "termination_reason": termination_reason,
                },
            )

        return handoff_result

    async def run_suite(
        self,
        suite: TestSuite,
        runs_per_test: int | None = None,
    ) -> MultiAgentSuiteResult:
        """
        Execute a complete test suite across all agents.

        Args:
            suite: Test suite to execute.
            runs_per_test: Override number of runs per test.

        Returns:
            MultiAgentSuiteResult with results from all agents.
        """
        num_runs = runs_per_test if runs_per_test is not None else self.runs_per_test
        total_tests = len(suite.tests)

        result = MultiAgentSuiteResult(
            suite_name=suite.test_suite,
            mode=self.mode,
            agents=[a.name for a in self.agents],
            start_time=datetime.now(),
        )

        # Record suite start in metrics
        metrics = get_metrics()
        if metrics:
            metrics.record_suite_start(pending_tests=total_tests * len(self.agents))

        with tracer.start_as_current_span(
            f"multi_agent_suite:{suite.test_suite}",
            kind=SpanKind.INTERNAL,
            attributes={
                "atp.suite.name": suite.test_suite,
                "atp.multi_agent.mode": self.mode.value,
                "atp.multi_agent.agent_count": len(self.agents),
                "atp.suite.total_tests": total_tests,
                "atp.suite.runs_per_test": num_runs,
            },
        ) as span:
            # Emit suite started event
            self._emit_progress(
                ProgressEvent(
                    event_type=ProgressEventType.SUITE_STARTED,
                    suite_name=suite.test_suite,
                    total_tests=total_tests,
                    details={
                        "mode": self.mode.value,
                        "agents": [a.name for a in self.agents],
                        "runs_per_test": num_runs,
                    },
                )
            )
            add_span_event(
                "multi_agent_suite_started",
                {
                    "total_tests": total_tests,
                    "agents": [a.name for a in self.agents],
                },
            )

            # Apply defaults to tests
            suite.apply_defaults()

            try:
                # Run all tests across all agents
                for idx, test in enumerate(suite.tests):
                    logger.info(
                        "Running test %d/%d: %s across %d agents",
                        idx + 1,
                        total_tests,
                        test.name,
                        len(self.agents),
                    )

                    multi_test_result = await self.run_single_test(
                        test=test,
                        runs=num_runs,
                    )
                    result.test_results.append(multi_test_result)

                # Build per-agent suite results
                for agent in self.agents:
                    agent_test_results: list[TestResult] = []
                    for mtr in result.test_results:
                        agent_result = mtr.get_result_for_agent(agent.name)
                        if agent_result:
                            agent_test_results.append(agent_result.test_result)

                    suite_result = SuiteResult(
                        suite_name=suite.test_suite,
                        agent_name=agent.name,
                        tests=agent_test_results,
                        start_time=result.start_time,
                        end_time=datetime.now(),
                    )
                    result.agent_suite_results[agent.name] = suite_result

                # Calculate comparison metrics
                for agent_name, suite_result in result.agent_suite_results.items():
                    comparison = self._calculate_metrics_for_agent(
                        agent_name, suite_result
                    )
                    result.comparison_metrics.append(comparison)

                # Generate overall rankings for each metric
                for metric in self.ranking_metrics:
                    rankings = self._rank_agents(result.comparison_metrics, metric)
                    result.overall_rankings.extend(rankings)

            except Exception as e:
                result.error = str(e)
                logger.error("Multi-agent suite execution failed: %s", e)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

            result.end_time = datetime.now()

            # Record suite end in metrics
            if metrics:
                metrics.record_suite_end()

            # Set suite result attributes
            set_span_attributes(
                **{
                    "atp.multi_agent.best_agent": result.best_agent or "none",
                    "atp.multi_agent.duration_seconds": result.duration_seconds,
                    "atp.suite.total_tests": result.total_tests,
                }
            )

            # Emit suite completed event
            self._emit_progress(
                ProgressEvent(
                    event_type=ProgressEventType.SUITE_COMPLETED,
                    suite_name=suite.test_suite,
                    total_tests=result.total_tests,
                    completed_tests=len(result.test_results),
                    success=result.error is None,
                    error=result.error,
                    details={
                        "mode": self.mode.value,
                        "best_agent": result.best_agent,
                        "agent_results": {
                            name: sr.success_rate
                            for name, sr in result.agent_suite_results.items()
                        },
                    },
                )
            )
            add_span_event(
                "multi_agent_suite_completed",
                {
                    "best_agent": result.best_agent or "none",
                    "total_tests": result.total_tests,
                },
            )

        return result

    async def cleanup(self) -> None:
        """Clean up all orchestrator resources."""
        for orchestrator in self._orchestrators.values():
            await orchestrator.cleanup()
        self._orchestrators.clear()

        for agent in self.agents:
            await agent.adapter.cleanup()

    async def __aenter__(self) -> "MultiAgentOrchestrator":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.cleanup()


async def run_comparison(
    agents: list[AgentConfig],
    test: TestDefinition,
    progress_callback: ProgressCallback | None = None,
    runs: int = 1,
    parallel: bool = True,
    ranking_metrics: list[RankingMetric] | None = None,
) -> MultiAgentTestResult:
    """
    Convenience function to run a single test in comparison mode.

    Args:
        agents: List of agent configurations.
        test: Test definition.
        progress_callback: Optional progress callback.
        runs: Number of runs per agent.
        parallel: Whether to run agents in parallel.
        ranking_metrics: Metrics for ranking.

    Returns:
        MultiAgentTestResult.
    """
    async with MultiAgentOrchestrator(
        agents=agents,
        mode=MultiAgentMode.COMPARISON,
        progress_callback=progress_callback,
        runs_per_test=runs,
        parallel_agents=parallel,
        ranking_metrics=ranking_metrics,
    ) as orchestrator:
        return await orchestrator.run_single_test(test)


async def run_suite_comparison(
    agents: list[AgentConfig],
    suite: TestSuite,
    progress_callback: ProgressCallback | None = None,
    runs_per_test: int = 1,
    parallel: bool = True,
    ranking_metrics: list[RankingMetric] | None = None,
) -> MultiAgentSuiteResult:
    """
    Convenience function to run a test suite in comparison mode.

    Args:
        agents: List of agent configurations.
        suite: Test suite.
        progress_callback: Optional progress callback.
        runs_per_test: Number of runs per test per agent.
        parallel: Whether to run agents in parallel.
        ranking_metrics: Metrics for ranking.

    Returns:
        MultiAgentSuiteResult.
    """
    async with MultiAgentOrchestrator(
        agents=agents,
        mode=MultiAgentMode.COMPARISON,
        progress_callback=progress_callback,
        runs_per_test=runs_per_test,
        parallel_agents=parallel,
        ranking_metrics=ranking_metrics,
    ) as orchestrator:
        return await orchestrator.run_suite(suite, runs_per_test)


async def run_collaboration(
    agents: list[AgentConfig],
    test: TestDefinition,
    collaboration_config: CollaborationConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> MultiAgentTestResult:
    """
    Convenience function to run a single test in collaboration mode.

    Args:
        agents: List of agent configurations.
        test: Test definition.
        collaboration_config: Configuration for collaboration behavior.
        progress_callback: Optional progress callback.

    Returns:
        MultiAgentTestResult with collaboration_result populated.
    """
    async with MultiAgentOrchestrator(
        agents=agents,
        mode=MultiAgentMode.COLLABORATION,
        collaboration_config=collaboration_config,
        progress_callback=progress_callback,
    ) as orchestrator:
        return await orchestrator.run_single_test(test)


async def run_suite_collaboration(
    agents: list[AgentConfig],
    suite: TestSuite,
    collaboration_config: CollaborationConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> MultiAgentSuiteResult:
    """
    Convenience function to run a test suite in collaboration mode.

    Args:
        agents: List of agent configurations.
        suite: Test suite.
        collaboration_config: Configuration for collaboration behavior.
        progress_callback: Optional progress callback.

    Returns:
        MultiAgentSuiteResult.
    """
    async with MultiAgentOrchestrator(
        agents=agents,
        mode=MultiAgentMode.COLLABORATION,
        collaboration_config=collaboration_config,
        progress_callback=progress_callback,
    ) as orchestrator:
        return await orchestrator.run_suite(suite)


async def run_handoff(
    agents: list[AgentConfig],
    test: TestDefinition,
    handoff_config: HandoffConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> MultiAgentTestResult:
    """
    Convenience function to run a single test in handoff mode.

    Agents execute sequentially, each receiving context and results
    from previous agents in the chain.

    Args:
        agents: List of agent configurations (order determines handoff sequence).
        test: Test definition.
        handoff_config: Configuration for handoff behavior.
        progress_callback: Optional progress callback.

    Returns:
        MultiAgentTestResult with handoff_result populated.
    """
    async with MultiAgentOrchestrator(
        agents=agents,
        mode=MultiAgentMode.HANDOFF,
        handoff_config=handoff_config,
        progress_callback=progress_callback,
    ) as orchestrator:
        return await orchestrator.run_single_test(test)


async def run_suite_handoff(
    agents: list[AgentConfig],
    suite: TestSuite,
    handoff_config: HandoffConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> MultiAgentSuiteResult:
    """
    Convenience function to run a test suite in handoff mode.

    Each test is executed with agents in sequence, passing context
    from one agent to the next.

    Args:
        agents: List of agent configurations (order determines handoff sequence).
        suite: Test suite.
        handoff_config: Configuration for handoff behavior.
        progress_callback: Optional progress callback.

    Returns:
        MultiAgentSuiteResult.
    """
    async with MultiAgentOrchestrator(
        agents=agents,
        mode=MultiAgentMode.HANDOFF,
        handoff_config=handoff_config,
        progress_callback=progress_callback,
    ) as orchestrator:
        return await orchestrator.run_suite(suite)
