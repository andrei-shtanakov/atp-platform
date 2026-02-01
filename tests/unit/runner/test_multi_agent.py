"""Tests for multi-agent orchestrator."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from atp.adapters.base import AgentAdapter
from atp.loader.models import (
    Constraints,
    TaskDefinition,
    TestDefaults,
    TestDefinition,
    TestSuite,
)
from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    Metrics,
    ResponseStatus,
)
from atp.runner.models import ProgressEvent, ProgressEventType
from atp.runner.multi_agent import (
    AgentConfig,
    AgentRanking,
    AgentTestResult,
    CollaborationConfig,
    CollaborationMessage,
    CollaborationMessageType,
    CollaborationMetrics,
    CollaborationResult,
    CollaborationTurnResult,
    ComparisonMetrics,
    MultiAgentMode,
    MultiAgentOrchestrator,
    MultiAgentSuiteResult,
    MultiAgentTestResult,
    RankingMetric,
    SharedContext,
    run_collaboration,
    run_comparison,
    run_suite_collaboration,
    run_suite_comparison,
)


class MockAdapter(AgentAdapter):
    """Mock adapter for testing."""

    def __init__(
        self,
        name: str = "mock",
        response: ATPResponse | None = None,
        events: list[ATPEvent] | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
        metrics: Metrics | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._response = response or ATPResponse(
            task_id="test-task",
            status=ResponseStatus.COMPLETED,
            metrics=metrics,
        )
        self._events = events or []
        self._error = error
        self._delay = delay

    @property
    def adapter_type(self) -> str:
        return self._name

    async def execute(self, request: ATPRequest) -> ATPResponse:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return ATPResponse(
            task_id=request.task_id,
            status=self._response.status,
            artifacts=self._response.artifacts,
            metrics=self._response.metrics,
            error=self._response.error,
        )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        for event in self._events:
            yield event
        yield ATPResponse(
            task_id=request.task_id,
            status=self._response.status,
            artifacts=self._response.artifacts,
            metrics=self._response.metrics,
            error=self._response.error,
        )


@pytest.fixture
def test_definition() -> TestDefinition:
    """Create a test definition."""
    return TestDefinition(
        id="test-001",
        name="Sample Test",
        task=TaskDefinition(description="Test task"),
        constraints=Constraints(timeout_seconds=10),
    )


@pytest.fixture
def test_suite(test_definition: TestDefinition) -> TestSuite:
    """Create a test suite."""
    test2 = TestDefinition(
        id="test-002",
        name="Second Test",
        task=TaskDefinition(description="Second task"),
        constraints=Constraints(timeout_seconds=10),
    )
    return TestSuite(
        test_suite="test-suite",
        tests=[test_definition, test2],
        defaults=TestDefaults(runs_per_test=1),
    )


@pytest.fixture
def agent_configs() -> list[AgentConfig]:
    """Create agent configurations."""
    return [
        AgentConfig(
            name="agent-1",
            adapter=MockAdapter(
                name="agent-1",
                metrics=Metrics(
                    total_tokens=100,
                    input_tokens=50,
                    output_tokens=50,
                    total_steps=5,
                    wall_time_seconds=1.0,
                    cost_usd=0.01,
                ),
            ),
        ),
        AgentConfig(
            name="agent-2",
            adapter=MockAdapter(
                name="agent-2",
                metrics=Metrics(
                    total_tokens=200,
                    input_tokens=100,
                    output_tokens=100,
                    total_steps=10,
                    wall_time_seconds=2.0,
                    cost_usd=0.02,
                ),
            ),
        ),
    ]


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_create_agent_config(self) -> None:
        """AgentConfig can be created with required fields."""
        adapter = MockAdapter()
        config = AgentConfig(name="test-agent", adapter=adapter)

        assert config.name == "test-agent"
        assert config.adapter is adapter
        assert config.weight == 1.0

    def test_agent_config_with_weight(self) -> None:
        """AgentConfig can have custom weight."""
        adapter = MockAdapter()
        config = AgentConfig(name="test-agent", adapter=adapter, weight=2.0)

        assert config.weight == 2.0


class TestComparisonMetrics:
    """Tests for ComparisonMetrics."""

    def test_create_comparison_metrics(self) -> None:
        """ComparisonMetrics can be created with defaults."""
        metrics = ComparisonMetrics(agent_name="test-agent")

        assert metrics.agent_name == "test-agent"
        assert metrics.success_rate == 0.0
        assert metrics.avg_duration_seconds is None
        assert metrics.avg_tokens is None
        assert metrics.avg_steps is None
        assert metrics.avg_cost_usd is None
        assert metrics.total_tests == 0
        assert metrics.passed_tests == 0
        assert metrics.failed_tests == 0

    def test_comparison_metrics_with_values(self) -> None:
        """ComparisonMetrics can have all values set."""
        metrics = ComparisonMetrics(
            agent_name="test-agent",
            success_rate=0.9,
            avg_duration_seconds=1.5,
            avg_tokens=150.0,
            avg_steps=7.5,
            avg_cost_usd=0.015,
            total_tests=10,
            passed_tests=9,
            failed_tests=1,
        )

        assert metrics.success_rate == 0.9
        assert metrics.avg_duration_seconds == 1.5
        assert metrics.avg_tokens == 150.0
        assert metrics.avg_steps == 7.5
        assert metrics.avg_cost_usd == 0.015
        assert metrics.total_tests == 10
        assert metrics.passed_tests == 9
        assert metrics.failed_tests == 1


class TestAgentRanking:
    """Tests for AgentRanking."""

    def test_create_agent_ranking(self) -> None:
        """AgentRanking can be created."""
        ranking = AgentRanking(
            agent_name="agent-1",
            rank=1,
            score=0.95,
            metric=RankingMetric.SUCCESS_RATE,
        )

        assert ranking.agent_name == "agent-1"
        assert ranking.rank == 1
        assert ranking.score == 0.95
        assert ranking.metric == RankingMetric.SUCCESS_RATE


class TestMultiAgentTestResult:
    """Tests for MultiAgentTestResult."""

    def test_empty_result(self, test_definition: TestDefinition) -> None:
        """Empty result has correct defaults."""
        result = MultiAgentTestResult(test=test_definition)

        assert result.test is test_definition
        assert result.agent_results == []
        assert result.mode == MultiAgentMode.COMPARISON
        assert result.winner is None
        assert result.rankings == []
        assert result.all_succeeded is False
        assert result.any_succeeded is False

    def test_result_with_agent_results(self, test_definition: TestDefinition) -> None:
        """Result with agent results has correct properties."""
        from atp.runner.models import TestResult

        agent_result_1 = AgentTestResult(
            agent_name="agent-1",
            test_result=TestResult(test=test_definition),
        )
        # Make it successful by adding a successful run
        from datetime import datetime

        from atp.runner.models import RunResult

        agent_result_1.test_result.runs = [
            RunResult(
                test_id=test_definition.id,
                run_number=1,
                response=ATPResponse(
                    task_id="test-task",
                    status=ResponseStatus.COMPLETED,
                ),
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        ]

        result = MultiAgentTestResult(
            test=test_definition,
            agent_results=[agent_result_1],
        )

        assert result.any_succeeded is True
        assert result.all_succeeded is True
        assert result.agent_names == ["agent-1"]
        assert result.get_result_for_agent("agent-1") is agent_result_1
        assert result.get_result_for_agent("nonexistent") is None


class TestMultiAgentSuiteResult:
    """Tests for MultiAgentSuiteResult."""

    def test_empty_suite_result(self) -> None:
        """Empty suite result has correct defaults."""
        result = MultiAgentSuiteResult(
            suite_name="test-suite",
            mode=MultiAgentMode.COMPARISON,
            agents=["agent-1", "agent-2"],
        )

        assert result.suite_name == "test-suite"
        assert result.mode == MultiAgentMode.COMPARISON
        assert result.agents == ["agent-1", "agent-2"]
        assert result.test_results == []
        assert result.agent_suite_results == {}
        assert result.comparison_metrics == []
        assert result.overall_rankings == []
        assert result.total_tests == 0
        assert result.best_agent is None

    def test_suite_result_with_rankings(self) -> None:
        """Suite result with rankings returns best agent."""
        result = MultiAgentSuiteResult(
            suite_name="test-suite",
            mode=MultiAgentMode.COMPARISON,
            agents=["agent-1", "agent-2"],
            overall_rankings=[
                AgentRanking(
                    agent_name="agent-1",
                    rank=1,
                    score=0.95,
                    metric=RankingMetric.SUCCESS_RATE,
                ),
                AgentRanking(
                    agent_name="agent-2",
                    rank=2,
                    score=0.85,
                    metric=RankingMetric.SUCCESS_RATE,
                ),
            ],
        )

        assert result.best_agent == "agent-1"


class TestMultiAgentOrchestratorInit:
    """Tests for MultiAgentOrchestrator initialization."""

    def test_init_with_defaults(self, agent_configs: list[AgentConfig]) -> None:
        """Orchestrator initializes with defaults."""
        orchestrator = MultiAgentOrchestrator(agents=agent_configs)

        assert orchestrator.agents == agent_configs
        assert orchestrator.mode == MultiAgentMode.COMPARISON
        assert orchestrator.runs_per_test == 1
        assert orchestrator.parallel_agents is True
        assert orchestrator.max_parallel_agents == 5
        assert orchestrator.determine_winner is True
        assert orchestrator.ranking_metrics == [RankingMetric.SUCCESS_RATE]

    def test_init_with_options(self, agent_configs: list[AgentConfig]) -> None:
        """Orchestrator initializes with custom options."""
        callback = MagicMock()
        orchestrator = MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.COLLABORATION,
            progress_callback=callback,
            runs_per_test=3,
            parallel_agents=False,
            max_parallel_agents=2,
            ranking_metrics=[RankingMetric.SPEED, RankingMetric.COST],
            determine_winner=False,
        )

        assert orchestrator.mode == MultiAgentMode.COLLABORATION
        assert orchestrator.runs_per_test == 3
        assert orchestrator.parallel_agents is False
        assert orchestrator.max_parallel_agents == 2
        assert orchestrator.determine_winner is False
        assert orchestrator.ranking_metrics == [RankingMetric.SPEED, RankingMetric.COST]
        assert orchestrator.progress_callback is callback

    def test_init_requires_agents(self) -> None:
        """Orchestrator requires at least one agent."""
        with pytest.raises(ValueError, match="At least one agent is required"):
            MultiAgentOrchestrator(agents=[])


class TestMultiAgentOrchestratorRunSingleTest:
    """Tests for running a single test across multiple agents."""

    @pytest.mark.anyio
    async def test_successful_parallel_execution(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test successful parallel execution across agents."""
        async with MultiAgentOrchestrator(agents=agent_configs) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.test is test_definition
        assert len(result.agent_results) == 2
        assert result.any_succeeded is True
        assert result.all_succeeded is True
        assert result.agent_names == ["agent-1", "agent-2"]
        assert result.winner is not None

    @pytest.mark.anyio
    async def test_successful_sequential_execution(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test successful sequential execution across agents."""
        async with MultiAgentOrchestrator(
            agents=agent_configs,
            parallel_agents=False,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert len(result.agent_results) == 2
        assert result.all_succeeded is True

    @pytest.mark.anyio
    async def test_mixed_success_failure(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test with some agents succeeding and some failing."""
        agents = [
            AgentConfig(
                name="success-agent",
                adapter=MockAdapter(
                    name="success",
                    response=ATPResponse(
                        task_id="test",
                        status=ResponseStatus.COMPLETED,
                    ),
                ),
            ),
            AgentConfig(
                name="failure-agent",
                adapter=MockAdapter(
                    name="failure",
                    response=ATPResponse(
                        task_id="test",
                        status=ResponseStatus.FAILED,
                        error="Agent failed",
                    ),
                ),
            ),
        ]

        async with MultiAgentOrchestrator(agents=agents) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.any_succeeded is True
        assert result.all_succeeded is False
        assert result.winner == "success-agent"

    @pytest.mark.anyio
    async def test_agent_exception_handling(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test that agent exceptions are handled gracefully."""
        agents = [
            AgentConfig(
                name="good-agent",
                adapter=MockAdapter(name="good"),
            ),
            AgentConfig(
                name="bad-agent",
                adapter=MockAdapter(
                    name="bad",
                    error=RuntimeError("Simulated error"),
                ),
            ),
        ]

        async with MultiAgentOrchestrator(agents=agents) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert len(result.agent_results) == 2
        assert result.any_succeeded is True

        good_result = result.get_result_for_agent("good-agent")
        assert good_result is not None
        assert good_result.success is True

        bad_result = result.get_result_for_agent("bad-agent")
        assert bad_result is not None
        assert bad_result.success is False

    @pytest.mark.anyio
    async def test_winner_determination_disabled(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test with winner determination disabled."""
        async with MultiAgentOrchestrator(
            agents=agent_configs,
            determine_winner=False,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.winner is None
        assert result.rankings == []

    @pytest.mark.anyio
    async def test_parallel_respects_semaphore(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test that parallel execution respects semaphore limit."""
        max_concurrent = 0
        current_concurrent = 0

        class ConcurrencyTrackingAdapter(MockAdapter):
            async def stream_events(
                self, request: ATPRequest
            ) -> AsyncIterator[ATPEvent | ATPResponse]:
                nonlocal max_concurrent, current_concurrent
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
                await asyncio.sleep(0.05)
                current_concurrent -= 1
                yield ATPResponse(
                    task_id=request.task_id, status=ResponseStatus.COMPLETED
                )

        agents = [
            AgentConfig(
                name=f"agent-{i}",
                adapter=ConcurrencyTrackingAdapter(name=f"agent-{i}"),
            )
            for i in range(5)
        ]

        async with MultiAgentOrchestrator(
            agents=agents,
            max_parallel_agents=2,
        ) as orchestrator:
            await orchestrator.run_single_test(test_definition)

        assert max_concurrent <= 2


class TestMultiAgentOrchestratorRunSuite:
    """Tests for running a suite across multiple agents."""

    @pytest.mark.anyio
    async def test_suite_execution(
        self,
        test_suite: TestSuite,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test running a complete suite across agents."""
        async with MultiAgentOrchestrator(agents=agent_configs) as orchestrator:
            result = await orchestrator.run_suite(test_suite)

        assert result.suite_name == "test-suite"
        assert result.mode == MultiAgentMode.COMPARISON
        assert result.agents == ["agent-1", "agent-2"]
        assert result.total_tests == 2
        assert len(result.test_results) == 2
        assert len(result.agent_suite_results) == 2
        assert len(result.comparison_metrics) == 2
        assert result.best_agent is not None

    @pytest.mark.anyio
    async def test_suite_comparison_metrics(
        self,
        test_suite: TestSuite,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that comparison metrics are calculated correctly."""
        async with MultiAgentOrchestrator(agents=agent_configs) as orchestrator:
            result = await orchestrator.run_suite(test_suite)

        for metrics in result.comparison_metrics:
            assert metrics.agent_name in ["agent-1", "agent-2"]
            assert metrics.total_tests == 2
            # Both agents succeed in this test
            assert metrics.success_rate == 1.0

    @pytest.mark.anyio
    async def test_suite_rankings(
        self,
        test_suite: TestSuite,
    ) -> None:
        """Test that rankings are generated correctly."""
        # Create agents with different metrics (cost difference is clear)
        agents = [
            AgentConfig(
                name="cheap-agent",
                adapter=MockAdapter(
                    name="cheap",
                    metrics=Metrics(
                        total_tokens=50,
                        cost_usd=0.01,
                    ),
                ),
            ),
            AgentConfig(
                name="expensive-agent",
                adapter=MockAdapter(
                    name="expensive",
                    metrics=Metrics(
                        total_tokens=100,
                        cost_usd=0.05,
                    ),
                ),
            ),
        ]

        async with MultiAgentOrchestrator(
            agents=agents,
            ranking_metrics=[RankingMetric.COST, RankingMetric.TOKENS],
        ) as orchestrator:
            result = await orchestrator.run_suite(test_suite)

        # Should have rankings for both metrics
        assert len(result.overall_rankings) == 4  # 2 agents x 2 metrics

        # Check cost rankings (lower cost is better)
        cost_rankings = [
            r for r in result.overall_rankings if r.metric == RankingMetric.COST
        ]
        assert len(cost_rankings) == 2
        # Cheap agent should be ranked 1st
        cheap_cost = next(r for r in cost_rankings if r.agent_name == "cheap-agent")
        assert cheap_cost.rank == 1

    @pytest.mark.anyio
    async def test_get_agent_suite_result(
        self,
        test_suite: TestSuite,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test getting suite result for a specific agent."""
        async with MultiAgentOrchestrator(agents=agent_configs) as orchestrator:
            result = await orchestrator.run_suite(test_suite)

        agent_1_result = result.get_agent_suite_result("agent-1")
        assert agent_1_result is not None
        assert agent_1_result.agent_name == "agent-1"
        assert agent_1_result.total_tests == 2

        nonexistent = result.get_agent_suite_result("nonexistent")
        assert nonexistent is None


class TestProgressCallback:
    """Tests for progress callback."""

    @pytest.mark.anyio
    async def test_progress_events_for_test(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that progress events are emitted for test execution."""
        events: list[ProgressEvent] = []

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            progress_callback=events.append,
        ) as orchestrator:
            await orchestrator.run_single_test(test_definition)

        event_types = [e.event_type for e in events]
        # Should have test started and completed/failed events
        assert ProgressEventType.TEST_STARTED in event_types
        assert (
            ProgressEventType.TEST_COMPLETED in event_types
            or ProgressEventType.TEST_FAILED in event_types
        )

    @pytest.mark.anyio
    async def test_progress_events_for_suite(
        self,
        test_suite: TestSuite,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that progress events are emitted for suite execution."""
        events: list[ProgressEvent] = []

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            progress_callback=events.append,
        ) as orchestrator:
            await orchestrator.run_suite(test_suite)

        event_types = [e.event_type for e in events]
        assert ProgressEventType.SUITE_STARTED in event_types
        assert ProgressEventType.SUITE_COMPLETED in event_types

    @pytest.mark.anyio
    async def test_callback_exception_handled(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that callback exceptions are handled gracefully."""

        def bad_callback(event: ProgressEvent) -> None:
            raise RuntimeError("Callback error")

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            progress_callback=bad_callback,
        ) as orchestrator:
            # Should not raise
            result = await orchestrator.run_single_test(test_definition)

        assert result.all_succeeded is True


class TestRankingMetrics:
    """Tests for ranking metrics."""

    @pytest.mark.anyio
    async def test_rank_by_success_rate(
        self,
        test_suite: TestSuite,
    ) -> None:
        """Test ranking by success rate."""
        agents = [
            AgentConfig(
                name="winner",
                adapter=MockAdapter(
                    name="winner",
                    response=ATPResponse(
                        task_id="test", status=ResponseStatus.COMPLETED
                    ),
                ),
            ),
            AgentConfig(
                name="loser",
                adapter=MockAdapter(
                    name="loser",
                    response=ATPResponse(
                        task_id="test",
                        status=ResponseStatus.FAILED,
                        error="Failed",
                    ),
                ),
            ),
        ]

        async with MultiAgentOrchestrator(
            agents=agents,
            ranking_metrics=[RankingMetric.SUCCESS_RATE],
        ) as orchestrator:
            result = await orchestrator.run_suite(test_suite)

        # Winner should be ranked first
        success_rankings = [
            r for r in result.overall_rankings if r.metric == RankingMetric.SUCCESS_RATE
        ]
        winner_ranking = next(r for r in success_rankings if r.agent_name == "winner")
        assert winner_ranking.rank == 1
        assert winner_ranking.score == 1.0

    @pytest.mark.anyio
    async def test_rank_by_cost(
        self,
        test_suite: TestSuite,
    ) -> None:
        """Test ranking by cost (lower is better)."""
        agents = [
            AgentConfig(
                name="cheap",
                adapter=MockAdapter(
                    name="cheap",
                    metrics=Metrics(cost_usd=0.01),
                ),
            ),
            AgentConfig(
                name="expensive",
                adapter=MockAdapter(
                    name="expensive",
                    metrics=Metrics(cost_usd=0.10),
                ),
            ),
        ]

        async with MultiAgentOrchestrator(
            agents=agents,
            ranking_metrics=[RankingMetric.COST],
        ) as orchestrator:
            result = await orchestrator.run_suite(test_suite)

        cost_rankings = [
            r for r in result.overall_rankings if r.metric == RankingMetric.COST
        ]
        cheap_ranking = next(r for r in cost_rankings if r.agent_name == "cheap")
        assert cheap_ranking.rank == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.anyio
    async def test_run_comparison(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test run_comparison convenience function."""
        result = await run_comparison(
            agents=agent_configs,
            test=test_definition,
        )

        assert isinstance(result, MultiAgentTestResult)
        assert result.mode == MultiAgentMode.COMPARISON
        assert result.all_succeeded is True

    @pytest.mark.anyio
    async def test_run_comparison_with_options(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test run_comparison with options."""
        events: list[ProgressEvent] = []

        result = await run_comparison(
            agents=agent_configs,
            test=test_definition,
            progress_callback=events.append,
            runs=2,
            parallel=True,
            ranking_metrics=[RankingMetric.SPEED],
        )

        assert result.all_succeeded is True
        assert len(events) > 0

    @pytest.mark.anyio
    async def test_run_suite_comparison(
        self,
        test_suite: TestSuite,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test run_suite_comparison convenience function."""
        result = await run_suite_comparison(
            agents=agent_configs,
            suite=test_suite,
        )

        assert isinstance(result, MultiAgentSuiteResult)
        assert result.mode == MultiAgentMode.COMPARISON
        assert result.total_tests == 2
        assert result.best_agent is not None


class TestContextManager:
    """Tests for context manager functionality."""

    @pytest.mark.anyio
    async def test_cleanup_on_exit(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that cleanup is called on context exit."""
        cleanup_called = False
        original_adapter = agent_configs[0].adapter

        class TrackingAdapter(MockAdapter):
            async def cleanup(self) -> None:
                nonlocal cleanup_called
                cleanup_called = True
                await super().cleanup()

        agent_configs[0].adapter = TrackingAdapter(name="tracking")

        async with MultiAgentOrchestrator(agents=agent_configs) as orchestrator:
            await orchestrator.run_single_test(test_definition)

        assert cleanup_called is True

        # Restore original adapter
        agent_configs[0].adapter = original_adapter

    @pytest.mark.anyio
    async def test_cleanup_on_exception(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test that cleanup is called even on exception."""
        cleanup_called = False

        class CleanupTrackingAdapter(MockAdapter):
            async def cleanup(self) -> None:
                nonlocal cleanup_called
                cleanup_called = True

        adapter = CleanupTrackingAdapter(name="test")
        agents = [
            AgentConfig(
                name="test",
                adapter=adapter,
            )
        ]

        try:
            async with MultiAgentOrchestrator(agents=agents) as orchestrator:
                # Run a test first so the orchestrator has something to clean up
                await orchestrator.run_single_test(test_definition)
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass

        assert cleanup_called is True


class TestMultiAgentMode:
    """Tests for MultiAgentMode enum."""

    def test_mode_values(self) -> None:
        """Test mode enum values."""
        assert MultiAgentMode.COMPARISON.value == "comparison"
        assert MultiAgentMode.COLLABORATION.value == "collaboration"
        assert MultiAgentMode.HANDOFF.value == "handoff"


class TestRankingMetricEnum:
    """Tests for RankingMetric enum."""

    def test_ranking_metric_values(self) -> None:
        """Test ranking metric enum values."""
        assert RankingMetric.QUALITY.value == "quality"
        assert RankingMetric.SPEED.value == "speed"
        assert RankingMetric.COST.value == "cost"
        assert RankingMetric.SUCCESS_RATE.value == "success_rate"
        assert RankingMetric.TOKENS.value == "tokens"
        assert RankingMetric.STEPS.value == "steps"


# ===========================================================================
# Collaboration Mode Tests
# ===========================================================================


class TestCollaborationMessage:
    """Tests for CollaborationMessage model."""

    def test_create_message(self) -> None:
        """CollaborationMessage can be created with required fields."""
        msg = CollaborationMessage(
            message_id="msg-001",
            message_type=CollaborationMessageType.TASK_ASSIGNMENT,
            from_agent="orchestrator",
            to_agent="agent-1",
            content={"task": "do something"},
        )

        assert msg.message_id == "msg-001"
        assert msg.message_type == CollaborationMessageType.TASK_ASSIGNMENT
        assert msg.from_agent == "orchestrator"
        assert msg.to_agent == "agent-1"
        assert msg.content == {"task": "do something"}
        assert msg.turn_number == 0
        assert msg.in_reply_to is None

    def test_broadcast_message(self) -> None:
        """CollaborationMessage can be a broadcast (to_agent=None)."""
        msg = CollaborationMessage(
            message_id="msg-002",
            message_type=CollaborationMessageType.STATUS_UPDATE,
            from_agent="agent-1",
            content={"status": "working"},
        )

        assert msg.to_agent is None

    def test_message_types(self) -> None:
        """All message types are accessible."""
        assert CollaborationMessageType.TASK_ASSIGNMENT.value == "task_assignment"
        assert CollaborationMessageType.RESULT.value == "result"
        assert CollaborationMessageType.QUERY.value == "query"
        assert CollaborationMessageType.RESPONSE.value == "response"
        assert CollaborationMessageType.HANDOFF.value == "handoff"
        assert CollaborationMessageType.STATUS_UPDATE.value == "status_update"


class TestSharedContext:
    """Tests for SharedContext model."""

    def test_create_shared_context(self) -> None:
        """SharedContext can be created with required fields."""
        ctx = SharedContext(task_description="Test task")

        assert ctx.task_description == "Test task"
        assert ctx.artifacts == {}
        assert ctx.variables == {}
        assert ctx.messages == []
        assert ctx.current_turn == 0
        assert ctx.completed_subtasks == []

    def test_add_message(self) -> None:
        """SharedContext.add_message adds messages to history."""
        ctx = SharedContext(task_description="Test task")
        msg = CollaborationMessage(
            message_id="msg-001",
            message_type=CollaborationMessageType.RESULT,
            from_agent="agent-1",
            content={"result": "done"},
        )

        ctx.add_message(msg)

        assert len(ctx.messages) == 1
        assert ctx.messages[0] is msg

    def test_get_messages_for_agent(self) -> None:
        """SharedContext.get_messages_for_agent filters correctly."""
        ctx = SharedContext(task_description="Test task")

        # Add messages
        msg1 = CollaborationMessage(
            message_id="msg-001",
            message_type=CollaborationMessageType.TASK_ASSIGNMENT,
            from_agent="orchestrator",
            to_agent="agent-1",
            content={},
        )
        msg2 = CollaborationMessage(
            message_id="msg-002",
            message_type=CollaborationMessageType.RESULT,
            from_agent="agent-1",
            content={},  # broadcast
        )
        msg3 = CollaborationMessage(
            message_id="msg-003",
            message_type=CollaborationMessageType.TASK_ASSIGNMENT,
            from_agent="orchestrator",
            to_agent="agent-2",
            content={},
        )

        ctx.add_message(msg1)
        ctx.add_message(msg2)
        ctx.add_message(msg3)

        agent1_msgs = ctx.get_messages_for_agent("agent-1")

        # Should get msg1 (to agent-1), msg2 (from agent-1), and broadcast
        assert len(agent1_msgs) == 2
        assert msg1 in agent1_msgs
        assert msg2 in agent1_msgs

    def test_get_messages_by_turn(self) -> None:
        """SharedContext.get_messages_by_turn filters by turn number."""
        ctx = SharedContext(task_description="Test task")

        msg1 = CollaborationMessage(
            message_id="msg-001",
            message_type=CollaborationMessageType.RESULT,
            from_agent="agent-1",
            content={},
            turn_number=1,
        )
        msg2 = CollaborationMessage(
            message_id="msg-002",
            message_type=CollaborationMessageType.RESULT,
            from_agent="agent-2",
            content={},
            turn_number=2,
        )

        ctx.add_message(msg1)
        ctx.add_message(msg2)

        turn1_msgs = ctx.get_messages_by_turn(1)

        assert len(turn1_msgs) == 1
        assert turn1_msgs[0] is msg1

    def test_set_and_get_artifact(self) -> None:
        """SharedContext artifact management works correctly."""
        ctx = SharedContext(task_description="Test task")

        ctx.set_artifact("output", {"data": "value"}, "agent-1")

        assert ctx.get_artifact("output") == {"data": "value"}
        assert ctx.artifacts["output"]["set_by"] == "agent-1"
        assert ctx.get_artifact("nonexistent") is None

    def test_set_and_get_variable(self) -> None:
        """SharedContext variable management works correctly."""
        ctx = SharedContext(task_description="Test task")

        ctx.set_variable("counter", 5)

        assert ctx.get_variable("counter") == 5
        assert ctx.get_variable("missing", default=0) == 0


class TestCollaborationConfig:
    """Tests for CollaborationConfig model."""

    def test_default_config(self) -> None:
        """CollaborationConfig has sensible defaults."""
        config = CollaborationConfig()

        assert config.max_turns == 10
        assert config.turn_timeout_seconds == 60.0
        assert config.require_consensus is False
        assert config.allow_parallel_turns is False
        assert config.coordinator_agent is None
        assert config.termination_condition == "all_complete"

    def test_custom_config(self) -> None:
        """CollaborationConfig can be customized."""
        config = CollaborationConfig(
            max_turns=5,
            turn_timeout_seconds=30.0,
            require_consensus=True,
            allow_parallel_turns=True,
            coordinator_agent="coordinator",
            termination_condition="consensus",
        )

        assert config.max_turns == 5
        assert config.turn_timeout_seconds == 30.0
        assert config.require_consensus is True
        assert config.allow_parallel_turns is True
        assert config.coordinator_agent == "coordinator"
        assert config.termination_condition == "consensus"


class TestCollaborationMetrics:
    """Tests for CollaborationMetrics model."""

    def test_default_metrics(self) -> None:
        """CollaborationMetrics has correct defaults."""
        metrics = CollaborationMetrics()

        assert metrics.total_turns == 0
        assert metrics.total_messages == 0
        assert metrics.messages_per_agent == {}
        assert metrics.turns_per_agent == {}
        assert metrics.avg_turn_duration_seconds is None
        assert metrics.consensus_reached is False
        assert metrics.termination_reason is None
        assert metrics.agent_contributions == {}
        assert metrics.total_tokens == 0
        assert metrics.total_cost_usd == 0.0

    def test_custom_metrics(self) -> None:
        """CollaborationMetrics can store all metric values."""
        metrics = CollaborationMetrics(
            total_turns=5,
            total_messages=10,
            messages_per_agent={"agent-1": 5, "agent-2": 5},
            turns_per_agent={"agent-1": 3, "agent-2": 2},
            avg_turn_duration_seconds=2.5,
            consensus_reached=True,
            termination_reason="consensus_reached",
            agent_contributions={"agent-1": 0.6, "agent-2": 0.4},
            total_tokens=500,
            total_cost_usd=0.05,
        )

        assert metrics.total_turns == 5
        assert metrics.total_messages == 10
        assert metrics.messages_per_agent == {"agent-1": 5, "agent-2": 5}
        assert metrics.consensus_reached is True
        assert metrics.total_tokens == 500
        assert metrics.total_cost_usd == 0.05


class TestCollaborationTurnResult:
    """Tests for CollaborationTurnResult model."""

    def test_successful_turn(self) -> None:
        """CollaborationTurnResult tracks successful turn."""
        response = ATPResponse(
            task_id="test-task",
            status=ResponseStatus.COMPLETED,
        )
        turn_result = CollaborationTurnResult(
            turn_number=1,
            agent_name="agent-1",
            response=response,
        )

        assert turn_result.turn_number == 1
        assert turn_result.agent_name == "agent-1"
        assert turn_result.success is True
        assert turn_result.error is None

    def test_failed_turn(self) -> None:
        """CollaborationTurnResult tracks failed turn."""
        turn_result = CollaborationTurnResult(
            turn_number=1,
            agent_name="agent-1",
            error="Something went wrong",
        )

        assert turn_result.success is False
        assert turn_result.error == "Something went wrong"

    def test_turn_with_messages(self) -> None:
        """CollaborationTurnResult tracks messages sent and received."""
        msg = CollaborationMessage(
            message_id="msg-001",
            message_type=CollaborationMessageType.RESULT,
            from_agent="agent-1",
            content={"result": "done"},
        )
        turn_result = CollaborationTurnResult(
            turn_number=1,
            agent_name="agent-1",
            messages_sent=[msg],
        )

        assert len(turn_result.messages_sent) == 1
        assert turn_result.messages_sent[0] is msg


class TestCollaborationResult:
    """Tests for CollaborationResult model."""

    def test_empty_result(self, test_definition: TestDefinition) -> None:
        """CollaborationResult with no turns."""
        shared_ctx = SharedContext(task_description="Test task")
        result = CollaborationResult(
            test=test_definition,
            shared_context=shared_ctx,
        )

        assert result.test is test_definition
        assert result.turn_results == []
        assert result.total_turns == 0
        # Empty result is successful (no failures)
        assert result.success is True

    def test_successful_result(self, test_definition: TestDefinition) -> None:
        """CollaborationResult with successful turns."""
        shared_ctx = SharedContext(task_description="Test task")
        turn = CollaborationTurnResult(
            turn_number=1,
            agent_name="agent-1",
            response=ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
            ),
        )

        result = CollaborationResult(
            test=test_definition,
            shared_context=shared_ctx,
            turn_results=[turn],
            final_response=ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
            ),
        )

        assert result.success is True
        assert result.total_turns == 1

    def test_result_with_error(self, test_definition: TestDefinition) -> None:
        """CollaborationResult with error."""
        shared_ctx = SharedContext(task_description="Test task")
        result = CollaborationResult(
            test=test_definition,
            shared_context=shared_ctx,
            error="Collaboration failed",
        )

        assert result.success is False
        assert result.error == "Collaboration failed"

    def test_get_turns_for_agent(self, test_definition: TestDefinition) -> None:
        """CollaborationResult.get_turns_for_agent filters correctly."""
        shared_ctx = SharedContext(task_description="Test task")
        turn1 = CollaborationTurnResult(turn_number=1, agent_name="agent-1")
        turn2 = CollaborationTurnResult(turn_number=2, agent_name="agent-2")
        turn3 = CollaborationTurnResult(turn_number=3, agent_name="agent-1")

        result = CollaborationResult(
            test=test_definition,
            shared_context=shared_ctx,
            turn_results=[turn1, turn2, turn3],
        )

        agent1_turns = result.get_turns_for_agent("agent-1")

        assert len(agent1_turns) == 2
        assert turn1 in agent1_turns
        assert turn3 in agent1_turns


class TestMultiAgentTestResultCollaboration:
    """Tests for MultiAgentTestResult in collaboration mode."""

    def test_collaboration_mode_result(self, test_definition: TestDefinition) -> None:
        """MultiAgentTestResult supports collaboration mode."""
        shared_ctx = SharedContext(task_description="Test task")
        collab_result = CollaborationResult(
            test=test_definition,
            shared_context=shared_ctx,
            final_response=ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
            ),
        )

        result = MultiAgentTestResult(
            test=test_definition,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_result=collab_result,
        )

        assert result.mode == MultiAgentMode.COLLABORATION
        assert result.collaboration_result is collab_result
        assert result.all_succeeded is True
        assert result.any_succeeded is True


class TestMultiAgentOrchestratorCollaboration:
    """Tests for MultiAgentOrchestrator in collaboration mode."""

    @pytest.mark.anyio
    async def test_collaboration_mode_execution(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test basic collaboration mode execution."""
        collab_config = CollaborationConfig(
            max_turns=3,
            termination_condition="max_turns",
        )

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.mode == MultiAgentMode.COLLABORATION
        assert result.collaboration_result is not None
        assert result.collaboration_result.total_turns == 3
        assert result.all_succeeded is True

    @pytest.mark.anyio
    async def test_collaboration_with_any_complete_termination(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test collaboration terminates on first agent completion."""
        collab_config = CollaborationConfig(
            max_turns=10,
            termination_condition="any_complete",
        )

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.collaboration_result is not None
        # Should terminate after first successful turn
        assert result.collaboration_result.total_turns >= 1
        assert (
            result.collaboration_result.collaboration_metrics.termination_reason
            == "agent_completed"
        )

    @pytest.mark.anyio
    async def test_collaboration_with_parallel_turns(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test collaboration with parallel turn execution."""
        collab_config = CollaborationConfig(
            max_turns=2,
            allow_parallel_turns=True,
            termination_condition="max_turns",
        )

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.collaboration_result is not None
        # With 2 agents and 2 turns with parallel execution,
        # we get 2 agents * 2 turns = 4 turn results
        assert result.collaboration_result.total_turns == 4

    @pytest.mark.anyio
    async def test_collaboration_with_coordinator(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test collaboration with a designated coordinator agent."""
        collab_config = CollaborationConfig(
            max_turns=4,
            coordinator_agent="agent-1",
            termination_condition="max_turns",
        )

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.collaboration_result is not None
        # Coordinator should take odd turns (1, 3), others take even turns (2, 4)
        turn_agents = [tr.agent_name for tr in result.collaboration_result.turn_results]
        assert turn_agents[0] == "agent-1"  # Turn 1: coordinator
        assert turn_agents[1] == "agent-2"  # Turn 2: other agent
        assert turn_agents[2] == "agent-1"  # Turn 3: coordinator

    @pytest.mark.anyio
    async def test_collaboration_shared_context(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that shared context is populated during collaboration."""
        collab_config = CollaborationConfig(
            max_turns=2,
            termination_condition="max_turns",
        )

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.collaboration_result is not None
        shared_ctx = result.collaboration_result.shared_context

        # Should have initial task assignment messages + result messages
        assert len(shared_ctx.messages) > 0
        assert shared_ctx.current_turn == 2

    @pytest.mark.anyio
    async def test_collaboration_metrics_calculation(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test that collaboration metrics are calculated correctly."""
        # Create agents with specific metrics
        agents = [
            AgentConfig(
                name="agent-1",
                adapter=MockAdapter(
                    name="agent-1",
                    metrics=Metrics(
                        total_tokens=100,
                        cost_usd=0.01,
                    ),
                ),
            ),
            AgentConfig(
                name="agent-2",
                adapter=MockAdapter(
                    name="agent-2",
                    metrics=Metrics(
                        total_tokens=200,
                        cost_usd=0.02,
                    ),
                ),
            ),
        ]

        collab_config = CollaborationConfig(
            max_turns=4,
            termination_condition="max_turns",
        )

        async with MultiAgentOrchestrator(
            agents=agents,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.collaboration_result is not None
        metrics = result.collaboration_result.collaboration_metrics

        assert metrics.total_turns == 4
        assert metrics.termination_reason == "max_turns_reached"
        # Each agent should have participated twice (turns 1,3 and 2,4)
        assert metrics.turns_per_agent.get("agent-1", 0) == 2
        assert metrics.turns_per_agent.get("agent-2", 0) == 2
        # Check aggregated tokens (100 * 2 + 200 * 2 = 600)
        assert metrics.total_tokens == 600
        # Check aggregated cost (0.01 * 2 + 0.02 * 2 = 0.06)
        assert abs(metrics.total_cost_usd - 0.06) < 0.001

    @pytest.mark.anyio
    async def test_collaboration_turn_timeout(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test that turn timeout is enforced."""
        # Create a slow adapter
        slow_adapter = MockAdapter(name="slow", delay=2.0)
        agents = [
            AgentConfig(name="slow-agent", adapter=slow_adapter),
        ]

        collab_config = CollaborationConfig(
            max_turns=1,
            turn_timeout_seconds=0.1,  # Very short timeout
            termination_condition="max_turns",
        )

        async with MultiAgentOrchestrator(
            agents=agents,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.collaboration_result is not None
        # Turn should have timed out
        turn_result = result.collaboration_result.turn_results[0]
        assert turn_result.error is not None
        assert "timeout" in turn_result.error.lower()

    @pytest.mark.anyio
    async def test_collaboration_agent_error_handling(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test that agent errors during collaboration are handled."""
        agents = [
            AgentConfig(
                name="error-agent",
                adapter=MockAdapter(
                    name="error",
                    error=RuntimeError("Agent crashed"),
                ),
            ),
        ]

        collab_config = CollaborationConfig(
            max_turns=1,
            termination_condition="max_turns",
        )

        async with MultiAgentOrchestrator(
            agents=agents,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.collaboration_result is not None
        turn_result = result.collaboration_result.turn_results[0]
        assert turn_result.error is not None
        assert "crashed" in turn_result.error.lower()


class TestCollaborationProgressEvents:
    """Tests for progress events in collaboration mode."""

    @pytest.mark.anyio
    async def test_collaboration_progress_events(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that progress events are emitted during collaboration."""
        events: list[ProgressEvent] = []

        collab_config = CollaborationConfig(
            max_turns=2,
            termination_condition="max_turns",
        )

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
            progress_callback=events.append,
        ) as orchestrator:
            await orchestrator.run_single_test(test_definition)

        event_types = [e.event_type for e in events]

        # Should have test started, run started/completed for each turn
        assert ProgressEventType.TEST_STARTED in event_types
        assert ProgressEventType.RUN_STARTED in event_types
        assert ProgressEventType.RUN_COMPLETED in event_types
        assert (
            ProgressEventType.TEST_COMPLETED in event_types
            or ProgressEventType.TEST_FAILED in event_types
        )


class TestCollaborationConvenienceFunctions:
    """Tests for collaboration convenience functions."""

    @pytest.mark.anyio
    async def test_run_collaboration(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test run_collaboration convenience function."""
        collab_config = CollaborationConfig(
            max_turns=2,
            termination_condition="max_turns",
        )

        result = await run_collaboration(
            agents=agent_configs,
            test=test_definition,
            collaboration_config=collab_config,
        )

        assert isinstance(result, MultiAgentTestResult)
        assert result.mode == MultiAgentMode.COLLABORATION
        assert result.collaboration_result is not None
        assert result.collaboration_result.total_turns == 2

    @pytest.mark.anyio
    async def test_run_suite_collaboration(
        self,
        test_suite: TestSuite,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test run_suite_collaboration convenience function."""
        collab_config = CollaborationConfig(
            max_turns=2,
            termination_condition="max_turns",
        )

        result = await run_suite_collaboration(
            agents=agent_configs,
            suite=test_suite,
            collaboration_config=collab_config,
        )

        assert isinstance(result, MultiAgentSuiteResult)
        assert result.mode == MultiAgentMode.COLLABORATION
        assert result.total_tests == 2


class TestCollaborationConsensusTermination:
    """Tests for consensus-based termination."""

    @pytest.mark.anyio
    async def test_consensus_termination(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that consensus termination works."""
        collab_config = CollaborationConfig(
            max_turns=10,
            termination_condition="consensus",
        )

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.collaboration_result is not None
        # With 2 agents succeeding, consensus should be reached after 2 turns
        assert result.collaboration_result.total_turns >= len(agent_configs)
        assert (
            result.collaboration_result.collaboration_metrics.termination_reason
            == "consensus_reached"
        )
        assert result.collaboration_result.collaboration_metrics.consensus_reached


class TestCollaborationAllCompleteTermination:
    """Tests for all_complete termination condition."""

    @pytest.mark.anyio
    async def test_all_complete_termination(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that all_complete waits for all agents."""
        collab_config = CollaborationConfig(
            max_turns=10,
            termination_condition="all_complete",
        )

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.COLLABORATION,
            collaboration_config=collab_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.collaboration_result is not None
        # Should terminate once all agents have completed at least once
        assert result.collaboration_result.total_turns >= len(agent_configs)
        assert (
            result.collaboration_result.collaboration_metrics.termination_reason
            == "all_agents_completed"
        )


# ===========================================================================
# Handoff Mode Tests
# ===========================================================================


class TestHandoffConfig:
    """Tests for HandoffConfig model."""

    def test_default_config(self) -> None:
        """HandoffConfig has sensible defaults."""
        from atp.runner.multi_agent import (
            ContextAccumulationMode,
            HandoffConfig,
            HandoffTrigger,
        )

        config = HandoffConfig()

        assert config.handoff_trigger == HandoffTrigger.ALWAYS
        assert config.context_accumulation == ContextAccumulationMode.APPEND
        assert config.max_context_size is None
        assert config.allow_backtrack is False
        assert config.final_agent_decides is True
        assert config.agent_timeout_seconds == 120.0
        assert config.continue_on_failure is False

    def test_custom_config(self) -> None:
        """HandoffConfig can be customized."""
        from atp.runner.multi_agent import (
            ContextAccumulationMode,
            HandoffConfig,
            HandoffTrigger,
        )

        config = HandoffConfig(
            handoff_trigger=HandoffTrigger.ON_SUCCESS,
            context_accumulation=ContextAccumulationMode.MERGE,
            max_context_size=5000,
            allow_backtrack=True,
            final_agent_decides=False,
            agent_timeout_seconds=60.0,
            continue_on_failure=True,
        )

        assert config.handoff_trigger == HandoffTrigger.ON_SUCCESS
        assert config.context_accumulation == ContextAccumulationMode.MERGE
        assert config.max_context_size == 5000
        assert config.allow_backtrack is True
        assert config.final_agent_decides is False
        assert config.agent_timeout_seconds == 60.0
        assert config.continue_on_failure is True


class TestHandoffContext:
    """Tests for HandoffContext model."""

    def test_create_context(self) -> None:
        """HandoffContext can be created with required fields."""
        from atp.runner.multi_agent import HandoffContext

        ctx = HandoffContext(original_task="Test task")

        assert ctx.original_task == "Test task"
        assert ctx.previous_outputs == []
        assert ctx.agent_sequence == []
        assert ctx.current_position == 0
        assert ctx.accumulated_artifacts == {}
        assert ctx.handoff_notes == []

    def test_add_output(self) -> None:
        """HandoffContext.add_output adds outputs correctly."""
        from atp.runner.multi_agent import AgentHandoffOutput, HandoffContext

        ctx = HandoffContext(
            original_task="Test task",
            agent_sequence=["agent-1", "agent-2"],
        )

        output = AgentHandoffOutput(
            agent_name="agent-1",
            sequence_position=0,
            artifacts=[{"name": "result", "data": "value"}],
        )

        ctx.add_output(output)

        assert len(ctx.previous_outputs) == 1
        assert ctx.previous_outputs[0] is output
        assert "result" in ctx.accumulated_artifacts

    def test_get_last_output(self) -> None:
        """HandoffContext.get_last_output returns most recent output."""
        from atp.runner.multi_agent import AgentHandoffOutput, HandoffContext

        ctx = HandoffContext(original_task="Test task")

        assert ctx.get_last_output() is None

        output1 = AgentHandoffOutput(agent_name="agent-1", sequence_position=0)
        output2 = AgentHandoffOutput(agent_name="agent-2", sequence_position=1)

        ctx.previous_outputs = [output1, output2]

        assert ctx.get_last_output() is output2

    def test_get_successful_outputs(self) -> None:
        """HandoffContext.get_successful_outputs filters correctly."""
        from atp.runner.multi_agent import AgentHandoffOutput, HandoffContext

        ctx = HandoffContext(original_task="Test task")

        output1 = AgentHandoffOutput(agent_name="agent-1", sequence_position=0)
        output2 = AgentHandoffOutput(
            agent_name="agent-2", sequence_position=1, error="Failed"
        )
        output3 = AgentHandoffOutput(agent_name="agent-3", sequence_position=2)

        ctx.previous_outputs = [output1, output2, output3]

        successful = ctx.get_successful_outputs()

        assert len(successful) == 2
        assert output1 in successful
        assert output3 in successful
        assert output2 not in successful

    def test_add_handoff_note(self) -> None:
        """HandoffContext.add_handoff_note adds notes correctly."""
        from atp.runner.multi_agent import HandoffContext

        ctx = HandoffContext(original_task="Test task")

        ctx.add_handoff_note("First note")
        ctx.add_handoff_note("Second note")

        assert len(ctx.handoff_notes) == 2
        assert "First note" in ctx.handoff_notes
        assert "Second note" in ctx.handoff_notes


class TestHandoffMetrics:
    """Tests for HandoffMetrics model."""

    def test_default_metrics(self) -> None:
        """HandoffMetrics has correct defaults."""
        from atp.runner.multi_agent import HandoffMetrics

        metrics = HandoffMetrics()

        assert metrics.total_agents == 0
        assert metrics.agents_executed == 0
        assert metrics.successful_handoffs == 0
        assert metrics.failed_handoffs == 0
        assert metrics.total_duration_seconds == 0.0
        assert metrics.duration_per_agent == {}
        assert metrics.agent_contributions == {}
        assert metrics.total_tokens == 0
        assert metrics.tokens_per_agent == {}
        assert metrics.total_cost_usd == 0.0
        assert metrics.cost_per_agent == {}
        assert metrics.termination_reason is None
        assert metrics.final_agent is None

    def test_custom_metrics(self) -> None:
        """HandoffMetrics can store all metric values."""
        from atp.runner.multi_agent import HandoffMetrics

        metrics = HandoffMetrics(
            total_agents=3,
            agents_executed=2,
            successful_handoffs=1,
            failed_handoffs=0,
            total_duration_seconds=5.5,
            duration_per_agent={"agent-1": 2.0, "agent-2": 3.5},
            agent_contributions={"agent-1": 0.4, "agent-2": 0.6},
            total_tokens=500,
            tokens_per_agent={"agent-1": 200, "agent-2": 300},
            total_cost_usd=0.05,
            cost_per_agent={"agent-1": 0.02, "agent-2": 0.03},
            termination_reason="chain_completed",
            final_agent="agent-2",
        )

        assert metrics.total_agents == 3
        assert metrics.agents_executed == 2
        assert metrics.successful_handoffs == 1
        assert metrics.total_tokens == 500
        assert metrics.final_agent == "agent-2"


class TestHandoffTurnResult:
    """Tests for HandoffTurnResult model."""

    def test_successful_turn(self) -> None:
        """HandoffTurnResult tracks successful turn."""
        from atp.runner.multi_agent import HandoffTurnResult

        response = ATPResponse(
            task_id="test-task",
            status=ResponseStatus.COMPLETED,
        )
        turn_result = HandoffTurnResult(
            agent_name="agent-1",
            sequence_position=0,
            response=response,
        )

        assert turn_result.agent_name == "agent-1"
        assert turn_result.sequence_position == 0
        assert turn_result.success is True
        assert turn_result.error is None

    def test_failed_turn(self) -> None:
        """HandoffTurnResult tracks failed turn."""
        from atp.runner.multi_agent import HandoffTurnResult

        turn_result = HandoffTurnResult(
            agent_name="agent-1",
            sequence_position=0,
            error="Something went wrong",
        )

        assert turn_result.success is False
        assert turn_result.error == "Something went wrong"

    def test_handoff_triggered_flag(self) -> None:
        """HandoffTurnResult tracks handoff trigger state."""
        from atp.runner.multi_agent import HandoffTurnResult

        turn_result = HandoffTurnResult(
            agent_name="agent-1",
            sequence_position=0,
            handoff_triggered=True,
        )

        assert turn_result.handoff_triggered is True


class TestHandoffResult:
    """Tests for HandoffResult model."""

    def test_empty_result(self, test_definition: TestDefinition) -> None:
        """HandoffResult with no turns."""
        from atp.runner.multi_agent import HandoffContext, HandoffResult

        ctx = HandoffContext(original_task="Test task")
        result = HandoffResult(
            test=test_definition,
            handoff_context=ctx,
        )

        assert result.test is test_definition
        assert result.turn_results == []
        assert result.total_agents_executed == 0
        # Empty result is not successful (no successful turns)
        assert result.success is False

    def test_successful_result(self, test_definition: TestDefinition) -> None:
        """HandoffResult with successful turns."""
        from atp.runner.multi_agent import (
            HandoffContext,
            HandoffResult,
            HandoffTurnResult,
        )

        ctx = HandoffContext(original_task="Test task")
        turn = HandoffTurnResult(
            agent_name="agent-1",
            sequence_position=0,
            response=ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
            ),
        )

        result = HandoffResult(
            test=test_definition,
            handoff_context=ctx,
            turn_results=[turn],
            final_response=ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
            ),
        )

        assert result.success is True
        assert result.total_agents_executed == 1

    def test_result_with_error(self, test_definition: TestDefinition) -> None:
        """HandoffResult with error."""
        from atp.runner.multi_agent import HandoffContext, HandoffResult

        ctx = HandoffContext(original_task="Test task")
        result = HandoffResult(
            test=test_definition,
            handoff_context=ctx,
            error="Handoff failed",
        )

        assert result.success is False
        assert result.error == "Handoff failed"

    def test_get_result_for_agent(self, test_definition: TestDefinition) -> None:
        """HandoffResult.get_result_for_agent works correctly."""
        from atp.runner.multi_agent import (
            HandoffContext,
            HandoffResult,
            HandoffTurnResult,
        )

        ctx = HandoffContext(original_task="Test task")
        turn1 = HandoffTurnResult(agent_name="agent-1", sequence_position=0)
        turn2 = HandoffTurnResult(agent_name="agent-2", sequence_position=1)

        result = HandoffResult(
            test=test_definition,
            handoff_context=ctx,
            turn_results=[turn1, turn2],
        )

        assert result.get_result_for_agent("agent-1") is turn1
        assert result.get_result_for_agent("agent-2") is turn2
        assert result.get_result_for_agent("nonexistent") is None

    def test_get_successful_and_failed_agents(
        self, test_definition: TestDefinition
    ) -> None:
        """HandoffResult tracks successful and failed agents."""
        from atp.runner.multi_agent import (
            HandoffContext,
            HandoffResult,
            HandoffTurnResult,
        )

        ctx = HandoffContext(original_task="Test task")
        turn1 = HandoffTurnResult(
            agent_name="agent-1",
            sequence_position=0,
            response=ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
        )
        turn2 = HandoffTurnResult(
            agent_name="agent-2", sequence_position=1, error="Failed"
        )

        result = HandoffResult(
            test=test_definition,
            handoff_context=ctx,
            turn_results=[turn1, turn2],
        )

        assert result.get_successful_agents() == ["agent-1"]
        assert result.get_failed_agents() == ["agent-2"]


class TestMultiAgentTestResultHandoff:
    """Tests for MultiAgentTestResult in handoff mode."""

    def test_handoff_mode_result(self, test_definition: TestDefinition) -> None:
        """MultiAgentTestResult supports handoff mode."""
        from atp.runner.multi_agent import (
            HandoffContext,
            HandoffResult,
            HandoffTurnResult,
        )

        ctx = HandoffContext(original_task="Test task")
        turn = HandoffTurnResult(
            agent_name="agent-1",
            sequence_position=0,
            response=ATPResponse(task_id="test", status=ResponseStatus.COMPLETED),
        )
        handoff_result = HandoffResult(
            test=test_definition,
            handoff_context=ctx,
            turn_results=[turn],
            final_response=ATPResponse(
                task_id="test",
                status=ResponseStatus.COMPLETED,
            ),
        )

        result = MultiAgentTestResult(
            test=test_definition,
            mode=MultiAgentMode.HANDOFF,
            handoff_result=handoff_result,
        )

        assert result.mode == MultiAgentMode.HANDOFF
        assert result.handoff_result is handoff_result
        assert result.all_succeeded is True
        assert result.any_succeeded is True
        assert result.agent_names == ["agent-1"]


class TestMultiAgentOrchestratorHandoff:
    """Tests for MultiAgentOrchestrator in handoff mode."""

    @pytest.mark.anyio
    async def test_basic_handoff_execution(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test basic handoff mode execution."""
        from atp.runner.multi_agent import HandoffConfig

        handoff_config = HandoffConfig()

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.mode == MultiAgentMode.HANDOFF
        assert result.handoff_result is not None
        assert result.handoff_result.total_agents_executed == 2
        assert result.all_succeeded is True

    @pytest.mark.anyio
    async def test_handoff_with_on_success_trigger(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test handoff with ON_SUCCESS trigger."""
        from atp.runner.multi_agent import HandoffConfig, HandoffTrigger

        handoff_config = HandoffConfig(
            handoff_trigger=HandoffTrigger.ON_SUCCESS,
        )

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.handoff_result is not None
        # Both agents should execute since first agent succeeds
        assert result.handoff_result.total_agents_executed == 2

    @pytest.mark.anyio
    async def test_handoff_with_on_failure_trigger(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test handoff with ON_FAILURE trigger."""
        from atp.runner.multi_agent import HandoffConfig, HandoffTrigger

        agents = [
            AgentConfig(name="agent-1", adapter=MockAdapter(name="agent-1")),
            AgentConfig(name="agent-2", adapter=MockAdapter(name="agent-2")),
        ]

        handoff_config = HandoffConfig(
            handoff_trigger=HandoffTrigger.ON_FAILURE,
        )

        async with MultiAgentOrchestrator(
            agents=agents,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.handoff_result is not None
        # Only first agent should execute since it succeeds
        # and ON_FAILURE trigger doesn't fire
        assert result.handoff_result.total_agents_executed == 1
        assert (
            result.handoff_result.handoff_metrics.termination_reason
            == "handoff_not_triggered"
        )

    @pytest.mark.anyio
    async def test_handoff_context_passing(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that context is passed between agents."""
        from atp.runner.multi_agent import HandoffConfig

        handoff_config = HandoffConfig()

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.handoff_result is not None
        ctx = result.handoff_result.handoff_context

        # Context should have outputs from both agents
        assert len(ctx.previous_outputs) == 2
        assert ctx.previous_outputs[0].agent_name == "agent-1"
        assert ctx.previous_outputs[1].agent_name == "agent-2"

    @pytest.mark.anyio
    async def test_handoff_agent_failure_stops_chain(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test that agent failure stops the handoff chain."""
        from atp.runner.multi_agent import HandoffConfig

        agents = [
            AgentConfig(
                name="fail-agent",
                adapter=MockAdapter(
                    name="fail",
                    response=ATPResponse(
                        task_id="test",
                        status=ResponseStatus.FAILED,
                        error="Agent failed",
                    ),
                ),
            ),
            AgentConfig(name="never-runs", adapter=MockAdapter(name="never")),
        ]

        handoff_config = HandoffConfig(continue_on_failure=False)

        async with MultiAgentOrchestrator(
            agents=agents,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.handoff_result is not None
        # Only first agent should execute
        assert result.handoff_result.total_agents_executed == 1
        assert (
            result.handoff_result.handoff_metrics.termination_reason == "agent_failed"
        )

    @pytest.mark.anyio
    async def test_handoff_continue_on_failure(
        self,
        test_definition: TestDefinition,
    ) -> None:
        """Test that handoff can continue on failure if configured."""
        from atp.runner.multi_agent import HandoffConfig

        agents = [
            AgentConfig(
                name="fail-agent",
                adapter=MockAdapter(
                    name="fail",
                    response=ATPResponse(
                        task_id="test",
                        status=ResponseStatus.FAILED,
                        error="Agent failed",
                    ),
                ),
            ),
            AgentConfig(name="success-agent", adapter=MockAdapter(name="success")),
        ]

        handoff_config = HandoffConfig(continue_on_failure=True)

        async with MultiAgentOrchestrator(
            agents=agents,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.handoff_result is not None
        # Both agents should execute
        assert result.handoff_result.total_agents_executed == 2
        assert result.handoff_result.get_failed_agents() == ["fail-agent"]
        assert result.handoff_result.get_successful_agents() == ["success-agent"]

    @pytest.mark.anyio
    async def test_handoff_timeout(self, test_definition: TestDefinition) -> None:
        """Test that agent timeout is enforced in handoff mode."""
        from atp.runner.multi_agent import HandoffConfig

        slow_adapter = MockAdapter(name="slow", delay=2.0)
        agents = [AgentConfig(name="slow-agent", adapter=slow_adapter)]

        handoff_config = HandoffConfig(agent_timeout_seconds=0.1)

        async with MultiAgentOrchestrator(
            agents=agents,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.handoff_result is not None
        turn_result = result.handoff_result.turn_results[0]
        assert turn_result.error is not None
        assert "timeout" in turn_result.error.lower()

    @pytest.mark.anyio
    async def test_handoff_metrics_calculation(
        self, test_definition: TestDefinition
    ) -> None:
        """Test that handoff metrics are calculated correctly."""
        from atp.runner.multi_agent import HandoffConfig

        agents = [
            AgentConfig(
                name="agent-1",
                adapter=MockAdapter(
                    name="agent-1",
                    metrics=Metrics(total_tokens=100, cost_usd=0.01),
                ),
            ),
            AgentConfig(
                name="agent-2",
                adapter=MockAdapter(
                    name="agent-2",
                    metrics=Metrics(total_tokens=200, cost_usd=0.02),
                ),
            ),
        ]

        handoff_config = HandoffConfig()

        async with MultiAgentOrchestrator(
            agents=agents,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.handoff_result is not None
        metrics = result.handoff_result.handoff_metrics

        assert metrics.total_agents == 2
        assert metrics.agents_executed == 2
        assert metrics.total_tokens == 300
        assert abs(metrics.total_cost_usd - 0.03) < 0.001
        assert metrics.termination_reason == "chain_completed"
        assert metrics.final_agent == "agent-2"

    @pytest.mark.anyio
    async def test_handoff_agent_contributions(
        self, test_definition: TestDefinition
    ) -> None:
        """Test that agent contributions are calculated."""
        from atp.runner.multi_agent import HandoffConfig

        agents = [
            AgentConfig(name="agent-1", adapter=MockAdapter(name="agent-1")),
            AgentConfig(name="agent-2", adapter=MockAdapter(name="agent-2")),
        ]

        handoff_config = HandoffConfig()

        async with MultiAgentOrchestrator(
            agents=agents,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.handoff_result is not None
        contributions = result.handoff_result.handoff_metrics.agent_contributions

        assert "agent-1" in contributions
        assert "agent-2" in contributions
        # Both agents succeeded, so both should have contributions
        assert contributions["agent-1"] > 0
        assert contributions["agent-2"] > 0

    @pytest.mark.anyio
    async def test_handoff_suite_execution(
        self,
        test_suite: TestSuite,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test running a suite in handoff mode."""
        from atp.runner.multi_agent import HandoffConfig

        handoff_config = HandoffConfig()

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_suite(test_suite)

        assert result.mode == MultiAgentMode.HANDOFF
        assert result.total_tests == 2
        # Each test should have handoff result
        for test_result in result.test_results:
            assert test_result.handoff_result is not None


class TestHandoffProgressEvents:
    """Tests for progress events in handoff mode."""

    @pytest.mark.anyio
    async def test_handoff_progress_events(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test that progress events are emitted during handoff."""
        from atp.runner.multi_agent import HandoffConfig

        events: list[ProgressEvent] = []

        handoff_config = HandoffConfig()

        async with MultiAgentOrchestrator(
            agents=agent_configs,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
            progress_callback=events.append,
        ) as orchestrator:
            await orchestrator.run_single_test(test_definition)

        event_types = [e.event_type for e in events]

        assert ProgressEventType.TEST_STARTED in event_types
        assert ProgressEventType.RUN_STARTED in event_types
        assert ProgressEventType.RUN_COMPLETED in event_types
        assert (
            ProgressEventType.TEST_COMPLETED in event_types
            or ProgressEventType.TEST_FAILED in event_types
        )


class TestHandoffConvenienceFunctions:
    """Tests for handoff convenience functions."""

    @pytest.mark.anyio
    async def test_run_handoff(
        self,
        test_definition: TestDefinition,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test run_handoff convenience function."""
        from atp.runner.multi_agent import HandoffConfig, run_handoff

        handoff_config = HandoffConfig()

        result = await run_handoff(
            agents=agent_configs,
            test=test_definition,
            handoff_config=handoff_config,
        )

        assert isinstance(result, MultiAgentTestResult)
        assert result.mode == MultiAgentMode.HANDOFF
        assert result.handoff_result is not None
        assert result.handoff_result.total_agents_executed == 2

    @pytest.mark.anyio
    async def test_run_suite_handoff(
        self,
        test_suite: TestSuite,
        agent_configs: list[AgentConfig],
    ) -> None:
        """Test run_suite_handoff convenience function."""
        from atp.runner.multi_agent import HandoffConfig, run_suite_handoff

        handoff_config = HandoffConfig()

        result = await run_suite_handoff(
            agents=agent_configs,
            suite=test_suite,
            handoff_config=handoff_config,
        )

        assert isinstance(result, MultiAgentSuiteResult)
        assert result.mode == MultiAgentMode.HANDOFF
        assert result.total_tests == 2


class TestHandoffTriggerTypes:
    """Tests for different handoff trigger types."""

    def test_trigger_enum_values(self) -> None:
        """Test HandoffTrigger enum values."""
        from atp.runner.multi_agent import HandoffTrigger

        assert HandoffTrigger.ALWAYS.value == "always"
        assert HandoffTrigger.ON_SUCCESS.value == "on_success"
        assert HandoffTrigger.ON_FAILURE.value == "on_failure"
        assert HandoffTrigger.ON_PARTIAL.value == "on_partial"
        assert HandoffTrigger.EXPLICIT.value == "explicit"


class TestContextAccumulationModes:
    """Tests for different context accumulation modes."""

    def test_accumulation_enum_values(self) -> None:
        """Test ContextAccumulationMode enum values."""
        from atp.runner.multi_agent import ContextAccumulationMode

        assert ContextAccumulationMode.APPEND.value == "append"
        assert ContextAccumulationMode.REPLACE.value == "replace"
        assert ContextAccumulationMode.MERGE.value == "merge"
        assert ContextAccumulationMode.SUMMARY.value == "summary"

    @pytest.mark.anyio
    async def test_replace_context_accumulation(
        self, test_definition: TestDefinition
    ) -> None:
        """Test REPLACE context accumulation mode."""
        from atp.runner.multi_agent import (
            ContextAccumulationMode,
            HandoffConfig,
        )

        agents = [
            AgentConfig(name="agent-1", adapter=MockAdapter(name="agent-1")),
            AgentConfig(name="agent-2", adapter=MockAdapter(name="agent-2")),
        ]

        handoff_config = HandoffConfig(
            context_accumulation=ContextAccumulationMode.REPLACE
        )

        async with MultiAgentOrchestrator(
            agents=agents,
            mode=MultiAgentMode.HANDOFF,
            handoff_config=handoff_config,
        ) as orchestrator:
            result = await orchestrator.run_single_test(test_definition)

        assert result.handoff_result is not None
        assert result.handoff_result.success is True
