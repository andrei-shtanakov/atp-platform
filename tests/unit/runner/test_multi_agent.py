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
    ComparisonMetrics,
    MultiAgentMode,
    MultiAgentOrchestrator,
    MultiAgentSuiteResult,
    MultiAgentTestResult,
    RankingMetric,
    run_comparison,
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
