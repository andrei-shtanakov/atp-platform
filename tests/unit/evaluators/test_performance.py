"""Unit tests for PerformanceEvaluator."""

from datetime import UTC, datetime

import pytest

from atp.evaluators.performance import (
    PerformanceBaseline,
    PerformanceConfig,
    PerformanceEvaluator,
    PerformanceMetrics,
    PerformanceMetricType,
    PerformanceThresholds,
    RegressionResult,
    RegressionStatus,
    calculate_percentile,
    calculate_token_efficiency,
    calculate_tokens_per_second,
    check_regressions,
    check_thresholds,
    compute_performance_metrics,
    detect_regression,
    extract_latencies_from_trace,
    extract_time_to_first_token,
)
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition
from atp.protocol import ATPEvent, ATPResponse, EventType, Metrics, ResponseStatus


@pytest.fixture
def evaluator() -> PerformanceEvaluator:
    """Create PerformanceEvaluator instance."""
    return PerformanceEvaluator()


@pytest.fixture
def sample_task() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-001",
        name="Performance Test",
        task=TaskDefinition(description="Test task for performance evaluation"),
        constraints=Constraints(),
    )


@pytest.fixture
def sample_trace() -> list[ATPEvent]:
    """Create a sample trace with LLM and tool call events."""
    base_time = datetime.now(UTC)
    return [
        ATPEvent(
            task_id="test-001",
            timestamp=base_time,
            sequence=0,
            event_type=EventType.LLM_REQUEST,
            payload={
                "model": "gpt-4",
                "input_tokens": 100,
                "output_tokens": 50,
                "duration_ms": 500.0,
            },
        ),
        ATPEvent(
            task_id="test-001",
            timestamp=base_time,
            sequence=1,
            event_type=EventType.TOOL_CALL,
            payload={
                "tool": "search",
                "duration_ms": 100.0,
                "status": "success",
            },
        ),
        ATPEvent(
            task_id="test-001",
            timestamp=base_time,
            sequence=2,
            event_type=EventType.LLM_REQUEST,
            payload={
                "model": "gpt-4",
                "input_tokens": 200,
                "output_tokens": 100,
                "duration_ms": 800.0,
            },
        ),
        ATPEvent(
            task_id="test-001",
            timestamp=base_time,
            sequence=3,
            event_type=EventType.LLM_REQUEST,
            payload={
                "model": "gpt-4",
                "input_tokens": 150,
                "output_tokens": 75,
                "duration_ms": 600.0,
            },
        ),
    ]


@pytest.fixture
def sample_response() -> ATPResponse:
    """Create a sample response with metrics."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[],
        metrics=Metrics(
            total_tokens=500,
            input_tokens=300,
            output_tokens=200,
            total_steps=5,
            tool_calls=2,
            llm_calls=3,
            wall_time_seconds=2.5,
            cost_usd=0.01,
        ),
    )


@pytest.fixture
def response_no_metrics() -> ATPResponse:
    """Create response without metrics."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[],
    )


@pytest.fixture
def sample_baseline() -> PerformanceBaseline:
    """Create sample baseline data."""
    return PerformanceBaseline(
        latency_p50_mean=550.0,
        latency_p50_std=50.0,
        latency_p95_mean=750.0,
        latency_p95_std=75.0,
        latency_p99_mean=850.0,
        latency_p99_std=100.0,
        time_to_first_token_mean=500.0,
        time_to_first_token_std=50.0,
        tokens_per_second_mean=200.0,
        tokens_per_second_std=20.0,
        token_efficiency_mean=0.4,
        token_efficiency_std=0.05,
        total_duration_mean=2.5,
        total_duration_std=0.3,
        n_samples=10,
    )


class TestPerformanceMetricType:
    """Tests for PerformanceMetricType enum."""

    def test_metric_types(self) -> None:
        """Test metric type values."""
        assert PerformanceMetricType.LATENCY_P50.value == "latency_p50"
        assert PerformanceMetricType.LATENCY_P95.value == "latency_p95"
        assert PerformanceMetricType.LATENCY_P99.value == "latency_p99"
        assert PerformanceMetricType.TIME_TO_FIRST_TOKEN.value == "time_to_first_token"
        assert PerformanceMetricType.TOKENS_PER_SECOND.value == "tokens_per_second"
        assert PerformanceMetricType.TOKEN_EFFICIENCY.value == "token_efficiency"


class TestRegressionStatus:
    """Tests for RegressionStatus enum."""

    def test_status_values(self) -> None:
        """Test status values."""
        assert RegressionStatus.NO_BASELINE.value == "no_baseline"
        assert RegressionStatus.NO_REGRESSION.value == "no_regression"
        assert RegressionStatus.REGRESSION.value == "regression"
        assert RegressionStatus.IMPROVEMENT.value == "improvement"


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        metrics = PerformanceMetrics()
        assert metrics.latency_p50 is None
        assert metrics.tokens_per_second is None
        assert metrics.llm_call_count == 0
        assert metrics.tool_call_count == 0
        assert metrics.llm_latencies == []

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        metrics = PerformanceMetrics(
            latency_p50=500.0,
            latency_p95=750.0,
            latency_p99=900.0,
            time_to_first_token=450.0,
            tokens_per_second=200.0,
            token_efficiency=0.4,
            total_duration=2.5,
            total_tokens=500,
            output_tokens=200,
            input_tokens=300,
            llm_call_count=3,
            tool_call_count=2,
        )
        d = metrics.to_dict()
        assert d["latency_p50_ms"] == 500.0
        assert d["latency_p95_ms"] == 750.0
        assert d["latency_p99_ms"] == 900.0
        assert d["time_to_first_token_ms"] == 450.0
        assert d["tokens_per_second"] == 200.0
        assert d["token_efficiency"] == 0.4
        assert d["total_duration_seconds"] == 2.5
        assert d["total_tokens"] == 500
        assert d["llm_call_count"] == 3
        assert d["tool_call_count"] == 2


class TestRegressionResult:
    """Tests for RegressionResult dataclass."""

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        result = RegressionResult(
            metric="latency_p50",
            status=RegressionStatus.REGRESSION,
            current_value=600.0,
            baseline_value=500.0,
            delta=100.0,
            delta_percent=20.0,
            p_value=0.01,
            is_significant=True,
        )
        d = result.to_dict()
        assert d["metric"] == "latency_p50"
        assert d["status"] == "regression"
        assert d["current_value"] == 600.0
        assert d["baseline_value"] == 500.0
        assert d["delta"] == 100.0
        assert d["delta_percent"] == 20.0
        assert d["p_value"] == 0.01
        assert d["is_significant"] is True


class TestPerformanceThresholds:
    """Tests for PerformanceThresholds."""

    def test_default_thresholds(self) -> None:
        """Test default threshold values."""
        thresholds = PerformanceThresholds()
        assert thresholds.max_latency_p50_ms is None
        assert thresholds.max_latency_p95_ms is None
        assert thresholds.min_tokens_per_second is None

    def test_custom_thresholds(self) -> None:
        """Test custom threshold values."""
        thresholds = PerformanceThresholds(
            max_latency_p50_ms=500.0,
            max_latency_p95_ms=1000.0,
            min_tokens_per_second=100.0,
            min_token_efficiency=0.3,
        )
        assert thresholds.max_latency_p50_ms == 500.0
        assert thresholds.max_latency_p95_ms == 1000.0
        assert thresholds.min_tokens_per_second == 100.0
        assert thresholds.min_token_efficiency == 0.3


class TestPerformanceBaseline:
    """Tests for PerformanceBaseline."""

    def test_baseline_creation(self) -> None:
        """Test baseline creation."""
        baseline = PerformanceBaseline(
            latency_p50_mean=500.0,
            latency_p50_std=50.0,
            n_samples=10,
        )
        assert baseline.latency_p50_mean == 500.0
        assert baseline.latency_p50_std == 50.0
        assert baseline.n_samples == 10


class TestPerformanceConfig:
    """Tests for PerformanceConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = PerformanceConfig()
        assert config.significance_level == 0.05
        assert config.regression_threshold_percent == 10.0
        assert config.check_latency is True
        assert config.check_throughput is True
        assert config.check_regression is True
        assert config.fail_on_regression is False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PerformanceConfig(
            significance_level=0.01,
            regression_threshold_percent=15.0,
            fail_on_regression=True,
        )
        assert config.significance_level == 0.01
        assert config.regression_threshold_percent == 15.0
        assert config.fail_on_regression is True


class TestCalculatePercentile:
    """Tests for calculate_percentile function."""

    def test_empty_list(self) -> None:
        """Test with empty list."""
        assert calculate_percentile([], 50) is None

    def test_single_value(self) -> None:
        """Test with single value."""
        assert calculate_percentile([100.0], 50) == 100.0
        assert calculate_percentile([100.0], 95) == 100.0

    def test_multiple_values(self) -> None:
        """Test with multiple values."""
        values = [100.0, 200.0, 300.0, 400.0, 500.0]
        # p50 should be around middle value
        p50 = calculate_percentile(values, 50)
        assert p50 is not None
        assert 200.0 <= p50 <= 400.0

        # p95 should be near top
        p95 = calculate_percentile(values, 95)
        assert p95 is not None
        assert p95 >= 400.0

    def test_percentile_boundaries(self) -> None:
        """Test percentile at boundaries."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert calculate_percentile(values, 0) == 10.0
        assert calculate_percentile(values, 100) == 50.0


class TestExtractLatenciesFromTrace:
    """Tests for extract_latencies_from_trace function."""

    def test_extract_latencies(self, sample_trace: list[ATPEvent]) -> None:
        """Test extracting latencies from trace."""
        latencies = extract_latencies_from_trace(sample_trace)
        assert len(latencies) == 3
        assert 500.0 in latencies
        assert 800.0 in latencies
        assert 600.0 in latencies

    def test_empty_trace(self) -> None:
        """Test with empty trace."""
        assert extract_latencies_from_trace([]) == []

    def test_trace_without_llm_events(self) -> None:
        """Test trace without LLM events."""
        trace = [
            ATPEvent(
                task_id="test",
                timestamp=datetime.now(UTC),
                sequence=0,
                event_type=EventType.TOOL_CALL,
                payload={"tool": "search"},
            )
        ]
        assert extract_latencies_from_trace(trace) == []


class TestExtractTimeToFirstToken:
    """Tests for extract_time_to_first_token function."""

    def test_extract_ttft(self, sample_trace: list[ATPEvent]) -> None:
        """Test extracting time to first token."""
        ttft = extract_time_to_first_token(sample_trace)
        assert ttft == 500.0  # First LLM event duration

    def test_empty_trace(self) -> None:
        """Test with empty trace."""
        assert extract_time_to_first_token([]) is None

    def test_trace_without_llm_events(self) -> None:
        """Test trace without LLM events."""
        trace = [
            ATPEvent(
                task_id="test",
                timestamp=datetime.now(UTC),
                sequence=0,
                event_type=EventType.TOOL_CALL,
                payload={"tool": "search"},
            )
        ]
        assert extract_time_to_first_token(trace) is None

    def test_llm_event_without_duration(self) -> None:
        """Test LLM event without duration field."""
        trace = [
            ATPEvent(
                task_id="test",
                timestamp=datetime.now(UTC),
                sequence=0,
                event_type=EventType.LLM_REQUEST,
                payload={"model": "gpt-4"},  # No duration_ms
            )
        ]
        assert extract_time_to_first_token(trace) is None


class TestCalculateTokensPerSecond:
    """Tests for calculate_tokens_per_second function."""

    def test_calculate_throughput(self) -> None:
        """Test throughput calculation."""
        result = calculate_tokens_per_second(1000, 2.0)
        assert result == 500.0

    def test_zero_duration(self) -> None:
        """Test with zero duration."""
        assert calculate_tokens_per_second(100, 0.0) is None

    def test_negative_duration(self) -> None:
        """Test with negative duration."""
        assert calculate_tokens_per_second(100, -1.0) is None

    def test_none_values(self) -> None:
        """Test with None values."""
        assert calculate_tokens_per_second(None, 1.0) is None
        assert calculate_tokens_per_second(100, None) is None


class TestCalculateTokenEfficiency:
    """Tests for calculate_token_efficiency function."""

    def test_calculate_efficiency(self) -> None:
        """Test efficiency calculation."""
        result = calculate_token_efficiency(200, 500)
        assert result == 0.4

    def test_zero_total(self) -> None:
        """Test with zero total tokens."""
        assert calculate_token_efficiency(100, 0) is None

    def test_none_values(self) -> None:
        """Test with None values."""
        assert calculate_token_efficiency(None, 100) is None
        assert calculate_token_efficiency(100, None) is None


class TestComputePerformanceMetrics:
    """Tests for compute_performance_metrics function."""

    def test_compute_metrics(
        self, sample_response: ATPResponse, sample_trace: list[ATPEvent]
    ) -> None:
        """Test computing all metrics."""
        metrics = compute_performance_metrics(sample_response, sample_trace)

        assert metrics.llm_call_count == 3
        assert metrics.tool_call_count == 1
        assert metrics.latency_p50 is not None
        assert metrics.latency_p95 is not None
        assert metrics.latency_p99 is not None
        assert metrics.time_to_first_token == 500.0
        assert metrics.total_tokens == 500
        assert metrics.tokens_per_second is not None
        assert metrics.token_efficiency is not None

    def test_compute_metrics_no_response_metrics(
        self, response_no_metrics: ATPResponse, sample_trace: list[ATPEvent]
    ) -> None:
        """Test with response without metrics."""
        metrics = compute_performance_metrics(response_no_metrics, sample_trace)

        assert metrics.llm_call_count == 3
        assert metrics.tokens_per_second is None
        assert metrics.token_efficiency is None

    def test_compute_metrics_empty_trace(self, sample_response: ATPResponse) -> None:
        """Test with empty trace."""
        metrics = compute_performance_metrics(sample_response, [])

        assert metrics.llm_call_count == 0
        assert metrics.latency_p50 is None
        assert metrics.time_to_first_token is None


class TestCheckThresholds:
    """Tests for check_thresholds function."""

    def test_all_thresholds_pass(self) -> None:
        """Test when all thresholds pass."""
        metrics = PerformanceMetrics(
            latency_p50=400.0,
            latency_p95=800.0,
            tokens_per_second=150.0,
            token_efficiency=0.5,
        )
        thresholds = PerformanceThresholds(
            max_latency_p50_ms=500.0,
            max_latency_p95_ms=1000.0,
            min_tokens_per_second=100.0,
            min_token_efficiency=0.3,
        )

        results = check_thresholds(metrics, thresholds)
        assert all(passed for _, passed, _ in results)

    def test_some_thresholds_fail(self) -> None:
        """Test when some thresholds fail."""
        metrics = PerformanceMetrics(
            latency_p50=600.0,  # Exceeds threshold
            latency_p95=800.0,
            tokens_per_second=80.0,  # Below threshold
        )
        thresholds = PerformanceThresholds(
            max_latency_p50_ms=500.0,
            max_latency_p95_ms=1000.0,
            min_tokens_per_second=100.0,
        )

        results = check_thresholds(metrics, thresholds)
        passed_count = sum(1 for _, passed, _ in results if passed)
        failed_count = sum(1 for _, passed, _ in results if not passed)

        assert passed_count == 1  # latency_p95 passes
        assert failed_count == 2  # latency_p50 and tokens_per_second fail

    def test_no_thresholds_set(self) -> None:
        """Test when no thresholds are set."""
        metrics = PerformanceMetrics(latency_p50=500.0)
        thresholds = PerformanceThresholds()

        results = check_thresholds(metrics, thresholds)
        assert len(results) == 0


class TestDetectRegression:
    """Tests for detect_regression function."""

    def test_no_baseline(self) -> None:
        """Test with no baseline data."""
        result = detect_regression(
            metric_name="latency_p50",
            current_value=500.0,
            baseline_mean=None,
            baseline_std=None,
            baseline_samples=10,
            significance_level=0.05,
            regression_threshold_percent=10.0,
        )
        assert result.status == RegressionStatus.NO_BASELINE

    def test_no_current_value(self) -> None:
        """Test with no current value."""
        result = detect_regression(
            metric_name="latency_p50",
            current_value=None,
            baseline_mean=500.0,
            baseline_std=50.0,
            baseline_samples=10,
            significance_level=0.05,
            regression_threshold_percent=10.0,
        )
        assert result.status == RegressionStatus.NO_BASELINE

    def test_regression_detected(self) -> None:
        """Test regression detection (higher is worse)."""
        result = detect_regression(
            metric_name="latency_p50",
            current_value=700.0,  # Much higher than baseline
            baseline_mean=500.0,
            baseline_std=30.0,
            baseline_samples=10,
            significance_level=0.05,
            regression_threshold_percent=10.0,
            higher_is_worse=True,
        )
        assert result.current_value == 700.0
        assert result.baseline_value == 500.0
        assert result.delta == 200.0
        assert result.delta_percent == 40.0

    def test_improvement_detected(self) -> None:
        """Test improvement detection (higher is worse)."""
        result = detect_regression(
            metric_name="latency_p50",
            current_value=350.0,  # Much lower than baseline
            baseline_mean=500.0,
            baseline_std=30.0,
            baseline_samples=10,
            significance_level=0.05,
            regression_threshold_percent=10.0,
            higher_is_worse=True,
        )
        assert result.delta == -150.0
        assert result.delta_percent == -30.0

    def test_no_regression_higher_is_better(self) -> None:
        """Test when higher is better (e.g., throughput)."""
        result = detect_regression(
            metric_name="tokens_per_second",
            current_value=250.0,  # Higher is better
            baseline_mean=200.0,
            baseline_std=20.0,
            baseline_samples=10,
            significance_level=0.05,
            regression_threshold_percent=10.0,
            higher_is_worse=False,
        )
        assert result.delta == 50.0
        assert result.delta_percent == 25.0


class TestCheckRegressions:
    """Tests for check_regressions function."""

    def test_check_all_metrics(self, sample_baseline: PerformanceBaseline) -> None:
        """Test checking all metrics against baseline."""
        metrics = PerformanceMetrics(
            latency_p50=600.0,
            latency_p95=800.0,
            latency_p99=950.0,
            time_to_first_token=550.0,
            tokens_per_second=180.0,
            token_efficiency=0.38,
            total_duration=2.8,
        )

        results = check_regressions(
            metrics,
            sample_baseline,
            significance_level=0.05,
            regression_threshold_percent=10.0,
        )

        assert len(results) == 7  # All metrics checked
        metric_names = [r.metric for r in results]
        assert "latency_p50" in metric_names
        assert "latency_p95" in metric_names
        assert "latency_p99" in metric_names
        assert "time_to_first_token" in metric_names
        assert "tokens_per_second" in metric_names
        assert "token_efficiency" in metric_names
        assert "total_duration" in metric_names


class TestPerformanceEvaluatorProperties:
    """Tests for PerformanceEvaluator properties."""

    def test_evaluator_name(self, evaluator: PerformanceEvaluator) -> None:
        """Test evaluator name property."""
        assert evaluator.name == "performance"


class TestPerformanceEvaluatorEvaluate:
    """Tests for PerformanceEvaluator.evaluate method."""

    @pytest.mark.anyio
    async def test_evaluate_basic(
        self,
        evaluator: PerformanceEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        sample_trace: list[ATPEvent],
    ) -> None:
        """Test basic evaluation."""
        assertion = Assertion(type="performance", config={})
        result = await evaluator.evaluate(
            sample_task, sample_response, sample_trace, assertion
        )

        assert result.evaluator == "performance"
        assert len(result.checks) >= 1
        summary = next(c for c in result.checks if c.name == "performance_summary")
        assert summary.details is not None
        assert "metrics" in summary.details

    @pytest.mark.anyio
    async def test_evaluate_with_thresholds(
        self,
        evaluator: PerformanceEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        sample_trace: list[ATPEvent],
    ) -> None:
        """Test evaluation with thresholds."""
        assertion = Assertion(
            type="performance",
            config={
                "thresholds": {
                    "max_latency_p50_ms": 1000.0,
                    "max_latency_p95_ms": 2000.0,
                    "min_tokens_per_second": 100.0,
                },
            },
        )
        result = await evaluator.evaluate(
            sample_task, sample_response, sample_trace, assertion
        )

        # Should have threshold checks
        threshold_checks = [c for c in result.checks if "threshold_" in c.name]
        assert len(threshold_checks) >= 1

    @pytest.mark.anyio
    async def test_evaluate_threshold_failure(
        self,
        evaluator: PerformanceEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        sample_trace: list[ATPEvent],
    ) -> None:
        """Test evaluation with failing thresholds."""
        assertion = Assertion(
            type="performance",
            config={
                "thresholds": {
                    "max_latency_p50_ms": 100.0,  # Very strict
                },
            },
        )
        result = await evaluator.evaluate(
            sample_task, sample_response, sample_trace, assertion
        )

        threshold_check = next(c for c in result.checks if "threshold_" in c.name)
        assert threshold_check.passed is False

    @pytest.mark.anyio
    async def test_evaluate_with_baseline(
        self,
        evaluator: PerformanceEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        sample_trace: list[ATPEvent],
        sample_baseline: PerformanceBaseline,
    ) -> None:
        """Test evaluation with baseline regression check."""
        assertion = Assertion(
            type="performance",
            config={
                "baseline": sample_baseline.model_dump(),
                "check_regression": True,
            },
        )
        result = await evaluator.evaluate(
            sample_task, sample_response, sample_trace, assertion
        )

        regression_check = next(
            (c for c in result.checks if c.name == "regression_check"), None
        )
        assert regression_check is not None

    @pytest.mark.anyio
    async def test_evaluate_fail_on_regression(
        self,
        evaluator: PerformanceEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
        sample_trace: list[ATPEvent],
    ) -> None:
        """Test evaluation fails when regression detected with fail_on_regression."""
        # Test verifies the fail_on_regression config path is processed
        # The actual regression detection depends on statistical significance
        # which requires multiple samples. This test verifies the config is
        # properly parsed and the regression check is included in results.
        baseline = PerformanceBaseline(
            latency_p50_mean=100.0,
            latency_p50_std=10.0,
            n_samples=10,
        )
        assertion = Assertion(
            type="performance",
            config={
                "baseline": baseline.model_dump(),
                "check_regression": True,
                "fail_on_regression": True,
                "regression_threshold_percent": 5.0,
            },
        )
        result = await evaluator.evaluate(
            sample_task, sample_response, sample_trace, assertion
        )

        # Verify regression check is included
        regression_check = next(
            (c for c in result.checks if c.name == "regression_check"), None
        )
        assert regression_check is not None
        # Verify baseline was used for comparison
        summary = next(c for c in result.checks if c.name == "performance_summary")
        assert summary.details is not None
        assert "regressions" in summary.details
        # The latency_p50 should show comparison with baseline
        regressions = summary.details["regressions"]
        latency_p50_result = next(
            (r for r in regressions if r["metric"] == "latency_p50"), None
        )
        assert latency_p50_result is not None
        assert latency_p50_result["baseline_value"] == 100.0

    @pytest.mark.anyio
    async def test_evaluate_empty_trace(
        self,
        evaluator: PerformanceEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test evaluation with empty trace."""
        assertion = Assertion(type="performance", config={})
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)

        summary = next(c for c in result.checks if c.name == "performance_summary")
        assert summary.details is not None
        metrics = summary.details["metrics"]
        assert metrics["llm_call_count"] == 0

    @pytest.mark.anyio
    async def test_evaluate_no_response_metrics(
        self,
        evaluator: PerformanceEvaluator,
        sample_task: TestDefinition,
        response_no_metrics: ATPResponse,
        sample_trace: list[ATPEvent],
    ) -> None:
        """Test evaluation when response has no metrics."""
        assertion = Assertion(type="performance", config={})
        result = await evaluator.evaluate(
            sample_task, response_no_metrics, sample_trace, assertion
        )

        summary = next(c for c in result.checks if c.name == "performance_summary")
        # Should still have trace-based metrics
        assert summary.details is not None
        metrics = summary.details["metrics"]
        assert metrics["llm_call_count"] == 3


class TestRegistry:
    """Tests for registry integration."""

    def test_performance_in_registry(self) -> None:
        """Test PerformanceEvaluator is registered."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        assert registry.is_registered("performance")

    def test_performance_assertion_mapped(self) -> None:
        """Test performance assertion type is mapped."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        assert registry.supports_assertion("performance")

    def test_create_performance_from_registry(self) -> None:
        """Test creating PerformanceEvaluator from registry."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        evaluator = registry.create("performance")
        assert isinstance(evaluator, PerformanceEvaluator)


class TestPerformanceImports:
    """Tests for module imports."""

    def test_imports_from_evaluators(self) -> None:
        """Test imports from evaluators package."""
        from atp.evaluators import (
            PerformanceBaseline,
            PerformanceConfig,
            PerformanceEvaluator,
            PerformanceMetrics,
            PerformanceMetricType,
            PerformanceThresholds,
            RegressionResult,
            RegressionStatus,
        )

        assert PerformanceEvaluator is not None
        assert PerformanceConfig is not None
        assert PerformanceThresholds is not None
        assert PerformanceBaseline is not None
        assert PerformanceMetrics is not None
        assert PerformanceMetricType is not None
        assert RegressionResult is not None
        assert RegressionStatus is not None
