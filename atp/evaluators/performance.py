"""Performance evaluator for tracking agent performance metrics.

This module provides an evaluator that:
- Tracks latency percentiles (p50, p95, p99)
- Measures time to first token for streaming
- Calculates tokens per second throughput
- Computes token efficiency ratio
- Supports configurable thresholds
- Detects performance regressions vs baseline
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from atp.baseline.comparison import welchs_t_test
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse, EventType

from .base import EvalCheck, EvalResult, Evaluator

logger = logging.getLogger(__name__)


class PerformanceMetricType(str, Enum):
    """Types of performance metrics."""

    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    TIME_TO_FIRST_TOKEN = "time_to_first_token"
    TOKENS_PER_SECOND = "tokens_per_second"
    TOKEN_EFFICIENCY = "token_efficiency"
    TOTAL_DURATION = "total_duration"
    MEMORY_USAGE = "memory_usage"


class RegressionStatus(str, Enum):
    """Status of regression detection."""

    NO_BASELINE = "no_baseline"
    NO_REGRESSION = "no_regression"
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"


@dataclass
class PerformanceMetrics:
    """Container for computed performance metrics."""

    latency_p50: float | None = None
    latency_p95: float | None = None
    latency_p99: float | None = None
    time_to_first_token: float | None = None
    tokens_per_second: float | None = None
    token_efficiency: float | None = None
    total_duration: float | None = None
    memory_usage_mb: float | None = None
    total_tokens: int | None = None
    output_tokens: int | None = None
    input_tokens: int | None = None
    llm_call_count: int = 0
    tool_call_count: int = 0
    llm_latencies: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        result: dict[str, Any] = {}
        if self.latency_p50 is not None:
            result["latency_p50_ms"] = round(self.latency_p50, 2)
        if self.latency_p95 is not None:
            result["latency_p95_ms"] = round(self.latency_p95, 2)
        if self.latency_p99 is not None:
            result["latency_p99_ms"] = round(self.latency_p99, 2)
        if self.time_to_first_token is not None:
            result["time_to_first_token_ms"] = round(self.time_to_first_token, 2)
        if self.tokens_per_second is not None:
            result["tokens_per_second"] = round(self.tokens_per_second, 2)
        if self.token_efficiency is not None:
            result["token_efficiency"] = round(self.token_efficiency, 4)
        if self.total_duration is not None:
            result["total_duration_seconds"] = round(self.total_duration, 3)
        if self.memory_usage_mb is not None:
            result["memory_usage_mb"] = round(self.memory_usage_mb, 2)
        if self.total_tokens is not None:
            result["total_tokens"] = self.total_tokens
        if self.output_tokens is not None:
            result["output_tokens"] = self.output_tokens
        if self.input_tokens is not None:
            result["input_tokens"] = self.input_tokens
        result["llm_call_count"] = self.llm_call_count
        result["tool_call_count"] = self.tool_call_count
        return result


@dataclass
class RegressionResult:
    """Result of regression detection for a metric."""

    metric: str
    status: RegressionStatus
    current_value: float | None = None
    baseline_value: float | None = None
    delta: float | None = None
    delta_percent: float | None = None
    p_value: float | None = None
    is_significant: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "metric": self.metric,
            "status": self.status.value,
        }
        if self.current_value is not None:
            result["current_value"] = round(self.current_value, 4)
        if self.baseline_value is not None:
            result["baseline_value"] = round(self.baseline_value, 4)
        if self.delta is not None:
            result["delta"] = round(self.delta, 4)
        if self.delta_percent is not None:
            result["delta_percent"] = round(self.delta_percent, 2)
        if self.p_value is not None:
            result["p_value"] = round(self.p_value, 6)
        result["is_significant"] = self.is_significant
        return result


class PerformanceThresholds(BaseModel):
    """Configurable thresholds for performance evaluation."""

    max_latency_p50_ms: float | None = Field(
        None, description="Maximum acceptable p50 latency in milliseconds"
    )
    max_latency_p95_ms: float | None = Field(
        None, description="Maximum acceptable p95 latency in milliseconds"
    )
    max_latency_p99_ms: float | None = Field(
        None, description="Maximum acceptable p99 latency in milliseconds"
    )
    max_time_to_first_token_ms: float | None = Field(
        None, description="Maximum acceptable time to first token in milliseconds"
    )
    min_tokens_per_second: float | None = Field(
        None, description="Minimum acceptable tokens per second throughput"
    )
    min_token_efficiency: float | None = Field(
        None, description="Minimum acceptable token efficiency ratio (0-1)"
    )
    max_total_duration_seconds: float | None = Field(
        None, description="Maximum acceptable total duration in seconds"
    )
    max_memory_usage_mb: float | None = Field(
        None, description="Maximum acceptable memory usage in megabytes"
    )


class PerformanceBaseline(BaseModel):
    """Baseline data for regression detection."""

    latency_p50_mean: float | None = Field(None, description="Mean p50 latency")
    latency_p50_std: float | None = Field(None, description="Std dev of p50 latency")
    latency_p95_mean: float | None = Field(None, description="Mean p95 latency")
    latency_p95_std: float | None = Field(None, description="Std dev of p95 latency")
    latency_p99_mean: float | None = Field(None, description="Mean p99 latency")
    latency_p99_std: float | None = Field(None, description="Std dev of p99 latency")
    time_to_first_token_mean: float | None = Field(
        None, description="Mean time to first token"
    )
    time_to_first_token_std: float | None = Field(
        None, description="Std dev of time to first token"
    )
    tokens_per_second_mean: float | None = Field(
        None, description="Mean tokens per second"
    )
    tokens_per_second_std: float | None = Field(
        None, description="Std dev of tokens per second"
    )
    token_efficiency_mean: float | None = Field(
        None, description="Mean token efficiency"
    )
    token_efficiency_std: float | None = Field(
        None, description="Std dev of token efficiency"
    )
    total_duration_mean: float | None = Field(None, description="Mean total duration")
    total_duration_std: float | None = Field(
        None, description="Std dev of total duration"
    )
    n_samples: int = Field(1, description="Number of samples in baseline", ge=1)


class PerformanceConfig(BaseModel):
    """Configuration for Performance evaluator."""

    thresholds: PerformanceThresholds = Field(
        default_factory=PerformanceThresholds,
        description="Configurable thresholds for metrics",
    )
    baseline: PerformanceBaseline | None = Field(
        None, description="Baseline data for regression detection"
    )
    significance_level: float = Field(
        0.05, ge=0.0, le=1.0, description="P-value threshold for significance"
    )
    regression_threshold_percent: float = Field(
        10.0,
        ge=0.0,
        description="Percentage degradation to consider a regression",
    )
    check_latency: bool = Field(True, description="Whether to check latency metrics")
    check_throughput: bool = Field(
        True, description="Whether to check throughput metrics"
    )
    check_efficiency: bool = Field(
        True, description="Whether to check efficiency metrics"
    )
    check_regression: bool = Field(
        True, description="Whether to check for regressions against baseline"
    )
    fail_on_regression: bool = Field(
        False, description="Whether to fail the evaluation if regression detected"
    )


def calculate_percentile(values: list[float], percentile: float) -> float | None:
    """Calculate a percentile from a list of values.

    Args:
        values: List of numeric values.
        percentile: Percentile to calculate (0-100).

    Returns:
        The calculated percentile value, or None if list is empty.
    """
    if not values:
        return None

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    # Linear interpolation method
    index = (percentile / 100.0) * (n - 1)
    lower_index = int(math.floor(index))
    upper_index = int(math.ceil(index))

    if lower_index == upper_index:
        return sorted_values[lower_index]

    fraction = index - lower_index
    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )


def extract_latencies_from_trace(trace: list[ATPEvent]) -> list[float]:
    """Extract LLM call latencies from event trace.

    Args:
        trace: List of ATP events.

    Returns:
        List of latencies in milliseconds.
    """
    latencies: list[float] = []

    for event in trace:
        if event.event_type == EventType.LLM_REQUEST:
            duration_ms = event.payload.get("duration_ms")
            if duration_ms is not None:
                latencies.append(float(duration_ms))

    return latencies


def extract_time_to_first_token(trace: list[ATPEvent]) -> float | None:
    """Extract time to first token from event trace.

    This is the time from the first LLM request to when the first
    token was generated (approximated by first response).

    Args:
        trace: List of ATP events.

    Returns:
        Time to first token in milliseconds, or None if not available.
    """
    if not trace:
        return None

    # Find the first LLM request event
    first_llm_event: ATPEvent | None = None
    for event in trace:
        if event.event_type == EventType.LLM_REQUEST:
            first_llm_event = event
            break

    if first_llm_event is None:
        return None

    # The duration_ms of the first LLM request approximates TTFT
    # In a streaming scenario, this would be the time to first chunk
    ttft = first_llm_event.payload.get("duration_ms")
    if ttft is not None:
        return float(ttft)

    return None


def calculate_tokens_per_second(
    total_tokens: int | None,
    duration_seconds: float | None,
) -> float | None:
    """Calculate tokens per second throughput.

    Args:
        total_tokens: Total tokens generated.
        duration_seconds: Total duration in seconds.

    Returns:
        Tokens per second, or None if calculation not possible.
    """
    if total_tokens is None or duration_seconds is None:
        return None
    if duration_seconds <= 0:
        return None
    return float(total_tokens) / duration_seconds


def calculate_token_efficiency(
    output_tokens: int | None,
    total_tokens: int | None,
) -> float | None:
    """Calculate token efficiency ratio.

    Token efficiency is the ratio of output tokens to total tokens,
    representing how much of the token budget went to useful output.

    Args:
        output_tokens: Number of output tokens.
        total_tokens: Total tokens used.

    Returns:
        Efficiency ratio (0-1), or None if calculation not possible.
    """
    if output_tokens is None or total_tokens is None:
        return None
    if total_tokens <= 0:
        return None
    return float(output_tokens) / float(total_tokens)


def compute_performance_metrics(
    response: ATPResponse,
    trace: list[ATPEvent],
) -> PerformanceMetrics:
    """Compute all performance metrics from response and trace.

    Args:
        response: ATP response with metrics.
        trace: List of ATP events.

    Returns:
        Computed performance metrics.
    """
    metrics = PerformanceMetrics()

    # Extract latencies from trace
    latencies = extract_latencies_from_trace(trace)
    metrics.llm_latencies = latencies
    metrics.llm_call_count = len(latencies)

    # Calculate latency percentiles
    if latencies:
        metrics.latency_p50 = calculate_percentile(latencies, 50)
        metrics.latency_p95 = calculate_percentile(latencies, 95)
        metrics.latency_p99 = calculate_percentile(latencies, 99)

    # Extract time to first token
    metrics.time_to_first_token = extract_time_to_first_token(trace)

    # Count tool calls
    tool_call_count = sum(
        1 for event in trace if event.event_type == EventType.TOOL_CALL
    )
    metrics.tool_call_count = tool_call_count

    # Extract metrics from response
    if response.metrics:
        response_metrics = response.metrics
        metrics.total_tokens = response_metrics.total_tokens
        metrics.output_tokens = response_metrics.output_tokens
        metrics.input_tokens = response_metrics.input_tokens
        metrics.total_duration = response_metrics.wall_time_seconds

        # Calculate throughput
        metrics.tokens_per_second = calculate_tokens_per_second(
            response_metrics.total_tokens,
            response_metrics.wall_time_seconds,
        )

        # Calculate efficiency
        metrics.token_efficiency = calculate_token_efficiency(
            response_metrics.output_tokens,
            response_metrics.total_tokens,
        )

    return metrics


def check_thresholds(
    metrics: PerformanceMetrics,
    thresholds: PerformanceThresholds,
) -> list[tuple[str, bool, str]]:
    """Check metrics against configured thresholds.

    Args:
        metrics: Computed performance metrics.
        thresholds: Configured thresholds.

    Returns:
        List of (check_name, passed, message) tuples.
    """
    results: list[tuple[str, bool, str]] = []

    if thresholds.max_latency_p50_ms is not None and metrics.latency_p50 is not None:
        passed = metrics.latency_p50 <= thresholds.max_latency_p50_ms
        msg = (
            f"p50 latency {metrics.latency_p50:.2f}ms "
            f"{'<=' if passed else '>'} threshold {thresholds.max_latency_p50_ms:.2f}ms"
        )
        results.append(("latency_p50", passed, msg))

    if thresholds.max_latency_p95_ms is not None and metrics.latency_p95 is not None:
        passed = metrics.latency_p95 <= thresholds.max_latency_p95_ms
        msg = (
            f"p95 latency {metrics.latency_p95:.2f}ms "
            f"{'<=' if passed else '>'} threshold {thresholds.max_latency_p95_ms:.2f}ms"
        )
        results.append(("latency_p95", passed, msg))

    if thresholds.max_latency_p99_ms is not None and metrics.latency_p99 is not None:
        passed = metrics.latency_p99 <= thresholds.max_latency_p99_ms
        msg = (
            f"p99 latency {metrics.latency_p99:.2f}ms "
            f"{'<=' if passed else '>'} threshold {thresholds.max_latency_p99_ms:.2f}ms"
        )
        results.append(("latency_p99", passed, msg))

    if (
        thresholds.max_time_to_first_token_ms is not None
        and metrics.time_to_first_token is not None
    ):
        passed = metrics.time_to_first_token <= thresholds.max_time_to_first_token_ms
        msg = (
            f"TTFT {metrics.time_to_first_token:.2f}ms "
            f"{'<=' if passed else '>'} threshold "
            f"{thresholds.max_time_to_first_token_ms:.2f}ms"
        )
        results.append(("time_to_first_token", passed, msg))

    if (
        thresholds.min_tokens_per_second is not None
        and metrics.tokens_per_second is not None
    ):
        passed = metrics.tokens_per_second >= thresholds.min_tokens_per_second
        threshold_val = thresholds.min_tokens_per_second
        msg = (
            f"Throughput {metrics.tokens_per_second:.2f} tok/s "
            f"{'>=' if passed else '<'} threshold {threshold_val:.2f} tok/s"
        )
        results.append(("tokens_per_second", passed, msg))

    if (
        thresholds.min_token_efficiency is not None
        and metrics.token_efficiency is not None
    ):
        passed = metrics.token_efficiency >= thresholds.min_token_efficiency
        msg = (
            f"Token efficiency {metrics.token_efficiency:.4f} "
            f"{'>=' if passed else '<'} threshold {thresholds.min_token_efficiency:.4f}"
        )
        results.append(("token_efficiency", passed, msg))

    if (
        thresholds.max_total_duration_seconds is not None
        and metrics.total_duration is not None
    ):
        passed = metrics.total_duration <= thresholds.max_total_duration_seconds
        msg = (
            f"Duration {metrics.total_duration:.3f}s "
            f"{'<=' if passed else '>'} threshold "
            f"{thresholds.max_total_duration_seconds:.3f}s"
        )
        results.append(("total_duration", passed, msg))

    if (
        thresholds.max_memory_usage_mb is not None
        and metrics.memory_usage_mb is not None
    ):
        passed = metrics.memory_usage_mb <= thresholds.max_memory_usage_mb
        threshold_val = thresholds.max_memory_usage_mb
        msg = (
            f"Memory {metrics.memory_usage_mb:.2f}MB "
            f"{'<=' if passed else '>'} threshold {threshold_val:.2f}MB"
        )
        results.append(("memory_usage", passed, msg))

    return results


def detect_regression(
    metric_name: str,
    current_value: float | None,
    baseline_mean: float | None,
    baseline_std: float | None,
    baseline_samples: int,
    significance_level: float,
    regression_threshold_percent: float,
    higher_is_worse: bool = True,
) -> RegressionResult:
    """Detect regression for a single metric.

    Args:
        metric_name: Name of the metric.
        current_value: Current metric value.
        baseline_mean: Baseline mean value.
        baseline_std: Baseline standard deviation.
        baseline_samples: Number of samples in baseline.
        significance_level: P-value threshold for significance.
        regression_threshold_percent: Percentage change threshold.
        higher_is_worse: If True, higher values indicate regression.

    Returns:
        RegressionResult with detection outcome.
    """
    if current_value is None:
        return RegressionResult(
            metric=metric_name,
            status=RegressionStatus.NO_BASELINE,
        )

    if baseline_mean is None or baseline_std is None:
        return RegressionResult(
            metric=metric_name,
            status=RegressionStatus.NO_BASELINE,
            current_value=current_value,
        )

    # Calculate delta
    delta = current_value - baseline_mean
    delta_percent = (delta / baseline_mean * 100) if baseline_mean != 0 else 0.0

    # Perform Welch's t-test (treating current as single sample)
    # For single current value, we use a one-sample approximation
    t_stat, p_value = welchs_t_test(
        mean1=current_value,
        std1=0.0,  # Single sample, no variance
        n1=1,
        mean2=baseline_mean,
        std2=baseline_std,
        n2=baseline_samples,
    )

    is_significant = p_value < significance_level

    # Determine regression status
    threshold_exceeded = abs(delta_percent) > regression_threshold_percent

    if higher_is_worse:
        is_regression = delta > 0 and threshold_exceeded
        is_improvement = delta < 0 and threshold_exceeded
    else:
        is_regression = delta < 0 and threshold_exceeded
        is_improvement = delta > 0 and threshold_exceeded

    if is_regression and is_significant:
        status = RegressionStatus.REGRESSION
    elif is_improvement and is_significant:
        status = RegressionStatus.IMPROVEMENT
    else:
        status = RegressionStatus.NO_REGRESSION

    return RegressionResult(
        metric=metric_name,
        status=status,
        current_value=current_value,
        baseline_value=baseline_mean,
        delta=delta,
        delta_percent=delta_percent,
        p_value=p_value,
        is_significant=is_significant,
    )


def check_regressions(
    metrics: PerformanceMetrics,
    baseline: PerformanceBaseline,
    significance_level: float,
    regression_threshold_percent: float,
) -> list[RegressionResult]:
    """Check for regressions against baseline.

    Args:
        metrics: Current performance metrics.
        baseline: Baseline data.
        significance_level: P-value threshold.
        regression_threshold_percent: Percentage threshold for regression.

    Returns:
        List of regression results for each metric.
    """
    results: list[RegressionResult] = []

    # Latency metrics (higher is worse)
    results.append(
        detect_regression(
            "latency_p50",
            metrics.latency_p50,
            baseline.latency_p50_mean,
            baseline.latency_p50_std,
            baseline.n_samples,
            significance_level,
            regression_threshold_percent,
            higher_is_worse=True,
        )
    )

    results.append(
        detect_regression(
            "latency_p95",
            metrics.latency_p95,
            baseline.latency_p95_mean,
            baseline.latency_p95_std,
            baseline.n_samples,
            significance_level,
            regression_threshold_percent,
            higher_is_worse=True,
        )
    )

    results.append(
        detect_regression(
            "latency_p99",
            metrics.latency_p99,
            baseline.latency_p99_mean,
            baseline.latency_p99_std,
            baseline.n_samples,
            significance_level,
            regression_threshold_percent,
            higher_is_worse=True,
        )
    )

    results.append(
        detect_regression(
            "time_to_first_token",
            metrics.time_to_first_token,
            baseline.time_to_first_token_mean,
            baseline.time_to_first_token_std,
            baseline.n_samples,
            significance_level,
            regression_threshold_percent,
            higher_is_worse=True,
        )
    )

    results.append(
        detect_regression(
            "total_duration",
            metrics.total_duration,
            baseline.total_duration_mean,
            baseline.total_duration_std,
            baseline.n_samples,
            significance_level,
            regression_threshold_percent,
            higher_is_worse=True,
        )
    )

    # Throughput/efficiency metrics (higher is better)
    results.append(
        detect_regression(
            "tokens_per_second",
            metrics.tokens_per_second,
            baseline.tokens_per_second_mean,
            baseline.tokens_per_second_std,
            baseline.n_samples,
            significance_level,
            regression_threshold_percent,
            higher_is_worse=False,
        )
    )

    results.append(
        detect_regression(
            "token_efficiency",
            metrics.token_efficiency,
            baseline.token_efficiency_mean,
            baseline.token_efficiency_std,
            baseline.n_samples,
            significance_level,
            regression_threshold_percent,
            higher_is_worse=False,
        )
    )

    return results


class PerformanceEvaluator(Evaluator):
    """Evaluator for tracking and assessing agent performance metrics.

    This evaluator computes performance metrics from agent execution,
    checks them against configurable thresholds, and detects performance
    regressions compared to a baseline.

    Features:
    - Latency percentiles (p50, p95, p99) from LLM calls
    - Time to first token (TTFT) for streaming responses
    - Tokens per second throughput
    - Token efficiency ratio (output/total tokens)
    - Configurable thresholds for pass/fail
    - Regression detection using statistical tests

    Configuration options:
        thresholds: Dict with threshold values for metrics
        baseline: Dict with baseline statistics for regression detection
        significance_level: P-value threshold (default: 0.05)
        regression_threshold_percent: Percentage change threshold (default: 10%)
        check_latency: Whether to evaluate latency metrics (default: true)
        check_throughput: Whether to evaluate throughput (default: true)
        check_efficiency: Whether to evaluate efficiency (default: true)
        check_regression: Whether to check for regressions (default: true)
        fail_on_regression: Whether regression causes failure (default: false)

    Example usage:
        ```yaml
        assertions:
          - type: "performance"
            config:
              thresholds:
                max_latency_p95_ms: 2000
                min_tokens_per_second: 50
              check_regression: true
              fail_on_regression: true
        ```
    """

    def __init__(self) -> None:
        """Initialize the Performance evaluator."""
        pass

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "performance"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate agent performance metrics.

        Args:
            task: Test definition containing task details.
            response: ATP Response from the agent.
            trace: List of ATP Events from execution.
            assertion: Assertion configuration.

        Returns:
            EvalResult containing performance check results.
        """
        config = self._parse_config(assertion.config)
        metrics = compute_performance_metrics(response, trace)
        checks: list[EvalCheck] = []

        # Check thresholds
        threshold_results = check_thresholds(metrics, config.thresholds)
        threshold_passed = all(passed for _, passed, _ in threshold_results)

        for check_name, passed, message in threshold_results:
            checks.append(
                self._create_check(
                    name=f"threshold_{check_name}",
                    passed=passed,
                    message=message,
                )
            )

        # Check regressions
        regression_results: list[RegressionResult] = []
        has_regression = False

        if config.check_regression and config.baseline is not None:
            regression_results = check_regressions(
                metrics,
                config.baseline,
                config.significance_level,
                config.regression_threshold_percent,
            )

            for result in regression_results:
                if result.status == RegressionStatus.REGRESSION:
                    has_regression = True

            # Add regression check
            regression_message = self._build_regression_message(regression_results)
            regression_passed = not (config.fail_on_regression and has_regression)
            checks.append(
                self._create_check(
                    name="regression_check",
                    passed=regression_passed,
                    message=regression_message,
                    details={
                        "regressions": [
                            r.to_dict()
                            for r in regression_results
                            if r.status == RegressionStatus.REGRESSION
                        ],
                        "improvements": [
                            r.to_dict()
                            for r in regression_results
                            if r.status == RegressionStatus.IMPROVEMENT
                        ],
                    },
                )
            )

        # Create summary check with all metrics
        summary_passed = threshold_passed and not (
            config.fail_on_regression and has_regression
        )
        summary_message = self._build_summary_message(
            metrics, threshold_results, has_regression
        )

        checks.append(
            EvalCheck(
                name="performance_summary",
                passed=summary_passed,
                score=self._calculate_score(
                    threshold_results, regression_results, config
                ),
                message=summary_message,
                details={
                    "metrics": metrics.to_dict(),
                    "threshold_checks": [
                        {"name": name, "passed": passed, "message": msg}
                        for name, passed, msg in threshold_results
                    ],
                    "regressions": [r.to_dict() for r in regression_results],
                },
            )
        )

        return self._create_result(checks)

    def _parse_config(self, config: dict[str, Any]) -> PerformanceConfig:
        """Parse assertion config into PerformanceConfig.

        Args:
            config: Raw config dictionary.

        Returns:
            Parsed PerformanceConfig.
        """
        thresholds_data = config.get("thresholds", {})
        thresholds = PerformanceThresholds(**thresholds_data)

        baseline_data = config.get("baseline")
        baseline = None
        if baseline_data:
            baseline = PerformanceBaseline(**baseline_data)

        return PerformanceConfig(
            thresholds=thresholds,
            baseline=baseline,
            significance_level=config.get("significance_level", 0.05),
            regression_threshold_percent=config.get(
                "regression_threshold_percent", 10.0
            ),
            check_latency=config.get("check_latency", True),
            check_throughput=config.get("check_throughput", True),
            check_efficiency=config.get("check_efficiency", True),
            check_regression=config.get("check_regression", True),
            fail_on_regression=config.get("fail_on_regression", False),
        )

    def _build_summary_message(
        self,
        metrics: PerformanceMetrics,
        threshold_results: list[tuple[str, bool, str]],
        has_regression: bool,
    ) -> str:
        """Build summary message for performance check.

        Args:
            metrics: Computed metrics.
            threshold_results: Threshold check results.
            has_regression: Whether regressions were detected.

        Returns:
            Summary message string.
        """
        parts: list[str] = []

        if metrics.latency_p50 is not None:
            parts.append(f"p50={metrics.latency_p50:.0f}ms")
        if metrics.latency_p95 is not None:
            parts.append(f"p95={metrics.latency_p95:.0f}ms")
        if metrics.tokens_per_second is not None:
            parts.append(f"throughput={metrics.tokens_per_second:.1f}tok/s")

        threshold_fails = sum(1 for _, passed, _ in threshold_results if not passed)
        if threshold_fails > 0:
            parts.append(f"{threshold_fails} threshold(s) exceeded")

        if has_regression:
            parts.append("regression detected")

        return "; ".join(parts) if parts else "No metrics available"

    def _build_regression_message(
        self,
        regression_results: list[RegressionResult],
    ) -> str:
        """Build message summarizing regression results.

        Args:
            regression_results: List of regression results.

        Returns:
            Regression summary message.
        """
        regressions = [
            r for r in regression_results if r.status == RegressionStatus.REGRESSION
        ]
        improvements = [
            r for r in regression_results if r.status == RegressionStatus.IMPROVEMENT
        ]

        if not regressions and not improvements:
            return "No significant changes from baseline"

        parts: list[str] = []
        if regressions:
            names = [r.metric for r in regressions]
            parts.append(f"Regressions: {', '.join(names)}")
        if improvements:
            names = [r.metric for r in improvements]
            parts.append(f"Improvements: {', '.join(names)}")

        return "; ".join(parts)

    def _calculate_score(
        self,
        threshold_results: list[tuple[str, bool, str]],
        regression_results: list[RegressionResult],
        config: PerformanceConfig,
    ) -> float:
        """Calculate overall performance score.

        Args:
            threshold_results: Threshold check results.
            regression_results: Regression detection results.
            config: Evaluator configuration.

        Returns:
            Score from 0.0 to 1.0.
        """
        if not threshold_results and not regression_results:
            return 1.0

        # Threshold score: percentage of thresholds passed
        threshold_score = 1.0
        if threshold_results:
            passed_count = sum(1 for _, passed, _ in threshold_results if passed)
            threshold_score = passed_count / len(threshold_results)

        # Regression score: penalize for regressions
        regression_score = 1.0
        if regression_results:
            regression_count = sum(
                1 for r in regression_results if r.status == RegressionStatus.REGRESSION
            )
            # Each regression reduces score by 0.15, up to a max penalty of 0.5
            penalty = min(regression_count * 0.15, 0.5)
            regression_score = 1.0 - penalty

        # Weight threshold checks more heavily
        final_score = threshold_score * 0.7 + regression_score * 0.3
        return max(0.0, min(1.0, final_score))
