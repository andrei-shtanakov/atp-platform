"""Prometheus metrics for ATP Platform.

This module provides Prometheus metrics instrumentation for monitoring
the ATP platform's performance, usage, and health. It includes:
- Counters for tracking totals (tests, LLM calls, errors)
- Histograms for tracking durations (test execution, evaluation)
- Gauges for tracking current state (active tests, pending tests)
- Configuration settings for metrics collection
- Helper functions for instrumentation

Example usage:
    from atp.core.metrics import (
        configure_metrics,
        get_metrics,
        record_test_execution,
        record_adapter_call,
        record_evaluator_duration,
    )

    # Configure metrics at startup
    configure_metrics(enabled=True)

    # Record test execution
    with record_test_execution(suite="my_suite", status="passed"):
        # ... run test

    # Record adapter call
    record_adapter_call(adapter="http", provider="openai", model="gpt-4")

    # Record evaluator duration
    with record_evaluator_duration(evaluator_type="llm_judge"):
        # ... run evaluation
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache
from threading import Lock
from typing import Any

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Lock for thread-safe initialization
_init_lock = Lock()

# Global flag to track if metrics are configured
_metrics_configured = False

# Custom registry for ATP metrics (allows isolated testing)
_atp_registry: CollectorRegistry | None = None


class MetricsSettings(BaseSettings):
    """Metrics configuration settings.

    These settings control Prometheus metrics collection behavior.
    All settings can be overridden via environment variables with ATP_METRICS_ prefix.
    """

    enabled: bool = Field(
        default=True,
        description="Enable or disable metrics collection",
    )
    prefix: str = Field(
        default="atp",
        description="Prefix for all metric names",
    )
    default_buckets: list[float] = Field(
        default=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
        description="Default histogram buckets for duration metrics (seconds)",
    )
    multiprocess_mode: bool = Field(
        default=False,
        description="Enable multiprocess mode for gunicorn/uwsgi deployments",
    )

    model_config = {
        "env_prefix": "ATP_METRICS_",
        "env_file": ".env",
        "extra": "ignore",
    }


@lru_cache
def get_metrics_settings() -> MetricsSettings:
    """Get cached metrics settings.

    Returns:
        MetricsSettings instance.
    """
    return MetricsSettings()


class ATPMetrics:
    """Container for all ATP Prometheus metrics.

    This class holds all the metric objects and provides methods for
    recording metric values. It follows the singleton pattern to ensure
    only one set of metrics is registered with Prometheus.
    """

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        settings: MetricsSettings | None = None,
    ) -> None:
        """Initialize ATP metrics.

        Args:
            registry: Optional custom registry. Uses default if not provided.
            settings: Optional settings. Uses defaults if not provided.
        """
        self.registry = registry or REGISTRY
        self.settings = settings or get_metrics_settings()
        prefix = self.settings.prefix
        buckets = tuple(self.settings.default_buckets)

        # ===== Counters =====

        # Test execution counters
        self.tests_total = Counter(
            f"{prefix}_tests_total",
            "Total number of tests executed",
            ["suite", "status"],
            registry=self.registry,
        )

        # LLM call counters
        self.llm_calls_total = Counter(
            f"{prefix}_llm_calls_total",
            "Total number of LLM API calls",
            ["provider", "model"],
            registry=self.registry,
        )

        # Token usage counters
        self.llm_tokens_total = Counter(
            f"{prefix}_llm_tokens_total",
            "Total number of LLM tokens used",
            ["provider", "type"],  # type: input/output
            registry=self.registry,
        )

        # Adapter error counters
        self.adapter_errors_total = Counter(
            f"{prefix}_adapter_errors_total",
            "Total number of adapter errors",
            ["adapter", "error"],
            registry=self.registry,
        )

        # Evaluator call counters
        self.evaluator_calls_total = Counter(
            f"{prefix}_evaluator_calls_total",
            "Total number of evaluator invocations",
            ["type", "passed"],
            registry=self.registry,
        )

        # ===== Histograms =====

        # Test duration histogram
        self.test_duration_seconds = Histogram(
            f"{prefix}_test_duration_seconds",
            "Test execution duration in seconds",
            ["suite", "test"],
            buckets=buckets,
            registry=self.registry,
        )

        # Evaluator duration histogram
        self.evaluator_duration_seconds = Histogram(
            f"{prefix}_evaluator_duration_seconds",
            "Evaluator execution duration in seconds",
            ["type"],
            buckets=buckets,
            registry=self.registry,
        )

        # Adapter duration histogram
        self.adapter_duration_seconds = Histogram(
            f"{prefix}_adapter_duration_seconds",
            "Adapter execution duration in seconds",
            ["adapter", "operation"],
            buckets=buckets,
            registry=self.registry,
        )

        # ===== Gauges =====

        # Active test gauge
        self.active_tests = Gauge(
            f"{prefix}_active_tests",
            "Number of currently running tests",
            registry=self.registry,
        )

        # Pending test gauge
        self.pending_tests = Gauge(
            f"{prefix}_pending_tests",
            "Number of tests waiting to be executed",
            registry=self.registry,
        )

        # Active suite gauge
        self.active_suites = Gauge(
            f"{prefix}_active_suites",
            "Number of currently running test suites",
            registry=self.registry,
        )

    def record_test_start(
        self,
        suite: str = "default",
    ) -> None:
        """Record the start of a test execution.

        Args:
            suite: Test suite name.
        """
        self.active_tests.inc()

    def record_test_end(
        self,
        suite: str,
        status: str,
        duration_seconds: float,
        test_name: str = "unknown",
    ) -> None:
        """Record the completion of a test execution.

        Args:
            suite: Test suite name.
            status: Test status (passed, failed, timeout, error).
            duration_seconds: Test execution duration.
            test_name: Name of the test.
        """
        self.active_tests.dec()
        self.tests_total.labels(suite=suite, status=status).inc()
        self.test_duration_seconds.labels(suite=suite, test=test_name).observe(
            duration_seconds
        )

    def record_suite_start(self, pending_tests: int = 0) -> None:
        """Record the start of a test suite execution.

        Args:
            pending_tests: Number of tests in the suite.
        """
        self.active_suites.inc()
        self.pending_tests.inc(pending_tests)

    def record_suite_end(self) -> None:
        """Record the completion of a test suite execution."""
        self.active_suites.dec()

    def record_test_dequeued(self) -> None:
        """Record that a test has been dequeued for execution."""
        self.pending_tests.dec()

    def record_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record an LLM API call.

        Args:
            provider: LLM provider name (openai, anthropic, etc.).
            model: Model name (gpt-4, claude-3, etc.).
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        self.llm_calls_total.labels(provider=provider, model=model).inc()
        if input_tokens > 0:
            self.llm_tokens_total.labels(provider=provider, type="input").inc(
                input_tokens
            )
        if output_tokens > 0:
            self.llm_tokens_total.labels(provider=provider, type="output").inc(
                output_tokens
            )

    def record_adapter_error(
        self,
        adapter: str,
        error: str,
    ) -> None:
        """Record an adapter error.

        Args:
            adapter: Adapter type (http, cli, docker, etc.).
            error: Error type (timeout, connection, validation, etc.).
        """
        self.adapter_errors_total.labels(adapter=adapter, error=error).inc()

    def record_adapter_call(
        self,
        adapter: str,
        operation: str,
        duration_seconds: float,
    ) -> None:
        """Record an adapter call.

        Args:
            adapter: Adapter type.
            operation: Operation type (execute, stream, health_check).
            duration_seconds: Call duration.
        """
        self.adapter_duration_seconds.labels(
            adapter=adapter, operation=operation
        ).observe(duration_seconds)

    def record_evaluator_call(
        self,
        evaluator_type: str,
        passed: bool,
        duration_seconds: float,
    ) -> None:
        """Record an evaluator call.

        Args:
            evaluator_type: Evaluator type (artifact, llm_judge, etc.).
            passed: Whether the evaluation passed.
            duration_seconds: Evaluation duration.
        """
        self.evaluator_calls_total.labels(
            type=evaluator_type, passed=str(passed).lower()
        ).inc()
        self.evaluator_duration_seconds.labels(type=evaluator_type).observe(
            duration_seconds
        )


# Global metrics instance
_metrics: ATPMetrics | None = None


def configure_metrics(
    enabled: bool | None = None,
    settings: MetricsSettings | None = None,
    registry: CollectorRegistry | None = None,
) -> ATPMetrics | None:
    """Configure Prometheus metrics for ATP Platform.

    This function initializes the metrics collection system. It should be
    called once during application startup.

    Args:
        enabled: Override enabled setting. If None, uses settings value.
        settings: Complete settings override.
        registry: Custom registry for metrics. Useful for testing.

    Returns:
        Configured ATPMetrics instance, or None if disabled.
    """
    global _metrics_configured, _metrics, _atp_registry

    with _init_lock:
        if _metrics_configured:
            logger.debug("Metrics already configured, returning existing instance")
            return _metrics

        # Get settings
        if settings is None:
            settings = get_metrics_settings()

        # Check if enabled
        is_enabled = enabled if enabled is not None else settings.enabled
        if not is_enabled:
            logger.info("Metrics collection is disabled")
            _metrics_configured = True
            return None

        # Handle multiprocess mode
        if settings.multiprocess_mode:
            try:
                _atp_registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(_atp_registry)
                registry = _atp_registry
                logger.info("Multiprocess metrics mode enabled")
            except Exception as e:
                logger.warning(f"Failed to enable multiprocess mode: {e}")

        # Create metrics instance
        _metrics = ATPMetrics(registry=registry, settings=settings)
        _atp_registry = registry

        _metrics_configured = True
        logger.info("Prometheus metrics configured with prefix '%s'", settings.prefix)

        return _metrics


def reset_metrics() -> None:
    """Reset metrics configuration state.

    This is primarily useful for testing. Clears the global metrics
    instance and configuration flag, and unregisters all ATP metrics
    from the default Prometheus registry.
    """
    global _metrics_configured, _metrics, _atp_registry

    with _init_lock:
        # Unregister metrics from the default registry if they exist
        if _metrics is not None:
            _unregister_all_collectors(_metrics, REGISTRY)

        _metrics_configured = False
        _metrics = None
        _atp_registry = None
        get_metrics_settings.cache_clear()


def _unregister_all_collectors(
    metrics: ATPMetrics, registry: CollectorRegistry
) -> None:
    """Unregister all collectors from a registry.

    This is a helper function for test cleanup. It safely unregisters
    all metric collectors that were created by an ATPMetrics instance.

    Args:
        metrics: ATPMetrics instance with collectors to unregister.
        registry: CollectorRegistry to unregister from.
    """
    collectors_to_unregister = [
        # Counters
        metrics.tests_total,
        metrics.llm_calls_total,
        metrics.llm_tokens_total,
        metrics.adapter_errors_total,
        metrics.evaluator_calls_total,
        # Histograms
        metrics.test_duration_seconds,
        metrics.evaluator_duration_seconds,
        metrics.adapter_duration_seconds,
        # Gauges
        metrics.active_tests,
        metrics.pending_tests,
        metrics.active_suites,
    ]

    for collector in collectors_to_unregister:
        try:
            registry.unregister(collector)
        except Exception:
            # Collector might not be registered or already unregistered
            pass


def get_metrics() -> ATPMetrics | None:
    """Get the global metrics instance.

    Returns:
        ATPMetrics instance if configured and enabled, None otherwise.
    """
    global _metrics
    if not _metrics_configured:
        configure_metrics()
    return _metrics


def get_registry() -> CollectorRegistry:
    """Get the metrics registry.

    Returns:
        The CollectorRegistry being used for metrics.
    """
    if _atp_registry is not None:
        return _atp_registry
    return REGISTRY


def generate_metrics() -> bytes:
    """Generate metrics output in Prometheus format.

    Returns:
        Prometheus metrics output as bytes.
    """
    registry = get_registry()
    return generate_latest(registry)


# ===== Convenience Context Managers =====


@contextmanager
def record_test_execution(
    suite: str = "default",
    test_name: str = "unknown",
) -> Generator[dict[str, Any], None, None]:
    """Context manager for recording test execution metrics.

    Args:
        suite: Test suite name.
        test_name: Test name.

    Yields:
        Dictionary to be populated with execution results.
        Expected keys: 'status' (str), 'duration_seconds' (float).

    Example:
        with record_test_execution(suite="my_suite", test_name="test_1") as ctx:
            result = run_test()
            ctx['status'] = 'passed' if result.passed else 'failed'
            ctx['duration_seconds'] = result.duration
    """
    metrics = get_metrics()
    result: dict[str, Any] = {"status": "error", "duration_seconds": 0.0}

    if metrics:
        metrics.record_test_start(suite=suite)

    try:
        yield result
    finally:
        if metrics:
            metrics.record_test_end(
                suite=suite,
                status=result.get("status", "error"),
                duration_seconds=result.get("duration_seconds", 0.0),
                test_name=test_name,
            )


@contextmanager
def record_evaluator_duration(
    evaluator_type: str,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for recording evaluator duration metrics.

    Args:
        evaluator_type: Type of evaluator.

    Yields:
        Dictionary to be populated with evaluation results.
        Expected keys: 'passed' (bool), 'duration_seconds' (float).

    Example:
        with record_evaluator_duration(evaluator_type="llm_judge") as ctx:
            result = evaluator.evaluate(...)
            ctx['passed'] = result.passed
            ctx['duration_seconds'] = elapsed_time
    """
    metrics = get_metrics()
    result: dict[str, Any] = {"passed": False, "duration_seconds": 0.0}

    try:
        yield result
    finally:
        if metrics:
            metrics.record_evaluator_call(
                evaluator_type=evaluator_type,
                passed=result.get("passed", False),
                duration_seconds=result.get("duration_seconds", 0.0),
            )


@contextmanager
def record_adapter_duration(
    adapter: str,
    operation: str = "execute",
) -> Generator[dict[str, Any], None, None]:
    """Context manager for recording adapter duration metrics.

    Args:
        adapter: Adapter type.
        operation: Operation being performed.

    Yields:
        Dictionary to be populated with call results.
        Expected keys: 'duration_seconds' (float), 'error' (str | None).

    Example:
        with record_adapter_duration(adapter="http", operation="execute") as ctx:
            try:
                result = adapter.execute(request)
                ctx['duration_seconds'] = elapsed_time
            except Exception as e:
                ctx['error'] = type(e).__name__
                raise
    """
    metrics = get_metrics()
    result: dict[str, Any] = {"duration_seconds": 0.0, "error": None}

    try:
        yield result
    finally:
        if metrics:
            metrics.record_adapter_call(
                adapter=adapter,
                operation=operation,
                duration_seconds=result.get("duration_seconds", 0.0),
            )
            error = result.get("error")
            if error:
                metrics.record_adapter_error(adapter=adapter, error=error)


# ===== Convenience Functions =====


def record_llm_call(
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Record an LLM API call.

    Args:
        provider: LLM provider name.
        model: Model name.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
    """
    metrics = get_metrics()
    if metrics:
        metrics.record_llm_call(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def record_adapter_error(adapter: str, error: str) -> None:
    """Record an adapter error.

    Args:
        adapter: Adapter type.
        error: Error type.
    """
    metrics = get_metrics()
    if metrics:
        metrics.record_adapter_error(adapter=adapter, error=error)
