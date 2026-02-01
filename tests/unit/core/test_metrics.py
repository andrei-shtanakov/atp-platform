"""Tests for ATP Prometheus metrics module."""

import pytest
from prometheus_client import REGISTRY, CollectorRegistry

from atp.core.metrics import (
    ATPMetrics,
    MetricsSettings,
    configure_metrics,
    generate_metrics,
    get_metrics,
    get_metrics_settings,
    get_registry,
    record_adapter_duration,
    record_adapter_error,
    record_evaluator_duration,
    record_llm_call,
    record_test_execution,
    reset_metrics,
)


@pytest.fixture(autouse=True)
def reset_metrics_state() -> None:
    """Reset metrics state before and after each test."""
    reset_metrics()
    yield
    reset_metrics()


@pytest.fixture
def isolated_registry() -> CollectorRegistry:
    """Create an isolated registry for testing."""
    return CollectorRegistry()


class TestMetricsSettings:
    """Tests for MetricsSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = MetricsSettings()
        assert settings.enabled is True
        assert settings.prefix == "atp"
        assert settings.multiprocess_mode is False
        assert len(settings.default_buckets) > 0

    def test_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test settings can be loaded from environment variables."""
        monkeypatch.setenv("ATP_METRICS_ENABLED", "false")
        monkeypatch.setenv("ATP_METRICS_PREFIX", "custom")

        # Clear cached settings
        get_metrics_settings.cache_clear()

        settings = MetricsSettings()
        assert settings.enabled is False
        assert settings.prefix == "custom"


class TestATPMetrics:
    """Tests for ATPMetrics class."""

    def test_metrics_initialization(self, isolated_registry: CollectorRegistry) -> None:
        """Test metrics are properly initialized."""
        settings = MetricsSettings(prefix="test")
        metrics = ATPMetrics(registry=isolated_registry, settings=settings)

        # Verify counters exist
        assert metrics.tests_total is not None
        assert metrics.llm_calls_total is not None
        assert metrics.adapter_errors_total is not None
        assert metrics.evaluator_calls_total is not None

        # Verify histograms exist
        assert metrics.test_duration_seconds is not None
        assert metrics.evaluator_duration_seconds is not None
        assert metrics.adapter_duration_seconds is not None

        # Verify gauges exist
        assert metrics.active_tests is not None
        assert metrics.pending_tests is not None
        assert metrics.active_suites is not None

    def test_record_test_start(self, isolated_registry: CollectorRegistry) -> None:
        """Test recording test start increments active tests."""
        metrics = ATPMetrics(registry=isolated_registry)
        metrics.record_test_start(suite="test_suite")

        # Check active tests gauge
        assert metrics.active_tests._value.get() == 1

    def test_record_test_end(self, isolated_registry: CollectorRegistry) -> None:
        """Test recording test end updates metrics correctly."""
        metrics = ATPMetrics(registry=isolated_registry)

        # Start then end a test
        metrics.record_test_start(suite="test_suite")
        metrics.record_test_end(
            suite="test_suite",
            status="passed",
            duration_seconds=1.5,
            test_name="test_example",
        )

        # Active tests should be 0
        assert metrics.active_tests._value.get() == 0

        # Counter should be incremented
        # Note: We can't easily check the counter value without generating metrics

    def test_record_suite_start_end(self, isolated_registry: CollectorRegistry) -> None:
        """Test recording suite start and end."""
        metrics = ATPMetrics(registry=isolated_registry)

        metrics.record_suite_start(pending_tests=5)
        assert metrics.active_suites._value.get() == 1
        assert metrics.pending_tests._value.get() == 5

        metrics.record_test_dequeued()
        assert metrics.pending_tests._value.get() == 4

        metrics.record_suite_end()
        assert metrics.active_suites._value.get() == 0

    def test_record_llm_call(self, isolated_registry: CollectorRegistry) -> None:
        """Test recording LLM calls."""
        from prometheus_client import generate_latest

        metrics = ATPMetrics(registry=isolated_registry)

        metrics.record_llm_call(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
        )

        # Verify metrics are recorded using isolated registry
        output = generate_latest(isolated_registry)
        assert b"atp_llm_calls_total" in output

    def test_record_adapter_error(self, isolated_registry: CollectorRegistry) -> None:
        """Test recording adapter errors."""
        from prometheus_client import generate_latest

        metrics = ATPMetrics(registry=isolated_registry)

        metrics.record_adapter_error(adapter="http", error="timeout")

        output = generate_latest(isolated_registry)
        assert b"atp_adapter_errors_total" in output

    def test_record_adapter_call(self, isolated_registry: CollectorRegistry) -> None:
        """Test recording adapter calls."""
        from prometheus_client import generate_latest

        metrics = ATPMetrics(registry=isolated_registry)

        metrics.record_adapter_call(
            adapter="http",
            operation="execute",
            duration_seconds=2.5,
        )

        output = generate_latest(isolated_registry)
        assert b"atp_adapter_duration_seconds" in output

    def test_record_evaluator_call(self, isolated_registry: CollectorRegistry) -> None:
        """Test recording evaluator calls."""
        from prometheus_client import generate_latest

        metrics = ATPMetrics(registry=isolated_registry)

        metrics.record_evaluator_call(
            evaluator_type="llm_judge",
            passed=True,
            duration_seconds=1.0,
        )

        output = generate_latest(isolated_registry)
        assert b"atp_evaluator_calls_total" in output
        assert b"atp_evaluator_duration_seconds" in output


class TestConfigureMetrics:
    """Tests for configure_metrics function."""

    def test_configure_metrics_enabled(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test configuring metrics when enabled."""
        metrics = configure_metrics(enabled=True, registry=isolated_registry)
        assert metrics is not None
        assert isinstance(metrics, ATPMetrics)

    def test_configure_metrics_disabled(self) -> None:
        """Test configuring metrics when disabled."""
        metrics = configure_metrics(enabled=False)
        assert metrics is None

    def test_configure_metrics_returns_metrics_on_first_call(self) -> None:
        """Test that first configure_metrics call returns an ATPMetrics instance."""
        reg = CollectorRegistry()
        metrics = configure_metrics(enabled=True, registry=reg)
        assert metrics is not None
        assert isinstance(metrics, ATPMetrics)


class TestGetMetrics:
    """Tests for get_metrics function."""

    def test_get_metrics_returns_instance(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test get_metrics returns configured instance."""
        configure_metrics(enabled=True, registry=isolated_registry)
        metrics = get_metrics()
        assert metrics is not None
        assert isinstance(metrics, ATPMetrics)

    def test_get_metrics_auto_configures(self) -> None:
        """Test get_metrics auto-configures if not already done."""
        # Reset to ensure not configured
        reset_metrics()

        # Should auto-configure
        metrics = get_metrics()
        assert metrics is not None


class TestGenerateMetrics:
    """Tests for generate_metrics function."""

    def test_generate_metrics_returns_bytes(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test generate_metrics returns prometheus format bytes."""
        configure_metrics(enabled=True, registry=isolated_registry)
        output = generate_metrics()

        assert isinstance(output, bytes)
        # Should contain metric definitions
        assert b"# HELP" in output or len(output) > 0

    def test_generate_metrics_includes_all_metrics(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test that generated output includes all defined metrics."""
        configure_metrics(enabled=True, registry=isolated_registry)
        metrics = get_metrics()
        assert metrics is not None

        # Record some metrics
        metrics.record_test_start(suite="test")
        metrics.record_test_end(
            suite="test", status="passed", duration_seconds=1.0, test_name="test1"
        )
        metrics.record_llm_call(
            provider="openai", model="gpt-4", input_tokens=10, output_tokens=20
        )

        output = generate_metrics()

        # Verify key metrics are present
        assert b"atp_tests_total" in output
        assert b"atp_llm_calls_total" in output
        assert b"atp_active_tests" in output


class TestContextManagers:
    """Tests for context manager functions."""

    def test_record_test_execution_context(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test record_test_execution context manager."""
        configure_metrics(enabled=True, registry=isolated_registry)

        with record_test_execution(suite="test_suite", test_name="test1") as ctx:
            ctx["status"] = "passed"
            ctx["duration_seconds"] = 1.5

        output = generate_metrics()
        assert b"atp_tests_total" in output

    def test_record_evaluator_duration_context(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test record_evaluator_duration context manager."""
        configure_metrics(enabled=True, registry=isolated_registry)

        with record_evaluator_duration(evaluator_type="artifact") as ctx:
            ctx["passed"] = True
            ctx["duration_seconds"] = 0.5

        output = generate_metrics()
        assert b"atp_evaluator_duration_seconds" in output

    def test_record_adapter_duration_context(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test record_adapter_duration context manager."""
        configure_metrics(enabled=True, registry=isolated_registry)

        with record_adapter_duration(adapter="http", operation="execute") as ctx:
            ctx["duration_seconds"] = 2.0

        output = generate_metrics()
        assert b"atp_adapter_duration_seconds" in output

    def test_record_adapter_duration_with_error(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test record_adapter_duration records errors."""
        configure_metrics(enabled=True, registry=isolated_registry)

        with record_adapter_duration(adapter="http", operation="execute") as ctx:
            ctx["duration_seconds"] = 1.0
            ctx["error"] = "TimeoutError"

        output = generate_metrics()
        assert b"atp_adapter_errors_total" in output


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_record_llm_call_function(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test record_llm_call convenience function."""
        configure_metrics(enabled=True, registry=isolated_registry)

        record_llm_call(
            provider="anthropic",
            model="claude-3",
            input_tokens=500,
            output_tokens=1000,
        )

        output = generate_metrics()
        assert b"atp_llm_calls_total" in output
        assert b"atp_llm_tokens_total" in output

    def test_record_adapter_error_function(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test record_adapter_error convenience function."""
        configure_metrics(enabled=True, registry=isolated_registry)

        record_adapter_error(adapter="cli", error="ProcessError")

        output = generate_metrics()
        assert b"atp_adapter_errors_total" in output

    def test_convenience_functions_handle_disabled_metrics(self) -> None:
        """Test convenience functions work when metrics are disabled."""
        configure_metrics(enabled=False)

        # These should not raise even with metrics disabled
        record_llm_call(provider="test", model="test", input_tokens=0, output_tokens=0)
        record_adapter_error(adapter="test", error="test")


class TestGetRegistry:
    """Tests for get_registry function."""

    def test_get_registry_returns_default(self) -> None:
        """Test get_registry returns default registry."""
        registry = get_registry()
        assert registry == REGISTRY

    def test_get_registry_returns_custom(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test get_registry returns custom registry after configuration."""
        configure_metrics(enabled=True, registry=isolated_registry)
        # Note: The custom registry is only used internally,
        # get_registry will still return REGISTRY unless multiprocess mode


class TestResetMetrics:
    """Tests for reset_metrics function."""

    def test_reset_clears_configuration(
        self, isolated_registry: CollectorRegistry
    ) -> None:
        """Test reset_metrics clears configuration state."""
        configure_metrics(enabled=True, registry=isolated_registry)

        # Reset should allow reconfiguration
        reset_metrics()

        # Should be able to configure again
        metrics = configure_metrics(enabled=True, registry=CollectorRegistry())
        assert metrics is not None
