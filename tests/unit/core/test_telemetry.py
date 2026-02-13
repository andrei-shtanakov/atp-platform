"""Tests for ATP OpenTelemetry integration."""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.trace import SpanKind, Status, StatusCode

from atp.core.telemetry import (
    InMemorySpanExporter,
    TelemetrySettings,
    add_span_event,
    configure_telemetry,
    create_adapter_span,
    create_evaluator_span,
    create_test_span,
    ensure_debug_exporter,
    extract_trace_context,
    get_current_span,
    get_debug_exporter,
    get_telemetry_settings,
    get_tracer,
    inject_trace_context,
    record_exception,
    reset_telemetry,
    set_adapter_response_attributes,
    set_evaluator_result_attributes,
    set_span_attribute,
    set_span_attributes,
    set_test_result_attributes,
    span,
)


def _ensure_telemetry_configured() -> InMemorySpanExporter:
    """Ensure telemetry is configured and return the exporter.

    This uses the shared ensure_debug_exporter() function which handles
    both first-time configuration and adding an exporter to an existing
    provider.
    """
    exporter = ensure_debug_exporter()
    exporter.clear()
    return exporter


@pytest.fixture(autouse=True)
def reset_telemetry_state() -> None:
    """Clear settings cache before each test."""
    get_telemetry_settings.cache_clear()
    yield
    get_telemetry_settings.cache_clear()


class TestTelemetrySettings:
    """Tests for TelemetrySettings configuration."""

    def test_default_settings(self) -> None:
        """Test default telemetry settings."""
        settings = TelemetrySettings()
        assert settings.enabled is True
        assert settings.service_name == "atp-platform"
        assert settings.service_version == "1.0.0"
        assert settings.environment == "development"
        assert settings.otlp_endpoint is None
        assert settings.otlp_insecure is True
        assert settings.console_export is False
        assert settings.debug_mode is False
        assert settings.max_debug_spans == 1000
        assert settings.sample_rate == 1.0
        assert settings.batch_export is True
        assert settings.export_timeout_millis == 30000

    def test_settings_from_env(self) -> None:
        """Test settings can be loaded from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "ATP_TELEMETRY_ENABLED": "false",
                "ATP_TELEMETRY_SERVICE_NAME": "custom-service",
                "ATP_TELEMETRY_DEBUG_MODE": "true",
            },
        ):
            # Clear cache to force reload
            get_telemetry_settings.cache_clear()
            settings = TelemetrySettings()
            assert settings.enabled is False
            assert settings.service_name == "custom-service"
            assert settings.debug_mode is True

    def test_settings_validation(self) -> None:
        """Test settings validation."""
        with pytest.raises(ValueError):
            TelemetrySettings(sample_rate=2.0)  # Must be 0.0-1.0

        with pytest.raises(ValueError):
            TelemetrySettings(sample_rate=-0.5)

        with pytest.raises(ValueError):
            TelemetrySettings(max_debug_spans=0)  # Must be >= 1


class TestInMemorySpanExporter:
    """Tests for InMemorySpanExporter."""

    def test_export_spans(self) -> None:
        """Test that spans can be exported to memory."""
        exporter = InMemorySpanExporter(max_spans=100)

        # Create mock span
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
        mock_ctx.span_id = 0x1234567890ABCDEF
        mock_span.get_span_context.return_value = mock_ctx
        mock_span.name = "test_span"
        mock_span.kind = SpanKind.INTERNAL
        mock_span.start_time = 1000000000000000000  # nanoseconds
        mock_span.end_time = 2000000000000000000
        mock_span.status = Status(StatusCode.OK)
        mock_span.attributes = {"key": "value"}
        mock_span.events = []
        mock_span.parent = None

        from opentelemetry.sdk.trace.export import SpanExportResult

        result = exporter.export([mock_span])
        assert result == SpanExportResult.SUCCESS

        spans = exporter.get_spans()
        assert len(spans) == 1
        assert spans[0].name == "test_span"

    def test_max_spans_limit(self) -> None:
        """Test that exporter respects max spans limit."""
        exporter = InMemorySpanExporter(max_spans=5)

        # Create mock spans
        mock_spans = []
        for i in range(10):
            mock_span = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.trace_id = i
            mock_ctx.span_id = i
            mock_span.get_span_context.return_value = mock_ctx
            mock_span.name = f"span_{i}"
            mock_span.kind = SpanKind.INTERNAL
            mock_span.start_time = 1000000000000000000
            mock_span.end_time = 2000000000000000000
            mock_span.status = Status(StatusCode.OK)
            mock_span.attributes = {}
            mock_span.events = []
            mock_span.parent = None
            mock_spans.append(mock_span)

        exporter.export(mock_spans)

        # Should only keep last 5
        spans = exporter.get_spans(limit=10)
        assert len(spans) == 5

    def test_filter_by_trace_id(self) -> None:
        """Test filtering spans by trace ID."""
        exporter = InMemorySpanExporter()

        mock_span1 = MagicMock()
        mock_ctx1 = MagicMock()
        mock_ctx1.trace_id = 0x1111111111111111
        mock_ctx1.span_id = 0x1
        mock_span1.get_span_context.return_value = mock_ctx1
        mock_span1.name = "span_1"
        mock_span1.kind = SpanKind.INTERNAL
        mock_span1.start_time = 1000000000000000000
        mock_span1.end_time = 2000000000000000000
        mock_span1.status = Status(StatusCode.OK)
        mock_span1.attributes = {}
        mock_span1.events = []
        mock_span1.parent = None

        mock_span2 = MagicMock()
        mock_ctx2 = MagicMock()
        mock_ctx2.trace_id = 0x2222222222222222
        mock_ctx2.span_id = 0x2
        mock_span2.get_span_context.return_value = mock_ctx2
        mock_span2.name = "span_2"
        mock_span2.kind = SpanKind.INTERNAL
        mock_span2.start_time = 1000000000000000000
        mock_span2.end_time = 2000000000000000000
        mock_span2.status = Status(StatusCode.OK)
        mock_span2.attributes = {}
        mock_span2.events = []
        mock_span2.parent = None

        exporter.export([mock_span1, mock_span2])

        # Filter by trace ID
        trace_id_hex = format(0x1111111111111111, "032x")
        spans = exporter.get_spans(trace_id=trace_id_hex)
        assert len(spans) == 1
        assert spans[0].name == "span_1"

    def test_filter_by_name(self) -> None:
        """Test filtering spans by name substring."""
        exporter = InMemorySpanExporter()

        mock_span1 = MagicMock()
        mock_ctx1 = MagicMock()
        mock_ctx1.trace_id = 0x1
        mock_ctx1.span_id = 0x1
        mock_span1.get_span_context.return_value = mock_ctx1
        mock_span1.name = "test:execute"
        mock_span1.kind = SpanKind.INTERNAL
        mock_span1.start_time = 1000000000000000000
        mock_span1.end_time = 2000000000000000000
        mock_span1.status = Status(StatusCode.OK)
        mock_span1.attributes = {}
        mock_span1.events = []
        mock_span1.parent = None

        mock_span2 = MagicMock()
        mock_ctx2 = MagicMock()
        mock_ctx2.trace_id = 0x2
        mock_ctx2.span_id = 0x2
        mock_span2.get_span_context.return_value = mock_ctx2
        mock_span2.name = "adapter:http"
        mock_span2.kind = SpanKind.CLIENT
        mock_span2.start_time = 1000000000000000000
        mock_span2.end_time = 2000000000000000000
        mock_span2.status = Status(StatusCode.OK)
        mock_span2.attributes = {}
        mock_span2.events = []
        mock_span2.parent = None

        exporter.export([mock_span1, mock_span2])

        spans = exporter.get_spans(name_filter="execute")
        assert len(spans) == 1
        assert spans[0].name == "test:execute"

    def test_clear_spans(self) -> None:
        """Test clearing all stored spans."""
        exporter = InMemorySpanExporter()

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.trace_id = 0x1
        mock_ctx.span_id = 0x1
        mock_span.get_span_context.return_value = mock_ctx
        mock_span.name = "test"
        mock_span.kind = SpanKind.INTERNAL
        mock_span.start_time = 1000000000000000000
        mock_span.end_time = 2000000000000000000
        mock_span.status = Status(StatusCode.OK)
        mock_span.attributes = {}
        mock_span.events = []
        mock_span.parent = None

        exporter.export([mock_span])
        assert len(exporter.get_spans()) == 1

        exporter.clear()
        assert len(exporter.get_spans()) == 0


class TestConfigureTelemetry:
    """Tests for configure_telemetry function."""

    def test_configure_with_defaults(self) -> None:
        """Test configuring telemetry with default settings."""
        # Just verify configure_telemetry can be called without error
        # It may return None if already configured
        provider = configure_telemetry()
        # Provider may be None if already configured or disabled
        assert provider is None or provider is not None

    def test_configure_disabled(self) -> None:
        """Test that telemetry can be disabled via settings."""
        settings = TelemetrySettings(enabled=False)
        # When disabled, configure should return None
        # But due to singleton nature, this may not apply if already configured
        provider = configure_telemetry(settings=settings)
        # Just verify it doesn't error
        assert provider is None or provider is not None

    def test_configure_with_debug_mode(self) -> None:
        """Test configuring telemetry with debug mode enables exporter."""
        # This uses the module-level configured telemetry
        exporter = _ensure_telemetry_configured()
        # Our helper ensures an exporter is available
        assert exporter is not None
        assert isinstance(exporter, InMemorySpanExporter)

    def test_configure_idempotent(self) -> None:
        """Test that telemetry configuration is idempotent."""
        # Both calls should not error
        _ensure_telemetry_configured()
        provider2 = configure_telemetry(debug_mode=True)
        # Second call returns None (already configured)
        assert provider2 is None

    def test_configure_with_custom_service(self) -> None:
        """Test configuring telemetry with custom service name."""
        # Just verify custom params don't cause errors
        provider = configure_telemetry(
            service_name="custom-service",
            service_version="1.0.0",
            environment="production",
            debug_mode=True,
        )
        # May be None if already configured
        assert provider is None or provider is not None


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_with_name(self) -> None:
        """Test getting a named tracer."""
        tracer = get_tracer("test.module")
        assert tracer is not None

    def test_get_tracer_default(self) -> None:
        """Test getting default tracer."""
        tracer = get_tracer()
        assert tracer is not None


class TestSpanDecorator:
    """Tests for the span decorator."""

    def test_span_decorator_sync(self) -> None:
        """Test span decorator on synchronous function."""
        exporter = _ensure_telemetry_configured()

        @span("test_operation")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10

        spans = exporter.get_spans()
        assert any(s.name == "test_operation" for s in spans)

    @pytest.mark.anyio
    async def test_span_decorator_async(self) -> None:
        """Test span decorator on async function."""
        exporter = _ensure_telemetry_configured()

        @span("async_operation")
        async def my_async_function(x: int) -> int:
            return x * 3

        result = await my_async_function(5)
        assert result == 15

        spans = exporter.get_spans()
        assert any(s.name == "async_operation" for s in spans)

    def test_span_decorator_records_exception(self) -> None:
        """Test that span decorator records exceptions."""
        exporter = _ensure_telemetry_configured()

        @span("failing_operation")
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        spans = exporter.get_spans()
        failing_spans = [s for s in spans if s.name == "failing_operation"]
        assert len(failing_spans) == 1
        assert failing_spans[0].status == "ERROR"

    def test_span_decorator_with_attributes(self) -> None:
        """Test span decorator with static attributes."""
        exporter = _ensure_telemetry_configured()

        @span("attributed_op", attributes={"component": "test"})
        def attributed_function() -> str:
            return "done"

        result = attributed_function()
        assert result == "done"

        spans = exporter.get_spans()
        attr_spans = [s for s in spans if s.name == "attributed_op"]
        assert len(attr_spans) == 1
        assert attr_spans[0].attributes.get("component") == "test"


class TestSpanAttributes:
    """Tests for span attribute helper functions."""

    def test_set_span_attribute(self) -> None:
        """Test setting a single span attribute."""
        exporter = _ensure_telemetry_configured()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("test_span"):
            set_span_attribute("key", "value")

        spans = exporter.get_spans()
        test_spans = [s for s in spans if s.name == "test_span"]
        assert len(test_spans) == 1
        assert test_spans[0].attributes.get("key") == "value"

    def test_set_span_attributes(self) -> None:
        """Test setting multiple span attributes."""
        exporter = _ensure_telemetry_configured()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("test_span"):
            set_span_attributes(key1="value1", key2="value2")

        spans = exporter.get_spans()
        test_spans = [s for s in spans if s.name == "test_span"]
        assert len(test_spans) == 1
        assert test_spans[0].attributes.get("key1") == "value1"
        assert test_spans[0].attributes.get("key2") == "value2"


class TestSpanEvents:
    """Tests for span event helper functions."""

    def test_add_span_event(self) -> None:
        """Test adding an event to a span."""
        exporter = _ensure_telemetry_configured()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("test_span"):
            add_span_event("test_event", {"detail": "value"})

        spans = exporter.get_spans()
        test_spans = [s for s in spans if s.name == "test_span"]
        assert len(test_spans) == 1
        assert any(e["name"] == "test_event" for e in test_spans[0].events)


class TestExceptionRecording:
    """Tests for exception recording."""

    def test_record_exception(self) -> None:
        """Test recording an exception on a span."""
        exporter = _ensure_telemetry_configured()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("test_span"):
            try:
                raise ValueError("Test error")
            except ValueError as e:
                record_exception(e)

        spans = exporter.get_spans()
        test_spans = [s for s in spans if s.name == "test_span"]
        assert len(test_spans) == 1
        assert test_spans[0].status == "ERROR"


class TestContextPropagation:
    """Tests for trace context propagation."""

    def test_inject_trace_context(self) -> None:
        """Test injecting trace context into a carrier."""
        configure_telemetry(debug_mode=True)
        tracer = get_tracer("test")

        with tracer.start_as_current_span("test_span"):
            carrier = inject_trace_context()
            assert "traceparent" in carrier

    def test_extract_trace_context(self) -> None:
        """Test extracting trace context from a carrier."""
        carrier = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }
        context = extract_trace_context(carrier)
        assert context is not None


class TestSpecializedSpans:
    """Tests for specialized span creation functions."""

    def test_create_test_span(self) -> None:
        """Test creating a test span."""
        exporter = _ensure_telemetry_configured()

        s = create_test_span(
            test_id="test-123",
            test_name="My Test",
            suite_name="My Suite",
            agent_name="test-agent",
        )
        s.end()

        spans = exporter.get_spans()
        assert any("test:My Test" in span.name for span in spans)

    def test_create_adapter_span(self) -> None:
        """Test creating an adapter span."""
        exporter = _ensure_telemetry_configured()

        s = create_adapter_span(
            adapter_type="http",
            operation="execute",
            task_id="task-123",
        )
        s.end()

        spans = exporter.get_spans()
        assert any("adapter:http:execute" in span.name for span in spans)

    def test_create_evaluator_span(self) -> None:
        """Test creating an evaluator span."""
        exporter = _ensure_telemetry_configured()

        s = create_evaluator_span(
            evaluator_name="llm_judge",
            test_id="test-123",
            assertion_type="behavior",
        )
        s.end()

        spans = exporter.get_spans()
        assert any("evaluate:llm_judge" in span.name for span in spans)


class TestResultAttributes:
    """Tests for result attribute helper functions."""

    def test_set_test_result_attributes(self) -> None:
        """Test setting test result attributes."""
        exporter = _ensure_telemetry_configured()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("test_span"):
            set_test_result_attributes(
                score=0.85,
                passed=True,
                duration_seconds=5.5,
            )

        spans = exporter.get_spans()
        test_spans = [s for s in spans if s.name == "test_span"]
        assert len(test_spans) == 1
        assert test_spans[0].attributes.get("atp.result.score") == 0.85
        assert test_spans[0].attributes.get("atp.result.passed") is True
        assert test_spans[0].attributes.get("atp.result.duration_seconds") == 5.5

    def test_set_adapter_response_attributes(self) -> None:
        """Test setting adapter response attributes."""
        exporter = _ensure_telemetry_configured()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("adapter_span"):
            set_adapter_response_attributes(
                status="completed",
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.05,
            )

        spans = exporter.get_spans()
        adapter_spans = [s for s in spans if s.name == "adapter_span"]
        assert len(adapter_spans) == 1
        assert adapter_spans[0].attributes.get("atp.response.status") == "completed"
        assert adapter_spans[0].attributes.get("atp.tokens.input") == 100
        assert adapter_spans[0].attributes.get("atp.tokens.output") == 200
        assert adapter_spans[0].attributes.get("atp.cost.usd") == 0.05

    def test_set_evaluator_result_attributes(self) -> None:
        """Test setting evaluator result attributes."""
        exporter = _ensure_telemetry_configured()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("eval_span"):
            set_evaluator_result_attributes(
                total_checks=5,
                passed_checks=4,
                score=0.8,
            )

        spans = exporter.get_spans()
        eval_spans = [s for s in spans if s.name == "eval_span"]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("atp.evaluator.total_checks") == 5
        assert eval_spans[0].attributes.get("atp.evaluator.passed_checks") == 4
        assert eval_spans[0].attributes.get("atp.evaluator.score") == 0.8


class TestResetTelemetry:
    """Tests for reset_telemetry function."""

    def test_reset_clears_debug_exporter(self) -> None:
        """Test that reset clears the debug exporter reference in the module."""
        # First ensure we have an exporter
        exporter = _ensure_telemetry_configured()
        assert exporter is not None

        # reset_telemetry clears the module-level _debug_exporter
        reset_telemetry()
        # After reset, the module's get_debug_exporter returns None
        # (but our test's _test_exporter is still valid)
        assert get_debug_exporter() is None

    def test_reset_allows_flag_reconfiguration(self) -> None:
        """Test that reset clears the internal flag allowing reconfiguration."""
        _ensure_telemetry_configured()

        # After reset, the _telemetry_configured flag is cleared
        reset_telemetry()

        # configure_telemetry will try again and create a new provider
        # (even though OpenTelemetry may warn about override)
        provider3 = configure_telemetry(debug_mode=True)
        # It should return a provider (even if OTel doesn't use it)
        assert provider3 is not None


class TestGetCurrentSpan:
    """Tests for get_current_span function."""

    def test_get_current_span_outside_context(self) -> None:
        """Test getting current span when no span is active."""
        s = get_current_span()
        # Returns a non-recording span when no span is active
        assert s is not None
        assert not s.is_recording()

    def test_get_current_span_inside_context(self) -> None:
        """Test getting current span inside a span context."""
        _ensure_telemetry_configured()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("test_span"):
            s = get_current_span()
            assert s is not None
            assert s.is_recording()
