"""OpenTelemetry integration for ATP Platform.

This module provides distributed tracing capabilities using OpenTelemetry.
It includes:
- Tracer configuration and initialization
- OTLP exporter configuration for external trace backends
- In-memory span storage for debugging (dev mode)
- Context propagation utilities
- Span decorators and helpers for instrumentation

Example usage:
    from atp.core.telemetry import configure_telemetry, get_tracer, span

    # Configure telemetry at startup
    configure_telemetry(service_name="atp-runner")

    # Get a tracer for your module
    tracer = get_tracer(__name__)

    # Create spans manually
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("test_id", "test-123")
        # ... do work

    # Or use the decorator
    @span("process_task")
    async def process_task(task_id: str):
        # ... do work
"""

import logging
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache, wraps
from threading import Lock
from typing import Any, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    get_tracer_provider,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Type variables for generic decorator
P = ParamSpec("P")
R = TypeVar("R")

# Context variable for propagating trace context
_trace_context: ContextVar[dict[str, str]] = ContextVar("trace_context", default={})

# Global flag to track if telemetry is configured
_telemetry_configured = False

# Lock for thread-safe initialization
_init_lock = Lock()


class TelemetrySettings(BaseSettings):
    """Telemetry configuration settings.

    These settings control OpenTelemetry tracing behavior.
    All settings can be overridden via environment variables with ATP_TELEMETRY_ prefix.
    """

    enabled: bool = Field(
        default=True,
        description="Enable or disable telemetry collection",
    )
    service_name: str = Field(
        default="atp-platform",
        description="Service name for trace identification",
    )
    service_version: str = Field(
        default="0.1.0",
        description="Service version for trace metadata",
    )
    environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production)",
    )
    otlp_endpoint: str | None = Field(
        default=None,
        description="OTLP exporter endpoint (e.g., http://localhost:4317)",
    )
    otlp_insecure: bool = Field(
        default=True,
        description="Use insecure connection for OTLP exporter",
    )
    console_export: bool = Field(
        default=False,
        description="Export spans to console (for debugging)",
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with in-memory span storage",
    )
    max_debug_spans: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of spans to store in debug mode",
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sampling rate for traces (0.0 to 1.0)",
    )
    batch_export: bool = Field(
        default=True,
        description="Use batch span processor for better performance",
    )
    export_timeout_millis: int = Field(
        default=30000,
        ge=1000,
        description="Timeout for span export in milliseconds",
    )

    model_config = {
        "env_prefix": "ATP_TELEMETRY_",
        "env_file": ".env",
        "extra": "ignore",
    }


@dataclass
class SpanData:
    """Simplified span data for API responses."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    start_time: datetime
    end_time: datetime | None
    status: str
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


class InMemorySpanExporter(SpanExporter):
    """Span exporter that stores spans in memory for debugging.

    This exporter is intended for development and debugging purposes.
    It maintains a fixed-size buffer of recent spans that can be queried
    via the /traces debug endpoint.
    """

    def __init__(self, max_spans: int = 1000) -> None:
        """Initialize the in-memory exporter.

        Args:
            max_spans: Maximum number of spans to store.
        """
        self._spans: list[ReadableSpan] = []
        self._max_spans = max_spans
        self._lock = Lock()

    def export(
        self,
        spans: Sequence[ReadableSpan],  # type: ignore[override]
    ) -> SpanExportResult:
        """Export spans to in-memory storage.

        Args:
            spans: Spans to export.

        Returns:
            SpanExportResult indicating success.
        """
        with self._lock:
            self._spans.extend(spans)
            # Trim to max size
            if len(self._spans) > self._max_spans:
                self._spans = self._spans[-self._max_spans :]
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown the exporter (no-op for in-memory)."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush (no-op for in-memory).

        Args:
            timeout_millis: Timeout in milliseconds (unused).

        Returns:
            Always returns True.
        """
        return True

    def get_spans(
        self,
        limit: int | None = None,
        trace_id: str | None = None,
        name_filter: str | None = None,
    ) -> list[SpanData]:
        """Get stored spans with optional filtering.

        Args:
            limit: Maximum number of spans to return.
            trace_id: Filter by trace ID.
            name_filter: Filter by span name substring.

        Returns:
            List of SpanData objects.
        """
        with self._lock:
            spans = self._spans.copy()

        result: list[SpanData] = []
        for span in reversed(spans):  # Most recent first
            ctx = span.get_span_context()
            # SpanContext is not None for valid spans
            trace_id_hex = format(ctx.trace_id, "032x")  # type: ignore[union-attr]

            # Apply filters
            if trace_id and trace_id_hex != trace_id:
                continue
            if name_filter and name_filter.lower() not in span.name.lower():
                continue

            # Convert attributes to serializable format
            attributes: dict[str, Any] = {}
            if span.attributes:
                for key, value in span.attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        attributes[key] = value
                    elif isinstance(value, (list, tuple)):
                        attributes[key] = list(value)
                    else:
                        attributes[key] = str(value)

            # Convert events
            events = []
            if span.events:
                for event in span.events:
                    event_data: dict[str, Any] = {
                        "name": event.name,
                        "timestamp": datetime.fromtimestamp(
                            event.timestamp / 1e9
                        ).isoformat(),
                    }
                    if event.attributes:
                        event_data["attributes"] = dict(event.attributes)
                    events.append(event_data)

            # Determine parent span ID
            parent_span_id = None
            if span.parent:
                parent_span_id = format(span.parent.span_id, "016x")

            # Get status
            status_str = "UNSET"
            if span.status:
                status_str = span.status.status_code.name

            # Handle start/end times which may be None
            start_ts = span.start_time or 0
            end_ts = span.end_time

            result.append(
                SpanData(
                    trace_id=trace_id_hex,
                    span_id=format(ctx.span_id, "016x"),  # type: ignore[union-attr]
                    parent_span_id=parent_span_id,
                    name=span.name,
                    kind=span.kind.name if span.kind else "INTERNAL",
                    start_time=datetime.fromtimestamp(start_ts / 1e9),
                    end_time=(datetime.fromtimestamp(end_ts / 1e9) if end_ts else None),
                    status=status_str,
                    attributes=attributes,
                    events=events,
                )
            )

            if limit and len(result) >= limit:
                break

        return result

    def clear(self) -> None:
        """Clear all stored spans."""
        with self._lock:
            self._spans.clear()


# Global in-memory exporter instance for debug mode
_debug_exporter: InMemorySpanExporter | None = None

# Singleton exporter that persists across reset_telemetry calls
# (because OpenTelemetry doesn't allow TracerProvider override)
_singleton_exporter: InMemorySpanExporter | None = None
_singleton_attached: bool = False


def get_debug_exporter() -> InMemorySpanExporter | None:
    """Get the debug span exporter if available.

    Returns:
        InMemorySpanExporter if debug mode is enabled, None otherwise.
    """
    return _debug_exporter


@lru_cache
def get_telemetry_settings() -> TelemetrySettings:
    """Get cached telemetry settings.

    Returns:
        TelemetrySettings instance.
    """
    return TelemetrySettings()


def configure_telemetry(
    service_name: str | None = None,
    service_version: str | None = None,
    environment: str | None = None,
    otlp_endpoint: str | None = None,
    debug_mode: bool | None = None,
    console_export: bool | None = None,
    settings: TelemetrySettings | None = None,
) -> TracerProvider | None:
    """Configure OpenTelemetry tracing for ATP Platform.

    This function initializes the OpenTelemetry SDK with appropriate exporters
    based on the configuration. It should be called once during application
    startup.

    Args:
        service_name: Service name override.
        service_version: Service version override.
        environment: Environment override.
        otlp_endpoint: OTLP endpoint override.
        debug_mode: Debug mode override.
        console_export: Console export override.
        settings: Complete settings override.

    Returns:
        Configured TracerProvider, or None if telemetry is disabled.
    """
    global _telemetry_configured, _debug_exporter

    with _init_lock:
        if _telemetry_configured:
            logger.debug("Telemetry already configured, skipping")
            return None

        # Get settings
        if settings is None:
            settings = get_telemetry_settings()

        # Apply overrides
        actual_service_name = service_name or settings.service_name
        actual_service_version = service_version or settings.service_version
        actual_environment = environment or settings.environment
        actual_otlp_endpoint = otlp_endpoint or settings.otlp_endpoint
        actual_debug_mode = (
            debug_mode if debug_mode is not None else settings.debug_mode
        )
        actual_console_export = (
            console_export if console_export is not None else settings.console_export
        )

        if not settings.enabled:
            logger.info("Telemetry is disabled")
            return None

        # Create resource with service information
        resource = Resource.create(
            {
                "service.name": actual_service_name,
                "service.version": actual_service_version,
                "deployment.environment": actual_environment,
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python",
            }
        )

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add exporters
        exporters_added = False

        # OTLP exporter for production
        if actual_otlp_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=actual_otlp_endpoint,
                    insecure=settings.otlp_insecure,
                    timeout=settings.export_timeout_millis // 1000,
                )
                if settings.batch_export:
                    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                else:
                    provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
                exporters_added = True
                logger.info("OTLP exporter configured: %s", actual_otlp_endpoint)
            except Exception as e:
                logger.warning("Failed to configure OTLP exporter: %s", e)

        # Console exporter for debugging
        if actual_console_export:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(SimpleSpanProcessor(console_exporter))
            exporters_added = True
            logger.info("Console span exporter enabled")

        # In-memory exporter for debug endpoint
        if actual_debug_mode:
            _debug_exporter = InMemorySpanExporter(max_spans=settings.max_debug_spans)
            provider.add_span_processor(SimpleSpanProcessor(_debug_exporter))
            exporters_added = True
            logger.info(
                "Debug mode enabled with in-memory span storage (max %d spans)",
                settings.max_debug_spans,
            )

        if not exporters_added:
            # Add a no-op console exporter to avoid dropping spans entirely
            logger.debug("No exporters configured, spans will not be exported")

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        _telemetry_configured = True
        logger.info(
            "Telemetry configured for service=%s, version=%s, env=%s",
            actual_service_name,
            actual_service_version,
            actual_environment,
        )

        return provider


def reset_telemetry() -> None:
    """Reset telemetry configuration state.

    This is primarily useful for testing. Note that due to OpenTelemetry's
    design, the global TracerProvider cannot be truly replaced once set.
    This function clears internal flags to allow re-configuration attempts,
    but the original provider may still be in use.

    For tests, prefer using ensure_debug_exporter() and calling clear()
    on the returned exporter to reset span collection between tests.
    """
    global _telemetry_configured, _debug_exporter

    with _init_lock:
        # Clear the debug exporter if it exists (but keep spans)
        # Tests should call exporter.clear() themselves
        _telemetry_configured = False
        _debug_exporter = None

        # Clear the settings cache
        get_telemetry_settings.cache_clear()


def add_exporter_to_provider(exporter: SpanExporter) -> bool:
    """Add an exporter to the current tracer provider.

    This is useful in testing scenarios where telemetry is already configured
    but you need to add an additional exporter to capture spans.

    Args:
        exporter: The span exporter to add.

    Returns:
        True if the exporter was successfully added, False otherwise.
    """
    try:
        current_provider = get_tracer_provider()
        if isinstance(current_provider, TracerProvider):
            current_provider.add_span_processor(SimpleSpanProcessor(exporter))
            return True
        return False
    except Exception as e:
        logger.warning("Failed to add exporter to provider: %s", e)
        return False


def ensure_debug_exporter() -> InMemorySpanExporter:
    """Ensure a debug exporter is available and attached to the provider.

    This function implements a singleton pattern for the debug exporter to
    handle OpenTelemetry's restriction that TracerProvider can only be set once.
    Once an exporter is created and attached, it will be reused for the
    lifetime of the process, even across reset_telemetry() calls.

    For tests, call exporter.clear() to reset the collected spans.

    Returns:
        The InMemorySpanExporter instance.
    """
    global _debug_exporter, _singleton_exporter, _singleton_attached

    # Fast path - return existing singleton
    if _singleton_exporter is not None:
        # Update the module-level reference in case it was cleared by reset
        _debug_exporter = _singleton_exporter
        return _singleton_exporter

    with _init_lock:
        # Double-check after acquiring lock
        if _singleton_exporter is not None:
            _debug_exporter = _singleton_exporter
            return _singleton_exporter

        # Create the singleton exporter
        settings = get_telemetry_settings()
        exporter = InMemorySpanExporter(max_spans=settings.max_debug_spans)
        _singleton_exporter = exporter
        _debug_exporter = exporter

        # Ensure telemetry is configured
        current_provider = get_tracer_provider()
        if not isinstance(current_provider, TracerProvider):
            # Configure telemetry to get an SDK provider
            # Note: we temporarily release lock to avoid deadlock
            pass  # We'll configure outside the lock

    # Configure outside the lock if needed
    current_provider = get_tracer_provider()
    if not isinstance(current_provider, TracerProvider):
        configure_telemetry(debug_mode=True)
        current_provider = get_tracer_provider()

    # Attach the exporter to the provider (idempotent check)
    with _init_lock:
        if not _singleton_attached and isinstance(current_provider, TracerProvider):
            current_provider.add_span_processor(
                SimpleSpanProcessor(_singleton_exporter)
            )
            _singleton_attached = True
            logger.debug("Attached debug exporter to tracer provider")

    return _singleton_exporter  # type: ignore[return-value]


def get_tracer(name: str | None = None) -> Tracer:
    """Get a tracer instance.

    Args:
        name: Tracer name (typically __name__). If None, returns default tracer.

    Returns:
        Tracer instance for creating spans.
    """
    return trace.get_tracer(name or "atp")


def get_current_span() -> Span:
    """Get the current active span.

    Returns:
        The current span, or a non-recording span if none is active.
    """
    return trace.get_current_span()


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current span.

    This is a convenience function for adding attributes to the currently
    active span without having to obtain a reference to it first.

    Args:
        key: Attribute key.
        value: Attribute value (must be a primitive type).
    """
    span = get_current_span()
    if span.is_recording():
        span.set_attribute(key, value)


def set_span_attributes(**attributes: Any) -> None:
    """Set multiple attributes on the current span.

    Args:
        **attributes: Key-value pairs to set as attributes.
    """
    span = get_current_span()
    if span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def add_span_event(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Add an event to the current span.

    Events are timestamped annotations that can be used to record notable
    occurrences during a span's lifetime.

    Args:
        name: Event name.
        attributes: Optional event attributes.
    """
    span = get_current_span()
    if span.is_recording():
        span.add_event(name, attributes=attributes)


def record_exception(
    exception: BaseException,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record an exception on the current span.

    This marks the span as having an error and records exception details.

    Args:
        exception: Exception to record.
        attributes: Optional additional attributes.
    """
    span = get_current_span()
    if span.is_recording():
        span.record_exception(exception, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(exception)))


# Context propagation utilities

_propagator = TraceContextTextMapPropagator()


def inject_trace_context(carrier: dict[str, str] | None = None) -> dict[str, str]:
    """Inject trace context into a carrier for propagation.

    Use this when making external calls (HTTP requests, queue messages)
    to propagate the trace context.

    Args:
        carrier: Optional carrier dict. Creates new dict if None.

    Returns:
        Carrier dict with trace context headers.
    """
    if carrier is None:
        carrier = {}
    _propagator.inject(carrier)
    return carrier


def extract_trace_context(carrier: dict[str, str]) -> Context:
    """Extract trace context from a carrier.

    Use this when receiving external calls to extract and continue
    an existing trace.

    Args:
        carrier: Carrier dict with trace context headers.

    Returns:
        Context object to be used with attach() or start_span().
    """
    return _propagator.extract(carrier)


# Decorator for creating spans


def span(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
    record_exceptions: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to create a span for a function.

    This decorator wraps a function to automatically create a span when
    the function is called. It supports both sync and async functions.

    Args:
        name: Span name. Defaults to function name if not provided.
        kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER).
        attributes: Static attributes to add to the span.
        record_exceptions: Whether to automatically record exceptions.

    Returns:
        Decorated function.

    Example:
        @span("process_request", attributes={"component": "api"})
        async def process_request(request_id: str):
            set_span_attribute("request_id", request_id)
            # ... process request
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        import asyncio

        span_name = name or func.__name__
        tracer = get_tracer(func.__module__)

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with tracer.start_as_current_span(
                    span_name,
                    kind=kind,
                    attributes=attributes,
                ) as s:
                    try:
                        result = await func(*args, **kwargs)  # type: ignore[misc]
                        return result
                    except Exception as e:
                        if record_exceptions:
                            s.record_exception(e)
                            s.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return async_wrapper  # type: ignore[return-value]
        else:

            @wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with tracer.start_as_current_span(
                    span_name,
                    kind=kind,
                    attributes=attributes,
                ) as s:
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        if record_exceptions:
                            s.record_exception(e)
                            s.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# Convenience functions for common span patterns


def create_test_span(
    test_id: str,
    test_name: str,
    suite_name: str | None = None,
    agent_name: str | None = None,
) -> Span:
    """Create a span for a test execution.

    Args:
        test_id: Unique test identifier.
        test_name: Human-readable test name.
        suite_name: Optional test suite name.
        agent_name: Optional agent name being tested.

    Returns:
        Started span for the test.
    """
    tracer = get_tracer("atp.runner")
    attributes: dict[str, Any] = {
        "atp.test.id": test_id,
        "atp.test.name": test_name,
    }
    if suite_name:
        attributes["atp.suite.name"] = suite_name
    if agent_name:
        attributes["atp.agent.name"] = agent_name

    return tracer.start_span(
        f"test:{test_name}",
        kind=SpanKind.INTERNAL,
        attributes=attributes,
    )


def create_adapter_span(
    adapter_type: str,
    operation: str,
    task_id: str | None = None,
) -> Span:
    """Create a span for an adapter operation.

    Args:
        adapter_type: Type of adapter (http, cli, etc.).
        operation: Operation being performed (execute, stream, etc.).
        task_id: Optional task ID.

    Returns:
        Started span for the adapter operation.
    """
    tracer = get_tracer("atp.adapters")
    attributes: dict[str, Any] = {
        "atp.adapter.type": adapter_type,
        "atp.adapter.operation": operation,
    }
    if task_id:
        attributes["atp.task.id"] = task_id

    return tracer.start_span(
        f"adapter:{adapter_type}:{operation}",
        kind=SpanKind.CLIENT,
        attributes=attributes,
    )


def create_evaluator_span(
    evaluator_name: str,
    test_id: str,
    assertion_type: str | None = None,
) -> Span:
    """Create a span for an evaluator operation.

    Args:
        evaluator_name: Name of the evaluator.
        test_id: Test ID being evaluated.
        assertion_type: Optional assertion type.

    Returns:
        Started span for the evaluator operation.
    """
    tracer = get_tracer("atp.evaluators")
    attributes: dict[str, Any] = {
        "atp.evaluator.name": evaluator_name,
        "atp.test.id": test_id,
    }
    if assertion_type:
        attributes["atp.assertion.type"] = assertion_type

    return tracer.start_span(
        f"evaluate:{evaluator_name}",
        kind=SpanKind.INTERNAL,
        attributes=attributes,
    )


def set_test_result_attributes(
    score: float,
    passed: bool,
    duration_seconds: float | None = None,
    error: str | None = None,
) -> None:
    """Set test result attributes on the current span.

    Args:
        score: Test score (0.0 to 1.0).
        passed: Whether the test passed.
        duration_seconds: Optional test duration.
        error: Optional error message.
    """
    span = get_current_span()
    if span.is_recording():
        span.set_attribute("atp.result.score", score)
        span.set_attribute("atp.result.passed", passed)
        if duration_seconds is not None:
            span.set_attribute("atp.result.duration_seconds", duration_seconds)
        if error:
            span.set_attribute("atp.result.error", error)
            span.set_status(Status(StatusCode.ERROR, error))
        else:
            span.set_status(Status(StatusCode.OK))


def set_adapter_response_attributes(
    status: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> None:
    """Set adapter response attributes on the current span.

    Args:
        status: Response status (completed, failed, timeout).
        input_tokens: Optional input token count.
        output_tokens: Optional output token count.
        cost_usd: Optional cost in USD.
    """
    span = get_current_span()
    if span.is_recording():
        span.set_attribute("atp.response.status", status)
        if input_tokens is not None:
            span.set_attribute("atp.tokens.input", input_tokens)
        if output_tokens is not None:
            span.set_attribute("atp.tokens.output", output_tokens)
        if cost_usd is not None:
            span.set_attribute("atp.cost.usd", cost_usd)


def set_evaluator_result_attributes(
    total_checks: int,
    passed_checks: int,
    score: float,
) -> None:
    """Set evaluator result attributes on the current span.

    Args:
        total_checks: Total number of checks performed.
        passed_checks: Number of checks that passed.
        score: Average score across all checks.
    """
    span = get_current_span()
    if span.is_recording():
        span.set_attribute("atp.evaluator.total_checks", total_checks)
        span.set_attribute("atp.evaluator.passed_checks", passed_checks)
        span.set_attribute("atp.evaluator.score", score)
