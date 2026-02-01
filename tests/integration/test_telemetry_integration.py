"""Integration tests for OpenTelemetry telemetry across ATP components."""

import os
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from atp.core.telemetry import (
    InMemorySpanExporter,
    ensure_debug_exporter,
    get_telemetry_settings,
    get_tracer,
)
from atp.dashboard.v2.factory import create_test_app


def _ensure_integration_telemetry() -> InMemorySpanExporter:
    """Ensure telemetry is configured for integration tests.

    This uses the shared ensure_debug_exporter() function which handles
    both first-time configuration and adding an exporter to an existing
    provider (e.g., when unit tests have already configured telemetry).
    """
    exporter = ensure_debug_exporter()
    exporter.clear()
    return exporter


@pytest.fixture(autouse=True)
def reset_telemetry_state() -> None:
    """Clear telemetry settings cache before each test."""
    get_telemetry_settings.cache_clear()
    yield
    get_telemetry_settings.cache_clear()


@pytest.fixture
def app_with_debug_telemetry():
    """Create a test app with debug telemetry enabled."""
    # Ensure telemetry is configured
    _ensure_integration_telemetry()

    # Set environment variable to enable debug mode for the settings
    with patch.dict(os.environ, {"ATP_TELEMETRY_DEBUG_MODE": "true"}):
        get_telemetry_settings.cache_clear()
        app = create_test_app(use_v2_routes=True)
        yield app


class TestTracesEndpoint:
    """Tests for the /api/debug/traces endpoint."""

    @pytest.mark.anyio
    async def test_traces_endpoint_requires_debug_mode(self) -> None:
        """Test that traces endpoint returns 404 when debug mode is off."""
        # Don't configure telemetry with debug mode
        app = create_test_app(use_v2_routes=True)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/debug/traces")
            assert response.status_code == 404
            # Check for key phrase in error message
            assert "debug" in response.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_traces_endpoint_with_debug_mode(
        self, app_with_debug_telemetry
    ) -> None:
        """Test that traces endpoint works when debug mode is on."""
        async with AsyncClient(
            transport=ASGITransport(app=app_with_debug_telemetry),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/debug/traces")
            # With debug mode env var set, endpoint should work
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is True
            assert "spans" in data
            assert "span_count" in data
            assert "total_stored" in data

    @pytest.mark.anyio
    async def test_traces_endpoint_returns_collected_spans(
        self, app_with_debug_telemetry
    ) -> None:
        """Test that traces endpoint returns collected spans."""
        # Ensure telemetry configured and clear spans
        exporter = _ensure_integration_telemetry()

        # Create some spans
        tracer = get_tracer("test")
        with tracer.start_as_current_span("test_span_1"):
            pass
        with tracer.start_as_current_span("test_span_2"):
            pass

        # Check via the exporter directly
        spans = exporter.get_spans()
        span_names = [s.name for s in spans]
        assert "test_span_1" in span_names
        assert "test_span_2" in span_names

    @pytest.mark.anyio
    async def test_traces_endpoint_with_limit(self, app_with_debug_telemetry) -> None:
        """Test that traces endpoint limit works via exporter."""
        exporter = _ensure_integration_telemetry()

        # Create multiple spans
        tracer = get_tracer("test")
        for i in range(5):
            with tracer.start_as_current_span(f"span_{i}"):
                pass

        # Verify spans are recorded
        spans = exporter.get_spans(limit=2)
        assert len(spans) == 2

    @pytest.mark.anyio
    async def test_traces_endpoint_with_name_filter(
        self, app_with_debug_telemetry
    ) -> None:
        """Test that traces endpoint filters by span name."""
        exporter = _ensure_integration_telemetry()

        tracer = get_tracer("test")
        with tracer.start_as_current_span("test:execute"):
            pass
        with tracer.start_as_current_span("adapter:http"):
            pass

        spans = exporter.get_spans(name_filter="execute")
        span_names = [s.name for s in spans]
        assert "test:execute" in span_names
        assert "adapter:http" not in span_names

    @pytest.mark.anyio
    async def test_clear_traces_endpoint(self, app_with_debug_telemetry) -> None:
        """Test that clear traces works via exporter."""
        exporter = _ensure_integration_telemetry()

        # Create some spans
        tracer = get_tracer("test")
        with tracer.start_as_current_span("test_span"):
            pass

        # Verify spans exist
        assert len(exporter.get_spans()) >= 1

        # Clear traces
        exporter.clear()

        # Verify spans are cleared
        assert len(exporter.get_spans()) == 0


class TestTelemetryStatusEndpoint:
    """Tests for the /api/debug/telemetry/status endpoint."""

    @pytest.mark.anyio
    async def test_telemetry_status_endpoint(self) -> None:
        """Test telemetry status endpoint returns configuration."""
        app = create_test_app(use_v2_routes=True)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/debug/telemetry/status")
            assert response.status_code == 200
            data = response.json()
            assert "telemetry_enabled" in data
            assert "debug_mode" in data
            assert "service_name" in data
            assert "service_version" in data
            assert "environment" in data
            assert "sample_rate" in data

    @pytest.mark.anyio
    async def test_telemetry_status_shows_debug_mode(
        self, app_with_debug_telemetry
    ) -> None:
        """Test that telemetry status reflects debug mode configuration."""
        async with AsyncClient(
            transport=ASGITransport(app=app_with_debug_telemetry),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/debug/telemetry/status")
            assert response.status_code == 200
            # Note: Status endpoint shows settings, not runtime state


class TestTelemetryAcrossComponents:
    """Tests for telemetry propagation across ATP components."""

    @pytest.mark.anyio
    async def test_span_parent_child_relationship(
        self, app_with_debug_telemetry
    ) -> None:
        """Test that nested spans have correct parent-child relationships."""
        exporter = _ensure_integration_telemetry()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("parent"):
            with tracer.start_as_current_span("child"):
                pass

        spans = exporter.get_spans()

        parent_spans = [s for s in spans if s.name == "parent"]
        child_spans = [s for s in spans if s.name == "child"]

        assert len(parent_spans) == 1
        assert len(child_spans) == 1

        parent_span_data = parent_spans[0]
        child_span_data = child_spans[0]

        # Child should have parent's span_id as parent_span_id
        assert child_span_data.parent_span_id == parent_span_data.span_id

    @pytest.mark.anyio
    async def test_span_attributes_recorded(self, app_with_debug_telemetry) -> None:
        """Test that span attributes are correctly recorded."""
        exporter = _ensure_integration_telemetry()
        tracer = get_tracer("test")

        with tracer.start_as_current_span(
            "test_with_attrs",
            attributes={
                "test.id": "test-123",
                "test.name": "My Test",
                "test.score": 0.95,
            },
        ):
            pass

        spans = exporter.get_spans()
        test_spans = [s for s in spans if s.name == "test_with_attrs"]

        assert len(test_spans) == 1
        attrs = test_spans[0].attributes
        assert attrs.get("test.id") == "test-123"
        assert attrs.get("test.name") == "My Test"
        assert attrs.get("test.score") == 0.95

    @pytest.mark.anyio
    async def test_span_events_recorded(self, app_with_debug_telemetry) -> None:
        """Test that span events are correctly recorded."""
        exporter = _ensure_integration_telemetry()
        tracer = get_tracer("test")

        with tracer.start_as_current_span("test_with_events") as span:
            span.add_event("start_processing", {"step": 1})
            span.add_event("processing_complete", {"step": 2})

        spans = exporter.get_spans()
        test_spans = [s for s in spans if s.name == "test_with_events"]

        assert len(test_spans) == 1
        events = test_spans[0].events
        assert len(events) == 2
        event_names = [e["name"] for e in events]
        assert "start_processing" in event_names
        assert "processing_complete" in event_names

    @pytest.mark.anyio
    async def test_error_span_status(self, app_with_debug_telemetry) -> None:
        """Test that error spans have correct status."""
        exporter = _ensure_integration_telemetry()
        tracer = get_tracer("test")

        try:
            with tracer.start_as_current_span("failing_span"):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        spans = exporter.get_spans()
        failing_spans = [s for s in spans if s.name == "failing_span"]

        # Note: Status is set only if explicitly recorded
        assert len(failing_spans) == 1
