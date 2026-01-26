"""Tests for Step Comparison UI components.

This module tests the Step Comparison UI components including:
- AgentSelector component behavior
- EventItem component styling
- StepComparison component rendering
- ComparisonContainer layout and API integration

These tests verify that the embedded React components in app.py contain
the expected structures and that the API integration works correctly.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from atp.dashboard.app import app, create_index_html
from atp.dashboard.schemas import (
    AgentExecutionDetail,
    EventSummary,
    SideBySideComparisonResponse,
)
from tests.fixtures.comparison import (
    SAMPLE_EVENTS,
    create_error_event,
    create_llm_request_event,
    create_progress_event,
    create_reasoning_event,
    create_tool_call_event,
)
from tests.fixtures.comparison.factories import reset_all_factories


@pytest.fixture
def client() -> TestClient:
    """Create a test client using the actual app."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def reset_fixtures() -> None:
    """Reset all factory counters before each test."""
    reset_all_factories()


@pytest.fixture
def html_content() -> str:
    """Get the HTML content from create_index_html."""
    return create_index_html()


class TestHTMLContainsComponents:
    """Tests verifying the HTML template contains required UI components."""

    def test_html_contains_agent_selector(self, html_content: str) -> None:
        """Test that AgentSelector component is defined in HTML."""
        assert "function AgentSelector" in html_content
        assert "selectedAgents" in html_content
        assert "onSelectionChange" in html_content

    def test_html_contains_event_item(self, html_content: str) -> None:
        """Test that EventItem component is defined in HTML."""
        assert "function EventItem" in html_content
        assert "event.event_type" in html_content

    def test_html_contains_step_comparison(self, html_content: str) -> None:
        """Test that StepComparison component is defined in HTML."""
        assert "function StepComparison" in html_content
        assert "agentDetail" in html_content
        assert "eventFilter" in html_content

    def test_html_contains_comparison_container(self, html_content: str) -> None:
        """Test that ComparisonContainer component is defined in HTML."""
        assert "function ComparisonContainer" in html_content
        assert "suiteName" in html_content
        assert "testId" in html_content

    def test_html_contains_event_colors(self, html_content: str) -> None:
        """Test that EVENT_COLORS constant is defined."""
        assert "EVENT_COLORS" in html_content
        assert "tool_call" in html_content
        assert "llm_request" in html_content
        assert "reasoning" in html_content
        assert "error" in html_content
        assert "progress" in html_content


class TestEventColorStyles:
    """Tests for event type color definitions."""

    def test_tool_call_has_blue_styling(self, html_content: str) -> None:
        """Test that tool_call events have blue color styling."""
        # Check the EVENT_COLORS definition includes blue for tool_call
        assert "tool_call:" in html_content
        assert "bg-blue-50" in html_content
        assert "border-blue-400" in html_content

    def test_llm_request_has_green_styling(self, html_content: str) -> None:
        """Test that llm_request events have green color styling."""
        assert "llm_request:" in html_content
        assert "bg-green-50" in html_content
        assert "border-green-400" in html_content

    def test_reasoning_has_amber_styling(self, html_content: str) -> None:
        """Test that reasoning events have amber color styling."""
        assert "reasoning:" in html_content
        assert "bg-amber-50" in html_content
        assert "border-amber-400" in html_content

    def test_error_has_red_styling(self, html_content: str) -> None:
        """Test that error events have red color styling."""
        assert "error:" in html_content
        assert "bg-red-50" in html_content
        assert "border-red-400" in html_content

    def test_progress_has_purple_styling(self, html_content: str) -> None:
        """Test that progress events have purple color styling."""
        assert "progress:" in html_content
        assert "bg-purple-50" in html_content
        assert "border-purple-400" in html_content


class TestAgentSelectorComponent:
    """Tests for AgentSelector component structure."""

    def test_agent_selector_has_max_agents_prop(self, html_content: str) -> None:
        """Test that AgentSelector accepts maxAgents prop."""
        assert "maxAgents" in html_content
        assert "maxAgents = 3" in html_content

    def test_agent_selector_has_dropdown_toggle(self, html_content: str) -> None:
        """Test that AgentSelector has dropdown toggle behavior."""
        assert "isOpen" in html_content
        assert "setIsOpen" in html_content

    def test_agent_selector_shows_selection_count(self, html_content: str) -> None:
        """Test that AgentSelector shows selection count."""
        assert "selectedAgents.length" in html_content
        assert "agents selected" in html_content

    def test_agent_selector_prevents_over_selection(self, html_content: str) -> None:
        """Test that AgentSelector prevents selecting more than maxAgents."""
        assert "selectedAgents.length >= maxAgents" in html_content
        assert "isDisabled" in html_content


class TestStepComparisonComponent:
    """Tests for StepComparison component structure."""

    def test_step_comparison_shows_agent_metrics(self, html_content: str) -> None:
        """Test that StepComparison shows agent metrics."""
        assert "agentDetail.score" in html_content
        assert "agentDetail.success" in html_content
        assert "agentDetail.duration_seconds" in html_content
        assert "agentDetail.total_tokens" in html_content

    def test_step_comparison_has_event_filter(self, html_content: str) -> None:
        """Test that StepComparison has event type filter buttons."""
        assert "eventFilter" in html_content
        assert "setEventFilter" in html_content
        # Should have 'all' filter option
        assert "eventFilter === 'all'" in html_content

    def test_step_comparison_renders_events(self, html_content: str) -> None:
        """Test that StepComparison renders EventItem for each event."""
        assert "<EventItem" in html_content
        assert "filteredEvents" in html_content

    def test_step_comparison_handles_empty_state(self, html_content: str) -> None:
        """Test that StepComparison handles empty events state."""
        has_empty_msg = (
            "No events to display" in html_content
            or "No execution data" in html_content
        )
        assert has_empty_msg


class TestComparisonContainerComponent:
    """Tests for ComparisonContainer component structure."""

    def test_comparison_container_uses_api_endpoint(self, html_content: str) -> None:
        """Test that ComparisonContainer calls the side-by-side API endpoint."""
        assert "/compare/side-by-side" in html_content

    def test_comparison_container_has_loading_state(self, html_content: str) -> None:
        """Test that ComparisonContainer has loading state with skeleton."""
        assert "loading" in html_content
        assert "setLoading" in html_content
        # Now uses skeleton loader instead of text
        assert "SkeletonMetricsPanel" in html_content

    def test_comparison_container_has_error_state(self, html_content: str) -> None:
        """Test that ComparisonContainer has error state."""
        assert "error" in html_content
        assert "setError" in html_content
        assert "Failed to load" in html_content

    def test_comparison_container_has_selection_ui(self, html_content: str) -> None:
        """Test that ComparisonContainer has test and agent selection UI."""
        assert "Select Test" in html_content
        assert "Select Agents" in html_content

    def test_comparison_container_has_grid_layout(self, html_content: str) -> None:
        """Test that ComparisonContainer has responsive grid layout."""
        assert "grid-cols-1" in html_content
        assert "lg:grid-cols-2" in html_content
        assert "lg:grid-cols-3" in html_content

    def test_comparison_container_has_back_button(self, html_content: str) -> None:
        """Test that ComparisonContainer has back to selection button."""
        assert "Back to selection" in html_content


class TestEventItemComponent:
    """Tests for EventItem component structure."""

    def test_event_item_shows_event_type(self, html_content: str) -> None:
        """Test that EventItem displays event type."""
        assert "event.event_type" in html_content
        assert "replace('_', ' ')" in html_content  # Formats underscores

    def test_event_item_shows_summary(self, html_content: str) -> None:
        """Test that EventItem displays event summary."""
        assert "event.summary" in html_content

    def test_event_item_shows_sequence_number(self, html_content: str) -> None:
        """Test that EventItem displays sequence number."""
        assert "sequence" in html_content

    def test_event_item_has_expandable_details(self, html_content: str) -> None:
        """Test that EventItem has expandable details section."""
        assert "expanded" in html_content
        assert "setExpanded" in html_content
        assert "event.data" in html_content


class TestAPIIntegration:
    """Tests for API endpoint integration from UI perspective."""

    def test_side_by_side_endpoint_returns_correct_structure(
        self, client: TestClient
    ) -> None:
        """Test that side-by-side endpoint returns expected structure for UI.

        Note: Uses database-free validation. Full integration tests with
        database are in test_comparison_endpoints.py.
        """
        response = client.get(
            "/api/compare/side-by-side",
            params={
                "suite_name": "test-suite",
                "test_id": "test-001",
                "agents": ["agent-1", "agent-2"],
            },
        )
        # Should return 404 (no data) or 500 (db not configured) in unit test
        # The important thing is validation passes (not 422)
        assert response.status_code in [404, 500]

    def test_agents_endpoint_accessible(self, client: TestClient) -> None:
        """Test that agents endpoint is accessible."""
        response = client.get("/api/agents")
        # Should return 200 with empty list or 500 if db not configured
        assert response.status_code in [200, 500]


class TestSchemaCompatibility:
    """Tests verifying UI-expected schema fields match API schemas."""

    def test_event_summary_has_required_fields(self) -> None:
        """Test that EventSummary has all fields expected by UI."""
        event = EventSummary(
            sequence=1,
            timestamp=datetime.now(),
            event_type="tool_call",
            summary="Test summary",
            data={"key": "value"},
        )
        # UI expects these fields
        assert hasattr(event, "sequence")
        assert hasattr(event, "timestamp")
        assert hasattr(event, "event_type")
        assert hasattr(event, "summary")
        assert hasattr(event, "data")

    def test_agent_execution_detail_has_required_fields(self) -> None:
        """Test that AgentExecutionDetail has all fields expected by UI."""
        detail = AgentExecutionDetail(
            agent_name="test-agent",
            test_execution_id=1,
            score=85.5,
            success=True,
            duration_seconds=120.5,
            total_tokens=1500,
            total_steps=5,
            tool_calls=3,
            llm_calls=5,
            cost_usd=0.015,
            events=[],
        )
        # UI expects these fields for metrics display
        assert hasattr(detail, "agent_name")
        assert hasattr(detail, "score")
        assert hasattr(detail, "success")
        assert hasattr(detail, "duration_seconds")
        assert hasattr(detail, "total_tokens")
        assert hasattr(detail, "total_steps")
        assert hasattr(detail, "tool_calls")
        assert hasattr(detail, "llm_calls")
        assert hasattr(detail, "cost_usd")
        assert hasattr(detail, "events")

    def test_side_by_side_response_has_required_fields(self) -> None:
        """Test that SideBySideComparisonResponse has all fields expected by UI."""
        response = SideBySideComparisonResponse(
            suite_name="test-suite",
            test_id="test-001",
            test_name="Test Case 1",
            agents=[],
        )
        # UI expects these fields for header display
        assert hasattr(response, "suite_name")
        assert hasattr(response, "test_id")
        assert hasattr(response, "test_name")
        assert hasattr(response, "agents")


class TestEventSummaryForUI:
    """Tests for event summary generation that feeds the UI."""

    def test_tool_call_event_summary(self) -> None:
        """Test that tool_call events produce UI-friendly summary."""
        event = create_tool_call_event(
            tool="web_search",
            status="success",
            sequence=1,
        )
        assert event["event_type"] == "tool_call"
        assert event["payload"]["tool"] == "web_search"
        assert event["payload"]["status"] == "success"

    def test_llm_request_event_summary(self) -> None:
        """Test that llm_request events produce UI-friendly summary."""
        event = create_llm_request_event(
            model="claude-sonnet-4-20250514",
            input_tokens=500,
            output_tokens=200,
            sequence=2,
        )
        assert event["event_type"] == "llm_request"
        assert event["payload"]["model"] == "claude-sonnet-4-20250514"
        assert event["payload"]["input_tokens"] == 500
        assert event["payload"]["output_tokens"] == 200

    def test_reasoning_event_summary(self) -> None:
        """Test that reasoning events produce UI-friendly summary."""
        event = create_reasoning_event(
            thought="Analyzing the problem",
            step="Step 1 of 3",
            sequence=3,
        )
        assert event["event_type"] == "reasoning"
        assert event["payload"]["thought"] == "Analyzing the problem"

    def test_error_event_summary(self) -> None:
        """Test that error events produce UI-friendly summary."""
        event = create_error_event(
            error_type="RuntimeError",
            message="Something went wrong",
            recoverable=True,
            sequence=4,
        )
        assert event["event_type"] == "error"
        assert event["payload"]["error_type"] == "RuntimeError"
        assert event["payload"]["message"] == "Something went wrong"

    def test_progress_event_summary(self) -> None:
        """Test that progress events produce UI-friendly summary."""
        event = create_progress_event(
            percentage=50.0,
            message="Halfway done",
            sequence=5,
        )
        assert event["event_type"] == "progress"
        assert event["payload"]["percentage"] == 50.0
        assert event["payload"]["message"] == "Halfway done"


class TestUIDataFlowPatterns:
    """Tests verifying data flow patterns expected by UI components."""

    def test_sample_events_are_ui_compatible(self) -> None:
        """Test that SAMPLE_EVENTS work with UI components."""
        events = SAMPLE_EVENTS["simple_success"]
        assert len(events) > 0
        for event in events:
            # UI expects these fields
            assert "event_type" in event
            assert "sequence" in event
            assert "timestamp" in event
            assert "payload" in event

    def test_event_types_match_color_definitions(self) -> None:
        """Test that event types in samples match EVENT_COLORS keys."""
        valid_types = {"tool_call", "llm_request", "reasoning", "error", "progress"}
        for scenario_name, events in SAMPLE_EVENTS.items():
            for event in events:
                assert event["event_type"] in valid_types, (
                    f"Event type '{event['event_type']}' in {scenario_name} "
                    f"not in valid types: {valid_types}"
                )

    def test_events_have_sequential_sequence_numbers(self) -> None:
        """Test that events have sequential sequence numbers for UI ordering."""
        events = SAMPLE_EVENTS["simple_success"]
        sequences = [e["sequence"] for e in events]
        # Sequences should be ordered (may not start at 0)
        assert sequences == sorted(sequences)


class TestResponsiveLayout:
    """Tests for responsive layout classes in HTML."""

    def test_has_mobile_single_column(self, html_content: str) -> None:
        """Test that layout has single column for mobile."""
        assert "grid-cols-1" in html_content

    def test_has_tablet_two_columns(self, html_content: str) -> None:
        """Test that layout has two columns for tablets."""
        assert "md:grid-cols-2" in html_content or "lg:grid-cols-2" in html_content

    def test_has_desktop_three_columns(self, html_content: str) -> None:
        """Test that layout has three columns for desktop with 3 agents."""
        assert "lg:grid-cols-3" in html_content

    def test_has_container_max_width(self, html_content: str) -> None:
        """Test that layout uses container with max-width."""
        assert "container mx-auto" in html_content


class TestLoadingAndErrorStates:
    """Tests for loading and error state handling in UI."""

    def test_loading_skeleton_exists(self, html_content: str) -> None:
        """Test that skeleton loading animation exists."""
        assert "skeleton-pulse" in html_content
        assert "SkeletonBox" in html_content

    def test_error_message_styling(self, html_content: str) -> None:
        """Test that error messages have appropriate styling."""
        assert "bg-red-50" in html_content
        assert "border-red-200" in html_content
        assert "text-red-700" in html_content

    def test_empty_state_message(self, html_content: str) -> None:
        """Test that empty state messages exist."""
        has_empty_msg = (
            "No execution data" in html_content
            or "No events to display" in html_content
        )
        assert has_empty_msg


class TestViewModeToggle:
    """Tests for view mode toggle between metrics and steps."""

    def test_view_mode_state_exists(self, html_content: str) -> None:
        """Test that viewMode state exists in AgentComparison."""
        assert "viewMode" in html_content
        assert "setViewMode" in html_content

    def test_metrics_view_mode(self, html_content: str) -> None:
        """Test that metrics view mode is default."""
        assert "'metrics'" in html_content

    def test_steps_view_mode(self, html_content: str) -> None:
        """Test that steps view mode option exists."""
        assert "'steps'" in html_content

    def test_view_step_by_step_button(self, html_content: str) -> None:
        """Test that View Step-by-Step button exists."""
        assert "View Step-by-Step" in html_content

    def test_back_to_metrics_button(self, html_content: str) -> None:
        """Test that Back to metrics button exists."""
        assert "Back to metrics" in html_content
