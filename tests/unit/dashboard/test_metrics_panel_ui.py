"""Tests for Metrics Panel UI component.

This module tests the MetricsPanel UI component including:
- MetricValue component behavior
- MetricsPanel component rendering
- Best/worst value highlighting
- Percentage difference calculations
- Responsive grid layouts

These tests verify that the embedded React components in app.py contain
the expected structures and that metrics comparison works correctly.
"""

import pytest
from fastapi.testclient import TestClient

from atp.dashboard.app import app, create_index_html
from atp.dashboard.schemas import AgentExecutionDetail
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


class TestHTMLContainsMetricsPanelComponents:
    """Tests verifying the HTML template contains required MetricsPanel components."""

    def test_html_contains_metrics_panel(self, html_content: str) -> None:
        """Test that MetricsPanel component is defined in HTML."""
        assert "function MetricsPanel" in html_content
        assert "agents" in html_content
        assert "showPercentDiff" in html_content

    def test_html_contains_metric_value(self, html_content: str) -> None:
        """Test that MetricValue component is defined in HTML."""
        assert "function MetricValue" in html_content
        assert "isBest" in html_content
        assert "isWorst" in html_content

    def test_html_contains_find_best_metrics(self, html_content: str) -> None:
        """Test that findBestMetrics helper function is defined."""
        assert "function findBestMetrics" in html_content

    def test_html_contains_calculate_percent_diff(self, html_content: str) -> None:
        """Test that calculatePercentDiff helper function is defined."""
        assert "function calculatePercentDiff" in html_content


class TestMetricsPanelDisplaysAllMetrics:
    """Tests verifying MetricsPanel displays all required metrics."""

    def test_displays_score_metric(self, html_content: str) -> None:
        """Test that MetricsPanel displays score metric."""
        # Check for score label and formatting
        assert 'label="Score"' in html_content
        assert "/100" in html_content

    def test_displays_tokens_metric(self, html_content: str) -> None:
        """Test that MetricsPanel displays tokens metric."""
        assert 'label="Tokens"' in html_content
        assert "total_tokens" in html_content

    def test_displays_steps_metric(self, html_content: str) -> None:
        """Test that MetricsPanel displays steps metric."""
        assert 'label="Steps"' in html_content
        assert "total_steps" in html_content

    def test_displays_duration_metric(self, html_content: str) -> None:
        """Test that MetricsPanel displays duration metric."""
        assert 'label="Duration"' in html_content
        assert "duration_seconds" in html_content

    def test_displays_cost_metric(self, html_content: str) -> None:
        """Test that MetricsPanel displays cost metric."""
        assert 'label="Cost"' in html_content
        assert "cost_usd" in html_content

    def test_displays_tool_calls_metric(self, html_content: str) -> None:
        """Test that MetricsPanel displays tool calls metric."""
        assert 'label="Tool Calls"' in html_content
        assert "tool_calls" in html_content


class TestBestValueHighlighting:
    """Tests for best/worst value highlighting in MetricsPanel."""

    def test_best_value_has_green_styling(self, html_content: str) -> None:
        """Test that best values are highlighted with green styling."""
        assert "bg-green-50" in html_content
        assert "text-green-700" in html_content

    def test_worst_value_has_red_styling(self, html_content: str) -> None:
        """Test that worst values are highlighted with red styling."""
        assert "bg-red-50" in html_content
        assert "text-red-700" in html_content

    def test_best_value_shows_star_indicator(self, html_content: str) -> None:
        """Test that best values show a star indicator."""
        # The star indicator for best values
        assert "★" in html_content

    def test_find_best_metrics_handles_score_highest(self, html_content: str) -> None:
        """Test that findBestMetrics identifies highest score as best."""
        # Score should be maximized (higher is better)
        assert "score: { best: -Infinity, isBest: (a, b) => a > b }" in html_content

    def test_find_best_metrics_handles_tokens_lowest(self, html_content: str) -> None:
        """Test that findBestMetrics identifies lowest tokens as best."""
        # Tokens should be minimized (lower is better)
        expected = "total_tokens: { best: Infinity, isBest: (a, b) => a < b }"
        assert expected in html_content

    def test_find_best_metrics_handles_steps_lowest(self, html_content: str) -> None:
        """Test that findBestMetrics identifies lowest steps as best."""
        expected = "total_steps: { best: Infinity, isBest: (a, b) => a < b }"
        assert expected in html_content

    def test_find_best_metrics_handles_duration_lowest(self, html_content: str) -> None:
        """Test that findBestMetrics identifies lowest duration as best."""
        expected = "duration_seconds: { best: Infinity, isBest: (a, b) => a < b }"
        assert expected in html_content

    def test_find_best_metrics_handles_cost_lowest(self, html_content: str) -> None:
        """Test that findBestMetrics identifies lowest cost as best."""
        expected = "cost_usd: { best: Infinity, isBest: (a, b) => a < b }"
        assert expected in html_content


class TestPercentageDifferenceCalculation:
    """Tests for percentage difference calculation and display."""

    def test_percent_diff_is_calculated(self, html_content: str) -> None:
        """Test that percentage difference is calculated."""
        assert "calculatePercentDiff" in html_content

    def test_percent_diff_handles_null_values(self, html_content: str) -> None:
        """Test that percent diff handles null/undefined values."""
        assert "baseline === null" in html_content
        assert "value === null" in html_content

    def test_percent_diff_handles_zero_baseline(self, html_content: str) -> None:
        """Test that percent diff handles zero baseline."""
        assert "baseline === 0" in html_content

    def test_percent_diff_positive_formatting(self, html_content: str) -> None:
        """Test that positive percent diff shows plus sign."""
        assert "percentDiff > 0 ? '+' : ''" in html_content

    def test_percent_diff_color_varies_by_metric(self, html_content: str) -> None:
        """Test that percent diff color varies based on metric type."""
        # For score, positive is good (green), for others positive is bad (red)
        assert "label === 'Score'" in html_content
        assert "text-green-600" in html_content
        assert "text-red-600" in html_content

    def test_baseline_agent_indicator(self, html_content: str) -> None:
        """Test that baseline agent is indicated in UI."""
        assert "Baseline" in html_content
        assert "baselineIndex" in html_content


class TestResponsiveGridLayout:
    """Tests for responsive grid layout in MetricsPanel."""

    def test_single_agent_single_column(self, html_content: str) -> None:
        """Test that single agent uses single column layout."""
        assert "agents.length === 1" in html_content
        assert "grid-cols-1" in html_content

    def test_two_agents_two_columns(self, html_content: str) -> None:
        """Test that two agents use two column layout on medium screens."""
        assert "agents.length === 2" in html_content
        assert "md:grid-cols-2" in html_content

    def test_three_agents_three_columns(self, html_content: str) -> None:
        """Test that three agents use three column layout on large screens."""
        assert "lg:grid-cols-3" in html_content

    def test_mobile_single_column_always(self, html_content: str) -> None:
        """Test that mobile always uses single column."""
        # First column is always grid-cols-1 before responsive breakpoints
        assert "grid-cols-1" in html_content


class TestMetricValueComponent:
    """Tests for MetricValue sub-component structure."""

    def test_metric_value_has_label(self, html_content: str) -> None:
        """Test that MetricValue displays label."""
        assert "label" in html_content
        assert "{label}" in html_content

    def test_metric_value_has_value_formatting(self, html_content: str) -> None:
        """Test that MetricValue formats value based on type."""
        assert "formatValue" in html_content
        assert "format === 'number'" in html_content
        assert "format === 'decimal'" in html_content
        assert "format === 'score'" in html_content
        assert "format === 'cost'" in html_content

    def test_metric_value_shows_unit(self, html_content: str) -> None:
        """Test that MetricValue shows unit if provided."""
        assert "{unit" in html_content

    def test_metric_value_has_conditional_styling(self, html_content: str) -> None:
        """Test that MetricValue has conditional bg/text colors."""
        assert "getBgColor" in html_content
        assert "getTextColor" in html_content


class TestMetricsPanelHeader:
    """Tests for MetricsPanel header section."""

    def test_header_shows_agent_count(self, html_content: str) -> None:
        """Test that header shows agent count."""
        assert "agents.length" in html_content
        assert "agent{agents.length > 1 ? 's' : ''} compared" in html_content

    def test_header_shows_baseline_info(self, html_content: str) -> None:
        """Test that header shows baseline information when showing diff."""
        assert "% diff vs" in html_content

    def test_panel_title(self, html_content: str) -> None:
        """Test that panel has title."""
        assert "Metrics Comparison" in html_content


class TestMetricsPanelLegend:
    """Tests for MetricsPanel legend section."""

    def test_legend_explains_best_value(self, html_content: str) -> None:
        """Test that legend explains best value styling."""
        assert "Best value" in html_content

    def test_legend_explains_worst_value(self, html_content: str) -> None:
        """Test that legend explains worst value styling."""
        assert "Worst value" in html_content

    def test_legend_explains_baseline(self, html_content: str) -> None:
        """Test that legend explains baseline percentage."""
        assert "relative to baseline" in html_content


class TestAgentStatusBadge:
    """Tests for agent pass/fail status badge."""

    def test_shows_passed_badge(self, html_content: str) -> None:
        """Test that passed agents show 'Passed' badge."""
        assert "Passed" in html_content
        assert "bg-green-100 text-green-800" in html_content

    def test_shows_failed_badge(self, html_content: str) -> None:
        """Test that failed agents show 'Failed' badge."""
        assert "Failed" in html_content
        assert "bg-red-100 text-red-800" in html_content


class TestMetricsPanelEmptyState:
    """Tests for MetricsPanel empty state handling."""

    def test_empty_state_message(self, html_content: str) -> None:
        """Test that empty state shows appropriate message."""
        assert "No agent data available" in html_content

    def test_handles_null_agents(self, html_content: str) -> None:
        """Test that component handles null agents prop."""
        assert "!agents || agents.length === 0" in html_content


class TestMetricsPanelIntegration:
    """Tests for MetricsPanel integration in ComparisonContainer."""

    def test_metrics_panel_in_comparison_container(self, html_content: str) -> None:
        """Test that MetricsPanel is used in ComparisonContainer."""
        assert "<MetricsPanel" in html_content
        assert "agents={comparison.agents}" in html_content

    def test_metrics_panel_shown_conditionally(self, html_content: str) -> None:
        """Test that MetricsPanel is shown only when agents exist."""
        assert "comparison.agents.length > 0" in html_content


class TestSchemaCompatibility:
    """Tests verifying UI-expected schema fields match API schemas."""

    def test_agent_execution_detail_has_required_fields(self) -> None:
        """Test that AgentExecutionDetail has all fields expected by MetricsPanel."""
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
        assert hasattr(detail, "cost_usd")

    def test_agent_detail_score_can_be_none(self) -> None:
        """Test that AgentExecutionDetail score can be None."""
        detail = AgentExecutionDetail(
            agent_name="test-agent",
            test_execution_id=1,
            score=None,
            success=False,
            duration_seconds=None,
            total_tokens=None,
            total_steps=None,
            tool_calls=None,
            llm_calls=None,
            cost_usd=None,
            events=[],
        )
        assert detail.score is None
        assert detail.total_tokens is None


class TestMetricFormats:
    """Tests for metric value formatting."""

    def test_number_format_uses_locale_string(self, html_content: str) -> None:
        """Test that number format uses toLocaleString."""
        assert "toLocaleString()" in html_content

    def test_decimal_format_uses_fixed_2(self, html_content: str) -> None:
        """Test that decimal format uses toFixed(2)."""
        assert "toFixed(2)" in html_content

    def test_score_format_uses_fixed_1(self, html_content: str) -> None:
        """Test that score format uses toFixed(1)."""
        assert "toFixed(1)" in html_content

    def test_cost_format_uses_fixed_4(self, html_content: str) -> None:
        """Test that cost format uses toFixed(4)."""
        assert "toFixed(4)" in html_content

    def test_null_value_shows_dash(self, html_content: str) -> None:
        """Test that null values display as dash."""
        assert "return '-'" in html_content
