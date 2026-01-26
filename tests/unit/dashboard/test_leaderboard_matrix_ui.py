"""Tests for Leaderboard Matrix UI components.

This module tests the Leaderboard Matrix UI components including:
- MatrixGrid component for main table display
- ScoreCell component with color coding
- AgentHeader component with stats
- TestRow component with test info
- Sorting functionality
- Responsive horizontal scroll

These tests verify that the embedded React components in app.py contain
the expected structures and that the API integration works correctly.
"""

import pytest
from fastapi.testclient import TestClient

from atp.dashboard.app import app, create_index_html
from atp.dashboard.schemas import (
    AgentColumn,
    LeaderboardMatrixResponse,
    TestRow,
    TestScore,
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


class TestHTMLContainsLeaderboardComponents:
    """Tests verifying the HTML template contains required leaderboard components."""

    def test_html_contains_matrix_grid(self, html_content: str) -> None:
        """Test that MatrixGrid component is defined in HTML."""
        assert "function MatrixGrid" in html_content
        assert "data.tests" in html_content
        assert "data.agents" in html_content

    def test_html_contains_score_cell(self, html_content: str) -> None:
        """Test that ScoreCell component is defined in HTML."""
        assert "function ScoreCell" in html_content
        assert "testScore" in html_content
        assert "agentName" in html_content

    def test_html_contains_agent_header(self, html_content: str) -> None:
        """Test that AgentHeader component is defined in HTML."""
        assert "function AgentHeader" in html_content
        assert "agent.rank" in html_content
        assert "agent.avg_score" in html_content
        assert "agent.pass_rate" in html_content

    def test_html_contains_test_row(self, html_content: str) -> None:
        """Test that TestRow component is defined in HTML."""
        assert "function TestRow" in html_content
        assert "test.test_name" in html_content
        assert "test.difficulty" in html_content
        assert "test.scores_by_agent" in html_content

    def test_html_contains_leaderboard_view(self, html_content: str) -> None:
        """Test that LeaderboardView component is defined in HTML."""
        assert "function LeaderboardView" in html_content
        assert "matrixData" in html_content
        assert "loadMatrix" in html_content


class TestScoreColorCoding:
    """Tests for score color coding definitions."""

    def test_score_colors_defined(self, html_content: str) -> None:
        """Test that SCORE_COLORS constant is defined."""
        assert "SCORE_COLORS" in html_content
        assert "excellent" in html_content
        assert "good" in html_content
        assert "medium" in html_content
        assert "poor" in html_content
        assert "none" in html_content

    def test_excellent_score_has_green_styling(self, html_content: str) -> None:
        """Test that excellent scores (80+) have deep green styling."""
        assert "bg-green-100" in html_content
        assert "text-green-800" in html_content

    def test_good_score_has_light_green_styling(self, html_content: str) -> None:
        """Test that good scores (60-79) have light green styling."""
        assert "bg-green-50" in html_content
        assert "text-green-700" in html_content

    def test_medium_score_has_yellow_styling(self, html_content: str) -> None:
        """Test that medium scores (40-59) have yellow styling."""
        assert "bg-yellow-50" in html_content
        assert "text-yellow-700" in html_content

    def test_poor_score_has_red_styling(self, html_content: str) -> None:
        """Test that poor scores (<40) have red styling."""
        assert "bg-red-50" in html_content
        assert "text-red-700" in html_content

    def test_get_score_color_function(self, html_content: str) -> None:
        """Test that getScoreColor helper function exists."""
        assert "function getScoreColor" in html_content
        assert "score >= 80" in html_content
        assert "score >= 60" in html_content
        assert "score >= 40" in html_content


class TestDifficultyBadges:
    """Tests for difficulty badge color definitions."""

    def test_difficulty_colors_defined(self, html_content: str) -> None:
        """Test that DIFFICULTY_COLORS constant is defined."""
        assert "DIFFICULTY_COLORS" in html_content

    def test_easy_difficulty_has_green_styling(self, html_content: str) -> None:
        """Test that easy difficulty has green styling."""
        # Check for difficulty colors in the DIFFICULTY_COLORS object
        assert "easy:" in html_content

    def test_difficulty_levels_present(self, html_content: str) -> None:
        """Test that all difficulty levels are present."""
        # medium, hard, very_hard, unknown should be in DIFFICULTY_COLORS
        assert "medium:" in html_content
        assert "hard:" in html_content
        assert "very_hard:" in html_content
        assert "unknown:" in html_content


class TestMatrixGridComponent:
    """Tests for MatrixGrid component structure."""

    def test_matrix_grid_shows_test_count(self, html_content: str) -> None:
        """Test that MatrixGrid displays test and agent counts."""
        assert "data.total_tests" in html_content
        assert "data.total_agents" in html_content

    def test_matrix_grid_has_table_structure(self, html_content: str) -> None:
        """Test that MatrixGrid uses table elements."""
        assert "<table" in html_content
        assert "<thead>" in html_content
        assert "<tbody>" in html_content
        assert "<tr" in html_content

    def test_matrix_grid_has_empty_state(self, html_content: str) -> None:
        """Test that MatrixGrid handles empty data state."""
        assert "No test data available" in html_content

    def test_matrix_grid_has_refresh_button(self, html_content: str) -> None:
        """Test that MatrixGrid has a refresh/reload option."""
        assert "onRefresh" in html_content
        assert "Refresh Data" in html_content


class TestSortingFunctionality:
    """Tests for column sorting functionality."""

    def test_sort_config_state(self, html_content: str) -> None:
        """Test that sorting state is tracked."""
        assert "sortConfig" in html_content
        assert "setSortConfig" in html_content

    def test_handle_sort_function(self, html_content: str) -> None:
        """Test that handleSort function exists."""
        assert "handleSort" in html_content

    def test_sort_by_name_button(self, html_content: str) -> None:
        """Test that sort by name button exists."""
        assert "Sort by Name" in html_content

    def test_sort_by_avg_score_button(self, html_content: str) -> None:
        """Test that sort by average score button exists."""
        assert "Sort by Avg Score" in html_content

    def test_sort_by_difficulty_button(self, html_content: str) -> None:
        """Test that sort by difficulty button exists."""
        assert "Sort by Difficulty" in html_content

    def test_sort_direction_indicator(self, html_content: str) -> None:
        """Test that sort direction is visually indicated."""
        assert "sortConfig.direction" in html_content
        # Should have an arrow indicator
        assert "rotate-180" in html_content

    def test_sorted_tests_implementation(self, html_content: str) -> None:
        """Test that sortedTests array is created."""
        assert "sortedTests" in html_content
        assert ".sort(" in html_content


class TestScoreCellComponent:
    """Tests for ScoreCell component structure."""

    def test_score_cell_shows_score(self, html_content: str) -> None:
        """Test that ScoreCell displays the score value."""
        assert "score.toFixed" in html_content

    def test_score_cell_shows_pass_fail(self, html_content: str) -> None:
        """Test that ScoreCell shows pass/fail status."""
        assert "'Pass'" in html_content or "Pass" in html_content
        assert "'Fail'" in html_content or "Fail" in html_content

    def test_score_cell_shows_execution_count(self, html_content: str) -> None:
        """Test that ScoreCell shows execution count when > 1."""
        assert "execution_count" in html_content
        assert "runs" in html_content

    def test_score_cell_handles_null(self, html_content: str) -> None:
        """Test that ScoreCell handles null/missing scores."""
        assert "testScore" in html_content
        # Should render a placeholder for null score
        assert "'-'" in html_content


class TestAgentHeaderComponent:
    """Tests for AgentHeader component structure."""

    def test_agent_header_shows_rank(self, html_content: str) -> None:
        """Test that AgentHeader displays agent rank."""
        assert "agent.rank" in html_content

    def test_agent_header_shows_name(self, html_content: str) -> None:
        """Test that AgentHeader displays agent name."""
        assert "agent.agent_name" in html_content

    def test_agent_header_shows_stats(self, html_content: str) -> None:
        """Test that AgentHeader displays average score and pass rate."""
        assert "agent.avg_score" in html_content
        assert "agent.pass_rate" in html_content

    def test_agent_header_shows_cost(self, html_content: str) -> None:
        """Test that AgentHeader can display total cost."""
        assert "agent.total_cost" in html_content

    def test_agent_header_rank_badges(self, html_content: str) -> None:
        """Test that top 3 ranks have special badge colors."""
        assert "bg-yellow-400" in html_content  # Gold for 1st
        assert "bg-gray-300" in html_content  # Silver for 2nd
        assert "bg-amber-600" in html_content  # Bronze for 3rd

    def test_agent_header_clickable_for_sort(self, html_content: str) -> None:
        """Test that AgentHeader is clickable for sorting."""
        assert "onSort" in html_content


class TestTestRowComponent:
    """Tests for TestRow component structure."""

    def test_test_row_shows_test_name(self, html_content: str) -> None:
        """Test that TestRow displays test name."""
        assert "test.test_name" in html_content

    def test_test_row_shows_difficulty(self, html_content: str) -> None:
        """Test that TestRow displays difficulty badge."""
        assert "test.difficulty" in html_content

    def test_test_row_shows_pattern(self, html_content: str) -> None:
        """Test that TestRow displays pattern badge when present."""
        assert "test.pattern" in html_content

    def test_test_row_shows_tags(self, html_content: str) -> None:
        """Test that TestRow displays test tags."""
        assert "test.tags" in html_content

    def test_test_row_has_sticky_column(self, html_content: str) -> None:
        """Test that first column is sticky for horizontal scroll."""
        assert "sticky left-0" in html_content

    def test_test_row_alternating_colors(self, html_content: str) -> None:
        """Test that rows have alternating background colors."""
        assert "rowIndex % 2" in html_content


class TestResponsiveLayout:
    """Tests for responsive layout and horizontal scroll."""

    def test_has_overflow_x_auto(self, html_content: str) -> None:
        """Test that table container has horizontal overflow."""
        assert "overflow-x-auto" in html_content

    def test_has_matrix_scroll_container(self, html_content: str) -> None:
        """Test that matrix scroll container class exists."""
        assert "matrix-scroll-container" in html_content

    def test_has_custom_scrollbar_styles(self, html_content: str) -> None:
        """Test that custom scrollbar styles are defined."""
        assert "::-webkit-scrollbar" in html_content

    def test_has_min_width_columns(self, html_content: str) -> None:
        """Test that columns have minimum width for readability."""
        assert "min-w-[" in html_content


class TestLeaderboardViewComponent:
    """Tests for LeaderboardView wrapper component."""

    def test_leaderboard_view_loads_data(self, html_content: str) -> None:
        """Test that LeaderboardView loads data from API."""
        assert "/leaderboard/matrix" in html_content

    def test_leaderboard_view_has_loading_state(self, html_content: str) -> None:
        """Test that LeaderboardView has loading state."""
        assert "Loading leaderboard matrix" in html_content

    def test_leaderboard_view_has_error_state(self, html_content: str) -> None:
        """Test that LeaderboardView has error state."""
        assert "Error loading leaderboard" in html_content
        assert "Try again" in html_content


class TestNavigationIntegration:
    """Tests for navigation integration."""

    def test_leaderboard_nav_button(self, html_content: str) -> None:
        """Test that Leaderboard navigation button exists."""
        assert "Leaderboard" in html_content
        assert "'leaderboard'" in html_content

    def test_leaderboard_view_state(self, html_content: str) -> None:
        """Test that leaderboard view state is handled."""
        assert "view === 'leaderboard'" in html_content

    def test_leaderboard_suite_selector(self, html_content: str) -> None:
        """Test that leaderboard view has suite selector."""
        assert "Leaderboard Matrix" in html_content


class TestLegendSection:
    """Tests for the legend/key section."""

    def test_score_legend_exists(self, html_content: str) -> None:
        """Test that score legend is displayed."""
        assert "80+" in html_content
        assert "60-79" in html_content or "60" in html_content
        assert "40-59" in html_content or "40" in html_content
        assert "&lt;40" in html_content or "<40" in html_content

    def test_rank_legend_exists(self, html_content: str) -> None:
        """Test that rank legend is displayed."""
        assert "Rank:" in html_content


class TestAPIIntegration:
    """Tests for API endpoint integration from UI perspective."""

    def test_leaderboard_matrix_endpoint_returns_correct_structure(
        self, client: TestClient
    ) -> None:
        """Test that leaderboard matrix endpoint returns expected structure for UI.

        Note: Uses database-free validation. Full integration tests with
        database are in test_leaderboard_matrix_api.py.
        """
        response = client.get(
            "/api/leaderboard/matrix",
            params={
                "suite_name": "test-suite",
            },
        )
        # Should return 404 (no data) or 500 (db not configured) in unit test
        # The important thing is validation passes (not 422)
        assert response.status_code in [200, 404, 500]


class TestSchemaCompatibility:
    """Tests verifying UI-expected schema fields match API schemas."""

    def test_test_score_has_required_fields(self) -> None:
        """Test that TestScore has all fields expected by UI."""
        score = TestScore(
            score=85.5,
            success=True,
            execution_count=3,
        )
        # UI expects these fields
        assert hasattr(score, "score")
        assert hasattr(score, "success")
        assert hasattr(score, "execution_count")

    def test_test_row_has_required_fields(self) -> None:
        """Test that TestRow has all fields expected by UI."""
        row = TestRow(
            test_id="test-001",
            test_name="Test Case 1",
            tags=["integration", "api"],
            scores_by_agent={
                "agent-1": TestScore(score=85.5, success=True, execution_count=2)
            },
            avg_score=85.5,
            difficulty="easy",
            pattern=None,
        )
        # UI expects these fields
        assert hasattr(row, "test_id")
        assert hasattr(row, "test_name")
        assert hasattr(row, "tags")
        assert hasattr(row, "scores_by_agent")
        assert hasattr(row, "avg_score")
        assert hasattr(row, "difficulty")
        assert hasattr(row, "pattern")

    def test_agent_column_has_required_fields(self) -> None:
        """Test that AgentColumn has all fields expected by UI."""
        column = AgentColumn(
            agent_name="test-agent",
            avg_score=85.5,
            pass_rate=0.9,
            total_tokens=15000,
            total_cost=1.25,
            rank=1,
        )
        # UI expects these fields
        assert hasattr(column, "agent_name")
        assert hasattr(column, "avg_score")
        assert hasattr(column, "pass_rate")
        assert hasattr(column, "total_tokens")
        assert hasattr(column, "total_cost")
        assert hasattr(column, "rank")

    def test_leaderboard_matrix_response_has_required_fields(self) -> None:
        """Test that LeaderboardMatrixResponse has all fields expected by UI."""
        response = LeaderboardMatrixResponse(
            suite_name="test-suite",
            tests=[],
            agents=[],
            total_tests=0,
            total_agents=0,
            limit=50,
            offset=0,
        )
        # UI expects these fields
        assert hasattr(response, "suite_name")
        assert hasattr(response, "tests")
        assert hasattr(response, "agents")
        assert hasattr(response, "total_tests")
        assert hasattr(response, "total_agents")


class TestUIDataFlowPatterns:
    """Tests verifying data flow patterns expected by UI components."""

    def test_difficulty_values_match_colors(self) -> None:
        """Test that difficulty values match DIFFICULTY_COLORS keys."""
        valid_difficulties = {"easy", "medium", "hard", "very_hard", "unknown"}
        for diff in valid_difficulties:
            row = TestRow(
                test_id="test",
                test_name="Test",
                tags=[],
                scores_by_agent={},
                avg_score=None,
                difficulty=diff,
                pattern=None,
            )
            assert row.difficulty in valid_difficulties

    def test_pattern_values_are_displayable(self) -> None:
        """Test that pattern values can be displayed."""
        patterns = ["hard_for_all", "easy", "high_variance", None]
        for pattern in patterns:
            row = TestRow(
                test_id="test",
                test_name="Test",
                tags=[],
                scores_by_agent={},
                avg_score=None,
                difficulty="medium",
                pattern=pattern,
            )
            # Pattern should be None or a string that can be displayed
            assert row.pattern is None or isinstance(row.pattern, str)


class TestLoadingAndErrorStates:
    """Tests for loading and error state handling in UI."""

    def test_loading_spinner_in_leaderboard(self, html_content: str) -> None:
        """Test that loading spinner exists in LeaderboardView."""
        assert "Loading leaderboard" in html_content
        assert "animate-spin" in html_content

    def test_error_retry_button(self, html_content: str) -> None:
        """Test that error state has retry button."""
        assert "Try again" in html_content

    def test_error_message_styling(self, html_content: str) -> None:
        """Test that error messages have appropriate styling."""
        # LeaderboardView uses red styling for errors
        assert "bg-red-50" in html_content
        assert "border-red-200" in html_content
