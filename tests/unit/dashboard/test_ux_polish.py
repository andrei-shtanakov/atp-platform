"""Tests for UX Polish components in ATP Dashboard.

Tests for TASK-012:
- Skeleton loaders for all views
- Error boundaries with retry buttons
- Responsive design (1280px, 1440px, 1920px)
- Keyboard navigation for timeline
- User-friendly API error messages
"""

from atp.dashboard.app import create_index_html


class TestSkeletonLoaders:
    """Tests for skeleton loader components."""

    def test_html_has_skeleton_box_component(self) -> None:
        """Test that HTML includes SkeletonBox component."""
        html = create_index_html()
        assert "function SkeletonBox" in html
        assert "skeleton-pulse" in html

    def test_html_has_skeleton_text_component(self) -> None:
        """Test that HTML includes SkeletonText component."""
        html = create_index_html()
        assert "function SkeletonText" in html

    def test_html_has_skeleton_dashboard_summary(self) -> None:
        """Test that HTML includes SkeletonDashboardSummary component."""
        html = create_index_html()
        assert "function SkeletonDashboardSummary" in html

    def test_html_has_skeleton_suite_list(self) -> None:
        """Test that HTML includes SkeletonSuiteList component."""
        html = create_index_html()
        assert "function SkeletonSuiteList" in html
        assert "SkeletonTableRow" in html

    def test_html_has_skeleton_metrics_panel(self) -> None:
        """Test that HTML includes SkeletonMetricsPanel component."""
        html = create_index_html()
        assert "function SkeletonMetricsPanel" in html

    def test_html_has_skeleton_leaderboard_matrix(self) -> None:
        """Test that HTML includes SkeletonLeaderboardMatrix component."""
        html = create_index_html()
        assert "function SkeletonLeaderboardMatrix" in html

    def test_html_has_skeleton_timeline(self) -> None:
        """Test that HTML includes SkeletonTimeline component."""
        html = create_index_html()
        assert "function SkeletonTimeline" in html

    def test_html_has_skeleton_chart(self) -> None:
        """Test that HTML includes SkeletonChart component."""
        html = create_index_html()
        assert "function SkeletonChart" in html

    def test_skeleton_pulse_animation_defined(self) -> None:
        """Test that skeleton pulse animation CSS is defined."""
        html = create_index_html()
        assert "@keyframes skeleton-pulse" in html
        assert "animation: skeleton-pulse" in html


class TestErrorBoundary:
    """Tests for error boundary component."""

    def test_html_has_error_boundary_class(self) -> None:
        """Test that HTML includes ErrorBoundary class component."""
        html = create_index_html()
        assert "class ErrorBoundary extends React.Component" in html

    def test_error_boundary_has_get_derived_state(self) -> None:
        """Test that ErrorBoundary has getDerivedStateFromError."""
        html = create_index_html()
        assert "getDerivedStateFromError" in html

    def test_error_boundary_has_component_did_catch(self) -> None:
        """Test that ErrorBoundary has componentDidCatch."""
        html = create_index_html()
        assert "componentDidCatch" in html

    def test_error_boundary_has_retry_handler(self) -> None:
        """Test that ErrorBoundary has retry handler."""
        html = create_index_html()
        assert "handleRetry" in html

    def test_error_boundary_shows_retry_button(self) -> None:
        """Test that ErrorBoundary renders retry button."""
        html = create_index_html()
        assert "Try Again" in html

    def test_html_has_error_display_component(self) -> None:
        """Test that HTML includes ErrorDisplay functional component."""
        html = create_index_html()
        assert "function ErrorDisplay" in html

    def test_error_display_accepts_retry_callback(self) -> None:
        """Test that ErrorDisplay accepts onRetry callback."""
        html = create_index_html()
        assert "ErrorDisplay" in html
        assert "onRetry" in html

    def test_views_wrapped_in_error_boundary(self) -> None:
        """Test that main views are wrapped in ErrorBoundary."""
        html = create_index_html()
        # Dashboard view
        assert '<ErrorBoundary title="Dashboard Error"' in html
        # Suites view
        assert '<ErrorBoundary title="Suites Error"' in html
        # Compare view
        assert '<ErrorBoundary title="Comparison Error"' in html
        # Leaderboard view
        assert '<ErrorBoundary title="Leaderboard Error"' in html
        # Timeline view
        assert '<ErrorBoundary title="Timeline Error"' in html


class TestKeyboardNavigation:
    """Tests for keyboard navigation in timeline."""

    def test_timeline_row_has_keyboard_handler(self) -> None:
        """Test that TimelineRow has keyboard event handler."""
        html = create_index_html()
        assert "handleKeyDown" in html
        assert "onKeyDown" in html

    def test_keyboard_navigation_supports_arrow_keys(self) -> None:
        """Test that keyboard navigation supports arrow keys."""
        html = create_index_html()
        assert "ArrowLeft" in html
        assert "ArrowRight" in html

    def test_keyboard_navigation_supports_enter(self) -> None:
        """Test that keyboard navigation supports Enter key."""
        html = create_index_html()
        # Enter key for activating event
        assert "'Enter'" in html or '"Enter"' in html

    def test_keyboard_navigation_supports_home_end(self) -> None:
        """Test that keyboard navigation supports Home/End keys."""
        html = create_index_html()
        assert "'Home'" in html or '"Home"' in html
        assert "'End'" in html or '"End"' in html

    def test_timeline_events_are_focusable(self) -> None:
        """Test that timeline events are keyboard-focusable."""
        html = create_index_html()
        assert "tabIndex" in html
        assert "timeline-event-focusable" in html

    def test_timeline_has_aria_labels(self) -> None:
        """Test that timeline has accessibility aria labels."""
        html = create_index_html()
        assert "aria-label" in html
        assert 'role="listbox"' in html or 'role="option"' in html

    def test_focus_styles_defined(self) -> None:
        """Test that focus styles are defined in CSS."""
        html = create_index_html()
        assert ".timeline-event-focusable:focus" in html


class TestUserFriendlyErrors:
    """Tests for user-friendly API error messages."""

    def test_api_helper_has_error_messages(self) -> None:
        """Test that API helper has user-friendly error messages."""
        html = create_index_html()
        assert "const errorMessages" in html

    def test_error_message_for_400(self) -> None:
        """Test user-friendly message for 400 Bad Request."""
        html = create_index_html()
        assert "Invalid request" in html

    def test_error_message_for_401(self) -> None:
        """Test user-friendly message for 401 Unauthorized."""
        html = create_index_html()
        assert "not authorized" in html or "log in" in html

    def test_error_message_for_403(self) -> None:
        """Test user-friendly message for 403 Forbidden."""
        html = create_index_html()
        assert "Access denied" in html or "permission" in html

    def test_error_message_for_404(self) -> None:
        """Test user-friendly message for 404 Not Found."""
        html = create_index_html()
        assert "not found" in html

    def test_error_message_for_500(self) -> None:
        """Test user-friendly message for 500 Server Error."""
        html = create_index_html()
        assert "Server error" in html

    def test_error_message_for_network_failure(self) -> None:
        """Test user-friendly message for network failures."""
        html = create_index_html()
        assert "internet connection" in html or "Unable to connect" in html


class TestResponsiveDesign:
    """Tests for responsive design patterns."""

    def test_html_has_responsive_grid_classes(self) -> None:
        """Test that HTML uses responsive grid classes."""
        html = create_index_html()
        # Tailwind responsive breakpoints
        assert "md:" in html
        assert "lg:" in html
        assert "grid-cols-1" in html

    def test_html_has_responsive_metrics_panel_grid(self) -> None:
        """Test that MetricsPanel has responsive grid layout."""
        html = create_index_html()
        # Check for responsive grid in MetricsPanel
        assert "md:grid-cols-2" in html
        assert "lg:grid-cols-3" in html

    def test_html_has_responsive_comparison_grid(self) -> None:
        """Test that comparison views have responsive grid."""
        html = create_index_html()
        assert "lg:grid-cols-2" in html

    def test_html_has_container_class(self) -> None:
        """Test that HTML uses container class for centering."""
        html = create_index_html()
        assert "container mx-auto" in html

    def test_html_has_min_width_on_table_columns(self) -> None:
        """Test that tables have min-width for readability."""
        html = create_index_html()
        assert "min-w-" in html

    def test_html_has_overflow_handling(self) -> None:
        """Test that HTML has overflow handling for scroll."""
        html = create_index_html()
        assert "overflow-x-auto" in html
        assert "overflow-y-auto" in html


class TestLoadingStatesUseSkeleton:
    """Tests that loading states use skeleton loaders."""

    def test_leaderboard_loading_uses_skeleton(self) -> None:
        """Test that LeaderboardView loading state uses skeleton."""
        html = create_index_html()
        # Check that SkeletonLeaderboardMatrix is used in LeaderboardView
        assert "SkeletonLeaderboardMatrix" in html

    def test_timeline_loading_uses_skeleton(self) -> None:
        """Test that Timeline loading state uses skeleton."""
        html = create_index_html()
        assert "SkeletonTimeline" in html

    def test_comparison_loading_uses_skeleton(self) -> None:
        """Test that Comparison loading state uses skeleton."""
        html = create_index_html()
        assert "SkeletonMetricsPanel" in html

    def test_main_app_loading_uses_skeleton(self) -> None:
        """Test that main App loading state uses skeleton components."""
        html = create_index_html()
        # Main App should use SkeletonDashboardSummary
        assert "SkeletonDashboardSummary" in html
        assert "SkeletonSuiteList" in html
        assert "SkeletonChart" in html


class TestRetryFunctionality:
    """Tests for retry functionality on errors."""

    def test_leaderboard_error_has_retry(self) -> None:
        """Test that leaderboard error display has retry."""
        html = create_index_html()
        # ErrorDisplay is used with onRetry callback
        assert "onRetry={loadMatrix}" in html

    def test_timeline_error_has_retry(self) -> None:
        """Test that timeline error display has retry."""
        html = create_index_html()
        assert "onRetry={loadTimeline}" in html

    def test_comparison_error_has_retry(self) -> None:
        """Test that comparison error display has retry."""
        html = create_index_html()
        assert "onRetry={loadComparison}" in html


class TestAccessibility:
    """Tests for accessibility features."""

    def test_keyboard_hint_displayed(self) -> None:
        """Test that keyboard navigation hint is displayed."""
        html = create_index_html()
        assert "arrow keys" in html.lower()
        assert "Enter to view details" in html

    def test_timeline_has_region_role(self) -> None:
        """Test that timeline has region role for screen readers."""
        html = create_index_html()
        assert 'role="region"' in html

    def test_events_have_descriptive_aria_labels(self) -> None:
        """Test that events have descriptive aria labels."""
        html = create_index_html()
        assert "aria-selected" in html
