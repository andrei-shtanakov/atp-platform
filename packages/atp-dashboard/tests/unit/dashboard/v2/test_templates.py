"""Tests for ATP Dashboard v2 templates.

These tests verify that templates render correctly and contain expected elements.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.templating import Jinja2Templates

# Template directory
TEMPLATES_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "atp/dashboard/v2/templates"
)


@pytest.fixture
def templates() -> Jinja2Templates:
    """Create Jinja2Templates instance for testing."""
    return Jinja2Templates(directory=str(TEMPLATES_DIR))


@pytest.fixture
def mock_request() -> Request:
    """Create a mock Request object for template rendering."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "server": ("localhost", 8000),
        "root_path": "",
        "app": MagicMock(),
    }
    # Mock url_for function for static file URLs
    scope["app"].url_path_for = MagicMock(return_value="/static/v2/test.js")

    request = Request(scope)
    return request


class TestBaseTemplate:
    """Tests for base.html template."""

    def test_base_template_exists(self) -> None:
        """Test that base.html template exists."""
        template_path = TEMPLATES_DIR / "base.html"
        assert template_path.exists(), "base.html template should exist"

    def test_base_template_contains_doctype(self) -> None:
        """Test that base.html contains DOCTYPE declaration."""
        template_path = TEMPLATES_DIR / "base.html"
        content = template_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_base_template_contains_tailwind(self) -> None:
        """Test that base.html includes Tailwind CSS."""
        template_path = TEMPLATES_DIR / "base.html"
        content = template_path.read_text()
        assert "tailwindcss" in content

    def test_base_template_contains_react(self) -> None:
        """Test that base.html includes React."""
        template_path = TEMPLATES_DIR / "base.html"
        content = template_path.read_text()
        assert "react" in content.lower()

    def test_base_template_contains_chartjs(self) -> None:
        """Test that base.html includes Chart.js."""
        template_path = TEMPLATES_DIR / "base.html"
        content = template_path.read_text()
        assert "chart.js" in content.lower()

    def test_base_template_has_blocks(self) -> None:
        """Test that base.html defines expected blocks."""
        template_path = TEMPLATES_DIR / "base.html"
        content = template_path.read_text()
        assert "{% block title %}" in content
        assert "{% block content %}" in content
        assert "{% block scripts %}" in content

    def test_base_template_has_root_div(self) -> None:
        """Test that base.html has root div for React."""
        template_path = TEMPLATES_DIR / "base.html"
        content = template_path.read_text()
        assert 'id="root"' in content


class TestHomeTemplate:
    """Tests for home.html template."""

    def test_home_template_exists(self) -> None:
        """Test that home.html template exists."""
        template_path = TEMPLATES_DIR / "home.html"
        assert template_path.exists(), "home.html template should exist"

    def test_home_template_extends_base(self) -> None:
        """Test that home.html extends base.html."""
        template_path = TEMPLATES_DIR / "home.html"
        content = template_path.read_text()
        assert '{% extends "base.html" %}' in content

    def test_home_template_has_title_block(self) -> None:
        """Test that home.html defines title block."""
        template_path = TEMPLATES_DIR / "home.html"
        content = template_path.read_text()
        assert "{% block title %}" in content
        assert "Home" in content or "Dashboard" in content

    def test_home_template_has_root_element(self) -> None:
        """Test that home.html has a root element for React."""
        template_path = TEMPLATES_DIR / "home.html"
        content = template_path.read_text()
        assert "home-root" in content

    def test_home_template_has_react_components(self) -> None:
        """Test that home.html contains React components."""
        template_path = TEMPLATES_DIR / "home.html"
        content = template_path.read_text()
        assert "function" in content  # JSX function components
        assert "React" in content


class TestTestResultsTemplate:
    """Tests for test_results.html template."""

    def test_test_results_template_exists(self) -> None:
        """Test that test_results.html template exists."""
        template_path = TEMPLATES_DIR / "test_results.html"
        assert template_path.exists(), "test_results.html template should exist"

    def test_test_results_template_extends_base(self) -> None:
        """Test that test_results.html extends base.html."""
        template_path = TEMPLATES_DIR / "test_results.html"
        content = template_path.read_text()
        assert '{% extends "base.html" %}' in content

    def test_test_results_template_has_title(self) -> None:
        """Test that test_results.html has appropriate title."""
        template_path = TEMPLATES_DIR / "test_results.html"
        content = template_path.read_text()
        assert "Test Results" in content

    def test_test_results_template_has_root_element(self) -> None:
        """Test that test_results.html has a root element for React."""
        template_path = TEMPLATES_DIR / "test_results.html"
        content = template_path.read_text()
        assert "test-results-root" in content

    def test_test_results_template_has_suite_detail(self) -> None:
        """Test that test_results.html contains SuiteDetail component."""
        template_path = TEMPLATES_DIR / "test_results.html"
        content = template_path.read_text()
        assert "SuiteDetail" in content


class TestComparisonTemplate:
    """Tests for comparison.html template."""

    def test_comparison_template_exists(self) -> None:
        """Test that comparison.html template exists."""
        template_path = TEMPLATES_DIR / "comparison.html"
        assert template_path.exists(), "comparison.html template should exist"

    def test_comparison_template_extends_base(self) -> None:
        """Test that comparison.html extends base.html."""
        template_path = TEMPLATES_DIR / "comparison.html"
        content = template_path.read_text()
        assert '{% extends "base.html" %}' in content

    def test_comparison_template_has_title(self) -> None:
        """Test that comparison.html has appropriate title."""
        template_path = TEMPLATES_DIR / "comparison.html"
        content = template_path.read_text()
        assert "Comparison" in content

    def test_comparison_template_has_root_element(self) -> None:
        """Test that comparison.html has a root element for React."""
        template_path = TEMPLATES_DIR / "comparison.html"
        content = template_path.read_text()
        assert "comparison-root" in content

    def test_comparison_template_has_agent_selector(self) -> None:
        """Test that comparison.html contains AgentSelector component."""
        template_path = TEMPLATES_DIR / "comparison.html"
        content = template_path.read_text()
        assert "AgentSelector" in content


class TestChartsComponent:
    """Tests for components/charts.html template."""

    def test_charts_component_exists(self) -> None:
        """Test that charts.html component exists."""
        template_path = TEMPLATES_DIR / "components/charts.html"
        assert template_path.exists(), "components/charts.html should exist"

    def test_charts_component_has_macros(self) -> None:
        """Test that charts.html defines Jinja2 macros."""
        template_path = TEMPLATES_DIR / "components/charts.html"
        content = template_path.read_text()
        assert "{% macro" in content

    def test_charts_component_has_trend_chart(self) -> None:
        """Test that charts.html has trend chart macro."""
        template_path = TEMPLATES_DIR / "components/charts.html"
        content = template_path.read_text()
        assert "trend_chart" in content

    def test_charts_component_has_bar_chart(self) -> None:
        """Test that charts.html has bar chart macro."""
        template_path = TEMPLATES_DIR / "components/charts.html"
        content = template_path.read_text()
        assert "bar_chart" in content

    def test_charts_component_has_pie_chart(self) -> None:
        """Test that charts.html has pie chart macro."""
        template_path = TEMPLATES_DIR / "components/charts.html"
        content = template_path.read_text()
        assert "pie_chart" in content

    def test_charts_component_has_chart_scripts(self) -> None:
        """Test that charts.html has JavaScript for chart initialization."""
        template_path = TEMPLATES_DIR / "components/charts.html"
        content = template_path.read_text()
        assert "createTrendChart" in content
        assert "createBarChart" in content


class TestTablesComponent:
    """Tests for components/tables.html template."""

    def test_tables_component_exists(self) -> None:
        """Test that tables.html component exists."""
        template_path = TEMPLATES_DIR / "components/tables.html"
        assert template_path.exists(), "components/tables.html should exist"

    def test_tables_component_has_macros(self) -> None:
        """Test that tables.html defines Jinja2 macros."""
        template_path = TEMPLATES_DIR / "components/tables.html"
        content = template_path.read_text()
        assert "{% macro" in content

    def test_tables_component_has_data_table(self) -> None:
        """Test that tables.html has data table macro."""
        template_path = TEMPLATES_DIR / "components/tables.html"
        content = template_path.read_text()
        assert "data_table" in content

    def test_tables_component_has_table_header(self) -> None:
        """Test that tables.html has table header macro."""
        template_path = TEMPLATES_DIR / "components/tables.html"
        content = template_path.read_text()
        assert "table_header" in content

    def test_tables_component_has_status_cell(self) -> None:
        """Test that tables.html has status cell macro."""
        template_path = TEMPLATES_DIR / "components/tables.html"
        content = template_path.read_text()
        assert "status_cell" in content

    def test_tables_component_has_pagination(self) -> None:
        """Test that tables.html has pagination macro."""
        template_path = TEMPLATES_DIR / "components/tables.html"
        content = template_path.read_text()
        assert "pagination" in content

    def test_tables_component_has_empty_state(self) -> None:
        """Test that tables.html has empty state macro."""
        template_path = TEMPLATES_DIR / "components/tables.html"
        content = template_path.read_text()
        assert "empty_state" in content

    def test_tables_component_has_skeleton_rows(self) -> None:
        """Test that tables.html has skeleton rows macro."""
        template_path = TEMPLATES_DIR / "components/tables.html"
        content = template_path.read_text()
        assert "skeleton_rows" in content


class TestNavigationComponent:
    """Tests for components/navigation.html template."""

    def test_navigation_component_exists(self) -> None:
        """Test that navigation.html component exists."""
        template_path = TEMPLATES_DIR / "components/navigation.html"
        assert template_path.exists(), "components/navigation.html should exist"

    def test_navigation_component_has_navbar_macro(self) -> None:
        """Test that navigation.html has navbar macro."""
        template_path = TEMPLATES_DIR / "components/navigation.html"
        content = template_path.read_text()
        assert "navbar" in content

    def test_navigation_component_has_breadcrumb_macro(self) -> None:
        """Test that navigation.html has breadcrumb macro."""
        template_path = TEMPLATES_DIR / "components/navigation.html"
        content = template_path.read_text()
        assert "breadcrumb" in content

    def test_navigation_component_has_tabs_macro(self) -> None:
        """Test that navigation.html has tabs macro."""
        template_path = TEMPLATES_DIR / "components/navigation.html"
        content = template_path.read_text()
        assert "tabs" in content

    def test_navigation_component_has_mobile_menu(self) -> None:
        """Test that navigation.html has mobile menu functionality."""
        template_path = TEMPLATES_DIR / "components/navigation.html"
        content = template_path.read_text()
        assert "mobile-menu" in content


class TestStaticFiles:
    """Tests for static CSS and JS files."""

    def test_dashboard_css_exists(self) -> None:
        """Test that dashboard.css exists."""
        css_path = TEMPLATES_DIR.parent / "static/css/dashboard.css"
        assert css_path.exists(), "static/css/dashboard.css should exist"

    def test_dashboard_css_has_chart_styles(self) -> None:
        """Test that dashboard.css has chart container styles."""
        css_path = TEMPLATES_DIR.parent / "static/css/dashboard.css"
        content = css_path.read_text()
        assert ".chart-container" in content

    def test_dashboard_css_has_skeleton_animation(self) -> None:
        """Test that dashboard.css has skeleton loader animation."""
        css_path = TEMPLATES_DIR.parent / "static/css/dashboard.css"
        content = css_path.read_text()
        assert "skeleton-pulse" in content

    def test_api_js_exists(self) -> None:
        """Test that api.js exists."""
        js_path = TEMPLATES_DIR.parent / "static/js/api.js"
        assert js_path.exists(), "static/js/api.js should exist"

    def test_api_js_has_get_method(self) -> None:
        """Test that api.js has GET method."""
        js_path = TEMPLATES_DIR.parent / "static/js/api.js"
        content = js_path.read_text()
        assert "async get" in content

    def test_api_js_has_post_method(self) -> None:
        """Test that api.js has POST method."""
        js_path = TEMPLATES_DIR.parent / "static/js/api.js"
        content = js_path.read_text()
        assert "async post" in content

    def test_components_js_exists(self) -> None:
        """Test that components.js exists."""
        js_path = TEMPLATES_DIR.parent / "static/js/components.js"
        assert js_path.exists(), "static/js/components.js should exist"

    def test_components_js_has_skeleton_components(self) -> None:
        """Test that components.js has skeleton loader components."""
        js_path = TEMPLATES_DIR.parent / "static/js/components.js"
        content = js_path.read_text()
        assert "SkeletonBox" in content
        assert "SkeletonSuiteList" in content

    def test_components_js_has_error_boundary(self) -> None:
        """Test that components.js has ErrorBoundary component."""
        js_path = TEMPLATES_DIR.parent / "static/js/components.js"
        content = js_path.read_text()
        assert "ErrorBoundary" in content

    def test_components_js_exports_globally(self) -> None:
        """Test that components.js exports to window object."""
        js_path = TEMPLATES_DIR.parent / "static/js/components.js"
        content = js_path.read_text()
        assert "window.SkeletonBox" in content
        assert "window.ErrorBoundary" in content


class TestTemplateSnapshot:
    """Snapshot tests for template content stability."""

    def test_base_template_structure(self) -> None:
        """Test base.html has stable structure with key elements."""
        template_path = TEMPLATES_DIR / "base.html"
        content = template_path.read_text()

        # Essential structural elements
        essential_elements = [
            "<!DOCTYPE html>",
            "<html",
            "<head>",
            "<body",
            "{% block title %}",
            "{% block content %}",
            "{% block scripts %}",
            "tailwindcss",
            "react",
            "chart.js",
        ]

        for element in essential_elements:
            assert element.lower() in content.lower(), (
                f"base.html should contain: {element}"
            )

    def test_home_template_structure(self) -> None:
        """Test home.html has stable structure with key elements."""
        template_path = TEMPLATES_DIR / "home.html"
        content = template_path.read_text()

        # Essential structural elements
        essential_elements = [
            '{% extends "base.html" %}',
            "{% block title %}",
            "{% block content %}",
            "{% block scripts %}",
            "DashboardSummary",
            "SuiteList",
            "HomePage",
        ]

        for element in essential_elements:
            assert element in content, f"home.html should contain: {element}"

    def test_comparison_template_structure(self) -> None:
        """Test comparison.html has stable structure with key elements."""
        template_path = TEMPLATES_DIR / "comparison.html"
        content = template_path.read_text()

        # Essential structural elements
        essential_elements = [
            '{% extends "base.html" %}',
            "{% block title %}",
            "{% block content %}",
            "AgentSelector",
            "MetricsPanel",
            "ComparisonPage",
        ]

        for element in essential_elements:
            assert element in content, f"comparison.html should contain: {element}"


class TestDirectoryStructure:
    """Tests for template directory structure."""

    def test_templates_directory_exists(self) -> None:
        """Test that templates directory exists."""
        assert TEMPLATES_DIR.exists()

    def test_components_subdirectory_exists(self) -> None:
        """Test that components subdirectory exists."""
        components_dir = TEMPLATES_DIR / "components"
        assert components_dir.exists()

    def test_static_css_directory_exists(self) -> None:
        """Test that static/css directory exists."""
        css_dir = TEMPLATES_DIR.parent / "static/css"
        assert css_dir.exists()

    def test_static_js_directory_exists(self) -> None:
        """Test that static/js directory exists."""
        js_dir = TEMPLATES_DIR.parent / "static/js"
        assert js_dir.exists()

    def test_all_expected_templates_exist(self) -> None:
        """Test that all expected template files exist."""
        expected_templates = [
            "base.html",
            "home.html",
            "test_results.html",
            "comparison.html",
            "components/charts.html",
            "components/tables.html",
            "components/navigation.html",
        ]

        for template in expected_templates:
            template_path = TEMPLATES_DIR / template
            assert template_path.exists(), f"Template {template} should exist"
