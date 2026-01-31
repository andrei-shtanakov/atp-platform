"""Tests for ATP Dashboard FastAPI application."""

from atp.dashboard.app import app, create_index_html


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_app_is_fastapi_instance(self) -> None:
        """Test that app is a FastAPI instance."""
        assert app is not None
        assert hasattr(app, "router")

    def test_app_has_title(self) -> None:
        """Test that app has correct title."""
        assert app.title == "ATP Dashboard"

    def test_app_includes_api_router(self) -> None:
        """Test that app includes API router."""
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert any("/api" in str(p) for p in route_paths)

    def test_app_has_cors_middleware(self) -> None:
        """Test that app has CORS middleware."""
        # Check middleware stack includes CORS
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes


class TestHTMLContent:
    """Tests for HTML content generation."""

    def test_create_index_html_returns_string(self) -> None:
        """Test that create_index_html returns HTML string."""
        html = create_index_html()
        assert isinstance(html, str)
        assert len(html) > 0

    def test_html_content_has_doctype(self) -> None:
        """Test that HTML has DOCTYPE."""
        html = create_index_html()
        assert "<!DOCTYPE html>" in html

    def test_html_content_has_html_tag(self) -> None:
        """Test that HTML has html tag."""
        html = create_index_html()
        assert "<html" in html
        assert "</html>" in html

    def test_html_content_has_head(self) -> None:
        """Test that HTML has head section."""
        html = create_index_html()
        assert "<head>" in html
        assert "</head>" in html

    def test_html_content_has_body(self) -> None:
        """Test that HTML has body section."""
        html = create_index_html()
        assert "<body" in html
        assert "</body>" in html

    def test_html_content_has_title(self) -> None:
        """Test that HTML has title."""
        html = create_index_html()
        assert "<title>" in html
        assert "ATP Dashboard" in html

    def test_html_content_has_react(self) -> None:
        """Test that HTML includes React."""
        html = create_index_html()
        assert "react" in html.lower()

    def test_html_content_has_root_div(self) -> None:
        """Test that HTML has root div for React."""
        html = create_index_html()
        assert 'id="root"' in html

    def test_html_content_has_script(self) -> None:
        """Test that HTML has script tags."""
        html = create_index_html()
        assert "<script" in html
        assert "</script>" in html

    def test_html_content_has_style(self) -> None:
        """Test that HTML has style tags."""
        html = create_index_html()
        assert "<style>" in html

    def test_html_content_has_tailwind(self) -> None:
        """Test that HTML includes Tailwind CSS."""
        html = create_index_html()
        assert "tailwindcss" in html.lower()

    def test_html_content_has_chartjs(self) -> None:
        """Test that HTML includes Chart.js."""
        html = create_index_html()
        assert "chart.js" in html.lower()

    def test_html_content_has_timeline_components(self) -> None:
        """Test that HTML includes timeline UI components."""
        html = create_index_html()
        assert "TimelineContainer" in html
        assert "TimelineRow" in html
        assert "TimeScale" in html
        assert "EventMarker" in html

    def test_html_content_has_timeline_view(self) -> None:
        """Test that HTML includes TimelineView component."""
        html = create_index_html()
        assert "TimelineView" in html
        assert "view === 'timeline'" in html

    def test_html_content_has_timeline_navigation(self) -> None:
        """Test that HTML includes Timeline navigation button."""
        html = create_index_html()
        assert "setView('timeline')" in html

    def test_html_content_has_event_detail_panel(self) -> None:
        """Test that HTML includes EventDetailPanel component."""
        html = create_index_html()
        assert "EventDetailPanel" in html

    def test_html_content_has_event_detail_panel_with_type_specific_displays(
        self,
    ) -> None:
        """Test that EventDetailPanel has type-specific display components."""
        html = create_index_html()
        # Check for type-specific helper components
        assert "ToolCallDetails" in html
        assert "LLMRequestDetails" in html
        assert "ErrorDetails" in html
        assert "ReasoningDetails" in html
        assert "ProgressDetails" in html

    def test_html_content_has_event_filters_component(self) -> None:
        """Test that HTML includes EventFilters component."""
        html = create_index_html()
        assert "EventFilters" in html
        assert "onFilterChange" in html

    def test_html_content_has_copy_json_functionality(self) -> None:
        """Test that HTML includes copy JSON functionality."""
        html = create_index_html()
        assert "handleCopyJSON" in html
        assert "Copy JSON" in html
        assert "navigator.clipboard" in html

    def test_html_content_has_tool_call_details(self) -> None:
        """Test that ToolCallDetails shows tool name, args, result."""
        html = create_index_html()
        # Check for tool_call specific labels/elements
        assert "Tool Name" in html
        assert "Arguments" in html
        assert "Result" in html

    def test_html_content_has_llm_request_details(self) -> None:
        """Test that LLMRequestDetails shows prompt, response, tokens."""
        html = create_index_html()
        # Check for llm_request specific labels/elements
        assert "Prompt" in html
        assert "Response" in html
        assert "Input Tokens" in html
        assert "Output Tokens" in html
        assert "Total Tokens" in html

    def test_html_content_has_error_details(self) -> None:
        """Test that ErrorDetails shows error message and stack trace."""
        html = create_index_html()
        # Check for error specific labels/elements
        assert "Error Message" in html
        assert "Stack Trace" in html

    def test_html_content_has_event_tooltip(self) -> None:
        """Test that HTML includes EventTooltip component."""
        html = create_index_html()
        assert "EventTooltip" in html

    def test_html_content_has_zoom_controls(self) -> None:
        """Test that HTML includes zoom controls for timeline."""
        html = create_index_html()
        assert "handleZoomIn" in html
        assert "handleZoomOut" in html
        assert "handleZoomReset" in html
        assert "zoomLevel" in html

    def test_html_content_has_timeline_styles(self) -> None:
        """Test that HTML includes timeline CSS styles."""
        html = create_index_html()
        assert "timeline-container" in html or "timeline" in html.lower()


class TestTestCreatorFormComponents:
    """Tests for TestCreatorForm UI components."""

    def test_html_content_has_test_creator_form(self) -> None:
        """Test that HTML includes TestCreatorForm component."""
        html = create_index_html()
        assert "TestCreatorForm" in html
        assert "TestCreatorView" in html

    def test_html_content_has_step_indicator(self) -> None:
        """Test that HTML includes StepIndicator component."""
        html = create_index_html()
        assert "StepIndicator" in html
        assert "currentStep" in html

    def test_html_content_has_suite_details_step(self) -> None:
        """Test that HTML includes SuiteDetailsStep component."""
        html = create_index_html()
        assert "SuiteDetailsStep" in html
        assert "Suite Name" in html
        assert "runs_per_test" in html

    def test_html_content_has_template_selection_step(self) -> None:
        """Test that HTML includes TemplateSelectionStep component."""
        html = create_index_html()
        assert "TemplateSelectionStep" in html
        assert "TemplateCard" in html

    def test_html_content_has_yaml_preview_step(self) -> None:
        """Test that HTML includes YAMLPreviewStep component."""
        html = create_index_html()
        assert "YAMLPreviewStep" in html
        assert "YAML Preview" in html
        assert "generateYAMLPreview" in html

    def test_html_content_has_template_card(self) -> None:
        """Test that HTML includes TemplateCard component."""
        html = create_index_html()
        assert "TemplateCard" in html
        assert "isSelected" in html
        assert "onSelect" in html

    def test_html_content_has_test_item(self) -> None:
        """Test that HTML includes TestItem component."""
        html = create_index_html()
        assert "TestItem" in html
        assert "onRemove" in html
        assert "onEdit" in html

    def test_html_content_has_test_edit_modal(self) -> None:
        """Test that HTML includes TestEditModal component."""
        html = create_index_html()
        assert "TestEditModal" in html
        assert "Test ID" in html
        assert "Test Name" in html
        assert "Task Description" in html

    def test_html_content_has_create_navigation(self) -> None:
        """Test that HTML includes Create navigation button."""
        html = create_index_html()
        assert "setView('create')" in html
        assert "+ Create" in html

    def test_html_content_has_create_view(self) -> None:
        """Test that HTML includes create view handler."""
        html = create_index_html()
        assert "view === 'create'" in html

    def test_html_content_has_category_colors(self) -> None:
        """Test that HTML includes category color constants."""
        html = create_index_html()
        assert "CATEGORY_COLORS" in html
        assert "getCategoryColor" in html

    def test_html_content_has_scoring_weights(self) -> None:
        """Test that HTML includes scoring weight controls."""
        html = create_index_html()
        assert "quality_weight" in html
        assert "completeness_weight" in html
        assert "efficiency_weight" in html
        assert "cost_weight" in html

    def test_html_content_has_suite_api_post(self) -> None:
        """Test that HTML includes API POST for suite definitions."""
        html = create_index_html()
        assert "api.post('/suite-definitions'" in html

    def test_html_content_has_templates_api_get(self) -> None:
        """Test that HTML includes API GET for templates."""
        html = create_index_html()
        assert "api.get('/templates')" in html

    def test_html_content_has_copy_to_clipboard(self) -> None:
        """Test that HTML includes copy to clipboard functionality."""
        html = create_index_html()
        assert "Copy to Clipboard" in html
        assert "handleCopy" in html

    def test_html_content_has_skeleton_test_creator(self) -> None:
        """Test that HTML includes SkeletonTestCreator component."""
        html = create_index_html()
        assert "SkeletonTestCreator" in html

    def test_html_content_has_multi_step_wizard(self) -> None:
        """Test that HTML includes multi-step wizard logic."""
        html = create_index_html()
        assert "handleNext" in html
        assert "handleBack" in html
        assert "validateStep" in html


class TestAppRoutes:
    """Tests for app routes."""

    def test_root_route_exists(self) -> None:
        """Test that root route exists."""
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/" in route_paths

    def test_api_routes_exist(self) -> None:
        """Test that API routes exist."""
        route_paths = [str(r.path) for r in app.routes if hasattr(r, "path")]
        # Check some API endpoints exist
        assert any("/api/dashboard" in p for p in route_paths)
        assert any("/api/agents" in p for p in route_paths)
        assert any("/api/suites" in p for p in route_paths)
