"""Tests for ATP Dashboard v2 HTMX templates.

Verifies that templates exist, have expected structure, and include
key elements (Pico CSS, HTMX, Chart.js).
"""

from pathlib import Path

import pytest

# Template directory (v2 HTMX templates)
TEMPLATES_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "atp/dashboard/v2/templates"
)


class TestBaseUITemplate:
    """Tests for base_ui.html (v2 HTMX layout)."""

    def test_exists(self) -> None:
        path = TEMPLATES_DIR / "ui" / "base_ui.html"
        assert path.exists()

    def test_has_doctype(self) -> None:
        content = (TEMPLATES_DIR / "ui" / "base_ui.html").read_text()
        assert "<!DOCTYPE html>" in content

    def test_has_pico_css(self) -> None:
        content = (TEMPLATES_DIR / "ui" / "base_ui.html").read_text()
        assert "pico" in content.lower()

    def test_has_htmx(self) -> None:
        content = (TEMPLATES_DIR / "ui" / "base_ui.html").read_text()
        assert "htmx" in content.lower()

    def test_has_content_block(self) -> None:
        content = (TEMPLATES_DIR / "ui" / "base_ui.html").read_text()
        assert "{% block content %}" in content

    def test_has_sidebar_nav(self) -> None:
        content = (TEMPLATES_DIR / "ui" / "base_ui.html").read_text()
        assert "sidebar" in content.lower()
        for page in ["Benchmarks", "Runs", "Leaderboard", "Analytics"]:
            assert page in content


class TestUIPages:
    """Tests that expected UI page templates exist."""

    @pytest.mark.parametrize(
        "template",
        [
            "ui/base_ui.html",
            "ui/login.html",
            "ui/home.html",
            "ui/benchmarks.html",
            "ui/runs.html",
            "ui/analytics.html",
            "ui/leaderboard.html",
            "ui/games.html",
            "ui/suites.html",
        ],
    )
    def test_template_exists(self, template: str) -> None:
        path = TEMPLATES_DIR / template
        assert path.exists(), f"Template {template} should exist"


class TestAnalyticsTemplate:
    """Tests for analytics.html with Chart.js."""

    def test_has_chartjs_cdn(self) -> None:
        content = (TEMPLATES_DIR / "ui" / "analytics.html").read_text()
        assert "chart.js" in content.lower()

    def test_has_status_chart_canvas(self) -> None:
        content = (TEMPLATES_DIR / "ui" / "analytics.html").read_text()
        assert "statusChart" in content

    def test_has_agents_chart_canvas(self) -> None:
        content = (TEMPLATES_DIR / "ui" / "analytics.html").read_text()
        assert "agentsChart" in content

    def test_has_stat_cards(self) -> None:
        content = (TEMPLATES_DIR / "ui" / "analytics.html").read_text()
        assert "stat-card" in content


class TestStaticFiles:
    """Tests for v2 static assets."""

    def test_css_directory_exists(self) -> None:
        css_dir = TEMPLATES_DIR.parent / "static" / "css"
        assert css_dir.exists()

    def test_ui_css_exists(self) -> None:
        css_path = TEMPLATES_DIR.parent / "static" / "css" / "ui.css"
        assert css_path.exists()
