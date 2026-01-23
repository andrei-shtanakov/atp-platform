"""HTML reporter for rich visual output."""

import html
from datetime import datetime
from pathlib import Path
from typing import TextIO

from jinja2 import BaseLoader, Environment

from atp.reporters.base import Reporter, SuiteReport

# Chart.js CDN URL embedded in a data URI approach or inline script
CHARTJS_VERSION = "4.4.1"
CHARTJS_CDN = (
    f"https://cdn.jsdelivr.net/npm/chart.js@{CHARTJS_VERSION}/dist/chart.umd.min.js"
)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --color-success: #22c55e;
            --color-success-bg: #dcfce7;
            --color-error: #ef4444;
            --color-error-bg: #fee2e2;
            --color-warning: #f59e0b;
            --color-warning-bg: #fef3c7;
            --color-info: #3b82f6;
            --color-info-bg: #dbeafe;
            --color-gray-50: #f9fafb;
            --color-gray-100: #f3f4f6;
            --color-gray-200: #e5e7eb;
            --color-gray-300: #d1d5db;
            --color-gray-500: #6b7280;
            --color-gray-700: #374151;
            --color-gray-900: #111827;
            --border-radius: 8px;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--color-gray-900);
            background-color: var(--color-gray-50);
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        h1, h2, h3 {
            font-weight: 600;
        }

        h1 {
            font-size: 1.875rem;
            margin-bottom: 1.5rem;
        }

        h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--color-gray-700);
        }

        h3 {
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }

        .header {
            background: white;
            border-radius: var(--border-radius);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-gray-200);
        }

        .header-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .header-meta {
            display: flex;
            gap: 2rem;
            color: var(--color-gray-500);
            font-size: 0.875rem;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .summary-card {
            background: white;
            border-radius: var(--border-radius);
            padding: 1.25rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-gray-200);
        }

        .summary-card.success {
            border-left: 4px solid var(--color-success);
        }

        .summary-card.error {
            border-left: 4px solid var(--color-error);
        }

        .summary-card.info {
            border-left: 4px solid var(--color-info);
        }

        .summary-card-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--color-gray-500);
            margin-bottom: 0.25rem;
        }

        .summary-card-value {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .summary-card-value.success {
            color: var(--color-success);
        }

        .summary-card-value.error {
            color: var(--color-error);
        }

        .charts-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .chart-container {
            background: white;
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-gray-200);
        }

        .chart-wrapper {
            position: relative;
            height: 250px;
        }

        .tests-section {
            background: white;
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-gray-200);
        }

        .test-item {
            border: 1px solid var(--color-gray-200);
            border-radius: var(--border-radius);
            margin-bottom: 0.75rem;
            overflow: hidden;
        }

        .test-item:last-child {
            margin-bottom: 0;
        }

        .test-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            cursor: pointer;
            background: var(--color-gray-50);
            transition: background-color 0.15s ease;
        }

        .test-header:hover {
            background: var(--color-gray-100);
        }

        .test-header-left {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .test-status {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }

        .test-status.success {
            background-color: var(--color-success);
        }

        .test-status.error {
            background-color: var(--color-error);
        }

        .test-name {
            font-weight: 500;
        }

        .test-header-right {
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: 0.875rem;
            color: var(--color-gray-500);
        }

        .test-score {
            font-weight: 600;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
        }

        .test-score.high {
            background-color: var(--color-success-bg);
            color: var(--color-success);
        }

        .test-score.medium {
            background-color: var(--color-warning-bg);
            color: var(--color-warning);
        }

        .test-score.low {
            background-color: var(--color-error-bg);
            color: var(--color-error);
        }

        .expand-icon {
            width: 20px;
            height: 20px;
            transition: transform 0.2s ease;
        }

        .test-item.expanded .expand-icon {
            transform: rotate(180deg);
        }

        .test-details {
            display: none;
            padding: 1rem;
            border-top: 1px solid var(--color-gray-200);
            background: white;
        }

        .test-item.expanded .test-details {
            display: block;
        }

        .detail-section {
            margin-bottom: 1rem;
        }

        .detail-section:last-child {
            margin-bottom: 0;
        }

        .detail-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--color-gray-500);
            margin-bottom: 0.5rem;
        }

        .checks-list {
            list-style: none;
        }

        .check-item {
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--color-gray-100);
        }

        .check-item:last-child {
            border-bottom: none;
        }

        .check-icon {
            width: 16px;
            height: 16px;
            flex-shrink: 0;
            margin-top: 0.125rem;
        }

        .check-icon.success {
            color: var(--color-success);
        }

        .check-icon.error {
            color: var(--color-error);
        }

        .check-content {
            flex: 1;
        }

        .check-name {
            font-weight: 500;
        }

        .check-evaluator {
            font-size: 0.75rem;
            color: var(--color-gray-500);
        }

        .check-message {
            font-size: 0.875rem;
            color: var(--color-gray-700);
            margin-top: 0.25rem;
        }

        .check-failed {
            background-color: var(--color-error-bg);
            padding: 0.5rem 0.75rem;
            border-radius: var(--border-radius);
            margin: 0.25rem 0;
        }

        .score-breakdown {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.5rem;
        }

        .score-component {
            text-align: center;
            padding: 0.75rem;
            background: var(--color-gray-50);
            border-radius: var(--border-radius);
        }

        .score-component-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            color: var(--color-gray-500);
        }

        .score-component-value {
            font-size: 1.125rem;
            font-weight: 600;
        }

        .trace-viewer {
            background: var(--color-gray-900);
            color: var(--color-gray-100);
            border-radius: var(--border-radius);
            overflow: hidden;
        }

        .trace-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: var(--color-gray-700);
            cursor: pointer;
        }

        .trace-header:hover {
            background: var(--color-gray-600);
        }

        .trace-content {
            display: none;
            padding: 1rem;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            font-size: 0.8rem;
            line-height: 1.5;
        }

        .trace-viewer.expanded .trace-content {
            display: block;
        }

        .trace-event {
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--color-gray-700);
        }

        .trace-event:last-child {
            border-bottom: none;
        }

        .trace-event-type {
            color: var(--color-info);
            font-weight: 500;
        }

        .trace-event-data {
            color: var(--color-gray-300);
            margin-left: 1rem;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .statistics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.5rem;
        }

        .stat-item {
            background: var(--color-gray-50);
            padding: 0.75rem;
            border-radius: var(--border-radius);
            text-align: center;
        }

        .stat-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            color: var(--color-gray-500);
        }

        .stat-value {
            font-weight: 600;
        }

        .stat-range {
            font-size: 0.75rem;
            color: var(--color-gray-500);
        }

        .error-message {
            background: var(--color-error-bg);
            color: var(--color-error);
            padding: 1rem;
            border-radius: var(--border-radius);
            margin-top: 1rem;
            font-family: monospace;
            white-space: pre-wrap;
        }

        .stability-badge {
            display: inline-block;
            padding: 0.125rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .stability-badge.stable {
            background-color: var(--color-success-bg);
            color: var(--color-success);
        }

        .stability-badge.moderate {
            background-color: var(--color-warning-bg);
            color: var(--color-warning);
        }

        .stability-badge.unstable, .stability-badge.critical {
            background-color: var(--color-error-bg);
            color: var(--color-error);
        }

        .footer {
            text-align: center;
            color: var(--color-gray-500);
            font-size: 0.75rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--color-gray-200);
        }

        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }

            .summary-grid {
                grid-template-columns: 1fr 1fr;
            }

            .charts-section {
                grid-template-columns: 1fr;
            }

            .score-breakdown {
                grid-template-columns: repeat(2, 1fr);
            }

            .test-header-right {
                flex-direction: column;
                align-items: flex-end;
                gap: 0.25rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>{{ title }}</h1>
        </div>
        <div class="header-meta">
            <span><strong>Suite:</strong> {{ report.suite_name }}</span>
            <span><strong>Agent:</strong> {{ report.agent_name }}</span>
            <span><strong>Generated:</strong> {{ generated_at }}</span>
            {% if report.runs_per_test > 1 %}
            <span><strong>Runs per test:</strong> {{ report.runs_per_test }}</span>
            {% endif %}
        </div>
    </div>

    <div class="summary-grid">
        <div class="summary-card {{ 'success' if report.passed_tests == report.total_tests else 'info' }}">
            <div class="summary-card-label">Total Tests</div>
            <div class="summary-card-value">{{ report.total_tests }}</div>
        </div>
        <div class="summary-card success">
            <div class="summary-card-label">Passed</div>
            <div class="summary-card-value success">{{ report.passed_tests }}</div>
        </div>
        <div class="summary-card {{ 'error' if report.failed_tests > 0 else 'success' }}">
            <div class="summary-card-label">Failed</div>
            <div class="summary-card-value {{ 'error' if report.failed_tests > 0 else '' }}">{{ report.failed_tests }}</div>
        </div>
        <div class="summary-card info">
            <div class="summary-card-label">Success Rate</div>
            <div class="summary-card-value">{{ "%.1f"|format(report.success_rate * 100) }}%</div>
        </div>
        <div class="summary-card info">
            <div class="summary-card-label">Duration</div>
            <div class="summary-card-value">{{ format_duration(report.duration_seconds) }}</div>
        </div>
    </div>

    {% if report.error %}
    <div class="error-message">
        <strong>Suite Error:</strong> {{ report.error }}
    </div>
    {% endif %}

    <div class="charts-section">
        <div class="chart-container">
            <h2>Test Results</h2>
            <div class="chart-wrapper">
                <canvas id="resultsChart"></canvas>
            </div>
        </div>
        {% if has_scores %}
        <div class="chart-container">
            <h2>Score Distribution</h2>
            <div class="chart-wrapper">
                <canvas id="scoresChart"></canvas>
            </div>
        </div>
        {% endif %}
    </div>

    <div class="tests-section">
        <h2>Test Details</h2>
        {% for test in report.tests %}
        <div class="test-item" id="test-{{ test.test_id }}">
            <div class="test-header" onclick="toggleTest('test-{{ test.test_id }}')">
                <div class="test-header-left">
                    <div class="test-status {{ 'success' if test.success else 'error' }}"></div>
                    <span class="test-name">{{ test.test_name }}</span>
                    {% if test.statistics and test.statistics.score_stability %}
                    <span class="stability-badge {{ test.statistics.score_stability.level.value }}">
                        {{ test.statistics.score_stability.level.value }}
                    </span>
                    {% endif %}
                </div>
                <div class="test-header-right">
                    {% if test.score is not none %}
                    <span class="test-score {{ score_class(test.score) }}">{{ "%.1f"|format(test.score) }}/100</span>
                    {% endif %}
                    <span>{{ format_duration(test.duration_seconds) }}</span>
                    <svg class="expand-icon" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
                    </svg>
                </div>
            </div>
            <div class="test-details">
                {% if test.error %}
                <div class="detail-section">
                    <div class="error-message">{{ test.error }}</div>
                </div>
                {% endif %}

                {% if test.total_runs > 1 %}
                <div class="detail-section">
                    <div class="detail-label">Run Statistics</div>
                    <div class="statistics-grid">
                        <div class="stat-item">
                            <div class="stat-label">Runs</div>
                            <div class="stat-value">{{ test.successful_runs }}/{{ test.total_runs }}</div>
                        </div>
                        {% if test.statistics and test.statistics.score_stats %}
                        <div class="stat-item">
                            <div class="stat-label">Mean Score</div>
                            <div class="stat-value">{{ "%.1f"|format(test.statistics.score_stats.mean) }}</div>
                            <div class="stat-range">{{ "%.1f"|format(test.statistics.score_stats.min) }} - {{ "%.1f"|format(test.statistics.score_stats.max) }}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Std Dev</div>
                            <div class="stat-value">{{ "%.2f"|format(test.statistics.score_stats.std) }}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">CV</div>
                            <div class="stat-value">{{ "%.2f"|format(test.statistics.score_stats.coefficient_of_variation) }}</div>
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endif %}

                {% if test.scored_result and test.scored_result.breakdown %}
                <div class="detail-section">
                    <div class="detail-label">Score Breakdown</div>
                    <div class="score-breakdown">
                        {% for component in ['quality', 'completeness', 'efficiency', 'cost'] %}
                        {% set comp = test.scored_result.breakdown[component] %}
                        <div class="score-component">
                            <div class="score-component-label">{{ component|capitalize }}</div>
                            <div class="score-component-value">{{ "%.0f"|format(comp.normalized_value * 100) }}</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}

                {% if test.eval_results %}
                <div class="detail-section">
                    <div class="detail-label">Evaluation Checks</div>
                    <ul class="checks-list">
                        {% for eval_result in test.eval_results %}
                        {% for check in eval_result.checks %}
                        <li class="check-item {{ '' if check.passed else 'check-failed' }}">
                            <svg class="check-icon {{ 'success' if check.passed else 'error' }}" viewBox="0 0 20 20" fill="currentColor">
                                {% if check.passed %}
                                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
                                {% else %}
                                <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
                                {% endif %}
                            </svg>
                            <div class="check-content">
                                <div class="check-name">{{ check.name }}</div>
                                <div class="check-evaluator">{{ eval_result.evaluator }} - Score: {{ "%.2f"|format(check.score) }}</div>
                                {% if check.message %}
                                <div class="check-message">{{ check.message }}</div>
                                {% endif %}
                            </div>
                        </li>
                        {% endfor %}
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% if test.trace %}
                <div class="detail-section">
                    <div class="trace-viewer" id="trace-{{ test.test_id }}">
                        <div class="trace-header" onclick="toggleTrace('trace-{{ test.test_id }}')">
                            <span>Trace Events ({{ test.trace|length }})</span>
                            <svg class="expand-icon" width="16" height="16" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
                            </svg>
                        </div>
                        <div class="trace-content">
                            {% for event in test.trace %}
                            <div class="trace-event">
                                <span class="trace-event-type">{{ event.type }}</span>
                                <div class="trace-event-data">{{ event.data | tojson(indent=2) }}</div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="footer">
        Generated by ATP (Agent Test Platform) on {{ generated_at }}
    </div>

    <script>
        // Chart.js inline (minified version placeholder - will load from CDN in production)
        // For single-file output, we embed chart rendering logic
    </script>
    <script src="{{ chartjs_cdn }}"></script>
    <script>
        // Toggle test details accordion
        function toggleTest(testId) {
            const element = document.getElementById(testId);
            element.classList.toggle('expanded');
        }

        // Toggle trace viewer
        function toggleTrace(traceId) {
            const element = document.getElementById(traceId);
            element.classList.toggle('expanded');
            event.stopPropagation();
        }

        // Initialize charts when DOM is ready
        document.addEventListener('DOMContentLoaded', function() {
            // Results pie chart
            const resultsCtx = document.getElementById('resultsChart');
            if (resultsCtx) {
                new Chart(resultsCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Passed', 'Failed'],
                        datasets: [{
                            data: [{{ report.passed_tests }}, {{ report.failed_tests }}],
                            backgroundColor: ['#22c55e', '#ef4444'],
                            borderWidth: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom'
                            }
                        }
                    }
                });
            }

            // Score distribution bar chart
            {% if has_scores %}
            const scoresCtx = document.getElementById('scoresChart');
            if (scoresCtx) {
                new Chart(scoresCtx, {
                    type: 'bar',
                    data: {
                        labels: {{ test_names | tojson }},
                        datasets: [{
                            label: 'Score',
                            data: {{ test_scores | tojson }},
                            backgroundColor: {{ test_colors | tojson }},
                            borderWidth: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        indexAxis: 'y',
                        scales: {
                            x: {
                                beginAtZero: true,
                                max: 100
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            }
                        }
                    }
                });
            }
            {% endif %}

            // Auto-expand failed tests
            {% for test in report.tests %}
            {% if not test.success %}
            toggleTest('test-{{ test.test_id }}');
            {% endif %}
            {% endfor %}
        });
    </script>
</body>
</html>
"""


class HTMLReporter(Reporter):
    """Reporter that outputs results as a rich HTML report.

    Produces a single-file HTML report with:
    - Summary section with key metrics
    - Test results pie chart
    - Score distribution bar chart (using Chart.js)
    - Collapsible test details accordion
    - Score breakdown visualization
    - Failed checks highlighting
    - Trace viewer (collapsible)
    """

    def __init__(
        self,
        output_file: Path | str | None = None,
        output: TextIO | None = None,
        title: str = "ATP Test Results",
        include_trace: bool = True,
        auto_expand_failed: bool = True,
    ) -> None:
        """Initialize the HTML reporter.

        Args:
            output_file: Path to write HTML file (takes precedence over output).
            output: Output stream (defaults to stdout if no file specified).
            title: Report title.
            include_trace: Whether to include trace events.
            auto_expand_failed: Whether to auto-expand failed test details.
        """
        self._output_file = Path(output_file) if output_file else None
        self._output = output
        self._title = title
        self._include_trace = include_trace
        self._auto_expand_failed = auto_expand_failed
        self._env = Environment(loader=BaseLoader(), autoescape=True)

    @property
    def name(self) -> str:
        """Return the reporter name."""
        return "html"

    def report(self, report: SuiteReport) -> None:
        """Generate and output the HTML report.

        Args:
            report: Suite report data to output.
        """
        html_content = self._render_html(report)

        if self._output_file:
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
            self._output_file.write_text(html_content)
        elif self._output:
            self._output.write(html_content)
        else:
            import sys

            sys.stdout.write(html_content)

    def _render_html(self, report: SuiteReport) -> str:
        """Render the HTML report from template.

        Args:
            report: Suite report data.

        Returns:
            Rendered HTML string.
        """
        template = self._env.from_string(HTML_TEMPLATE)

        # Check if any tests have scores
        has_scores = any(t.score is not None for t in report.tests)

        # Prepare chart data
        test_names = [self._truncate_name(t.test_name, 30) for t in report.tests]
        test_scores = [t.score if t.score is not None else 0 for t in report.tests]
        test_colors = [self._get_score_color(s) for s in test_scores]

        return template.render(
            title=self._title,
            report=report,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            has_scores=has_scores,
            test_names=test_names,
            test_scores=test_scores,
            test_colors=test_colors,
            chartjs_cdn=CHARTJS_CDN,
            format_duration=self._format_duration,
            score_class=self._get_score_class,
        )

    def _truncate_name(self, name: str, max_length: int) -> str:
        """Truncate a name to max length with ellipsis.

        Args:
            name: Name to truncate.
            max_length: Maximum length.

        Returns:
            Truncated name.
        """
        if len(name) <= max_length:
            return name
        return name[: max_length - 3] + "..."

    def _get_score_color(self, score: float) -> str:
        """Get chart color based on score.

        Args:
            score: Score value (0-100).

        Returns:
            Hex color string.
        """
        if score >= 80:
            return "#22c55e"  # Green
        elif score >= 50:
            return "#f59e0b"  # Yellow/amber
        else:
            return "#ef4444"  # Red

    def _get_score_class(self, score: float | None) -> str:
        """Get CSS class based on score.

        Args:
            score: Score value (0-100).

        Returns:
            CSS class name.
        """
        if score is None:
            return ""
        if score >= 80:
            return "high"
        elif score >= 50:
            return "medium"
        else:
            return "low"

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Text to escape.

        Returns:
            Escaped text.
        """
        return html.escape(text)
