"""Base reporter interface and result models."""

from abc import ABC, abstractmethod

from atp.core.results import (  # noqa: F401 — re-export
    SuiteReport,
    TestReport,
    rebuild_report_models,
)

# Ensure models are fully defined when importing from reporters.base
# (scoring/statistics are always available in the full platform)
rebuild_report_models()


class Reporter(ABC):
    """Base class for reporters.

    Reporters format and output test results in various formats.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the reporter name."""

    @abstractmethod
    def report(self, report: SuiteReport) -> None:
        """Generate and output the report.

        Args:
            report: Suite report data to output.
        """

    @property
    def supports_streaming(self) -> bool:
        """Return whether this reporter supports streaming output.

        Default is False. Override in subclasses that support streaming.
        """
        return False

    def _format_duration(self, seconds: float | None) -> str:
        """Format duration in human-readable format.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted duration string.
        """
        if seconds is None:
            return "N/A"

        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _format_score(self, score: float | None) -> str:
        """Format score for display.

        Args:
            score: Score value (0-100).

        Returns:
            Formatted score string.
        """
        if score is None:
            return "N/A"
        return f"{score:.1f}/100"
