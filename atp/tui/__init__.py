"""ATP TUI (Terminal User Interface) package.

This package provides a terminal-based user interface for ATP,
built with Textual framework.

Usage:
    # Start the TUI
    atp tui

    # Or programmatically
    from atp.tui import ATPTUI
    app = ATPTUI()
    app.run()

Requirements:
    This package requires optional dependencies. Install with:
    uv add atp-platform[tui]
"""

from atp.tui.app import ATPTUI, run_tui

__all__ = ["ATPTUI", "run_tui"]
