"""Tests that the CLI entry point configures structured logging."""

import logging

import pytest
import structlog
from click.testing import CliRunner

from atp.cli.main import cli


@pytest.fixture(autouse=True)
def restore_root_logger():
    """Save and restore root logger handlers/level around each test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_cli_installs_structlog_formatter() -> None:
    result = CliRunner().invoke(cli, ["version"])
    assert result.exit_code == 0
    root = logging.getLogger()
    assert any(
        isinstance(h.formatter, structlog.stdlib.ProcessorFormatter)
        for h in root.handlers
    ), "cli() must route stdlib logging through structlog ProcessorFormatter"


def test_cli_verbose_sets_debug_level() -> None:
    result = CliRunner().invoke(cli, ["--verbose", "version"])
    assert result.exit_code == 0
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_console_handler_uses_stderr() -> None:
    """stdout is reserved for machine output (--output=json); logs go to stderr."""
    import sys

    from atp.core.logging import configure_logging

    configure_logging(level="INFO")
    root = logging.getLogger()
    streams = [h.stream for h in root.handlers if isinstance(h, logging.StreamHandler)]
    assert sys.stderr in streams
    assert sys.stdout not in streams
