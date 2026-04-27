"""Tests for the application-level logging configuration.

Pins the contract that ``configure_app_logging()`` is idempotent and
that ``ExtrasFormatter`` renders ``logger.info(msg, extra={...})``
fields as ``key='value'`` pairs after the message — without that, the
structured events shipped in PR #101 (Task 4 of the MCP reliability
plan) are silently dropped by uvicorn's default config.
"""

from __future__ import annotations

import logging

import pytest

from atp.dashboard.v2.logging_config import (
    ExtrasFormatter,
    configure_app_logging,
)


@pytest.fixture(autouse=True)
def _reset_atp_logger() -> None:
    """Strip any handlers we attach so each test starts from a clean
    state. Without this the autouse-fixture-style idempotency check
    passes only on the first test and silently regresses afterwards."""
    atp_logger = logging.getLogger("atp")
    original_level = atp_logger.level
    original_propagate = atp_logger.propagate
    original_handlers = list(atp_logger.handlers)
    atp_logger.handlers.clear()
    try:
        yield
    finally:
        atp_logger.handlers.clear()
        atp_logger.handlers.extend(original_handlers)
        atp_logger.setLevel(original_level)
        atp_logger.propagate = original_propagate


# ---------------------------------------------------------------------------
# ExtrasFormatter
# ---------------------------------------------------------------------------


def _build_record(message: str, **extras: object) -> logging.LogRecord:
    """Construct a record exactly the way ``logger.info(msg, extra=...)``
    would. ``LogRecord`` does not normally accept arbitrary kwargs, so
    we set them on ``__dict__`` after construction — same path the
    standard ``logging.Logger._log`` takes for ``extra``."""
    record = logging.LogRecord(
        name="atp.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg=message,
        args=(),
        exc_info=None,
    )
    for k, v in extras.items():
        setattr(record, k, v)
    return record


def test_extras_formatter_emits_message_alone_when_no_extras() -> None:
    """A standard ``logger.info(msg)`` call (no ``extra=``) must format
    cleanly without a trailing space or fabricated key=value pairs."""
    formatter = ExtrasFormatter()
    record = _build_record("plain message")

    output = formatter.format(record)

    assert output.endswith("atp.test plain message")
    assert "=" not in output.split(" plain message", 1)[1]


def test_extras_formatter_appends_extras_as_key_repr_pairs() -> None:
    """Structured ``extra={...}`` fields must render after the message
    using ``repr()`` so strings stay quoted and operators can grep
    on either field name or full token."""
    formatter = ExtrasFormatter()
    record = _build_record(
        "mcp_handshake_started",
        request_id="abcd1234",
        client_ip="10.0.0.1",
        duration_ms=0.42,
    )

    output = formatter.format(record)

    assert "mcp_handshake_started" in output
    assert "request_id='abcd1234'" in output
    assert "client_ip='10.0.0.1'" in output
    assert "duration_ms=0.42" in output


def test_extras_formatter_skips_underscore_prefixed_attrs() -> None:
    """LogRecord internals (``_*``) and pytest's tweaks must not leak
    into the rendered tail — only explicitly-passed ``extra`` keys do."""
    formatter = ExtrasFormatter()
    record = _build_record("evt", request_id="rid", _internal="should-be-hidden")

    output = formatter.format(record)

    assert "request_id='rid'" in output
    assert "_internal" not in output


# ---------------------------------------------------------------------------
# configure_app_logging
# ---------------------------------------------------------------------------


def test_configure_attaches_handler_with_extras_formatter() -> None:
    configure_app_logging()
    atp_logger = logging.getLogger("atp")

    handlers_with_extras = [
        h for h in atp_logger.handlers if isinstance(h.formatter, ExtrasFormatter)
    ]
    assert len(handlers_with_extras) == 1
    assert atp_logger.level == logging.INFO


def test_configure_is_idempotent() -> None:
    """Repeated calls must not stack handlers — tests that ``import``
    factory will trigger ``create_app`` setup multiple times across a
    suite, and double-printed log lines would surface as flaky test
    output."""
    configure_app_logging()
    configure_app_logging()
    configure_app_logging()

    atp_logger = logging.getLogger("atp")
    handlers_with_extras = [
        h for h in atp_logger.handlers if isinstance(h.formatter, ExtrasFormatter)
    ]
    assert len(handlers_with_extras) == 1


def test_info_records_from_child_logger_pass_through(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end smoke: after configure, an ``INFO`` record from
    ``atp.mcp.observability`` (the Task 4 logger) reaches caplog,
    confirming that ``propagate=True`` was preserved and the level
    was lifted from the default WARNING."""
    configure_app_logging()
    caplog.set_level(logging.INFO, logger="atp")

    logging.getLogger("atp.mcp.observability").info(
        "mcp_handshake_started",
        extra={"request_id": "rid", "client_ip": "10.0.0.1"},
    )

    matching = [r for r in caplog.records if r.message == "mcp_handshake_started"]
    assert len(matching) == 1
    assert matching[0].request_id == "rid"  # type: ignore[attr-defined]
