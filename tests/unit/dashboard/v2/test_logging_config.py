"""Tests for the application-level logging configuration.

Pins the contract that ``configure_app_logging()`` is idempotent and
that ``ExtrasFormatter`` renders ``logger.info(msg, extra={...})``
fields as ``key='value'`` pairs after the message — without that, the
structured events shipped in PR #101 (Task 4 of the MCP reliability
plan) are silently dropped by uvicorn's default config.
"""

from __future__ import annotations

import logging
import sys

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


def test_extras_formatter_does_not_duplicate_event_field() -> None:
    """``emit_event`` passes ``extra={"event": <name>, ...}`` so the
    name is available on the record for filtering. The formatter must
    not render it as a tail key=value because the same string is
    already the message body — without this skip, every line shows
    the event name twice (Copilot review on PR #102)."""
    formatter = ExtrasFormatter()
    record = _build_record(
        "mcp_handshake_started",
        event="mcp_handshake_started",
        request_id="rid",
    )

    output = formatter.format(record)

    # Message body present, request_id rendered, no second copy of the
    # event name as ``event='...'``.
    assert "mcp_handshake_started" in output
    assert "request_id='rid'" in output
    assert "event=" not in output


def test_extras_formatter_strips_default_millisecond_asctime() -> None:
    """Default ``%(asctime)s`` includes ``,mmm`` ms suffix; we use an
    explicit ``datefmt`` so log lines stay grep-friendly. Pin that —
    tooling assumes the documented shape."""
    formatter = ExtrasFormatter()
    record = _build_record("evt")

    output = formatter.format(record)

    timestamp = output.split(" ", 1)[0]
    # ``YYYY-MM-DD`` then a space, then ``HH:MM:SS`` — no comma, no
    # millisecond suffix.
    assert "," not in timestamp + output.split(" ", 2)[1]


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
    # Pin the stdout target — ``StreamHandler()`` with no args defaults
    # to stderr, which gets split from stdout in some log pipelines.
    # Co-locate with uvicorn's access logs (also stdout).
    assert handlers_with_extras[0].stream is sys.stdout


def test_configure_disables_propagation_in_production() -> None:
    """``atp.propagate`` must be False in production so records do
    not bubble to a handler attached to root by some other module —
    this caused the duplicated log lines on prod (tournament 35,
    2026-04-28). If a future refactor flips this back to True
    unconditionally, double-print returns silently. Pin the
    invariant by simulating a non-pytest environment.

    Under pytest, ``configure_app_logging`` deliberately keeps
    ``propagate=True`` so the ``caplog`` fixture (which captures
    via a handler on root) continues to work; that branch is
    covered by ``test_configure_keeps_propagation_under_pytest``
    below."""
    # Temporarily remove ``pytest`` from sys.modules so the helper's
    # detection branch matches production. Restore in finally so the
    # rest of the suite is unaffected.
    pytest_mod = sys.modules.pop("pytest", None)
    try:
        configure_app_logging()
        atp_logger = logging.getLogger("atp")
        assert atp_logger.propagate is False, (
            "atp.propagate must be False in production to prevent "
            "double-print via root handlers; see tournament-35 prod "
            "log incident."
        )
    finally:
        if pytest_mod is not None:
            sys.modules["pytest"] = pytest_mod


def test_configure_keeps_propagation_under_pytest() -> None:
    """When pytest is loaded (every test run), propagation stays on
    so caplog can capture via root. The complementary production
    branch is pinned in the test above."""
    assert "pytest" in sys.modules, "this test only meaningful under pytest"

    configure_app_logging()
    atp_logger = logging.getLogger("atp")
    assert atp_logger.propagate is True


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
    """Under pytest, propagation stays on (so caplog works), level
    is lifted from WARNING to INFO, and an ``atp.mcp.observability``
    record reaches caplog with its ``extra`` fields preserved on
    the LogRecord. The production path (propagate=False, no
    propagation to caplog) is covered by
    ``test_configure_disables_propagation_in_production``."""
    configure_app_logging()
    caplog.set_level(logging.INFO, logger="atp")

    logging.getLogger("atp.mcp.observability").info(
        "mcp_handshake_started",
        extra={"request_id": "rid", "client_ip": "10.0.0.1"},
    )

    matching = [r for r in caplog.records if r.message == "mcp_handshake_started"]
    assert len(matching) == 1
    assert matching[0].request_id == "rid"  # type: ignore[attr-defined]
