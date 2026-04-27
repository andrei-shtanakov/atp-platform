"""Application-level logging configuration for the ATP Dashboard.

Without this, only uvicorn's own loggers (``uvicorn``, ``uvicorn.access``)
reach stdout — every ``logger.info(...)`` from app modules is silently
dropped because the Python root logger defaults to ``WARNING``. The
gap was discovered while diagnosing the cold-start race on tournament
33 (2026-04-27): the structured ``mcp_handshake_*`` events shipped in
PR #101 (Task 4 of the MCP reliability plan) were emitted but never
appeared in container logs, making the new observability invisible.

This module sets ``INFO`` on the ``atp`` logger hierarchy and attaches
a single ``StreamHandler`` whose formatter appends every non-stdlib
``LogRecord`` attribute as a ``key=value`` pair — so the structured
``extra={...}`` fields passed to ``logger.info(msg, extra=...)`` are
visible in plain log output, not just in JSON aggregators.

Idempotent: ``configure_app_logging()`` only attaches a handler if
none exists yet on the ``atp`` logger, so repeated calls (e.g. from
test imports of ``factory``) do not duplicate output.
"""

from __future__ import annotations

import logging

# These attributes are set by the standard library on every
# ``LogRecord``. We exclude them from the "extras" rendering so the
# formatter only shows what callers explicitly passed via ``extra=``.
# Sourced from CPython 3.12 ``logging.LogRecord.__init__`` plus the
# attributes set during formatting (``message``, ``asctime``).
_STANDARD_LOG_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",  # Python 3.12+
        "thread",
        "threadName",
    }
)


class ExtrasFormatter(logging.Formatter):
    """Logging formatter that appends ``extra`` fields after the message.

    Output shape (with the default base format below) — wrapped here
    for line length, but on stdout it is one line::

        2026-04-27 16:32:11 INFO atp.mcp.observability
          mcp_handshake_started request_id='abc...'
          user_agent='pytest/1.0' duration_ms=0.41

    The base format is fixed via the constructor; the ``extra`` rendering
    is unconditional and uses ``repr()`` so strings stay quoted and
    nested structures stay disambiguable in plain text. Operators can
    grep by either field name or event name; downstream JSON
    aggregators can be added later without breaking this format.
    """

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_LOG_RECORD_ATTRS and not k.startswith("_")
        }
        if not extras:
            return base
        rendered = " ".join(f"{k}={v!r}" for k, v in extras.items())
        return f"{base} {rendered}"


def configure_app_logging(level: int = logging.INFO) -> None:
    """Configure the ``atp`` logger hierarchy for stdout-visible output.

    Idempotent: subsequent calls do not stack handlers. Tests that
    import ``factory`` to call ``create_app()`` therefore do not see
    duplicated log output.

    Does **not** touch the root logger or uvicorn's loggers — those
    keep their existing config. We only own the ``atp`` namespace.
    """
    atp_logger = logging.getLogger("atp")
    atp_logger.setLevel(level)

    # Attach our handler exactly once. Detecting "is our handler
    # already there" by type+formatter would be overkill; checking
    # for any ExtrasFormatter handler suffices because nothing else
    # in this codebase installs one.
    already_attached = any(
        isinstance(h.formatter, ExtrasFormatter) for h in atp_logger.handlers
    )
    if already_attached:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(ExtrasFormatter())
    atp_logger.addHandler(handler)
    # ``propagate=True`` (default) is intentional. Uvicorn's default
    # ``LOGGING_CONFIG`` does NOT attach a handler to the root logger —
    # only to ``uvicorn``, ``uvicorn.error``, ``uvicorn.access`` —
    # so records bubbling up from ``atp.*`` print exactly once via
    # our handler. Pytest's ``caplog`` fixture relies on propagation
    # to root to capture records; switching it off would silently
    # break every test that asserts on ``caplog.records``.
