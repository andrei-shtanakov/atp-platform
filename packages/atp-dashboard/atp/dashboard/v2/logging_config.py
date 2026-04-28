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
import sys

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

# ``event`` is callers' convention for tagging structured emits (see
# ``atp.dashboard.mcp.observability``). It always equals ``record.msg``
# at emit time, so rendering it as a tail key=value pair would
# duplicate the event name on every line. Keep it on the record so
# tests / filters can read ``record.event`` directly, but skip it in
# the formatted output.
_EXTRAS_RENDER_SKIP = frozenset({"event"})


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

    The reserved attribute ``event`` (set by ``observability.emit_event``)
    is NOT rendered — it always equals ``record.msg`` and would
    otherwise duplicate the event name on every line. It stays
    available on the record for tests / filters via ``record.event``.
    """

    def __init__(self) -> None:
        # Explicit ``datefmt`` strips the default millisecond suffix
        # (``,mmm``) so log lines stay grep-friendly and match the
        # examples in the runbook. Sub-second granularity is available
        # via the ``duration_ms`` field on observability events when
        # actually needed.
        super().__init__(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_LOG_RECORD_ATTRS
            and k not in _EXTRAS_RENDER_SKIP
            and not k.startswith("_")
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

    Sets ``atp.propagate = False`` in production to defend against
    doubled output: PR #102 originally kept propagation on for
    caplog compatibility, but tournament-35 prod logs (2026-04-28)
    showed every event printed twice — once via our
    ``ExtrasFormatter`` handler, once via a default-format handler
    somewhere upstream. The exact upstream handler is not yet
    identified (a one-shot startup dump below captures the live
    process state for follow-up). Cutting propagation eliminates
    the double-print regardless of cause.

    Under pytest, propagation stays on so the ``caplog`` fixture
    (which captures via a handler on root) keeps working without
    forcing every test to attach a handler manually. Detection
    uses ``sys.modules`` because ``factory`` module-imports
    ``create_app`` at import time — by the moment any test runs,
    ``pytest`` is already loaded, so the check is reliable.
    """
    atp_logger = logging.getLogger("atp")
    atp_logger.setLevel(level)
    atp_logger.propagate = "pytest" in sys.modules

    # Attach our handler exactly once. Detecting "is our handler
    # already there" by type+formatter would be overkill; checking
    # for any ExtrasFormatter handler suffices because nothing else
    # in this codebase installs one.
    already_attached = any(
        isinstance(h.formatter, ExtrasFormatter) for h in atp_logger.handlers
    )
    if already_attached:
        return

    # Explicit stdout stream: ``StreamHandler()`` defaults to stderr,
    # which docker captures but separates from stdout in some pipelines.
    # Uvicorn's access logs already go to stdout; co-locate ours.
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(ExtrasFormatter())
    atp_logger.addHandler(handler)

    _dump_logger_state_once(atp_logger)


def _dump_logger_state_once(atp_logger: logging.Logger) -> None:
    """One-shot startup diagnostic — dump every logger that has at
    least one handler attached, plus the propagation flag, to stdout.

    Captures the live uvicorn-process logger state at the precise
    moment ``configure_app_logging`` finishes setup. Runs once per
    process (gated by the function's own first-call check). Output
    is grep-friendly so on-call can scan ``docker compose logs
    platform | grep LOGGER_DUMP`` after a deploy.

    Lives in code (not env-gated) because (a) it's a single block of
    output per process — negligible cost — and (b) it stays useful
    across future logging changes when a similar question recurs.
    Drop or switch to env-gated mode in a follow-up PR once the
    upstream handler attaching to root has been identified and
    eliminated.
    """
    if getattr(_dump_logger_state_once, "_done", False):
        return
    _dump_logger_state_once._done = True  # type: ignore[attr-defined]

    atp_logger.info(
        "LOGGER_DUMP root_handlers=%r root_level=%d atp_handlers=%r atp_propagate=%s",
        logging.getLogger().handlers,
        logging.getLogger().level,
        atp_logger.handlers,
        atp_logger.propagate,
    )
    for name, lg in sorted(logging.Logger.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger) and lg.handlers:
            atp_logger.info(
                "LOGGER_DUMP name=%s handlers=%r propagate=%s",
                name,
                lg.handlers,
                lg.propagate,
            )
