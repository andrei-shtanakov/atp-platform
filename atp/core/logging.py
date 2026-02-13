"""Structured logging implementation for ATP Platform.

This module provides structured logging with:
- Correlation ID generation and propagation
- JSON output for production, pretty output for development
- Context processors for common fields (version, hostname)
- Log level configuration per module
- Sensitive data redaction
- Integration with Python's standard logging

Example usage:
    from atp.core.logging import configure_logging, get_logger, correlation_context

    # Configure logging at startup
    configure_logging()

    # Get a structured logger
    logger = get_logger(__name__)

    # Log with context
    logger.info("processing_request", user_id="123", action="create")

    # Use correlation ID context
    async with correlation_context("request-123"):
        logger.info("handling_request")  # Automatically includes correlation_id
"""

import logging
import socket
import sys
import uuid
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

import structlog
from structlog.types import EventDict, Processor, WrappedLogger

from atp.core.security import redact_secrets, sanitize_log_message

# Version of ATP platform - imported dynamically to avoid circular imports
ATP_VERSION = "1.0.0"

# Context variable for correlation ID propagation across async operations
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Context variable for additional bound context
_bound_context: ContextVar[dict[str, Any]] = ContextVar("bound_context", default={})

# Module-level log level overrides
_module_log_levels: dict[str, int] = {}

# Patterns for sensitive keys to redact in structured logs
SENSITIVE_KEY_PATTERNS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "credential",
        "private_key",
        "access_key",
        "secret_key",
    }
)


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracing.

    Returns:
        A unique string identifier (UUID4 format).
    """
    return str(uuid.uuid4())


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context.

    Returns:
        The current correlation ID or None if not set.
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None) -> None:
    """Set the correlation ID in the current context.

    Args:
        correlation_id: The correlation ID to set, or None to clear.
    """
    _correlation_id.set(correlation_id)


class correlation_context:
    """Context manager for correlation ID propagation.

    Use this to establish a correlation ID scope for a request or operation.

    Example:
        async with correlation_context("req-123"):
            logger.info("processing")  # Includes correlation_id="req-123"

        # Or generate a new ID automatically:
        async with correlation_context():
            logger.info("processing")  # Includes auto-generated correlation_id
    """

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize the correlation context.

        Args:
            correlation_id: Optional correlation ID. If None, generates a new one.
        """
        self.correlation_id = correlation_id or generate_correlation_id()
        self._token: Any = None

    def __enter__(self) -> str:
        """Enter the context and set the correlation ID."""
        self._token = _correlation_id.set(self.correlation_id)
        return self.correlation_id

    def __exit__(self, *args: Any) -> None:
        """Exit the context and restore the previous correlation ID."""
        _correlation_id.reset(self._token)

    async def __aenter__(self) -> str:
        """Async enter the context and set the correlation ID."""
        return self.__enter__()

    async def __aexit__(self, *args: Any) -> None:
        """Async exit the context and restore the previous correlation ID."""
        self.__exit__(*args)


class bind_context:
    """Context manager to bind additional context to logs.

    Example:
        with bind_context(user_id="123", tenant="acme"):
            logger.info("action")  # Includes user_id and tenant
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the context binder.

        Args:
            **kwargs: Context fields to bind to logs.
        """
        self.ctx = kwargs
        self._token: Any = None
        self._old_context: dict[str, Any] = {}

    def __enter__(self) -> "bind_context":
        """Enter the context and bind the fields."""
        self._old_context = _bound_context.get().copy()
        new_context = {**self._old_context, **self.ctx}
        self._token = _bound_context.set(new_context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context and restore previous bindings."""
        _bound_context.reset(self._token)


# =============================================================================
# Structlog Processors
# =============================================================================


def add_correlation_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add correlation ID to log events if available."""
    correlation_id = get_correlation_id()
    if correlation_id is not None:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_bound_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add bound context variables to log events."""
    ctx = _bound_context.get()
    if ctx:
        for key, value in ctx.items():
            if key not in event_dict:
                event_dict[key] = value
    return event_dict


@lru_cache(maxsize=1)
def _get_hostname() -> str:
    """Get the hostname (cached)."""
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def add_common_fields(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add common fields like version, hostname, and environment."""
    event_dict.setdefault("atp_version", ATP_VERSION)
    event_dict.setdefault("hostname", _get_hostname())
    return event_dict


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name indicates sensitive data."""
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in SENSITIVE_KEY_PATTERNS)


def redact_sensitive_data(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Redact sensitive data from log events.

    This processor:
    1. Redacts values of keys that match sensitive patterns
    2. Applies text-based secret detection to string values
    """
    redacted_dict: EventDict = {}

    for key, value in event_dict.items():
        if _is_sensitive_key(key):
            # Redact the entire value for sensitive keys
            redacted_dict[key] = "[REDACTED]"
        elif isinstance(value, str):
            # Apply secret pattern detection to string values
            redacted_dict[key] = redact_secrets(value)
        elif isinstance(value, dict):
            # Recursively handle nested dicts
            redacted_dict[key] = _redact_dict(value)
        else:
            redacted_dict[key] = value

    return redacted_dict


def _redact_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact sensitive data from a dictionary."""
    result: dict[str, Any] = {}
    for key, value in d.items():
        if _is_sensitive_key(key):
            result[key] = "[REDACTED]"
        elif isinstance(value, str):
            result[key] = redact_secrets(value)
        elif isinstance(value, dict):
            result[key] = _redact_dict(value)
        else:
            result[key] = value
    return result


def sanitize_event(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Sanitize log messages to prevent log injection."""
    event = event_dict.get("event", "")
    if isinstance(event, str):
        event_dict["event"] = sanitize_log_message(event)
    return event_dict


def filter_by_module_level(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Filter log events based on module-specific log levels."""
    if not _module_log_levels:
        return event_dict

    logger_name = event_dict.get("logger", "")
    if not logger_name:
        return event_dict

    # Find the most specific module level that matches
    level_threshold = None
    matched_prefix = ""

    for module, level in _module_log_levels.items():
        if logger_name == module or logger_name.startswith(f"{module}."):
            if len(module) > len(matched_prefix):
                level_threshold = level
                matched_prefix = module

    if level_threshold is not None:
        # Get the numeric level for the current log method
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "warn": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
            "exception": logging.ERROR,
        }
        current_level = level_map.get(method_name.lower(), logging.INFO)

        if current_level < level_threshold:
            raise structlog.DropEvent

    return event_dict


# =============================================================================
# Logging Configuration
# =============================================================================


def set_module_log_level(module: str, level: int | str) -> None:
    """Set the log level for a specific module.

    Args:
        module: Module name (e.g., "atp.runner", "atp.evaluators.llm_judge")
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int
    """
    if isinstance(level, str):
        numeric_level: int = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level
    _module_log_levels[module] = numeric_level


def get_module_log_level(module: str) -> int | None:
    """Get the log level for a specific module.

    Args:
        module: Module name

    Returns:
        The log level if set, None otherwise
    """
    return _module_log_levels.get(module)


def clear_module_log_levels() -> None:
    """Clear all module-specific log level overrides."""
    _module_log_levels.clear()


def _get_processors(json_output: bool = False) -> list[Processor]:
    """Get the list of structlog processors.

    Args:
        json_output: Whether to use JSON output format

    Returns:
        List of processors for structlog configuration
    """
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_correlation_id,
        add_bound_context,
        add_common_fields,
        filter_by_module_level,
        redact_sensitive_data,
        sanitize_event,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # For production: JSON output
        shared_processors.append(
            structlog.processors.format_exc_info,
        )
    else:
        # For development: colored console output
        shared_processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    return shared_processors


class StructlogHandler(logging.Handler):
    """Logging handler that routes stdlib logging to structlog."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record through structlog."""
        try:
            # Get the structlog logger
            sl_logger = structlog.get_logger(record.name)

            # Get the log level method
            level_name = record.levelname.lower()
            if level_name == "warning":
                level_name = "warn"

            log_method = getattr(sl_logger, level_name, None)
            if log_method is None:
                log_method = sl_logger.info

            # Extract extra fields from the record
            extra: dict[str, Any] = {}
            for key, value in record.__dict__.items():
                if key not in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                    "taskName",
                }:
                    extra[key] = value

            # Format the message
            msg = record.getMessage()

            # Include exception info if present
            if record.exc_info:
                extra["exc_info"] = record.exc_info

            # Log through structlog
            log_method(msg, **extra)

        except Exception:
            # Fallback to basic logging if structlog fails
            self.handleError(record)


def configure_logging(
    level: str | int = logging.INFO,
    json_output: bool | None = None,
    log_file: str | None = None,
    module_levels: dict[str, str | int] | None = None,
) -> None:
    """Configure structured logging for the application.

    This function sets up structlog with the appropriate processors and
    integrates with Python's standard logging.

    Args:
        level: Global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON output format. If None, auto-detects:
                     True if not a TTY (production), False otherwise (dev)
        log_file: Optional file path for log output
        module_levels: Dict of module name to log level for per-module config
    """
    # Determine output format
    if json_output is None:
        json_output = not sys.stdout.isatty()

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Set up module-specific log levels
    if module_levels:
        for module, mod_level in module_levels.items():
            set_module_log_level(module, mod_level)

    # Build the shared processors (without the final renderer)
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_correlation_id,
        add_bound_context,
        add_common_fields,
        filter_by_module_level,
        redact_sensitive_data,
        sanitize_event,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.processors.format_exc_info,
    ]

    # Configure structlog to use ProcessorFormatter for stdlib integration
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Create formatter for stdlib handlers
    if json_output:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared_processors,
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def configure_logging_from_settings() -> None:
    """Configure logging from ATP settings.

    This is a convenience function that reads logging configuration
    from ATPSettings and applies it.
    """
    # Import here to avoid circular imports
    from atp.core.settings import get_cached_settings

    settings = get_cached_settings()

    configure_logging(
        level=settings.logging.level,
        json_output=settings.logging.json_output,
        log_file=settings.logging.file,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__). If None, returns the root logger.

    Returns:
        A structured logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("request_received", method="GET", path="/api/test")
    """
    return structlog.get_logger(name)


# =============================================================================
# Convenience Functions
# =============================================================================


def log_with_context(
    logger: structlog.stdlib.BoundLogger,
    level: str,
    event: str,
    correlation_id: str | None = None,
    **kwargs: Any,
) -> None:
    """Log an event with optional correlation ID.

    Args:
        logger: The structlog logger instance
        level: Log level (debug, info, warning, error, critical)
        event: Event name/message
        correlation_id: Optional correlation ID to include
        **kwargs: Additional context fields
    """
    if correlation_id:
        kwargs["correlation_id"] = correlation_id

    log_method = getattr(logger, level, logger.info)
    log_method(event, **kwargs)


def create_test_logger(
    name: str = "test", json_output: bool = False
) -> structlog.stdlib.BoundLogger:
    """Create a logger configured for testing.

    This creates an isolated logger that doesn't affect the global
    logging configuration, useful for unit tests.

    Args:
        name: Logger name
        json_output: Whether to use JSON output

    Returns:
        Configured test logger
    """
    processors = _get_processors(json_output=json_output)

    # Create an isolated configuration
    return structlog.wrap_logger(
        logging.getLogger(name),
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
    )


# =============================================================================
# Reset function for testing
# =============================================================================


def reset_logging() -> None:
    """Reset logging configuration to defaults.

    This is primarily useful for testing to ensure clean state between tests.
    """
    clear_module_log_levels()
    _correlation_id.set(None)

    # Try to reset bound context to empty dict
    try:
        _bound_context.set({})
    except LookupError:
        pass

    # Reset structlog configuration
    structlog.reset_defaults()

    # Reset root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
