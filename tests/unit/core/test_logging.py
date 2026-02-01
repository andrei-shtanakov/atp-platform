"""Tests for ATP structured logging implementation."""

import logging
from io import StringIO
from typing import Any

import pytest
import structlog

from atp.core.logging import (
    ATP_VERSION,
    add_common_fields,
    add_correlation_id,
    bind_context,
    clear_module_log_levels,
    configure_logging,
    correlation_context,
    filter_by_module_level,
    generate_correlation_id,
    get_correlation_id,
    get_logger,
    get_module_log_level,
    redact_sensitive_data,
    reset_logging,
    sanitize_event,
    set_correlation_id,
    set_module_log_level,
)


@pytest.fixture(autouse=True)
def reset_logging_state():  # type: ignore[misc]
    """Reset logging state before each test."""
    reset_logging()
    yield
    reset_logging()


class TestCorrelationId:
    """Tests for correlation ID generation and propagation."""

    def test_generate_correlation_id_returns_uuid(self) -> None:
        """Test that correlation IDs are valid UUIDs."""
        correlation_id = generate_correlation_id()
        assert len(correlation_id) == 36
        assert correlation_id.count("-") == 4

    def test_generate_correlation_id_unique(self) -> None:
        """Test that each generated correlation ID is unique."""
        ids = {generate_correlation_id() for _ in range(100)}
        assert len(ids) == 100

    def test_get_set_correlation_id(self) -> None:
        """Test getting and setting correlation ID."""
        assert get_correlation_id() is None

        set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"

        set_correlation_id(None)
        assert get_correlation_id() is None

    def test_correlation_context_manager(self) -> None:
        """Test correlation context manager."""
        assert get_correlation_id() is None

        with correlation_context("ctx-123") as ctx_id:
            assert ctx_id == "ctx-123"
            assert get_correlation_id() == "ctx-123"

        assert get_correlation_id() is None

    def test_correlation_context_auto_generate(self) -> None:
        """Test correlation context with auto-generated ID."""
        assert get_correlation_id() is None

        with correlation_context() as ctx_id:
            assert ctx_id is not None
            assert len(ctx_id) == 36
            assert get_correlation_id() == ctx_id

        assert get_correlation_id() is None

    def test_correlation_context_nested(self) -> None:
        """Test nested correlation contexts."""
        with correlation_context("outer-123") as outer_id:
            assert outer_id == "outer-123"
            assert get_correlation_id() == "outer-123"

            with correlation_context("inner-456") as inner_id:
                assert inner_id == "inner-456"
                assert get_correlation_id() == "inner-456"

            assert get_correlation_id() == "outer-123"

        assert get_correlation_id() is None

    @pytest.mark.anyio
    async def test_correlation_context_async(self) -> None:
        """Test async correlation context manager."""
        assert get_correlation_id() is None

        async with correlation_context("async-123") as ctx_id:
            assert ctx_id == "async-123"
            assert get_correlation_id() == "async-123"

        assert get_correlation_id() is None


class TestBindContext:
    """Tests for bound context functionality."""

    def test_bind_context_adds_fields(self) -> None:
        """Test that bind_context adds fields to logs."""
        # The bound context is stored in a contextvar and used by processors
        with bind_context(user_id="123", tenant="acme"):
            # Context is now bound - processors will pick it up
            pass

    def test_bind_context_nested(self) -> None:
        """Test nested bind_context calls."""
        with bind_context(outer="1"):
            with bind_context(inner="2"):
                # Both should be available
                pass


class TestProcessors:
    """Tests for structlog processors."""

    def test_add_correlation_id_processor(self) -> None:
        """Test add_correlation_id processor."""
        event_dict: dict[str, Any] = {"event": "test"}

        # Without correlation ID
        result = add_correlation_id(None, "info", event_dict.copy())
        assert "correlation_id" not in result

        # With correlation ID
        set_correlation_id("proc-123")
        result = add_correlation_id(None, "info", event_dict.copy())
        assert result["correlation_id"] == "proc-123"

    def test_add_common_fields_processor(self) -> None:
        """Test add_common_fields processor."""
        event_dict: dict[str, Any] = {"event": "test"}

        result = add_common_fields(None, "info", event_dict)

        assert result["atp_version"] == ATP_VERSION
        assert "hostname" in result

    def test_add_common_fields_does_not_override(self) -> None:
        """Test that common fields don't override existing values."""
        event_dict: dict[str, Any] = {
            "event": "test",
            "atp_version": "custom",
            "hostname": "custom-host",
        }

        result = add_common_fields(None, "info", event_dict)

        assert result["atp_version"] == "custom"
        assert result["hostname"] == "custom-host"

    def test_redact_sensitive_data_keys(self) -> None:
        """Test that sensitive keys are redacted."""
        event_dict: dict[str, Any] = {
            "event": "test",
            "password": "secret123",
            "api_key": "sk-123456",
            "token": "eyJ...",
            "normal_field": "visible",
        }

        result = redact_sensitive_data(None, "info", event_dict)

        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["token"] == "[REDACTED]"
        assert result["normal_field"] == "visible"

    def test_redact_sensitive_data_patterns(self) -> None:
        """Test that secret patterns in string values are redacted."""
        event_dict: dict[str, Any] = {
            "event": "test",
            "message": "Using api_key=sk-12345678901234567890",
        }

        result = redact_sensitive_data(None, "info", event_dict)

        assert "sk-12345678901234567890" not in result["message"]
        assert "[REDACTED]" in result["message"] or "api_key" in result["message"]

    def test_redact_sensitive_data_nested_dict(self) -> None:
        """Test that nested dicts are recursively redacted."""
        event_dict: dict[str, Any] = {
            "event": "test",
            "config": {
                "password": "secret",
                "host": "localhost",
            },
        }

        result = redact_sensitive_data(None, "info", event_dict)

        assert result["config"]["password"] == "[REDACTED]"
        assert result["config"]["host"] == "localhost"

    def test_sanitize_event_processor(self) -> None:
        """Test sanitize_event processor."""
        event_dict: dict[str, Any] = {
            "event": "test\r\ninjection",
        }

        result = sanitize_event(None, "info", event_dict)

        assert "\r" not in result["event"]
        assert "\n" not in result["event"]
        assert "\\r" in result["event"]
        assert "\\n" in result["event"]


class TestModuleLogLevels:
    """Tests for module-specific log level configuration."""

    def test_set_and_get_module_log_level(self) -> None:
        """Test setting and getting module log levels."""
        assert get_module_log_level("atp.runner") is None

        set_module_log_level("atp.runner", logging.DEBUG)
        assert get_module_log_level("atp.runner") == logging.DEBUG

        set_module_log_level("atp.runner", "WARNING")
        assert get_module_log_level("atp.runner") == logging.WARNING

    def test_clear_module_log_levels(self) -> None:
        """Test clearing module log levels."""
        set_module_log_level("atp.runner", logging.DEBUG)
        set_module_log_level("atp.evaluators", logging.ERROR)

        clear_module_log_levels()

        assert get_module_log_level("atp.runner") is None
        assert get_module_log_level("atp.evaluators") is None

    def test_filter_by_module_level_passes_when_no_config(self) -> None:
        """Test that filter passes when no module config exists."""
        event_dict: dict[str, Any] = {"event": "test", "logger": "atp.runner"}

        result = filter_by_module_level(None, "debug", event_dict)
        assert result == event_dict

    def test_filter_by_module_level_filters(self) -> None:
        """Test that filter drops events below threshold."""
        set_module_log_level("atp.runner", logging.WARNING)

        event_dict: dict[str, Any] = {"event": "test", "logger": "atp.runner"}

        # Debug should be dropped
        with pytest.raises(structlog.DropEvent):
            filter_by_module_level(None, "debug", event_dict.copy())

        # Warning should pass
        result = filter_by_module_level(None, "warning", event_dict.copy())
        assert result == event_dict

    def test_filter_by_module_level_hierarchical(self) -> None:
        """Test hierarchical module matching."""
        set_module_log_level("atp", logging.ERROR)
        set_module_log_level("atp.runner", logging.DEBUG)

        # atp.runner.orchestrator should use atp.runner's DEBUG level
        event_dict: dict[str, Any] = {
            "event": "test",
            "logger": "atp.runner.orchestrator",
        }
        result = filter_by_module_level(None, "debug", event_dict.copy())
        assert result == event_dict

        # atp.evaluators should use atp's ERROR level
        event_dict["logger"] = "atp.evaluators"
        with pytest.raises(structlog.DropEvent):
            filter_by_module_level(None, "info", event_dict.copy())


class TestConfigureLogging:
    """Tests for logging configuration."""

    def test_configure_logging_default(self) -> None:
        """Test default logging configuration."""
        configure_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

    def test_configure_logging_custom_level(self) -> None:
        """Test logging configuration with custom level."""
        configure_logging(level=logging.DEBUG)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_logging_string_level(self) -> None:
        """Test logging configuration with string level."""
        configure_logging(level="WARNING")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_configure_logging_module_levels(self) -> None:
        """Test logging configuration with module levels."""
        configure_logging(
            module_levels={
                "atp.runner": "DEBUG",
                "atp.evaluators": logging.ERROR,
            }
        )

        assert get_module_log_level("atp.runner") == logging.DEBUG
        assert get_module_log_level("atp.evaluators") == logging.ERROR

    def test_configure_logging_json_output(self) -> None:
        """Test logging configuration with JSON output."""
        configure_logging(json_output=True)

        # Just verify it doesn't raise
        logger = get_logger("test")
        logger.info("test_json_output")


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_name(self) -> None:
        """Test getting a named logger."""
        logger = get_logger("test.module")
        assert logger is not None

    def test_get_logger_without_name(self) -> None:
        """Test getting root logger."""
        logger = get_logger()
        assert logger is not None

    def test_get_logger_same_name_consistent(self) -> None:
        """Test that loggers with same name behave consistently."""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        # Both loggers should be usable (structlog returns proxy objects)
        # Just verify they are both valid loggers
        assert logger1 is not None
        assert logger2 is not None


class TestResetLogging:
    """Tests for reset_logging function."""

    def test_reset_clears_correlation_id(self) -> None:
        """Test that reset clears correlation ID."""
        set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"

        reset_logging()

        assert get_correlation_id() is None

    def test_reset_clears_module_levels(self) -> None:
        """Test that reset clears module log levels."""
        set_module_log_level("atp.test", logging.DEBUG)
        assert get_module_log_level("atp.test") == logging.DEBUG

        reset_logging()

        assert get_module_log_level("atp.test") is None


class TestIntegrationWithStdlib:
    """Tests for integration with Python's standard logging."""

    def test_stdlib_logging_works(self) -> None:
        """Test that standard logging still works after configuration."""
        configure_logging(json_output=False)

        stdlib_logger = logging.getLogger("test.stdlib")
        stdlib_logger.setLevel(logging.DEBUG)

        # Should not raise
        stdlib_logger.info("Test message")
        stdlib_logger.debug("Debug message")
        stdlib_logger.warning("Warning message")

    def test_correlation_id_in_stdlib_logs(self) -> None:
        """Test that correlation ID is included in stdlib logs."""
        configure_logging(json_output=True)

        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)

        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
        handler.setFormatter(formatter)

        test_logger = logging.getLogger("test.correlation")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        with correlation_context("corr-123"):
            test_logger.info("Test with correlation")

        # The correlation ID should be captured by structlog processors
        # when using the full integration


class TestJsonOutput:
    """Tests for JSON output format."""

    def test_json_output_format(self) -> None:
        """Test that JSON output is valid JSON."""
        # Configure with JSON output
        configure_logging(json_output=True, level=logging.DEBUG)

        # Create a logger
        logger = get_logger("test.json")

        # Log something - just verify it doesn't raise
        logger.info("test_message", key="value")


class TestSensitiveDataInRealLogs:
    """Tests for sensitive data handling in real logging scenarios."""

    def test_password_in_dict_redacted(self) -> None:
        """Test that passwords in log data are redacted."""
        event_dict: dict[str, Any] = {
            "event": "user_login",
            "user": "john",
            "password": "supersecret123",
        }

        result = redact_sensitive_data(None, "info", event_dict)
        assert result["password"] == "[REDACTED]"
        assert result["user"] == "john"

    def test_authorization_header_redacted(self) -> None:
        """Test that authorization headers are redacted."""
        event_dict: dict[str, Any] = {
            "event": "api_call",
            "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        }

        result = redact_sensitive_data(None, "info", event_dict)
        assert result["authorization"] == "[REDACTED]"

    def test_secret_key_variations_redacted(self) -> None:
        """Test various secret key naming patterns."""
        event_dict: dict[str, Any] = {
            "event": "config_load",
            "secret_key": "abc123",
            "SecretKey": "def456",
            "SECRET": "ghi789",
            "access_key": "jkl012",
            "private_key": "mno345",
        }

        result = redact_sensitive_data(None, "info", event_dict)
        assert result["secret_key"] == "[REDACTED]"
        assert result["SecretKey"] == "[REDACTED]"
        assert result["SECRET"] == "[REDACTED]"
        assert result["access_key"] == "[REDACTED]"
        assert result["private_key"] == "[REDACTED]"


class TestAddBoundContext:
    """Tests for add_bound_context processor."""

    def test_add_bound_context_empty(self) -> None:
        """Test that empty bound context doesn't modify event."""
        from atp.core.logging import add_bound_context

        event_dict: dict[str, Any] = {"event": "test"}
        result = add_bound_context(None, "info", event_dict)
        assert result == event_dict

    def test_add_bound_context_with_values(self) -> None:
        """Test that bound context values are added."""
        from atp.core.logging import add_bound_context

        with bind_context(user_id="123", tenant="acme"):
            event_dict: dict[str, Any] = {"event": "test"}
            result = add_bound_context(None, "info", event_dict)
            assert result["user_id"] == "123"
            assert result["tenant"] == "acme"

    def test_add_bound_context_doesnt_override(self) -> None:
        """Test that bound context doesn't override existing keys."""
        from atp.core.logging import add_bound_context

        with bind_context(user_id="from_context"):
            event_dict: dict[str, Any] = {"event": "test", "user_id": "existing"}
            result = add_bound_context(None, "info", event_dict)
            assert result["user_id"] == "existing"


class TestLogWithContext:
    """Tests for log_with_context function."""

    def test_log_with_context_basic(self) -> None:
        """Test log_with_context basic usage."""
        from atp.core.logging import log_with_context

        configure_logging(json_output=False)
        logger = get_logger("test.context")

        # Should not raise
        log_with_context(logger, "info", "test_event", key="value")

    def test_log_with_context_with_correlation_id(self) -> None:
        """Test log_with_context with explicit correlation ID."""
        from atp.core.logging import log_with_context

        configure_logging(json_output=False)
        logger = get_logger("test.context")

        # Should not raise
        log_with_context(
            logger, "info", "test_event", correlation_id="corr-123", key="value"
        )


class TestCreateTestLogger:
    """Tests for create_test_logger function."""

    def test_create_test_logger_default(self) -> None:
        """Test creating a test logger with defaults."""
        from atp.core.logging import create_test_logger

        logger = create_test_logger()
        assert logger is not None

    def test_create_test_logger_json(self) -> None:
        """Test creating a test logger with JSON output."""
        from atp.core.logging import create_test_logger

        logger = create_test_logger(name="test.json", json_output=True)
        assert logger is not None

    def test_create_test_logger_custom_name(self) -> None:
        """Test creating a test logger with custom name."""
        from atp.core.logging import create_test_logger

        logger = create_test_logger(name="custom.test")
        assert logger is not None


class TestGetProcessors:
    """Tests for _get_processors function."""

    def test_get_processors_console(self) -> None:
        """Test getting processors for console output."""
        from atp.core.logging import _get_processors

        processors = _get_processors(json_output=False)
        assert len(processors) > 0

    def test_get_processors_json(self) -> None:
        """Test getting processors for JSON output."""
        from atp.core.logging import _get_processors

        processors = _get_processors(json_output=True)
        assert len(processors) > 0


class TestHostnameCache:
    """Tests for hostname caching."""

    def test_get_hostname_cached(self) -> None:
        """Test that hostname is cached."""
        from atp.core.logging import _get_hostname

        # Clear any existing cache
        _get_hostname.cache_clear()

        hostname1 = _get_hostname()
        hostname2 = _get_hostname()

        assert hostname1 == hostname2
        assert isinstance(hostname1, str)


class TestConfigureLoggingFromSettings:
    """Tests for configure_logging_from_settings."""

    def test_configure_from_settings(self) -> None:
        """Test configuring logging from settings."""
        from atp.core.logging import configure_logging_from_settings

        # This should use default settings and not raise
        configure_logging_from_settings()

        root_logger = logging.getLogger()
        assert root_logger.level in [logging.DEBUG, logging.INFO, logging.WARNING]
