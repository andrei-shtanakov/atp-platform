"""Unit tests for base adapter classes and exceptions."""

import pytest

from atp.adapters import (
    AdapterConfig,
    AdapterConnectionError,
    AdapterError,
    AdapterNotFoundError,
    AdapterResponseError,
    AdapterTimeoutError,
)


class TestAdapterConfig:
    """Tests for AdapterConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AdapterConfig()

        assert config.timeout_seconds == 300.0
        assert config.retry_count == 0
        assert config.retry_delay_seconds == 1.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AdapterConfig(
            timeout_seconds=60.0,
            retry_count=3,
            retry_delay_seconds=2.0,
        )

        assert config.timeout_seconds == 60.0
        assert config.retry_count == 3
        assert config.retry_delay_seconds == 2.0

    def test_timeout_must_be_positive(self) -> None:
        """Test that timeout must be positive."""
        with pytest.raises(ValueError):
            AdapterConfig(timeout_seconds=0)

        with pytest.raises(ValueError):
            AdapterConfig(timeout_seconds=-1)

    def test_retry_count_non_negative(self) -> None:
        """Test that retry_count must be non-negative."""
        with pytest.raises(ValueError):
            AdapterConfig(retry_count=-1)

    def test_retry_delay_non_negative(self) -> None:
        """Test that retry_delay must be non-negative."""
        with pytest.raises(ValueError):
            AdapterConfig(retry_delay_seconds=-1)


class TestAdapterError:
    """Tests for AdapterError."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = AdapterError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.adapter_type is None
        assert error.cause is None

    def test_error_with_adapter_type(self) -> None:
        """Test error with adapter type."""
        error = AdapterError("Error", adapter_type="http")

        assert error.adapter_type == "http"

    def test_error_with_cause(self) -> None:
        """Test error with cause."""
        original = ValueError("Original error")
        error = AdapterError("Wrapped error", cause=original)

        assert error.cause is original


class TestAdapterTimeoutError:
    """Tests for AdapterTimeoutError."""

    def test_basic_timeout(self) -> None:
        """Test basic timeout error."""
        error = AdapterTimeoutError()

        assert "timed out" in str(error)
        assert error.timeout_seconds is None

    def test_timeout_with_duration(self) -> None:
        """Test timeout error with duration."""
        error = AdapterTimeoutError(
            message="Request timed out after 30s",
            timeout_seconds=30.0,
        )

        assert error.timeout_seconds == 30.0
        assert "30s" in str(error)

    def test_timeout_with_adapter_type(self) -> None:
        """Test timeout error with adapter type."""
        error = AdapterTimeoutError(
            timeout_seconds=60.0,
            adapter_type="http",
        )

        assert error.adapter_type == "http"


class TestAdapterConnectionError:
    """Tests for AdapterConnectionError."""

    def test_basic_connection_error(self) -> None:
        """Test basic connection error."""
        error = AdapterConnectionError()

        assert "Connection failed" in str(error)
        assert error.endpoint is None

    def test_connection_error_with_endpoint(self) -> None:
        """Test connection error with endpoint."""
        error = AdapterConnectionError(
            message="Failed to connect to http://localhost:8000",
            endpoint="http://localhost:8000",
        )

        assert error.endpoint == "http://localhost:8000"
        assert "localhost" in str(error)

    def test_connection_error_with_cause(self) -> None:
        """Test connection error with cause."""
        original = OSError("Connection refused")
        error = AdapterConnectionError(cause=original)

        assert error.cause is original


class TestAdapterResponseError:
    """Tests for AdapterResponseError."""

    def test_basic_response_error(self) -> None:
        """Test basic response error."""
        error = AdapterResponseError()

        assert "Invalid response" in str(error)
        assert error.status_code is None
        assert error.response_body is None

    def test_response_error_with_status(self) -> None:
        """Test response error with status code."""
        error = AdapterResponseError(
            message="Server error",
            status_code=500,
        )

        assert error.status_code == 500

    def test_response_error_with_body(self) -> None:
        """Test response error with response body."""
        error = AdapterResponseError(
            message="Invalid JSON",
            response_body="not valid json",
        )

        assert error.response_body == "not valid json"

    def test_response_error_with_all_fields(self) -> None:
        """Test response error with all fields."""
        error = AdapterResponseError(
            message="Bad request",
            status_code=400,
            response_body='{"error": "invalid input"}',
            adapter_type="http",
        )

        assert error.status_code == 400
        assert "invalid input" in error.response_body  # type: ignore
        assert error.adapter_type == "http"


class TestAdapterNotFoundError:
    """Tests for AdapterNotFoundError."""

    def test_not_found_error(self) -> None:
        """Test adapter not found error."""
        error = AdapterNotFoundError("custom")

        assert "custom" in str(error)
        assert "not found" in str(error)
        assert error.adapter_type == "custom"
