"""Tests for runner exceptions."""

from atp.core.exceptions import ATPError
from atp.runner.exceptions import (
    RunnerError,
    RunnerTimeoutError,
    SandboxError,
    TestExecutionError,
)


class TestRunnerError:
    """Tests for RunnerError."""

    def test_base_error_inheritance(self) -> None:
        """RunnerError inherits from ATPError."""
        error = RunnerError("test error")
        assert isinstance(error, ATPError)
        assert isinstance(error, Exception)

    def test_message_only(self) -> None:
        """RunnerError with message only."""
        error = RunnerError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.test_id is None
        assert error.cause is None

    def test_with_test_id(self) -> None:
        """RunnerError with test_id."""
        error = RunnerError("Error", test_id="test-001")
        assert error.test_id == "test-001"

    def test_with_cause(self) -> None:
        """RunnerError with cause exception."""
        cause = ValueError("Original error")
        error = RunnerError("Wrapped error", cause=cause)
        assert error.cause is cause


class TestTestExecutionError:
    """Tests for TestExecutionError."""

    def test_inheritance(self) -> None:
        """TestExecutionError inherits from RunnerError."""
        error = TestExecutionError("test error")
        assert isinstance(error, RunnerError)

    def test_with_run_number(self) -> None:
        """TestExecutionError with run number."""
        error = TestExecutionError(
            "Run failed",
            test_id="test-001",
            run_number=3,
        )
        assert error.test_id == "test-001"
        assert error.run_number == 3

    def test_with_cause(self) -> None:
        """TestExecutionError with cause."""
        cause = RuntimeError("Adapter failed")
        error = TestExecutionError("Execution failed", cause=cause)
        assert error.cause is cause


class TestRunnerTimeoutError:
    """Tests for RunnerTimeoutError."""

    def test_inheritance(self) -> None:
        """RunnerTimeoutError inherits from RunnerError."""
        error = RunnerTimeoutError()
        assert isinstance(error, RunnerError)

    def test_default_message(self) -> None:
        """RunnerTimeoutError has default message."""
        error = RunnerTimeoutError()
        assert str(error) == "Test execution timed out"

    def test_custom_message(self) -> None:
        """RunnerTimeoutError with custom message."""
        error = RunnerTimeoutError("Timed out after 30s")
        assert str(error) == "Timed out after 30s"

    def test_with_timeout_seconds(self) -> None:
        """RunnerTimeoutError with timeout info."""
        error = RunnerTimeoutError(
            "Timeout",
            test_id="test-001",
            timeout_seconds=60.0,
        )
        assert error.test_id == "test-001"
        assert error.timeout_seconds == 60.0


class TestSandboxError:
    """Tests for SandboxError."""

    def test_inheritance(self) -> None:
        """SandboxError inherits from RunnerError."""
        error = SandboxError()
        assert isinstance(error, RunnerError)

    def test_default_message(self) -> None:
        """SandboxError has default message."""
        error = SandboxError()
        assert str(error) == "Sandbox error"

    def test_with_sandbox_id(self) -> None:
        """SandboxError with sandbox_id."""
        error = SandboxError(
            "Container failed",
            sandbox_id="sandbox-abc123",
        )
        assert error.sandbox_id == "sandbox-abc123"

    def test_with_all_params(self) -> None:
        """SandboxError with all parameters."""
        cause = OSError("Disk full")
        error = SandboxError(
            "Failed to create workspace",
            test_id="test-001",
            sandbox_id="sandbox-abc123",
            cause=cause,
        )
        assert error.test_id == "test-001"
        assert error.sandbox_id == "sandbox-abc123"
        assert error.cause is cause
