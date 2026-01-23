"""Runner-specific exceptions."""

from atp.core.exceptions import ATPError


class RunnerError(ATPError):
    """Base exception for runner errors."""

    def __init__(
        self,
        message: str,
        test_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.test_id = test_id
        self.cause = cause
        super().__init__(message)


class TestExecutionError(RunnerError):
    """Raised when test execution fails."""

    def __init__(
        self,
        message: str,
        test_id: str | None = None,
        run_number: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.run_number = run_number
        super().__init__(message, test_id=test_id, cause=cause)


class RunnerTimeoutError(RunnerError):
    """Raised when test execution times out."""

    def __init__(
        self,
        message: str = "Test execution timed out",
        test_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(message, test_id=test_id)


class SandboxError(RunnerError):
    """Raised when sandbox operations fail."""

    def __init__(
        self,
        message: str = "Sandbox error",
        test_id: str | None = None,
        sandbox_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.sandbox_id = sandbox_id
        super().__init__(message, test_id=test_id, cause=cause)
