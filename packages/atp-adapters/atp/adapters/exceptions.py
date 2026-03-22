"""Adapter-specific exceptions."""

from atp.core.exceptions import ATPError


class AdapterError(ATPError):
    """Base exception for adapter errors."""

    def __init__(
        self,
        message: str,
        adapter_type: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.adapter_type = adapter_type
        self.cause = cause
        super().__init__(message)


class AdapterTimeoutError(AdapterError):
    """Raised when an adapter operation times out."""

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: float | None = None,
        adapter_type: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(message, adapter_type=adapter_type)


class AdapterConnectionError(AdapterError):
    """Raised when adapter cannot connect to the agent."""

    def __init__(
        self,
        message: str = "Connection failed",
        endpoint: str | None = None,
        adapter_type: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.endpoint = endpoint
        super().__init__(message, adapter_type=adapter_type, cause=cause)


class AdapterResponseError(AdapterError):
    """Raised when the agent returns an invalid response."""

    def __init__(
        self,
        message: str = "Invalid response",
        status_code: int | None = None,
        response_body: str | None = None,
        adapter_type: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message, adapter_type=adapter_type)


class AdapterNotFoundError(AdapterError):
    """Raised when a requested adapter type is not registered."""

    def __init__(self, adapter_type: str) -> None:
        super().__init__(
            f"Adapter type '{adapter_type}' not found in registry",
            adapter_type=adapter_type,
        )
