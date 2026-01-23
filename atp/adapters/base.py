"""Base adapter interface for agent communication."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from atp.protocol import ATPEvent, ATPRequest, ATPResponse


class AdapterConfig(BaseModel):
    """Base configuration for adapters."""

    timeout_seconds: float = Field(
        default=300.0, description="Timeout for execution in seconds", gt=0
    )
    retry_count: int = Field(
        default=0, description="Number of retries on failure", ge=0
    )
    retry_delay_seconds: float = Field(
        default=1.0, description="Delay between retries in seconds", ge=0
    )


class AgentAdapter(ABC):
    """
    Base class for agent adapters.

    Adapters translate between the ATP Protocol and agent-specific
    communication mechanisms (HTTP, Docker, CLI, etc.).
    """

    def __init__(self, config: AdapterConfig | None = None) -> None:
        """
        Initialize the adapter.

        Args:
            config: Adapter configuration. Uses defaults if not provided.
        """
        self.config = config or AdapterConfig()

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""

    @abstractmethod
    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task synchronously.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse with execution results.

        Raises:
            AdapterError: If execution fails.
            AdapterTimeoutError: If execution times out.
        """

    @abstractmethod
    def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """
        Execute a task with event streaming.

        Yields ATP Events during execution and ends with an ATP Response.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects during execution.
            Final ATPResponse when complete.

        Raises:
            AdapterError: If execution fails.
            AdapterTimeoutError: If execution times out.
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if the agent is available and healthy.

        Returns:
            True if agent is healthy, False otherwise.
        """
        return True

    async def cleanup(self) -> None:
        """
        Release any resources held by the adapter.

        Called when the adapter is no longer needed.
        """

    async def __aenter__(self) -> "AgentAdapter":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.cleanup()
