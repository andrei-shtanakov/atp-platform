"""SDK adapter for in-process agent communication.

The SDKAdapter acts as a bridge between the ATP runner (platform side)
and an agent running in the same process via the ATP SDK. The platform
calls ``execute()`` which enqueues the request; the agent polls with
``pull_task()`` and resolves via ``resolve_task()``.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator

from atp.protocol import ATPEvent, ATPRequest, ATPResponse
from pydantic import Field

from atp.adapters.base import AdapterConfig, AgentAdapter

logger = logging.getLogger(__name__)


class SDKAdapterConfig(AdapterConfig):
    """Configuration for the SDK adapter."""

    timeout_seconds: float = Field(
        default=3600.0,
        description="Timeout waiting for agent response",
        gt=0,
    )


class SDKAdapter(AgentAdapter):
    """Adapter for in-process agent communication via the ATP SDK.

    The platform side calls :meth:`execute` which parks the request
    until the agent side pulls it with :meth:`pull_task` and resolves
    it with :meth:`resolve_task`.
    """

    def __init__(self, config: SDKAdapterConfig | None = None) -> None:
        """Initialize the SDK adapter.

        Args:
            config: Optional adapter configuration.
        """
        super().__init__(config or SDKAdapterConfig())
        self._pending_tasks: OrderedDict[str, ATPRequest] = OrderedDict()
        self._events: dict[str, asyncio.Event] = {}
        self._results: dict[str, ATPResponse] = {}

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "sdk"

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Execute a task by enqueuing it for the agent.

        The request is stored in a pending queue. An asyncio Event
        is created so we can wait for the agent to resolve the task.

        Args:
            request: ATP request with task specification.

        Returns:
            ATPResponse provided by the agent via resolve_task.

        Raises:
            TimeoutError: If the agent does not resolve in time.
        """
        task_id = request.task_id
        self._pending_tasks[task_id] = request
        self._events[task_id] = asyncio.Event()

        try:
            await asyncio.wait_for(
                self._events[task_id].wait(),
                timeout=self.config.timeout_seconds,
            )
        except TimeoutError:
            self._pending_tasks.pop(task_id, None)
            self._events.pop(task_id, None)
            self._results.pop(task_id, None)
            raise TimeoutError(
                f"SDK adapter timed out waiting for task "
                f"{task_id!r} after "
                f"{self.config.timeout_seconds}s"
            ) from None

        result = self._results.pop(task_id)
        self._events.pop(task_id, None)
        return result

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """Execute and yield the response (no streaming in MVP).

        Args:
            request: ATP request with task specification.

        Yields:
            The final ATPResponse.
        """
        response = await self.execute(request)
        yield response

    def pull_task(self) -> ATPRequest | None:
        """Pull the next pending task (FIFO).

        Returns:
            The oldest pending ATPRequest, or None if empty.
        """
        if not self._pending_tasks:
            return None
        _, request = self._pending_tasks.popitem(last=False)
        return request

    def resolve_task(self, task_id: str, response: ATPResponse) -> None:
        """Resolve a task with the agent's response.

        Stores the response and signals the waiting execute() call.

        Args:
            task_id: The task ID to resolve.
            response: The agent's response.
        """
        self._results[task_id] = response
        event = self._events.get(task_id)
        if event is not None:
            event.set()
