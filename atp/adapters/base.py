"""Base adapter interface for agent communication."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel, Field

from atp.core.metrics import get_metrics, record_llm_call
from atp.core.telemetry import (
    add_span_event,
    get_tracer,
    set_adapter_response_attributes,
    set_span_attribute,
)
from atp.protocol import ATPEvent, ATPRequest, ATPResponse

if TYPE_CHECKING:
    from atp.analytics.cost import CostTracker

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


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
    enable_cost_tracking: bool = Field(
        default=True, description="Enable cost tracking via CostTracker"
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

    async def execute_with_tracing(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task with OpenTelemetry tracing.

        This method wraps execute() with automatic span creation and
        attribute recording. Subclasses should override execute() as usual.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse with execution results.
        """
        import time

        start_time = time.perf_counter()
        metrics = get_metrics()

        with tracer.start_as_current_span(
            f"adapter:{self.adapter_type}:execute",
            kind=SpanKind.CLIENT,
            attributes={
                "atp.adapter.type": self.adapter_type,
                "atp.adapter.operation": "execute",
                "atp.task.id": request.task_id,
                "atp.timeout_seconds": self.config.timeout_seconds,
            },
        ) as span:
            if request.metadata:
                test_id = request.metadata.get("test_id")
                if test_id:
                    set_span_attribute("atp.test.id", test_id)

            add_span_event("adapter_execute_start")

            try:
                response = await self.execute(request)

                # Record duration in metrics
                duration = time.perf_counter() - start_time
                if metrics:
                    metrics.record_adapter_call(
                        adapter=self.adapter_type,
                        operation="execute",
                        duration_seconds=duration,
                    )

                # Record response attributes
                set_adapter_response_attributes(
                    status=response.status.value,
                    input_tokens=(
                        response.metrics.input_tokens if response.metrics else None
                    ),
                    output_tokens=(
                        response.metrics.output_tokens if response.metrics else None
                    ),
                )

                # Record LLM call metrics if available
                if response.metrics:
                    input_tokens = response.metrics.input_tokens or 0
                    output_tokens = response.metrics.output_tokens or 0
                    if input_tokens > 0 or output_tokens > 0:
                        # Use adapter type as provider since ATPResponse
                        # doesn't have metadata for provider/model info
                        record_llm_call(
                            provider=self.adapter_type,
                            model="unknown",
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        )

                if response.status.value == "completed":
                    span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(
                        Status(StatusCode.ERROR, response.error or "Unknown error")
                    )

                add_span_event(
                    "adapter_execute_complete",
                    {"status": response.status.value},
                )

                return response

            except Exception as e:
                # Record error in metrics
                duration = time.perf_counter() - start_time
                if metrics:
                    metrics.record_adapter_call(
                        adapter=self.adapter_type,
                        operation="execute",
                        duration_seconds=duration,
                    )
                    metrics.record_adapter_error(
                        adapter=self.adapter_type,
                        error=type(e).__name__,
                    )
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                add_span_event("adapter_execute_error", {"error": str(e)})
                raise

    async def stream_events_with_tracing(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """
        Execute a task with event streaming and OpenTelemetry tracing.

        This method wraps stream_events() with automatic span creation.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects during execution.
            Final ATPResponse when complete.
        """
        with tracer.start_as_current_span(
            f"adapter:{self.adapter_type}:stream",
            kind=SpanKind.CLIENT,
            attributes={
                "atp.adapter.type": self.adapter_type,
                "atp.adapter.operation": "stream",
                "atp.task.id": request.task_id,
            },
        ) as span:
            if request.metadata:
                test_id = request.metadata.get("test_id")
                if test_id:
                    set_span_attribute("atp.test.id", test_id)

            event_count = 0
            final_response: ATPResponse | None = None

            try:
                async for item in self.stream_events(request):
                    if isinstance(item, ATPEvent):
                        event_count += 1
                        yield item
                    else:
                        final_response = item
                        yield item

                set_span_attribute("atp.stream.event_count", event_count)

                if final_response:
                    set_adapter_response_attributes(
                        status=final_response.status.value,
                        input_tokens=(
                            final_response.metrics.input_tokens
                            if final_response.metrics
                            else None
                        ),
                        output_tokens=(
                            final_response.metrics.output_tokens
                            if final_response.metrics
                            else None
                        ),
                    )
                    if final_response.status.value == "completed":
                        span.set_status(Status(StatusCode.OK))
                    else:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                final_response.error or "Unknown error",
                            )
                        )

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    async def __aenter__(self) -> AgentAdapter:
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


async def track_response_cost(
    response: ATPResponse,
    provider: str,
    model: str | None = None,
    test_id: str | None = None,
    suite_id: str | None = None,
    agent_name: str | None = None,
    cost_tracker: CostTracker | None = None,
) -> None:
    """Track cost from an ATPResponse with metrics.

    This utility function extracts token usage from an ATPResponse's metrics
    and tracks the cost using the CostTracker.

    Args:
        response: ATPResponse with metrics containing token usage.
        provider: LLM provider name.
        model: Model name. If not provided, tries to extract from response.
        test_id: Optional test ID for association.
        suite_id: Optional suite ID for association.
        agent_name: Optional agent name for association.
        cost_tracker: Optional CostTracker. Uses global tracker if not provided.
    """
    if response.metrics is None:
        return

    metrics = response.metrics
    input_tokens = metrics.input_tokens
    output_tokens = metrics.output_tokens

    if input_tokens is None and output_tokens is None:
        return

    try:
        if cost_tracker is None:
            from atp.analytics.cost import get_cost_tracker

            cost_tracker = await get_cost_tracker()

        from atp.analytics.cost import CostEvent

        await cost_tracker.track(
            CostEvent(
                timestamp=datetime.now(),
                provider=provider,
                model=model or "unknown",
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
                test_id=test_id or response.task_id,
                suite_id=suite_id,
                agent_name=agent_name,
                metadata={
                    "source": "adapter",
                    "trace_id": response.trace_id,
                    "status": response.status.value,
                },
            )
        )
    except Exception as e:
        logger.warning("Failed to track response cost: %s", e)
