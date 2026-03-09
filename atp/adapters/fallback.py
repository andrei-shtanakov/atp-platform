"""Fallback adapter chain for resilient agent execution."""

import logging
from collections.abc import AsyncIterator

from atp.adapters.base import AdapterConfig, AgentAdapter
from atp.adapters.exceptions import AdapterError
from atp.protocol import ATPEvent, ATPRequest, ATPResponse

logger = logging.getLogger(__name__)


class FallbackAdapter(AgentAdapter):
    """Adapter that tries multiple adapters in sequence.

    When the primary adapter fails, falls back to the next adapter
    in the chain. Useful for resilient multi-provider setups.
    """

    def __init__(
        self,
        chain: list[AgentAdapter],
        config: AdapterConfig | None = None,
    ) -> None:
        """Initialize fallback adapter.

        Args:
            chain: Ordered list of adapters to try.
            config: Optional config override.

        Raises:
            ValueError: If chain is empty.
        """
        if not chain:
            raise ValueError("Fallback chain must have at least one adapter")
        super().__init__(config)
        self.chain = chain

    @property
    def adapter_type(self) -> str:
        return "fallback"

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Execute request, falling back through the chain on failure."""
        errors: list[tuple[str, Exception]] = []

        for adapter in self.chain:
            try:
                response = await adapter.execute(request)
                if errors:
                    logger.info(
                        "Fallback succeeded with adapter '%s' after %d failure(s)",
                        adapter.adapter_type,
                        len(errors),
                    )
                return response
            except Exception as exc:
                logger.warning(
                    "Adapter '%s' failed: %s. Trying next.",
                    adapter.adapter_type,
                    exc,
                )
                errors.append((adapter.adapter_type, exc))

        adapter_names = [name for name, _ in errors]
        raise AdapterError(
            f"All adapters failed: {', '.join(adapter_names)}",
            adapter_type="fallback",
        )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """Stream events, falling back through the chain on failure."""
        errors: list[tuple[str, Exception]] = []

        for adapter in self.chain:
            try:
                async for item in adapter.stream_events(request):
                    yield item
                return
            except Exception as exc:
                logger.warning(
                    "Adapter '%s' streaming failed: %s. Trying next.",
                    adapter.adapter_type,
                    exc,
                )
                errors.append((adapter.adapter_type, exc))

        adapter_names = [name for name, _ in errors]
        raise AdapterError(
            f"All adapters failed: {', '.join(adapter_names)}",
            adapter_type="fallback",
        )

    async def health_check(self) -> bool:
        """Return True if any adapter in the chain is healthy."""
        for adapter in self.chain:
            if await adapter.health_check():
                return True
        return False

    async def cleanup(self) -> None:
        """Clean up all adapters in the chain."""
        for adapter in self.chain:
            await adapter.cleanup()
