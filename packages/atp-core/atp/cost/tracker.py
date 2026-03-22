"""Cost tracker with pluggable persistence backend."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from atp.cost.models import CostEvent, PricingConfig

logger = logging.getLogger(__name__)


class CostPersistenceBackend(Protocol):
    """Protocol for cost persistence backends.

    Implement this to persist cost events to a database or other storage.
    The default CostTracker works without a backend (in-memory only).
    """

    async def persist_batch(
        self,
        events: list[CostEvent],
        pricing: PricingConfig,
    ) -> None:
        """Persist a batch of cost events.

        Args:
            events: Cost events to persist.
            pricing: Pricing config for cost calculation.
        """
        ...


class CostTracker:
    """Non-blocking cost tracking service.

    Features:
    - Async queue for non-blocking cost event tracking
    - Background processor for batch persistence
    - Pluggable persistence backend (optional)
    - Configurable batch size and timeout
    - Budget checking support

    Example:
        tracker = CostTracker(pricing=PricingConfig.default())
        await tracker.start()

        await tracker.track(CostEvent(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
        ))

        await tracker.stop()
    """

    def __init__(
        self,
        pricing: PricingConfig | None = None,
        backend: CostPersistenceBackend | None = None,
        batch_size: int = 100,
        batch_timeout: float = 5.0,
        max_queue_size: int = 10000,
    ):
        """Initialize cost tracker.

        Args:
            pricing: Pricing configuration. Uses defaults if not provided.
            backend: Optional persistence backend for storing events.
            batch_size: Number of events to batch before processing.
            batch_timeout: Seconds to wait before processing incomplete batch.
            max_queue_size: Maximum queue size before dropping events.
        """
        self.pricing = pricing or PricingConfig.default()
        self._backend = backend
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_queue_size = max_queue_size

        self._queue: asyncio.Queue[CostEvent] = asyncio.Queue(maxsize=max_queue_size)
        self._processor_task: asyncio.Task[None] | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._events_processed = 0
        self._events_failed = 0
        self._batches_processed = 0

    @property
    def is_running(self) -> bool:
        """Check if the processor is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def stats(self) -> dict[str, int]:
        """Get tracking statistics."""
        return {
            "events_processed": self._events_processed,
            "events_failed": self._events_failed,
            "batches_processed": self._batches_processed,
            "queue_size": self.queue_size,
        }

    def set_backend(self, backend: CostPersistenceBackend) -> None:
        """Set the persistence backend.

        Can be called after construction to register a backend
        (e.g., when dashboard package is loaded).

        Args:
            backend: Persistence backend to use.
        """
        self._backend = backend

    async def start(self) -> None:
        """Start background cost processor."""
        if self._running:
            logger.warning("Cost tracker is already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._processor_task = asyncio.create_task(
            self._process_loop(), name="cost_tracker_processor"
        )
        logger.info("Cost tracker started")

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop background processor and process remaining events."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        if self._processor_task:
            try:
                await asyncio.wait_for(self._processor_task, timeout=timeout)
            except TimeoutError:
                logger.warning("Cost tracker shutdown timed out, cancelling")
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass
            self._processor_task = None

        logger.info(
            f"Cost tracker stopped. Stats: {self._events_processed} "
            f"processed, {self._events_failed} failed, "
            f"{self._batches_processed} batches"
        )

    async def track(self, event: CostEvent) -> None:
        """Queue a cost event for processing.

        Non-blocking — returns immediately after queueing.
        """
        if not self._running:
            logger.warning("Cost tracker is not running, event may not be processed")

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.error(
                f"Cost tracking queue is full ({self.max_queue_size}). "
                "Event dropped. Consider increasing max_queue_size."
            )
            self._events_failed += 1

    async def track_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        test_id: str | None = None,
        suite_id: str | None = None,
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Decimal:
        """Track an LLM call and return calculated cost."""
        event = CostEvent(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            test_id=test_id,
            suite_id=suite_id,
            agent_name=agent_name,
            metadata=metadata,
        )
        await self.track(event)
        return self.pricing.calculate(provider, model, input_tokens, output_tokens)

    async def _process_loop(self) -> None:
        """Background loop to process cost events in batches."""
        batch: list[CostEvent] = []

        while self._running or not self._queue.empty():
            try:
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(), timeout=self.batch_timeout
                    )
                    batch.append(event)
                    self._queue.task_done()
                except TimeoutError:
                    pass

                while len(batch) < self.batch_size and not self._queue.empty():
                    try:
                        event = self._queue.get_nowait()
                        batch.append(event)
                        self._queue.task_done()
                    except asyncio.QueueEmpty:
                        break

                if batch and (
                    len(batch) >= self.batch_size
                    or self._shutdown_event.is_set()
                    or not self._running
                ):
                    await self._process_batch(batch)
                    batch = []

            except asyncio.CancelledError:
                if batch:
                    try:
                        await self._process_batch(batch)
                    except Exception as e:
                        logger.error(f"Failed to process final batch: {e}")
                raise

            except Exception as e:
                logger.error(f"Error in cost processor loop: {e}")
                await asyncio.sleep(1.0)

        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[CostEvent]) -> None:
        """Process a batch of cost events."""
        if not batch:
            return

        try:
            if self._backend is not None:
                await self._backend.persist_batch(batch, self.pricing)

            self._events_processed += len(batch)
            self._batches_processed += 1
            logger.debug(f"Processed batch of {len(batch)} cost events")

        except Exception as e:
            self._events_failed += len(batch)
            logger.error(f"Failed to process cost batch: {e}")
            raise

    async def flush(self) -> None:
        """Force processing of all queued events."""
        if not self._running:
            logger.warning("Cost tracker is not running")
            return
        await self._queue.join()

    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        """Calculate cost without tracking."""
        return self.pricing.calculate(provider, model, input_tokens, output_tokens)


# Global cost tracker instance
_cost_tracker: CostTracker | None = None


async def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance.

    Creates and starts tracker if not already running.
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
        await _cost_tracker.start()
    return _cost_tracker


def set_cost_tracker(tracker: CostTracker) -> None:
    """Set the global cost tracker instance (useful for testing)."""
    global _cost_tracker
    _cost_tracker = tracker


async def shutdown_cost_tracker() -> None:
    """Shutdown the global cost tracker."""
    global _cost_tracker
    if _cost_tracker is not None:
        await _cost_tracker.stop()
        _cost_tracker = None
