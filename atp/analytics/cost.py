"""Cost tracking service for ATP Platform.

This module provides non-blocking cost tracking for all LLM operations,
with support for multiple providers (OpenAI, Anthropic, Google, Azure, Bedrock).

Features:
- Async event queue for non-blocking cost tracking
- Background processor for batch inserts
- Configurable pricing for all major providers
- Custom pricing configuration support
- Budget checking and alerts

Example usage:
    from atp.analytics.cost import get_cost_tracker, CostEvent

    tracker = await get_cost_tracker()
    await tracker.track(CostEvent(
        timestamp=datetime.now(),
        provider="anthropic",
        model="claude-3-sonnet",
        input_tokens=1000,
        output_tokens=500,
    ))
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from atp.analytics.database import get_analytics_database
from atp.analytics.models import CostRecord
from atp.analytics.repository import CostRepository

logger = logging.getLogger(__name__)


@dataclass
class CostEvent:
    """Event representing an LLM operation cost.

    Attributes:
        timestamp: When the operation occurred.
        provider: LLM provider (anthropic, openai, google, azure, bedrock).
        model: Model name (claude-3-sonnet, gpt-4, etc.).
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        test_id: Optional test ID for association.
        suite_id: Optional suite ID for association.
        agent_name: Optional agent name for association.
        metadata: Optional additional metadata.
    """

    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    test_id: str | None = None
    suite_id: str | None = None
    agent_name: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ModelPricing:
    """Pricing configuration for a specific model.

    Prices are in USD per 1,000 tokens.
    """

    input_per_1k: Decimal
    output_per_1k: Decimal
    name: str = ""

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Calculate cost for given token counts.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.
        """
        input_cost = (Decimal(input_tokens) / Decimal(1000)) * self.input_per_1k
        output_cost = (Decimal(output_tokens) / Decimal(1000)) * self.output_per_1k
        return input_cost + output_cost


@dataclass
class PricingConfig:
    """Pricing configuration for all models.

    Contains pricing for built-in models and custom configurations.
    """

    models: dict[str, ModelPricing] = field(default_factory=dict)
    provider_defaults: dict[str, ModelPricing] = field(default_factory=dict)

    @classmethod
    def default(cls) -> PricingConfig:
        """Create default pricing configuration with all major providers.

        Pricing is based on public pricing as of early 2026.
        Prices are in USD per 1,000 tokens.
        """
        models = {
            # Anthropic Claude models
            "claude-opus-4-5-20251101": ModelPricing(
                input_per_1k=Decimal("0.015"),
                output_per_1k=Decimal("0.075"),
                name="Claude Opus 4.5",
            ),
            "claude-sonnet-4-20250514": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Claude Sonnet 4",
            ),
            "claude-3-5-sonnet-20241022": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Claude 3.5 Sonnet",
            ),
            "claude-3-5-haiku-20241022": ModelPricing(
                input_per_1k=Decimal("0.0008"),
                output_per_1k=Decimal("0.004"),
                name="Claude 3.5 Haiku",
            ),
            "claude-3-opus-20240229": ModelPricing(
                input_per_1k=Decimal("0.015"),
                output_per_1k=Decimal("0.075"),
                name="Claude 3 Opus",
            ),
            "claude-3-sonnet-20240229": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Claude 3 Sonnet",
            ),
            "claude-3-haiku-20240307": ModelPricing(
                input_per_1k=Decimal("0.00025"),
                output_per_1k=Decimal("0.00125"),
                name="Claude 3 Haiku",
            ),
            # OpenAI GPT models
            "gpt-4o": ModelPricing(
                input_per_1k=Decimal("0.0025"),
                output_per_1k=Decimal("0.01"),
                name="GPT-4o",
            ),
            "gpt-4o-mini": ModelPricing(
                input_per_1k=Decimal("0.00015"),
                output_per_1k=Decimal("0.0006"),
                name="GPT-4o Mini",
            ),
            "gpt-4-turbo": ModelPricing(
                input_per_1k=Decimal("0.01"),
                output_per_1k=Decimal("0.03"),
                name="GPT-4 Turbo",
            ),
            "gpt-4": ModelPricing(
                input_per_1k=Decimal("0.03"),
                output_per_1k=Decimal("0.06"),
                name="GPT-4",
            ),
            "gpt-3.5-turbo": ModelPricing(
                input_per_1k=Decimal("0.0005"),
                output_per_1k=Decimal("0.0015"),
                name="GPT-3.5 Turbo",
            ),
            "o1": ModelPricing(
                input_per_1k=Decimal("0.015"),
                output_per_1k=Decimal("0.06"),
                name="O1",
            ),
            "o1-mini": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.012"),
                name="O1 Mini",
            ),
            "o1-pro": ModelPricing(
                input_per_1k=Decimal("0.15"),
                output_per_1k=Decimal("0.6"),
                name="O1 Pro",
            ),
            # Google Gemini models
            "gemini-1.5-pro": ModelPricing(
                input_per_1k=Decimal("0.00125"),
                output_per_1k=Decimal("0.005"),
                name="Gemini 1.5 Pro",
            ),
            "gemini-1.5-flash": ModelPricing(
                input_per_1k=Decimal("0.000075"),
                output_per_1k=Decimal("0.0003"),
                name="Gemini 1.5 Flash",
            ),
            "gemini-2.0-flash": ModelPricing(
                input_per_1k=Decimal("0.0001"),
                output_per_1k=Decimal("0.0004"),
                name="Gemini 2.0 Flash",
            ),
            "gemini-2.0-pro": ModelPricing(
                input_per_1k=Decimal("0.00125"),
                output_per_1k=Decimal("0.005"),
                name="Gemini 2.0 Pro",
            ),
            # AWS Bedrock - Claude models (same as Anthropic but through Bedrock)
            "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Claude 3.5 Sonnet (Bedrock)",
            ),
            "anthropic.claude-3-sonnet-20240229-v1:0": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Claude 3 Sonnet (Bedrock)",
            ),
            "anthropic.claude-3-haiku-20240307-v1:0": ModelPricing(
                input_per_1k=Decimal("0.00025"),
                output_per_1k=Decimal("0.00125"),
                name="Claude 3 Haiku (Bedrock)",
            ),
            "anthropic.claude-3-opus-20240229-v1:0": ModelPricing(
                input_per_1k=Decimal("0.015"),
                output_per_1k=Decimal("0.075"),
                name="Claude 3 Opus (Bedrock)",
            ),
            # AWS Bedrock - Titan models
            "amazon.titan-text-express-v1": ModelPricing(
                input_per_1k=Decimal("0.0002"),
                output_per_1k=Decimal("0.0006"),
                name="Titan Text Express",
            ),
            "amazon.titan-text-lite-v1": ModelPricing(
                input_per_1k=Decimal("0.00015"),
                output_per_1k=Decimal("0.0002"),
                name="Titan Text Lite",
            ),
            # Azure OpenAI (same as OpenAI but through Azure)
            "azure/gpt-4o": ModelPricing(
                input_per_1k=Decimal("0.0025"),
                output_per_1k=Decimal("0.01"),
                name="GPT-4o (Azure)",
            ),
            "azure/gpt-4o-mini": ModelPricing(
                input_per_1k=Decimal("0.00015"),
                output_per_1k=Decimal("0.0006"),
                name="GPT-4o Mini (Azure)",
            ),
            "azure/gpt-4-turbo": ModelPricing(
                input_per_1k=Decimal("0.01"),
                output_per_1k=Decimal("0.03"),
                name="GPT-4 Turbo (Azure)",
            ),
            "azure/gpt-4": ModelPricing(
                input_per_1k=Decimal("0.03"),
                output_per_1k=Decimal("0.06"),
                name="GPT-4 (Azure)",
            ),
        }

        # Provider defaults for unknown models
        provider_defaults = {
            "anthropic": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Anthropic Default",
            ),
            "openai": ModelPricing(
                input_per_1k=Decimal("0.0025"),
                output_per_1k=Decimal("0.01"),
                name="OpenAI Default",
            ),
            "google": ModelPricing(
                input_per_1k=Decimal("0.00125"),
                output_per_1k=Decimal("0.005"),
                name="Google Default",
            ),
            "azure": ModelPricing(
                input_per_1k=Decimal("0.0025"),
                output_per_1k=Decimal("0.01"),
                name="Azure Default",
            ),
            "bedrock": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Bedrock Default",
            ),
        }

        return cls(models=models, provider_defaults=provider_defaults)

    @classmethod
    def from_yaml(cls, path: Path) -> PricingConfig:
        """Load pricing configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            PricingConfig instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If configuration is invalid.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Pricing configuration must be a dictionary")

        config = cls.default()

        # Load custom model pricing
        if "models" in data:
            for model_name, pricing_data in data["models"].items():
                if not isinstance(pricing_data, dict):
                    raise ValueError(f"Invalid pricing for model {model_name}")

                config.models[model_name] = ModelPricing(
                    input_per_1k=Decimal(str(pricing_data.get("input_per_1k", 0))),
                    output_per_1k=Decimal(str(pricing_data.get("output_per_1k", 0))),
                    name=str(pricing_data.get("name", model_name)),
                )

        # Load provider defaults
        if "provider_defaults" in data:
            for provider, pricing_data in data["provider_defaults"].items():
                if not isinstance(pricing_data, dict):
                    raise ValueError(f"Invalid default pricing for provider {provider}")

                config.provider_defaults[provider] = ModelPricing(
                    input_per_1k=Decimal(str(pricing_data.get("input_per_1k", 0))),
                    output_per_1k=Decimal(str(pricing_data.get("output_per_1k", 0))),
                    name=pricing_data.get("name", f"{provider} Default"),
                )

        return config

    def calculate(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal:
        """Calculate cost for an LLM operation.

        Args:
            provider: LLM provider name.
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.
        """
        # Try exact model match first
        if model in self.models:
            return self.models[model].calculate_cost(input_tokens, output_tokens)

        # Try provider/model format for Azure
        azure_key = f"azure/{model}"
        if azure_key in self.models:
            return self.models[azure_key].calculate_cost(input_tokens, output_tokens)

        # Try provider default
        provider_lower = provider.lower()
        if provider_lower in self.provider_defaults:
            logger.debug(
                f"Using default pricing for unknown model {model} "
                f"from provider {provider}"
            )
            return self.provider_defaults[provider_lower].calculate_cost(
                input_tokens, output_tokens
            )

        # Fall back to generic default
        logger.warning(
            f"No pricing found for model {model} from provider {provider}. "
            "Using zero cost."
        )
        return Decimal("0")

    def get_model_pricing(self, model: str) -> ModelPricing | None:
        """Get pricing for a specific model.

        Args:
            model: Model name.

        Returns:
            ModelPricing if found, None otherwise.
        """
        return self.models.get(model)

    def add_custom_pricing(
        self,
        model: str,
        input_per_1k: Decimal | float | str,
        output_per_1k: Decimal | float | str,
        name: str | None = None,
    ) -> None:
        """Add custom pricing for a model.

        Args:
            model: Model name.
            input_per_1k: Input price per 1,000 tokens.
            output_per_1k: Output price per 1,000 tokens.
            name: Optional display name.
        """
        self.models[model] = ModelPricing(
            input_per_1k=Decimal(str(input_per_1k)),
            output_per_1k=Decimal(str(output_per_1k)),
            name=name or model,
        )


class CostTracker:
    """Non-blocking cost tracking service.

    Features:
    - Async queue for non-blocking cost event tracking
    - Background processor for batch database inserts
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
        batch_size: int = 100,
        batch_timeout: float = 5.0,
        max_queue_size: int = 10000,
    ):
        """Initialize cost tracker.

        Args:
            pricing: Pricing configuration. Uses defaults if not provided.
            batch_size: Number of events to batch before insert.
            batch_timeout: Seconds to wait before processing incomplete batch.
            max_queue_size: Maximum queue size before blocking.
        """
        self.pricing = pricing or PricingConfig.default()
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
        """Stop background processor and process remaining events.

        Args:
            timeout: Maximum seconds to wait for remaining events.
        """
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
            f"Cost tracker stopped. Stats: {self._events_processed} processed, "
            f"{self._events_failed} failed, {self._batches_processed} batches"
        )

    async def track(self, event: CostEvent) -> None:
        """Queue a cost event for processing.

        This method is non-blocking and returns immediately after queueing.

        Args:
            event: Cost event to track.

        Raises:
            RuntimeError: If tracker is not running and queue is full.
        """
        if not self._running:
            logger.warning("Cost tracker is not running, event may not be processed")

        try:
            # Use put_nowait to avoid blocking, but handle full queue gracefully
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
        """Convenience method to track an LLM call and return calculated cost.

        Args:
            provider: LLM provider name.
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            test_id: Optional test ID.
            suite_id: Optional suite ID.
            agent_name: Optional agent name.
            metadata: Optional additional metadata.

        Returns:
            Calculated cost in USD.
        """
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
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(), timeout=self.batch_timeout
                    )
                    batch.append(event)
                    self._queue.task_done()
                except TimeoutError:
                    # Timeout reached, process whatever we have
                    pass

                # Also drain queue up to batch size
                while len(batch) < self.batch_size and not self._queue.empty():
                    try:
                        event = self._queue.get_nowait()
                        batch.append(event)
                        self._queue.task_done()
                    except asyncio.QueueEmpty:
                        break

                # Process batch if we have events
                if batch and (
                    len(batch) >= self.batch_size
                    or self._shutdown_event.is_set()
                    or not self._running
                ):
                    await self._process_batch(batch)
                    batch = []

            except asyncio.CancelledError:
                # Process remaining batch on cancellation
                if batch:
                    try:
                        await self._process_batch(batch)
                    except Exception as e:
                        logger.error(f"Failed to process final batch: {e}")
                raise

            except Exception as e:
                logger.error(f"Error in cost processor loop: {e}")
                # Don't lose the batch on error, try to reprocess
                await asyncio.sleep(1.0)

        # Process any remaining events
        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[CostEvent]) -> None:
        """Process a batch of cost events.

        Args:
            batch: List of cost events to process.
        """
        if not batch:
            return

        try:
            db = get_analytics_database()
            records = []

            for event in batch:
                cost = self.pricing.calculate(
                    event.provider,
                    event.model,
                    event.input_tokens,
                    event.output_tokens,
                )
                records.append(
                    CostRecord(
                        timestamp=event.timestamp,
                        provider=event.provider,
                        model=event.model,
                        input_tokens=event.input_tokens,
                        output_tokens=event.output_tokens,
                        cost_usd=cost,
                        test_id=event.test_id,
                        suite_id=event.suite_id,
                        agent_name=event.agent_name,
                        metadata_json=event.metadata,
                    )
                )

            async with db.session() as session:
                repo = CostRepository(session)
                await repo.create_cost_records_batch(records)

            self._events_processed += len(batch)
            self._batches_processed += 1
            logger.debug(f"Processed batch of {len(batch)} cost events")

        except Exception as e:
            self._events_failed += len(batch)
            logger.error(f"Failed to process cost batch: {e}")
            raise

    async def flush(self) -> None:
        """Force processing of all queued events.

        Blocks until queue is empty.
        """
        if not self._running:
            logger.warning("Cost tracker is not running")
            return

        # Wait for queue to be empty
        await self._queue.join()

    def calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal:
        """Calculate cost without tracking.

        Args:
            provider: LLM provider name.
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.
        """
        return self.pricing.calculate(provider, model, input_tokens, output_tokens)


# Global cost tracker instance
_cost_tracker: CostTracker | None = None


async def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance.

    Creates and starts tracker if not already running.

    Returns:
        CostTracker instance.
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
        await _cost_tracker.start()
    return _cost_tracker


def set_cost_tracker(tracker: CostTracker) -> None:
    """Set the global cost tracker instance.

    Useful for testing.

    Args:
        tracker: CostTracker instance to use.
    """
    global _cost_tracker
    _cost_tracker = tracker


async def shutdown_cost_tracker() -> None:
    """Shutdown the global cost tracker."""
    global _cost_tracker
    if _cost_tracker is not None:
        await _cost_tracker.stop()
        _cost_tracker = None
