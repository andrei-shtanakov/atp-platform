"""Cost tracking service — backward compatibility re-exports.

The core cost tracking logic has moved to atp.cost.
This module re-exports everything for backward compatibility
and provides the AnalyticsCostBackend that persists to the analytics DB.
"""

from __future__ import annotations

import logging

from atp.cost.models import CostEvent, ModelPricing, PricingConfig
from atp.cost.tracker import (
    CostPersistenceBackend,
    CostTracker,
    get_cost_tracker,
    set_cost_tracker,
    shutdown_cost_tracker,
)

logger = logging.getLogger(__name__)


class AnalyticsCostBackend:
    """Persistence backend that stores cost events in the analytics DB."""

    async def persist_batch(
        self,
        events: list[CostEvent],
        pricing: PricingConfig,
    ) -> None:
        """Persist cost events to analytics database."""
        from atp.analytics.database import get_analytics_database
        from atp.analytics.models import CostRecord
        from atp.analytics.repository import CostRepository

        db = get_analytics_database()
        records = []

        for event in events:
            cost = pricing.calculate(
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


async def get_cost_tracker_with_db() -> CostTracker:
    """Get a cost tracker with analytics DB persistence enabled."""
    tracker = await get_cost_tracker()
    if tracker._backend is None:
        tracker.set_backend(AnalyticsCostBackend())
    return tracker


__all__ = [
    "AnalyticsCostBackend",
    "CostEvent",
    "CostPersistenceBackend",
    "CostTracker",
    "ModelPricing",
    "PricingConfig",
    "get_cost_tracker",
    "get_cost_tracker_with_db",
    "set_cost_tracker",
    "shutdown_cost_tracker",
]
