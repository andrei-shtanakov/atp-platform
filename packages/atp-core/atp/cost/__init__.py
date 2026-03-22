"""ATP Cost tracking module.

Standalone cost tracking for LLM operations, decoupled from analytics DB.
This module can be used without the dashboard/analytics packages.

Usage:
    from atp.cost import CostTracker, CostEvent, PricingConfig

    tracker = CostTracker(pricing=PricingConfig.default())
    await tracker.start()
    await tracker.track(CostEvent(
        timestamp=datetime.now(),
        provider="anthropic",
        model="claude-3-sonnet",
        input_tokens=1000,
        output_tokens=500,
    ))
"""

from atp.cost.models import CostEvent, ModelPricing, PricingConfig
from atp.cost.tracker import (
    CostPersistenceBackend,
    CostTracker,
    get_cost_tracker,
    set_cost_tracker,
    shutdown_cost_tracker,
)

__all__ = [
    "CostEvent",
    "CostPersistenceBackend",
    "CostTracker",
    "ModelPricing",
    "PricingConfig",
    "get_cost_tracker",
    "set_cost_tracker",
    "shutdown_cost_tracker",
]
