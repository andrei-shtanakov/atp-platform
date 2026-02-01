"""Chaos engineering module for testing agent resilience.

This module provides fault injection capabilities to test how agents handle
various failure scenarios:

- Tool failures (timeouts, connection errors, rate limits)
- Latency injection (configurable delays)
- Token limit simulation
- Partial response simulation
- Rate limit simulation

Example usage:

    from atp.chaos import ChaosInjector, ChaosConfig, get_profile, ChaosProfile

    # Using a predefined profile
    config = get_profile(ChaosProfile.MODERATE)
    injector = ChaosInjector(config)

    # Custom configuration
    config = ChaosConfig(
        enabled=True,
        tool_failures=[
            ToolFailureConfig(
                tool="web_search", probability=0.3, error_type=ErrorType.TIMEOUT
            ),
        ],
        latency=LatencyConfig(min_ms=100, max_ms=500),
    )
    injector = ChaosInjector(config)

    # Apply chaos during tool execution
    await injector.inject_latency("web_search")
    injector.inject_tool_failure("web_search")  # Raises if failure triggered

    # Process responses
    response = injector.process_response(response)
"""

from .injectors import (
    ChaosError,
    ChaosInjector,
    LatencyInjector,
    PartialResponseInjector,
    RateLimiter,
    RateLimitError,
    TokenLimitError,
    TokenLimitInjector,
    ToolFailureError,
    ToolFailureInjector,
)
from .models import (
    DEFAULT_ERROR_MESSAGES,
    ChaosConfig,
    ChaosProfile,
    ErrorType,
    LatencyConfig,
    PartialResponseConfig,
    RateLimitConfig,
    TokenLimitConfig,
    ToolFailureConfig,
)
from .profiles import get_profile, get_profile_description, list_profiles

__all__ = [
    # Main injector
    "ChaosInjector",
    # Configuration models
    "ChaosConfig",
    "ChaosProfile",
    "ErrorType",
    "ToolFailureConfig",
    "LatencyConfig",
    "TokenLimitConfig",
    "PartialResponseConfig",
    "RateLimitConfig",
    "DEFAULT_ERROR_MESSAGES",
    # Exceptions
    "ChaosError",
    "ToolFailureError",
    "RateLimitError",
    "TokenLimitError",
    # Standalone injectors
    "LatencyInjector",
    "ToolFailureInjector",
    "TokenLimitInjector",
    "PartialResponseInjector",
    "RateLimiter",
    # Profile utilities
    "get_profile",
    "get_profile_description",
    "list_profiles",
]
