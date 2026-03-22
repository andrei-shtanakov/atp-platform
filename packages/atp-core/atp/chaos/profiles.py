"""Predefined chaos profiles for common test scenarios."""

from .models import (
    ChaosConfig,
    ChaosProfile,
    ErrorType,
    LatencyConfig,
    PartialResponseConfig,
    RateLimitConfig,
    TokenLimitConfig,
    ToolFailureConfig,
)


def get_profile(profile: ChaosProfile | str) -> ChaosConfig:
    """Get a predefined chaos configuration by profile name.

    Args:
        profile: Profile name or ChaosProfile enum value.

    Returns:
        ChaosConfig with the predefined settings for the profile.

    Raises:
        ValueError: If the profile name is unknown.
    """
    if isinstance(profile, str):
        try:
            profile = ChaosProfile(profile)
        except ValueError as e:
            valid = [p.value for p in ChaosProfile]
            raise ValueError(
                f"Unknown chaos profile: '{profile}'. Valid profiles: {valid}"
            ) from e

    return _PROFILES[profile]


def list_profiles() -> list[str]:
    """List all available chaos profile names.

    Returns:
        List of profile names.
    """
    return [p.value for p in ChaosProfile]


def get_profile_description(profile: ChaosProfile | str) -> str:
    """Get a human-readable description of a chaos profile.

    Args:
        profile: Profile name or ChaosProfile enum value.

    Returns:
        Description of what the profile does.
    """
    if isinstance(profile, str):
        profile = ChaosProfile(profile)

    return _PROFILE_DESCRIPTIONS[profile]


# Profile descriptions
_PROFILE_DESCRIPTIONS: dict[ChaosProfile, str] = {
    ChaosProfile.NONE: "No chaos injection - normal operation",
    ChaosProfile.LIGHT: "Light chaos: occasional latency (50-200ms), 5% tool failures",
    ChaosProfile.MODERATE: (
        "Moderate chaos: noticeable latency (100-500ms), 15% tool failures, "
        "occasional partial responses"
    ),
    ChaosProfile.HEAVY: (
        "Heavy chaos: high latency (200-2000ms), 30% tool failures, "
        "frequent partial responses, rate limiting"
    ),
    ChaosProfile.HIGH_LATENCY: (
        "High latency profile: significant delays (500-3000ms) on all operations"
    ),
    ChaosProfile.FLAKY_TOOLS: (
        "Flaky tools profile: 25% chance of tool failures with various error types"
    ),
    ChaosProfile.RATE_LIMITED: (
        "Rate limited profile: strict rate limiting (10 req/min) to test backoff"
    ),
    ChaosProfile.RESOURCE_CONSTRAINED: (
        "Resource constrained profile: low token limits and partial responses"
    ),
}


# Predefined profile configurations
_PROFILES: dict[ChaosProfile, ChaosConfig] = {
    ChaosProfile.NONE: ChaosConfig(enabled=False),
    ChaosProfile.LIGHT: ChaosConfig(
        enabled=True,
        tool_failures=[
            ToolFailureConfig(
                tool="*",
                probability=0.05,
                error_type=ErrorType.TIMEOUT,
            ),
        ],
        latency=LatencyConfig(
            min_ms=50,
            max_ms=200,
            affected_tools=["*"],
        ),
    ),
    ChaosProfile.MODERATE: ChaosConfig(
        enabled=True,
        tool_failures=[
            ToolFailureConfig(
                tool="*",
                probability=0.15,
                error_type=ErrorType.INTERNAL_ERROR,
            ),
        ],
        latency=LatencyConfig(
            min_ms=100,
            max_ms=500,
            affected_tools=["*"],
        ),
        partial_response=PartialResponseConfig(
            enabled=True,
            truncate_probability=0.1,
            min_percentage=0.7,
            max_percentage=0.95,
        ),
    ),
    ChaosProfile.HEAVY: ChaosConfig(
        enabled=True,
        tool_failures=[
            ToolFailureConfig(
                tool="*",
                probability=0.30,
                error_type=ErrorType.INTERNAL_ERROR,
            ),
        ],
        latency=LatencyConfig(
            min_ms=200,
            max_ms=2000,
            affected_tools=["*"],
        ),
        partial_response=PartialResponseConfig(
            enabled=True,
            truncate_probability=0.25,
            min_percentage=0.5,
            max_percentage=0.9,
        ),
        rate_limit=RateLimitConfig(
            enabled=True,
            requests_per_minute=30,
            burst_size=5,
            retry_after_seconds=30,
        ),
    ),
    ChaosProfile.HIGH_LATENCY: ChaosConfig(
        enabled=True,
        latency=LatencyConfig(
            min_ms=500,
            max_ms=3000,
            affected_tools=["*"],
        ),
    ),
    ChaosProfile.FLAKY_TOOLS: ChaosConfig(
        enabled=True,
        tool_failures=[
            ToolFailureConfig(
                tool="*",
                probability=0.25,
                error_type=ErrorType.INTERNAL_ERROR,
            ),
            ToolFailureConfig(
                tool="web_search",
                probability=0.40,
                error_type=ErrorType.TIMEOUT,
            ),
            ToolFailureConfig(
                tool="api_call",
                probability=0.35,
                error_type=ErrorType.RATE_LIMIT,
            ),
        ],
    ),
    ChaosProfile.RATE_LIMITED: ChaosConfig(
        enabled=True,
        rate_limit=RateLimitConfig(
            enabled=True,
            requests_per_minute=10,
            burst_size=3,
            retry_after_seconds=60,
        ),
    ),
    ChaosProfile.RESOURCE_CONSTRAINED: ChaosConfig(
        enabled=True,
        token_limits=TokenLimitConfig(
            max_input=1000,
            max_output=500,
        ),
        partial_response=PartialResponseConfig(
            enabled=True,
            truncate_probability=0.3,
            min_percentage=0.4,
            max_percentage=0.8,
        ),
    ),
}
