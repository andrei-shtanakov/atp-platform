"""Unit tests for chaos engineering models."""

import pytest
from pydantic import ValidationError

from atp.chaos import (
    ChaosConfig,
    ChaosProfile,
    ErrorType,
    LatencyConfig,
    PartialResponseConfig,
    RateLimitConfig,
    TokenLimitConfig,
    ToolFailureConfig,
)


class TestToolFailureConfig:
    """Tests for ToolFailureConfig model."""

    def test_basic_config(self) -> None:
        """Test basic tool failure configuration."""
        config = ToolFailureConfig(
            tool="web_search",
            probability=0.3,
            error_type=ErrorType.TIMEOUT,
        )
        assert config.tool == "web_search"
        assert config.probability == 0.3
        assert config.error_type == ErrorType.TIMEOUT

    def test_default_values(self) -> None:
        """Test default values for tool failure config."""
        config = ToolFailureConfig(tool="*")
        assert config.probability == 0.0
        assert config.error_type == ErrorType.INTERNAL_ERROR
        assert config.error_message is None

    def test_wildcard_pattern(self) -> None:
        """Test wildcard tool pattern."""
        config = ToolFailureConfig(tool="*", probability=0.5)
        assert config.tool == "*"

    def test_custom_error_message(self) -> None:
        """Test custom error message."""
        config = ToolFailureConfig(
            tool="api_call",
            probability=0.1,
            error_message="Custom failure message",
        )
        assert config.error_message == "Custom failure message"

    def test_probability_validation_min(self) -> None:
        """Test probability minimum validation."""
        with pytest.raises(ValidationError):
            ToolFailureConfig(tool="test", probability=-0.1)

    def test_probability_validation_max(self) -> None:
        """Test probability maximum validation."""
        with pytest.raises(ValidationError):
            ToolFailureConfig(tool="test", probability=1.1)

    def test_empty_tool_validation(self) -> None:
        """Test empty tool name validation."""
        with pytest.raises(ValidationError):
            ToolFailureConfig(tool="", probability=0.1)

    def test_whitespace_tool_validation(self) -> None:
        """Test whitespace-only tool name validation."""
        with pytest.raises(ValidationError):
            ToolFailureConfig(tool="   ", probability=0.1)


class TestLatencyConfig:
    """Tests for LatencyConfig model."""

    def test_basic_config(self) -> None:
        """Test basic latency configuration."""
        config = LatencyConfig(min_ms=100, max_ms=500)
        assert config.min_ms == 100
        assert config.max_ms == 500
        assert config.affected_tools == ["*"]

    def test_default_values(self) -> None:
        """Test default latency values."""
        config = LatencyConfig()
        assert config.min_ms == 0
        assert config.max_ms == 0
        assert config.affected_tools == ["*"]

    def test_custom_affected_tools(self) -> None:
        """Test custom affected tools list."""
        config = LatencyConfig(
            min_ms=50,
            max_ms=200,
            affected_tools=["web_search", "api_call"],
        )
        assert config.affected_tools == ["web_search", "api_call"]

    def test_max_less_than_min_validation(self) -> None:
        """Test max_ms must be >= min_ms."""
        with pytest.raises(ValidationError):
            LatencyConfig(min_ms=500, max_ms=100)

    def test_negative_min_validation(self) -> None:
        """Test negative min_ms validation."""
        with pytest.raises(ValidationError):
            LatencyConfig(min_ms=-1, max_ms=100)


class TestTokenLimitConfig:
    """Tests for TokenLimitConfig model."""

    def test_basic_config(self) -> None:
        """Test basic token limit configuration."""
        config = TokenLimitConfig(max_input=1000, max_output=500)
        assert config.max_input == 1000
        assert config.max_output == 500

    def test_input_only(self) -> None:
        """Test input limit only."""
        config = TokenLimitConfig(max_input=2000)
        assert config.max_input == 2000
        assert config.max_output is None

    def test_output_only(self) -> None:
        """Test output limit only."""
        config = TokenLimitConfig(max_output=1000)
        assert config.max_input is None
        assert config.max_output == 1000

    def test_zero_limit_validation(self) -> None:
        """Test zero limit validation."""
        with pytest.raises(ValidationError):
            TokenLimitConfig(max_input=0)


class TestPartialResponseConfig:
    """Tests for PartialResponseConfig model."""

    def test_basic_config(self) -> None:
        """Test basic partial response configuration."""
        config = PartialResponseConfig(
            enabled=True,
            truncate_probability=0.3,
            min_percentage=0.5,
            max_percentage=0.9,
        )
        assert config.enabled is True
        assert config.truncate_probability == 0.3
        assert config.min_percentage == 0.5
        assert config.max_percentage == 0.9

    def test_default_values(self) -> None:
        """Test default partial response values."""
        config = PartialResponseConfig()
        assert config.enabled is False
        assert config.truncate_probability == 0.0
        assert config.min_percentage == 0.5
        assert config.max_percentage == 0.9

    def test_max_less_than_min_validation(self) -> None:
        """Test max_percentage must be >= min_percentage."""
        with pytest.raises(ValidationError):
            PartialResponseConfig(
                enabled=True,
                min_percentage=0.8,
                max_percentage=0.5,
            )


class TestRateLimitConfig:
    """Tests for RateLimitConfig model."""

    def test_basic_config(self) -> None:
        """Test basic rate limit configuration."""
        config = RateLimitConfig(
            enabled=True,
            requests_per_minute=30,
            burst_size=5,
            retry_after_seconds=60,
        )
        assert config.enabled is True
        assert config.requests_per_minute == 30
        assert config.burst_size == 5
        assert config.retry_after_seconds == 60

    def test_default_values(self) -> None:
        """Test default rate limit values."""
        config = RateLimitConfig()
        assert config.enabled is False
        assert config.requests_per_minute == 60
        assert config.burst_size == 10
        assert config.retry_after_seconds == 60

    def test_zero_requests_validation(self) -> None:
        """Test zero requests per minute validation."""
        with pytest.raises(ValidationError):
            RateLimitConfig(requests_per_minute=0)


class TestChaosConfig:
    """Tests for ChaosConfig model."""

    def test_empty_config(self) -> None:
        """Test empty chaos configuration."""
        config = ChaosConfig()
        assert config.enabled is True
        assert config.tool_failures == []
        assert config.latency is None
        assert config.token_limits is None
        assert config.partial_response is None
        assert config.rate_limit is None

    def test_is_empty_no_injections(self) -> None:
        """Test is_empty when no injections configured."""
        config = ChaosConfig(enabled=True)
        assert config.is_empty() is True

    def test_is_empty_disabled(self) -> None:
        """Test is_empty when disabled."""
        config = ChaosConfig(
            enabled=False,
            tool_failures=[ToolFailureConfig(tool="*", probability=0.5)],
        )
        assert config.is_empty() is True

    def test_is_empty_with_tool_failures(self) -> None:
        """Test is_empty with tool failures configured."""
        config = ChaosConfig(
            enabled=True,
            tool_failures=[ToolFailureConfig(tool="*", probability=0.5)],
        )
        assert config.is_empty() is False

    def test_is_empty_with_latency(self) -> None:
        """Test is_empty with latency configured."""
        config = ChaosConfig(
            enabled=True,
            latency=LatencyConfig(min_ms=100, max_ms=500),
        )
        assert config.is_empty() is False

    def test_is_empty_with_zero_latency(self) -> None:
        """Test is_empty with zero latency."""
        config = ChaosConfig(
            enabled=True,
            latency=LatencyConfig(min_ms=0, max_ms=0),
        )
        assert config.is_empty() is True

    def test_is_empty_with_token_limits(self) -> None:
        """Test is_empty with token limits configured."""
        config = ChaosConfig(
            enabled=True,
            token_limits=TokenLimitConfig(max_input=1000),
        )
        assert config.is_empty() is False

    def test_is_empty_with_partial_response(self) -> None:
        """Test is_empty with partial response configured."""
        config = ChaosConfig(
            enabled=True,
            partial_response=PartialResponseConfig(enabled=True),
        )
        assert config.is_empty() is False

    def test_is_empty_with_rate_limit(self) -> None:
        """Test is_empty with rate limit configured."""
        config = ChaosConfig(
            enabled=True,
            rate_limit=RateLimitConfig(enabled=True),
        )
        assert config.is_empty() is False

    def test_full_config(self) -> None:
        """Test full chaos configuration."""
        config = ChaosConfig(
            enabled=True,
            tool_failures=[
                ToolFailureConfig(tool="web_search", probability=0.3),
                ToolFailureConfig(tool="*", probability=0.1),
            ],
            latency=LatencyConfig(min_ms=100, max_ms=500),
            token_limits=TokenLimitConfig(max_input=1000, max_output=500),
            partial_response=PartialResponseConfig(
                enabled=True, truncate_probability=0.2
            ),
            rate_limit=RateLimitConfig(enabled=True, requests_per_minute=30),
            seed=42,
        )
        assert config.is_empty() is False
        assert len(config.tool_failures) == 2
        assert config.seed == 42


class TestChaosProfile:
    """Tests for ChaosProfile enum."""

    def test_all_profiles_exist(self) -> None:
        """Test all expected profiles exist."""
        expected = [
            "none",
            "light",
            "moderate",
            "heavy",
            "high_latency",
            "flaky_tools",
            "rate_limited",
            "resource_constrained",
        ]
        actual = [p.value for p in ChaosProfile]
        assert set(expected) == set(actual)

    def test_profile_from_string(self) -> None:
        """Test creating profile from string."""
        profile = ChaosProfile("light")
        assert profile == ChaosProfile.LIGHT

    def test_invalid_profile(self) -> None:
        """Test invalid profile name."""
        with pytest.raises(ValueError):
            ChaosProfile("invalid_profile")
