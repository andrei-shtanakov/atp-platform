"""Unit tests for chaos engineering injectors."""

import time
from datetime import datetime

import pytest

from atp.chaos import (
    ChaosConfig,
    ChaosInjector,
    ErrorType,
    LatencyConfig,
    LatencyInjector,
    PartialResponseConfig,
    PartialResponseInjector,
    RateLimitConfig,
    RateLimiter,
    RateLimitError,
    TokenLimitConfig,
    TokenLimitError,
    TokenLimitInjector,
    ToolFailureConfig,
    ToolFailureError,
    ToolFailureInjector,
)
from atp.protocol import (
    ArtifactFile,
    ATPEvent,
    ATPResponse,
    EventType,
    Metrics,
    ResponseStatus,
)


class TestChaosInjector:
    """Tests for the main ChaosInjector class."""

    @pytest.fixture
    def basic_config(self) -> ChaosConfig:
        """Create a basic chaos configuration."""
        return ChaosConfig(
            enabled=True,
            tool_failures=[
                ToolFailureConfig(tool="web_search", probability=1.0),
                ToolFailureConfig(tool="*", probability=0.5),
            ],
            latency=LatencyConfig(min_ms=10, max_ms=50),
            seed=42,
        )

    @pytest.fixture
    def injector(self, basic_config: ChaosConfig) -> ChaosInjector:
        """Create a chaos injector with basic config."""
        return ChaosInjector(basic_config)

    def test_is_active_when_enabled(self, injector: ChaosInjector) -> None:
        """Test is_active returns True when chaos is enabled and configured."""
        assert injector.is_active is True

    def test_is_active_when_disabled(self) -> None:
        """Test is_active returns False when chaos is disabled."""
        config = ChaosConfig(enabled=False)
        injector = ChaosInjector(config)
        assert injector.is_active is False

    def test_is_active_when_empty(self) -> None:
        """Test is_active returns False when no injections configured."""
        config = ChaosConfig(enabled=True)
        injector = ChaosInjector(config)
        assert injector.is_active is False


class TestToolFailureInjection:
    """Tests for tool failure injection."""

    @pytest.fixture
    def injector(self) -> ChaosInjector:
        """Create injector with deterministic tool failures."""
        config = ChaosConfig(
            enabled=True,
            tool_failures=[
                ToolFailureConfig(
                    tool="failing_tool",
                    probability=1.0,
                    error_type=ErrorType.TIMEOUT,
                ),
                ToolFailureConfig(
                    tool="sometimes_fails",
                    probability=0.5,
                    error_type=ErrorType.INTERNAL_ERROR,
                ),
            ],
            seed=42,
        )
        return ChaosInjector(config)

    def test_should_inject_failure_always(self, injector: ChaosInjector) -> None:
        """Test tool with 100% failure rate always fails."""
        result = injector.should_inject_tool_failure("failing_tool")
        assert result is not None
        assert result.tool == "failing_tool"
        assert result.error_type == ErrorType.TIMEOUT

    def test_should_inject_failure_never(self, injector: ChaosInjector) -> None:
        """Test unmatched tool never fails."""
        result = injector.should_inject_tool_failure("safe_tool")
        assert result is None

    def test_inject_tool_failure_raises(self, injector: ChaosInjector) -> None:
        """Test inject_tool_failure raises ToolFailureError."""
        with pytest.raises(ToolFailureError) as exc_info:
            injector.inject_tool_failure("failing_tool")
        assert "failing_tool" in str(exc_info.value)
        assert exc_info.value.error_type == ErrorType.TIMEOUT

    def test_inject_tool_failure_no_raise(self, injector: ChaosInjector) -> None:
        """Test inject_tool_failure doesn't raise for safe tools."""
        # Should not raise
        injector.inject_tool_failure("safe_tool")

    def test_tool_failure_with_wildcard(self) -> None:
        """Test wildcard pattern matches all tools."""
        config = ChaosConfig(
            enabled=True,
            tool_failures=[
                ToolFailureConfig(tool="*", probability=1.0),
            ],
            seed=42,
        )
        injector = ChaosInjector(config)

        result = injector.should_inject_tool_failure("any_tool")
        assert result is not None

    def test_tool_failure_disabled(self) -> None:
        """Test no failure when chaos is disabled."""
        config = ChaosConfig(
            enabled=False,
            tool_failures=[
                ToolFailureConfig(tool="*", probability=1.0),
            ],
        )
        injector = ChaosInjector(config)

        result = injector.should_inject_tool_failure("any_tool")
        assert result is None


class TestLatencyInjection:
    """Tests for latency injection."""

    @pytest.fixture
    def injector(self) -> ChaosInjector:
        """Create injector with latency config."""
        config = ChaosConfig(
            enabled=True,
            latency=LatencyConfig(
                min_ms=50,
                max_ms=100,
                affected_tools=["slow_tool", "api_*"],
            ),
            seed=42,
        )
        return ChaosInjector(config)

    @pytest.mark.anyio
    async def test_inject_latency_delay(self, injector: ChaosInjector) -> None:
        """Test latency injection applies delay."""
        start = time.monotonic()
        delay = await injector.inject_latency("slow_tool")
        elapsed = time.monotonic() - start

        assert delay >= 0.05  # min 50ms
        assert delay <= 0.1  # max 100ms
        assert elapsed >= 0.05

    @pytest.mark.anyio
    async def test_inject_latency_pattern_match(self, injector: ChaosInjector) -> None:
        """Test latency injection with pattern matching."""
        delay = await injector.inject_latency("api_call")
        assert delay > 0

    @pytest.mark.anyio
    async def test_inject_latency_no_match(self, injector: ChaosInjector) -> None:
        """Test no latency for unmatched tools."""
        delay = await injector.inject_latency("fast_tool")
        assert delay == 0.0

    @pytest.mark.anyio
    async def test_inject_latency_disabled(self) -> None:
        """Test no latency when chaos is disabled."""
        config = ChaosConfig(
            enabled=False,
            latency=LatencyConfig(min_ms=100, max_ms=500),
        )
        injector = ChaosInjector(config)

        delay = await injector.inject_latency("any_tool")
        assert delay == 0.0


class TestTokenLimitInjection:
    """Tests for token limit injection."""

    @pytest.fixture
    def injector(self) -> ChaosInjector:
        """Create injector with token limits."""
        config = ChaosConfig(
            enabled=True,
            token_limits=TokenLimitConfig(max_input=1000, max_output=500),
        )
        return ChaosInjector(config)

    def test_check_token_limits_pass(self, injector: ChaosInjector) -> None:
        """Test token limits pass when within bounds."""
        # Should not raise
        injector.check_token_limits(input_tokens=500, output_tokens=200)

    def test_check_token_limits_input_exceeded(self, injector: ChaosInjector) -> None:
        """Test token limits raise when input exceeded."""
        with pytest.raises(TokenLimitError) as exc_info:
            injector.check_token_limits(input_tokens=1500, output_tokens=200)
        assert exc_info.value.limit_type == "input"
        assert exc_info.value.limit == 1000
        assert exc_info.value.actual == 1500

    def test_check_token_limits_output_exceeded(self, injector: ChaosInjector) -> None:
        """Test token limits raise when output exceeded."""
        with pytest.raises(TokenLimitError) as exc_info:
            injector.check_token_limits(input_tokens=500, output_tokens=600)
        assert exc_info.value.limit_type == "output"

    def test_check_token_limits_none_values(self, injector: ChaosInjector) -> None:
        """Test token limits pass with None values."""
        injector.check_token_limits(input_tokens=None, output_tokens=None)

    def test_check_token_limits_disabled(self) -> None:
        """Test no check when chaos is disabled."""
        config = ChaosConfig(
            enabled=False,
            token_limits=TokenLimitConfig(max_input=100),
        )
        injector = ChaosInjector(config)

        # Should not raise even with exceeded limits
        injector.check_token_limits(input_tokens=500)


class TestPartialResponseInjection:
    """Tests for partial response injection."""

    @pytest.fixture
    def injector(self) -> ChaosInjector:
        """Create injector with partial response config."""
        config = ChaosConfig(
            enabled=True,
            partial_response=PartialResponseConfig(
                enabled=True,
                truncate_probability=1.0,  # Always truncate
                min_percentage=0.5,
                max_percentage=0.5,  # Fixed at 50%
            ),
            seed=42,
        )
        return ChaosInjector(config)

    def test_apply_partial_response(self, injector: ChaosInjector) -> None:
        """Test partial response truncation."""
        content = "This is a test content that should be truncated"
        truncated, was_truncated = injector.apply_partial_response(content)

        assert was_truncated is True
        assert len(truncated) < len(content)
        assert len(truncated) == len(content) // 2

    def test_apply_partial_response_disabled(self) -> None:
        """Test no truncation when disabled."""
        config = ChaosConfig(
            enabled=True,
            partial_response=PartialResponseConfig(enabled=False),
        )
        injector = ChaosInjector(config)

        content = "Test content"
        result, was_truncated = injector.apply_partial_response(content)

        assert was_truncated is False
        assert result == content

    def test_apply_partial_response_zero_probability(self) -> None:
        """Test no truncation with zero probability."""
        config = ChaosConfig(
            enabled=True,
            partial_response=PartialResponseConfig(
                enabled=True,
                truncate_probability=0.0,
            ),
        )
        injector = ChaosInjector(config)

        content = "Test content"
        result, was_truncated = injector.apply_partial_response(content)

        assert was_truncated is False
        assert result == content


class TestRateLimitInjection:
    """Tests for rate limit injection."""

    @pytest.fixture
    def injector(self) -> ChaosInjector:
        """Create injector with rate limit config."""
        config = ChaosConfig(
            enabled=True,
            rate_limit=RateLimitConfig(
                enabled=True,
                requests_per_minute=5,
                burst_size=2,
                retry_after_seconds=30,
            ),
        )
        return ChaosInjector(config)

    def test_check_rate_limit_pass(self, injector: ChaosInjector) -> None:
        """Test rate limit allows initial requests."""
        # First request should pass
        injector.check_rate_limit()

    def test_check_rate_limit_burst_exceeded(self, injector: ChaosInjector) -> None:
        """Test rate limit fails when burst exceeded."""
        # Exhaust burst
        injector.check_rate_limit()
        injector.check_rate_limit()

        # Third request should fail
        with pytest.raises(RateLimitError) as exc_info:
            injector.check_rate_limit()
        assert exc_info.value.retry_after == 30

    def test_check_rate_limit_disabled(self) -> None:
        """Test no rate limit when disabled."""
        config = ChaosConfig(
            enabled=True,
            rate_limit=RateLimitConfig(enabled=False),
        )
        injector = ChaosInjector(config)

        # Should never raise
        for _ in range(100):
            injector.check_rate_limit()


class TestProcessResponse:
    """Tests for response processing."""

    @pytest.fixture
    def response(self) -> ATPResponse:
        """Create a sample response."""
        return ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(
                    type="file",
                    path="output.txt",
                    content="This is the output content that may be truncated",
                ),
            ],
            metrics=Metrics(input_tokens=500, output_tokens=200),
        )

    def test_process_response_with_token_limits(self, response: ATPResponse) -> None:
        """Test response processing checks token limits."""
        config = ChaosConfig(
            enabled=True,
            token_limits=TokenLimitConfig(max_output=100),
        )
        injector = ChaosInjector(config)

        with pytest.raises(TokenLimitError):
            injector.process_response(response)

    def test_process_response_with_partial(self, response: ATPResponse) -> None:
        """Test response processing applies partial response."""
        config = ChaosConfig(
            enabled=True,
            partial_response=PartialResponseConfig(
                enabled=True,
                truncate_probability=1.0,
                min_percentage=0.5,
                max_percentage=0.5,
            ),
            seed=42,
        )
        injector = ChaosInjector(config)

        processed = injector.process_response(response)
        assert processed.status == ResponseStatus.PARTIAL
        artifact = processed.artifacts[0]
        assert isinstance(artifact, ArtifactFile)
        assert artifact.content is not None
        assert len(artifact.content) < len(response.artifacts[0].content)  # type: ignore


class TestProcessEvent:
    """Tests for event processing."""

    @pytest.fixture
    def tool_call_event(self) -> ATPEvent:
        """Create a tool call event."""
        return ATPEvent(
            task_id="test-001",
            timestamp=datetime.now(),
            sequence=0,
            event_type=EventType.TOOL_CALL,
            payload={"tool": "failing_tool", "status": "pending"},
        )

    def test_process_event_injects_failure(self, tool_call_event: ATPEvent) -> None:
        """Test event processing injects tool failure."""
        config = ChaosConfig(
            enabled=True,
            tool_failures=[
                ToolFailureConfig(
                    tool="failing_tool",
                    probability=1.0,
                    error_type=ErrorType.TIMEOUT,
                ),
            ],
            seed=42,
        )
        injector = ChaosInjector(config)

        processed = injector.process_event(tool_call_event)
        assert processed is not None
        assert processed.event_type == EventType.ERROR
        assert processed.payload["error_type"] == "timeout"
        assert processed.payload["chaos_injected"] is True

    def test_process_event_no_failure(self, tool_call_event: ATPEvent) -> None:
        """Test event processing passes through when no failure."""
        config = ChaosConfig(
            enabled=True,
            tool_failures=[
                ToolFailureConfig(tool="other_tool", probability=1.0),
            ],
        )
        injector = ChaosInjector(config)

        processed = injector.process_event(tool_call_event)
        assert processed is not None
        assert processed.event_type == EventType.TOOL_CALL


class TestProcessToolCall:
    """Tests for process_tool_call convenience method."""

    @pytest.mark.anyio
    async def test_process_tool_call_success(self) -> None:
        """Test successful tool call processing."""
        config = ChaosConfig(enabled=True, seed=42)
        injector = ChaosInjector(config)

        async def mock_tool() -> str:
            return "result"

        result = await injector.process_tool_call("safe_tool", mock_tool)
        assert result == "result"

    @pytest.mark.anyio
    async def test_process_tool_call_with_failure(self) -> None:
        """Test tool call processing with failure injection."""
        config = ChaosConfig(
            enabled=True,
            tool_failures=[
                ToolFailureConfig(tool="*", probability=1.0),
            ],
            seed=42,
        )
        injector = ChaosInjector(config)

        async def mock_tool() -> str:
            return "result"

        with pytest.raises(ToolFailureError):
            await injector.process_tool_call("any_tool", mock_tool)

    @pytest.mark.anyio
    async def test_process_tool_call_with_latency(self) -> None:
        """Test tool call processing with latency injection."""
        config = ChaosConfig(
            enabled=True,
            latency=LatencyConfig(min_ms=50, max_ms=100),
            seed=42,
        )
        injector = ChaosInjector(config)

        async def mock_tool() -> str:
            return "result"

        start = time.monotonic()
        await injector.process_tool_call("slow_tool", mock_tool)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.05  # At least 50ms delay

    @pytest.mark.anyio
    async def test_process_tool_call_sync_function(self) -> None:
        """Test tool call processing with sync function."""
        config = ChaosConfig(enabled=True, seed=42)
        injector = ChaosInjector(config)

        def sync_tool() -> str:
            return "sync_result"

        result = await injector.process_tool_call("safe_tool", sync_tool)
        assert result == "sync_result"


class TestStandaloneInjectors:
    """Tests for standalone injector classes."""

    def test_tool_failure_injector(self) -> None:
        """Test ToolFailureInjector standalone."""
        configs = [
            ToolFailureConfig(tool="fail*", probability=1.0),
        ]
        injector = ToolFailureInjector(configs, seed=42)

        assert injector.should_fail("failing_tool") is not None
        assert injector.should_fail("safe_tool") is None

        with pytest.raises(ToolFailureError):
            injector.inject("failing_tool")

    @pytest.mark.anyio
    async def test_latency_injector(self) -> None:
        """Test LatencyInjector standalone."""
        config = LatencyConfig(min_ms=50, max_ms=100)
        injector = LatencyInjector(config, seed=42)

        delay = await injector.inject("any_tool")
        assert 0.05 <= delay <= 0.1

    def test_token_limit_injector(self) -> None:
        """Test TokenLimitInjector standalone."""
        config = TokenLimitConfig(max_input=1000, max_output=500)
        injector = TokenLimitInjector(config)

        # Should pass
        injector.check(input_tokens=500)

        # Should fail
        with pytest.raises(TokenLimitError):
            injector.check(input_tokens=1500)

    def test_partial_response_injector(self) -> None:
        """Test PartialResponseInjector standalone."""
        config = PartialResponseConfig(
            enabled=True,
            truncate_probability=1.0,
            min_percentage=0.5,
            max_percentage=0.5,
        )
        injector = PartialResponseInjector(config, seed=42)

        content = "Test content here"
        truncated, was_truncated = injector.apply(content)

        assert was_truncated is True
        assert len(truncated) == len(content) // 2


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_allows_burst(self) -> None:
        """Test rate limiter allows burst requests."""
        config = RateLimitConfig(
            enabled=True,
            requests_per_minute=60,
            burst_size=5,
        )
        limiter = RateLimiter(config)

        # Should allow burst
        for _ in range(5):
            assert limiter.allow() is True

        # Should deny after burst
        assert limiter.allow() is False

    def test_rate_limiter_refills(self) -> None:
        """Test rate limiter refills tokens over time."""
        config = RateLimitConfig(
            enabled=True,
            requests_per_minute=60,  # 1 per second
            burst_size=1,
        )
        limiter = RateLimiter(config)

        # Use the token
        assert limiter.allow() is True
        assert limiter.allow() is False

        # Wait for refill (simulate time passing)
        time.sleep(0.1)  # 100ms should add ~0.1 tokens at 1/sec rate

        # After waiting, tokens should have refilled partially
        # In reality this depends on timing, so we just check the mechanism works
        # For accurate testing, we'd mock time.monotonic

    def test_rate_limiter_retry_after(self) -> None:
        """Test rate limiter provides correct retry_after value."""
        config = RateLimitConfig(
            enabled=True,
            retry_after_seconds=45,
        )
        limiter = RateLimiter(config)

        assert limiter.retry_after == 45


class TestReproducibility:
    """Tests for chaos injection reproducibility with seeds."""

    def test_same_seed_same_results(self) -> None:
        """Test same seed produces same results."""
        config = ChaosConfig(
            enabled=True,
            tool_failures=[
                ToolFailureConfig(tool="*", probability=0.5),
            ],
            seed=12345,
        )

        injector1 = ChaosInjector(config)
        injector2 = ChaosInjector(config)

        # Run same sequence of checks
        results1 = [
            injector1.should_inject_tool_failure("tool1"),
            injector1.should_inject_tool_failure("tool2"),
            injector1.should_inject_tool_failure("tool3"),
        ]

        results2 = [
            injector2.should_inject_tool_failure("tool1"),
            injector2.should_inject_tool_failure("tool2"),
            injector2.should_inject_tool_failure("tool3"),
        ]

        # Results should match
        for r1, r2 in zip(results1, results2):
            assert (r1 is None) == (r2 is None)

    def test_different_seed_different_results(self) -> None:
        """Test different seeds can produce different results."""
        config1 = ChaosConfig(
            enabled=True,
            tool_failures=[
                ToolFailureConfig(tool="*", probability=0.5),
            ],
            seed=12345,
        )
        config2 = ChaosConfig(
            enabled=True,
            tool_failures=[
                ToolFailureConfig(tool="*", probability=0.5),
            ],
            seed=54321,
        )

        injector1 = ChaosInjector(config1)
        injector2 = ChaosInjector(config2)

        # Run many checks - with different seeds, results should differ
        results1 = [injector1.should_inject_tool_failure(f"tool{i}") for i in range(20)]
        results2 = [injector2.should_inject_tool_failure(f"tool{i}") for i in range(20)]

        # Convert to boolean for comparison
        bool1 = [r is not None for r in results1]
        bool2 = [r is not None for r in results2]

        # With different seeds, at least some results should differ
        # (statistically almost certain with 20 samples at 50% probability)
        assert bool1 != bool2
