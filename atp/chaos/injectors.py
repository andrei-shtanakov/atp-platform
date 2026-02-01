"""Chaos engineering fault injectors for testing agent resilience."""

import asyncio
import fnmatch
import logging
import random
import time
from collections import deque
from typing import Any

from atp.protocol import ATPEvent, ATPResponse, EventType, ResponseStatus

from .models import (
    DEFAULT_ERROR_MESSAGES,
    ChaosConfig,
    ErrorType,
    LatencyConfig,
    PartialResponseConfig,
    RateLimitConfig,
    TokenLimitConfig,
    ToolFailureConfig,
)

logger = logging.getLogger(__name__)


class ChaosError(Exception):
    """Base exception for chaos-induced errors."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        recoverable: bool = True,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.recoverable = recoverable


class ToolFailureError(ChaosError):
    """Error raised when tool failure is injected."""

    def __init__(
        self,
        tool_name: str,
        error_type: ErrorType,
        message: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        msg = message or DEFAULT_ERROR_MESSAGES.get(error_type, "Tool execution failed")
        super().__init__(
            f"Chaos-injected failure for tool '{tool_name}': {msg}",
            error_type,
            recoverable=error_type != ErrorType.PERMISSION_DENIED,
        )


class RateLimitError(ChaosError):
    """Error raised when rate limit is exceeded."""

    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after} seconds.",
            ErrorType.RATE_LIMIT,
            recoverable=True,
        )


class TokenLimitError(ChaosError):
    """Error raised when token limits are exceeded."""

    def __init__(self, limit_type: str, limit: int, actual: int) -> None:
        self.limit_type = limit_type
        self.limit = limit
        self.actual = actual
        super().__init__(
            f"Token limit exceeded: {limit_type} tokens ({actual}) > limit ({limit})",
            ErrorType.VALIDATION_ERROR,
            recoverable=False,
        )


class ChaosInjector:
    """Main chaos injection engine that coordinates all fault injectors."""

    def __init__(self, config: ChaosConfig, seed: int | None = None) -> None:
        """Initialize the chaos injector.

        Args:
            config: Chaos configuration specifying what faults to inject.
            seed: Random seed for reproducibility. Overrides config seed if provided.
        """
        self.config = config
        self._rng = random.Random(seed if seed is not None else config.seed)
        self._rate_limiter: RateLimiter | None = None

        if config.rate_limit and config.rate_limit.enabled:
            self._rate_limiter = RateLimiter(config.rate_limit, self._rng)

    @property
    def is_active(self) -> bool:
        """Check if chaos injection is enabled and has active injections."""
        return self.config.enabled and not self.config.is_empty()

    def should_inject_tool_failure(self, tool_name: str) -> ToolFailureConfig | None:
        """Check if a tool failure should be injected for the given tool.

        Args:
            tool_name: Name of the tool being called.

        Returns:
            ToolFailureConfig if failure should be injected, None otherwise.
        """
        if not self.config.enabled:
            return None

        for failure_config in self.config.tool_failures:
            if self._matches_pattern(tool_name, failure_config.tool):
                if self._rng.random() < failure_config.probability:
                    logger.debug(
                        "Chaos: injecting %s failure for tool '%s'",
                        failure_config.error_type.value,
                        tool_name,
                    )
                    return failure_config

        return None

    def inject_tool_failure(self, tool_name: str) -> None:
        """Raise an exception if tool failure should be injected.

        Args:
            tool_name: Name of the tool being called.

        Raises:
            ToolFailureError: If failure should be injected.
        """
        failure_config = self.should_inject_tool_failure(tool_name)
        if failure_config:
            raise ToolFailureError(
                tool_name,
                failure_config.error_type,
                failure_config.error_message,
            )

    async def inject_latency(self, tool_name: str | None = None) -> float:
        """Inject latency delay if configured.

        Args:
            tool_name: Optional tool name to check against affected_tools pattern.

        Returns:
            The actual delay in seconds that was applied.
        """
        if not self.config.enabled or not self.config.latency:
            return 0.0

        latency_config = self.config.latency

        if latency_config.max_ms <= 0:
            return 0.0

        # Check if this tool should be affected
        if tool_name:
            affected = any(
                self._matches_pattern(tool_name, pattern)
                for pattern in latency_config.affected_tools
            )
            if not affected:
                return 0.0

        # Calculate random delay within range
        delay_ms = self._rng.randint(latency_config.min_ms, latency_config.max_ms)
        delay_seconds = delay_ms / 1000.0

        logger.debug("Chaos: injecting %dms latency", delay_ms)
        await asyncio.sleep(delay_seconds)

        return delay_seconds

    def check_token_limits(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Check if token counts exceed configured limits.

        Args:
            input_tokens: Number of input tokens to check.
            output_tokens: Number of output tokens to check.

        Raises:
            TokenLimitError: If any limit is exceeded.
        """
        if not self.config.enabled or not self.config.token_limits:
            return

        limits = self.config.token_limits

        if limits.max_input is not None and input_tokens is not None:
            if input_tokens > limits.max_input:
                logger.debug(
                    "Chaos: input token limit exceeded (%d > %d)",
                    input_tokens,
                    limits.max_input,
                )
                raise TokenLimitError("input", limits.max_input, input_tokens)

        if limits.max_output is not None and output_tokens is not None:
            if output_tokens > limits.max_output:
                logger.debug(
                    "Chaos: output token limit exceeded (%d > %d)",
                    output_tokens,
                    limits.max_output,
                )
                raise TokenLimitError("output", limits.max_output, output_tokens)

    def apply_partial_response(self, content: str) -> tuple[str, bool]:
        """Potentially truncate response content.

        Args:
            content: Original response content.

        Returns:
            Tuple of (potentially truncated content, whether truncation occurred).
        """
        if not self.config.enabled or not self.config.partial_response:
            return content, False

        partial_config = self.config.partial_response

        if not partial_config.enabled:
            return content, False

        if self._rng.random() >= partial_config.truncate_probability:
            return content, False

        # Calculate truncation percentage
        keep_percentage = self._rng.uniform(
            partial_config.min_percentage,
            partial_config.max_percentage,
        )

        # Truncate content
        original_length = len(content)
        truncate_at = int(original_length * keep_percentage)
        truncated = content[:truncate_at]

        logger.debug(
            "Chaos: truncating response from %d to %d chars (%.1f%%)",
            original_length,
            truncate_at,
            keep_percentage * 100,
        )

        return truncated, True

    def check_rate_limit(self) -> None:
        """Check if rate limit is exceeded.

        Raises:
            RateLimitError: If rate limit is exceeded.
        """
        if not self._rate_limiter:
            return

        if not self._rate_limiter.allow():
            raise RateLimitError(self._rate_limiter.retry_after)

    async def process_tool_call(
        self,
        tool_name: str,
        execute_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Process a tool call with chaos injection.

        This is a convenience method that applies all relevant chaos injections
        around a tool call.

        Args:
            tool_name: Name of the tool being called.
            execute_fn: The actual tool execution function.
            *args: Arguments to pass to execute_fn.
            **kwargs: Keyword arguments to pass to execute_fn.

        Returns:
            Result from execute_fn, potentially modified.

        Raises:
            ChaosError: If any chaos-injected error occurs.
        """
        # Check rate limit first
        self.check_rate_limit()

        # Inject latency before execution
        await self.inject_latency(tool_name)

        # Check for tool failure injection
        self.inject_tool_failure(tool_name)

        # Execute the actual tool
        if asyncio.iscoroutinefunction(execute_fn):
            result = await execute_fn(*args, **kwargs)
        else:
            result = execute_fn(*args, **kwargs)

        return result

    def process_response(self, response: ATPResponse) -> ATPResponse:
        """Process an ATPResponse with chaos injections.

        Applies token limit checks and partial response simulation.

        Args:
            response: Original ATPResponse.

        Returns:
            Potentially modified ATPResponse.

        Raises:
            TokenLimitError: If token limits are exceeded.
        """
        if not self.config.enabled:
            return response

        # Check token limits if metrics are available
        if response.metrics:
            self.check_token_limits(
                input_tokens=response.metrics.input_tokens,
                output_tokens=response.metrics.output_tokens,
            )

        # Apply partial response to artifacts if configured
        if self.config.partial_response and self.config.partial_response.enabled:
            response = self._apply_partial_response_to_artifacts(response)

        return response

    def process_event(self, event: ATPEvent) -> ATPEvent | None:
        """Process an ATPEvent with chaos injections.

        May modify or suppress events based on chaos configuration.

        Args:
            event: Original ATPEvent.

        Returns:
            Potentially modified event, or None if event should be suppressed.
        """
        if not self.config.enabled:
            return event

        # For tool calls, check if we should inject a failure
        if event.event_type == EventType.TOOL_CALL:
            tool_name = event.payload.get("tool", "unknown")
            failure_config = self.should_inject_tool_failure(tool_name)

            if failure_config:
                # Transform the event to show an error
                return ATPEvent(
                    version=event.version,
                    task_id=event.task_id,
                    timestamp=event.timestamp,
                    sequence=event.sequence,
                    event_type=EventType.ERROR,
                    payload={
                        "error_type": failure_config.error_type.value,
                        "message": failure_config.error_message
                        or DEFAULT_ERROR_MESSAGES[failure_config.error_type],
                        "recoverable": failure_config.error_type
                        != ErrorType.PERMISSION_DENIED,
                        "chaos_injected": True,
                        "original_tool": tool_name,
                    },
                )

        return event

    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Check if a value matches a pattern (supports * wildcard)."""
        if pattern == "*":
            return True
        return fnmatch.fnmatch(value.lower(), pattern.lower())

    def _apply_partial_response_to_artifacts(
        self, response: ATPResponse
    ) -> ATPResponse:
        """Apply partial response truncation to artifact contents."""
        from atp.protocol import (
            ArtifactFile,
            ArtifactReference,
            ArtifactStructured,
        )

        modified_artifacts: list[
            ArtifactFile | ArtifactStructured | ArtifactReference
        ] = []
        any_modified = False

        for artifact in response.artifacts:
            if isinstance(artifact, ArtifactFile) and artifact.content:
                truncated, was_truncated = self.apply_partial_response(artifact.content)
                if was_truncated:
                    any_modified = True
                    modified_artifacts.append(
                        ArtifactFile(
                            type="file",
                            path=artifact.path,
                            content_type=artifact.content_type,
                            size_bytes=len(truncated.encode()),
                            content_hash=None,  # Hash is now invalid
                            content=truncated,
                        )
                    )
                else:
                    modified_artifacts.append(artifact)
            else:
                modified_artifacts.append(artifact)

        if any_modified:
            return ATPResponse(
                version=response.version,
                task_id=response.task_id,
                status=ResponseStatus.PARTIAL,  # Mark as partial due to truncation
                artifacts=modified_artifacts,
                metrics=response.metrics,
                error=response.error,
                trace_id=response.trace_id,
            )

        return response


class RateLimiter:
    """Token bucket rate limiter for chaos injection."""

    def __init__(
        self, config: RateLimitConfig, rng: random.Random | None = None
    ) -> None:
        """Initialize the rate limiter.

        Args:
            config: Rate limit configuration.
            rng: Random number generator for variability.
        """
        self.config = config
        self._rng = rng or random.Random()
        self._tokens = float(config.burst_size)
        self._last_update = time.monotonic()
        self._request_times: deque[float] = deque()

    @property
    def retry_after(self) -> int:
        """Get retry-after value for rate limit errors."""
        return self.config.retry_after_seconds

    def allow(self) -> bool:
        """Check if a request is allowed under the rate limit.

        Returns:
            True if request is allowed, False if rate limited.
        """
        now = time.monotonic()
        self._refill_tokens(now)
        self._cleanup_old_requests(now)

        # Check sliding window limit
        if len(self._request_times) >= self.config.requests_per_minute:
            logger.debug(
                "Chaos: rate limit exceeded (%d requests in window)",
                len(self._request_times),
            )
            return False

        # Check token bucket for burst
        if self._tokens < 1:
            logger.debug("Chaos: rate limit exceeded (token bucket empty)")
            return False

        # Allow the request
        self._tokens -= 1
        self._request_times.append(now)
        return True

    def _refill_tokens(self, now: float) -> None:
        """Refill tokens based on elapsed time."""
        elapsed = now - self._last_update
        self._last_update = now

        # Calculate tokens to add based on rate
        tokens_per_second = self.config.requests_per_minute / 60.0
        tokens_to_add = elapsed * tokens_per_second

        self._tokens = min(self._tokens + tokens_to_add, float(self.config.burst_size))

    def _cleanup_old_requests(self, now: float) -> None:
        """Remove requests older than 1 minute from the sliding window."""
        cutoff = now - 60.0
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()


class LatencyInjector:
    """Standalone latency injector for simpler use cases."""

    def __init__(self, config: LatencyConfig, seed: int | None = None) -> None:
        """Initialize the latency injector.

        Args:
            config: Latency configuration.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self._rng = random.Random(seed)

    async def inject(self, tool_name: str | None = None) -> float:
        """Inject latency delay.

        Args:
            tool_name: Optional tool name to check against affected_tools.

        Returns:
            The delay in seconds that was applied.
        """
        if self.config.max_ms <= 0:
            return 0.0

        if tool_name:
            affected = any(
                fnmatch.fnmatch(tool_name.lower(), pattern.lower())
                for pattern in self.config.affected_tools
            )
            if not affected:
                return 0.0

        delay_ms = self._rng.randint(self.config.min_ms, self.config.max_ms)
        delay_seconds = delay_ms / 1000.0

        await asyncio.sleep(delay_seconds)
        return delay_seconds


class ToolFailureInjector:
    """Standalone tool failure injector for simpler use cases."""

    def __init__(
        self, configs: list[ToolFailureConfig], seed: int | None = None
    ) -> None:
        """Initialize the tool failure injector.

        Args:
            configs: List of tool failure configurations.
            seed: Random seed for reproducibility.
        """
        self.configs = configs
        self._rng = random.Random(seed)

    def should_fail(self, tool_name: str) -> ToolFailureConfig | None:
        """Check if the tool should fail.

        Args:
            tool_name: Name of the tool.

        Returns:
            ToolFailureConfig if failure should be injected, None otherwise.
        """
        for config in self.configs:
            pattern = config.tool
            if pattern == "*" or fnmatch.fnmatch(tool_name.lower(), pattern.lower()):
                if self._rng.random() < config.probability:
                    return config
        return None

    def inject(self, tool_name: str) -> None:
        """Inject a tool failure if configured.

        Args:
            tool_name: Name of the tool.

        Raises:
            ToolFailureError: If failure is injected.
        """
        config = self.should_fail(tool_name)
        if config:
            raise ToolFailureError(
                tool_name,
                config.error_type,
                config.error_message,
            )


class TokenLimitInjector:
    """Standalone token limit injector for simpler use cases."""

    def __init__(self, config: TokenLimitConfig) -> None:
        """Initialize the token limit injector.

        Args:
            config: Token limit configuration.
        """
        self.config = config

    def check(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Check if token limits are exceeded.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Raises:
            TokenLimitError: If any limit is exceeded.
        """
        if self.config.max_input is not None and input_tokens is not None:
            if input_tokens > self.config.max_input:
                raise TokenLimitError("input", self.config.max_input, input_tokens)

        if self.config.max_output is not None and output_tokens is not None:
            if output_tokens > self.config.max_output:
                raise TokenLimitError("output", self.config.max_output, output_tokens)


class PartialResponseInjector:
    """Standalone partial response injector for simpler use cases."""

    def __init__(self, config: PartialResponseConfig, seed: int | None = None) -> None:
        """Initialize the partial response injector.

        Args:
            config: Partial response configuration.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self._rng = random.Random(seed)

    def apply(self, content: str) -> tuple[str, bool]:
        """Apply partial response truncation.

        Args:
            content: Original content.

        Returns:
            Tuple of (content, was_truncated).
        """
        if not self.config.enabled:
            return content, False

        if self._rng.random() >= self.config.truncate_probability:
            return content, False

        keep_percentage = self._rng.uniform(
            self.config.min_percentage,
            self.config.max_percentage,
        )

        truncate_at = int(len(content) * keep_percentage)
        return content[:truncate_at], True
