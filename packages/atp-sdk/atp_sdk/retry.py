"""Retry logic for HTTP requests with exponential backoff and full jitter."""

from __future__ import annotations

import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import anyio
import httpx

logger = logging.getLogger("atp_sdk")

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({502, 503, 504})
_RETRY_ON_429_STATUS = 429


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries).
        retry_backoff: Base backoff multiplier in seconds.
        max_retry_delay: Maximum delay between retries in seconds.
        retry_on_timeout: Whether to retry on httpx.TimeoutException.
    """

    max_retries: int = 3
    retry_backoff: float = 1.0
    max_retry_delay: float = 30.0
    retry_on_timeout: bool = True


def _jitter_delay(attempt: int, config: RetryConfig) -> float:
    """Compute full-jitter delay: random(0, min(max_retry_delay, backoff * 2^n))."""
    cap = min(config.max_retry_delay, config.retry_backoff * (2**attempt))
    return random.uniform(0, cap)


def _is_retryable_transport_error(exc: Exception, config: RetryConfig) -> bool:
    """Return True if the exception should trigger a retry."""
    if isinstance(exc, httpx.TimeoutException):
        return config.retry_on_timeout
    return isinstance(exc, httpx.TransportError)


async def retry_request(
    sender: Callable[[], Awaitable[httpx.Response]],
    config: RetryConfig,
) -> httpx.Response:
    """Execute ``sender`` with retry logic.

    Retries on:
    - httpx.TransportError (including ConnectError)
    - httpx.TimeoutException (when retry_on_timeout=True)
    - HTTP 502, 503, 504
    - HTTP 429 (using Retry-After header if present)

    Does NOT retry on other 4xx responses.

    When all retries are exhausted:
    - Raises the last exception for transport errors.
    - Returns the last response for status-code-based failures.

    Args:
        sender: Async callable that performs the HTTP request.
        config: Retry configuration.

    Returns:
        The first successful httpx.Response or the last failing one.

    Raises:
        httpx.TransportError: When retries exhausted due to transport errors.
        httpx.TimeoutException: When retries exhausted (or disabled) on timeout.
    """
    last_response: httpx.Response | None = None
    last_exc: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            response = await sender()
        except Exception as exc:
            if not _is_retryable_transport_error(exc, config):
                raise
            last_exc = exc
            last_response = None
            if attempt < config.max_retries:
                delay = _jitter_delay(attempt, config)
                logger.warning(
                    "Retry %d/%d after transport error (%s); sleeping %.2fs",
                    attempt + 1,
                    config.max_retries,
                    type(exc).__name__,
                    delay,
                )
                await anyio.sleep(delay)
            continue

        status = response.status_code

        # Success — return immediately
        if status < 400:
            return response

        # 429 Too Many Requests — honour Retry-After
        if status == _RETRY_ON_429_STATUS:
            last_response = response
            if attempt < config.max_retries:
                retry_after_raw = response.headers.get("Retry-After")
                try:
                    delay = min(float(retry_after_raw or 0), config.max_retry_delay)
                except ValueError:
                    delay = _jitter_delay(attempt, config)
                logger.warning(
                    "Retry %d/%d after 429; sleeping %.2fs",
                    attempt + 1,
                    config.max_retries,
                    delay,
                )
                await anyio.sleep(delay)
            continue

        # Other retryable status codes (502, 503, 504)
        if status in RETRYABLE_STATUS_CODES:
            last_response = response
            if attempt < config.max_retries:
                delay = _jitter_delay(attempt, config)
                logger.warning(
                    "Retry %d/%d after HTTP %d; sleeping %.2fs",
                    attempt + 1,
                    config.max_retries,
                    status,
                    delay,
                )
                await anyio.sleep(delay)
            continue

        # Non-retryable status (e.g. 400, 401, 404, 500)
        return response

    # All retries exhausted
    if last_exc is not None:
        raise last_exc
    assert last_response is not None
    return last_response
