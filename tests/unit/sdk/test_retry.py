"""Tests for retry module (TDD)."""

from __future__ import annotations

import httpx
import pytest
from atp_sdk.retry import RETRYABLE_STATUS_CODES, RetryConfig, retry_request

# ---------------------------------------------------------------------------
# RetryConfig defaults
# ---------------------------------------------------------------------------


def test_retry_config_defaults() -> None:
    """RetryConfig has correct default values."""
    cfg = RetryConfig()
    assert cfg.max_retries == 3
    assert cfg.retry_backoff == 1.0
    assert cfg.max_retry_delay == 30.0
    assert cfg.retry_on_timeout is True


def test_retry_config_frozen() -> None:
    """RetryConfig is immutable (frozen dataclass)."""
    cfg = RetryConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.max_retries = 99  # type: ignore[misc]


def test_retryable_status_codes() -> None:
    """RETRYABLE_STATUS_CODES contains 502, 503, 504."""
    assert RETRYABLE_STATUS_CODES == {502, 503, 504}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_response(
    status_code: int, headers: dict[str, str] | None = None
) -> httpx.Response:
    """Build a minimal httpx.Response."""
    return httpx.Response(status_code, headers=headers or {})


# ---------------------------------------------------------------------------
# No retry on success
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_retry_on_success() -> None:
    """200 response is returned immediately without retrying."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return make_response(200)

    cfg = RetryConfig(max_retries=3)
    response = await retry_request(sender, cfg)
    assert response.status_code == 200
    assert call_count == 1


# ---------------------------------------------------------------------------
# Retry on retryable status codes
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_retry_on_502() -> None:
    """Retries on HTTP 502 and returns last response when exhausted."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return make_response(502)

    cfg = RetryConfig(max_retries=2, retry_backoff=0.0)
    response = await retry_request(sender, cfg)
    # initial attempt + 2 retries = 3 total
    assert call_count == 3
    assert response.status_code == 502


@pytest.mark.anyio
async def test_retry_on_503() -> None:
    """Retries on HTTP 503."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return make_response(503)

    cfg = RetryConfig(max_retries=1, retry_backoff=0.0)
    response = await retry_request(sender, cfg)
    assert call_count == 2
    assert response.status_code == 503


@pytest.mark.anyio
async def test_retry_on_504() -> None:
    """Retries on HTTP 504."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return make_response(504)

    cfg = RetryConfig(max_retries=1, retry_backoff=0.0)
    response = await retry_request(sender, cfg)
    assert call_count == 2
    assert response.status_code == 504


# ---------------------------------------------------------------------------
# Retry on transport error
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_retry_on_connect_error() -> None:
    """Retries on httpx.ConnectError (transport error)."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("connection refused")

    cfg = RetryConfig(max_retries=2, retry_backoff=0.0)
    with pytest.raises(httpx.ConnectError):
        await retry_request(sender, cfg)
    assert call_count == 3


@pytest.mark.anyio
async def test_retry_exhausted_raises_last_exception() -> None:
    """When retries exhausted on transport error, raises the last exception."""
    errors: list[httpx.ConnectError] = []

    async def sender() -> httpx.Response:
        err = httpx.ConnectError("attempt failed")
        errors.append(err)
        raise err

    cfg = RetryConfig(max_retries=1, retry_backoff=0.0)
    with pytest.raises(httpx.ConnectError, match="attempt failed"):
        await retry_request(sender, cfg)


@pytest.mark.anyio
async def test_retry_exhausted_returns_last_response() -> None:
    """When retries exhausted on bad status, returns the last response."""

    async def sender() -> httpx.Response:
        return make_response(503)

    cfg = RetryConfig(max_retries=2, retry_backoff=0.0)
    response = await retry_request(sender, cfg)
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# No retry on non-retryable status codes
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_retry_on_400() -> None:
    """Does NOT retry on 400 Bad Request."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return make_response(400)

    cfg = RetryConfig(max_retries=3, retry_backoff=0.0)
    response = await retry_request(sender, cfg)
    assert call_count == 1
    assert response.status_code == 400


@pytest.mark.anyio
async def test_no_retry_on_401() -> None:
    """Does NOT retry on 401 Unauthorized."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return make_response(401)

    cfg = RetryConfig(max_retries=3, retry_backoff=0.0)
    await retry_request(sender, cfg)
    assert call_count == 1


@pytest.mark.anyio
async def test_no_retry_on_404() -> None:
    """Does NOT retry on 404 Not Found."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return make_response(404)

    cfg = RetryConfig(max_retries=3, retry_backoff=0.0)
    await retry_request(sender, cfg)
    assert call_count == 1


# ---------------------------------------------------------------------------
# 429 with Retry-After
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_retry_on_429_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries on 429 and respects Retry-After header (capped at max_retry_delay)."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("atp_sdk.retry.anyio.sleep", fake_sleep)

    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return make_response(429, headers={"Retry-After": "5"})
        return make_response(200)

    cfg = RetryConfig(max_retries=2, retry_backoff=0.0, max_retry_delay=30.0)
    response = await retry_request(sender, cfg)
    assert response.status_code == 200
    assert call_count == 2
    # Should have slept for 5 seconds (from Retry-After header)
    assert slept[0] == pytest.approx(5.0)


@pytest.mark.anyio
async def test_retry_on_429_retry_after_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry-After is capped at max_retry_delay."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("atp_sdk.retry.anyio.sleep", fake_sleep)

    async def sender() -> httpx.Response:
        return make_response(429, headers={"Retry-After": "9999"})

    cfg = RetryConfig(max_retries=1, retry_backoff=0.0, max_retry_delay=10.0)
    response = await retry_request(sender, cfg)
    assert response.status_code == 429
    assert slept[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# No retry on timeout when retry_on_timeout=False
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_retry_on_timeout_when_disabled() -> None:
    """TimeoutException is NOT retried when retry_on_timeout=False."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.TimeoutException("timed out")

    cfg = RetryConfig(max_retries=3, retry_backoff=0.0, retry_on_timeout=False)
    with pytest.raises(httpx.TimeoutException):
        await retry_request(sender, cfg)
    assert call_count == 1


@pytest.mark.anyio
async def test_retry_on_timeout_when_enabled() -> None:
    """TimeoutException IS retried when retry_on_timeout=True."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.TimeoutException("timed out")

    cfg = RetryConfig(max_retries=2, retry_backoff=0.0, retry_on_timeout=True)
    with pytest.raises(httpx.TimeoutException):
        await retry_request(sender, cfg)
    assert call_count == 3


# ---------------------------------------------------------------------------
# max_retry_delay caps backoff
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_max_retry_delay_caps_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff delay is capped at max_retry_delay."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("atp_sdk.retry.anyio.sleep", fake_sleep)
    # Patch random to always return 1.0 (max of uniform range) for determinism
    monkeypatch.setattr("atp_sdk.retry.random.uniform", lambda a, b: b)

    async def sender() -> httpx.Response:
        return make_response(503)

    cfg = RetryConfig(max_retries=3, retry_backoff=100.0, max_retry_delay=5.0)
    await retry_request(sender, cfg)

    # All delays should be capped at max_retry_delay=5.0
    for delay in slept:
        assert delay <= 5.0


# ---------------------------------------------------------------------------
# max_retries=0 disables retry
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_max_retries_zero_disables_retry() -> None:
    """max_retries=0 means no retry at all."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return make_response(502)

    cfg = RetryConfig(max_retries=0)
    response = await retry_request(sender, cfg)
    assert call_count == 1
    assert response.status_code == 502


@pytest.mark.anyio
async def test_max_retries_zero_raises_immediately() -> None:
    """max_retries=0 raises transport error immediately."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("refused")

    cfg = RetryConfig(max_retries=0)
    with pytest.raises(httpx.ConnectError):
        await retry_request(sender, cfg)
    assert call_count == 1


# ---------------------------------------------------------------------------
# Eventual success after retries
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_success_after_retries() -> None:
    """Succeeds on the Nth attempt after earlier failures."""
    call_count = 0

    async def sender() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return make_response(503)
        return make_response(200)

    cfg = RetryConfig(max_retries=3, retry_backoff=0.0)
    response = await retry_request(sender, cfg)
    assert response.status_code == 200
    assert call_count == 3
