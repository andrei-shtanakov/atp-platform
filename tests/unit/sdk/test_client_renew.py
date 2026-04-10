"""Tests for AsyncATPClient on_token_expired re-login flow (Fix A)."""

from __future__ import annotations

import httpx
import pytest
from atp_sdk.client import AsyncATPClient


def _transport(
    handler: httpx.MockTransport | None = None,
    responses: list[httpx.Response] | None = None,
    record: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """Build a MockTransport that returns a scripted sequence of responses."""
    assert responses is not None
    it = iter(responses)

    def _handle(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        return next(it)

    return httpx.MockTransport(_handle)


def _make_client(
    responses: list[httpx.Response],
    on_token_expired: object | None = None,
    token: str = "initial",
    record: list[httpx.Request] | None = None,
) -> AsyncATPClient:
    client = AsyncATPClient(
        platform_url="http://test",
        token=token,
        max_retries=0,  # isolate 401 logic from transient retry behavior
        on_token_expired=on_token_expired,  # type: ignore[arg-type]
    )
    client._http = httpx.AsyncClient(
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
        transport=_transport(responses=responses, record=record),
    )
    return client


# ---------------------------------------------------------------------------
# on_token_expired callback behavior
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_401_triggers_callback_and_retries_with_new_token() -> None:
    """On 401, callback is called once; second request uses the new token."""
    calls: list[int] = []

    async def renew() -> str:
        calls.append(1)
        return "new-token"

    record: list[httpx.Request] = []
    client = _make_client(
        responses=[httpx.Response(401), httpx.Response(200, json={"ok": True})],
        on_token_expired=renew,
        record=record,
    )

    try:
        resp = await client._request("GET", "/api/v1/benchmarks")
        assert resp.status_code == 200
        assert len(calls) == 1
        assert client.token == "new-token"
        assert len(record) == 2
        # The retried request carries the new Authorization header
        assert record[1].headers["Authorization"] == "Bearer new-token"
    finally:
        await client.close()


@pytest.mark.anyio
async def test_401_twice_does_not_loop() -> None:
    """If callback returns a still-invalid token, we return the second 401."""
    calls: list[int] = []

    async def renew() -> str:
        calls.append(1)
        return "still-bad"

    client = _make_client(
        responses=[httpx.Response(401), httpx.Response(401)],
        on_token_expired=renew,
    )

    try:
        resp = await client._request("GET", "/api/v1/benchmarks")
        assert resp.status_code == 401
        # Callback called exactly once — no infinite loop
        assert len(calls) == 1
    finally:
        await client.close()


@pytest.mark.anyio
async def test_401_without_callback_returns_401_unchanged() -> None:
    """When no on_token_expired callback is set, 401 passes through."""
    client = _make_client(
        responses=[httpx.Response(401)],
        on_token_expired=None,
    )
    try:
        resp = await client._request("GET", "/api/v1/benchmarks")
        assert resp.status_code == 401
    finally:
        await client.close()


@pytest.mark.anyio
async def test_200_does_not_trigger_callback() -> None:
    """Happy-path 200 never invokes the callback."""
    calls: list[int] = []

    async def renew() -> str:
        calls.append(1)
        return "unused"

    client = _make_client(
        responses=[httpx.Response(200, json={"ok": True})],
        on_token_expired=renew,
    )
    try:
        resp = await client._request("GET", "/api/v1/benchmarks")
        assert resp.status_code == 200
        assert calls == []
    finally:
        await client.close()


@pytest.mark.anyio
async def test_callback_exception_propagates() -> None:
    """Exceptions from the callback bubble up to the caller."""

    async def renew() -> str:
        raise RuntimeError("vault down")

    client = _make_client(
        responses=[httpx.Response(401)],
        on_token_expired=renew,
    )
    try:
        with pytest.raises(RuntimeError, match="vault down"):
            await client._request("GET", "/api/v1/benchmarks")
    finally:
        await client.close()


def test_on_token_expired_default_is_none() -> None:
    """Parameter defaults to None for backwards compatibility."""
    client = AsyncATPClient(platform_url="http://test", token="tok")
    assert client._on_token_expired is None
