"""Tests for MCP handshake observability events.

Covers ``packages/atp-dashboard/atp/dashboard/mcp/observability.py`` and
the events emitted by ``MCPAuthMiddleware``. The runbook at
``docs/runbooks/mcp-observability.md`` documents the schema; these
tests pin its observable shape so a downstream Prometheus/OTel mapping
can be written against a stable contract.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from atp.dashboard.mcp.auth import MCPAuthMiddleware
from atp.dashboard.mcp.observability import (
    MCP_FIRST_TOOL_CALL,
    MCP_HANDSHAKE_AUTHORIZED,
    MCP_HANDSHAKE_REJECTED,
    MCP_HANDSHAKE_STARTED,
    emit_event,
    maybe_emit_first_tool_call,
    new_request_id,
    reset_state,
)


@pytest.fixture(autouse=True)
def _isolate_observability_state() -> None:
    """The ``_first_tool_call_seen`` cache is module-level state. Clear
    between tests so the per-session dedup cannot leak across cases."""
    reset_state()


def _records_for(
    caplog: pytest.LogCaptureFixture, event: str
) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == event]


# ---------------------------------------------------------------------------
# observability helpers
# ---------------------------------------------------------------------------


def test_new_request_id_returns_16_char_hex_and_is_unique() -> None:
    """16 hex chars = 64 bits of entropy; collision-resistant for any
    realistic single-process run."""
    a = new_request_id()
    b = new_request_id()
    assert len(a) == 16 and len(b) == 16
    assert int(a, 16) >= 0  # parses as hex
    assert a != b


def test_emit_event_produces_structured_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``emit_event`` must surface the event name AND the structured
    fields as a ``LogRecord.extra`` so downstream JSON formatters or
    tests can pick them up."""
    caplog.set_level(logging.INFO, logger="atp.mcp.observability")
    emit_event("custom_event", request_id="abc", foo=1, bar="x")

    records = _records_for(caplog, "custom_event")
    assert len(records) == 1
    record = records[0]
    assert record.message == "custom_event"
    assert record.request_id == "abc"  # type: ignore[attr-defined]
    assert record.foo == 1  # type: ignore[attr-defined]
    assert record.bar == "x"  # type: ignore[attr-defined]


def test_maybe_emit_first_tool_call_dedupes_per_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Second call with the same ``session_id`` is a no-op; a different
    session emits independently."""
    caplog.set_level(logging.INFO, logger="atp.mcp.observability")

    assert (
        maybe_emit_first_tool_call(session_id="s1", request_id="r1", tool="ping")
        is True
    )
    assert (
        maybe_emit_first_tool_call(session_id="s1", request_id="r2", tool="make_move")
        is False
    )
    assert (
        maybe_emit_first_tool_call(session_id="s2", request_id="r3", tool="ping")
        is True
    )

    records = _records_for(caplog, MCP_FIRST_TOOL_CALL)
    # Only s1's first call (request_id=r1) and s2's first (r3) emit.
    assert {(r.session_id, r.request_id) for r in records} == {  # type: ignore[attr-defined]
        ("s1", "r1"),
        ("s2", "r3"),
    }


# ---------------------------------------------------------------------------
# emit_tool_call — session-id sourcing (PR #102 follow-up)
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Mimics the surface of ``starlette.requests.Request`` that
    ``emit_tool_call`` reads — ``state.mcp_request_id`` and
    ``query_params.get(...)`` — without needing a real HTTP scope."""

    def __init__(
        self,
        *,
        request_id: str | None = None,
        url_session_id: str | None = None,
    ) -> None:
        class _State:
            pass

        self.state = _State()
        if request_id is not None:
            self.state.mcp_request_id = request_id  # type: ignore[attr-defined]
        self.query_params: dict[str, str] = (
            {"session_id": url_session_id} if url_session_id is not None else {}
        )


class _FakeCtx:
    def __init__(self, session_id: str | None = None) -> None:
        if session_id is not None:
            self.session_id = session_id


def _patch_get_http_request(
    monkeypatch: pytest.MonkeyPatch, request: _FakeRequest | Exception
) -> None:
    """Make ``fastmcp.server.dependencies.get_http_request`` return
    ``request`` (or raise it, if an Exception). Tool wrappers import
    the helper inside the function body, so module-level patching
    via the original location works."""

    def _fake() -> _FakeRequest:
        if isinstance(request, Exception):
            raise request
        return request

    import fastmcp.server.dependencies as deps

    monkeypatch.setattr(deps, "get_http_request", _fake)


def test_emit_tool_call_uses_url_session_id_when_present(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary path: SSE messages-POST carries ``?session_id=`` on
    the URL. ``emit_tool_call`` must dedup against that, not
    ``ctx.session_id``."""
    from atp.dashboard.mcp.observability import emit_tool_call

    caplog.set_level(logging.INFO, logger="atp.mcp.observability")
    _patch_get_http_request(
        monkeypatch,
        _FakeRequest(request_id="rid-1", url_session_id="sse-abc"),
    )
    # ctx.session_id deliberately set to a DIFFERENT value to prove
    # the URL wins.
    ctx = _FakeCtx(session_id="ctx-xyz")

    emit_tool_call(ctx, tool="ping")
    emit_tool_call(ctx, tool="get_current_state")  # second call → dedup

    records = _records_for(caplog, MCP_FIRST_TOOL_CALL)
    assert len(records) == 1
    assert records[0].session_id == "sse-abc"  # type: ignore[attr-defined]
    assert records[0].request_id == "rid-1"  # type: ignore[attr-defined]
    assert records[0].tool == "ping"  # type: ignore[attr-defined]


def test_emit_tool_call_falls_back_to_ctx_session_id(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback path: URL has no session_id (e.g. a non-SSE transport
    or a request shape we haven't seen yet). Use ``ctx.session_id``
    when it's available."""
    from atp.dashboard.mcp.observability import emit_tool_call

    caplog.set_level(logging.INFO, logger="atp.mcp.observability")
    _patch_get_http_request(
        monkeypatch,
        _FakeRequest(request_id="rid-2", url_session_id=None),
    )
    ctx = _FakeCtx(session_id="ctx-fallback")

    emit_tool_call(ctx, tool="join_tournament")

    records = _records_for(caplog, MCP_FIRST_TOOL_CALL)
    assert len(records) == 1
    assert records[0].session_id == "ctx-fallback"  # type: ignore[attr-defined]
    assert records[0].request_id == "rid-2"  # type: ignore[attr-defined]


def test_emit_tool_call_skips_when_neither_source_has_session_id(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No URL query param, no ctx attribute → skip emit. Falling
    back to ``id(ctx)`` would re-emit on every dispatch because
    ``Context`` is reconstructed per call (Copilot review on PR #101)."""
    from atp.dashboard.mcp.observability import emit_tool_call

    caplog.set_level(logging.INFO, logger="atp.mcp.observability")
    _patch_get_http_request(
        monkeypatch,
        _FakeRequest(request_id="rid-3", url_session_id=None),
    )

    emit_tool_call(_FakeCtx(session_id=None), tool="ping")

    assert _records_for(caplog, MCP_FIRST_TOOL_CALL) == []


def test_emit_tool_call_tolerates_get_http_request_raising(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_http_request`` raises outside an HTTP-bound MCP call.
    ``emit_tool_call`` must not propagate that and should still try
    the ctx fallback."""
    from atp.dashboard.mcp.observability import emit_tool_call

    caplog.set_level(logging.INFO, logger="atp.mcp.observability")
    _patch_get_http_request(monkeypatch, RuntimeError("no http context"))

    emit_tool_call(_FakeCtx(session_id="ctx-only"), tool="make_move")

    records = _records_for(caplog, MCP_FIRST_TOOL_CALL)
    assert len(records) == 1
    assert records[0].session_id == "ctx-only"  # type: ignore[attr-defined]
    assert records[0].request_id == "no-request-id"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# MCPAuthMiddleware events
# ---------------------------------------------------------------------------


def _make_app(
    set_user_id: int | None,
    set_agent_id: int | None = None,
    set_agent_purpose: str | None = "tournament",
) -> FastAPI:
    """Build a tiny app that pre-populates request.state, then runs
    MCPAuthMiddleware. Mirrors the helper from test_auth_middleware.py."""
    app = FastAPI()
    app.add_middleware(MCPAuthMiddleware)

    @app.middleware("http")
    async def _set_state(request, call_next):  # type: ignore[no-untyped-def]
        if set_user_id is not None:
            request.state.user_id = set_user_id
        if set_agent_id is not None:
            request.state.agent_id = set_agent_id
        if set_agent_purpose is not None:
            request.state.agent_purpose = set_agent_purpose
        return await call_next(request)

    @app.get("/_handshake_test")
    async def _h() -> dict:
        return {"ok": True}

    return app


def test_handshake_started_then_authorized_for_tournament_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The happy path emits exactly two events in order, sharing the
    same ``request_id`` and carrying timing on the second."""
    caplog.set_level(logging.INFO, logger="atp.mcp.observability")

    client = TestClient(_make_app(set_user_id=7, set_agent_id=42))
    response = client.get(
        "/_handshake_test",
        headers={"User-Agent": "pytest-client/1.0"},
    )
    assert response.status_code == 200

    started = _records_for(caplog, MCP_HANDSHAKE_STARTED)
    authed = _records_for(caplog, MCP_HANDSHAKE_AUTHORIZED)
    assert len(started) == 1
    assert len(authed) == 1

    s, a = started[0], authed[0]
    assert s.request_id == a.request_id  # type: ignore[attr-defined]
    assert isinstance(s.request_id, str) and len(s.request_id) == 16  # type: ignore[attr-defined]
    assert s.user_agent == "pytest-client/1.0"  # type: ignore[attr-defined]
    assert s.path == "/_handshake_test"  # type: ignore[attr-defined]
    assert a.user_id == 7  # type: ignore[attr-defined]
    assert a.agent_id == 42  # type: ignore[attr-defined]
    assert a.agent_purpose == "tournament"  # type: ignore[attr-defined]
    assert isinstance(a.duration_ms, float) and a.duration_ms >= 0  # type: ignore[attr-defined]


def test_handshake_rejected_unauthenticated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="atp.mcp.observability")

    client = TestClient(_make_app(set_user_id=None))
    response = client.get("/_handshake_test")
    assert response.status_code == 401

    rejected = _records_for(caplog, MCP_HANDSHAKE_REJECTED)
    assert len(rejected) == 1
    r = rejected[0]
    assert r.reason == "unauthenticated"  # type: ignore[attr-defined]
    assert r.status == 401  # type: ignore[attr-defined]
    assert isinstance(r.duration_ms, float)  # type: ignore[attr-defined]
    # No HANDSHAKE_AUTHORIZED on a rejection.
    assert _records_for(caplog, MCP_HANDSHAKE_AUTHORIZED) == []


def test_handshake_rejected_user_level_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="atp.mcp.observability")

    client = TestClient(_make_app(set_user_id=11, set_agent_purpose=None))
    response = client.get("/_handshake_test")
    assert response.status_code == 403

    rejected = _records_for(caplog, MCP_HANDSHAKE_REJECTED)
    assert len(rejected) == 1
    assert rejected[0].reason == "user_level_token"  # type: ignore[attr-defined]
    assert rejected[0].status == 403  # type: ignore[attr-defined]


def test_handshake_rejected_benchmark_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="atp.mcp.observability")

    client = TestClient(_make_app(set_user_id=11, set_agent_purpose="benchmark"))
    response = client.get("/_handshake_test")
    assert response.status_code == 403

    rejected = _records_for(caplog, MCP_HANDSHAKE_REJECTED)
    assert len(rejected) == 1
    assert rejected[0].reason == "benchmark_token"  # type: ignore[attr-defined]
    assert rejected[0].status == 403  # type: ignore[attr-defined]


def test_request_id_propagated_to_scope_state() -> None:
    """Downstream tool wrappers read ``request.state.mcp_request_id``
    via FastMCP's ``get_http_request()`` helper; the middleware must
    have stamped it before the request reaches them."""
    seen: dict[str, str | None] = {}

    app = FastAPI()
    app.add_middleware(MCPAuthMiddleware)

    @app.middleware("http")
    async def _set_state(request, call_next):  # type: ignore[no-untyped-def]
        request.state.user_id = 7
        request.state.agent_purpose = "tournament"
        return await call_next(request)

    @app.get("/_state")
    async def _state(request: Request) -> dict:
        seen["request_id"] = getattr(request.state, "mcp_request_id", None)
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/_state")
    assert response.status_code == 200
    rid = seen["request_id"]
    assert isinstance(rid, str) and len(rid) == 16
