"""HTTP-level tests for the live tournament dashboard routes.

Covers:
  * GET /api/v1/tournaments/{id}/dashboard (one-shot JSON snapshot)
  * GET /api/v1/tournaments/{id}/dashboard/stream (SSE)

Both routes are anonymous-friendly for public tournaments
(``join_token IS NULL``) and 404 -- not 403 -- when the viewer cannot
see the tournament so the endpoints can't be used as an existence
oracle.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.tournament.events import TournamentEvent, TournamentEventBus
from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app
from atp.dashboard.v2.routes import tournament_live as tournament_live_module


@pytest.fixture(autouse=True)
def _short_sse_heartbeat(monkeypatch):
    """SSE generator uses a 15 s heartbeat; shrink it for tests so the
    bus.subscribe ``wait_for(queue.get())`` wakes quickly and the
    generator notices client disconnects without making the test wall
    clock 15 s long.
    """
    monkeypatch.setattr(tournament_live_module, "_HEARTBEAT_SECONDS", 0.1)


@pytest.fixture
def _patch_fresh_helpers(monkeypatch, test_database: Database):
    """Make ``_project_snapshot_fresh`` / ``_resolved_match_id`` read from
    the test ``Database`` rather than ``get_database()``.

    The default in-memory engine uses NullPool, so each new session
    creates a fresh DB without any of the test's seeded rows. The SSE
    generator opens its own short-lived session per snapshot via
    ``get_database()``, so we redirect both helpers to the test engine.
    """
    from sqlalchemy import select as _select

    from atp.dashboard.models import GameResult
    from atp.dashboard.v2.routes.el_farol_from_tournament import (
        _reshape_from_tournament,
    )

    async def _fresh_snapshot(tournament_id: int):
        async with test_database.session_factory() as s:
            return await _reshape_from_tournament(tournament_id, s)

    async def _fresh_match_id(tournament_id: int):
        async with test_database.session_factory() as s:
            row = (
                await s.execute(
                    _select(GameResult).where(GameResult.tournament_id == tournament_id)
                )
            ).scalar_one_or_none()
            return row.match_id if row is not None else None

    monkeypatch.setattr(
        tournament_live_module, "_project_snapshot_fresh", _fresh_snapshot
    )
    monkeypatch.setattr(tournament_live_module, "_resolved_match_id", _fresh_match_id)


# ---------- fixtures ----------


@pytest.fixture
def v2_app(test_database: Database):
    """Create a test app with v2 routes and the in-memory DB wired in.

    Same shape as ``test_el_farol_dashboard_endpoint.py`` so the live
    endpoints exercise the real router stack with our test DB.
    """
    app = create_test_app(use_v2_routes=True)

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


@pytest.fixture
def test_bus(v2_app) -> TournamentEventBus:
    """A dedicated TournamentEventBus stored on app.state.

    ``tournament_live._bus_from_request`` prefers ``app.state.tournament_event_bus``
    when present, so injecting one here lets us drive the SSE generator
    without touching the global mcp singleton.
    """
    bus = TournamentEventBus()
    v2_app.state.tournament_event_bus = bus
    return bus


@pytest.fixture
def sse_app(test_database: Database) -> FastAPI:
    """Build a minimal FastAPI app exposing only the tournament_live router.

    The full ``create_test_app`` stack installs ``SlowAPIMiddleware`` (a
    ``BaseHTTPMiddleware`` subclass) that buffers response bodies and
    therefore breaks long-lived SSE streams under ASGITransport. The
    factory's existing ``_slowapi_except_mcp`` shim only exempts
    ``/mcp/*``, not the live tournament dashboard mount, which is why a
    minimal app is needed here. See the comment in
    ``packages/atp-dashboard/atp/dashboard/v2/factory.py`` for the
    same caveat that already applies to the MCP SSE mount.
    """
    from fastapi import FastAPI

    from atp.dashboard.v2.routes.tournament_live import router as live_router

    app = FastAPI()
    # Mount under /api so the prefix matches production
    # ("/api/v1/tournaments/{id}/dashboard{/stream}").
    app.include_router(live_router, prefix="/api")

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session
    return app


@pytest.fixture
def sse_bus(sse_app) -> TournamentEventBus:
    bus = TournamentEventBus()
    sse_app.state.tournament_event_bus = bus
    return bus


async def _make_pending_el_farol(
    session: AsyncSession,
    *,
    creator_id: int | None = None,
    join_token: str | None = None,
    num_players: int = 2,
    total_rounds: int = 1,
) -> Tournament:
    t = Tournament(
        game_type="el_farol",
        status=TournamentStatus.PENDING,
        num_players=num_players,
        total_rounds=total_rounds,
        round_deadline_s=30,
        created_by=creator_id,
        config={},
        rules={},
        join_token=join_token,
    )
    session.add(t)
    await session.commit()
    await session.refresh(t)
    return t


async def _add_resolved_round(
    session: AsyncSession,
    tournament: Tournament,
    *,
    round_number: int = 1,
) -> Round:
    """Seed a tournament with two builtin participants and one COMPLETED round.

    Each participant submits ``intervals=[[0, 0]]`` (slot 0). Both attend
    the same crowded slot -> payoff -1.0 each, mirroring
    ``test_el_farol_resolve_round_writes_payoffs`` in the unit suite.
    """
    p1 = Participant(
        tournament_id=tournament.id,
        user_id=None,
        agent_name="builtin-1",
        agent_id=None,
        builtin_strategy="el_farol/calibrated",
    )
    p2 = Participant(
        tournament_id=tournament.id,
        user_id=None,
        agent_name="builtin-2",
        agent_id=None,
        builtin_strategy="el_farol/conservative",
    )
    session.add_all([p1, p2])
    await session.flush()

    r = Round(
        tournament_id=tournament.id,
        round_number=round_number,
        status=RoundStatus.COMPLETED,
        state={},
    )
    session.add(r)
    await session.flush()

    a1 = Action(
        round_id=r.id,
        participant_id=p1.id,
        action_data={"intervals": [[0, 0]]},
        payoff=-1.0,
        source="submitted",
    )
    a2 = Action(
        round_id=r.id,
        participant_id=p2.id,
        action_data={"intervals": [[0, 0]]},
        payoff=-1.0,
        source="submitted",
    )
    session.add_all([a1, a2])
    await session.commit()
    return r


# ---------- JSON dashboard endpoint ----------


class TestDashboardJSONEndpoint:
    @pytest.mark.anyio
    async def test_anonymous_public_no_rounds_returns_empty_data(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """GIVEN a public El Farol tournament with no resolved rounds
        WHEN an anonymous viewer GETs the dashboard
        THEN a 200 with empty DATA / NUM_DAYS=0 is returned.
        """
        t = await _make_pending_el_farol(db_session, join_token=None)

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get(f"/api/v1/tournaments/{t.id}/dashboard")

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["DATA"] == []
        assert data["NUM_DAYS"] == 0

    @pytest.mark.anyio
    async def test_anonymous_public_with_one_round_returns_one_day(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """One COMPLETED round -> DATA has one entry."""
        t = await _make_pending_el_farol(db_session, join_token=None)
        await _add_resolved_round(db_session, t, round_number=1)

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get(f"/api/v1/tournaments/{t.id}/dashboard")

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data["DATA"]) == 1
        assert data["NUM_DAYS"] == 1

    @pytest.mark.anyio
    async def test_anonymous_private_returns_404(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        """Private tournament 404s for anonymous viewers (existence-hiding)."""
        t = await _make_pending_el_farol(db_session, join_token="secret")

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get(f"/api/v1/tournaments/{t.id}/dashboard")

        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_unknown_tournament_returns_404(
        self,
        v2_app,
        db_session: AsyncSession,
        disable_dashboard_auth,
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/tournaments/99999/dashboard")

        assert resp.status_code == 404


# ---------- SSE stream endpoint ----------


def _parse_sse_chunks(buf: str) -> list[tuple[str, dict | None]]:
    """Parse an SSE buffer into a list of (event_type, parsed_data) tuples.

    Comments (``: ping``) are skipped. Frames without ``event:`` default
    to event type ``message``.
    """
    out: list[tuple[str, dict | None]] = []
    for frame in buf.split("\n\n"):
        frame = frame.strip("\n")
        if not frame or frame.startswith(":"):
            continue
        ev_type = "message"
        data_lines: list[str] = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                ev_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())
        data: dict | None = None
        if data_lines:
            try:
                data = json.loads("".join(data_lines))
            except json.JSONDecodeError:
                data = None
        out.append((ev_type, data))
    return out


class _FakeRequest:
    """Minimal stand-in for FastAPI ``Request`` objects.

    The SSE generator only consumes ``request.app.state`` (via
    ``_bus_from_request``) and ``request.is_disconnected()``; everything
    else on Request is irrelevant.  Using a fake here lets us drive
    ``_sse_event_generator`` deterministically without httpx
    ``ASGITransport``, which buffers/coalesces streamed bytes in a way
    that interacts poorly with long-lived ``request.is_disconnected()``
    polling under asyncio.
    """

    def __init__(self, app) -> None:
        self.app = app
        self._disconnected = False

    def disconnect(self) -> None:
        self._disconnected = True

    async def is_disconnected(self) -> bool:
        return self._disconnected


async def _read_one_frame(gen, *, timeout: float = 2.0) -> tuple[str, dict | None]:
    """Pull bytes from the SSE generator until we see one full SSE frame
    (terminated by ``\\n\\n``) and return its parsed (event_type, data).

    Heartbeats (``: ping\\n\\n``) are skipped so callers can wait for the
    next *real* event.
    """
    buf = b""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise AssertionError(f"no full SSE frame in {timeout}s; buf={buf!r}")
        chunk = await asyncio.wait_for(gen.__anext__(), timeout=remaining)
        buf += chunk
        if b"\n\n" in buf:
            parsed = _parse_sse_chunks(buf.decode())
            if parsed:
                return parsed[0]
            buf = b""  # was a heartbeat; keep reading


async def _wait_until_subscribed(
    bus: TournamentEventBus, tournament_id: int, *, timeout: float = 1.0
) -> None:
    """Block until at least one queue is registered for ``tournament_id``.

    The SSE generator only enters its ``bus.subscribe`` block on the
    *second* iteration past the initial snapshot. Tests need to wait
    for that registration before publishing, otherwise events are
    dropped to no subscribers (events are ephemeral, AD-7).
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        # ``_subscribers`` is private but stable across the codebase.
        if bus._subscribers.get(tournament_id):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"no subscriber for tournament {tournament_id} after {timeout}s"
    )


class TestDashboardSSEEndpoint:
    """SSE generator tests driven directly (no httpx) to avoid the
    ASGITransport buffering / disconnect-polling pitfalls. Visibility
    gating is exercised separately via the JSON endpoint and a
    targeted HTTP test below.
    """

    @pytest.mark.anyio
    async def test_initial_snapshot_emitted_first(
        self,
        sse_app,
        sse_bus: TournamentEventBus,
        db_session: AsyncSession,
        disable_dashboard_auth,
        _patch_fresh_helpers,
    ) -> None:
        """The very first frame on a fresh stream is an ``event: snapshot``
        with the empty initial DashboardPayload (no rounds resolved).
        """
        t = await _make_pending_el_farol(db_session, join_token=None)

        gen = tournament_live_module._sse_event_generator(_FakeRequest(sse_app), t.id)
        try:
            ev_type, data = await _read_one_frame(gen)
            assert ev_type == "snapshot"
            assert data is not None
            assert data["DATA"] == []
            assert data["NUM_DAYS"] == 0
        finally:
            await gen.aclose()

    @pytest.mark.anyio
    async def test_round_ended_event_triggers_new_snapshot(
        self,
        sse_app,
        sse_bus: TournamentEventBus,
        test_database: Database,
        db_session: AsyncSession,
        disable_dashboard_auth,
        _patch_fresh_helpers,
    ) -> None:
        """After ``round_ended`` lands on the bus, the generator
        re-projects and emits another ``event: snapshot``.
        """
        from datetime import datetime

        t = await _make_pending_el_farol(db_session, join_token=None)

        gen = tournament_live_module._sse_event_generator(_FakeRequest(sse_app), t.id)
        try:
            # 1. Initial empty snapshot.
            ev_type, data = await _read_one_frame(gen)
            assert ev_type == "snapshot"
            assert data["NUM_DAYS"] == 0

            # The subscribe context only opens AFTER the initial yield is
            # consumed.  Drive the generator forward in a background task
            # so it actually reaches ``bus.subscribe`` -- otherwise our
            # publish below races the registration and the event gets
            # dropped (events are ephemeral, AD-7).
            frames: list[bytes] = []

            async def _drain():
                async for chunk in gen:
                    frames.append(chunk)

            drain_task = asyncio.create_task(_drain())
            await _wait_until_subscribed(sse_bus, t.id)

            # 2. Seed a resolved round, then publish round_ended.
            async with test_database.session_factory() as s:
                t2 = await s.get(Tournament, t.id)
                await _add_resolved_round(s, t2, round_number=1)

            await sse_bus.publish(
                TournamentEvent(
                    event_type="round_ended",
                    tournament_id=t.id,
                    round_number=1,
                    data={
                        "tournament_completed": False,
                        "next_round_number": 2,
                    },
                    timestamp=datetime.now(),
                )
            )

            # 3. Wait for a snapshot frame with NUM_DAYS=1.
            deadline = asyncio.get_event_loop().time() + 3.0
            while True:
                if asyncio.get_event_loop().time() > deadline:
                    pytest.fail(
                        f"no NUM_DAYS=1 snapshot after round_ended; frames={frames!r}"
                    )
                buf = b"".join(frames).decode()
                parsed = _parse_sse_chunks(buf)
                day1_snapshots = [
                    e
                    for e in parsed
                    if e[0] == "snapshot"
                    and e[1] is not None
                    and e[1].get("NUM_DAYS") == 1
                ]
                if day1_snapshots:
                    break
                await asyncio.sleep(0.05)
        finally:
            drain_task.cancel()
            try:
                await drain_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            try:
                await gen.aclose()
            except RuntimeError:
                pass

    @pytest.mark.anyio
    async def test_tournament_completed_event_emits_completed_and_closes(
        self,
        sse_app,
        sse_bus: TournamentEventBus,
        db_session: AsyncSession,
        disable_dashboard_auth,
        _patch_fresh_helpers,
    ) -> None:
        """``tournament_completed`` produces a final snapshot, an
        ``event: completed`` frame, and ends the generator.
        """
        from datetime import datetime

        t = await _make_pending_el_farol(db_session, join_token=None)

        gen = tournament_live_module._sse_event_generator(_FakeRequest(sse_app), t.id)
        try:
            # Initial snapshot.
            ev0, _ = await _read_one_frame(gen)
            assert ev0 == "snapshot"

            frames: list[bytes] = []

            async def _drain():
                async for chunk in gen:
                    frames.append(chunk)

            drain_task = asyncio.create_task(_drain())
            await _wait_until_subscribed(sse_bus, t.id)

            await sse_bus.publish(
                TournamentEvent(
                    event_type="tournament_completed",
                    tournament_id=t.id,
                    round_number=None,
                    data={"final_scores": {}},
                    timestamp=datetime.now(),
                )
            )

            # The generator emits snapshot + completed and closes; the
            # drain task completes when StopAsyncIteration fires.
            await asyncio.wait_for(drain_task, timeout=3.0)

            buf = b"".join(frames).decode()
            parsed = _parse_sse_chunks(buf)
            event_types = [e[0] for e in parsed]
            assert "snapshot" in event_types
            assert "completed" in event_types
            # The completed frame should be the last with payload {"match_id": None}.
            completed = [e for e in parsed if e[0] == "completed"]
            assert completed
            assert completed[-1][1] == {"match_id": None}
        finally:
            await gen.aclose()

    @pytest.mark.anyio
    async def test_anonymous_private_stream_returns_404(
        self,
        sse_app,
        sse_bus: TournamentEventBus,
        db_session: AsyncSession,
        disable_dashboard_auth,
        _patch_fresh_helpers,
    ) -> None:
        """The HTTP gate (visibility check) 404s a private tournament for
        an anonymous viewer, before the streaming generator ever runs.
        """
        t = await _make_pending_el_farol(db_session, join_token="locked")

        async with AsyncClient(
            transport=ASGITransport(app=sse_app), base_url="http://test"
        ) as client:
            resp = await client.get(f"/api/v1/tournaments/{t.id}/dashboard/stream")

        assert resp.status_code == 404
