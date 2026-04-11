"""Tests for the two public cancel entry points: cancel_tournament
(user-authenticated) and cancel_tournament_system (trusted internal)."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.errors import NotFoundError
from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import Tournament, TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService


def _tournament(created_by: int | None = 42) -> Tournament:
    t = MagicMock(spec=Tournament)
    t.id = 1
    t.created_by = created_by
    t.status = TournamentStatus.ACTIVE
    return t


def _user(user_id: int, is_admin: bool = False) -> MagicMock:
    u = MagicMock()
    u.id = user_id
    u.is_admin = is_admin
    return u


def _make_service(tournament, cancel_event_to_return=None):
    session = MagicMock()
    session.get = AsyncMock(return_value=tournament)
    session.begin = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock()
    session.begin.return_value.__aexit__ = AsyncMock(return_value=False)
    bus = MagicMock()
    bus.publish = AsyncMock()
    svc = TournamentService(session=session, bus=bus)

    async def fake_cancel_impl(*args, **kwargs):
        return cancel_event_to_return

    svc._cancel_impl = fake_cancel_impl  # type: ignore[method-assign]
    return svc, session, bus


def _event():
    return TournamentCancelEvent(
        tournament_id=1,
        cancelled_at=datetime(2026, 4, 15, 10, 0, 0),
        cancelled_by=42,
        cancelled_reason=CancelReason.ADMIN_ACTION,
        cancelled_reason_detail=None,
        final_rounds_played=0,
        final_status=TournamentStatus.CANCELLED,
    )


@pytest.mark.anyio
async def test_cancel_tournament_by_owner_publishes_event():
    t = _tournament(created_by=42)
    svc, session, bus = _make_service(t, cancel_event_to_return=_event())

    await svc.cancel_tournament(
        user=_user(42, is_admin=False),
        tournament_id=1,
        reason_detail="manual",
    )

    bus.publish.assert_awaited_once()


@pytest.mark.anyio
async def test_cancel_tournament_by_admin_publishes_event():
    t = _tournament(created_by=99)
    svc, session, bus = _make_service(t, cancel_event_to_return=_event())

    await svc.cancel_tournament(
        user=_user(42, is_admin=True),
        tournament_id=1,
        reason_detail=None,
    )

    bus.publish.assert_awaited_once()


@pytest.mark.anyio
async def test_cancel_tournament_by_non_owner_raises_not_found():
    t = _tournament(created_by=99)
    svc, session, bus = _make_service(t)

    with pytest.raises(NotFoundError):
        await svc.cancel_tournament(
            user=_user(42, is_admin=False),
            tournament_id=1,
            reason_detail=None,
        )
    bus.publish.assert_not_called()


@pytest.mark.anyio
async def test_cancel_tournament_publish_failure_does_not_raise():
    """Bus publish failure after successful commit returns success to caller.
    Per error handling matrix: DB is single source of truth, session_sync
    recovers subscribers."""
    t = _tournament(created_by=42)
    svc, session, bus = _make_service(t, cancel_event_to_return=_event())
    bus.publish = AsyncMock(side_effect=ConnectionError("bus down"))

    # Must NOT raise
    await svc.cancel_tournament(
        user=_user(42, is_admin=False),
        tournament_id=1,
        reason_detail=None,
    )


@pytest.mark.anyio
async def test_cancel_tournament_system_no_auth_required():
    t = _tournament(created_by=99)
    svc, session, bus = _make_service(t, cancel_event_to_return=_event())

    await svc.cancel_tournament_system(
        tournament_id=1,
        reason=CancelReason.PENDING_TIMEOUT,
    )

    bus.publish.assert_awaited_once()
