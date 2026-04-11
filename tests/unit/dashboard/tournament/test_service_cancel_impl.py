"""Unit tests for TournamentService._cancel_impl — the private cancel
helper. Exercises control flow, idempotent guard, event building, and
the step-3-before-step-6 ordering regression."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from atp.dashboard.tournament.events import TournamentCancelEvent
from atp.dashboard.tournament.models import Tournament, TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason
from atp.dashboard.tournament.service import TournamentService


@pytest.fixture
def frozen_now(monkeypatch):
    fixed = datetime(2026, 4, 15, 10, 0, 0)

    class _FakeDatetime:
        @classmethod
        def utcnow(cls):
            return fixed

    monkeypatch.setattr("atp.dashboard.tournament.service.datetime", _FakeDatetime)
    return fixed


def _tournament(status: TournamentStatus) -> Tournament:
    t = MagicMock(spec=Tournament)
    t.id = 1
    t.status = status
    t.cancelled_at = None
    t.cancelled_by = None
    t.cancelled_reason = None
    t.cancelled_reason_detail = None
    return t


def _make_service(tournament: Tournament | None, completed_round_count: int = 0):
    session = MagicMock()
    session.get = AsyncMock(return_value=tournament)
    session.execute = AsyncMock()
    session.scalar = AsyncMock(return_value=completed_round_count)
    bus = MagicMock()
    return TournamentService(session=session, bus=bus), session


@pytest.mark.anyio
async def test_happy_path_pending_to_cancelled(frozen_now):
    t = _tournament(TournamentStatus.PENDING)
    svc, session = _make_service(t, completed_round_count=0)

    event = await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.ADMIN_ACTION,
        cancelled_by=42,
        reason_detail="stuck",
    )

    assert t.status == TournamentStatus.CANCELLED
    assert t.cancelled_at == frozen_now
    assert t.cancelled_by == 42
    assert t.cancelled_reason == CancelReason.ADMIN_ACTION
    assert t.cancelled_reason_detail == "stuck"
    assert isinstance(event, TournamentCancelEvent)
    assert event.tournament_id == 1
    assert event.final_rounds_played == 0


@pytest.mark.anyio
async def test_happy_path_active_to_cancelled(frozen_now):
    t = _tournament(TournamentStatus.ACTIVE)
    svc, _ = _make_service(t, completed_round_count=3)

    event = await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.ADMIN_ACTION,
        cancelled_by=42,
        reason_detail=None,
    )
    assert event is not None
    assert event.final_rounds_played == 3


@pytest.mark.anyio
async def test_idempotent_on_already_cancelled():
    t = _tournament(TournamentStatus.CANCELLED)
    svc, session = _make_service(t)

    event = await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.ADMIN_ACTION,
        cancelled_by=42,
        reason_detail=None,
    )
    assert event is None
    # No bulk updates executed when tournament already terminal
    session.execute.assert_not_called()


@pytest.mark.anyio
async def test_idempotent_on_already_completed():
    t = _tournament(TournamentStatus.COMPLETED)
    svc, session = _make_service(t)
    event = await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.ADMIN_ACTION,
        cancelled_by=42,
        reason_detail=None,
    )
    assert event is None
    session.execute.assert_not_called()


@pytest.mark.anyio
async def test_final_rounds_played_snapshots_before_bulk_update(frozen_now):
    """Regression guard for step-3-before-step-6 ordering. If anyone
    reorders _cancel_impl such that the scalar count runs AFTER the
    bulk UPDATE that transitions in-flight rounds to CANCELLED, this
    test fires — because the count would then reflect post-update
    state instead of pre-update state."""
    t = _tournament(TournamentStatus.ACTIVE)
    svc, session = _make_service(t, completed_round_count=2)

    # Record the order in which session methods are called
    call_order: list[str] = []
    original_scalar = session.scalar
    original_execute = session.execute

    async def record_scalar(*args, **kwargs):
        call_order.append("scalar")
        return await original_scalar(*args, **kwargs)

    async def record_execute(*args, **kwargs):
        call_order.append("execute")
        return await original_execute(*args, **kwargs)

    session.scalar = record_scalar
    session.execute = record_execute

    await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.ADMIN_ACTION,
        cancelled_by=42,
        reason_detail=None,
    )

    # scalar (final_rounds_played snapshot) must precede both bulk UPDATEs
    assert call_order[0] == "scalar"
    assert "execute" in call_order
    assert call_order.index("scalar") < call_order.index("execute")


@pytest.mark.anyio
async def test_system_cancel_with_none_actor_valid(frozen_now):
    t = _tournament(TournamentStatus.PENDING)
    svc, _ = _make_service(t)

    event = await svc._cancel_impl(
        tournament_id=1,
        reason=CancelReason.PENDING_TIMEOUT,
        cancelled_by=None,
        reason_detail=None,
    )
    assert event is not None
    assert event.cancelled_by is None
    assert event.cancelled_reason == CancelReason.PENDING_TIMEOUT
