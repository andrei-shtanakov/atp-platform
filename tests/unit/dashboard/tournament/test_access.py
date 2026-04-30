"""Unit tests for the can_view_reasoning access-control helper."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from atp.dashboard.tournament.access import can_view_reasoning
from atp.dashboard.tournament.models import TournamentStatus


@dataclass
class _FakeUser:
    id: int
    is_admin: bool = False


@dataclass
class _FakeTournament:
    created_by: int
    status: TournamentStatus


OWNER_ID = 1
PARTICIPANT_A_ID = 2
PARTICIPANT_B_ID = 3
ADMIN_ID = 99


def _t(status: TournamentStatus) -> _FakeTournament:
    return _FakeTournament(created_by=OWNER_ID, status=status)


@pytest.mark.parametrize(
    "status",
    [TournamentStatus.PENDING, TournamentStatus.ACTIVE, TournamentStatus.CANCELLED],
)
def test_anon_denied_until_completed(status: TournamentStatus) -> None:
    assert (
        can_view_reasoning(
            user=None, tournament=_t(status), action_user_id=PARTICIPANT_A_ID
        )
        is False
    )


def test_anon_allowed_when_completed() -> None:
    assert (
        can_view_reasoning(
            user=None,
            tournament=_t(TournamentStatus.COMPLETED),
            action_user_id=PARTICIPANT_A_ID,
        )
        is True
    )


@pytest.mark.parametrize(
    "status",
    [
        TournamentStatus.PENDING,
        TournamentStatus.ACTIVE,
        TournamentStatus.COMPLETED,
        TournamentStatus.CANCELLED,
    ],
)
def test_admin_always_allowed(status: TournamentStatus) -> None:
    admin = _FakeUser(id=ADMIN_ID, is_admin=True)
    assert (
        can_view_reasoning(
            user=admin, tournament=_t(status), action_user_id=PARTICIPANT_A_ID
        )
        is True
    )


def test_owner_sees_own_during_active() -> None:
    owner = _FakeUser(id=OWNER_ID)
    assert (
        can_view_reasoning(
            user=owner,
            tournament=_t(TournamentStatus.ACTIVE),
            action_user_id=PARTICIPANT_A_ID,
        )
        is True
    )


def test_self_sees_own_during_active() -> None:
    agent = _FakeUser(id=PARTICIPANT_A_ID)
    assert (
        can_view_reasoning(
            user=agent,
            tournament=_t(TournamentStatus.ACTIVE),
            action_user_id=PARTICIPANT_A_ID,
        )
        is True
    )


def test_self_denied_others_during_active() -> None:
    agent = _FakeUser(id=PARTICIPANT_A_ID)
    assert (
        can_view_reasoning(
            user=agent,
            tournament=_t(TournamentStatus.ACTIVE),
            action_user_id=PARTICIPANT_B_ID,
        )
        is False
    )


def test_unrelated_user_denied_during_active() -> None:
    other = _FakeUser(id=4242)
    assert (
        can_view_reasoning(
            user=other,
            tournament=_t(TournamentStatus.ACTIVE),
            action_user_id=PARTICIPANT_A_ID,
        )
        is False
    )


def test_unrelated_user_allowed_after_completed() -> None:
    other = _FakeUser(id=4242)
    assert (
        can_view_reasoning(
            user=other,
            tournament=_t(TournamentStatus.COMPLETED),
            action_user_id=PARTICIPANT_A_ID,
        )
        is True
    )


def test_none_action_user_id_during_active_denies_non_admin() -> None:
    # Defensive: an action whose participant has no linked user shouldn't
    # leak to random callers during live play.
    agent = _FakeUser(id=PARTICIPANT_A_ID)
    assert (
        can_view_reasoning(
            user=agent,
            tournament=_t(TournamentStatus.ACTIVE),
            action_user_id=None,
        )
        is False
    )
