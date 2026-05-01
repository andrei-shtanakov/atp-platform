"""Isolated unit tests for TournamentService._play_roster_participants.

The helper returns the participant set the engine should see in any given
round: anyone currently in the lobby (released_at IS NULL) plus anyone
who has played at least one Action in this tournament (so a player kicked
at round N still appears in rounds N+1+ for participant-index stability).

Without this helper, two bugs reappear:
1. Pending leavers (released_at != NULL, 0 Actions) pollute post-shrink
   state and break engine arity.
2. A participant kicked mid-tournament would silently vanish from later
   rounds and shift participant indices, corrupting payoffs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.models import (
    Action,
    ActionSource,
    Round,
    RoundStatus,
)
from atp.dashboard.tournament.service import TournamentService


async def _make_user(session: AsyncSession, username: str) -> User:
    user = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password="x",
        is_admin=False,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    return user


async def _make_pending_tournament(
    session: AsyncSession,
    creator: User,
    bus: TournamentEventBus,
    *,
    num_players: int = 4,
):
    svc = TournamentService(session, bus)
    tournament, _ = await svc.create_tournament(
        creator=creator,
        name="play-roster-fixture",
        game_type="el_farol",
        num_players=num_players,
        total_rounds=2,
        round_deadline_s=30,
    )
    return svc, tournament


async def _seed_round(
    session: AsyncSession, tournament_id: int, round_number: int
) -> Round:
    now = datetime.now()
    rnd = Round(
        tournament_id=tournament_id,
        round_number=round_number,
        status=RoundStatus.WAITING_FOR_ACTIONS,
        started_at=now,
        deadline=now + timedelta(seconds=30),
        state={},
    )
    session.add(rnd)
    await session.commit()
    return rnd


async def _seed_action(
    session: AsyncSession, round_id: int, participant_id: int
) -> Action:
    action = Action(
        round_id=round_id,
        participant_id=participant_id,
        action_data={"intervals": []},
        source=ActionSource.SUBMITTED.value,
    )
    session.add(action)
    await session.commit()
    return action


@pytest.mark.anyio
async def test_excludes_pending_leaver_without_actions(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """A participant who joined and left in PENDING (released_at != NULL,
    0 Actions) must NOT appear in the gameplay roster."""
    svc, t = await _make_pending_tournament(session, admin_user, event_bus)

    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")
    await svc.leave(t.id, alice)

    roster = await svc._play_roster_participants(t.id)

    assert len(roster) == 1
    assert roster[0].agent_name == "bob"


@pytest.mark.anyio
async def test_includes_kicked_participant_with_played_actions(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """A participant kicked AFTER playing at least one Action must REMAIN
    in the gameplay roster so participant indices stay stable across rounds."""
    svc, t = await _make_pending_tournament(session, admin_user, event_bus)

    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    rnd = await _seed_round(session, t.id, round_number=1)
    alice_p = (await svc._play_roster_participants(t.id))[0]
    await _seed_action(session, rnd.id, alice_p.id)

    # Now kick alice (sets released_at).
    alice_p.released_at = datetime.now()
    await session.commit()

    roster = await svc._play_roster_participants(t.id)
    names = {p.agent_name for p in roster}

    assert names == {"alice", "bob"}


@pytest.mark.anyio
async def test_includes_currently_active_participants(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """Active participants (released_at IS NULL) are always included
    regardless of whether they have played any Action."""
    svc, t = await _make_pending_tournament(session, admin_user, event_bus)

    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")

    roster = await svc._play_roster_participants(t.id)

    assert len(roster) == 2
    assert {p.agent_name for p in roster} == {"alice", "bob"}


@pytest.mark.anyio
async def test_returns_in_id_order(
    session: AsyncSession,
    admin_user: User,
    alice: User,
    bob: User,
    event_bus: TournamentEventBus,
) -> None:
    """Roster ordering MUST be by Participant.id ascending so engine
    indexes remain stable across rounds."""
    charlie = await _make_user(session, "charlie")
    svc, t = await _make_pending_tournament(
        session, admin_user, event_bus, num_players=4
    )

    await svc.join(t.id, alice, "alice")
    await svc.join(t.id, bob, "bob")
    await svc.join(t.id, charlie, "charlie")

    roster = await svc._play_roster_participants(t.id)
    ids = [p.id for p in roster]

    assert ids == sorted(ids)
    assert [p.agent_name for p in roster] == ["alice", "bob", "charlie"]
