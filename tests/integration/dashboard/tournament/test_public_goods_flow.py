"""End-to-end tournament flow for Public Goods.

Mirrors ``test_el_farol_flow.py`` — both are N-player simultaneous
games that share the ``pending_submission`` polling semantic. Runs a
3-round 4-player match with varied contributions, a timeout on the
final round, and asserts the state snapshot lines up with the
server-side payoff math.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select, text

from atp.dashboard.models import User
from atp.dashboard.tournament.models import (
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.service import TournamentService


class _DummyBus:
    async def publish(self, event):
        pass


async def _seed_users(session, n: int = 4) -> list[User]:
    for uid in range(1, n + 1):
        uname = f"u{uid}"
        await session.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (:id, 'default', :u, :e, 'x', 1, 0, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"id": uid, "u": uname, "e": f"{uname}@test.com"},
        )
    users: list[User] = []
    for uid in range(1, n + 1):
        users.append(await session.get(User, uid))
    return users


@pytest.mark.anyio
async def test_public_goods_full_flow_n4_r3(session_factory):
    async with session_factory() as session:
        users = await _seed_users(session, 4)
        await session.commit()

        svc = TournamentService(session, _DummyBus())
        t, _ = await svc.create_tournament(
            creator=users[0],
            name="e2e-pg",
            game_type="public_goods",
            num_players=4,
            total_rounds=3,
            round_deadline_s=60,
        )
        for u in users:
            await svc.join(t.id, u, agent_name=u.username)
        await session.commit()

        # All joined → tournament is ACTIVE.
        t = await session.get(Tournament, t.id)
        await session.refresh(t)
        assert t.status == TournamentStatus.ACTIVE

        # Round 1: varied contributions — one free-rider, one full contributor
        for u, contrib in zip(users, [20.0, 10.0, 5.0, 0.0], strict=True):
            await svc.submit_action(t.id, u, action={"contribution": contrib})
        await session.commit()

        # Round 2: everyone cooperates (full contribution)
        for u in users:
            await svc.submit_action(t.id, u, action={"contribution": 20.0})
        await session.commit()

        # Round 3: only 3 submit; 4th times out → force-resolve uses the
        # timeout default (contribution=0).
        for u in users[:3]:
            await svc.submit_action(t.id, u, action={"contribution": 15.0})
        await session.commit()

        active = (
            await session.execute(
                select(Round)
                .where(Round.tournament_id == t.id)
                .where(Round.status == RoundStatus.WAITING_FOR_ACTIONS)
            )
        ).scalar_one()
        await svc.force_resolve_round(active.id)
        await session.commit()

        await session.refresh(t)
        assert t.status == TournamentStatus.COMPLETED

        # State snapshot for player 0:
        # - four participants
        # - PG uses pending_submission (N-player), not your_turn
        # - history has 3 completed rounds, visible to everyone
        state = await svc.get_state_for(t.id, users[0])
        assert state.game_type == "public_goods"
        assert state.num_players == 4
        assert len(state.all_scores) == 4
        assert state.pending_submission is False
        assert len(state.your_history) == 3
        assert len(state.all_contributions_by_round) == 3

        # Spot-check round 2: everyone contributed 20 → payoff = 1.6 * 80 / 4 = 32
        # (discount_factor=1.0 default, so no extra scaling)
        r2_contributions = state.all_contributions_by_round[1]
        assert r2_contributions == [20.0, 20.0, 20.0, 20.0]

        # Round 3: last player timed out → their contribution should be 0.0
        r3_contributions = state.all_contributions_by_round[2]
        assert r3_contributions[:3] == [15.0, 15.0, 15.0]
        assert r3_contributions[3] == 0.0
