"""End-to-end tournament flow for El Farol (spec §7.2)."""

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


async def _seed_users(session, n: int = 5) -> list[User]:
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
async def test_el_farol_full_flow_n5_r3(session_factory):
    async with session_factory() as session:
        users = await _seed_users(session, 5)
        await session.commit()

        svc = TournamentService(session, _DummyBus())
        t, _ = await svc.create_tournament(
            creator=users[0],
            name="e2e",
            game_type="el_farol",
            num_players=5,
            total_rounds=3,
            round_deadline_s=60,
        )
        for u in users:
            await svc.join(t.id, u, agent_name=u.username)
        await session.commit()

        # Tournament should be ACTIVE (all players joined)
        t = await session.get(Tournament, t.id)
        await session.refresh(t)
        assert t.status == TournamentStatus.ACTIVE

        # Round 1: varied slot picks
        for u, slots in zip(users, [[0, 1], [2, 3], [0], [4, 5], []], strict=True):
            await svc.submit_action(t.id, u, action={"slots": slots})
        await session.commit()

        # Round 2: all go to slot 0 (crowded)
        for u in users:
            await svc.submit_action(t.id, u, action={"slots": [0]})
        await session.commit()

        # Round 3: only 4 submit; 5th times out -> force resolve
        for u in users[:4]:
            await svc.submit_action(t.id, u, action={"slots": [1, 2]})
        await session.commit()

        # Find the active round and force-resolve
        active = (
            await session.execute(
                select(Round)
                .where(Round.tournament_id == t.id)
                .where(Round.status == RoundStatus.WAITING_FOR_ACTIONS)
            )
        ).scalar_one()
        await svc.force_resolve_round(active.id)
        await session.commit()

        # Tournament should be COMPLETED
        await session.refresh(t)
        assert t.status == TournamentStatus.COMPLETED

        # State for first player: all_scores length == 5, pending_submission False
        state = await svc.get_state_for(t.id, users[0])
        assert state.game_type == "el_farol"
        assert len(state.all_scores) == 5
        assert state.pending_submission is False
