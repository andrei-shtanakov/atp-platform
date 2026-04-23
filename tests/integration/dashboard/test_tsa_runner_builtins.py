"""LABS-TSA PR-4 — runner branches on builtin_strategy.

Validates the synchronous builtin path inside ``TournamentService``:
builtin participants never submit via MCP, so before a round is
resolved, the service synthesises an ``Action`` for every builtin
slot by calling ``Strategy.choose_action(observation)``.

Determinism: resolving the same (tournament_id, participant_id)
twice yields the same action (stable blake2b seed — not
PYTHONHASHSEED-random ``hash()``).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.database import Database
from atp.dashboard.models import Agent
from atp.dashboard.tournament.events import TournamentEventBus
from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Round,
    RoundStatus,
    Tournament,
    TournamentStatus,
)
from atp.dashboard.tournament.service import TournamentService, _utc_now


class _NullBus(TournamentEventBus):
    """In-process event bus that drops everything — tests don't care."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        pass

    async def publish(self, event) -> None:  # type: ignore[override]
        return None

    async def subscribe(self, tournament_id):  # type: ignore[override]
        return  # pragma: no cover
        yield

    async def unsubscribe(self, tournament_id, queue) -> None:  # type: ignore[override]
        return None  # pragma: no cover


@pytest.fixture
async def session(test_database: Database) -> AsyncGenerator[AsyncSession, None]:
    async with test_database.session_factory() as s:
        yield s


async def _setup_2p_el_farol(
    session: AsyncSession,
) -> tuple[Tournament, Participant, Participant, Round, Agent]:
    """Create: 1 tournament, 1 agent-backed participant, 1 builtin, 1 round.

    Returns (tournament, agent_participant, builtin_participant, round, agent).
    """
    from atp.dashboard.models import User

    user = User(username="tester", email="t@x", hashed_password="x", is_active=True)
    session.add(user)
    await session.flush()

    agent = Agent(
        tenant_id="default",
        name="tester-agent",
        agent_type="mcp",
        owner_id=user.id,
        config={},
        purpose="tournament",
    )
    session.add(agent)
    await session.flush()

    now = _utc_now()
    t = Tournament(
        game_type="el_farol",
        status=TournamentStatus.ACTIVE,
        num_players=2,
        total_rounds=1,
        round_deadline_s=30,
        created_by=user.id,
        pending_deadline=now,
        config={"name": "t"},
    )
    session.add(t)
    await session.flush()

    p_agent = Participant(
        tournament_id=t.id,
        user_id=user.id,
        agent_id=agent.id,
        agent_name="tester-agent",
    )
    p_builtin = Participant(
        tournament_id=t.id,
        user_id=None,
        agent_id=None,
        agent_name="el_farol/traditionalist",
        builtin_strategy="el_farol/traditionalist",
    )
    session.add_all([p_agent, p_builtin])
    await session.flush()

    r = Round(
        tournament_id=t.id,
        round_number=1,
        status=RoundStatus.WAITING_FOR_ACTIONS,
        started_at=now,
        deadline=now + timedelta(seconds=30),
        state={},
    )
    session.add(r)
    await session.flush()

    return t, p_agent, p_builtin, r, agent


class TestRunnerBranch:
    @pytest.mark.anyio
    async def test_builtin_participant_resolved_synchronously(
        self, session: AsyncSession
    ) -> None:
        t, p_agent, p_builtin, r, _agent = await _setup_2p_el_farol(session)
        svc = TournamentService(session=session, bus=_NullBus())

        # The agent-backed participant submits first. Before resolution,
        # the service must synthesise the builtin's action.
        session.add(
            Action(
                round_id=r.id,
                participant_id=p_agent.id,
                action_data={"slots": [0, 1, 2]},
            )
        )
        await session.flush()

        await svc._ensure_builtin_actions(r, t)

        actions = (
            (await session.execute(select(Action).where(Action.round_id == r.id)))
            .scalars()
            .all()
        )
        pids = {a.participant_id for a in actions}
        assert p_builtin.id in pids
        # Builtin action must be a well-formed dict for the game.
        builtin_action = next(a for a in actions if a.participant_id == p_builtin.id)
        assert isinstance(builtin_action.action_data, dict)

    @pytest.mark.anyio
    async def test_same_participant_and_tournament_gives_same_action(
        self, session: AsyncSession
    ) -> None:
        """Determinism: rebuilding the same (t,p) pair yields the same action.

        Uses a seeded Gambler (namespaced ``el_farol/gambler``). The
        seed is derived inside ``resolve_builtin`` via blake2b of the
        identity pair — stable across process boundaries.
        """
        from atp.dashboard.tournament.builtins import resolve_builtin

        # Build the same Observation snapshot twice and compare actions.
        strat_a = resolve_builtin(
            "el_farol/gambler", tournament_id=42, participant_id=9
        )
        strat_b = resolve_builtin(
            "el_farol/gambler", tournament_id=42, participant_id=9
        )
        # choose_action against an empty observation state on day 0.
        from game_envs.core.state import Observation

        obs = Observation(
            player_id="1",
            game_state={"attendance_history": [], "num_slots": 16},
            available_actions=[],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        a = strat_a.choose_action(obs)
        b = strat_b.choose_action(obs)
        assert a == b
