"""Tournament → DashboardPayload reshape for El Farol (LABS-106).

Mirror of ``el_farol_dashboard._reshape`` but reads authoritative
``Round``/``Action``/``Participant`` rows instead of the pre-serialised
``actions_json``/``day_aggregates_json`` on ``GameResult`` (which are
intentionally NULL for tournament-written rows — see
``TournamentService._write_game_result_for_tournament``).

Output model (``DashboardPayload``) is imported from
``el_farol_dashboard`` so any bump of ``SHAPE_VERSION`` is picked up
here automatically.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.tournament.models import Participant
from atp.dashboard.v2.routes.el_farol_dashboard import (
    DashboardAgent,
    DashboardPayload,
)

_DEFAULT_COLOR = "#6e7781"
_BUILTIN_PREFIX = "el_farol/"


def _build_agents_from_participants(
    participants: list[Participant],
) -> list[DashboardAgent]:
    """Convert tournament Participant rows to DashboardAgent instances.

    Strips the ``el_farol/`` prefix from ``builtin_strategy`` so the
    profile field reads as the bare strategy name (e.g. ``"traditionalist"``).
    Builtins with no ``user_id`` map to ``user="unknown"``; real users map
    to ``user=str(user_id)``.
    """
    agents: list[DashboardAgent] = []
    for p in participants:
        profile = ""
        if p.builtin_strategy and p.builtin_strategy.startswith(_BUILTIN_PREFIX):
            profile = p.builtin_strategy[len(_BUILTIN_PREFIX) :]
        elif p.builtin_strategy:
            profile = p.builtin_strategy
        agents.append(
            DashboardAgent(
                id=p.agent_name,
                color=_DEFAULT_COLOR,
                user=str(p.user_id) if p.user_id is not None else "unknown",
                profile=profile,
            )
        )
    return agents


async def _reshape_from_tournament(
    tournament_id: int,
    session: AsyncSession,
) -> DashboardPayload:
    raise NotImplementedError("Task 1 stub; filled in Task 4")
