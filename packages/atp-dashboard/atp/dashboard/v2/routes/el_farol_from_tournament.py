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

from atp.dashboard.v2.routes.el_farol_dashboard import DashboardPayload


async def _reshape_from_tournament(
    tournament_id: int,
    session: AsyncSession,
) -> DashboardPayload:
    raise NotImplementedError("Task 1 stub; filled in Task 4")
