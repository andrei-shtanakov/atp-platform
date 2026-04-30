"""Tournament visibility predicate shared by UI and live-stream routes.

The rule mirrors the SQL filter in ``ui.py::ui_matches::_apply_visibility``
and the post-fetch helper ``_match_visible_to_user``. Both gate access to
match detail; the live tournament dashboard and SSE stream share this
same predicate so an anonymous spectator can watch a public tournament
without being able to enumerate private ones.

A tournament is visible to a viewer when ANY of:
  * ``tournament.join_token IS NULL`` (public tournament)
  * the viewer is an admin
  * the viewer is the tournament's creator
  * the viewer is a Participant of the tournament

Anonymous viewers (``user is None``) only pass the first clause.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atp.dashboard.models import User
from atp.dashboard.tournament.models import Participant, Tournament


async def is_tournament_visible_to(
    session: AsyncSession,
    tournament: Tournament,
    user: User | None,
) -> bool:
    """Return True iff ``tournament`` should be visible to ``user``."""
    if tournament.join_token is None:
        return True
    if user is None:
        return False
    if user.is_admin or tournament.created_by == user.id:
        return True
    participant_id = await session.scalar(
        select(Participant.id)
        .where(Participant.tournament_id == tournament.id)
        .where(Participant.user_id == user.id)
        .limit(1)
    )
    return participant_id is not None
