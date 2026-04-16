"""Access-control helpers for tournament data.

Single source of truth for authorization rules that span multiple render
surfaces (UI template, JSON API, reports). Avoids drift between if-chains
scattered across routes.
"""

from __future__ import annotations

from atp.dashboard.models import User
from atp.dashboard.tournament.models import Tournament, TournamentStatus


def can_view_reasoning(
    *,
    user: User | None,
    tournament: Tournament,
    action_user_id: int | None,
) -> bool:
    """Return True if ``user`` may read an ``Action.reasoning`` value.

    Rules (evaluated in order):

    - After ``tournament.status == COMPLETED``: always visible. The caller
      must have already passed a tournament-visibility check upstream
      (e.g. private tournaments gate on ``join_token`` / ownership).
    - During live play (``PENDING`` / ``ACTIVE``):
        * admins see everything;
        * the tournament creator sees everything;
        * the agent sees its own reasoning (when ``action_user_id`` matches).
    - Anonymous callers see nothing during live play.
    """
    if tournament.status == TournamentStatus.COMPLETED:
        return True
    if user is None:
        return False
    if getattr(user, "is_admin", False):
        return True
    if tournament.created_by == user.id:
        return True
    return action_user_id is not None and action_user_id == user.id
