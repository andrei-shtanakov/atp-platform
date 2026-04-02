"""Tournament API routes.

Provides endpoints for tournament listing and details, with stubs
for join, current-round, action, and results endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from atp.dashboard.tournament.models import Tournament
from atp.dashboard.tournament.schemas import (
    ActionRequest,
    JoinRequest,
    TournamentResponse,
)
from atp.dashboard.v2.dependencies import DBSession

router = APIRouter(prefix="/v1/tournaments", tags=["tournaments"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _tournament_to_response(t: Tournament) -> TournamentResponse:
    return TournamentResponse(
        id=t.id,
        game_type=t.game_type,
        status=t.status,
        starts_at=(t.starts_at.isoformat() if t.starts_at else None),
        ends_at=t.ends_at.isoformat() if t.ends_at else None,
    )


# ------------------------------------------------------------------
# Implemented endpoints
# ------------------------------------------------------------------


@router.get("", response_model=list[TournamentResponse])
async def list_tournaments(
    session: DBSession,
) -> list[TournamentResponse]:
    """List all tournaments."""
    result = await session.execute(select(Tournament).order_by(Tournament.id))
    tournaments = result.scalars().all()
    return [_tournament_to_response(t) for t in tournaments]


@router.get(
    "/{tournament_id}",
    response_model=TournamentResponse,
)
async def get_tournament(
    tournament_id: int,
    session: DBSession,
) -> TournamentResponse:
    """Get tournament details by id."""
    t = await session.get(Tournament, tournament_id)
    if t is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tournament {tournament_id} not found",
        )
    return _tournament_to_response(t)


# ------------------------------------------------------------------
# Stub endpoints (not yet implemented)
# ------------------------------------------------------------------


@router.post(
    "/{tournament_id}/join",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def join_tournament(
    tournament_id: int,
    data: JoinRequest,
) -> dict[str, str]:
    """Join a tournament (stub)."""
    return {"detail": "Not implemented"}


@router.get(
    "/{tournament_id}/current-round",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def current_round(
    tournament_id: int,
) -> dict[str, str]:
    """Get current round (stub)."""
    return {"detail": "Not implemented"}


@router.post(
    "/{tournament_id}/action",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def submit_action(
    tournament_id: int,
    data: ActionRequest,
) -> dict[str, str]:
    """Submit an action in the current round (stub)."""
    return {"detail": "Not implemented"}


@router.get(
    "/{tournament_id}/results",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def get_results(
    tournament_id: int,
) -> dict[str, str]:
    """Get tournament results (stub)."""
    return {"detail": "Not implemented"}
