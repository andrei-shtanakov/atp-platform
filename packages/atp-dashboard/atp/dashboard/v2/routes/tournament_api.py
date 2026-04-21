"""Tournament REST admin endpoints.

6 handlers for tournament lifecycle management:
  GET    /api/v1/tournaments              — list (visibility-filtered)
  GET    /api/v1/tournaments/{id}         — detail
  GET    /api/v1/tournaments/{id}/rounds  — round history with nested
                                            per-action data (action_data,
                                            payoff, reasoning; reasoning is
                                            gated — see access.can_view_reasoning)
  GET    /api/v1/tournaments/{id}/participants — participant list
  POST   /api/v1/tournaments             — create (returns join_token once)
  POST   /api/v1/tournaments/{id}/cancel — cancel

All handlers use the shared TournamentService. Transaction boundary is owned
by the FastAPI DB dependency (ambient autobegin + commit on success).
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.models import User
from atp.dashboard.tournament.access import can_view_reasoning
from atp.dashboard.tournament.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Round,
    TournamentStatus,
)
from atp.dashboard.tournament.service import TournamentService
from atp.dashboard.v2.dependencies import DBSession, get_db_session

router = APIRouter(prefix="/v1/tournaments", tags=["tournaments"])


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


async def get_current_user_for_tournament(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> User:
    """Resolve the calling user for tournament endpoints.

    JWTUserStateMiddleware decodes the Bearer JWT and populates
    ``request.state.user_id`` (integer) — this dependency then loads
    the corresponding User row from the DB and verifies it is active.
    In test mode (``ATP_DISABLE_AUTH=true``) falls back to user id=1.
    """
    user_id: int | None = getattr(request.state, "user_id", None)
    if user_id is not None:
        user = await session.get(User, user_id)
        if user is not None and user.is_active:
            return user

    if os.environ.get("ATP_DISABLE_AUTH") == "true":
        loaded = await session.get(User, 1)
        if loaded is not None:
            return loaded

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="unauthenticated",
    )


async def get_tournament_service(
    session: DBSession,
    request: Request,
) -> TournamentService:
    """Provide a TournamentService bound to the request's DB session."""
    from atp.dashboard.mcp import tournament_event_bus

    bus = getattr(request.app.state, "tournament_event_bus", tournament_event_bus)
    return TournamentService(session=session, bus=bus)


TournamentUser = Annotated[User, Depends(get_current_user_for_tournament)]
TournamentSvc = Annotated[TournamentService, Depends(get_tournament_service)]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateTournamentRequest(BaseModel):
    """Payload for creating a new tournament."""

    name: str
    game_type: str = "prisoners_dilemma"
    num_players: int = Field(ge=2)
    total_rounds: int = Field(ge=1)
    round_deadline_s: int = Field(ge=1)
    private: bool = False


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _serialize(t: Any, is_admin: bool) -> dict[str, Any]:
    """Serialize a Tournament ORM object to a response dict.

    join_token is NEVER included here — it is added only on creation.
    """
    base: dict[str, Any] = {
        "id": t.id,
        "name": (t.config or {}).get("name", ""),
        "status": t.status if isinstance(t.status, str) else t.status.value,
        "game_type": t.game_type,
        "num_players": t.num_players,
        "total_rounds": t.total_rounds,
        "round_deadline_s": t.round_deadline_s,
        "has_join_token": bool(t.join_token),
        "cancelled_reason": (
            t.cancelled_reason.value if t.cancelled_reason is not None else None
        ),
        "cancelled_reason_detail": t.cancelled_reason_detail,
    }
    if is_admin:
        base["cancelled_by"] = t.cancelled_by
    return base


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_tournaments_endpoint(
    user: TournamentUser,
    service: TournamentSvc,
    status_filter: str | None = None,
) -> dict[str, Any]:
    """List tournaments visible to the calling user."""
    filt: TournamentStatus | None = None
    if status_filter is not None:
        try:
            filt = TournamentStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unknown status filter: {status_filter!r}",
            )
    tournaments = await service.list_tournaments(user=user, status=filt)
    return {"tournaments": [_serialize(t, user.is_admin) for t in tournaments]}


@router.get("/{tournament_id}")
async def get_tournament_endpoint(
    tournament_id: int,
    user: TournamentUser,
    service: TournamentSvc,
) -> dict[str, Any]:
    """Return tournament details (visibility-filtered)."""
    try:
        t = await service.get_tournament(tournament_id, user)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="tournament not found",
        )
    return _serialize(t, user.is_admin)


@router.get("/{tournament_id}/rounds")
async def get_rounds_endpoint(
    tournament_id: int,
    user: TournamentUser,
    service: TournamentSvc,
    session: DBSession,
) -> dict[str, Any]:
    """Return round history with nested per-action data.

    Response shape (each round):

        {"round_number": N, "status": "completed|...",
         "actions": [
             {"agent_name": str, "action_data": {...},
              "payoff": float | null, "reasoning": str | null},
             ...
         ]}

    ``reasoning`` is masked to ``null`` for viewers who fail the gate in
    ``atp.dashboard.tournament.access.can_view_reasoning`` (e.g. non-owners
    reading opponent rows during live play). The original minimal shape
    (``round_number`` + ``status``) is a strict subset, so legacy clients
    that ignore the ``actions`` key keep working.
    """
    try:
        t = await service.get_tournament(tournament_id, user)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="tournament not found",
        )

    # Mirror service.get_history() cap: latest 100 rounds, returned in
    # ascending order. Prevents unbounded response growth now that nested
    # actions (with action_data + reasoning) inflate per-round payload.
    stmt = (
        select(Round)
        .where(Round.tournament_id == tournament_id)
        .order_by(Round.round_number.desc())
        .limit(100)
        .options(
            selectinload(Round.actions).selectinload(Action.participant),
        )
    )
    rounds = sorted(
        (await session.scalars(stmt)).all(),
        key=lambda r: r.round_number,
    )

    return {
        "rounds": [
            {
                "round_number": r.round_number,
                "status": (r.status if isinstance(r.status, str) else r.status.value),
                "actions": [
                    {
                        "agent_name": a.participant.agent_name,
                        "action_data": a.action_data,
                        "payoff": a.payoff,
                        "reasoning": (
                            a.reasoning
                            if can_view_reasoning(
                                user=user,
                                tournament=t,
                                action_user_id=a.participant.user_id,
                            )
                            else None
                        ),
                    }
                    for a in sorted(r.actions, key=lambda x: x.participant_id)
                ],
            }
            for r in rounds
        ]
    }


@router.get("/{tournament_id}/participants")
async def get_participants_endpoint(
    tournament_id: int,
    user: TournamentUser,
    service: TournamentSvc,
    session: DBSession,
) -> dict[str, Any]:
    """Return participants of a tournament."""
    try:
        await service.get_tournament(tournament_id, user)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="tournament not found",
        )

    # Explicitly load participants to avoid lazy-loading in async context.
    result = await session.execute(
        select(Participant)
        .where(Participant.tournament_id == tournament_id)
        .order_by(Participant.id)
    )
    raw_participants = result.scalars().all()

    participants: list[dict[str, Any]] = []
    for p in raw_participants:
        row: dict[str, Any] = {
            "id": p.id,
            "user_id": p.user_id,
            "agent_name": p.agent_name,
        }
        if p.user_id == user.id or user.is_admin:
            row["released_at"] = p.released_at.isoformat() if p.released_at else None
        participants.append(row)
    return {"participants": participants}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_tournament_endpoint(
    req: CreateTournamentRequest,
    user: TournamentUser,
    service: TournamentSvc,
) -> dict[str, Any]:
    """Create a tournament. Returns join_token once (private tournaments only)."""
    try:
        tournament, join_token = await service.create_tournament(
            creator=user,
            name=req.name,
            game_type=req.game_type,
            num_players=req.num_players,
            total_rounds=req.total_rounds,
            round_deadline_s=req.round_deadline_s,
            private=req.private,
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    response = _serialize(tournament, user.is_admin)
    response["join_token"] = join_token  # None for public, token string for private
    return response


@router.post("/{tournament_id}/cancel")
async def cancel_tournament_endpoint(
    tournament_id: int,
    user: TournamentUser,
    service: TournamentSvc,
) -> dict[str, Any]:
    """Cancel a tournament (admin or owner only)."""
    try:
        await service.cancel_tournament(user=user, tournament_id=tournament_id)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="tournament not found",
        )
    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    return {"cancelled": True}


@router.delete(
    "/{tournament_id}/participants/{participant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def kick_participant_endpoint(
    tournament_id: int,
    participant_id: int,
    user: TournamentUser,
    service: TournamentSvc,
) -> None:
    """Kick a participant from a live tournament (admin only).

    Sets Participant.released_at and, for in-progress rounds, inserts a
    TIMEOUT_DEFAULT action so round resolution is not blocked.
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    try:
        await service.kick_participant(tournament_id, participant_id)
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="participant not found",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
