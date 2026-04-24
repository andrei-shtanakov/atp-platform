"""Admin-gated UI routes for tournament management.

All routes are mounted under ``/ui/admin``. Authentication follows the
same pattern as the public ``/ui/*`` routes: ``JWTUserStateMiddleware``
populates ``request.state.user_id`` from a Bearer header or
``atp_token`` cookie, and each route calls ``_require_admin_ui_user``
to reject anonymous callers (401) and authenticated non-admins (403).

Routes added here:
- ``GET  /ui/admin``                              — admin landing
- ``GET  /ui/admin/tournaments``                   — full tournament list
- ``GET  /ui/admin/tournaments/new``               — create form
- ``POST /ui/admin/tournaments/new``               — submit form
- ``GET  /ui/admin/tournaments/{id}``              — detail (live or post-mortem)
- ``GET  /ui/admin/tournaments/{id}/activity``     — HTMX fragment (polled)
- ``GET  /ui/admin/users``                         — registered users + activity counts
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from atp.dashboard.models import Agent, User
from atp.dashboard.tournament.errors import ValidationError as TournamentValidationError
from atp.dashboard.tournament.models import Participant, Round, Tournament
from atp.dashboard.tournament.service import TournamentService
from atp.dashboard.v2.dependencies import DBSession
from atp.dashboard.v2.routes.tournament_api import get_tournament_service

router = APIRouter(prefix="/ui/admin", tags=["admin-ui"])


def _templates(request: Request):
    """Access the Jinja2Templates instance set up in the factory."""
    return request.app.state.templates


async def _require_admin_ui_user(request: Request, session: DBSession) -> User:
    """Resolve the UI caller and require ``is_admin=True``.

    Raises:
        HTTPException 401: caller is anonymous.
        HTTPException 403: caller is authenticated but not admin.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


@router.get("")
async def admin_home(request: Request, session: DBSession):
    """Admin landing page with quick-links to admin sections."""
    user = await _require_admin_ui_user(request, session)
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/admin/index.html",
        context={"user": user, "active_page": "admin"},
    )


@router.get("/tournaments")
async def admin_tournaments_list(request: Request, session: DBSession):
    """List every tournament (all statuses, all owners) for admins."""
    user = await _require_admin_ui_user(request, session)
    stmt = select(Tournament).order_by(Tournament.created_at.desc())
    result = await session.execute(stmt)
    tournaments = result.scalars().all()
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/admin/tournaments_list.html",
        context={
            "user": user,
            "tournaments": tournaments,
            "active_page": "admin",
        },
    )


@router.get("/tournaments/new")
async def admin_tournament_new_form(request: Request, session: DBSession):
    """Render the create-tournament form (El Farol only for MVP)."""
    user = await _require_admin_ui_user(request, session)
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/admin/tournament_new.html",
        context={"user": user, "error": None},
    )


@router.post("/tournaments/new")
async def admin_tournament_new_submit(
    request: Request,
    session: DBSession,
    service: TournamentService = Depends(get_tournament_service),
    name: str = Form(...),
    game_type: str = Form(...),
    num_players: int = Form(...),
    total_rounds: int = Form(...),
    round_deadline_s: int = Form(...),
):
    """Create a tournament and redirect to its admin detail page.

    El Farol's capacity threshold is derived from num_players inside
    the service (see ``_el_farol_for`` in tournament/service.py);
    the form does not expose it.
    """
    user = await _require_admin_ui_user(request, session)
    try:
        tournament, _join_token = await service.create_tournament(
            user,
            name=name,
            game_type=game_type,
            num_players=num_players,
            total_rounds=total_rounds,
            round_deadline_s=round_deadline_s,
        )
    except TournamentValidationError as exc:
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/admin/tournament_new.html",
            context={"user": user, "error": str(exc)},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    return RedirectResponse(
        url=f"/ui/admin/tournaments/{tournament.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


_LIVE_STATUSES = ("pending", "active")


@router.get("/tournaments/{tournament_id}")
async def admin_tournament_detail(
    tournament_id: int,
    request: Request,
    session: DBSession,
    service: TournamentService = Depends(get_tournament_service),
):
    """Render the admin detail page for a single tournament.

    Live (``pending`` / ``active``) tournaments get the Cancel button
    plus an HTMX-polled activity block. Post-mortem tournaments drop
    Cancel and server-render the activity block once.
    """
    user = await _require_admin_ui_user(request, session)
    stmt = (
        select(Tournament)
        .where(Tournament.id == tournament_id)
        .options(selectinload(Tournament.participants))
    )
    tournament = (await session.execute(stmt)).scalars().first()
    if tournament is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    # Derive the current round number from the Round rows.
    round_stmt = (
        select(Round)
        .where(Round.tournament_id == tournament_id)
        .order_by(Round.round_number.desc())
    )
    rounds = (await session.execute(round_stmt)).scalars().all()
    current_round = rounds[0].round_number if rounds else 0
    is_live = tournament.status in _LIVE_STATUSES

    # For post-mortem we server-render the activity block once so the
    # page is self-contained; live pages instead hit the fragment route
    # every 2 s.
    snap: dict | None = None
    if not is_live:
        try:
            snap = await service.get_admin_activity(tournament_id)
        except LookupError:
            snap = None

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/admin/tournament_detail.html",
        context={
            "user": user,
            "tournament": tournament,
            "current_round": current_round,
            "is_live": is_live,
            "snap": snap,
        },
    )


@router.get("/users")
async def admin_users_list(request: Request, session: DBSession):
    """List every registered user with minimal activity counters.

    Counts surfaced per row:
    * ``tournaments_created`` — rows in ``tournaments`` where
      ``created_by`` matches the user.
    * ``tournaments_joined`` — rows in ``tournament_participants``
      where ``user_id`` matches. Builtin-strategy participants have
      ``user_id IS NULL`` and are excluded naturally.
    * ``agents_owned`` — non-soft-deleted ``agents`` rows owned by
      the user.

    All three are computed via grouped subqueries joined to ``users``
    so a user with no activity still shows up with zeros (LEFT JOIN
    + COALESCE), and we stay at three aggregate scans regardless of
    user count.
    """
    user = await _require_admin_ui_user(request, session)

    tournaments_created_sq = (
        select(
            Tournament.created_by.label("uid"),
            func.count(Tournament.id).label("n"),
        )
        .where(Tournament.created_by.is_not(None))
        .group_by(Tournament.created_by)
        .subquery()
    )
    tournaments_joined_sq = (
        select(
            Participant.user_id.label("uid"),
            func.count(Participant.id).label("n"),
        )
        .where(Participant.user_id.is_not(None))
        .group_by(Participant.user_id)
        .subquery()
    )
    agents_owned_sq = (
        select(
            Agent.owner_id.label("uid"),
            func.count(Agent.id).label("n"),
        )
        .where(Agent.deleted_at.is_(None))
        .group_by(Agent.owner_id)
        .subquery()
    )

    stmt = (
        select(
            User,
            func.coalesce(tournaments_created_sq.c.n, 0).label("t_created"),
            func.coalesce(tournaments_joined_sq.c.n, 0).label("t_joined"),
            func.coalesce(agents_owned_sq.c.n, 0).label("a_owned"),
        )
        .outerjoin(
            tournaments_created_sq,
            User.id == tournaments_created_sq.c.uid,
        )
        .outerjoin(
            tournaments_joined_sq,
            User.id == tournaments_joined_sq.c.uid,
        )
        .outerjoin(
            agents_owned_sq,
            User.id == agents_owned_sq.c.uid,
        )
        .order_by(User.id)
    )
    rows = (await session.execute(stmt)).all()

    users = [
        {
            "id": row.User.id,
            "username": row.User.username,
            "email": row.User.email,
            "is_admin": row.User.is_admin,
            "is_active": row.User.is_active,
            "created_at": row.User.created_at,
            "tournaments_created": row.t_created,
            "tournaments_joined": row.t_joined,
            "agents_owned": row.a_owned,
        }
        for row in rows
    ]
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/admin/users_list.html",
        context={"user": user, "users": users, "active_page": "admin"},
    )


@router.get("/tournaments/{tournament_id}/activity")
async def admin_tournament_activity(
    tournament_id: int,
    request: Request,
    session: DBSession,
    service: TournamentService = Depends(get_tournament_service),
):
    """HTMX activity fragment (polled every 2 s by the detail page)."""
    user = await _require_admin_ui_user(request, session)
    try:
        snap = await service.get_admin_activity(tournament_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/admin/_activity_block.html",
        context={
            "user": user,
            "snap": snap,
            "is_live": snap["status"] in _LIVE_STATUSES,
        },
    )
