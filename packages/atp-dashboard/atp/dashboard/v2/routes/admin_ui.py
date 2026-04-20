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
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from atp.dashboard.models import User
from atp.dashboard.tournament.errors import ValidationError as TournamentValidationError
from atp.dashboard.tournament.models import Tournament
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
        context={"user": user},
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
        context={"user": user, "tournaments": tournaments},
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
