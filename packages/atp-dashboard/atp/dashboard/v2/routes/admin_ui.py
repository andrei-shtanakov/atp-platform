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

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select

from atp.dashboard.models import User
from atp.dashboard.tournament.models import Tournament
from atp.dashboard.v2.dependencies import DBSession

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
