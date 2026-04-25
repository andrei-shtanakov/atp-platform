"""Dashboard UI routes.

Server-rendered HTML pages using HTMX + Jinja2 + Pico CSS.
All UI routes are under /ui/ prefix.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus, TaskResult
from atp.dashboard.models import Agent, SuiteDefinition, User
from atp.dashboard.rbac.models import Role, UserRole
from atp.dashboard.tokens import APIToken, Invite
from atp.dashboard.tournament.models import Participant, TournamentStatus
from atp.dashboard.tournament.models import Tournament as TournamentModel
from atp.dashboard.v2.dependencies import DBSession
from atp.dashboard.v2.rate_limit import limiter

logger = logging.getLogger("atp.dashboard")

router = APIRouter(prefix="/ui", tags=["ui"])


def _templates(request: Request):
    """Get Jinja2Templates from app state."""
    return request.app.state.templates


async def _get_ui_user(request: Request, session: DBSession) -> User | None:
    """Resolve the current user from cookie/header for UI templates.

    Returns User object if authenticated, None otherwise.
    Uses request.state.user_id set by JWTUserStateMiddleware.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is None:
        return None
    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        return None
    return user


async def _needs_bootstrap(session: DBSession) -> bool:
    """True when the database has no users yet — the 'first admin' path."""
    result = await session.execute(select(func.count(User.id)))
    return (result.scalar_one() or 0) == 0


@functools.lru_cache(maxsize=1)
def _game_registry() -> Any:
    """Return the populated ``GameRegistry`` class, or ``None`` if unavailable.

    Imports every bundled game module once per process so the registry
    decorators fire. Cached via ``lru_cache`` so /ui/games and
    /ui/games/{name} don't pay the import cost on each request.
    """
    try:
        from game_envs.games import (  # noqa: PLC0415
            auction,
            battle_of_sexes,
            colonel_blotto,
            congestion,
            el_farol,
            prisoners_dilemma,
            public_goods,
            stag_hunt,
        )
        from game_envs.games.registry import GameRegistry  # noqa: PLC0415

        _ = (
            auction,
            battle_of_sexes,
            colonel_blotto,
            congestion,
            el_farol,
            prisoners_dilemma,
            public_goods,
            stag_hunt,
        )
        return GameRegistry
    except Exception:
        logger.exception("game_envs not importable; game registry disabled")
        return None


@router.get("/about", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_about(request: Request, session: DBSession) -> HTMLResponse:
    """Public landing page with a short platform description and repo link."""
    user = await _get_ui_user(request, session)
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/about.html",
        context={"active_page": "about", "user": user},
    )


@router.get("/login", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_login(request: Request, session: DBSession) -> HTMLResponse:
    """Render login page (redirects to /ui/setup while the DB is empty)."""
    from atp.dashboard.v2.config import get_config

    if await _needs_bootstrap(session):
        return RedirectResponse(url="/ui/setup", status_code=302)  # type: ignore[return-value]

    config = get_config()
    expired = request.query_params.get("expired")
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/login.html",
        context={
            "expired": expired,
            "registration_mode": config.registration_mode,
        },
    )


@router.post("/logout")
@limiter.limit("120/minute")
async def ui_logout(request: Request) -> HTMLResponse:
    """Clear auth cookie and redirect to login.

    The delete_cookie attributes (path, samesite) must match how the cookie
    was originally set by login.html; otherwise some browsers keep the
    cookie because attributes don't line up.
    """
    from starlette.responses import RedirectResponse

    response = RedirectResponse(url="/ui/login", status_code=303)
    response.delete_cookie("atp_token", path="/", samesite="strict")
    return response  # type: ignore[return-value]


@router.get("/register", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_register(request: Request, session: DBSession) -> HTMLResponse:
    """Render registration page (redirects to /ui/setup while the DB is empty)."""
    from atp.dashboard.v2.config import get_config

    if await _needs_bootstrap(session):
        return RedirectResponse(url="/ui/setup", status_code=302)  # type: ignore[return-value]

    config = get_config()
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/register.html",
        context={"registration_mode": config.registration_mode},
    )


@router.get("/setup", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_setup(
    request: Request, session: DBSession, error: str | None = None
) -> HTMLResponse:
    """Bootstrap page for creating the first admin user.

    Disabled (redirects to /ui/login) once any user exists.
    """
    if not await _needs_bootstrap(session):
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/setup.html",
        context={"error": error},
    )


@router.post("/setup", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def ui_setup_submit(
    request: Request,
    session: DBSession,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
) -> HTMLResponse:
    """Create the first admin user from the setup form."""
    from urllib.parse import quote

    from atp.dashboard.auth import create_user

    if not await _needs_bootstrap(session):
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    def _fail(msg: str) -> HTMLResponse:
        return RedirectResponse(  # type: ignore[return-value]
            url=f"/ui/setup?error={quote(msg)}", status_code=303
        )

    if password != password_confirm:
        return _fail("Passwords do not match")
    if len(password) < 8:
        return _fail("Password must be at least 8 characters")

    try:
        user = await create_user(
            session,
            username=username.strip(),
            email=email.strip(),
            password=password,
            is_admin=True,
        )
    except ValueError as exc:
        return _fail(str(exc))

    role_result = await session.execute(select(Role).where(Role.name == "admin"))
    role = role_result.scalar_one_or_none()
    if role is not None:
        session.add(UserRole(user_id=user.id, role_id=role.id))
    await session.flush()

    return RedirectResponse(url="/ui/login", status_code=303)  # type: ignore[return-value]


@router.get("/", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_home(request: Request, session: DBSession) -> HTMLResponse:
    """Render home page with summary stats and a unified activity feed."""
    user = await _get_ui_user(request, session)

    total_benchmarks = (
        await session.execute(select(func.count(Benchmark.id)))
    ).scalar() or 0
    total_runs = (await session.execute(select(func.count(Run.id)))).scalar() or 0
    active_runs = (
        await session.execute(
            select(func.count(Run.id)).where(Run.status == RunStatus.IN_PROGRESS)
        )
    ).scalar() or 0
    total_tournaments = (
        await session.execute(select(func.count(TournamentModel.id)))
    ).scalar() or 0

    recent_runs = (
        (await session.execute(select(Run).order_by(Run.started_at.desc()).limit(10)))
        .scalars()
        .all()
    )
    recent_tournaments = (
        (
            await session.execute(
                select(TournamentModel)
                .order_by(TournamentModel.created_at.desc())
                .limit(10)
            )
        )
        .scalars()
        .all()
    )

    fallback_ts = datetime.min
    recent_items: list[SimpleNamespace] = []
    for run in recent_runs:
        status = str(run.status or "").lower()
        verb = (
            "completed"
            if status == "completed"
            else "started"
            if status == "in_progress"
            else status or "unknown"
        )
        recent_items.append(
            SimpleNamespace(
                kind="run",
                label=(f"Run #{run.id} {verb} — {run.agent_name or 'unnamed'}"),
                href=f"/ui/runs/{run.id}",
                ts=run.started_at,
            )
        )
    for t in recent_tournaments:
        ts = t.cancelled_at or t.ends_at or t.starts_at or t.created_at
        status = str(t.status or "").lower()
        recent_items.append(
            SimpleNamespace(
                kind="tournament",
                label=(f"Tournament #{t.id} {status} — {t.game_type}"),
                href=f"/ui/tournaments/{t.id}",
                ts=ts,
            )
        )
    recent_items.sort(key=lambda item: item.ts or fallback_ts, reverse=True)
    recent_items = recent_items[:10]

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/home.html",
        context={
            "active_page": "home",
            "total_benchmarks": total_benchmarks,
            "total_runs": total_runs,
            "active_runs": active_runs,
            "total_tournaments": total_tournaments,
            "recent_items": recent_items,
            "user": user,
        },
    )


@router.get("/benchmarks", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_benchmarks(
    request: Request,
    session: DBSession,
    page: int = 1,
) -> HTMLResponse:
    """Render benchmarks list page."""
    user = await _get_ui_user(request, session)
    per_page = 50
    offset = (page - 1) * per_page

    result = await session.execute(select(func.count(Benchmark.id)))
    total = result.scalar() or 0

    result = await session.execute(
        select(Benchmark).order_by(Benchmark.id.desc()).limit(per_page).offset(offset)
    )
    benchmarks = result.scalars().all()

    total_pages = (total + per_page - 1) // per_page

    template_name = "ui/benchmarks.html"
    partial = request.query_params.get("partial")
    if partial:
        template_name = "ui/partials/benchmark_table.html"

    return _templates(request).TemplateResponse(
        request=request,
        name=template_name,
        context={
            "active_page": "benchmarks",
            "benchmarks": benchmarks,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "user": user,
        },
    )


@router.get("/benchmarks/{benchmark_id}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_benchmark_detail(
    request: Request,
    benchmark_id: int,
    session: DBSession,
) -> HTMLResponse:
    """Render benchmark detail page."""
    from atp.loader.models import TestSuite

    user = await _get_ui_user(request, session)
    benchmark = await session.get(Benchmark, benchmark_id)
    if benchmark is None:
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/error.html",
            context={
                "error_title": "Not Found",
                "error_message": f"Benchmark #{benchmark_id} not found.",
            },
            status_code=404,
        )

    tests = []
    if benchmark.suite:
        try:
            suite = TestSuite.model_validate(benchmark.suite)
            tests = suite.tests
        except Exception:
            pass

    result = await session.execute(
        select(Run)
        .where(Run.benchmark_id == benchmark_id)
        .order_by(Run.started_at.desc())
        .limit(10)
    )
    runs = result.scalars().all()

    result = await session.execute(
        select(
            Run.agent_name,
            func.max(Run.total_score).label("best_score"),
            func.count(Run.id).label("run_count"),
        )
        .where(
            Run.benchmark_id == benchmark_id,
            Run.status == RunStatus.COMPLETED,
        )
        .group_by(Run.agent_name)
        .order_by(func.max(Run.total_score).desc())
        .limit(5)
    )
    leaderboard = result.all()

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/benchmark_detail.html",
        context={
            "active_page": "benchmarks",
            "benchmark": benchmark,
            "tests": tests,
            "runs": runs,
            "leaderboard": leaderboard,
            "user": user,
        },
    )


@router.get("/games", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_games(request: Request, session: DBSession) -> HTMLResponse:
    """Render games page with game registry and tournaments."""
    user = await _get_ui_user(request, session)
    games: list[dict] = []
    registry = _game_registry()
    if registry is not None:
        games = registry.list_games(with_metadata=True)

    result = await session.execute(
        select(TournamentModel).order_by(TournamentModel.id.desc()).limit(50)
    )
    tournaments = result.scalars().all()

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/games.html",
        context={
            "active_page": "games",
            "games": games,
            "tournaments": tournaments,
            "user": user,
        },
    )


@router.get("/games/{game_name}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_game_detail(
    request: Request,
    game_name: str,
    session: DBSession,
) -> HTMLResponse:
    """Public per-game detail page: rules, payoffs, how to participate.

    Content comes from ``atp.dashboard.v2.game_copy.GAME_COPY`` (narrative
    prose authored separately from the game-environments engine package)
    and from ``GameRegistry.game_info()`` (technical metadata — action
    spaces, config schema, player count). The page is intentionally
    public: anonymous visitors see the same thing as authenticated users.
    """
    from atp.dashboard.v2.game_copy import get_copy  # noqa: PLC0415

    copy = get_copy(game_name)
    if copy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown game: {game_name}",
        )

    user = await _get_ui_user(request, session)

    # Pull live registry metadata (action spaces, num_players, etc.).
    # Cached module-level helper; first call imports game_envs, subsequent
    # calls are a dict lookup.
    registry_info: dict[str, Any] | None = None
    registry = _game_registry()
    if registry is not None:
        try:
            registry_info = registry.game_info(game_name)
        except KeyError:
            # Copy exists but engine doesn't know this game yet — still render.
            registry_info = None
        except Exception:
            logger.exception("game_envs metadata unavailable for %s", game_name)
            registry_info = None

    # Latest tournaments of this game_type for social proof.
    tournaments_stmt = (
        select(TournamentModel)
        .where(TournamentModel.game_type == game_name)
        .order_by(TournamentModel.id.desc())
        .limit(5)
    )
    recent_tournaments = list((await session.execute(tournaments_stmt)).scalars().all())

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/game_detail.html",
        context={
            "active_page": "games",
            "game_name": game_name,
            "copy": copy,
            "registry_info": registry_info,
            "recent_tournaments": recent_tournaments,
            "user": user,
        },
    )


@router.get("/matches", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_matches(
    request: Request,
    session: DBSession,
) -> HTMLResponse:
    """List El Farol matches that have the Phase-7 dashboard payload.

    Filters match the renderability criteria in :func:`ui_match_detail`:
    ``status == 'completed'`` plus both ``actions_json`` and
    ``day_aggregates_json`` populated. These Phase-7 JSON columns are
    only written by the CLI writer for El Farol-shaped games (see
    ``atp/cli/commands/game.py``), so the renderability filter
    implicitly selects El Farol without matching on ``game_name`` —
    the engine's pretty name
    (``"El Farol Bar (n=6, threshold=4, days=30)"``) would not match a
    literal ``"el_farol"`` anyway.
    """
    from atp.dashboard.models import GameResult
    from atp.dashboard.tournament.models import Participant
    from atp.dashboard.tournament.models import Tournament as _Tournament

    user = await _get_ui_user(request, session)
    user_id = user.id if user else None

    renderable_filters = [
        GameResult.status == "completed",
        GameResult.actions_json.is_not(None),
        GameResult.day_aggregates_json.is_not(None),
    ]

    def _apply_visibility(stmt):  # type: ignore[no-untyped-def]
        """LABS-TSA PR-5: outer-join Tournament and filter by visibility.

        Rows pass when any of:
          * ``tournament_id IS NULL`` (legacy / CLI standalone runs)
          * ``tournament.join_token IS NULL`` (public tournaments)
          * the current user is the tournament creator
          * the current user is a Participant of the tournament
          * the current user is an admin (no filter applied)

        The detail route at ``/ui/matches/{id}`` applies the same rule
        post-fetch via ``_match_visible_to_user`` below. Any change
        here MUST be mirrored in that helper or an IDOR bug sneaks
        back in.
        """
        stmt = stmt.outerjoin(_Tournament, GameResult.tournament_id == _Tournament.id)
        if user is not None and user.is_admin:
            return stmt
        visibility_clauses: list[Any] = [
            GameResult.tournament_id.is_(None),
            _Tournament.join_token.is_(None),
        ]
        if user_id is not None:
            visibility_clauses.append(_Tournament.created_by == user_id)
            visibility_clauses.append(
                _Tournament.id.in_(
                    select(Participant.tournament_id).where(
                        Participant.user_id == user_id
                    )
                )
            )
        return stmt.where(or_(*visibility_clauses))

    stmt = _apply_visibility(select(GameResult).where(*renderable_filters))
    stmt = stmt.order_by(GameResult.completed_at.desc().nulls_last()).limit(100)
    matches = list((await session.execute(stmt)).scalars().all())

    total_stmt = _apply_visibility(
        select(func.count(GameResult.id)).where(*renderable_filters)
    )
    total = (await session.execute(total_stmt)).scalar_one()

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/matches.html",
        context={
            "active_page": "matches",
            "matches": matches,
            "total": total,
            "user": user,
        },
    )


async def _match_visible_to_user(
    session: AsyncSession,
    match: Any,  # noqa: ANN401 — GameResult ORM row, avoids circular import
    user: User | None,
) -> bool:
    """Return True iff ``match`` should be visible to ``user``.

    Business rule mirrors the SQL predicate in the listing
    (``ui_matches._apply_visibility``). Any divergence between the
    two is a security bug — keep them lockstep:

      * ``tournament_id IS NULL`` (legacy / CLI standalone runs) → visible
      * ``tournament.join_token IS NULL`` (public tournament) → visible
      * the caller is an admin → visible
      * the caller is the tournament's creator → visible
      * the caller is a Participant of the tournament → visible
      * otherwise → hidden
    """
    if match.tournament_id is None:
        return True
    from atp.dashboard.tournament.models import Participant, Tournament

    tournament = await session.get(Tournament, match.tournament_id)
    if tournament is None or tournament.join_token is None:
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


@router.get("/matches/{match_id}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_match_detail(
    request: Request,
    match_id: str,
    session: DBSession,
) -> HTMLResponse:
    """Render the El Farol match dashboard for a single completed match.

    Wraps the standalone dashboard bundle
    (``/static/v2/js/el_farol/{data_helpers,dashboard}.js``) with a Jinja
    shell that extends ``base_ui.html`` so sidebar/auth/navigation stay
    consistent. The server reshape (from
    ``atp.dashboard.v2.routes.el_farol_dashboard``) runs inline during
    the request and serialises the payload into ``window.__ATP_MATCH__``
    so the JS boots from a single JSON blob rather than a second fetch.
    ADR-005 (LABS-98) documents the stack choice.
    """
    import json
    import os

    from atp.dashboard.models import GameResult
    from atp.dashboard.v2.routes.el_farol_dashboard import _reshape
    from atp.dashboard.v2.routes.el_farol_from_tournament import (
        _reshape_from_tournament,
    )

    user = await _get_ui_user(request, session)

    # entry-path hint — /ui/matches can be reached from runs listing or
    # tournaments; fall back to games list for anonymous traffic.
    referer = request.headers.get("referer", "") or ""
    if "/ui/tournaments" in referer:
        back_link_href, back_link_label = "/ui/tournaments", "All tournaments"
        active_page = "tournaments"
    elif "/ui/runs" in referer:
        back_link_href, back_link_label = "/ui/runs", "All runs"
        active_page = "runs"
    else:
        back_link_href, back_link_label = "/ui/games", "All games"
        active_page = "games"

    stmt = select(GameResult).where(GameResult.match_id == match_id)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None and match_id.isdigit():
        row = await session.get(GameResult, int(match_id))

    # Visibility gate: same business rule as the /ui/matches listing
    # filter (``_apply_visibility`` closure in ``ui_matches``) — kept
    # in sync via ``_match_visible_to_user`` below. Without this, a
    # private-tournament match_id would still be enumerable by
    # autoincrement PK (``/ui/matches/1``, ``/ui/matches/2``, …) or by
    # a leaked UUID.
    if row is not None and not await _match_visible_to_user(session, row, user):
        # Hide existence: render the same "not found" HTML path the
        # unknown-match branch uses (200 + friendly message). Avoids
        # using HTTP status as an existence oracle — same 200 for
        # "does not exist" and "you are not allowed to see it".
        row = None

    context: dict[str, Any] = {
        "active_page": active_page,
        "match_id": match_id,
        "user": user,
        "back_link_href": back_link_href,
        "back_link_label": back_link_label,
        "not_found": False,
        "predates_schema": False,
        "in_progress": False,
        "status": None,
    }

    if row is None:
        context["not_found"] = True
    elif row.status and row.status != "completed":
        context["in_progress"] = True
        context["status"] = row.status
    else:
        # Three render paths share the completed-match branch:
        # (a) tournament-backed matches → reshape from authoritative
        #     Round/Action ORM rows (LABS-106).
        # (b) Phase-7 CLI matches → existing reshape from
        #     actions_json/day_aggregates_json columns.
        # (c) legacy CLI matches from before PR #63 introduced those
        #     columns — genuinely unrecoverable, surface a friendly
        #     placeholder.
        payload = None
        if row.tournament_id is not None:
            payload = await _reshape_from_tournament(row.tournament_id, session)
        elif not row.actions_json or not row.day_aggregates_json:
            context["predates_schema"] = True
        else:
            payload = _reshape(row)

        if payload is not None:
            context.update(
                {
                    "payload_json": json.dumps(
                        payload.model_dump(), separators=(",", ":")
                    ),
                    "num_agents": len(payload.AGENTS),
                    "num_days": payload.NUM_DAYS,
                    "num_slots": payload.NUM_SLOTS,
                    "capacity": payload.CAPACITY,
                    "langfuse_base": os.environ.get("ATP_LANGFUSE_BASE_URL", ""),
                }
            )

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/match_detail.html",
        context=context,
    )


@router.get("/tournaments", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_tournaments(
    request: Request,
    session: DBSession,
    page: int = 1,
) -> HTMLResponse:
    """Render tournament list page."""
    from sqlalchemy import exists, or_

    from atp.dashboard.tournament.models import Participant, Tournament

    per_page = 50
    offset = (page - 1) * per_page

    # Resolve current user for visibility filtering
    user_id = getattr(request.state, "user_id", None)
    user: User | None = None
    if user_id:
        user = await session.get(User, user_id)

    def _visibility_filter(stmt):  # type: ignore[no-untyped-def]
        """Apply same visibility rules as TournamentService.list_tournaments."""
        if user and user.is_admin:
            return stmt
        if user:
            return stmt.where(
                or_(
                    Tournament.join_token.is_(None),
                    Tournament.created_by == user.id,
                    exists().where(
                        (Participant.tournament_id == Tournament.id)
                        & (Participant.user_id == user.id)
                    ),
                )
            )
        # Anonymous: public tournaments only
        return stmt.where(Tournament.join_token.is_(None))

    count_stmt = _visibility_filter(select(func.count(Tournament.id)))
    result = await session.execute(count_stmt)
    total = result.scalar() or 0

    list_stmt = _visibility_filter(
        select(Tournament)
        .options(
            selectinload(Tournament.participants),
            selectinload(Tournament.rounds),
        )
        .order_by(Tournament.id.desc())
        .limit(per_page)
        .offset(offset)
    )
    result = await session.execute(list_stmt)
    tournaments = result.scalars().all()

    # Batch-load creator usernames
    creator_ids = {t.created_by for t in tournaments if t.created_by}
    creators: dict[int, str] = {}
    if creator_ids:
        user_result = await session.execute(
            select(User).where(User.id.in_(creator_ids))
        )
        creators = {u.id: u.username for u in user_result.scalars()}

    total_pages = (total + per_page - 1) // per_page

    template_name = "ui/tournaments.html"
    partial = request.query_params.get("partial")
    if partial:
        template_name = "ui/partials/tournament_list_table.html"

    return _templates(request).TemplateResponse(
        request=request,
        name=template_name,
        context={
            "active_page": "tournaments",
            "tournaments": tournaments,
            "creators": creators,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "user": user,
        },
    )


# Whitelist of game_types surfaced by the /ui/tournaments/new form —
# derived from TournamentService.SUPPORTED_GAMES so the UI cannot drift
# from the service-level validator.
from atp.dashboard.tournament.service import (  # noqa: E402
    SUPPORTED_GAMES as _TOURNAMENT_SUPPORTED_GAMES,
)

_TOURNAMENT_NEW_GAMES: list[str] = sorted(_TOURNAMENT_SUPPORTED_GAMES)


def _resolve_form_user(request: Request, session_user: User | None) -> int | None:
    """Resolve the active user id for form POSTs.

    Mirrors ``get_current_user_for_tournament`` in ``tournament_api.py``:
    prefers the JWT-decoded ``request.state.user_id`` and falls back to
    id=1 when ``ATP_DISABLE_AUTH=true`` so integration tests that bypass
    auth can seed a user and still transit the POST path.
    """
    import os

    if session_user is not None:
        return session_user.id
    user_id: int | None = getattr(request.state, "user_id", None)
    if user_id is not None:
        return user_id
    if os.environ.get("ATP_DISABLE_AUTH") == "true":
        return 1
    return None


@router.get("/tournaments/new", response_class=HTMLResponse)
@limiter.limit("60/minute")
async def ui_tournaments_new(
    request: Request,
    session: DBSession,
    game_type: str = "el_farol",
) -> HTMLResponse:
    """Render the self-service tournament creation form.

    LABS-TSA PR-5. Non-admin users see a forced-private visibility
    widget (disabled radio backed by a hidden input so the POST body
    always carries ``private=on``); admins can toggle private/public.
    The builtin-strategy checklist is game-scoped via
    ``list_builtins_for_game``.
    """
    from atp.dashboard.tournament.builtins import list_builtins_for_game

    user = await _get_ui_user(request, session)
    builtins = list_builtins_for_game(game_type)
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/tournament_new.html",
        context={
            "active_page": "tournaments",
            "games": _TOURNAMENT_NEW_GAMES,
            "selected_game": game_type,
            "builtins": builtins,
            "user": user,
        },
    )


@router.post("/tournaments/new", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def ui_tournaments_new_submit(
    request: Request,
    session: DBSession,
) -> HTMLResponse:
    """Handle the self-service tournament creation POST.

    LABS-TSA PR-5. Renders the detail page inline on success (no
    redirect) so the one-time ``join_token`` reveal stays in the
    response body and is not exposed through a follow-up GET.
    """
    from atp.dashboard.mcp import tournament_event_bus
    from atp.dashboard.tournament.errors import (
        ConcurrentPrivateCapExceededError,
        RosterValidationError,
    )
    from atp.dashboard.tournament.errors import (
        ValidationError as TournamentValidationError,
    )
    from atp.dashboard.tournament.service import TournamentService

    user_row = await _get_ui_user(request, session)
    user_id = _resolve_form_user(request, user_row)
    if user_id is None:
        return RedirectResponse(url="/ui/login", status_code=303)  # type: ignore[return-value]
    creator = user_row or await session.get(User, user_id)
    if creator is None or not creator.is_active:
        return RedirectResponse(url="/ui/login", status_code=303)  # type: ignore[return-value]

    form = await request.form()
    game_type = str(form.get("game_type", "el_farol"))
    private = str(form.get("private", "")).lower() in ("on", "true", "1")
    try:
        num_players = int(str(form.get("num_players", "2")))
        total_rounds = int(str(form.get("total_rounds", "1")))
        round_deadline_s = int(str(form.get("round_deadline_s", "30")))
    except ValueError:
        return _render_tournament_new_form_error(
            request,
            creator,
            game_type,
            "num_players / total_rounds / round_deadline_s must be integers",
        )
    roster_raw = form.getlist("roster[]")
    roster = [str(r) for r in roster_raw if r]

    # Non-admins cannot create public tournaments — enforce server-side
    # regardless of what the form submitted. The template already hides
    # the "public" radio for non-admins, but the check here is the hard
    # gate.
    if not creator.is_admin:
        private = True

    bus = getattr(request.app.state, "tournament_event_bus", tournament_event_bus)
    svc = TournamentService(session=session, bus=bus)
    tournament_name = f"{game_type} #{creator.username}"
    try:
        tournament, join_token = await svc.create_tournament(
            creator=creator,
            name=tournament_name,
            game_type=game_type,
            num_players=num_players,
            total_rounds=total_rounds,
            round_deadline_s=round_deadline_s,
            private=private,
            roster=roster,
        )
    except (
        TournamentValidationError,
        RosterValidationError,
        ConcurrentPrivateCapExceededError,
    ) as exc:
        return _render_tournament_new_form_error(request, creator, game_type, str(exc))
    # Ensure the caller-owned transaction writes the Tournament row.
    await session.commit()

    # Re-read the tournament with participants and rounds eager-loaded so
    # the synchronous Jinja render doesn't trigger an async lazy-load
    # (which blows up with ``MissingGreenlet``).
    from atp.dashboard.tournament.models import Round
    from atp.dashboard.tournament.models import Tournament as _Tournament

    reloaded = (
        await session.execute(
            select(_Tournament)
            .where(_Tournament.id == tournament.id)
            .options(
                selectinload(_Tournament.participants),
                selectinload(_Tournament.rounds).selectinload(Round.actions),
            )
        )
    ).scalar_one()

    # Server-render the detail page directly so the join_token is never
    # exposed through a POST→GET redirect.
    creator_name = creator.username
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/tournament_detail.html",
        context={
            "active_page": "tournaments",
            "tournament": reloaded,
            "creator_name": creator_name,
            "cancelled_by_name": None,
            "sorted_rounds": sorted(
                list(reloaded.rounds), key=lambda r: r.round_number, reverse=True
            ),
            "sorted_participants": sorted(
                list(reloaded.participants), key=lambda p: p.id
            ),
            "participant_map": {p.id: p.agent_name for p in reloaded.participants},
            "completed_rounds": 0,
            "timeline": [],
            "user": creator,
            "is_admin": creator.is_admin,
            "visible_reasoning_action_ids": set(),
            "join_token_once": join_token,
        },
    )


def _render_tournament_new_form_error(
    request: Request,
    user: User,
    game_type: str,
    error: str,
) -> HTMLResponse:
    """Re-render /ui/tournaments/new with an error banner."""
    from atp.dashboard.tournament.builtins import list_builtins_for_game

    builtins = list_builtins_for_game(game_type)
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/tournament_new.html",
        context={
            "active_page": "tournaments",
            "games": _TOURNAMENT_NEW_GAMES,
            "selected_game": game_type,
            "builtins": builtins,
            "user": user,
            "error": error,
        },
        status_code=400,
    )


@router.get("/tournaments/{tournament_id}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_tournament_detail(
    request: Request,
    tournament_id: int,
    session: DBSession,
) -> HTMLResponse:
    """Render tournament detail page or HTMX live partial."""
    from atp.dashboard.tournament.access import can_view_reasoning
    from atp.dashboard.tournament.models import (
        Action,
        Round,
        Tournament,
    )

    result = await session.execute(
        select(Tournament)
        .where(Tournament.id == tournament_id)
        .options(
            selectinload(Tournament.participants),
            selectinload(Tournament.rounds)
            .selectinload(Round.actions)
            .selectinload(Action.participant),
        )
    )
    tournament = result.scalar_one_or_none()

    # Resolve current user
    user_id = getattr(request.state, "user_id", None)
    user: User | None = None
    is_admin = False
    if user_id:
        user = await session.get(User, user_id)
        if user and user.is_admin:
            is_admin = True

    # Visibility check (same rules as TournamentService.get_tournament)
    not_found_resp = _templates(request).TemplateResponse(
        request=request,
        name="ui/error.html",
        context={
            "error_title": "Not Found",
            "error_message": f"Tournament #{tournament_id} not found.",
        },
        status_code=404,
    )

    if tournament is None:
        return not_found_resp

    if not is_admin and tournament.join_token is not None:
        # Private tournament — check ownership or participation
        is_owner = user and tournament.created_by == user.id
        is_participant = False
        if user:
            is_participant = any(p.user_id == user.id for p in tournament.participants)
        if not is_owner and not is_participant:
            return not_found_resp

    # Creator username
    creator_name = "—"
    if tournament.created_by:
        creator = await session.get(User, tournament.created_by)
        if creator:
            creator_name = creator.username

    # Cancelled-by username
    cancelled_by_name = None
    if tournament.cancelled_by:
        cb_user = await session.get(User, tournament.cancelled_by)
        if cb_user:
            cancelled_by_name = cb_user.username

    # Sort rounds newest-first
    sorted_rounds = sorted(
        tournament.rounds,
        key=lambda r: r.round_number,
        reverse=True,
    )

    # Sort participants by id for consistent score display
    sorted_participants = sorted(
        tournament.participants,
        key=lambda p: p.id,
    )

    # Build participant id->name map for round history columns
    participant_map = {p.id: p.agent_name for p in sorted_participants}

    # Completed round count
    completed_rounds = sum(1 for r in tournament.rounds if r.status == "completed")

    # Build event timeline (admin only)
    timeline: list[tuple[str, datetime, str]] = []
    if is_admin:
        if tournament.created_at:
            timeline.append(
                (
                    "tournament_created",
                    tournament.created_at,
                    f"{tournament.game_type}, {tournament.num_players} players, "
                    f"{tournament.total_rounds} rounds",
                )
            )
        for p in sorted(
            tournament.participants, key=lambda p: p.joined_at or datetime.min
        ):
            if p.joined_at:
                timeline.append(("participant_joined", p.joined_at, p.agent_name))
        for r in sorted(tournament.rounds, key=lambda r: r.round_number):
            if r.started_at:
                timeline.append(
                    (
                        "round_started",
                        r.started_at,
                        f"Round {r.round_number} of {tournament.total_rounds}",
                    )
                )
        if tournament.ends_at and tournament.status == "completed":
            timeline.append(("tournament_completed", tournament.ends_at, ""))
        if tournament.cancelled_at and tournament.status == "cancelled":
            reason_text = ""
            if tournament.cancelled_reason:
                reason_text = tournament.cancelled_reason.value
            timeline.append(
                ("tournament_cancelled", tournament.cancelled_at, reason_text)
            )
        # Newest first
        timeline.sort(key=lambda e: e[1], reverse=True)

    # Precompute which Action.id values the current viewer may read the
    # reasoning of. Template-side gate is a simple membership test, keeping
    # ACL logic out of Jinja.
    visible_reasoning_action_ids = {
        a.id
        for r in sorted_rounds
        for a in r.actions
        if can_view_reasoning(
            user=user,
            tournament=tournament,
            action_user_id=(a.participant.user_id if a.participant else None),
        )
    }

    context = {
        "active_page": "tournaments",
        "tournament": tournament,
        "creator_name": creator_name,
        "is_admin": is_admin,
        "cancelled_by_name": cancelled_by_name,
        "sorted_rounds": sorted_rounds,
        "sorted_participants": sorted_participants,
        "participant_map": participant_map,
        "completed_rounds": completed_rounds,
        "timeline": timeline,
        "user": user,
        "visible_reasoning_action_ids": visible_reasoning_action_ids,
    }

    partial = request.query_params.get("partial")
    if partial == "live":
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/partials/tournament_live.html",
            context=context,
        )

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/tournament_detail.html",
        context=context,
    )


@router.get("/runs", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_runs(
    request: Request,
    session: DBSession,
    page: int = 1,
) -> HTMLResponse:
    """Render runs list page."""
    user = await _get_ui_user(request, session)
    per_page = 50
    offset = (page - 1) * per_page

    result = await session.execute(select(func.count(Run.id)))
    total = result.scalar() or 0

    stmt = (
        select(Run, Benchmark.name.label("benchmark_name"), Benchmark.tasks_count)
        .outerjoin(Benchmark, Run.benchmark_id == Benchmark.id)
        .order_by(Run.started_at.desc())
        .limit(per_page)
        .offset(offset)
    )
    result = await session.execute(stmt)
    rows = result.all()

    total_pages = (total + per_page - 1) // per_page

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/runs.html",
        context={
            "active_page": "runs",
            "runs": rows,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "user": user,
        },
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_run_detail(
    request: Request,
    run_id: int,
    session: DBSession,
) -> HTMLResponse:
    """Render run detail page or HTMX partial."""
    user = await _get_ui_user(request, session)
    stmt = (
        select(Run, Benchmark.name.label("benchmark_name"), Benchmark.tasks_count)
        .outerjoin(Benchmark, Run.benchmark_id == Benchmark.id)
        .where(Run.id == run_id)
    )
    result = await session.execute(stmt)
    row = result.one_or_none()

    if row is None:
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/error.html",
            context={
                "error_title": "Not Found",
                "error_message": f"Run #{run_id} not found.",
            },
            status_code=404,
        )

    run, benchmark_name, tasks_count = row

    task_result = await session.execute(
        select(TaskResult)
        .where(TaskResult.run_id == run_id)
        .order_by(TaskResult.task_index)
    )
    task_results = task_result.scalars().all()

    context = {
        "active_page": "runs",
        "run": run,
        "benchmark_name": benchmark_name or "Unknown",
        "tasks_count": tasks_count or 0,
        "task_results": task_results,
        "now": datetime.now(),
        "user": user,
    }

    # HTMX partial responses
    is_htmx = request.headers.get("HX-Request") == "true"
    if is_htmx:
        target = request.headers.get("HX-Target", "")
        if target == "run-header":
            return _templates(request).TemplateResponse(
                request=request,
                name="ui/partials/run_header.html",
                context=context,
            )
        if target == "run-tasks":
            return _templates(request).TemplateResponse(
                request=request,
                name="ui/partials/run_tasks.html",
                context=context,
            )

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/run_detail.html",
        context=context,
    )


@router.post("/runs/{run_id}/cancel", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_cancel_run(
    request: Request,
    run_id: int,
    session: DBSession,
) -> HTMLResponse:
    """Cancel an in-progress run."""
    user = await _get_ui_user(request, session)
    run = await session.get(Run, run_id)
    if run is None:
        return HTMLResponse("<p style='color:red'>Run not found</p>")
    run.status = RunStatus.CANCELLED
    run.finished_at = datetime.now()
    await session.commit()

    # If called from run detail page (hx-target is run-header),
    # return the updated header partial
    target = request.headers.get("HX-Target", "")
    if target == "run-header":
        stmt = select(Benchmark.name, Benchmark.tasks_count).where(
            Benchmark.id == run.benchmark_id
        )
        result = await session.execute(stmt)
        bm_row = result.one_or_none()
        benchmark_name = bm_row[0] if bm_row else "Unknown"
        tasks_count = bm_row[1] if bm_row else 0

        return _templates(request).TemplateResponse(
            request=request,
            name="ui/partials/run_header.html",
            context={
                "run": run,
                "benchmark_name": benchmark_name,
                "tasks_count": tasks_count,
                "now": datetime.now(),
                "user": user,
            },
        )

    return HTMLResponse(f"<p style='color:green'>Run #{run_id} cancelled</p>")


@router.get("/leaderboard", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_leaderboard(
    request: Request,
    session: DBSession,
    benchmark_id: int | None = None,
    game_type: str | None = None,
) -> HTMLResponse:
    """Render global leaderboard page (benchmark + tournament sections).

    Scores are never mixed across the two: benchmark runs and tournament
    payoffs live on different scales, and even across game types within
    tournaments they aren't directly comparable — so the tournament section
    requires a specific ``game_type`` filter before it aggregates.
    """
    user = await _get_ui_user(request, session)
    benchmarks = (
        (await session.execute(select(Benchmark).order_by(Benchmark.name)))
        .scalars()
        .all()
    )

    stmt = select(
        Run.agent_name,
        func.max(Run.total_score).label("best_score"),
        func.count(Run.id).label("run_count"),
        func.count(func.distinct(Run.benchmark_id)).label("benchmark_count"),
    ).where(Run.status == RunStatus.COMPLETED)

    if benchmark_id is not None:
        stmt = stmt.where(Run.benchmark_id == benchmark_id)

    stmt = stmt.group_by(Run.agent_name).order_by(func.max(Run.total_score).desc())
    entries = (await session.execute(stmt)).all()

    partial = request.query_params.get("partial")
    if partial == "benchmark":
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/partials/leaderboard_table.html",
            context={
                "entries": entries,
                "selected_benchmark_id": benchmark_id,
                "user": user,
            },
        )

    game_types = [
        gt
        for (gt,) in (
            await session.execute(
                select(TournamentModel.game_type)
                .where(TournamentModel.status == TournamentStatus.COMPLETED)
                .distinct()
                .order_by(TournamentModel.game_type)
            )
        ).all()
    ]

    tournament_entries: Sequence = []
    if game_type:
        tournament_stmt = (
            select(
                Participant.agent_name,
                func.sum(Participant.total_score).label("total_score"),
                func.count(func.distinct(Participant.tournament_id)).label(
                    "tournament_count"
                ),
            )
            .join(TournamentModel, TournamentModel.id == Participant.tournament_id)
            .where(
                TournamentModel.status == TournamentStatus.COMPLETED,
                TournamentModel.game_type == game_type,
                Participant.total_score.isnot(None),
            )
            .group_by(Participant.agent_name)
            .order_by(func.sum(Participant.total_score).desc())
        )
        tournament_entries = (await session.execute(tournament_stmt)).all()

    if partial == "tournament":
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/partials/tournament_leaderboard_table.html",
            context={
                "tournament_entries": tournament_entries,
                "selected_game_type": game_type,
                "user": user,
            },
        )

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/leaderboard.html",
        context={
            "active_page": "leaderboard",
            "benchmarks": benchmarks,
            "entries": entries,
            "selected_benchmark_id": benchmark_id,
            "game_types": game_types,
            "tournament_entries": tournament_entries,
            "selected_game_type": game_type,
            "user": user,
        },
    )


@router.get("/suites", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_suites(
    request: Request,
    session: DBSession,
    page: int = 1,
) -> HTMLResponse:
    """Render suites list page."""
    user = await _get_ui_user(request, session)
    per_page = 50
    offset = (page - 1) * per_page

    result = await session.execute(select(func.count(SuiteDefinition.id)))
    total = result.scalar() or 0

    result = await session.execute(
        select(SuiteDefinition)
        .order_by(SuiteDefinition.id.desc())
        .limit(per_page)
        .offset(offset)
    )
    suites = result.scalars().all()

    total_pages = (total + per_page - 1) // per_page

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/suites.html",
        context={
            "active_page": "suites",
            "suites": suites,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "user": user,
        },
    )


@router.post("/suites/upload", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_suites_upload(
    request: Request,
    file: UploadFile,
    session: DBSession,
) -> HTMLResponse:
    """Handle YAML suite file upload and return HTML fragment."""
    from atp.loader.models import TestSuite
    from sqlalchemy import select as sa_select

    from atp.dashboard.v2.routes.upload import _validate_yaml

    filename = file.filename or "unknown.yaml"
    raw = await file.read()

    parsed_data, report = _validate_yaml(raw, filename)

    if not report.valid:
        errors_html = "".join(f"<li>{e}</li>" for e in report.errors)
        return HTMLResponse(
            f'<p style="color:red"><strong>Upload failed:</strong></p>'
            f"<ul>{errors_html}</ul>"
        )

    assert parsed_data is not None

    suite_name = parsed_data.get("test_suite", filename)
    stmt = sa_select(SuiteDefinition).where(SuiteDefinition.name == suite_name)
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing is not None:
        return HTMLResponse(
            f'<p style="color:orange">Suite <strong>{suite_name}</strong> '
            f"already exists (id={existing.id}).</p>"
        )

    suite_model = TestSuite.model_validate(parsed_data)
    tests_list = [
        t.model_dump(mode="json", exclude_none=False) for t in suite_model.tests
    ]
    for test_dict in tests_list:
        if "constraints" in test_dict and isinstance(test_dict["constraints"], dict):
            test_dict["constraints"] = {
                k: v for k, v in test_dict["constraints"].items() if v is not None
            }

    suite_def = SuiteDefinition(
        name=suite_name,
        version=parsed_data.get("version", "1.0"),
        description=parsed_data.get("description"),
        defaults_json=parsed_data.get("defaults", {}),
        agents_json=parsed_data.get("agents", []),
        tests_json=tests_list,
    )
    session.add(suite_def)
    await session.commit()
    await session.refresh(suite_def)

    warnings_html = ""
    if report.warnings:
        w = "".join(f"<li>{w}</li>" for w in report.warnings)
        warnings_html = f"<ul>{w}</ul>"

    return HTMLResponse(
        f'<p style="color:green">Suite <strong>{suite_name}</strong> uploaded '
        f"successfully (id={suite_def.id}, {len(tests_list)} tests).</p>"
        f"{warnings_html}"
        "<script>window.location.reload();</script>"
    )


@router.post("/suites/{suite_id}/create-benchmark", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_create_benchmark(
    request: Request,
    suite_id: int,
    session: DBSession,
) -> HTMLResponse:
    """Create a benchmark from a suite definition and return HTML fragment."""
    from atp.loader.models import TestSuite

    sd = await session.get(SuiteDefinition, suite_id)
    if sd is None:
        return HTMLResponse(
            f'<p style="color:red">Suite #{suite_id} not found.</p>',
            status_code=404,
        )

    try:
        suite = TestSuite.model_validate(
            {
                "test_suite": sd.name,
                "version": sd.version or "1.0",
                "tests": sd.tests_json,
            }
        )
        suite_dict = suite.model_dump(mode="json")
        bm = Benchmark(
            name=sd.name,
            description=sd.description or "",
            suite=suite_dict,
            tasks_count=len(suite.tests),
            tags=[],
            version=sd.version,
        )
        session.add(bm)
        await session.commit()
        await session.refresh(bm)
    except Exception as exc:
        logger.exception("Failed to create benchmark from suite %d", suite_id)
        return HTMLResponse(
            f'<p style="color:red">Failed to create benchmark: {exc}</p>'
        )

    return HTMLResponse(
        f'<p style="color:green">Benchmark <strong>{bm.name}</strong> created '
        f'(id={bm.id}). <a href="/ui/benchmarks/{bm.id}">View</a></p>'
    )


@router.get("/analytics", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_analytics(request: Request, session: DBSession) -> HTMLResponse:
    """Render analytics page with platform stats and agent rankings."""
    user = await _get_ui_user(request, session)
    result = await session.execute(select(func.count(Benchmark.id)))
    total_benchmarks = result.scalar() or 0

    result = await session.execute(select(func.count(SuiteDefinition.id)))
    total_suites = result.scalar() or 0

    result = await session.execute(select(func.count(Run.id)))
    total_runs = result.scalar() or 0

    result = await session.execute(
        select(func.count(Run.id)).where(Run.status == RunStatus.COMPLETED)
    )
    completed_runs = result.scalar() or 0

    result = await session.execute(
        select(func.avg(Run.total_score)).where(Run.status == RunStatus.COMPLETED)
    )
    avg_score = result.scalar()

    result = await session.execute(select(func.count(func.distinct(Run.agent_name))))
    total_agents = result.scalar() or 0

    result = await session.execute(
        select(Run.status, func.count(Run.id).label("count"))
        .group_by(Run.status)
        .order_by(func.count(Run.id).desc())
    )
    runs_by_status = [
        {"status": str(row.status), "count": row.count} for row in result.all()
    ]

    result = await session.execute(
        select(
            Run.agent_name,
            func.avg(Run.total_score).label("avg_score"),
            func.max(Run.total_score).label("best_score"),
            func.count(Run.id).label("run_count"),
        )
        .where(Run.status == RunStatus.COMPLETED)
        .group_by(Run.agent_name)
        .order_by(func.avg(Run.total_score).desc())
        .limit(10)
    )
    top_agents = result.all()

    result = await session.execute(
        select(Run).order_by(Run.started_at.desc()).limit(20)
    )
    recent_runs = result.scalars().all()

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/analytics.html",
        context={
            "active_page": "analytics",
            "total_benchmarks": total_benchmarks,
            "total_suites": total_suites,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "avg_score": avg_score,
            "total_agents": total_agents,
            "runs_by_status": runs_by_status,
            "top_agents": top_agents,
            "recent_runs": recent_runs,
            "user": user,
        },
    )


@router.get("/agents", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_agents(request: Request, session: DBSession) -> HTMLResponse:
    """My Agents page.

    LABS-TSA PR-5: adds a per-purpose quota strip (benchmark + tournament
    agent counts vs. their configured caps) and surfaces ``purpose`` as a
    table column. ``counts`` and ``quota`` are keyed by the two
    ``Agent.purpose`` values.
    """
    from atp.dashboard.v2.config import get_config

    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    # Single query: agents + active token count via LEFT JOIN + GROUP BY
    token_count_sub = (
        select(
            APIToken.agent_id,
            func.count(APIToken.id).label("token_count"),
        )
        .where(APIToken.revoked_at.is_(None))
        .group_by(APIToken.agent_id)
        .subquery()
    )
    result = await session.execute(
        select(Agent, func.coalesce(token_count_sub.c.token_count, 0))
        .outerjoin(token_count_sub, Agent.id == token_count_sub.c.agent_id)
        .where(Agent.owner_id == user.id, Agent.deleted_at.is_(None))
        .order_by(Agent.created_at.desc())
    )
    agents = [
        SimpleNamespace(
            id=a.id,
            name=a.name,
            version=a.version,
            agent_type=a.agent_type,
            purpose=a.purpose,
            created_at=a.created_at,
            token_count=tc,
        )
        for a, tc in result.all()
    ]

    # Per-purpose counts for the quota strip. Exclude soft-deleted agents
    # so the UI matches the enforcement logic in create_agent_for_user.
    counts: dict[str, int] = {}
    for purpose in ("benchmark", "tournament"):
        counts[purpose] = (
            await session.scalar(
                select(func.count(Agent.id)).where(
                    Agent.owner_id == user.id,
                    Agent.purpose == purpose,
                    Agent.deleted_at.is_(None),
                )
            )
            or 0
        )
    cfg = get_config()
    quota = {
        "benchmark": cfg.max_benchmark_agents_per_user,
        "tournament": cfg.max_tournament_agents_per_user,
    }

    error = request.query_params.get("error")
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/agents.html",
        context={
            "active_page": "agents",
            "user": user,
            "agents": agents,
            "counts": counts,
            "quota": quota,
            "error": error,
        },
    )


@router.post("/agents", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def ui_create_agent(
    request: Request,
    session: DBSession,
    name: str = Form(...),
    agent_type: str = Form(...),
    version: str = Form("latest"),
    description: str = Form(""),
) -> HTMLResponse:
    """Handle the 'New Agent' form submission from /ui/agents."""
    from atp.dashboard.v2.routes.agent_management_api import create_agent_for_user

    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    clean_name = name.strip()
    clean_version = version.strip() or "latest"
    clean_type = agent_type.strip()
    clean_desc = description.strip() or None

    try:
        await create_agent_for_user(
            session=session,
            user=user,
            name=clean_name,
            version=clean_version,
            agent_type=clean_type,
            description=clean_desc,
        )
    except HTTPException as exc:
        from urllib.parse import quote

        detail = exc.detail if isinstance(exc.detail, str) else "Failed to create agent"
        return RedirectResponse(  # type: ignore[return-value]
            url=f"/ui/agents?error={quote(detail)}", status_code=303
        )

    return RedirectResponse(url="/ui/agents", status_code=303)  # type: ignore[return-value]


@router.post("/agents/{agent_id}/delete", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def ui_delete_agent(
    request: Request,
    session: DBSession,
    agent_id: int,
) -> HTMLResponse:
    """Soft-delete an agent via the cookie-auth UI.

    Mirrors ``DELETE /api/v1/agents/{id}`` but accepts a form POST so
    the agents table can expose a Delete button without JavaScript.
    """
    from urllib.parse import quote

    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    agent = await session.get(Agent, agent_id)
    if agent is None or agent.deleted_at is not None:
        return RedirectResponse(  # type: ignore[return-value]
            url="/ui/agents?error=Agent+not+found", status_code=303
        )
    if agent.owner_id != user.id and not user.is_admin:
        return RedirectResponse(  # type: ignore[return-value]
            url="/ui/agents?error=You+don%27t+own+this+agent", status_code=303
        )

    active = await session.execute(
        select(Participant.id)
        .join(TournamentModel, Participant.tournament_id == TournamentModel.id)
        .where(
            Participant.agent_id == agent_id,
            Participant.released_at.is_(None),
            TournamentModel.status.in_(
                [TournamentStatus.PENDING, TournamentStatus.ACTIVE]
            ),
        )
    )
    if active.scalar_one_or_none() is not None:
        return RedirectResponse(  # type: ignore[return-value]
            url=f"/ui/agents?error={quote('Agent is in an active tournament')}",
            status_code=303,
        )

    now = datetime.now()
    agent.deleted_at = now
    await session.execute(
        update(APIToken)
        .where(APIToken.agent_id == agent_id, APIToken.revoked_at.is_(None))
        .values(revoked_at=now)
    )
    await session.flush()

    return RedirectResponse(url="/ui/agents", status_code=303)  # type: ignore[return-value]


async def _render_agent_detail(
    request: Request,
    session: DBSession,
    user: User,
    agent_id: int,
    *,
    new_token: str | None = None,
    error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    """Build the /ui/agents/{id} response; reused by GET and token POST handlers."""
    agent = await session.get(Agent, agent_id)
    if (
        not agent
        or agent.deleted_at
        or (agent.owner_id != user.id and not user.is_admin)
    ):
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/error.html",
            context={
                "active_page": "agents",
                "user": user,
                "error_title": "Not Found",
                "error_message": "Agent not found",
            },
            status_code=404,
        )

    token_result = await session.execute(
        select(APIToken)
        .where(APIToken.agent_id == agent_id)
        .order_by(APIToken.created_at.desc())
    )
    tokens = token_result.scalars().all()

    history_result = await session.execute(
        select(Participant, TournamentModel.status)
        .join(TournamentModel, Participant.tournament_id == TournamentModel.id)
        .where(Participant.agent_id == agent_id)
        .order_by(Participant.joined_at.desc())
    )
    tournament_history = [
        SimpleNamespace(
            tournament_id=p.tournament_id,
            tournament_status=t_status,
            total_score=p.total_score,
            joined_at=p.joined_at,
        )
        for p, t_status in history_result.all()
    ]

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/agent_detail.html",
        context={
            "active_page": "agents",
            "user": user,
            "agent": agent,
            "tokens": tokens,
            "tournament_history": tournament_history,
            "now": datetime.now(),
            "new_token": new_token,
            "error": error,
        },
        status_code=status_code,
    )


@router.get("/agents/new", response_class=HTMLResponse)
@limiter.limit("60/minute")
async def ui_agent_new(
    request: Request,
    session: DBSession,
    purpose: str = "benchmark",
) -> HTMLResponse:
    """Render the agent-registration form.

    Reachable from the "Register benchmark agent" / "Register tournament
    agent" buttons on ``/ui/agents``. Must be declared BEFORE
    ``/agents/{agent_id}`` below — FastAPI route matching is
    declaration-order, and ``"new"`` would fail the
    ``agent_id: int`` coercion otherwise (this was the QA-blocking
    422 bug surfaced on prod).
    """
    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]
    if purpose not in ("benchmark", "tournament"):
        purpose = "benchmark"
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/agent_new.html",
        context={
            "active_page": "agents",
            "user": user,
            "purpose": purpose,
            "error": None,
        },
    )


@router.post("/agents/new", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def ui_agent_new_submit(
    request: Request,
    session: DBSession,
) -> HTMLResponse:
    """Create an agent from the UI form; redirect to /ui/agents on success."""
    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]
    form = await request.form()
    name = str(form.get("name", "")).strip()
    agent_type = str(form.get("agent_type", "mcp")).strip()
    purpose = str(form.get("purpose", "benchmark")).strip()
    description = str(form.get("description", "")).strip() or None

    if purpose not in ("benchmark", "tournament"):
        purpose = "benchmark"

    error: str | None = None
    if not name:
        error = "Name is required."

    if error is None:
        from atp.dashboard.v2.routes.agent_management_api import (
            create_agent_for_user,
        )

        try:
            await create_agent_for_user(
                session=session,
                user=user,
                name=name,
                version="latest",
                agent_type=agent_type,
                description=description,
                config=None,
                purpose=purpose,
            )
        except HTTPException as exc:
            # create_agent_for_user uses a SAVEPOINT around the INSERT, so
            # the outer transaction (and the ORM-loaded ``user`` instance
            # the template lazy-loads) stays live — no rollback needed here.
            error = str(exc.detail)
        except Exception as exc:  # pragma: no cover - defensive
            error = str(exc)

    if error is not None:
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/agent_new.html",
            context={
                "active_page": "agents",
                "user": user,
                "purpose": purpose,
                "form_name": name,
                "form_agent_type": agent_type,
                "form_description": description or "",
                "error": error,
            },
            status_code=400,
        )
    return RedirectResponse(url="/ui/agents", status_code=303)  # type: ignore[return-value]


@router.get("/agents/{agent_id}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_agent_detail(
    request: Request, session: DBSession, agent_id: int
) -> HTMLResponse:
    """Agent detail page."""
    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]
    return await _render_agent_detail(request, session, user, agent_id)


@router.post("/agents/{agent_id}/tokens", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def ui_create_agent_token(
    request: Request,
    session: DBSession,
    agent_id: int,
    name: str = Form(...),
    expires_in_days: str = Form(""),
) -> HTMLResponse:
    """Create an API token scoped to an owned agent."""
    from atp.dashboard.v2.routes.token_api import create_token_for_user

    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    days: int | None = None
    raw_days = expires_in_days.strip()
    if raw_days:
        try:
            days = int(raw_days)
        except ValueError:
            return await _render_agent_detail(
                request,
                session,
                user,
                agent_id,
                error="Expiry must be a whole number of days (or blank)",
                status_code=400,
            )

    try:
        _, raw = await create_token_for_user(
            session=session,
            user=user,
            name=name.strip(),
            agent_id=agent_id,
            expires_in_days=days,
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else "Failed to create token"
        return await _render_agent_detail(
            request,
            session,
            user,
            agent_id,
            error=detail,
            status_code=exc.status_code,
        )

    return await _render_agent_detail(
        request, session, user, agent_id, new_token=raw, status_code=201
    )


@router.post("/tokens/{token_id}/revoke", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def ui_revoke_token(
    request: Request,
    session: DBSession,
    token_id: int,
) -> HTMLResponse:
    """Revoke a token and redirect back to the owning agent (or /ui/tokens)."""
    from atp.dashboard.v2.routes.token_api import revoke_token_for_user

    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    token = await session.get(APIToken, token_id)
    target_agent_id = token.agent_id if token is not None else None

    try:
        await revoke_token_for_user(session=session, user=user, token_id=token_id)
    except HTTPException as exc:
        if target_agent_id is not None:
            detail = (
                exc.detail if isinstance(exc.detail, str) else "Failed to revoke token"
            )
            return await _render_agent_detail(
                request,
                session,
                user,
                target_agent_id,
                error=detail,
                status_code=exc.status_code,
            )
        from urllib.parse import quote

        detail = exc.detail if isinstance(exc.detail, str) else "Failed to revoke token"
        return RedirectResponse(  # type: ignore[return-value]
            url=f"/ui/tokens?error={quote(detail)}", status_code=303
        )

    target = (
        f"/ui/agents/{target_agent_id}" if target_agent_id is not None else "/ui/tokens"
    )
    return RedirectResponse(url=target, status_code=303)  # type: ignore[return-value]


async def _render_tokens_page(
    request: Request,
    session: DBSession,
    user: User,
    *,
    new_token: str | None = None,
    error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    """Shared renderer for the /ui/tokens page.

    Used by both the GET (list) and POST (create) handlers so the
    newly-minted raw token can be shown inline on submit without a
    second redirect.
    """
    token_result = await session.execute(
        select(APIToken, Agent.name.label("agent_name"))
        .outerjoin(Agent, APIToken.agent_id == Agent.id)
        .where(APIToken.user_id == user.id)
        .order_by(APIToken.created_at.desc())
    )
    tokens = [
        SimpleNamespace(
            id=t.id,
            name=t.name,
            token_prefix=t.token_prefix,
            agent_id=t.agent_id,
            expires_at=t.expires_at,
            last_used_at=t.last_used_at,
            revoked_at=t.revoked_at,
            created_at=t.created_at,
            agent_name=agent_name,
        )
        for t, agent_name in token_result.all()
    ]

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/tokens.html",
        context={
            "active_page": "tokens",
            "user": user,
            "tokens": tokens,
            "now": datetime.now(),
            "new_token": new_token,
            "error": error,
        },
        status_code=status_code,
    )


@router.get("/tokens", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_tokens(request: Request, session: DBSession) -> HTMLResponse:
    """My Tokens page."""
    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    return await _render_tokens_page(request, session, user)


@router.post("/tokens", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def ui_create_user_token(
    request: Request,
    session: DBSession,
    name: str = Form(...),
    expires_in_days: str = Form(""),
) -> HTMLResponse:
    """Create a user-level API token (``atp_u_...``) via the UI form.

    Companion of ``POST /ui/agents/{id}/tokens`` which creates
    agent-scoped tokens. Without this handler, user-level tokens could
    only be minted via ``POST /api/v1/tokens`` with curl — a friction
    point visible from the public About quickstart (PR #34).
    """
    from atp.dashboard.v2.routes.token_api import create_token_for_user

    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    days: int | None = None
    raw_days = expires_in_days.strip()
    if raw_days:
        try:
            days = int(raw_days)
        except ValueError:
            return await _render_tokens_page(
                request,
                session,
                user,
                error="Expiry must be a whole number of days (or blank)",
                status_code=400,
            )

    try:
        _, raw = await create_token_for_user(
            session=session,
            user=user,
            name=name.strip(),
            agent_id=None,
            expires_in_days=days,
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else "Failed to create token"
        return await _render_tokens_page(
            request,
            session,
            user,
            error=detail,
            status_code=exc.status_code,
        )

    return await _render_tokens_page(
        request, session, user, new_token=raw, status_code=201
    )


@router.get("/invites", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_invites(request: Request, session: DBSession) -> HTMLResponse:
    """Invite management page (admin only)."""
    user = await _get_ui_user(request, session)
    if not user or not user.is_admin:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    # Single query: invites + creator/used_by names via aliased JOINs
    from sqlalchemy.orm import aliased

    Creator = aliased(User)
    UsedBy = aliased(User)
    result = await session.execute(
        select(Invite, Creator.username, UsedBy.username)
        .outerjoin(Creator, Invite.created_by_id == Creator.id)
        .outerjoin(UsedBy, Invite.used_by_id == UsedBy.id)
        .order_by(Invite.created_at.desc())
    )
    invites = [
        SimpleNamespace(
            id=inv.id,
            code=inv.code,
            created_by_id=inv.created_by_id,
            used_by_id=inv.used_by_id,
            use_count=inv.use_count,
            max_uses=inv.max_uses,
            expires_at=inv.expires_at,
            created_at=inv.created_at,
            created_by_name=creator_name,
            used_by_name=used_by_name,
        )
        for inv, creator_name, used_by_name in result.all()
    ]

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/invites.html",
        context={
            "active_page": "invites",
            "user": user,
            "invites": invites,
            "now": datetime.now(),
            "new_code": request.query_params.get("new_code"),
            "error": request.query_params.get("error"),
        },
    )


@router.post("/invites", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def ui_create_invite(
    request: Request,
    session: DBSession,
    expires_in_days: str = Form(""),
) -> HTMLResponse:
    """Create an invite code from the admin UI."""
    from urllib.parse import quote

    from atp.dashboard.v2.routes.invite_api import create_invite_for_admin

    user = await _get_ui_user(request, session)
    if not user or not user.is_admin:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    days: int | None = 7
    raw = expires_in_days.strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            return RedirectResponse(  # type: ignore[return-value]
                url="/ui/invites?error=" + quote("Expiry must be a whole number"),
                status_code=303,
            )
        days = None if parsed == 0 else parsed

    invite = await create_invite_for_admin(
        session=session, admin=user, expires_in_days=days
    )
    return RedirectResponse(  # type: ignore[return-value]
        url=f"/ui/invites?new_code={quote(invite.code)}", status_code=303
    )


@router.post("/invites/{invite_id}/deactivate", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def ui_deactivate_invite(
    request: Request,
    session: DBSession,
    invite_id: int,
) -> HTMLResponse:
    """Deactivate an invite from the admin UI."""
    from urllib.parse import quote

    from atp.dashboard.v2.routes.invite_api import deactivate_invite_for_admin

    user = await _get_ui_user(request, session)
    if not user or not user.is_admin:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    try:
        await deactivate_invite_for_admin(session=session, invite_id=invite_id)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else "Failed to deactivate"
        return RedirectResponse(  # type: ignore[return-value]
            url=f"/ui/invites?error={quote(detail)}", status_code=303
        )
    return RedirectResponse(url="/ui/invites", status_code=303)  # type: ignore[return-value]
