"""Dashboard UI routes.

Server-rendered HTML pages using HTMX + Jinja2 + Pico CSS.
All UI routes are under /ui/ prefix.
"""

from __future__ import annotations

import logging
from datetime import datetime
from types import SimpleNamespace

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus, TaskResult
from atp.dashboard.models import Agent, SuiteDefinition, User
from atp.dashboard.tokens import APIToken, Invite
from atp.dashboard.tournament.models import Participant
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


@router.get("/login", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_login(request: Request) -> HTMLResponse:
    """Render login page."""
    from atp.dashboard.v2.config import get_config

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
    """Clear auth cookie and redirect to home."""
    from starlette.responses import RedirectResponse

    response = RedirectResponse(url="/ui/", status_code=303)
    response.delete_cookie("atp_token", path="/")
    return response  # type: ignore[return-value]


@router.get("/register", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_register(request: Request) -> HTMLResponse:
    """Render registration page."""
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/register.html",
    )


@router.get("/", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_home(request: Request, session: DBSession) -> HTMLResponse:
    """Render home page with summary stats."""
    user = await _get_ui_user(request, session)
    result = await session.execute(select(func.count(Benchmark.id)))
    total_benchmarks = result.scalar() or 0

    result = await session.execute(select(func.count(Run.id)))
    total_runs = result.scalar() or 0

    result = await session.execute(
        select(func.count(Run.id)).where(Run.status == RunStatus.IN_PROGRESS)
    )
    active_runs = result.scalar() or 0

    result = await session.execute(
        select(Run).order_by(Run.started_at.desc()).limit(10)
    )
    recent_runs = result.scalars().all()

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/home.html",
        context={
            "active_page": "home",
            "total_benchmarks": total_benchmarks,
            "total_runs": total_runs,
            "active_runs": active_runs,
            "recent_runs": recent_runs,
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
        games = GameRegistry.list_games(with_metadata=True)  # type: ignore[assignment]
    except Exception:
        logger.debug("game_envs not available; showing empty games list")

    from atp.dashboard.tournament.models import Tournament  # noqa: PLC0415

    result = await session.execute(
        select(Tournament).order_by(Tournament.id.desc()).limit(50)
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


@router.get("/tournaments/{tournament_id}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_tournament_detail(
    request: Request,
    tournament_id: int,
    session: DBSession,
) -> HTMLResponse:
    """Render tournament detail page or HTMX live partial."""
    from atp.dashboard.tournament.models import (
        Round,
        Tournament,
    )

    result = await session.execute(
        select(Tournament)
        .where(Tournament.id == tournament_id)
        .options(
            selectinload(Tournament.participants),
            selectinload(Tournament.rounds).selectinload(Round.actions),
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
) -> HTMLResponse:
    """Render global leaderboard page with optional benchmark filter."""
    user = await _get_ui_user(request, session)
    result = await session.execute(select(Benchmark).order_by(Benchmark.name))
    benchmarks = result.scalars().all()

    stmt = select(
        Run.agent_name,
        func.max(Run.total_score).label("best_score"),
        func.count(Run.id).label("run_count"),
        func.count(func.distinct(Run.benchmark_id)).label("benchmark_count"),
    ).where(Run.status == RunStatus.COMPLETED)

    if benchmark_id is not None:
        stmt = stmt.where(Run.benchmark_id == benchmark_id)

    stmt = stmt.group_by(Run.agent_name).order_by(func.max(Run.total_score).desc())
    result = await session.execute(stmt)
    entries = result.all()

    partial = request.query_params.get("partial")
    if partial:
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/partials/leaderboard_table.html",
            context={
                "entries": entries,
                "selected_benchmark_id": benchmark_id,
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
    """My Agents page."""
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
            created_at=a.created_at,
            token_count=tc,
        )
        for a, tc in result.all()
    ]

    return _templates(request).TemplateResponse(
        request=request,
        name="ui/agents.html",
        context={"active_page": "agents", "user": user, "agents": agents},
    )


@router.get("/agents/{agent_id}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_agent_detail(
    request: Request, session: DBSession, agent_id: int
) -> HTMLResponse:
    """Agent detail page."""
    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

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
        },
    )


@router.get("/tokens", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_tokens(request: Request, session: DBSession) -> HTMLResponse:
    """My Tokens page."""
    user = await _get_ui_user(request, session)
    if not user:
        return RedirectResponse(url="/ui/login", status_code=302)  # type: ignore[return-value]

    # Single query: tokens + agent name via LEFT JOIN
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
        },
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
        },
    )
