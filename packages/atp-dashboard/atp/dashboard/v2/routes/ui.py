"""Dashboard UI routes.

Server-rendered HTML pages using HTMX + Jinja2 + Pico CSS.
All UI routes are under /ui/ prefix.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus, TaskResult
from atp.dashboard.models import SuiteDefinition
from atp.dashboard.v2.dependencies import DBSession
from atp.dashboard.v2.rate_limit import limiter

logger = logging.getLogger("atp.dashboard")

router = APIRouter(prefix="/ui", tags=["ui"])


def _templates(request: Request):
    """Get Jinja2Templates from app state."""
    return request.app.state.templates


@router.get("/login", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_login(request: Request) -> HTMLResponse:
    """Render login page."""
    expired = request.query_params.get("expired")
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/login.html",
        context={"expired": expired},
    )


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
        },
    )


@router.get("/games", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_games(request: Request, session: DBSession) -> HTMLResponse:
    """Render games page with game registry and tournaments."""
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
    from atp.dashboard.models import User
    from atp.dashboard.tournament.models import Tournament

    per_page = 50
    offset = (page - 1) * per_page

    result = await session.execute(select(func.count(Tournament.id)))
    total = result.scalar() or 0

    result = await session.execute(
        select(Tournament)
        .options(
            selectinload(Tournament.participants),
            selectinload(Tournament.rounds),
        )
        .order_by(Tournament.id.desc())
        .limit(per_page)
        .offset(offset)
    )
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
        },
    )


@router.get("/runs", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_runs(
    request: Request,
    session: DBSession,
    page: int = 1,
) -> HTMLResponse:
    """Render runs list page."""
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
        },
    )
