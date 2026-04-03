"""Dashboard UI routes.

Server-rendered HTML pages using HTMX + Jinja2 + Pico CSS.
All UI routes are under /ui/ prefix.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select

from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus
from atp.dashboard.v2.dependencies import DBSession

logger = logging.getLogger("atp.dashboard")

router = APIRouter(prefix="/ui", tags=["ui"])


def _templates(request: Request):
    """Get Jinja2Templates from app state."""
    return request.app.state.templates


@router.get("/login", response_class=HTMLResponse)
async def ui_login(request: Request) -> HTMLResponse:
    """Render login page."""
    expired = request.query_params.get("expired")
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/login.html",
        context={"expired": expired},
    )


@router.get("/", response_class=HTMLResponse)
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
async def ui_games(request: Request) -> HTMLResponse:
    """Games placeholder."""
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/placeholder.html",
        context={"active_page": "games", "page_title": "Games"},
    )


@router.get("/runs", response_class=HTMLResponse)
async def ui_runs(request: Request) -> HTMLResponse:
    """Runs placeholder."""
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/placeholder.html",
        context={"active_page": "runs", "page_title": "Runs"},
    )


@router.get("/leaderboard", response_class=HTMLResponse)
async def ui_leaderboard(request: Request) -> HTMLResponse:
    """Leaderboard placeholder."""
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/placeholder.html",
        context={
            "active_page": "leaderboard",
            "page_title": "Leaderboard",
        },
    )


@router.get("/suites", response_class=HTMLResponse)
async def ui_suites(request: Request) -> HTMLResponse:
    """Suites placeholder."""
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/placeholder.html",
        context={"active_page": "suites", "page_title": "Suites"},
    )


@router.get("/analytics", response_class=HTMLResponse)
async def ui_analytics(request: Request) -> HTMLResponse:
    """Analytics placeholder."""
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/placeholder.html",
        context={
            "active_page": "analytics",
            "page_title": "Analytics",
        },
    )
