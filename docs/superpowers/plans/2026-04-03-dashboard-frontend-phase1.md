# Dashboard Frontend Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an HTMX + Jinja2 + Pico CSS web UI under `/ui/` with sidebar navigation, home page, and benchmarks pages.

**Architecture:** New templates in `templates/ui/` with a separate `base_ui.html` (HTMX + Pico CSS). New route module `ui.py` under `/ui/` prefix. Existing React-based templates at `/` are untouched. Auth via httpOnly JWT cookie with open-redirect-safe `?next=` parameter.

**Tech Stack:** FastAPI, Jinja2, HTMX 2.0 (CDN), Pico CSS (CDN), SQLAlchemy async

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html` | Layout: sidebar + content, CDN links for HTMX + Pico |
| Create | `packages/atp-dashboard/atp/dashboard/v2/templates/ui/login.html` | Device Flow login page |
| Create | `packages/atp-dashboard/atp/dashboard/v2/templates/ui/home.html` | Home with stat cards + activity |
| Create | `packages/atp-dashboard/atp/dashboard/v2/templates/ui/benchmarks.html` | Benchmark list table |
| Create | `packages/atp-dashboard/atp/dashboard/v2/templates/ui/benchmark_detail.html` | Benchmark detail |
| Create | `packages/atp-dashboard/atp/dashboard/v2/templates/ui/placeholder.html` | "Coming soon" for future pages |
| Create | `packages/atp-dashboard/atp/dashboard/v2/templates/ui/error.html` | Error page (404, 500) |
| Create | `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/benchmark_table.html` | tbody partial for HTMX |
| Create | `packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css` | Sidebar + layout CSS |
| Create | `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` | All UI routes |
| Modify | `packages/atp-dashboard/atp/dashboard/v2/factory.py` | Mount UI router |
| Create | `tests/unit/dashboard/test_ui_routes.py` | Tests for UI endpoints |

---

### Task 1: Base layout template + CSS + route skeleton

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/placeholder.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/error.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/factory.py`
- Create: `tests/unit/dashboard/test_ui_routes.py`

- [ ] **Step 1: Write tests for UI route skeleton**

Create `tests/unit/dashboard/test_ui_routes.py`:

```python
"""Tests for dashboard UI routes."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def app():
    """Create test app with UI routes."""
    return create_test_app()


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as c:
        yield c


class TestUIRoutes:
    """Tests for UI page routes."""

    @pytest.mark.anyio
    async def test_home_returns_html(self, client) -> None:
        """GET /ui/ returns HTML page."""
        resp = await client.get("/ui/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "ATP Platform" in resp.text

    @pytest.mark.anyio
    async def test_benchmarks_returns_html(self, client) -> None:
        """GET /ui/benchmarks returns HTML page."""
        resp = await client.get("/ui/benchmarks")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Benchmarks" in resp.text

    @pytest.mark.anyio
    async def test_placeholder_games(self, client) -> None:
        """GET /ui/games returns placeholder page."""
        resp = await client.get("/ui/games")
        assert resp.status_code == 200
        assert "Coming soon" in resp.text

    @pytest.mark.anyio
    async def test_placeholder_runs(self, client) -> None:
        """GET /ui/runs returns placeholder page."""
        resp = await client.get("/ui/runs")
        assert resp.status_code == 200
        assert "Coming soon" in resp.text

    @pytest.mark.anyio
    async def test_placeholder_leaderboard(self, client) -> None:
        """GET /ui/leaderboard returns placeholder page."""
        resp = await client.get("/ui/leaderboard")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_placeholder_suites(self, client) -> None:
        """GET /ui/suites returns placeholder page."""
        resp = await client.get("/ui/suites")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_placeholder_analytics(self, client) -> None:
        """GET /ui/analytics returns placeholder page."""
        resp = await client.get("/ui/analytics")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_login_page(self, client) -> None:
        """GET /ui/login returns login page."""
        resp = await client.get("/ui/login")
        assert resp.status_code == 200
        assert "Login" in resp.text or "login" in resp.text

    @pytest.mark.anyio
    async def test_sidebar_has_all_nav_links(self, client) -> None:
        """Home page sidebar contains all navigation links."""
        resp = await client.get("/ui/")
        html = resp.text
        assert "/ui/benchmarks" in html
        assert "/ui/games" in html
        assert "/ui/runs" in html
        assert "/ui/leaderboard" in html
        assert "/ui/suites" in html
        assert "/ui/analytics" in html

    @pytest.mark.anyio
    async def test_htmx_loaded(self, client) -> None:
        """Pages include HTMX CDN script."""
        resp = await client.get("/ui/")
        assert "htmx.org" in resp.text

    @pytest.mark.anyio
    async def test_pico_css_loaded(self, client) -> None:
        """Pages include Pico CSS CDN."""
        resp = await client.get("/ui/")
        assert "pico" in resp.text.lower()
```

- [ ] **Step 2: Run tests to see them fail**

Run: `uv run python -m pytest tests/unit/dashboard/test_ui_routes.py -v`
Expected: FAIL (routes not registered)

- [ ] **Step 3: Create base_ui.html template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`:

```html
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}ATP Platform{% endblock %}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <link rel="stylesheet" href="/static/v2/css/ui.css">
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
</head>
<body hx-headers='{"X-CSRF": "1"}'>
    <div class="app-layout">
        <nav class="sidebar">
            <div class="sidebar-header">
                <a href="/ui/" class="sidebar-logo">ATP Platform</a>
            </div>
            <ul class="sidebar-nav">
                <li><a href="/ui/" class="{% if active_page == 'home' %}active{% endif %}">Home</a></li>
                <li><a href="/ui/benchmarks" class="{% if active_page == 'benchmarks' %}active{% endif %}">Benchmarks</a></li>
                <li><a href="/ui/games" class="{% if active_page == 'games' %}active{% endif %}">Games</a></li>
                <li><a href="/ui/runs" class="{% if active_page == 'runs' %}active{% endif %}">Runs</a></li>
                <li><a href="/ui/leaderboard" class="{% if active_page == 'leaderboard' %}active{% endif %}">Leaderboard</a></li>
                <li><a href="/ui/suites" class="{% if active_page == 'suites' %}active{% endif %}">Suites</a></li>
                <li><a href="/ui/analytics" class="{% if active_page == 'analytics' %}active{% endif %}">Analytics</a></li>
            </ul>
            <div class="sidebar-footer">
                {% if user %}
                <span class="sidebar-user">{{ user.username }}</span>
                {% endif %}
            </div>
        </nav>
        <main class="content">
            {% block content %}{% endblock %}
        </main>
    </div>
</body>
</html>
```

- [ ] **Step 4: Create ui.css**

Create `packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css`:

```css
/* ATP Dashboard UI Layout */
.app-layout {
    display: flex;
    min-height: 100vh;
}

/* Sidebar */
.sidebar {
    width: 220px;
    background: #1a1a2e;
    color: #e0e0e0;
    display: flex;
    flex-direction: column;
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    padding: 1rem 0.75rem;
}

.sidebar-header {
    padding: 0 0.5rem;
    margin-bottom: 1.5rem;
}

.sidebar-logo {
    color: #7c3aed;
    font-weight: bold;
    font-size: 1.1rem;
    text-decoration: none;
}

.sidebar-nav {
    list-style: none;
    padding: 0;
    margin: 0;
}

.sidebar-nav li {
    margin-bottom: 0.25rem;
}

.sidebar-nav a {
    display: block;
    padding: 0.5rem 0.75rem;
    border-radius: 0.375rem;
    color: #c0c0c0;
    text-decoration: none;
    font-size: 0.9rem;
    transition: background 0.15s;
}

.sidebar-nav a:hover {
    background: #2d2d44;
    color: #fff;
}

.sidebar-nav a.active {
    background: #2d2d44;
    color: #fff;
}

.sidebar-footer {
    margin-top: auto;
    padding-top: 1rem;
    border-top: 1px solid #333;
}

.sidebar-user {
    font-size: 0.8rem;
    color: #888;
    padding: 0.5rem 0.75rem;
    display: block;
}

/* Content area */
.content {
    flex: 1;
    margin-left: 220px;
    padding: 2rem;
    background: #fafafa;
    min-height: 100vh;
}

/* Stat cards */
.stat-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 0.5rem;
    padding: 1.25rem;
}

.stat-card .value {
    font-size: 1.75rem;
    font-weight: bold;
    color: #7c3aed;
}

.stat-card .label {
    font-size: 0.85rem;
    color: #666;
    margin-top: 0.25rem;
}

/* Activity feed */
.activity-item {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.activity-item .time {
    font-size: 0.8rem;
    color: #888;
}

/* Override Pico's body margin for our layout */
body {
    margin: 0;
    padding: 0;
}
```

- [ ] **Step 5: Create placeholder.html**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/placeholder.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}{{ page_title }} - ATP Platform{% endblock %}

{% block content %}
<h2>{{ page_title }}</h2>
<p>Coming soon. This section is under development.</p>
{% endblock %}
```

- [ ] **Step 6: Create error.html**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/error.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}{{ error_title|default("Error") }} - ATP Platform{% endblock %}

{% block content %}
<h2>{{ error_title|default("Error") }}</h2>
<p>{{ error_message|default("Something went wrong.") }}</p>
<p><a href="/ui/">Back to Home</a></p>
{% endblock %}
```

- [ ] **Step 7: Create ui.py route module**

Create `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`:

```python
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
    # Count benchmarks
    result = await session.execute(
        select(func.count(Benchmark.id))
    )
    total_benchmarks = result.scalar() or 0

    # Count runs
    result = await session.execute(
        select(func.count(Run.id))
    )
    total_runs = result.scalar() or 0

    # Count active runs
    result = await session.execute(
        select(func.count(Run.id)).where(
            Run.status == RunStatus.IN_PROGRESS
        )
    )
    active_runs = result.scalar() or 0

    # Recent runs (last 10)
    result = await session.execute(
        select(Run)
        .order_by(Run.started_at.desc())
        .limit(10)
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

    # Get total count
    result = await session.execute(
        select(func.count(Benchmark.id))
    )
    total = result.scalar() or 0

    # Get page of benchmarks
    result = await session.execute(
        select(Benchmark)
        .order_by(Benchmark.id.desc())
        .limit(per_page)
        .offset(offset)
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

    # Parse suite tests
    tests = []
    if benchmark.suite:
        try:
            suite = TestSuite.model_validate(benchmark.suite)
            tests = suite.tests
        except Exception:
            pass

    # Recent runs for this benchmark
    result = await session.execute(
        select(Run)
        .where(Run.benchmark_id == benchmark_id)
        .order_by(Run.started_at.desc())
        .limit(10)
    )
    runs = result.scalars().all()

    # Leaderboard (top 5)
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


# Placeholder routes
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
```

- [ ] **Step 8: Mount UI router in factory.py**

In `packages/atp-dashboard/atp/dashboard/v2/factory.py`, add after `app.include_router(api_router, prefix="/api")`:

```python
    # Mount UI routes (HTMX + Pico CSS frontend)
    from atp.dashboard.v2.routes.ui import router as ui_router
    app.include_router(ui_router)
```

- [ ] **Step 9: Run tests**

Run: `uv run python -m pytest tests/unit/dashboard/test_ui_routes.py -v`
Expected: All PASS

- [ ] **Step 10: Run ruff + pyrefly**

Run: `uv run ruff format packages/atp-dashboard/ tests/unit/dashboard/test_ui_routes.py && uv run ruff check packages/atp-dashboard/ tests/unit/dashboard/test_ui_routes.py --fix && uv run pyrefly check`

- [ ] **Step 11: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/ packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css packages/atp-dashboard/atp/dashboard/v2/routes/ui.py packages/atp-dashboard/atp/dashboard/v2/factory.py tests/unit/dashboard/test_ui_routes.py
git commit -m "feat(dashboard): add UI shell with sidebar, HTMX + Pico CSS, placeholder pages"
```

---

### Task 2: Home page template

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/home.html`

- [ ] **Step 1: Create home.html**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/home.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}Home - ATP Platform{% endblock %}

{% block content %}
<h2>Welcome to ATP Platform</h2>
<p>Agent Test Platform — benchmark and evaluate AI agents</p>

<div class="stat-cards">
    <div class="stat-card">
        <div class="value">{{ total_benchmarks }}</div>
        <div class="label">Benchmarks</div>
    </div>
    <div class="stat-card">
        <div class="value">{{ total_runs }}</div>
        <div class="label">Total Runs</div>
    </div>
    <div class="stat-card">
        <div class="value">{{ active_runs }}</div>
        <div class="label">Active Runs</div>
    </div>
</div>

<h3>Recent Activity</h3>
{% if recent_runs %}
    {% for run in recent_runs %}
    <div class="activity-item">
        <span>
            Run #{{ run.id }}
            {% if run.status.value == "completed" %}completed{% elif run.status.value == "in_progress" %}started{% else %}{{ run.status.value }}{% endif %}
            — {{ run.agent_name or "unnamed" }}
        </span>
        <span class="time">
            {% if run.started_at %}{{ run.started_at.strftime("%Y-%m-%d %H:%M") }}{% endif %}
        </span>
    </div>
    {% endfor %}
{% else %}
    <p>No activity yet. Start a benchmark run to see results here.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 2: Run tests**

Run: `uv run python -m pytest tests/unit/dashboard/test_ui_routes.py::TestUIRoutes::test_home_returns_html -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/home.html
git commit -m "feat(dashboard): add home page template with stats and activity feed"
```

---

### Task 3: Benchmarks list page + partial

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/benchmarks.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/benchmark_table.html`

- [ ] **Step 1: Create benchmarks.html**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/benchmarks.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}Benchmarks - ATP Platform{% endblock %}

{% block content %}
<h2>Benchmarks</h2>
<p>{{ total }} benchmark{{ "s" if total != 1 else "" }} available</p>

<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Version</th>
            <th>Tasks</th>
            <th>Created</th>
        </tr>
    </thead>
    <tbody id="benchmark-table-body">
        {% include "ui/partials/benchmark_table.html" %}
    </tbody>
</table>

{% if total_pages > 1 %}
<nav>
    <ul>
        {% for p in range(1, total_pages + 1) %}
        <li>
            {% if p == page %}
            <a href="#" aria-current="page"><strong>{{ p }}</strong></a>
            {% else %}
            <a href="/ui/benchmarks?page={{ p }}"
               hx-get="/ui/benchmarks?partial=1&page={{ p }}"
               hx-target="#benchmark-table-body"
               hx-swap="innerHTML">{{ p }}</a>
            {% endif %}
        </li>
        {% endfor %}
    </ul>
</nav>
{% endif %}
{% endblock %}
```

- [ ] **Step 2: Create benchmark_table.html partial**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/benchmark_table.html`:

```html
{% for bm in benchmarks %}
<tr>
    <td><a href="/ui/benchmarks/{{ bm.id }}">{{ bm.name }}</a></td>
    <td>{{ bm.version or "—" }}</td>
    <td>{{ bm.tasks_count }}</td>
    <td>{% if bm.created_at %}{{ bm.created_at.strftime("%Y-%m-%d") }}{% else %}—{% endif %}</td>
</tr>
{% else %}
<tr>
    <td colspan="4">No benchmarks found.</td>
</tr>
{% endfor %}
```

- [ ] **Step 3: Run tests**

Run: `uv run python -m pytest tests/unit/dashboard/test_ui_routes.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/benchmarks.html packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/benchmark_table.html
git commit -m "feat(dashboard): add benchmarks list page with HTMX pagination"
```

---

### Task 4: Benchmark detail page

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/benchmark_detail.html`
- Modify: `tests/unit/dashboard/test_ui_routes.py`

- [ ] **Step 1: Add detail page test**

Add to `tests/unit/dashboard/test_ui_routes.py`:

```python
class TestBenchmarkDetail:
    """Tests for benchmark detail page."""

    @pytest.mark.anyio
    async def test_benchmark_not_found(self, client) -> None:
        """GET /ui/benchmarks/999 returns 404 error page."""
        resp = await client.get("/ui/benchmarks/999")
        assert resp.status_code == 404
        assert "Not Found" in resp.text
```

- [ ] **Step 2: Create benchmark_detail.html**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/benchmark_detail.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}{{ benchmark.name }} - ATP Platform{% endblock %}

{% block content %}
<h2>{{ benchmark.name }}</h2>
{% if benchmark.description %}
<p>{{ benchmark.description }}</p>
{% endif %}

<div class="stat-cards">
    <div class="stat-card">
        <div class="value">{{ benchmark.tasks_count }}</div>
        <div class="label">Tasks</div>
    </div>
    <div class="stat-card">
        <div class="value">{{ benchmark.version or "—" }}</div>
        <div class="label">Version</div>
    </div>
    {% if benchmark.family_tag %}
    <div class="stat-card">
        <div class="value">{{ benchmark.family_tag }}</div>
        <div class="label">Family</div>
    </div>
    {% endif %}
</div>

<h3>Tests</h3>
{% if tests %}
<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Description</th>
            <th>Assertions</th>
        </tr>
    </thead>
    <tbody>
        {% for test in tests %}
        <tr>
            <td>{{ test.id }}</td>
            <td>{{ test.name }}</td>
            <td>{{ test.task.description[:80] }}{% if test.task.description|length > 80 %}...{% endif %}</td>
            <td>{{ test.assertions|length }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No tests defined.</p>
{% endif %}

<h3>Recent Runs</h3>
{% if runs %}
<table>
    <thead>
        <tr>
            <th>Run</th>
            <th>Agent</th>
            <th>Status</th>
            <th>Score</th>
            <th>Date</th>
        </tr>
    </thead>
    <tbody>
        {% for run in runs %}
        <tr>
            <td>#{{ run.id }}</td>
            <td>{{ run.agent_name or "—" }}</td>
            <td>{{ run.status.value }}</td>
            <td>{% if run.total_score is not none %}{{ "%.1f"|format(run.total_score) }}{% else %}—{% endif %}</td>
            <td>{% if run.started_at %}{{ run.started_at.strftime("%Y-%m-%d %H:%M") }}{% else %}—{% endif %}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No runs yet.</p>
{% endif %}

<h3>Leaderboard (Top 5)</h3>
{% if leaderboard %}
<table>
    <thead>
        <tr>
            <th>#</th>
            <th>Agent</th>
            <th>Best Score</th>
            <th>Runs</th>
        </tr>
    </thead>
    <tbody>
        {% for entry in leaderboard %}
        <tr>
            <td>{{ loop.index }}</td>
            <td>{{ entry.agent_name }}</td>
            <td>{{ "%.1f"|format(entry.best_score or 0) }}</td>
            <td>{{ entry.run_count }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No completed runs yet.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 3: Run tests**

Run: `uv run python -m pytest tests/unit/dashboard/test_ui_routes.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/benchmark_detail.html tests/unit/dashboard/test_ui_routes.py
git commit -m "feat(dashboard): add benchmark detail page with tests, runs, leaderboard"
```

---

### Task 5: Login page

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/login.html`

- [ ] **Step 1: Create login.html**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/login.html`:

```html
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - ATP Platform</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <style>
        .login-container {
            max-width: 400px;
            margin: 10vh auto;
            padding: 2rem;
        }
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .device-code {
            font-size: 2rem;
            font-weight: bold;
            letter-spacing: 0.2em;
            text-align: center;
            padding: 1rem;
            background: #f0f0f0;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-header">
            <h1>ATP Platform</h1>
            <p>Agent Test Platform</p>
        </div>

        {% if expired %}
        <article aria-label="Session expired">
            Session expired. Please log in again.
        </article>
        {% endif %}

        <article>
            <h3>Login with GitHub</h3>
            <p>Click below to start the login process. You'll be redirected to GitHub to authorize.</p>
            <button id="start-login" onclick="startDeviceFlow()">
                Login with GitHub
            </button>
            <div id="device-flow" style="display:none">
                <p>Enter this code at GitHub:</p>
                <div class="device-code" id="user-code"></div>
                <p><a id="verify-link" href="#" target="_blank">Open GitHub →</a></p>
                <p id="poll-status" aria-busy="true">Waiting for authorization...</p>
            </div>
            <div id="error-msg" style="display:none; color: red;"></div>
        </article>

        <article>
            <h3>Login with credentials</h3>
            <form id="login-form">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required>
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
                <button type="submit">Sign in</button>
            </form>
        </article>
    </div>

    <script>
        async function startDeviceFlow() {
            const btn = document.getElementById('start-login');
            btn.disabled = true;
            btn.setAttribute('aria-busy', 'true');
            try {
                const resp = await fetch('/api/auth/device', {method: 'POST'});
                if (resp.status === 501) {
                    showError('GitHub OAuth not configured on this server.');
                    return;
                }
                const data = await resp.json();
                document.getElementById('user-code').textContent =
                    data.user_code.slice(0,4) + '-' + data.user_code.slice(4);
                document.getElementById('verify-link').href = data.verification_uri;
                document.getElementById('device-flow').style.display = 'block';
                pollDeviceFlow(data.device_code, data.interval);
            } catch(e) {
                showError('Failed to start login: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.removeAttribute('aria-busy');
            }
        }

        async function pollDeviceFlow(deviceCode, interval) {
            const poll = async () => {
                try {
                    const resp = await fetch('/api/auth/device/poll', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({device_code: deviceCode}),
                    });
                    if (resp.status === 428) {
                        setTimeout(poll, interval * 1000);
                        return;
                    }
                    if (resp.status === 410) {
                        showError('Code expired. Please try again.');
                        return;
                    }
                    if (resp.ok) {
                        const data = await resp.json();
                        document.cookie = 'atp_token=' + data.access_token + ';path=/;SameSite=Strict';
                        const next = new URLSearchParams(window.location.search).get('next');
                        window.location.href = (next && next.startsWith('/ui/')) ? next : '/ui/';
                    }
                } catch(e) {
                    showError('Poll error: ' + e.message);
                }
            };
            setTimeout(poll, interval * 1000);
        }

        document.getElementById('login-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            try {
                const resp = await fetch('/api/auth/token', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: new URLSearchParams({username, password}),
                });
                if (!resp.ok) {
                    const data = await resp.json().catch(() => ({}));
                    showError(data.detail || 'Invalid credentials');
                    return;
                }
                const data = await resp.json();
                document.cookie = 'atp_token=' + data.access_token + ';path=/;SameSite=Strict';
                const next = new URLSearchParams(window.location.search).get('next');
                window.location.href = (next && next.startsWith('/ui/')) ? next : '/ui/';
            } catch(e) {
                showError('Login failed: ' + e.message);
            }
        });

        function showError(msg) {
            const el = document.getElementById('error-msg');
            el.textContent = msg;
            el.style.display = 'block';
        }
    </script>
</body>
</html>
```

- [ ] **Step 2: Run tests**

Run: `uv run python -m pytest tests/unit/dashboard/test_ui_routes.py::TestUIRoutes::test_login_page -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/login.html
git commit -m "feat(dashboard): add login page with Device Flow and credentials form"
```

---

### Task 6: Integration verification

- [ ] **Step 1: Run full dashboard test suite**

Run: `uv run python -m pytest tests/unit/dashboard/ -v -x -q`
Expected: No regressions

- [ ] **Step 2: Verify all UI routes registered**

Run: `uv run python -c "from atp.dashboard.v2.factory import app; routes = [r.path for r in app.routes if hasattr(r, 'path')]; print([r for r in routes if '/ui' in r])"`
Expected: All `/ui/` routes listed

- [ ] **Step 3: Run ruff + pyrefly**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run pyrefly check`

- [ ] **Step 4: Commit any formatting changes**

```bash
git add -u && git diff --cached --stat
git commit -m "style: format and lint fixes for dashboard UI"
```
