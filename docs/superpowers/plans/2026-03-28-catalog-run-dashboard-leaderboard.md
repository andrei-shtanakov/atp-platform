# Catalog Run + Dashboard + Leaderboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `atp catalog run` to actually execute tests via TestOrchestrator and create submissions, add catalog browsing pages to the dashboard, and add a public leaderboard API.

**Architecture:** Three independent features built on the existing catalog infrastructure. (1) `catalog run` replaces the stub with real execution, reusing the `_run_suite` pattern from `test_cmd`. (2) Dashboard adds FastAPI routes + Jinja2 templates for catalog browsing. (3) Leaderboard adds API endpoints for cross-agent comparison. All three use `CatalogRepository` for data access.

**Tech Stack:** Python 3.12+, Click, SQLAlchemy async, FastAPI, Jinja2, Rich, existing TestOrchestrator/TestLoader/ResultStorage

---

## File Structure

### New files

```
# Feature 1: catalog run
(no new files — modifying atp/cli/commands/catalog.py)

# Feature 2: Dashboard catalog UI
packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py        — API routes for catalog
packages/atp-dashboard/atp/dashboard/v2/templates/catalog.html    — Catalog browse page
packages/atp-dashboard/atp/dashboard/v2/templates/catalog_suite.html — Suite detail page

# Feature 3: Leaderboard API
packages/atp-dashboard/atp/dashboard/v2/routes/catalog_leaderboard.py — Leaderboard endpoints

# Tests
tests/unit/catalog/test_catalog_run.py     — Integration test for run flow
tests/unit/dashboard/catalog/test_routes.py — Dashboard route tests
```

### Modified files

```
atp/cli/commands/catalog.py                              — Replace run stub with real execution
packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py — Register new routes
packages/atp-dashboard/atp/dashboard/v2/templates/base.html — Add catalog nav link
```

---

## Feature 1: `atp catalog run` Integration

### Task 1: Wire catalog run to TestOrchestrator

**Files:**
- Modify: `atp/cli/commands/catalog.py`
- Test: `tests/unit/catalog/test_catalog_run.py`

- [ ] **Step 1: Read the existing `_run_suite` pattern**

Read `atp/cli/main.py` — find the `_run_suite` async function (around lines 650-900). Understand:
- How it creates an adapter via `create_adapter(adapter_type, config_dict)`
- How it creates `TestOrchestrator` with `async with`
- How it calls `orchestrator.run_suite(suite, agent_name, runs_per_test)`
- How it calls `_save_results_to_db()`
- How it extracts scores from `ScoredTestResult`

Also read the current `catalog.py` run stub.

- [ ] **Step 2: Write test for catalog run**

```python
# tests/unit/catalog/test_catalog_run.py
"""Tests for atp catalog run execution flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.cli.commands.catalog import _execute_catalog_run


@pytest.mark.anyio
async def test_execute_catalog_run_calls_orchestrator() -> None:
    """Verify the run function loads suite, creates adapter, runs, and submits."""
    mock_suite_yaml = (
        "test_suite: test\nversion: '1.0'\n"
        "tests:\n  - id: t1\n    name: T1\n"
        "    task:\n      description: Do thing\n"
    )

    # Mock the DB session and repository
    mock_session = AsyncMock()
    mock_repo = AsyncMock()
    mock_suite = MagicMock()
    mock_suite.suite_yaml = mock_suite_yaml
    mock_suite.name = "Test Suite"
    mock_suite.tests = []
    mock_repo.get_suite_by_path.return_value = mock_suite

    # Mock TestLoader
    mock_loader = MagicMock()
    mock_test_suite = MagicMock()
    mock_test_suite.test_suite = "test"
    mock_loader.load_from_string.return_value = mock_test_suite

    # Mock orchestrator
    mock_result = MagicMock()
    mock_result.test_results = []
    mock_result.success = True
    mock_orchestrator = AsyncMock()
    mock_orchestrator.run_suite.return_value = mock_result
    mock_orchestrator.__aenter__ = AsyncMock(return_value=mock_orchestrator)
    mock_orchestrator.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("atp.cli.commands.catalog.CatalogRepository", return_value=mock_repo),
        patch("atp.cli.commands.catalog.TestLoader", return_value=mock_loader),
        patch("atp.cli.commands.catalog.create_adapter") as mock_create,
        patch("atp.cli.commands.catalog.TestOrchestrator", return_value=mock_orchestrator),
    ):
        mock_create.return_value = MagicMock()
        success = await _execute_catalog_run(
            session=mock_session,
            category_slug="coding",
            suite_slug="file-operations",
            adapter_type="http",
            adapter_config={"url": "http://localhost:8000"},
            agent_name="test-agent",
            runs_per_test=1,
        )

    assert success is True
    mock_repo.get_suite_by_path.assert_called_once_with("coding", "file-operations")
    mock_loader.load_from_string.assert_called_once()
    mock_orchestrator.run_suite.assert_called_once()
```

- [ ] **Step 3: Run test to verify failure**

Run: `uv run python -m pytest tests/unit/catalog/test_catalog_run.py -v`
Expected: FAIL — `_execute_catalog_run` not found

- [ ] **Step 4: Implement catalog run**

Replace the `_do_run` stub in `atp/cli/commands/catalog.py` with real execution. The key function:

```python
async def _execute_catalog_run(
    session: AsyncSession,
    category_slug: str,
    suite_slug: str,
    adapter_type: str,
    adapter_config: dict[str, str],
    agent_name: str,
    runs_per_test: int = 1,
) -> bool:
    """Execute a catalog suite and create submissions."""
    from atp.adapters import create_adapter
    from atp.catalog.comparison import format_comparison_table
    from atp.catalog.repository import CatalogRepository
    from atp.loader import TestLoader
    from atp.runner.orchestrator import TestOrchestrator

    repo = CatalogRepository(session)
    suite = await repo.get_suite_by_path(category_slug, suite_slug)
    if suite is None:
        click.echo(f"Suite not found: {category_slug}/{suite_slug}", err=True)
        return False

    # Load suite from YAML
    loader = TestLoader()
    test_suite = loader.load_from_string(suite.suite_yaml)

    # Create adapter
    adapter = create_adapter(adapter_type, adapter_config)

    # Run via orchestrator
    click.echo(f"Running: {category_slug}/{suite_slug} ({len(suite.tests)} tests)")
    async with TestOrchestrator(
        adapter=adapter,
        runs_per_test=runs_per_test,
    ) as orchestrator:
        result = await orchestrator.run_suite(
            suite=test_suite,
            agent_name=agent_name,
            runs_per_test=runs_per_test,
        )

    # Save to dashboard DB
    from atp.dashboard.storage import ResultStorage
    storage = ResultStorage(session)
    agent = await storage.get_or_create_agent(
        name=agent_name, agent_type=adapter_type, config=adapter_config,
    )
    suite_exec = await storage.persist_suite_result(
        result=result, agent_type=adapter_type,
    )

    # Create catalog submissions
    comparison_tests = []
    for test_result in result.test_results:
        test_slug = test_result.test.id
        score = getattr(test_result, "score", None) or 0.0

        catalog_test = None
        for ct in suite.tests:
            if ct.slug == test_slug:
                catalog_test = ct
                break

        if catalog_test:
            await repo.create_submission(
                test_id=catalog_test.id,
                agent_name=agent_name,
                agent_type=adapter_type,
                score=score,
                suite_execution_id=suite_exec.id if suite_exec else None,
            )
            await repo.update_test_stats(catalog_test.id)
            await session.refresh(catalog_test)

            comparison_tests.append({
                "name": catalog_test.slug,
                "score": score,
                "avg": catalog_test.avg_score,
                "best": catalog_test.best_score,
            })

    await session.commit()

    # Render comparison
    click.echo(f"\nResults: {category_slug}/{suite_slug}")
    click.echo(format_comparison_table(comparison_tests, []))
    return result.success
```

Update the `run` Click command to call this function:

```python
@catalog_command.command(name="run")
@click.argument("path")
@click.option("--adapter", required=True, help="Agent adapter type (http, cli, container)")
@click.option("--adapter-config", multiple=True, help="key=value adapter config pairs")
@click.option("--agent-name", default="my-agent", help="Name for this agent")
@click.option("--runs", default=1, type=int, help="Runs per test")
def run_cmd(path: str, adapter: str, adapter_config: tuple, agent_name: str, runs: int) -> None:
    """Run a catalog suite against your agent and submit results."""
    parts = path.split("/")
    if len(parts) != 2:
        click.echo("Error: PATH must be category/suite (e.g., coding/file-operations)", err=True)
        sys.exit(EXIT_ERROR)

    config_dict = {}
    for item in adapter_config:
        key, _, value = item.partition("=")
        config_dict[key] = value

    try:
        success = asyncio.run(_do_run_real(parts[0], parts[1], adapter, config_dict, agent_name, runs))
        sys.exit(EXIT_SUCCESS if success else EXIT_FAILURE)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


async def _do_run_real(
    category: str, suite: str, adapter_type: str,
    adapter_config: dict, agent_name: str, runs: int,
) -> bool:
    from atp.dashboard.database import init_database
    db = await init_database()
    async with db.session() as session:
        return await _execute_catalog_run(
            session, category, suite, adapter_type, adapter_config, agent_name, runs,
        )
```

- [ ] **Step 5: Run tests**

Run: `uv run python -m pytest tests/unit/catalog/test_catalog_run.py -v`
Expected: PASS

- [ ] **Step 6: Run all catalog tests for regression**

Run: `uv run python -m pytest tests/unit/catalog/ tests/unit/cli/test_catalog_cli.py -q`
Expected: All pass

- [ ] **Step 7: Ruff + commit**

```bash
uv run ruff format atp/cli/commands/catalog.py tests/unit/catalog/
uv run ruff check atp/cli/commands/catalog.py tests/unit/catalog/ --fix
git add atp/cli/commands/catalog.py tests/unit/catalog/test_catalog_run.py
git commit -m "feat(catalog): wire 'atp catalog run' to TestOrchestrator"
```

---

## Feature 2: Dashboard Catalog UI

### Task 2: Catalog API Routes

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`
- Test: `tests/unit/dashboard/catalog/test_routes.py`

- [ ] **Step 1: Read existing route patterns**

Read `packages/atp-dashboard/atp/dashboard/v2/routes/leaderboard.py` and `__init__.py`. Understand:
- How `APIRouter` is created with prefix and tags
- How `DBSession` is injected
- How responses are returned (Pydantic models or dicts)
- How routes are registered in `__init__.py`

- [ ] **Step 2: Create Pydantic response models and routes**

Create `packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py`:

```python
"""Catalog API routes for dashboard."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from atp.catalog.repository import CatalogRepository
from atp.catalog.sync import sync_builtin_catalog
from atp.dashboard.v2.dependencies import DBSession

router = APIRouter(prefix="/catalog", tags=["catalog"])


class CategoryResponse(BaseModel):
    slug: str
    name: str
    description: str | None
    suite_count: int
    test_count: int


class SuiteResponse(BaseModel):
    slug: str
    name: str
    description: str | None
    difficulty: str | None
    author: str
    source: str
    test_count: int


class TestResponse(BaseModel):
    slug: str
    name: str
    task_description: str
    difficulty: str | None
    total_submissions: int
    avg_score: float | None
    best_score: float | None


class SuiteDetailResponse(BaseModel):
    slug: str
    name: str
    description: str | None
    difficulty: str | None
    author: str
    source: str
    category: str
    tests: list[TestResponse]


class SubmissionResponse(BaseModel):
    rank: int
    agent_name: str
    score: float
    quality_score: float
    completeness_score: float
    efficiency_score: float
    cost_score: float


@router.get("/categories", response_model=list[CategoryResponse])
async def list_categories(session: DBSession) -> list[CategoryResponse]:
    """List all catalog categories."""
    repo = CatalogRepository(session)
    cats = await repo.list_categories()
    if not cats:
        await sync_builtin_catalog(session)
        cats = await repo.list_categories()
    return [
        CategoryResponse(
            slug=c.slug,
            name=c.name,
            description=c.description,
            suite_count=len(c.suites),
            test_count=sum(len(s.tests) for s in c.suites),
        )
        for c in cats
    ]


@router.get("/categories/{category_slug}/suites", response_model=list[SuiteResponse])
async def list_suites(category_slug: str, session: DBSession) -> list[SuiteResponse]:
    """List suites in a category."""
    repo = CatalogRepository(session)
    suites = await repo.list_suites(category_slug=category_slug)
    return [
        SuiteResponse(
            slug=s.slug,
            name=s.name,
            description=s.description,
            difficulty=s.difficulty,
            author=s.author,
            source=s.source,
            test_count=len(s.tests),
        )
        for s in suites
    ]


@router.get(
    "/categories/{category_slug}/suites/{suite_slug}",
    response_model=SuiteDetailResponse,
)
async def get_suite(
    category_slug: str, suite_slug: str, session: DBSession
) -> SuiteDetailResponse:
    """Get suite details with tests."""
    repo = CatalogRepository(session)
    suite = await repo.get_suite_by_path(category_slug, suite_slug)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite not found")
    return SuiteDetailResponse(
        slug=suite.slug,
        name=suite.name,
        description=suite.description,
        difficulty=suite.difficulty,
        author=suite.author,
        source=suite.source,
        category=category_slug,
        tests=[
            TestResponse(
                slug=t.slug,
                name=t.name,
                task_description=t.task_description,
                difficulty=t.difficulty,
                total_submissions=t.total_submissions,
                avg_score=t.avg_score,
                best_score=t.best_score,
            )
            for t in suite.tests
        ],
    )


@router.get(
    "/categories/{category_slug}/suites/{suite_slug}/leaderboard",
    response_model=dict[str, list[SubmissionResponse]],
)
async def get_suite_leaderboard(
    category_slug: str,
    suite_slug: str,
    session: DBSession,
    limit: int = 20,
) -> dict[str, list[SubmissionResponse]]:
    """Get leaderboard for all tests in a suite."""
    repo = CatalogRepository(session)
    suite = await repo.get_suite_by_path(category_slug, suite_slug)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite not found")

    result: dict[str, list[SubmissionResponse]] = {}
    for test in suite.tests:
        top = await repo.get_top_submissions(test.id, limit=limit)
        result[test.slug] = [
            SubmissionResponse(
                rank=i,
                agent_name=s.agent_name,
                score=s.score,
                quality_score=s.quality_score,
                completeness_score=s.completeness_score,
                efficiency_score=s.efficiency_score,
                cost_score=s.cost_score,
            )
            for i, s in enumerate(top, 1)
        ]
    return result
```

- [ ] **Step 3: Register routes**

In `packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py`, add:

```python
from atp.dashboard.v2.routes.catalog import router as catalog_router

router.include_router(catalog_router)
```

Follow the exact pattern used for other routers in that file.

- [ ] **Step 4: Write route tests**

Create `tests/unit/dashboard/catalog/__init__.py` and `tests/unit/dashboard/catalog/test_routes.py`:

```python
# tests/unit/dashboard/catalog/test_routes.py
"""Tests for catalog dashboard API routes."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from atp.dashboard.v2.routes.catalog import (
    list_categories,
    CategoryResponse,
)


@pytest.mark.anyio
async def test_list_categories_returns_data() -> None:
    """Categories endpoint returns formatted data."""
    mock_session = AsyncMock()
    mock_cat = MagicMock()
    mock_cat.slug = "coding"
    mock_cat.name = "Coding"
    mock_cat.description = "Code tests"
    mock_suite = MagicMock()
    mock_suite.tests = [MagicMock(), MagicMock()]
    mock_cat.suites = [mock_suite]

    mock_repo = AsyncMock()
    mock_repo.list_categories.return_value = [mock_cat]

    with patch(
        "atp.dashboard.v2.routes.catalog.CatalogRepository",
        return_value=mock_repo,
    ):
        result = await list_categories(mock_session)

    assert len(result) == 1
    assert result[0].slug == "coding"
    assert result[0].suite_count == 1
    assert result[0].test_count == 2
```

- [ ] **Step 5: Run tests**

Run: `uv run python -m pytest tests/unit/dashboard/catalog/test_routes.py -v`

- [ ] **Step 6: Ruff + commit**

```bash
uv run ruff format packages/atp-dashboard/ tests/unit/dashboard/catalog/
uv run ruff check packages/atp-dashboard/ tests/unit/dashboard/catalog/ --fix
git add packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py \
       packages/atp-dashboard/atp/dashboard/v2/routes/__init__.py \
       tests/unit/dashboard/catalog/
git commit -m "feat(catalog): add dashboard API routes for catalog browsing"
```

---

### Task 3: Catalog HTML Templates

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/catalog.html`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/catalog_suite.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/base.html` (add nav link)
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py` (add HTML routes)

- [ ] **Step 1: Read existing template patterns**

Read `packages/atp-dashboard/atp/dashboard/v2/templates/base.html` and one content template (e.g., `home.html`). Understand:
- Base layout structure (header, nav, content block)
- How to extend base
- CSS/styling approach

- [ ] **Step 2: Add HTML rendering routes**

Add to `catalog.py`:

```python
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(
    directory=str(Path(__file__).parent.parent / "templates")
)

@router.get("/", response_class=HTMLResponse)
async def catalog_page(request: Request, session: DBSession) -> HTMLResponse:
    """Render catalog browse page."""
    repo = CatalogRepository(session)
    cats = await repo.list_categories()
    if not cats:
        await sync_builtin_catalog(session)
        cats = await repo.list_categories()
    return templates.TemplateResponse(
        "catalog.html",
        {"request": request, "categories": cats},
    )

@router.get("/{category_slug}/{suite_slug}", response_class=HTMLResponse)
async def catalog_suite_page(
    request: Request, category_slug: str, suite_slug: str, session: DBSession,
) -> HTMLResponse:
    """Render suite detail page."""
    repo = CatalogRepository(session)
    suite = await repo.get_suite_by_path(category_slug, suite_slug)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite not found")
    leaderboard = {}
    for test in suite.tests:
        leaderboard[test.slug] = await repo.get_top_submissions(test.id, limit=10)
    return templates.TemplateResponse(
        "catalog_suite.html",
        {"request": request, "suite": suite, "category": category_slug, "leaderboard": leaderboard},
    )
```

- [ ] **Step 3: Create catalog.html template**

Create a simple, functional template that shows categories and their suites in a table. Use the existing base.html layout.

- [ ] **Step 4: Create catalog_suite.html template**

Shows suite details, test list with scores, and leaderboard per test.

- [ ] **Step 5: Add nav link in base.html**

Add a "Catalog" link to the navigation alongside existing links.

- [ ] **Step 6: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/catalog.html \
       packages/atp-dashboard/atp/dashboard/v2/templates/catalog_suite.html \
       packages/atp-dashboard/atp/dashboard/v2/templates/base.html \
       packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py
git commit -m "feat(catalog): add dashboard HTML pages for catalog browsing"
```

---

## Feature 3: Public Leaderboard API

### Task 4: Cross-Suite Leaderboard Endpoint

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py`
- Modify: `atp/catalog/repository.py`

- [ ] **Step 1: Add repository method for global leaderboard**

Add to `atp/catalog/repository.py`:

```python
async def get_global_leaderboard(self, limit: int = 50) -> list[dict]:
    """Get global agent rankings across all catalog tests.

    Returns list of dicts: agent_name, total_score, tests_completed, avg_score
    """
    result = await self._session.execute(
        select(
            CatalogSubmission.agent_name,
            func.count(CatalogSubmission.id).label("tests_completed"),
            func.avg(CatalogSubmission.score).label("avg_score"),
            func.sum(CatalogSubmission.score).label("total_score"),
        )
        .group_by(CatalogSubmission.agent_name)
        .order_by(func.avg(CatalogSubmission.score).desc())
        .limit(limit)
    )
    rows = result.all()
    return [
        {
            "agent_name": r.agent_name,
            "tests_completed": r.tests_completed,
            "avg_score": round(r.avg_score, 1) if r.avg_score else 0.0,
            "total_score": round(r.total_score, 1) if r.total_score else 0.0,
        }
        for r in rows
    ]
```

Add import: `from sqlalchemy import func`

- [ ] **Step 2: Add leaderboard API endpoint**

Add to `catalog.py` routes:

```python
class GlobalLeaderboardEntry(BaseModel):
    rank: int
    agent_name: str
    tests_completed: int
    avg_score: float
    total_score: float


@router.get("/leaderboard", response_model=list[GlobalLeaderboardEntry])
async def global_leaderboard(
    session: DBSession, limit: int = 50
) -> list[GlobalLeaderboardEntry]:
    """Global agent leaderboard across all catalog tests."""
    repo = CatalogRepository(session)
    entries = await repo.get_global_leaderboard(limit=limit)
    return [
        GlobalLeaderboardEntry(rank=i, **entry)
        for i, entry in enumerate(entries, 1)
    ]
```

- [ ] **Step 3: Write test**

```python
@pytest.mark.anyio
async def test_global_leaderboard_empty() -> None:
    """Global leaderboard returns empty list when no submissions."""
    mock_session = AsyncMock()
    mock_repo = AsyncMock()
    mock_repo.get_global_leaderboard.return_value = []

    with patch(
        "atp.dashboard.v2.routes.catalog.CatalogRepository",
        return_value=mock_repo,
    ):
        result = await global_leaderboard(mock_session, limit=50)

    assert result == []
```

- [ ] **Step 4: Ruff + commit**

```bash
uv run ruff format atp/catalog/repository.py packages/atp-dashboard/
uv run ruff check atp/catalog/ packages/atp-dashboard/ --fix
git add atp/catalog/repository.py packages/atp-dashboard/atp/dashboard/v2/routes/catalog.py \
       tests/unit/dashboard/catalog/test_routes.py
git commit -m "feat(catalog): add global leaderboard API endpoint"
```

---

## Task 5: Final Verification

- [ ] **Step 1: Run all catalog tests**

Run: `uv run python -m pytest tests/unit/catalog/ tests/unit/cli/test_catalog_cli.py -q`

- [ ] **Step 2: Run dashboard catalog tests**

Run: `uv run python -m pytest tests/unit/dashboard/catalog/ -q`

- [ ] **Step 3: Smoke test CLI**

```bash
uv run atp catalog sync
uv run atp catalog list
uv run atp catalog info coding/file-operations
uv run atp catalog run --help
```

- [ ] **Step 4: Quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run pyrefly check`

- [ ] **Step 5: Review commits**

Run: `git log --oneline -10`
