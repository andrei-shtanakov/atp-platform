# Run Detail Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/ui/runs/{run_id}` page that shows run summary stats, task results table with expandable details, and auto-refreshes while the run is in progress.

**Architecture:** Single route handler dispatching full page or HTMX partials based on `HX-Request`/`HX-Target` headers. Three Jinja2 templates (page + 2 partials). Minimal inline JS for row toggle. CSS additions for detail rows.

**Tech Stack:** FastAPI, SQLAlchemy, Jinja2, HTMX, Pico CSS

**Spec:** `docs/superpowers/specs/2026-04-04-run-detail-page-design.md`

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` | Modify | Add `ui_run_detail()` route + update `ui_cancel_run()` |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/run_detail.html` | Create | Full page template |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/run_header.html` | Create | Header card partial |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/run_tasks.html` | Create | Task table partial |
| `packages/atp-dashboard/atp/dashboard/v2/templates/ui/runs.html` | Modify | Make run ID a link to detail page |
| `packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css` | Modify | Add detail-row, JSON block styles |
| `tests/unit/dashboard/test_ui_routes.py` | Modify | Add run detail tests |

---

### Task 1: CSS styles for detail rows

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css`

- [ ] **Step 1: Add CSS for run detail page**

Append to end of `packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css`:

```css
/* Run detail page */
.run-meta {
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 1.5rem;
}

.run-meta a {
    color: #7c3aed;
}

.task-row {
    cursor: pointer;
}

.task-row:hover {
    background: #f0f0f0;
}

.detail-row {
    display: none;
    background: #f8f9fa;
}

.detail-row td {
    padding: 1rem 1.5rem;
}

.detail-row pre {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 0.375rem;
    padding: 0.75rem;
    max-height: 300px;
    overflow: auto;
    font-size: 0.8rem;
    margin: 0.5rem 0;
}

.detail-row details {
    margin-bottom: 0.5rem;
}

.detail-row summary {
    cursor: pointer;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.25rem 0;
}

.eval-summary {
    font-size: 0.85rem;
    color: #444;
    margin-bottom: 0.75rem;
}

.status-badge {
    color: #fff;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.85em;
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/static/css/ui.css
git commit -m "style: add CSS for run detail page"
```

---

### Task 2: Header partial template

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/run_header.html`

- [ ] **Step 1: Create the header partial**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/run_header.html`:

```html
<div id="run-header"
     {% if run.status.value in ("IN_PROGRESS", "PENDING") %}
     hx-get="/ui/runs/{{ run.id }}"
     hx-target="#run-header"
     hx-swap="outerHTML"
     hx-trigger="every 5s"
     {% endif %}>
    <div class="stat-cards">
        <div class="stat-card">
            <div class="label">Status</div>
            <div class="value">
                {% set s = run.status.value %}
                {% if s == "COMPLETED" %}
                <mark class="status-badge" style="background:var(--pico-color-green-550,#2d9e5f);">{{ s }}</mark>
                {% elif s == "IN_PROGRESS" %}
                <mark class="status-badge" style="background:var(--pico-color-azure-550,#1b6bb0);">{{ s }}</mark>
                {% elif s == "FAILED" %}
                <mark class="status-badge" style="background:var(--pico-color-red-550,#c0392b);">{{ s }}</mark>
                {% elif s == "PARTIAL" %}
                <mark class="status-badge" style="background:#e6a817;">{{ s }}</mark>
                {% else %}
                <mark class="status-badge" style="background:#aaa;">{{ s }}</mark>
                {% endif %}
            </div>
        </div>
        <div class="stat-card">
            <div class="label">Score</div>
            <div class="value">
                {% if run.total_score is not none %}
                {{ "%.2f"|format(run.total_score) }}
                {% else %}
                —
                {% endif %}
            </div>
        </div>
        <div class="stat-card">
            <div class="label">Progress</div>
            <div class="value">{{ run.current_task_index }} / {{ tasks_count }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Duration</div>
            <div class="value">
                {% if run.started_at is none %}
                    Not started
                {% elif run.finished_at is not none %}
                    {% set duration = (run.finished_at - run.started_at).total_seconds() %}
                    {% if duration >= 3600 %}
                        {{ "%.0f"|format(duration / 3600) }}h {{ "%.0f"|format((duration % 3600) / 60) }}m
                    {% elif duration >= 60 %}
                        {{ "%.0f"|format(duration / 60) }}m {{ "%.0f"|format(duration % 60) }}s
                    {% else %}
                        {{ "%.0f"|format(duration) }}s
                    {% endif %}
                {% else %}
                    {% set duration = (now - run.started_at).total_seconds() %}
                    {% if duration >= 3600 %}
                        {{ "%.0f"|format(duration / 3600) }}h {{ "%.0f"|format((duration % 3600) / 60) }}m (running)
                    {% elif duration >= 60 %}
                        {{ "%.0f"|format(duration / 60) }}m {{ "%.0f"|format(duration % 60) }}s (running)
                    {% else %}
                        {{ "%.0f"|format(duration) }}s (running)
                    {% endif %}
                {% endif %}
            </div>
        </div>
    </div>

    <div class="run-meta">
        Benchmark: <a href="/ui/benchmarks/{{ run.benchmark_id }}">{{ benchmark_name }}</a>
        &middot; Agent: <strong>{{ run.agent_name or "—" }}</strong>
        &middot; Run #{{ run.id }}
        {% if run.status.value in ("IN_PROGRESS", "PENDING") %}
        &middot;
        <button
            hx-post="/ui/runs/{{ run.id }}/cancel"
            hx-target="#run-header"
            hx-swap="outerHTML"
            style="padding:2px 10px;font-size:0.85em;display:inline;"
            onclick="return confirm('Cancel run #{{ run.id }}?')">
            Cancel
        </button>
        {% endif %}
    </div>
</div>
```

- [ ] **Step 2: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/run_header.html
git commit -m "feat(dashboard): add run detail header partial template"
```

---

### Task 3: Task table partial template

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/run_tasks.html`

- [ ] **Step 1: Create the task table partial**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/run_tasks.html`:

```html
<div id="run-tasks"
     {% if run.status.value in ("IN_PROGRESS", "PENDING") %}
     hx-get="/ui/runs/{{ run.id }}"
     hx-target="#run-tasks"
     hx-swap="outerHTML"
     hx-trigger="every 5s"
     {% endif %}>

<script>
function toggleDetail(row) {
    var detail = row.nextElementSibling;
    if (detail && detail.classList.contains('detail-row')) {
        detail.style.display = detail.style.display === 'table-row' ? 'none' : 'table-row';
    }
}
</script>

<table>
    <thead>
        <tr>
            <th>#</th>
            <th>Task</th>
            <th>Score</th>
            <th>Status</th>
        </tr>
    </thead>
    <tbody>
        {% for tr in task_results %}
        {% set req = tr.request or {} %}
        {% set task_desc = req.get("task", {}).get("description", "Task #" ~ tr.task_index) %}
        <tr class="task-row" onclick="toggleDetail(this)">
            <td>{{ tr.task_index }}</td>
            <td>{{ task_desc[:80] }}{% if task_desc|length > 80 %}…{% endif %}</td>
            <td>{% if tr.score is not none %}{{ "%.2f"|format(tr.score) }}{% else %}—{% endif %}</td>
            <td>
                {% if tr.score is not none %}
                <mark class="status-badge" style="background:#2d9e5f;">Scored</mark>
                {% elif tr.submitted_at is not none %}
                <mark class="status-badge" style="background:#1b6bb0;">Submitted</mark>
                {% else %}
                <mark class="status-badge" style="background:#aaa;">Pending</mark>
                {% endif %}
            </td>
        </tr>
        <tr class="detail-row">
            <td colspan="4">
                <div class="eval-summary">
                    {% set messages = [] %}
                    {% for er in (tr.eval_results or []) %}
                        {% for check in er.get("checks", []) %}
                            {% if check.get("message") %}
                                {% set _ = messages.append(check.get("name", "?") ~ ": " ~ check.get("message")) %}
                            {% endif %}
                        {% endfor %}
                    {% endfor %}
                    <strong>Evaluation:</strong> {{ messages|join("; ") or "No evaluation" }}
                </div>
                <details>
                    <summary>Request JSON</summary>
                    <pre><code>{{ tr.request|tojson(indent=2) }}</code></pre>
                </details>
                <details>
                    <summary>Response JSON</summary>
                    <pre><code>{{ tr.response|tojson(indent=2) }}</code></pre>
                </details>
                {% if tr.eval_results %}
                <details>
                    <summary>Eval Results JSON</summary>
                    <pre><code>{{ tr.eval_results|tojson(indent=2) }}</code></pre>
                </details>
                {% endif %}
            </td>
        </tr>
        {% else %}
        <tr>
            <td colspan="4" style="text-align:center;color:#888;">No task results yet.</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
</div>
```

- [ ] **Step 2: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/run_tasks.html
git commit -m "feat(dashboard): add run detail task table partial template"
```

---

### Task 4: Full page template

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/run_detail.html`

- [ ] **Step 1: Create the full page template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/run_detail.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}Run #{{ run.id }} - ATP Platform{% endblock %}

{% block content %}
<h2>Run #{{ run.id }}</h2>

{% include "ui/partials/run_header.html" %}

<h3>Task Results</h3>
{% include "ui/partials/run_tasks.html" %}
{% endblock %}
```

- [ ] **Step 2: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/run_detail.html
git commit -m "feat(dashboard): add run detail full page template"
```

---

### Task 5: Route handler

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`

- [ ] **Step 1: Add the import for TaskResult**

In `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`, update the import from `atp.dashboard.benchmark.models`:

Change:
```python
from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus
```
To:
```python
from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus, TaskResult
```

- [ ] **Step 2: Add the `ui_run_detail` route handler**

Add this route handler in `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`, **after** the `ui_runs()` function and **before** the `ui_cancel_run()` function:

```python
@router.get("/runs/{run_id}", response_class=HTMLResponse)
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
```

- [ ] **Step 3: Update `ui_cancel_run` to return header partial when called from detail page**

Replace the existing `ui_cancel_run` function with:

```python
@router.post("/runs/{run_id}/cancel", response_class=HTMLResponse)
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
        stmt = (
            select(Benchmark.name, Benchmark.tasks_count)
            .where(Benchmark.id == run.benchmark_id)
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
```

- [ ] **Step 4: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py
git commit -m "feat(dashboard): add run detail route handler with HTMX partial support"
```

---

### Task 6: Link from runs list to detail page

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/runs.html`

- [ ] **Step 1: Update the Run ID column to link to detail page**

In `packages/atp-dashboard/atp/dashboard/v2/templates/ui/runs.html`, replace:

```html
            <td>
                {% if run.benchmark_id %}
                <a href="/ui/benchmarks/{{ run.benchmark_id }}">#{{ run.id }}</a>
                {% else %}
                #{{ run.id }}
                {% endif %}
            </td>
```

With:

```html
            <td>
                <a href="/ui/runs/{{ run.id }}">#{{ run.id }}</a>
            </td>
```

- [ ] **Step 2: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/templates/ui/runs.html
git commit -m "feat(dashboard): link run IDs to detail page in runs list"
```

---

### Task 7: Tests

**Files:**
- Modify: `tests/unit/dashboard/test_ui_routes.py`

- [ ] **Step 1: Add run detail tests to the test file**

Add the following test class at the end of `tests/unit/dashboard/test_ui_routes.py`:

```python
class TestRunDetailPage:
    """Tests for run detail page /ui/runs/{run_id}."""

    @pytest.mark.anyio
    async def test_run_detail_returns_html(self, client) -> None:
        """GET /ui/runs/{id} returns HTML for existing run."""
        # First create a benchmark and run via the app's database
        from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus, TaskResult
        from atp.dashboard.database import get_database

        db = get_database()
        async with db.session() as session:
            bm = Benchmark(name="test-bm", tasks_count=2, suite={}, tags=[])
            session.add(bm)
            await session.flush()

            run = Run(
                benchmark_id=bm.id,
                agent_name="test-agent",
                status=RunStatus.COMPLETED,
                current_task_index=2,
                total_score=0.85,
            )
            session.add(run)
            await session.flush()

            tr = TaskResult(
                run_id=run.id,
                task_index=0,
                request={"task": {"description": "Solve 1+1"}},
                response={"answer": "2"},
                score=1.0,
            )
            session.add(tr)
            await session.commit()

            run_id = run.id

        resp = await client.get(f"/ui/runs/{run_id}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert f"Run #{run_id}" in resp.text
        assert "test-bm" in resp.text
        assert "test-agent" in resp.text
        assert "Solve 1+1" in resp.text

    @pytest.mark.anyio
    async def test_run_detail_404(self, client) -> None:
        """GET /ui/runs/99999 returns 404."""
        resp = await client.get("/ui/runs/99999")
        assert resp.status_code == 404
        assert "Not Found" in resp.text

    @pytest.mark.anyio
    async def test_run_detail_htmx_header_partial(self, client) -> None:
        """HTMX request with HX-Target: run-header returns partial."""
        from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus
        from atp.dashboard.database import get_database

        db = get_database()
        async with db.session() as session:
            bm = Benchmark(name="htmx-bm", tasks_count=1, suite={}, tags=[])
            session.add(bm)
            await session.flush()

            run = Run(
                benchmark_id=bm.id,
                agent_name="htmx-agent",
                status=RunStatus.IN_PROGRESS,
                current_task_index=0,
            )
            session.add(run)
            await session.commit()
            run_id = run.id

        resp = await client.get(
            f"/ui/runs/{run_id}",
            headers={"HX-Request": "true", "HX-Target": "run-header"},
        )
        assert resp.status_code == 200
        assert "run-header" in resp.text
        assert "hx-trigger" in resp.text  # Should have polling for in-progress

    @pytest.mark.anyio
    async def test_run_detail_htmx_tasks_partial(self, client) -> None:
        """HTMX request with HX-Target: run-tasks returns partial."""
        from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus
        from atp.dashboard.database import get_database

        db = get_database()
        async with db.session() as session:
            bm = Benchmark(name="tasks-bm", tasks_count=1, suite={}, tags=[])
            session.add(bm)
            await session.flush()

            run = Run(
                benchmark_id=bm.id,
                agent_name="tasks-agent",
                status=RunStatus.COMPLETED,
                current_task_index=1,
            )
            session.add(run)
            await session.commit()
            run_id = run.id

        resp = await client.get(
            f"/ui/runs/{run_id}",
            headers={"HX-Request": "true", "HX-Target": "run-tasks"},
        )
        assert resp.status_code == 200
        assert "run-tasks" in resp.text
        assert "hx-trigger" not in resp.text  # No polling for completed run
```

- [ ] **Step 2: Run the tests**

```bash
uv run python -m pytest tests/unit/dashboard/test_ui_routes.py::TestRunDetailPage -v
```

Expected: All 4 tests PASS.

- [ ] **Step 3: Run the full UI routes test suite**

```bash
uv run python -m pytest tests/unit/dashboard/test_ui_routes.py -v
```

Expected: All tests PASS (existing + new).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/dashboard/test_ui_routes.py
git commit -m "test(dashboard): add run detail page tests"
```

---

### Task 8: Lint and verify

- [ ] **Step 1: Run ruff format and check**

```bash
uv run ruff format packages/atp-dashboard/atp/dashboard/v2/routes/ui.py
uv run ruff check packages/atp-dashboard/atp/dashboard/v2/routes/ui.py
uv run ruff format tests/unit/dashboard/test_ui_routes.py
uv run ruff check tests/unit/dashboard/test_ui_routes.py
```

Expected: All pass.

- [ ] **Step 2: Run the full test suite for regressions**

```bash
uv run python -m pytest tests/unit/dashboard/ tests/unit/sdk/ tests/unit/evaluators/ -v --tb=short
```

Expected: No new failures.

- [ ] **Step 3: Manual smoke test (optional)**

```bash
uv run atp dashboard
```

Open `http://localhost:8000/ui/runs` in browser. Verify run IDs are now clickable links. Click one — should see the run detail page with header card and task table.

- [ ] **Step 4: Final commit if any lint fixes were needed**

```bash
git add -u
git commit -m "fix: lint fixes for run detail page"
```
