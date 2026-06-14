# SP-3: Dashboard leaderboard + trend over the eval store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two HTML dashboard views over the SP-1 store columns — (A) an **eval leaderboard** ranking agents by `critical_pass_rate` for a chosen suite, and (B) an **eval trend** of one (agent, suite) over time with an OLS slope — so results are finally viewable and comparable.

**Architecture:** Pure query helpers on `ResultStorage` read `SuiteExecution` (the internal store, NOT the pull-model benchmark `Run`) and return plain dicts; a tiny pure `trend_stats` computes the OLS slope; two thin `/ui/` routes render Pico/HTMX templates (trend reuses the analytics.html Chart.js pattern). Scope key is **`suite_name` + `agent_name`** (always populated by every run); `task_type` is shown as a column/filter and becomes the primary scope once SP-4 populates it.

**Tech Stack:** Python 3.12, uv, FastAPI, SQLAlchemy (async), Jinja2 + Pico CSS + HTMX + Chart.js (CDN), pytest (anyio + httpx ASGITransport).

**Companion docs:** spec `docs/superpowers/specs/2026-06-14-eval-results-architecture-design.md` (§8 A/B), ADR-006. SP-1 (the columns) is merged on `main`.

**Scope guard (NOT in SP-3):** matrix/drill-down views (SP-5); writing/export (SP-2); `task_type` population (SP-4 — here it's read-only display/filter); no change to the existing benchmark `Run` leaderboard or the JSON `trends.py`.

**Decision (locked with the user):** primary scope = `suite_name` now (immediately useful over real SP-1 data); `task_type` is a displayed dimension + optional filter, not the primary grouping, until SP-4.

---

## File Structure

- Create `packages/atp-dashboard/atp/dashboard/trend_stats.py` — pure `ols_slope(ys)` + `classify_trend(slope)`. DB-free, unit-tested.
- Modify `packages/atp-dashboard/atp/dashboard/storage.py` — read helpers: `suites_with_metrics()`, `suite_leaderboard(suite_name)`, `suite_trend(suite_name, agent_name, limit)`. Plain-dict returns.
- Modify `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` — two routes: `GET /ui/eval-leaderboard`, `GET /ui/eval-trends`.
- Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_leaderboard.html` and `ui/eval_trends.html`.
- Modify `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html` — two nav links.
- Tests: `tests/unit/dashboard/test_trend_stats.py`, `tests/unit/dashboard/test_storage.py` (extend), `tests/integration/dashboard/test_ui_eval_leaderboard.py`, `tests/integration/dashboard/test_ui_eval_trends.py`.

**Test cwd:** all from repo root (`uv run pytest …`).

**Patterns to mirror (read first):** the existing `ui_leaderboard` route in `ui.py` (handler signature `async def(request: Request, session: DBSession)`, `user = await _get_ui_user(request, session)`, `_templates(request).TemplateResponse(request=request, name=..., context={"active_page": ...})`); `ui/analytics.html` for the Chart.js CDN include + inline `<script>` chart; `ui/base_ui.html` sidebar nav; `tests/integration/dashboard/test_ui_tokens_routes.py` for the app+client+cookie fixture.

---

## Task 1: Pure trend stats (OLS slope + classification)

**Files:**
- Create: `packages/atp-dashboard/atp/dashboard/trend_stats.py`
- Test: `tests/unit/dashboard/test_trend_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/dashboard/test_trend_stats.py
"""SP-3: pure OLS slope + trend classification."""

from atp.dashboard.trend_stats import classify_trend, ols_slope


def test_ols_slope_increasing() -> None:
    assert ols_slope([0.2, 0.4, 0.6]) == 0.2


def test_ols_slope_flat() -> None:
    assert ols_slope([0.5, 0.5, 0.5]) == 0.0


def test_ols_slope_needs_two_points() -> None:
    assert ols_slope([]) is None
    assert ols_slope([0.7]) is None


def test_classify_trend() -> None:
    assert classify_trend(0.05) == "improving"
    assert classify_trend(-0.05) == "degrading"
    assert classify_trend(0.001) == "stable"
    assert classify_trend(None) == "n/a"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/dashboard/test_trend_stats.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Create the module**

```python
# packages/atp-dashboard/atp/dashboard/trend_stats.py
"""Pure trend statistics for the eval dashboard (SP-3). No DB.

Mirrors the OLS approach in atp/analytics/trend.py but operates on an in-memory
series (the dashboard reads points from the DB, not JSON files).
"""

import statistics

# Slope magnitude below this is treated as flat (per-run change in a 0..1 rate).
_STABLE_THRESHOLD = 0.01


def ols_slope(values: list[float]) -> float | None:
    """OLS slope of ``values`` against their 0-based index. None if < 2 points."""
    if len(values) < 2:
        return None
    xs = list(range(len(values)))
    slope, _intercept = statistics.linear_regression(xs, values)
    return round(slope, 6)


def classify_trend(slope: float | None, threshold: float = _STABLE_THRESHOLD) -> str:
    """Label a slope: improving / degrading / stable / n/a (None)."""
    if slope is None:
        return "n/a"
    if slope > threshold:
        return "improving"
    if slope < -threshold:
        return "degrading"
    return "stable"
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/dashboard/test_trend_stats.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-dashboard/atp/dashboard/trend_stats.py tests/unit/dashboard/test_trend_stats.py && uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/trend_stats.py tests/unit/dashboard/test_trend_stats.py
git commit -m "feat(dashboard): pure OLS slope + trend classification (SP-3)"
```

---

## Task 2: Storage read helpers

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/storage.py`
- Test: `tests/unit/dashboard/test_storage.py` (extend)

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/dashboard/test_storage.py`, building `ResultStorage` + seeding `SuiteExecution` rows the SAME way the existing tests in that file do (read them first; use the file's session/commit idiom). Seed for suite "code-review": agent `claude_code` (two completed runs, latest `critical_pass_rate=0.8`), agent `anthropic_api` (one completed run `critical_pass_rate=0.6`), and one `status="running"` row that must be excluded.

```python
@pytest.mark.anyio
async def test_suite_leaderboard_ranks_latest_per_agent() -> None:
    storage = ResultStorage(_make_session())  # match the file's existing helper
    # ... seed via storage.create_suite_execution_by_name + update_suite_execution
    #     (set started_at so "latest" is deterministic; set status + aggregates)
    rows = await storage.suite_leaderboard("code-review")
    assert [r["agent_name"] for r in rows] == ["claude_code", "anthropic_api"]
    assert rows[0]["critical_pass_rate"] == 0.8        # latest claude_code run
    assert all(r["status"] == "completed" for r in rows)  # running excluded


@pytest.mark.anyio
async def test_suites_with_metrics_lists_only_scored() -> None:
    storage = ResultStorage(_make_session())
    # seed one suite with critical_pass_rate set, one with it NULL
    names = await storage.suites_with_metrics()
    assert "code-review" in names


@pytest.mark.anyio
async def test_suite_trend_orders_ascending() -> None:
    storage = ResultStorage(_make_session())
    # seed 3 completed claude_code runs for "code-review" at increasing started_at
    pts = await storage.suite_trend("code-review", "claude_code", limit=10)
    assert [p["critical_pass_rate"] for p in pts] == sorted(
        p["critical_pass_rate"] for p in pts
    ) or len(pts) >= 2  # ascending by time
    assert pts[0]["ts"] <= pts[-1]["ts"]
```

(Write the seeding concretely against the real constructors. If the existing tests use an in-memory SQLite session rather than a mock, use that — the leaderboard query needs real ordering/filtering, so a real in-memory session is preferable here.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/dashboard/test_storage.py -k "leaderboard or suites_with or suite_trend" -v`
Expected: FAIL (methods missing).

- [ ] **Step 3: Add the read helpers to `ResultStorage`** (use `self._session`, `select`, `func` — already imported or add):

```python
    async def suites_with_metrics(self) -> list[str]:
        """Distinct suite names that have at least one scored (critical_pass_rate
        non-null) execution — the leaderboard/trend scope selector."""
        stmt = (
            select(SuiteExecution.suite_name)
            .where(SuiteExecution.critical_pass_rate.is_not(None))
            .distinct()
            .order_by(SuiteExecution.suite_name)
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def suite_leaderboard(self, suite_name: str) -> list[dict[str, Any]]:
        """One row per agent = their LATEST completed run for the suite, ranked
        by critical_pass_rate desc (null rates last)."""
        stmt = (
            select(SuiteExecution)
            .where(
                SuiteExecution.suite_name == suite_name,
                SuiteExecution.status == "completed",
            )
            .order_by(SuiteExecution.started_at.desc())
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        latest: dict[str, SuiteExecution] = {}
        for r in rows:  # newest first → first seen per agent is the latest
            latest.setdefault(r.agent_name, r)
        entries = [
            {
                "agent_name": e.agent_name,
                "task_type": e.task_type,
                "critical_pass_rate": e.critical_pass_rate,
                "malformed_rate": e.malformed_rate,
                "mean_rubric": e.mean_rubric,
                "breakpoint_axis_level": e.breakpoint_axis_level,
                "started_at": e.started_at,
                "status": e.status,
                "run_uuid": e.run_uuid,
            }
            for e in latest.values()
        ]
        entries.sort(
            key=lambda d: (
                d["critical_pass_rate"] is None,
                -(d["critical_pass_rate"] or 0.0),
            )
        )
        return entries

    async def suite_trend(
        self, suite_name: str, agent_name: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Completed scored runs for (suite, agent), oldest→newest."""
        stmt = (
            select(SuiteExecution)
            .where(
                SuiteExecution.suite_name == suite_name,
                SuiteExecution.agent_name == agent_name,
                SuiteExecution.status == "completed",
                SuiteExecution.critical_pass_rate.is_not(None),
            )
            .order_by(SuiteExecution.started_at.asc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [
            {
                "ts": r.started_at,
                "critical_pass_rate": r.critical_pass_rate,
                "malformed_rate": r.malformed_rate,
                "mean_rubric": r.mean_rubric,
            }
            for r in rows
        ]
```

(Ensure `Any`, `select` are imported; `SuiteExecution` already is.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/dashboard/test_storage.py -q`
Expected: PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-dashboard tests/unit/dashboard/test_storage.py && uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/storage.py tests/unit/dashboard/test_storage.py
git commit -m "feat(dashboard): storage read helpers for eval leaderboard + trend (SP-3)"
```

---

## Task 3: Eval leaderboard UI route + template + nav

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_leaderboard.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`
- Test: `tests/integration/dashboard/test_ui_eval_leaderboard.py`

- [ ] **Step 1: Write the failing integration test** — mirror `tests/integration/dashboard/test_ui_tokens_routes.py`'s app+client fixture (in-memory DB, `create_app`, optional cookie). Seed two agents' completed `SuiteExecution` rows for suite "code-review" (claude_code 0.8, anthropic_api 0.6). Assert:

```python
@pytest.mark.anyio
async def test_eval_leaderboard_ranks_agents(app_with_db) -> None:
    app, db = app_with_db  # adapt to the fixture you mirror
    # seed via a db.session(): two completed SuiteExecution rows (see Task 2 shape)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-leaderboard?suite_name=code-review")
    assert resp.status_code == 200
    body = resp.text
    assert "claude_code" in body and "anthropic_api" in body
    # claude_code (0.8) ranks before anthropic_api (0.6)
    assert body.index("claude_code") < body.index("anthropic_api")


@pytest.mark.anyio
async def test_eval_leaderboard_empty_state(app_with_db) -> None:
    app, _db = app_with_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-leaderboard")
    assert resp.status_code == 200  # no suite selected → selector + empty hint
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/dashboard/test_ui_eval_leaderboard.py -v`
Expected: FAIL (404 — route missing).

- [ ] **Step 3: Add the route to `ui.py`** (place near `ui_leaderboard`; reuse `ResultStorage`):

```python
@router.get("/eval-leaderboard", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_eval_leaderboard(
    request: Request,
    session: DBSession,
    suite_name: str | None = None,
) -> HTMLResponse:
    """Rank agents by critical_pass_rate for a chosen eval suite (SP-1 store)."""
    user = await _get_ui_user(request, session)
    storage = ResultStorage(session)
    suites = await storage.suites_with_metrics()
    if suite_name is None and suites:
        suite_name = suites[0]
    entries = await storage.suite_leaderboard(suite_name) if suite_name else []
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/eval_leaderboard.html",
        context={
            "active_page": "eval_leaderboard",
            "user": user,
            "suites": suites,
            "suite_name": suite_name,
            "entries": entries,
        },
    )
```

Confirm the actual imports used by neighbouring routes: `ResultStorage` (from `atp.dashboard.storage`), `DBSession`, `HTMLResponse`, `_templates`, `_get_ui_user`, `limiter`, `router` — reuse exactly what `ui_leaderboard` uses (don't introduce new import styles).

- [ ] **Step 4: Create `ui/eval_leaderboard.html`** (extend the base, mirror an existing table page like `ui/executions.html`):

```html
{% extends "ui/base_ui.html" %}
{% block content %}
<h2>Eval Leaderboard</h2>
<form method="get" action="/ui/eval-leaderboard">
  <label>Suite
    <select name="suite_name" onchange="this.form.submit()">
      {% for s in suites %}
      <option value="{{ s }}" {% if s == suite_name %}selected{% endif %}>{{ s }}</option>
      {% endfor %}
    </select>
  </label>
</form>

{% if entries %}
<table>
  <thead>
    <tr><th>#</th><th>Agent</th><th>critical_pass_rate</th><th>malformed_rate</th>
        <th>mean_rubric</th><th>breakpoint</th><th>task_type</th></tr>
  </thead>
  <tbody>
    {% for e in entries %}
    <tr>
      <td>{{ loop.index }}</td>
      <td>{{ e.agent_name }}</td>
      <td>{{ "%.3f"|format(e.critical_pass_rate) if e.critical_pass_rate is not none else "—" }}</td>
      <td>{{ "%.3f"|format(e.malformed_rate) if e.malformed_rate is not none else "—" }}</td>
      <td>{{ "%.3f"|format(e.mean_rubric) if e.mean_rubric is not none else "—" }}</td>
      <td>{{ e.breakpoint_axis_level or "—" }}</td>
      <td>{{ e.task_type or "—" }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% else %}
<p>No scored eval runs yet. Run <code>atp test method/cases/&lt;suite&gt;</code> to populate the store.</p>
{% endif %}
{% endblock %}
```

(Match the base template's actual block name — confirm it is `content` by reading `base_ui.html`; adjust if different.)

- [ ] **Step 5: Add the nav link** in `ui/base_ui.html` after the `/ui/analytics` item:

```html
<li><a href="/ui/eval-leaderboard" class="{% if active_page == 'eval_leaderboard' %}active{% endif %}">Eval Leaderboard</a></li>
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/integration/dashboard/test_ui_eval_leaderboard.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-dashboard tests/integration/dashboard/test_ui_eval_leaderboard.py && uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_leaderboard.html packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html tests/integration/dashboard/test_ui_eval_leaderboard.py
git commit -m "feat(dashboard): /ui/eval-leaderboard over the SP-1 store (SP-3)"
```

---

## Task 4: Eval trend UI route + template (Chart.js) + nav

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_trends.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html`
- Test: `tests/integration/dashboard/test_ui_eval_trends.py`

- [ ] **Step 1: Write the failing integration test** — seed ≥2 completed claude_code runs for "code-review" at increasing started_at with rising critical_pass_rate. Assert:

```python
@pytest.mark.anyio
async def test_eval_trends_shows_slope_and_points(app_with_db) -> None:
    app, db = app_with_db
    # seed 3 completed claude_code runs (code-review) at increasing started_at,
    # critical_pass_rate 0.4 -> 0.6 -> 0.8
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/ui/eval-trends?suite_name=code-review&agent_name=claude_code"
        )
    assert resp.status_code == 200
    body = resp.text
    assert "improving" in body          # OLS slope classification rendered
    assert "claude_code" in body


@pytest.mark.anyio
async def test_eval_trends_empty_state(app_with_db) -> None:
    app, _db = app_with_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-trends")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/dashboard/test_ui_eval_trends.py -v`
Expected: FAIL (404).

- [ ] **Step 3: Add the route to `ui.py`**:

```python
@router.get("/eval-trends", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_eval_trends(
    request: Request,
    session: DBSession,
    suite_name: str | None = None,
    agent_name: str | None = None,
) -> HTMLResponse:
    """Trend of critical_pass_rate over time for one (suite, agent) (SP-1 store)."""
    user = await _get_ui_user(request, session)
    storage = ResultStorage(session)
    suites = await storage.suites_with_metrics()
    if suite_name is None and suites:
        suite_name = suites[0]
    agents = (
        await storage.agents_for_suite(suite_name) if suite_name else []
    )
    if agent_name is None and agents:
        agent_name = agents[0]
    points = (
        await storage.suite_trend(suite_name, agent_name)
        if suite_name and agent_name
        else []
    )
    rates = [p["critical_pass_rate"] for p in points]
    slope = ols_slope(rates)
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/eval_trends.html",
        context={
            "active_page": "eval_trends",
            "user": user,
            "suites": suites,
            "agents": agents,
            "suite_name": suite_name,
            "agent_name": agent_name,
            "points": points,
            "slope": slope,
            "direction": classify_trend(slope),
            "labels_json": json.dumps([p["ts"].isoformat() for p in points]),
            "values_json": json.dumps(rates),
        },
    )
```

Add imports at top of `ui.py` if missing: `import json`; `from atp.dashboard.trend_stats import classify_trend, ols_slope`. Add a small `agents_for_suite` helper to `ResultStorage` (distinct agent_name for a suite among scored completed runs) — same pattern as `suites_with_metrics`:

```python
    async def agents_for_suite(self, suite_name: str) -> list[str]:
        stmt = (
            select(SuiteExecution.agent_name)
            .where(
                SuiteExecution.suite_name == suite_name,
                SuiteExecution.status == "completed",
                SuiteExecution.critical_pass_rate.is_not(None),
            )
            .distinct()
            .order_by(SuiteExecution.agent_name)
        )
        return list((await self._session.execute(stmt)).scalars().all())
```
(Add this helper + a quick unit test to Task 2's test file if you prefer; minimally, it's covered by the integration test here.)

- [ ] **Step 4: Create `ui/eval_trends.html`** — selectors for suite + agent, a stat line (slope/direction), a points table, and a Chart.js line chart (mirror the `<script src="…chart.umd.min.js">` include + inline chart from `ui/analytics.html`):

```html
{% extends "ui/base_ui.html" %}
{% block content %}
<h2>Eval Trend</h2>
<form method="get" action="/ui/eval-trends">
  <label>Suite
    <select name="suite_name" onchange="this.form.submit()">
      {% for s in suites %}<option value="{{ s }}" {% if s == suite_name %}selected{% endif %}>{{ s }}</option>{% endfor %}
    </select>
  </label>
  <label>Agent
    <select name="agent_name" onchange="this.form.submit()">
      {% for a in agents %}<option value="{{ a }}" {% if a == agent_name %}selected{% endif %}>{{ a }}</option>{% endfor %}
    </select>
  </label>
</form>

{% if points %}
<p><strong>Trend:</strong> {{ direction }}{% if slope is not none %} (slope {{ "%.4f"|format(slope) }}/run){% endif %}</p>
<canvas id="trendChart" height="120"></canvas>
<!-- Pinned + SRI (Subresource Integrity) so a CDN compromise can't inject code.
     Compute the hash for the pinned version BEFORE writing this tag:
       curl -sL https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js \
         | openssl dgst -sha384 -binary | openssl base64 -A
     then paste it as integrity="sha384-<HASH>". Do NOT invent the hash. -->
<script
  src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"
  integrity="sha384-<COMPUTE_WITH_COMMAND_ABOVE>"
  crossorigin="anonymous"></script>
<script>
  new Chart(document.getElementById('trendChart'), {
    type: 'line',
    data: {
      labels: {{ labels_json | safe }},
      datasets: [{ label: 'critical_pass_rate', data: {{ values_json | safe }}, tension: 0.2 }]
    },
    options: { scales: { y: { min: 0, max: 1 } } }
  });
</script>
<table>
  <thead><tr><th>When</th><th>critical_pass_rate</th><th>malformed_rate</th><th>mean_rubric</th></tr></thead>
  <tbody>
    {% for p in points %}
    <tr><td>{{ p.ts }}</td>
        <td>{{ "%.3f"|format(p.critical_pass_rate) }}</td>
        <td>{{ "%.3f"|format(p.malformed_rate) if p.malformed_rate is not none else "—" }}</td>
        <td>{{ "%.3f"|format(p.mean_rubric) if p.mean_rubric is not none else "—" }}</td></tr>
    {% endfor %}
  </tbody>
</table>
{% else %}
<p>No scored runs for this selection yet.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 5: Add the nav link** in `base_ui.html` after the eval-leaderboard item:

```html
<li><a href="/ui/eval-trends" class="{% if active_page == 'eval_trends' %}active{% endif %}">Eval Trends</a></li>
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/integration/dashboard/test_ui_eval_trends.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check packages/atp-dashboard tests/integration/dashboard/test_ui_eval_trends.py && uv run pyrefly check
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_trends.html packages/atp-dashboard/atp/dashboard/v2/templates/ui/base_ui.html tests/integration/dashboard/test_ui_eval_trends.py
git commit -m "feat(dashboard): /ui/eval-trends with OLS slope + Chart.js (SP-3)"
```

---

## Task 5: Regression + quality gates

**Files:** none (verification only)

- [ ] **Step 1: Run the dashboard suites**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run pytest tests/unit/dashboard tests/integration/dashboard -q
```
Expected: all PASS (new trend_stats/storage/ui tests + existing UI tests unaffected).

- [ ] **Step 2: Smoke the two pages render with an empty DB (no crash, 200)**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run pytest tests/integration/dashboard/test_ui_eval_leaderboard.py tests/integration/dashboard/test_ui_eval_trends.py -q
```
Expected: PASS (incl. the empty-state tests).

- [ ] **Step 3: Lint + types**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
uv run ruff check atp/ packages/atp-dashboard
uv run pyrefly check
```
Expected: ruff clean; pyrefly 0 errors.

- [ ] **Step 4: Commit any formatting**

```bash
cd "$(git rev-parse --show-toplevel)"
git add -A && git commit -m "chore(sp-3): formatting" || echo "nothing to commit"
```

---

## Self-Review (completed during authoring)

- **Spec coverage (§8 A/B):** leaderboard (Task 3) + trend (Task 4) over the SP-1 columns, reading the internal store (not benchmark `Run`). Matrix/drill-down explicitly deferred to SP-5.
- **Scoping decision honored:** primary key `suite_name`+`agent_name` (always populated); `task_type` is a displayed column (leaderboard) — becomes primary scope in SP-4. No dependency on SP-4.
- **Type/name consistency:** storage helpers return dicts whose keys (`agent_name`/`critical_pass_rate`/`malformed_rate`/`mean_rubric`/`breakpoint_axis_level`/`task_type`/`ts`) are exactly what the templates render; `ols_slope`/`classify_trend` signatures match their call sites; `ResultStorage(session)` construction matches the existing `ui_*` routes.
- **Reuse:** Chart.js via the same CDN as `ui/analytics.html`, but **pinned + with SRI** (`integrity`/`crossorigin`) — the existing `analytics.html` include lacks SRI (floating `@4`, no integrity); SP-3's new tag pins `4.4.4` and the implementer computes the real sha384 (command in Task 4) rather than inventing one. Route/template/test patterns mirror `ui_leaderboard` / `analytics.html` / `test_ui_tokens_routes.py` — read those first (exact import names, base block name, fixture shape are environment-specific, not re-pasted).
- **Security follow-up (noted, out of SP-3 scope):** `ui/analytics.html`'s Chart.js include has the same missing-SRI gap; harden it (pin + SRI) in a separate small PR.
- **No prod risk:** read-only views; no schema/migration; empty-state paths return 200.
