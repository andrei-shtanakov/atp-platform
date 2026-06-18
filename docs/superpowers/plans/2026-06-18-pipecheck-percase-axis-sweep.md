# Pipe-check per-case / axis-sweep dashboard view — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the per-case axis-level sweep and recall/precision/FP from pipe-check runs on the ATP dashboard, for visual analysis and a management demo.

**Architecture:** The pipe-check harness already writes full per-case grading to `case_details_<agent>.jsonl`. The dashboard `TestExecution` model already has the columns (`axis_level`, `critical_pass`, `malformed`, `recall`, `precision`, `fp_count`). So: (1) the importer reads the sibling `case_details` file and writes one `TestExecution` child row per case; (2) a new storage method aggregates the latest run into an axis sweep; (3) a new `/ui/eval-run/{suite}/{agent}` drill-down renders the sweep + per-case table, linked from the leaderboard. No DB migration.

**Tech Stack:** Python 3.12, SQLAlchemy async, FastAPI, Jinja2 (HTMX + Pico CSS), pytest + anyio, uv.

**Spec:** `docs/superpowers/specs/2026-06-18-pipecheck-percase-axis-sweep-design.md`

## Global Constraints

- Package manager: `uv` only (never `pip`). Run tools via `uv run`.
- Type hints on all code; `uv run pyrefly check` must pass (0 errors).
- `uv run ruff format .` + `uv run ruff check .` clean; line length 88.
- Async tests use `anyio` (the test file already defines an `anyio_backend` fixture returning `"asyncio"`).
- Branch: `r07/pipecheck-dashboard-import` (already created; do NOT work on `main`).
- Importer idempotency invariant: a parent `SuiteExecution` is skipped when its `run_uuid` already exists; child rows are written only when the parent is newly created.
- Dashboard must not import from `atp.reporters` (adapters package); define the axis order locally in the dashboard.

---

### Task 1: `parse_case_details()` + sibling-path helper

**Files:**
- Modify: `method/import_pipecheck_to_dashboard.py`
- Test: `tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py`

**Interfaces:**
- Produces: `parse_case_details(path: Path) -> list[dict[str, Any]]` — parse a `case_details_*.jsonl` file into per-case dicts; `[]` if missing/unreadable; skips blank and malformed lines.
- Produces: `case_details_path_for(report_path: Path) -> Path` — `report_benchmark_<agent>.json` → sibling `case_details_<agent>.jsonl`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py`:

```python
def test_parse_case_details_skips_blank_and_malformed(tmp_path: Path) -> None:
    p = tmp_path / "case_details_x.jsonl"
    p.write_text(
        '{"case_id":"a","axis_level":"clean","critical_pass":true}\n'
        "\n"
        "not json {\n"
        '{"case_id":"b","axis_level":"severe","critical_pass":false}'
    )
    rows = imp.parse_case_details(p)
    assert [r["case_id"] for r in rows] == ["a", "b"]


def test_parse_case_details_missing_file_is_empty(tmp_path: Path) -> None:
    assert imp.parse_case_details(tmp_path / "nope.jsonl") == []


def test_case_details_path_for_derives_sibling(tmp_path: Path) -> None:
    rp = tmp_path / "report_benchmark_claude_code.json"
    assert imp.case_details_path_for(rp).name == "case_details_claude_code.jsonl"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py -k "case_details or case_details_path" -v`
Expected: FAIL with `AttributeError: module 'import_pipecheck' has no attribute 'parse_case_details'`.

- [ ] **Step 3: Implement the helpers**

Add to `method/import_pipecheck_to_dashboard.py` (after `discover_reports`):

```python
def case_details_path_for(report_path: Path) -> Path:
    """Sibling ``case_details_<agent>.jsonl`` for a report_benchmark file.

    ``report_benchmark_<agent>.json`` -> ``case_details_<agent>.jsonl`` in the
    same directory.
    """
    stem = report_path.name.replace("report_benchmark_", "case_details_", 1)
    stem = stem.rsplit(".json", 1)[0]
    return report_path.with_name(f"{stem}.jsonl")


def parse_case_details(path: Path) -> list[dict[str, Any]]:
    """Parse a ``case_details_<agent>.jsonl`` file into per-case dicts.

    Returns [] if the file is missing or unreadable. Skips blank and malformed
    lines so a single bad line never sinks the whole import.
    """
    try:
        text = path.read_text()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py -k "case_details or case_details_path" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add method/import_pipecheck_to_dashboard.py tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py
git commit -m "feat(import): parse case_details jsonl + sibling-path helper"
```

---

### Task 2: Importer writes `TestExecution` child rows

**Files:**
- Modify: `method/import_pipecheck_to_dashboard.py` (the `import_reports` function)
- Test: `tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py`

**Interfaces:**
- Consumes: `parse_case_details`, `case_details_path_for` (Task 1);
  `ResultStorage.create_test_execution(suite_execution, test_id, test_name, *, started_at, dimensions)` and `ResultStorage.update_test_execution(execution, *, completed_at, success, score, status)` (existing storage).
- Produces: after each newly-created parent `SuiteExecution`, one `TestExecution` per case in the sibling `case_details` file (none when the file is absent).

- [ ] **Step 1: Write the failing tests**

Add to the test file:

```python
@pytest.mark.anyio
async def test_import_writes_case_rows_when_sibling_present(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    _write_report(tmp_path / "report_benchmark_claude_code.json", run_id="r1")
    (tmp_path / "case_details_claude_code.jsonl").write_text(
        '{"case_id":"c1","axis_level":"clean","critical_pass":true,'
        '"recall":1.0,"precision":1.0,"fp_count":0,"duration_seconds":1.0}\n'
        '{"case_id":"c2","axis_level":"severe","critical_pass":false,'
        '"recall":0.5,"precision":0.5,"fp_count":2,"duration_seconds":2.0}'
    )
    reports = imp.discover_reports(tmp_path)
    await imp.import_reports(reports, db_url=db_url)

    from atp.dashboard import init_database
    from atp.dashboard.models import TestExecution
    from sqlalchemy import select

    db = await init_database(url=db_url)
    async with db.session() as session:
        rows = (await session.execute(select(TestExecution))).scalars().all()
        by_id = {r.test_id: r for r in rows}
        assert set(by_id) == {"c1", "c2"}
        assert by_id["c2"].axis_level == "severe"
        assert by_id["c2"].fp_count == 2
        assert by_id["c1"].critical_pass is True
        assert by_id["c1"].task_type == "review"


@pytest.mark.anyio
async def test_import_writes_no_case_rows_when_sibling_absent(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    _write_report(
        tmp_path / "report_benchmark_codex_cli.json", run_id="r2", agent_id="codex_cli"
    )
    reports = imp.discover_reports(tmp_path)
    await imp.import_reports(reports, db_url=db_url)

    from atp.dashboard import init_database
    from atp.dashboard.models import TestExecution
    from sqlalchemy import select

    db = await init_database(url=db_url)
    async with db.session() as session:
        assert (await session.execute(select(TestExecution))).scalars().all() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py -k "case_rows" -v`
Expected: `test_import_writes_case_rows_when_sibling_present` FAILS (`set(by_id) == set()` — no child rows written yet); the absent-sibling test passes vacuously.

- [ ] **Step 3: Implement child-row writing in `import_reports`**

In `method/import_pipecheck_to_dashboard.py`, inside the per-report loop of `import_reports`, immediately after the existing `await storage.update_suite_execution(...)` call and before `imported += 1`, insert:

```python
            for c in parse_case_details(case_details_path_for(r.source_file)):
                started = r.started_at
                case_id = str(c.get("case_id") or "unknown")
                te = await storage.create_test_execution(
                    suite_execution=execution,
                    test_id=case_id,
                    test_name=case_id,
                    started_at=started,
                    dimensions={
                        "axis_level": c.get("axis_level"),
                        "critical_pass": c.get("critical_pass"),
                        "malformed": c.get("malformed"),
                        "recall": c.get("recall"),
                        "precision": c.get("precision"),
                        "fp_count": c.get("fp_count"),
                        "rubric_score": c.get("rubric_score"),
                        "task_type": r.task_type,
                    },
                )
                dur = float(c.get("duration_seconds") or 0.0)
                await storage.update_test_execution(
                    te,
                    completed_at=started + timedelta(seconds=dur),
                    success=bool(c.get("critical_pass")),
                    score=float(c.get("rubric_score") or 0.0),
                    status="completed",
                )
```

(`timedelta` is already imported at the top of the module.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py -v`
Expected: all pass (existing tests + the two new ones).

- [ ] **Step 5: Commit**

```bash
git add method/import_pipecheck_to_dashboard.py tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py
git commit -m "feat(import): write TestExecution child rows from case_details"
```

---

### Task 3: `--replace` purge flag

**Files:**
- Modify: `method/import_pipecheck_to_dashboard.py` (`import_reports` signature + `main` argparse)
- Test: `tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py`

**Interfaces:**
- Produces: `import_reports(reports, db_url=None, replace=False) -> tuple[int, int]`. When `replace=True`, delete existing `adapter="pipe-check"` rows (and their `TestExecution` children) for every `suite_name` present in `reports` before importing.
- Produces: `--replace` CLI flag wired into `main`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.anyio
async def test_replace_purges_prior_pipecheck_rows(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'd.db'}"
    _write_report(tmp_path / "report_benchmark_claude_code.json", run_id="r1")
    await imp.import_reports(imp.discover_reports(tmp_path), db_url=db_url)

    # second sweep: new run_uuid, replace=True must supersede the first
    _write_report(tmp_path / "report_benchmark_claude_code.json", run_id="r2")
    imported, skipped = await imp.import_reports(
        imp.discover_reports(tmp_path), db_url=db_url, replace=True
    )
    assert (imported, skipped) == (1, 0)

    from atp.dashboard import init_database
    from atp.dashboard.models import SuiteExecution
    from sqlalchemy import select

    db = await init_database(url=db_url)
    async with db.session() as session:
        runs = (
            await session.execute(
                select(SuiteExecution.run_uuid).where(
                    SuiteExecution.suite_name == "code-review"
                )
            )
        ).scalars().all()
        assert runs == ["r2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py -k replace -v`
Expected: FAIL — `import_reports()` got an unexpected keyword argument `replace`.

- [ ] **Step 3: Implement the purge + flag**

In `method/import_pipecheck_to_dashboard.py`, change the signature and add the purge at the start of the session block:

```python
async def import_reports(
    reports: list[ParsedReport], db_url: str | None = None, replace: bool = False
) -> tuple[int, int]:
    """Write reports into the dashboard store. Returns (imported, skipped).

    When ``replace`` is True, existing ``pipe-check`` rows for the suites in
    ``reports`` (and their child TestExecution rows) are deleted first, so a
    fresh sweep supersedes earlier partial data.
    """
```

Then, inside `async with db.session() as session:`, before `storage = ResultStorage(session)`, add:

```python
        from sqlalchemy import delete

        if replace:
            suite_names = {r.suite_name for r in reports}
            ids = (
                await session.execute(
                    select(SuiteExecution.id).where(
                        SuiteExecution.adapter == "pipe-check",
                        SuiteExecution.suite_name.in_(suite_names),
                    )
                )
            ).scalars().all()
            if ids:
                from atp.dashboard.models import TestExecution

                await session.execute(
                    delete(TestExecution).where(
                        TestExecution.suite_execution_id.in_(ids)
                    )
                )
                await session.execute(
                    delete(SuiteExecution).where(SuiteExecution.id.in_(ids))
                )
                print(f"--replace: purged {len(ids)} prior pipe-check run(s).")
```

In `main`, add the argparse flag (next to `--dry-run`):

```python
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Purge prior pipe-check runs for these suites before importing",
    )
```

And thread it into the call:

```python
    imported, skipped = asyncio.run(
        import_reports(reports, db_url=args.db, replace=args.replace)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add method/import_pipecheck_to_dashboard.py tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py
git commit -m "feat(import): --replace purges prior pipe-check rows"
```

---

### Task 4: Storage `suite_agent_case_detail()` + axis sweep

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/storage.py`
- Test: `tests/unit/dashboard/test_storage.py`

**Interfaces:**
- Produces: module-level `AXIS_ORDER = ["clean", "mild", "moderate", "severe", "very_severe"]` in `storage.py`.
- Produces: `ResultStorage.suite_agent_case_detail(suite_name: str, agent_name: str) -> dict[str, Any]` returning `{"run": dict | None, "cases": list[dict], "axis_sweep": list[dict]}`. `axis_sweep` entries are `{"axis_level", "n", "critical_pass", "pass_rate"}` in canonical order, only for levels with cases. Uses the latest completed run for the pair (newest `started_at`).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/dashboard/test_storage.py` (follow the file's existing fixture style for building a `ResultStorage` against an in-memory DB; if the file has a session/storage fixture, reuse it — otherwise build one as below):

```python
@pytest.mark.anyio
async def test_suite_agent_case_detail_axis_sweep() -> None:
    from datetime import datetime
    from atp.dashboard.database import Database
    from atp.dashboard.models import Base
    from atp.dashboard.storage import ResultStorage

    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with db.session() as session:
        storage = ResultStorage(session)
        ex = await storage.create_suite_execution_by_name(
            suite_name="code-review",
            agent_name="claude_code",
            started_at=datetime(2026, 6, 17, 10, 0, 0),
            adapter="pipe-check",
        )
        await storage.update_suite_execution(ex, status="completed")
        for axis, passed in [("clean", True), ("severe", True), ("severe", False)]:
            te = await storage.create_test_execution(
                suite_execution=ex,
                test_id=f"{axis}-{passed}",
                test_name=f"{axis}-{passed}",
                dimensions={"axis_level": axis, "critical_pass": passed},
            )
            await storage.update_test_execution(te, status="completed", success=passed)
        await session.commit()

        detail = await storage.suite_agent_case_detail("code-review", "claude_code")
        assert detail["run"] is not None
        assert len(detail["cases"]) == 3
        sweep = {a["axis_level"]: a for a in detail["axis_sweep"]}
        assert [a["axis_level"] for a in detail["axis_sweep"]] == ["clean", "severe"]
        assert sweep["clean"]["pass_rate"] == 1.0
        assert sweep["severe"]["n"] == 2
        assert sweep["severe"]["pass_rate"] == 0.5
    await db.close()


@pytest.mark.anyio
async def test_suite_agent_case_detail_no_run_is_empty() -> None:
    from atp.dashboard.database import Database
    from atp.dashboard.models import Base
    from atp.dashboard.storage import ResultStorage

    db = Database(url="sqlite+aiosqlite:///:memory:", echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with db.session() as session:
        storage = ResultStorage(session)
        detail = await storage.suite_agent_case_detail("code-review", "ghost")
        assert detail == {"run": None, "cases": [], "axis_sweep": []}
    await db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/dashboard/test_storage.py -k suite_agent_case_detail -v`
Expected: FAIL — `ResultStorage` has no attribute `suite_agent_case_detail`.

- [ ] **Step 3: Implement the method**

In `packages/atp-dashboard/atp/dashboard/storage.py`, add near the top (after imports):

```python
AXIS_ORDER = ["clean", "mild", "moderate", "severe", "very_severe"]
```

Add the method to `ResultStorage` (next to `suite_leaderboard`). Ensure `TestExecution` is imported (it already is — used by `create_test_execution`):

```python
    async def suite_agent_case_detail(
        self, suite_name: str, agent_name: str
    ) -> dict[str, Any]:
        """Latest completed run's per-case rows + axis sweep for (suite, agent).

        Returns {"run", "cases", "axis_sweep"}. ``axis_sweep`` carries the
        critical_pass rate per axis_level in canonical order (only levels that
        have cases). Empty when the pair has no completed run or no case rows.
        """
        run = (
            await self._session.execute(
                select(SuiteExecution)
                .where(
                    SuiteExecution.suite_name == suite_name,
                    SuiteExecution.agent_name == agent_name,
                    SuiteExecution.status == "completed",
                )
                .order_by(SuiteExecution.started_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if run is None:
            return {"run": None, "cases": [], "axis_sweep": []}
        rows = (
            await self._session.execute(
                select(TestExecution).where(
                    TestExecution.suite_execution_id == run.id
                )
            )
        ).scalars().all()
        cases = [
            {
                "case_id": t.test_id,
                "axis_level": t.axis_level,
                "critical_pass": t.critical_pass,
                "malformed": t.malformed,
                "recall": t.recall,
                "precision": t.precision,
                "fp_count": t.fp_count,
                "duration_seconds": t.duration_seconds,
            }
            for t in rows
        ]
        by_axis: dict[str, list[TestExecution]] = {}
        for t in rows:
            by_axis.setdefault(t.axis_level or "unknown", []).append(t)
        axis_sweep: list[dict[str, Any]] = []
        for level in AXIS_ORDER:
            group = by_axis.get(level)
            if not group:
                continue
            passed = sum(1 for t in group if t.critical_pass)
            axis_sweep.append(
                {
                    "axis_level": level,
                    "n": len(group),
                    "critical_pass": passed,
                    "pass_rate": round(passed / len(group), 6),
                }
            )
        return {
            "run": {
                "run_uuid": run.run_uuid,
                "started_at": run.started_at,
                "critical_pass_rate": run.critical_pass_rate,
                "breakpoint_axis_level": run.breakpoint_axis_level,
            },
            "cases": cases,
            "axis_sweep": axis_sweep,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/dashboard/test_storage.py -k suite_agent_case_detail -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/storage.py tests/unit/dashboard/test_storage.py
git commit -m "feat(dashboard): suite_agent_case_detail axis sweep query"
```

---

### Task 5: `/ui/eval-run/{suite}/{agent}` drill-down view + leaderboard link

**Files:**
- Modify: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` (add route after `ui_eval_trends`)
- Create: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_run_detail.html`
- Modify: `packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_leaderboard.html` (link agent → drill-down)
- Test: `tests/integration/dashboard/test_ui_eval_run_detail.py`

**Interfaces:**
- Consumes: `ResultStorage.suite_agent_case_detail` (Task 4).
- Produces: `GET /ui/eval-run/{suite_name}/{agent_name}` → HTML.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/dashboard/test_ui_eval_run_detail.py` (copy the `fresh_app` fixture and imports verbatim from `tests/integration/dashboard/test_ui_eval_leaderboard.py`, then add):

```python
@pytest.mark.anyio
async def test_eval_run_detail_renders_axis_sweep(fresh_app: tuple) -> None:
    from atp.dashboard.storage import ResultStorage

    app, db = fresh_app
    async with db.session() as session:
        storage = ResultStorage(session)
        ex = await storage.create_suite_execution_by_name(
            suite_name="code-review",
            agent_name="claude_code",
            started_at=datetime.now(),
            adapter="pipe-check",
        )
        await storage.update_suite_execution(ex, status="completed")
        for axis, passed in [("clean", True), ("severe", False)]:
            te = await storage.create_test_execution(
                suite_execution=ex,
                test_id=f"case-{axis}",
                test_name=f"case-{axis}",
                dimensions={"axis_level": axis, "critical_pass": passed},
            )
            await storage.update_test_execution(te, status="completed", success=passed)
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-run/code-review/claude_code")
    assert resp.status_code == 200
    assert "case-clean" in resp.text
    assert "severe" in resp.text


@pytest.mark.anyio
async def test_eval_run_detail_shows_notice_when_no_cases(fresh_app: tuple) -> None:
    from atp.dashboard.storage import ResultStorage

    app, db = fresh_app
    async with db.session() as session:
        storage = ResultStorage(session)
        ex = await storage.create_suite_execution_by_name(
            suite_name="code-review",
            agent_name="anthropic_api",
            started_at=datetime.now(),
            adapter="pipe-check",
        )
        await storage.update_suite_execution(ex, status="completed")
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ui/eval-run/code-review/anthropic_api")
    assert resp.status_code == 200
    assert "no per-case detail" in resp.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/dashboard/test_ui_eval_run_detail.py -v`
Expected: FAIL — 404 (route not registered) so `resp.status_code == 200` fails.

- [ ] **Step 3: Add the route**

In `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py`, after `ui_eval_trends`:

```python
@router.get("/eval-run/{suite_name}/{agent_name}", response_class=HTMLResponse)
@limiter.limit("120/minute")
async def ui_eval_run_detail(
    request: Request,
    session: DBSession,
    suite_name: str,
    agent_name: str,
) -> HTMLResponse:
    """Per-case axis-sweep drill-down for one (suite, agent) latest run."""
    user = await _get_ui_user(request, session)
    storage = ResultStorage(session)
    detail = await storage.suite_agent_case_detail(suite_name, agent_name)
    return _templates(request).TemplateResponse(
        request=request,
        name="ui/eval_run_detail.html",
        context={
            "active_page": "eval_leaderboard",
            "user": user,
            "suite_name": suite_name,
            "agent_name": agent_name,
            "run": detail["run"],
            "cases": detail["cases"],
            "axis_sweep": detail["axis_sweep"],
        },
    )
```

- [ ] **Step 4: Create the template**

Create `packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_run_detail.html`:

```html
{% extends "ui/base_ui.html" %}

{% block title %}Eval Run — {{ agent_name }} / {{ suite_name }} - ATP Platform{% endblock %}

{% block content %}
<h2>Eval Run: {{ agent_name }} <small>/ {{ suite_name }}</small></h2>
<p><a href="/ui/eval-leaderboard?suite_name={{ suite_name }}">← back to leaderboard</a></p>
{% if run %}
<p>Latest run <code>{{ run.run_uuid }}</code> ·
  critical_pass_rate
  {{ "%.3f"|format(run.critical_pass_rate) if run.critical_pass_rate is not none else "—" }}
  · breakpoint {{ run.breakpoint_axis_level or "—" }}</p>
{% endif %}
{% if not cases %}
<p>No per-case detail for this run (aggregate-only import — re-run the sweep with
  the current harness to capture <code>case_details</code>).</p>
{% else %}
<h3>Axis sweep</h3>
<table>
  <thead><tr><th>axis_level</th><th>cases</th><th>critical_pass</th><th>pass_rate</th></tr></thead>
  <tbody>
    {% for a in axis_sweep %}
    <tr><td>{{ a.axis_level }}</td><td>{{ a.n }}</td><td>{{ a.critical_pass }}</td>
      <td>{{ "%.3f"|format(a.pass_rate) }}</td></tr>
    {% endfor %}
  </tbody>
</table>
<h3>Per-case ({{ cases|length }})</h3>
<table>
  <thead><tr><th>case_id</th><th>axis_level</th><th>critical_pass</th><th>malformed</th>
    <th>recall</th><th>precision</th><th>fp_count</th></tr></thead>
  <tbody>
    {% for c in cases %}
    <tr><td>{{ c.case_id }}</td><td>{{ c.axis_level or "—" }}</td>
      <td>{{ "✓" if c.critical_pass else "✗" }}</td>
      <td>{{ "✓" if c.malformed else "" }}</td>
      <td>{{ "%.3f"|format(c.recall) if c.recall is not none else "—" }}</td>
      <td>{{ "%.3f"|format(c.precision) if c.precision is not none else "—" }}</td>
      <td>{{ c.fp_count if c.fp_count is not none else "—" }}</td></tr>
    {% endfor %}
  </tbody>
</table>
{% endif %}
{% endblock %}
```

- [ ] **Step 5: Link the leaderboard agent to the drill-down**

In `packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_leaderboard.html`, replace:

```html
    <tr><td>{{ loop.index }}</td><td>{{ e.agent_name }}</td>
```

with:

```html
    <tr><td>{{ loop.index }}</td>
      <td><a href="/ui/eval-run/{{ suite_name }}/{{ e.agent_name }}">{{ e.agent_name }}</a></td>
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/integration/dashboard/test_ui_eval_run_detail.py tests/integration/dashboard/test_ui_eval_leaderboard.py -v`
Expected: all pass (new route tests + unchanged leaderboard tests).

- [ ] **Step 7: Commit**

```bash
git add packages/atp-dashboard/atp/dashboard/v2/routes/ui.py \
  packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_run_detail.html \
  packages/atp-dashboard/atp/dashboard/v2/templates/ui/eval_leaderboard.html \
  tests/integration/dashboard/test_ui_eval_run_detail.py
git commit -m "feat(dashboard): /ui/eval-run drill-down with axis sweep + per-case table"
```

---

### Task 6: Full-suite verification + local re-import

**Files:** none (verification + operational).

- [ ] **Step 1: Format, type-check, lint**

```bash
uv run ruff format .
uv run ruff check .
uv run pyrefly check method/import_pipecheck_to_dashboard.py packages/atp-dashboard/atp/dashboard/storage.py packages/atp-dashboard/atp/dashboard/v2/routes/ui.py
```
Expected: format clean, ruff "All checks passed!", pyrefly "0 errors".

- [ ] **Step 2: Run the touched test suites**

```bash
uv run pytest tests/unit/method_spawners/test_import_pipecheck_to_dashboard.py \
  tests/unit/dashboard/test_storage.py \
  tests/integration/dashboard/test_ui_eval_run_detail.py \
  tests/integration/dashboard/test_ui_eval_leaderboard.py -v
```
Expected: all pass.

- [ ] **Step 3: Re-import current data with `--replace` and eyeball the view**

```bash
uv run python method/import_pipecheck_to_dashboard.py --replace
uv run atp dashboard
```
Open `http://localhost:8080/ui/eval-leaderboard?suite_name=code-review`, click
`claude_code` → confirm the axis sweep (clean→very_severe with gaps) and the
per-case recall/precision/fp table render. Click `ollama_qwen25_14b` (no
`case_details`) → confirm the "no per-case detail" notice.

- [ ] **Step 4: Commit any formatting-only changes**

```bash
git add -A && git commit -m "chore: format + lint pass for pipe-check axis-sweep view" || true
```

---

## Operational follow-up (NOT code — tracked, depends on external run)

> **This work is data-gated.** The view above is built for the full per-case
> shape but, until the weekend run, only `claude_code` / `codex_cli` /
> `deepseek` have `case_details`. The remaining agents render the
> "no per-case detail" notice — expected, not a bug.

- [ ] **Weekend paid run (planned ~2 days out, before the demo a week out).**
  Re-run `method/run_pipe_check.py` over **all required agents** so every agent
  gets `case_details_*.jsonl`. The current harness already writes them
  unconditionally (`run_pipe_check.py:403`).
- [ ] **New models join the roster** in that run — additional **local and API**
  models beyond today's 9. Register their shims/agent_ids in
  `method/run_pipe_check.py` (`SHIMS`, and `OLLAMA_MODELS` for local rows) and
  allowlist any new env vars in `ALLOWED_ENV`. The dashboard view is
  roster-agnostic (driven by whatever agents have rows) — **no dashboard code
  change needed** to absorb them.
- [ ] **Balance the axis grid** for the new sweep: today's case set skews to
  `severe`/`moderate` (clean/mild/very_severe = 9 each). For a clean demo curve,
  ensure the case set spans `clean → very_severe` more evenly (a
  `method/cases/` config concern for the run, not this plan).
- [ ] **After the run: re-import with `--replace`** to supersede today's partial
  data: `uv run python method/import_pipecheck_to_dashboard.py --replace
  --results-dir <new sweep dir>`.

## Self-Review

- **Spec coverage:** importer child rows (Tasks 1–2 ✓), keep-latest + `--replace`
  (Task 3 ✓, keep-latest is the existing `suite_leaderboard` behavior — no code),
  storage axis sweep (Task 4 ✓), drill-down view + leaderboard link + "no detail"
  notice (Task 5 ✓), coverage timeline / weekend run / new models / balanced grid
  (Operational follow-up ✓). Non-goals (no contract change, no `language`, no
  aggregation-semantics change) — respected; no task touches them.
- **Placeholder scan:** none — every code step carries complete code; every test
  step carries an assertion.
- **Type consistency:** `suite_agent_case_detail` returns `{"run","cases","axis_sweep"}`
  consumed verbatim by the Task 5 route/template; `import_reports(..., replace=False)`
  matches the Task 3 call site; `create_test_execution`/`update_test_execution`
  signatures match storage.py as read 2026-06-18; `AXIS_ORDER` defined once in
  storage.py (dashboard does not import the adapters reporter).
