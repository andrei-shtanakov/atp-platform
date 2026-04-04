# Run Detail Page — Design Spec

**Date**: 2026-04-04
**Status**: Approved (v2 — post-review)

## Overview

Add a run detail page at `/ui/runs/{run_id}` to the ATP dashboard. Users can drill into a specific benchmark run to see summary stats, task-by-task results, and raw request/response/eval data. The page auto-refreshes while the run is in progress.

## Layout

### Header Card

Four stat cards in a row:

| Stat | Source | Display |
|------|--------|---------|
| Status | `run.status` | Colored badge (green=COMPLETED, azure=IN_PROGRESS, red=FAILED, gray=CANCELLED/PENDING, yellow=PARTIAL) |
| Score | `run.total_score` | Float formatted to 2 decimals, or "—" if None |
| Progress | `current_task_index / benchmark.tasks_count` | "8 / 10" |
| Duration | see Duration Logic below | "2m 34s", "running...", or "Not started" |

**Duration logic:**
- `started_at` is None → "Not started"
- `started_at` set, `finished_at` is None → elapsed from `started_at` to server `now()`, display as "Xm Ys (running)"
- Both set → `finished_at - started_at`, display as "Xm Ys"

Below stats: breadcrumb links — Benchmark name → `/ui/benchmarks/{benchmark_id}`, Agent name, Run ID.

Cancel button visible when status is `IN_PROGRESS` or `PENDING`. Uses existing `POST /ui/runs/{run_id}/cancel` route in `routes/ui.py` (`ui_cancel_run`).

**Cancel UX flow**: Button uses `hx-post="/ui/runs/{id}/cancel"` with `hx-target="#run-header"` and `hx-swap="outerHTML"`. The server responds with the updated `run_header.html` partial showing CANCELLED status (no Cancel button). The task table refreshes on the next poll cycle (poll will also stop since status is now terminal).

### Auth

Follows existing pattern: all UI routes are gated by the `disable_auth` config check (same as other `/ui/*` pages). No per-run ownership check — any authenticated user can view any run.

### Task Results Table

Columns: `#` (task_index), Task (description), Score, Status.

**Task description extraction** from `TaskResult.request` JSON:
```python
request_data.get("task", {}).get("description", f"Task #{task_index}")
```
Fallback ensures display never breaks on missing/malformed JSON.

**Score display**: `task_result.score` formatted to 2 decimals, or "—" if None.

**Status column** — derived from TaskResult fields (no `status` field in the model):
- `score is not None` → "Scored" (green) — evaluation complete
- `score is None` and `submitted_at is not None` → "Submitted" (blue) — awaiting evaluation
- `score is None` and `submitted_at is None` → "Pending" (gray) — not yet submitted

**Row expansion**: Click on a data row toggles visibility of the next `<tr class="detail-row">` via minimal JS (`onclick` toggles `style.display` on the sibling row). This avoids invalid HTML from placing `<details>` inside `<table>`.

**Detail row content** (`<td colspan="4">`):
- **Eval summary line**: Extracted from `eval_results`:
  ```python
  # eval_results is list[dict] where each dict has "checks" list
  # Each check has: name, passed, score, message
  # Display: join all check messages, or "No evaluation" if empty/None
  messages = []
  for er in (task_result.eval_results or []):
      for check in er.get("checks", []):
          msg = check.get("message", "")
          if msg:
              messages.append(f"{check.get('name', '?')}: {msg}")
  summary = "; ".join(messages) or "No evaluation"
  ```
- **Collapsible sections** using `<details>`/`<summary>` (valid here — inside `<td>`, not `<tr>`):
  - Request JSON (`<pre><code>` pretty-printed)
  - Response JSON (`<pre><code>` pretty-printed)
  - Eval Results JSON (`<pre><code>` pretty-printed)

### HTMX Auto-Refresh

Partial detection uses the **HX-Request header** (standard HTMX pattern), not query params:

```python
is_htmx = request.headers.get("HX-Request") == "true"
target = request.headers.get("HX-Target", "")
```

When run status is non-terminal (`IN_PROGRESS` or `PENDING`):
- Header card: `hx-get="/ui/runs/{id}" hx-target="#run-header" hx-trigger="every 5s"`
- Task table: `hx-get="/ui/runs/{id}" hx-target="#run-tasks" hx-trigger="every 5s"`

Server checks `HX-Target` to determine which partial to return:
- `HX-Target: run-header` → returns `run_header.html`
- `HX-Target: run-tasks` → returns `run_tasks.html`
- No HX-Request → returns full `run_detail.html`

**Polling stops** when the partial response for terminal statuses omits the `hx-trigger` attribute. HTMX naturally stops polling when the replacement HTML has no trigger.

## Route Handler

```
GET /ui/runs/{run_id}
  HX-Target: run-header → returns run_header.html partial
  HX-Target: run-tasks  → returns run_tasks.html partial
  (full request)        → returns run_detail.html page
```

Query: join `Run` + `Benchmark` (for name, tasks_count) + eager-load `TaskResult` ordered by `task_index`.

Return 404 if run not found.

## Files

### New
- `templates/ui/run_detail.html` — full page extending `base_ui.html`
- `templates/ui/partials/run_header.html` — header card partial (included by run_detail.html and returned standalone for HTMX)
- `templates/ui/partials/run_tasks.html` — task table partial (same pattern)

### Modified
- `routes/ui.py` — add `ui_run_detail()` route handler
- `templates/ui/runs.html` — make run ID column a clickable link to `/ui/runs/{id}`
- `static/css/ui.css` — add styles for `.detail-row`, JSON blocks

### Tests
- `tests/unit/dashboard/test_ui_routes.py` — test run detail page returns HTML, test HTMX partials, test 404 for nonexistent run

## Data Flow

```
Browser GET /ui/runs/42
  → ui_run_detail(run_id=42)
    → SELECT Run + Benchmark + TaskResults WHERE run.id = 42
    → render run_detail.html with context {run, benchmark_name, tasks_count, task_results}
  → Full HTML response

Browser HTMX poll (every 5s while in-progress):
  GET /ui/runs/42 [HX-Request: true, HX-Target: run-header]
    → run_header.html partial (swaps #run-header div)
  GET /ui/runs/42 [HX-Request: true, HX-Target: run-tasks]
    → run_tasks.html partial (swaps #run-tasks div)
```

## Styling

Follow existing Pico CSS patterns. Status badge colors match `runs.html`. Detail rows: light gray background (`#f8f9fa`). JSON blocks: `<pre><code>` with monospace font, `overflow-x: auto`, max-height with scroll. Row toggle JS: ~5 lines inline.

## JS Philosophy

Minimal inline JS only where HTML cannot solve the problem:
- Row expand/collapse: ~5 lines of `onclick` toggle (HTML `<details>` is invalid inside `<table>`)
- Everything else is HTMX attributes or pure HTML

No external JS files, no build step, no framework. Consistent with Phase 1 approach.

## Non-Goals

- No WebSocket integration (HTMX polling is sufficient)
- No pagination for task results (benchmarks typically have <100 tasks)
- No editing/re-running from this page
- No per-run ownership check (all authenticated users see all runs)

## Future Considerations

- `agent_version` field on Run model — currently only `agent_name` (string) is stored, making it impossible to correlate historical runs with specific agent configs. Run detail page will surface this gap to users. Out of scope for this spec but noted for future model evolution.
