# El Farol Live Dashboard — Empty-State Skeleton

## Overview
Make the live tournament dashboard at `/ui/tournaments/{id}/live` render a usable, non-crashing skeleton when zero rounds have resolved, instead of the bare "Tournament starting…" placeholder. The dashboard then upgrades in place via the existing SSE-driven full-page reload once round 1 commits.

## Architecture Alignment
- Reuse the existing render pipeline (`match_detail.html` + `data_helpers.js` + `dashboard.js`) — no new templates, no new JS bundles.
- Reuse the existing SSE stream at `/api/v1/tournaments/{id}/dashboard/stream` and `live_subscriber.js` (no changes; it already triggers a full reload on `round_ended`, which is what gets us from skeleton → populated dashboard).
- Reuse `_reshape_from_tournament` as the single source of truth for the projection. The projection already emits `AGENTS`, `NUM_SLOTS`, `CAPACITY`, `NUM_DAYS=0`, `DATA=[]` correctly when no rounds are completed (verified by reading `el_farol_from_tournament.py:233-290`); no changes needed there.
- Drop the parallel `live_waiting=True` template branch — it was a workaround for an unsafe JS bundle, and once the JS is empty-state-safe the branch becomes dead code.

## Phase 1: Make `dashboard.js` empty-state safe

**Goal:** when `DATA.length === 0`, the JS renders a clean skeleton without throwing, without producing `NaN`/`Infinity` SVG attributes, and without clobbering the template's `—` KPI placeholders. The page must remain interactive enough to receive the SSE reload.

**Self-contained:** after this phase, the `live_waiting=True` template branch is no longer load-bearing for crash avoidance — it's just cosmetic, removed in Phase 2.

### Files
- [ ] `packages/atp-dashboard/atp/dashboard/v2/static/js/el_farol/dashboard.js` — MODIFIED

### Changes

#### 1. Fix `currentDay` initialization (line 17, 30)

```js
// Replace lines 17 + 30
const isEmpty = !Array.isArray(DATA) || DATA.length === 0 || NUM_DAYS === 0;
let currentDay = isEmpty ? 0 : Math.min(42, NUM_DAYS);
try {
  const d = parseInt(localStorage.getItem(LS_KEY_DAY) || '', 10);
  if (!isEmpty && d >= 1 && d <= NUM_DAYS) currentDay = d;
} catch (e) {}
```

Rationale: `currentDay = 0` is the canonical "no day selected" sentinel; every render function below short-circuits on it.

#### 2. Add an early-exit guard helper

Insert after `const $ = ...`:

```js
function isDataEmpty() { return !DATA || DATA.length === 0; }
```

#### 3. Guard each render function

Each function returns early when `isDataEmpty()`. The skeleton DOM (topbar, KPI strip with `—` placeholders, empty `#cards`, empty `#heatmap` SVG, `compare-empty` block) is whatever the template + previous render left in place — we explicitly do not write `NaN`/`undefined` into it.

- `renderRulesDiagram()` — early return if empty; clear `#rulesDiagram` and set `#rulesDayNum` text to `'—'`.
- `renderKPIs()` — early return if empty; explicitly set `kpiLeader/kpiLeaderSub/kpiSpread/kpiOver/kpiAtt/kpiBest/kpiBestSub/dayLabel/weekLabel` to `'—'` and the scrubber `value` to `0`. Skip the `localStorage.setItem` call.
- `renderCards()` — when empty, render one card per agent with: color swatch, agent id, profile, and microcopy `Waiting for round 1…` (no payoff, no sparkline, no pip row, no compare button — keep markup minimal). Wire no event listeners.
- `renderHeatmap()` — when empty, write a single centered `<text>` element to the SVG: `No rounds resolved yet`. No cells, no playhead line, no axis labels.
- `renderCompare()` — already handles `showSet.length === 0` with the "Hover a card to preview" empty state. **No change needed**, but verify the path `pinnedCompare = []` (empty initial value is fine) doesn't fall through into chart-drawing code. (It doesn't — the early `return` on line 284 covers it.)

#### 4. Disable interactive controls when empty

After `renderAll()` runs once, if `isDataEmpty()`:

```js
['playBtn'].forEach(id => $(id).disabled = true);
$('scrubber').disabled = true;
$('scrubber').max = 0;
// step()/jump() inline onclicks: guard inside the functions instead
```

Update `togglePlay()`, `step()`, `jump()` to early-return when `isDataEmpty()`. The keyboard handler (`ArrowLeft`/`ArrowRight`/`Space`) calls those functions, so it's covered transitively.

The scrubber `max="{{ num_days }}"` rendered server-side will be `0` when empty — set to `1` in the template instead (see Phase 2), and let the JS disable it.

### Tests First
No JS test infrastructure exists; manual verification only (see Quality Gates below). The Phase 2 Python integration tests cover the server-rendered HTML shape.

### Acceptance Criteria
- [ ] Loading `/ui/matches/{id}` for a synthetic match where `DATA=[]` no longer throws; opening DevTools console shows zero errors.
- [ ] Page shows agent cards with names + colors + "Waiting for round 1…" microcopy.
- [ ] Heatmap SVG contains a single text label, no `NaN` attributes anywhere in the DOM.
- [ ] KPI strip shows `—` placeholders.
- [ ] Scrubber, Play, Step, Jump are disabled.

---

## Phase 2: Drop the `live_waiting` branch from the route + template

**Goal:** with the JS now safe for empty `DATA`, route always renders the full dashboard. Removes a dead-code branch and a duplicate "starting" placeholder.

**Self-contained:** Phase 1 already lets the dashboard render with `NUM_DAYS=0`; this phase wires the route to actually use that path.

### Files
- [ ] `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` — MODIFIED (drop the `live_waiting` branch at lines 856–866)
- [ ] `packages/atp-dashboard/atp/dashboard/v2/templates/ui/match_detail.html` — MODIFIED (remove `{% elif live_waiting %}` block, lines 33–40; ensure `num_days` defaults survive)

### Changes

#### `ui.py` — collapse the branch

Replace lines 852–879 with a single unconditional context update:

```python
payload = await _reshape_from_tournament(tournament_id, session)
live_stream_url = f"/api/v1/tournaments/{tournament_id}/dashboard/stream"

context.update({
    "match_id": f"tournament-{tournament_id}",
    "payload_json": json.dumps(payload.model_dump(), separators=(",", ":")),
    "num_agents": len(payload.AGENTS),
    "num_days": payload.NUM_DAYS,         # 0 when no rounds resolved
    "num_slots": payload.NUM_SLOTS,
    "capacity": payload.CAPACITY,
    "langfuse_base": os.environ.get("ATP_LANGFUSE_BASE_URL", ""),
    "live_stream_url": live_stream_url,
})
```

No helper extraction needed — the projection already returns the right shape with zero rounds (`AGENTS` from participants, `NUM_SLOTS=16`, `CAPACITY` from `tournament.num_players`).

#### `match_detail.html` — remove dead branch

Delete lines 33–40 (the `{% elif live_waiting %}` … `{% endif %}` cluster including its `live_subscriber.js` script tag — the same script is already loaded inside the `{% else %}` branch when `live_stream_url` is set, lines 160–163).

In the `{% else %}` branch, the scrubber `max="{{ num_days }}"` will be `0` when empty — change to:

```html
<input type="range" min="1" max="{{ num_days or 1 }}" value="1" class="scrubber" id="scrubber">
```

so the slider DOM is well-formed; the JS disables it anyway when `isDataEmpty()`.

The topbar match-meta line `· {{ num_days }} days ·` is acceptable as `0 days` — leave it; it'll re-render correctly after the SSE reload.

### Tests First (test_*.py — `unit-tester` subagent owns these)

#### Test file: `tests/integration/dashboard/tournament/test_live_dashboard_ui.py` (NEW)

The existing `test_live_dashboard_endpoints.py` covers the JSON+SSE routes; this is the HTML/UI counterpart for the `/ui/tournaments/{id}/live` route specifically.

Pattern: use the same `v2_app` + `_patch_fresh_helpers` fixtures already established in `test_live_dashboard_endpoints.py` (import them or duplicate following the conftest pattern). Seed an `el_farol` `Tournament` via the same model imports.

Key test cases:
- `test_live_ui_zero_rounds_renders_skeleton` — seed a tournament with 2 participants and **zero** completed `Round` rows; GET `/ui/tournaments/{id}/live`. Assert:
  - HTTP 200
  - response text contains `id="atp-el-farol"` (full dashboard root)
  - response text contains `id="cards"`
  - response text contains `id="heatmap"`
  - response text contains `live_stream_url` (the SSE script tag is present)
  - response text does **not** contain `Tournament starting…` (the removed microcopy)
  - response text contains `0 days` in the match-meta line (sanity check on `num_days=0` plumbing)
- `test_live_ui_with_rounds_renders_dashboard` — seed a tournament with 2 participants and ≥1 completed round + actions; GET `/ui/tournaments/{id}/live`. Assert HTTP 200, contains `id="atp-el-farol"`, contains `payload_json` script block, `num_days` reflects the resolved round count. Regression guard for the route refactor.

Both tests follow the seeding pattern from `test_live_dashboard_endpoints.py::TestLiveDashboardSnapshot::test_anonymous_public_no_rounds_returns_empty_data` (seed `Tournament` + `Participant` rows; for the populated case, seed `Round` with `status=RoundStatus.COMPLETED` and matching `Action` rows).

#### Existing tests to verify still pass
- `tests/integration/dashboard/tournament/test_live_dashboard_endpoints.py::test_anonymous_public_no_rounds_returns_empty_data` — no projection changes were made, must remain green.
- `tests/integration/dashboard/test_match_detail_ui.py::TestMatchDetailUI::test_happy_path_renders_dashboard_skeleton` — `match_detail.html` change (`max="{{ num_days or 1 }}"`) must not affect the populated-match path.

### Acceptance Criteria
- [ ] `live_waiting` no longer appears in `ui.py` or `match_detail.html`.
- [ ] `GET /ui/tournaments/{id}/live` for a fresh tournament returns the full dashboard skeleton HTML with `live_stream_url` present.
- [ ] `GET /ui/tournaments/{id}/live` for a tournament with resolved rounds renders identically to before this plan.
- [ ] Manual: open `/ui/tournaments/{id}/live` in a browser before round 1 resolves, observe the skeleton with no console errors; wait for round 1 to commit; page reloads via SSE and shows the populated dashboard.

---

## Risks & Dependencies

- **`renderRulesDiagram` is invoked from the rules popover lifecycle, not just `renderAll`.** Verify the early-return is safe when the user opens the popover before round 1 — it currently triggers `renderRulesDiagram()` reading `DATA[currentDay-1]`. The Phase 1 guard covers this path.
- **`localStorage` restore corner case.** Pre-PR-108 sessions may have a stored day from a different match id; the new `if (!isEmpty && d >= 1 && d <= NUM_DAYS)` guard rejects it cleanly. (Note: a separate plan, out of scope here, will address localStorage `||'1'` defaulting on populated dashboards — Concern 2 in the source brief.)
- **No JS unit tests** — visual regressions in the skeleton aren't caught by CI. Mitigated by the Phase 2 integration tests asserting the HTML shape and by manual verification on the staging URL.
- **No SSE/bus changes.** PR #108's live-update infrastructure is untouched; the only behavioral change visible to the user is the first render before round 1.
