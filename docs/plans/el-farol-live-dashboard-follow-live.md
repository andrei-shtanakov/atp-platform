# El Farol Live Dashboard — Follow-Live Mode

## Overview
The live dashboard at `/ui/tournaments/{id}/live` snaps to **Day 1** every time a round resolves, because `dashboard.js:30` restores `currentDay` from `localStorage` with the literal fallback `'1'` and the SSE listener triggers a full-page reload on every `round_ended`. The user has to click ⏭ / Forward after every reload to see the latest round.

Add an opt-out **follow-live** mode that auto-snaps to the latest day on each reload, while still letting the user scrub backwards to inspect history.

This plan assumes [`el-farol-live-dashboard-empty-skeleton.md`](el-farol-live-dashboard-empty-skeleton.md) has shipped — `currentDay = 0` and `isDataEmpty()` are pre-existing concepts.

## Architecture Alignment
- Reuse the existing full-page-reload update model. No `dashboard.js` refactor to support in-place updates (out of scope).
- Reuse the existing `atp-live-pill` topbar element (`match_detail.html:47-49`, styled in `el_farol.css:257-262`); add one sibling element for the resume affordance.
- Reuse the existing `window.jump` / `window.step` / `window.togglePlay` globals exposed by `dashboard.js:42`.
- Coordinate the two scripts (`dashboard.js`, `live_subscriber.js`) via a single `sessionStorage` key — no direct cross-script function calls.
- Replay mode (`/ui/matches/{match_id}` when `window.__ATP_LIVE_URL__` is unset) is **untouched**.

## Behavior Matrix

| live mode? | follow-live flag | event                      | resulting `currentDay` | resulting flag |
| ---------- | ---------------- | -------------------------- | ---------------------- | -------------- |
| no         | n/a              | page load                  | `localStorage` restore | n/a            |
| yes        | `true` (default) | page load, `NUM_DAYS > 0`  | `NUM_DAYS`             | `true`         |
| yes        | `true`           | page load, `NUM_DAYS = 0`  | `0` (skeleton, Plan 1) | `true` (no-op) |
| yes        | `true`           | scrubber `input`           | user-selected day      | **`false`**    |
| yes        | `true`           | `step(±1)` / arrow keys    | clamped day            | **`false`**    |
| yes        | `true`           | `togglePlay()`             | unchanged              | **`false`**    |
| yes        | `true`           | `jump(d)` where `d < NUM_DAYS` | `d`               | **`false`**    |
| yes        | `true`           | `jump(NUM_DAYS)` (⏭ Last)  | `NUM_DAYS`             | `true` (stays) |
| yes        | `false`          | page load                  | `localStorage` restore | `false`        |
| yes        | `false`          | resume-pill click          | `NUM_DAYS`             | **`true`**     |
| yes        | `false`          | `jump(NUM_DAYS)` (⏭ Last)  | `NUM_DAYS`             | **`true`**     |

Rationale:
- `jump(NUM_DAYS)` is a "show me the latest" intent; treat it as a re-enable signal.
- `jump(d < NUM_DAYS)` (heatmap cell click, ⏮ Day 1, etc.) is "show me a specific older day"; treat as user opting out.
- `togglePlay()` starts auto-stepping from `currentDay` forward; once the user is in playback control, follow-live is off by definition.

## Storage Contract

- **Key**: `atp-follow-live-tournament-{tournament_id}`
- **Scope**: `sessionStorage` (per-tab; survives reloads in same tab; dies on tab close).
- **Values**:
  - `'false'` → follow-live disabled
  - any other value (including absent) → follow-live enabled (default)
- **Why sessionStorage not localStorage**: the user's "I want to scrub history" intent is tab-bound. Opening a fresh tab on the same tournament should default to follow-live = on.

The tournament id is read from the existing `window.__ATP_MATCH__.match_id` payload, which for live mode is `tournament-{tournament_id}` (set in `ui.py:818`). No new globals required.

---

## Phase 1: `dashboard.js` — read flag, force latest, flip flag on user nav

**Goal:** when `window.__ATP_LIVE_URL__` is set, read the `sessionStorage` flag during init and override `currentDay` to `NUM_DAYS`. On user navigation, set the flag to `'false'`. Replay mode behavior is identical to today.

**Self-contained:** after this phase the dashboard correctly auto-advances on each SSE reload, but the user has no UI affordance to re-enable follow-live once they've opted out (Phase 2 adds the pill).

### Files
- [ ] `packages/atp-dashboard/atp/dashboard/v2/static/js/el_farol/dashboard.js` — MODIFIED

### Changes

#### 1. Add helpers near the existing `LS_KEY_DAY` block (after line 31)

```js
// Live-mode follow-latest support. The SSE-driven page reload pattern
// makes the per-load currentDay decision the only place to enforce
// "auto-advance to the latest round" — once the page is rendered we
// can't update window.ATP without a refactor. The sessionStorage flag
// survives reloads but dies on tab close so a fresh tab defaults to
// follow-live = on.
const isLiveMode = !!window.__ATP_LIVE_URL__;
const FOLLOW_LIVE_KEY = 'atp-follow-live-' + (window.__ATP_MATCH__ ? window.__ATP_MATCH__.match_id : 'default');

function isFollowingLive() {
  if (!isLiveMode) return false;
  try { return sessionStorage.getItem(FOLLOW_LIVE_KEY) !== 'false'; }
  catch (e) { return true; }
}

function setFollowingLive(enabled) {
  if (!isLiveMode) return;
  try {
    if (enabled) sessionStorage.removeItem(FOLLOW_LIVE_KEY);
    else sessionStorage.setItem(FOLLOW_LIVE_KEY, 'false');
  } catch (e) {}
  // Notify live_subscriber.js (Phase 2) that the pill state changed.
  window.dispatchEvent(new CustomEvent('atp:follow-live-changed', {
    detail: { enabled }
  }));
}

window.atpIsFollowingLive = isFollowingLive;  // for live_subscriber.js
window.atpSetFollowingLive = setFollowingLive;  // for live_subscriber.js
```

#### 2. Override `currentDay` after the existing localStorage restore

Insert after the existing `try { const d = parseInt(localStorage.getItem(LS_KEY_DAY)... } catch(e){}` block (line 30 / post-Plan-1 location):

```js
// Live mode + follow-live enabled: ignore the localStorage restore and
// snap to the latest resolved day. Plan 1's isDataEmpty() handling
// stays first — when DATA is empty, currentDay = 0 already.
if (!isDataEmpty() && isFollowingLive() && NUM_DAYS > 0) {
  currentDay = NUM_DAYS;
}
```

#### 3. Flip the flag off on manual nav

Modify the existing handlers:

```js
// Line ~40: step() — flip off
function step(d) {
  setFollowingLive(false);
  currentDay = Math.max(1, Math.min(NUM_DAYS, currentDay + d));
  renderAll();
}

// Line ~41: jump() — flip on iff jumping to NUM_DAYS, else off
function jump(d) {
  setFollowingLive(d === NUM_DAYS);
  currentDay = d;
  renderAll();
}

// Line ~34: togglePlay() — flip off when starting playback
function togglePlay() {
  playing = !playing;
  if (playing) setFollowingLive(false);
  // ...rest unchanged
}

// Line ~43: scrubber input listener — flip off
$('scrubber').addEventListener('input', e => {
  setFollowingLive(false);
  currentDay = parseInt(e.target.value);
  renderAll();
});
```

The keyboard handler (line ~430) calls `step()` and `togglePlay()`, so it's covered transitively.

Heatmap cell `jump(parseInt(el.dataset.day))` calls (line 267-268) flip off because the day is almost always `< NUM_DAYS`; if the user happens to click the latest column it flips on, which is the right intent.

### Tests First
No JS test infrastructure. Manual verification:
- Open a live tournament with rounds resolved. On load, `currentDay === NUM_DAYS` (latest day visible).
- Trigger a new round. After the SSE reload, still on the latest day (no manual click needed).
- Scrub backwards. After the next SSE reload, stay on the scrubbed day (no jump-forward).
- Click ⏭ Last. `currentDay === NUM_DAYS`; the next SSE reload keeps it at the new latest.
- Open `/ui/matches/{match_id}` (replay mode). No follow-live behavior; `localStorage` day restoration works as today.

### Acceptance Criteria
- [ ] Live mode + first load with rounds resolved → `currentDay === NUM_DAYS`.
- [ ] Live mode + SSE reload after `round_ended` → still on `NUM_DAYS` (auto-advances).
- [ ] Live mode + user scrubs to day N where `N < NUM_DAYS` → next SSE reload keeps `currentDay === N`.
- [ ] Live mode + user clicks ⏭ Last → `currentDay === NUM_DAYS`; next SSE reload still on `NUM_DAYS`.
- [ ] Replay mode (`window.__ATP_LIVE_URL__` unset) → `localStorage` restore behavior unchanged; `sessionStorage` flag never written.
- [ ] Empty-state (Plan 1, `DATA=[]`) → flag respected but no-op; `currentDay === 0`.

---

## Phase 2: Resume-live pill UI

**Goal:** when follow-live is **off** in live mode, show a clickable pill the user can click to resume. When **on**, hide it. The existing `atp-live-pill` (showing "Live") stays in place — the resume pill is a sibling.

**Self-contained:** Phase 1 already gives correct behavior on click of ⏭ Last. This phase adds a less-discoverable affordance for users who don't know about the Last button.

### Files
- [ ] `packages/atp-dashboard/atp/dashboard/v2/templates/ui/match_detail.html` — MODIFIED (add sibling pill markup near line 48)
- [ ] `packages/atp-dashboard/atp/dashboard/v2/static/js/el_farol/live_subscriber.js` — MODIFIED (manage pill visibility + click)
- [ ] `packages/atp-dashboard/atp/dashboard/v2/static/css/el_farol.css` — MODIFIED (add `.atp-resume-pill` rule)

### Changes

#### `match_detail.html` — add resume-pill markup

In the `{% else %}` block, after the existing `atp-live-pill` (currently lines 47-49):

```html
{% if live_stream_url %}
  <span id="atp-live-pill" class="atp-live-pill live" title="Streaming live tournament updates">Live</span>
  <button id="atp-resume-pill" class="atp-resume-pill" type="button" hidden
          title="Resume following the latest round">▶ Resume live</button>
{% endif %}
```

The `hidden` attribute is the default state; `live_subscriber.js` reveals it when follow-live is off.

#### `live_subscriber.js` — manage pill visibility + click

Add to the IIFE (after the existing `setStatus` helper):

```js
const resumeBtn = document.getElementById('atp-resume-pill');

function syncResumePill() {
  if (!resumeBtn) return;
  const following = typeof window.atpIsFollowingLive === 'function'
    && window.atpIsFollowingLive();
  resumeBtn.hidden = following;
}

if (resumeBtn) {
  resumeBtn.addEventListener('click', () => {
    if (typeof window.atpSetFollowingLive === 'function') {
      window.atpSetFollowingLive(true);
    }
    if (typeof window.jump === 'function' && typeof window.ATP === 'object'
        && window.ATP && window.ATP.NUM_DAYS > 0) {
      window.jump(window.ATP.NUM_DAYS);
    }
    syncResumePill();
  });
}

// Listen for the custom event dashboard.js dispatches when the flag changes
// (e.g. user scrubs backwards → flag flips to false → reveal the pill).
window.addEventListener('atp:follow-live-changed', syncResumePill);

// Initial sync, after dashboard.js has loaded and possibly written the flag.
syncResumePill();
```

Place the resume-pill code path **after** the existing `EventSource` setup so the IIFE's early `return` (when `__ATP_LIVE_URL__` is unset) keeps replay mode untouched.

#### `el_farol.css` — add the new pill style

Append after the existing `.atp-live-pill` block (line 262):

```css
.atp-resume-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 10px;
  margin-left: 6px;
  border-radius: 10px;
  border: 1px solid rgba(96, 165, 250, 0.5);
  background: rgba(96, 165, 250, 0.18);
  color: #60a5fa;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  cursor: pointer;
  vertical-align: middle;
  font-family: inherit;
}
.atp-resume-pill:hover {
  background: rgba(96, 165, 250, 0.32);
}
.atp-resume-pill[hidden] {
  display: none;
}
```

The `[hidden]` rule is defensive — Pico CSS doesn't reset it for `inline-flex` elements.

### Tests First (test_*.py — `unit-tester` subagent owns these)

Optional, low-value: extend the existing `test_live_dashboard_ui.py` (added by Plan 1) with one assertion that the rendered HTML for a live tournament contains the `id="atp-resume-pill"` element. The dynamic visibility is JS-only and not testable without a JS runner.

```python
def test_live_ui_renders_resume_pill_markup(...):
    # Seed a live tournament; GET /ui/tournaments/{id}/live
    # Assert: 'id="atp-resume-pill"' in response.text
    # Assert: 'hidden' attribute on the resume pill (initial state)
```

Skip if Plan 1's `test_live_ui_zero_rounds_renders_skeleton` already does an HTML-snapshot grep — extending it with one more `assert` line is enough.

### Acceptance Criteria
- [ ] In live mode + follow-live ON → resume pill is hidden.
- [ ] In live mode + follow-live OFF → resume pill is visible in the topbar next to the existing "Live" pill.
- [ ] Clicking the resume pill: flag flips to ON, `currentDay = NUM_DAYS`, dashboard re-renders, pill hides.
- [ ] In replay mode → resume pill is not rendered at all (template `{% if live_stream_url %}` gate).
- [ ] CSS: pill matches the visual weight of the existing `atp-live-pill` (small, uppercase, badge-style); blue accent so it reads as an action, not a status.

---

## Risks & Edge Cases

- **`window.__ATP_MATCH__.match_id` for live mode is `tournament-{id}` (string).** The sessionStorage key derived from it is unique per tournament. ✅
- **User opens two tabs on the same tournament.** Each tab has its own `sessionStorage`; behavior is independent. Acceptable.
- **`NUM_DAYS = 0` (Plan 1 empty state).** `isFollowingLive()` returns true but the override block guards on `NUM_DAYS > 0`, so `currentDay` stays at `0`. Resume pill stays hidden because there's nothing to resume to. After the first round resolves and the SSE reload fires, `NUM_DAYS = 1` and follow-live snaps to it as expected.
- **User has `dashboard.js` load after `live_subscriber.js` (script order change in template).** The `syncResumePill()` initial call on `live_subscriber.js` won't see `window.atpIsFollowingLive` yet. Mitigation: the `typeof === 'function'` guards already handle this, and the `atp:follow-live-changed` event will fire on the first user nav. Best-effort defense; the current template (`match_detail.html:158-159`) loads `dashboard.js` first.
- **Tournament completes while user is on day < NUM_DAYS.** `tournament_completed` triggers an SSE reload. With follow-live OFF, the user stays on their scrubbed day (correct — they were inspecting history). The terminal `live_subscriber.js` redirect to `/ui/matches/{match_id}` (existing behavior, line ~50-60) takes over — replay mode does not honor `sessionStorage` follow-live, so the user lands on whatever `localStorage` says, which is fine.
- **`localStorage` restore conflict with follow-live override.** Phase 1's override block runs *after* the localStorage restore, so the override always wins when follow-live is on. When follow-live is off, the localStorage restore runs unmodified — i.e., the existing replay-mode behavior, which is what we want.
- **Custom event compatibility.** `CustomEvent` is supported in all browsers we target. No polyfill needed.
- **Concern 2 partial fix only.** The deeper question ("`||'1'` falls back to day 1 in localStorage when key absent") is left untouched intentionally — for **replay mode** that's actually the right default for old finished matches. Live mode bypasses it via the override block.

## Out of Scope
- In-place `dashboard.js` updates (replacing `window.ATP` and re-rendering without reload).
- Per-user persistence of follow-live across tabs / devices.
- "Auto-resume after N seconds idle" or other implicit re-enable.
- A `localStorage` migration for users who previously stored a stale day.
- Concern 1 (empty skeleton) — covered by `el-farol-live-dashboard-empty-skeleton.md`.
- Backend changes — none.
