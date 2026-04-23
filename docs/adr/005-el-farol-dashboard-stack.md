# ADR-005: El Farol Dashboard — Rendering Stack

**Status**: Accepted
**Date**: 2026-04-23
**Context**: PR #63 (merged 2026-04-23) landed the El Farol data model
(Phases 1–7 of `docs/plans/el-farol-dashboard-data-model.md`). Two
standalone HTML mockups checked in under
`docs/mockups/el-farol/cards.html` and
`docs/mockups/el-farol/scaled.html` (LABS-97 / PR #65) describe the
target UX. They proved sufficient fidelity of the schema but are
packaged as a self-contained SPA (vanilla DOM + SVG, ~22 k chars
dashboard code + ~9 k chars data/helpers + inlined fonts/assets). The
rest of `/ui/*` is Jinja2 templates extending `base_ui.html`, Pico CSS
(light theme), HTMX 2.0.4, no client JS framework. LABS-98 asks: how do
we integrate the mockup into the running dashboard?

## Decision

**Hybrid — Jinja shell + scoped JS+SVG dashboard module.** Render a thin
Jinja page that extends `base_ui.html` (so sidebar, auth, navigation are
reused) and embeds a root `<div id="atp-el-farol">` plus a dedicated JS
bundle served from `/static/v2/js/el_farol/`. The bundle contains the
mockup's rendering logic, adapted to read from `window.__ATP_MATCH__`
(server-injected JSON) instead of the mockup's `generateData()`.

Reject both pure alternatives:

- **Pure SPA-subroute** — serve the standalone HTML as-is under
  `/ui/matches/{match_id}`. Cheapest, but loses sidebar, loses
  consistent auth UX, duplicates navigation, and forces a theme break
  (the mockup's dark theme vs the rest of `/ui/`'s light Pico).
- **Pure HTMX rewrite** — decompose scrubber, heatmap, sparklines,
  pin-overlay into Jinja fragments driven by HTMX partials. Consistent,
  but the mockup's interactions are genuinely client-side (hover-to-
  preview on cards, drag-scrubber at 60 fps, 3-mode heatmap retoggle,
  localStorage persistence). HTMX round-trips would kill the polish.

The hybrid keeps the page shell in the existing stack and treats the
dashboard widget itself as a single client-side module. Visually, we
scope the mockup's dark palette to `.atp-dashboard-dark` so Pico's
light theme remains default outside.

## Specifics

### Route and page

- New route `GET /ui/matches/{match_id}` in `routes/ui.py` (not
  `/ui/games/{game_name}/…` — that path is already the public per-game
  rules page).
- Jinja template `templates/ui/match_detail.html` extends `base_ui.html`.
- The template fetches the `GameResult` row server-side, serialises the
  `window.__ATP_MATCH__` payload inline as `<script>` JSON (no CORS, no
  extra round-trip for the initial render), and renders a single
  `<div id="atp-el-farol" class="atp-dashboard-dark">`.
- The page pulls `/static/v2/js/el_farol/data_helpers.js` and
  `.../dashboard.js` via two `<script>` tags at the end of `<body>`.
- `active_page` stays on whichever parent is appropriate
  (`"tournaments"` or `"runs"` depending on entry path).

### JS bundle

- Two files under `packages/atp-dashboard/atp/dashboard/v2/static/js/el_farol/`:
  1. `data_helpers.js` — `cumPayoff`, `cumSeries`, `leaderboard`,
     `nonEmptyIntervals`, `slotsInIntervals`. Derived from the mockup's
     9 k-char file, minus `generateData()` and its seeded RNG.
  2. `dashboard.js` — renderers + event wiring. Derived from the
     mockup's 22 k-char file unchanged, entry point becomes a function
     `initDashboard(root, atp)` instead of module-level `renderAll()`.
- The bootstrap at `<script>` tag bottom-of-body:
  ```js
  const root = document.getElementById('atp-el-farol');
  window.ATP = shimFromPayload(window.__ATP_MATCH__);
  initDashboard(root, window.ATP);
  ```
- No build step, no bundler — these are plain ES scripts loaded in
  order. Matches the rest of the dashboard's zero-toolchain approach.

### Style

- The mockup's inline `<style>` block extracts into
  `packages/atp-dashboard/atp/dashboard/v2/static/css/el_farol.css` and
  is referenced from the page as `/static/v2/css/el_farol.css`
  (matching the `/static/v2` mount in `v2/factory.py`), scoped under
  `.atp-dashboard-dark`. Fonts (Inter, JetBrains Mono) load from Google
  Fonts CDN the way the mockup does, scoped so they don't leak to Pico.
- No CSS framework collision because `.atp-dashboard-dark` isolates the
  tree.

### Fetch adapter boundary (LABS-99)

- The server-side endpoint
  `GET /api/v1/games/{match_id}/dashboard` returns a Pydantic model
  shaped as `window.ATP` expects. The Jinja template calls the same
  endpoint internally during render so SSR and client share one schema.
- A follow-up subroute could stream updates via HTMX `hx-swap` on a
  partial if we ever render match detail in real time. Not in scope
  for Cards phase.

### Scope boundary

- Phase 1 ships the Cards variant only (8–16 agents).
- The Scaled variant (LABS-TBD) reuses the same bundle loading pattern
  with a different top-level JS module. Keeping both under
  `/static/v2/js/el_farol/` means the second variant is a new file, not
  a rewrite.

## Consequences

### Wins

- Sidebar, auth, navigation, ownership checks come free from
  `base_ui.html`.
- Mockup-to-prod fidelity stays high because the interactive guts are
  essentially the mockup's own code.
- Zero new runtime dependencies — no Alpine, no React, no bundler. Two
  static JS files.
- Scaled variant's future integration is a drop-in second bundle.

### Costs

- Two styling contexts coexist (Pico light + scoped dark). Requires
  discipline: anything outside `.atp-dashboard-dark` uses Pico; anything
  inside never pulls Pico selectors. CSS linter not planned —
  spot-check in code review.
- The JS is hand-rolled. Debugging is easier than React but harder than
  HTMX partials. Accept the cost because the alternative is a
  multi-week rewrite.
- Server-injected JSON puts the entire match (100 days × 8–16 agents ≈
  30–80 KB) in the initial HTML. Fine for Cards. For Scaled (64 agents
  × 100 days ≈ 500 KB) revisit: switch to an XHR fetch after first
  paint, or server-side page the data.

## Risks for paths we didn't pick

**Pure SPA-subroute (rejected):**

1. Theme split. Dark El Farol page feels like a different product from
   the rest of `/ui/`. Users mentally context-switch.
2. Auth drift. The standalone HTML has no concept of our login
   redirect, CSRF header, or ownership checks; bolting them on
   duplicates logic already in `base_ui.html` + middleware.
3. Navigation dead-end. No sidebar from inside the SPA — user has to
   use the back button to leave, losing place in tournaments or runs
   listings.

**Pure HTMX rewrite (rejected):**

1. Scrubber regression. The mockup's drag-scrub at 60 fps re-renders
   KPIs, heatmap playhead, and cards synchronously. HTMX round-trips
   per scrub event (even debounced) will judder visibly.
2. Heatmap complexity. 16 slots × 100 days × 3 modes = ~48 000
   conceptual cells. HTMX's pattern is "swap HTML from server";
   re-streaming that on every mode toggle is expensive, caching it per
   mode on the client defeats the HTMX simplicity win.
3. Pin-overlay compare. Multi-agent overlay line chart needs live
   x/y-range recomputation when a pin is added. Done on client in
   <10 ms; via HTMX it means either server holds per-user pin state
   (new table) or the server always computes all 4! overlays (wasteful).

## References

- `docs/plans/el-farol-dashboard-data-model.md` — data-model plan (PR #63)
- `docs/mockups/el-farol/cards.html` — Cards mockup (LABS-97 / PR #65)
- `docs/mockups/el-farol/scaled.html` — Scaled mockup (phase 2)
- LABS-97 — Cards dashboard epic
- LABS-99 — fetch adapter sub-issue
- LABS-100 — `floor`/`round` alignment sub-issue
- LABS-101 — drawer Tier-2 fields verification sub-issue
- LABS-102 — ship Cards sub-issue
