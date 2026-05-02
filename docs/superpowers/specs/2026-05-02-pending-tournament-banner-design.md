# Pending Tournament Banner — Design

**Status:** Draft (revision 3 — addresses follow-up reviews 2026-05-02)
**Date:** 2026-05-02
**Owner:** prosto.andrey.g@gmail.com

## Summary

Add a small live-updating banner to the top of two tournament pages
during the **pending** registration phase:

- `/ui/tournaments/{id}` (the tournament detail page)
- `/ui/tournaments/{id}/live` (the El Farol live cards dashboard)

The banner shows two pieces of information:

- **Registration progress** — `Registered: N / num_players`
- **Time remaining** — countdown to `pending_deadline`, ticking once per
  second on the client

The banner exists only while the tournament is in `pending` status. As
soon as the tournament transitions to `active` (either because the full
roster joined or the deadline triggered the shrink-on-deadline
behaviour added by PR #117), the banner disappears.

## Goals

- Give an admin watching a 100-bot tournament real-time visibility into
  how many bots have arrived and how long until registration closes.
- Update without page reloads.
- Keep the change strictly additive — no logic changes to the tournament
  lifecycle.

## Non-goals

- No urgency cues (color coding, animations) — plain text in v1.
- No new SSE channel — re-use existing 10 s HTMX polling for the counter
  and a 1 s client-side JS interval for the countdown.
- No state for `active` / `completed` / `cancelled` tournaments — banner
  is gated to `pending` only.
- No multi-tenant. Single-tenant (`DEFAULT_TENANT_ID`) only, matching
  the rest of v1 dashboard surfaces.
- No new logic for cancellation. The shrink-on-deadline path keeps the
  tournament moving to `active`; we only display data, never decide.

## Surfaces

| Surface | URL | Method | Response |
|---|---|---|---|
| Detail page (full) | `/ui/tournaments/{id}` | GET | Existing template + new banner include |
| Live page (full) | `/ui/tournaments/{id}/live` | GET | Existing template + new banner include |
| Banner partial | `/ui/tournaments/{id}?partial=pending-banner` | GET | New `pending_banner_wrapper.html` |

The partial endpoint reuses the existing `partial=` query-param pattern
already used by `?partial=live` on the detail route. Both pages embed
the same banner wrapper, so they both poll the same partial URL.

### Visibility

The banner inherits whatever visibility the partial endpoint enforces,
which is the **detail route's** strict 404 — the partial branch is a
sub-route of `ui_tournament_detail`, not `ui_tournament_live`.

Important nuance: the live route (`ui_tournament_live`) does NOT raise
HTTP 404 for invisible tournaments; it renders `match_detail.html` with
`not_found=True` as an in-page placeholder (see `routes/ui.py:822`).
That asymmetry doesn't hurt us because:

- The HTMX poll target is `/ui/tournaments/{id}?partial=pending-banner`
  (the detail route), which DOES 404 invisible tournaments — pollers
  from the live page never leak data through that channel.
- The host live page only renders the wrapper when
  `pending_banner_show=True`. For an invisible tournament the live
  route hits the `not_found=True` placeholder branch BEFORE calling the
  helper, so `pending_banner_show` is never set and the include is
  skipped (the conditional `{% if pending_banner_show is defined %}`
  guards this).

Net effect for a private tournament + non-allowed viewer: detail page
404, live page placeholder, banner absent on both.

## Data model touchpoints

No schema migrations.

- `Tournament.status == "pending"` — gate condition.
- `Tournament.tenant_id == DEFAULT_TENANT_ID` — defensive parity with
  v1 surfaces.
- `Tournament.pending_deadline` — `Mapped[datetime]`, `nullable=False`,
  `server_default=now()`. Already populated for every tournament.
- `Tournament.num_players` — planned roster size.
- `Tournament.participants` — registered participants. The counter
  excludes those whose `released_at IS NOT NULL` (released = left or
  shrunk out).

## Components

### New files

```
packages/atp-dashboard/atp/dashboard/v2/
├── templates/ui/partials/
│   ├── pending_banner.html              # banner content (counter + countdown span)
│   └── pending_banner_wrapper.html      # outer div with hx-* attrs + include
└── static/v2/js/
    └── pending_banner.js                # 1 s JS-tick countdown formatter

tests/unit/dashboard/
└── test_pending_banner_context.py       # _pending_banner_context helper

tests/integration/dashboard/
└── test_pending_banner.py               # render, gate, counter, partial endpoint
```

### Modified files

```
packages/atp-dashboard/atp/dashboard/v2/
├── routes/ui.py                         # context helper + partial branch
└── templates/ui/
    ├── tournament_detail.html           # +include wrapper before <h2>
    ├── match_detail.html                # +include wrapper before page heading
    └── base_ui.html                     # +<script src=".../pending_banner.js" defer>
```

### Helper shape (`routes/ui.py`)

```python
from datetime import UTC

def _pending_banner_context(tournament: Tournament) -> dict[str, Any]:
    """Return template context fragments for the pending banner.

    Returns ``{"pending_banner_show": False}`` when the banner is not
    applicable; otherwise returns the full set of four keys.

    NB: ``Tournament.pending_deadline`` is declared as
    ``Mapped[datetime]`` with no ``timezone=True`` flag, so the value
    comes back tz-naive. We must coerce to UTC before
    ``isoformat()`` — otherwise the ISO string lacks a ``Z``/offset
    suffix and the browser's ``new Date(...)`` will parse it in the
    user's local timezone, producing a countdown that is wrong by the
    user-vs-server clock skew. The same pattern is already used at
    ``routes/ui.py:877-887`` for ``starts_at``.
    """
    if (
        tournament.status != TournamentStatus.PENDING
        or tournament.tenant_id != DEFAULT_TENANT_ID
    ):
        return {"pending_banner_show": False}
    deadline = tournament.pending_deadline
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    registered = sum(
        1 for p in tournament.participants if p.released_at is None
    )
    return {
        "pending_banner_show": True,
        "pending_deadline_iso": deadline.isoformat(),
        "pending_registered_count": registered,
        "pending_planned_count": tournament.num_players,
    }
```

The helper is called from both `ui_tournament_detail` and
`ui_tournament_live` after the visibility gate has admitted the viewer.
Spread into the existing context dict via `**_pending_banner_context(t)`.

**Hard prerequisite — eager load in `ui_tournament_live`:** the existing
`session.get(Tournament, ...)` call at `routes/ui.py:804` does NOT load
participants. The helper accesses `tournament.participants`, which in
async SQLAlchemy raises `MissingGreenlet` (HTTP 500), not a silent zero.
Before this change ships, `ui_tournament_live` MUST be converted to:

```python
tournament = (
    await session.execute(
        select(Tournament)
        .options(selectinload(Tournament.participants))
        .where(Tournament.id == tournament_id)
    )
).scalar_one_or_none()
```

`ui_tournament_detail` already eager-loads participants for the
scoreboard (see `routes/ui.py:1232`); no change needed there.

### `pending_banner_wrapper.html`

```jinja
<div id="pending-banner"
     {% if pending_banner_show %}
     hx-get="/ui/tournaments/{{ tournament_id }}?partial=pending-banner"
     hx-trigger="every 10s"
     hx-swap="outerHTML"
     {% endif %}
>
  {% if pending_banner_show %}
    {% include "ui/partials/pending_banner.html" %}
  {% endif %}
</div>
```

When `pending_banner_show=False` the wrapper has no `hx-trigger`, so
HTMX polling stops naturally — no need for explicit cancellation.

### `pending_banner.html`

```jinja
<div class="pending-banner-row" style="
    background:#fffbe6;
    border:1px solid #ffe58f;
    border-radius:4px;
    padding:0.5rem 1rem;
    margin-bottom:1rem;
    display:flex;
    gap:1.5rem;
    align-items:center;
">
  <span class="pending-banner-count">
    Registered: <strong>{{ pending_registered_count }}</strong>
    / {{ pending_planned_count }}
  </span>
  <span class="js-countdown"
        data-deadline-iso="{{ pending_deadline_iso }}">—:—</span>
</div>
```

### `pending_banner.js`

```js
(function () {
  // ONE global interval polls the DOM every second and updates every
  // ``.js-countdown`` element it finds. This avoids the per-element
  // interval-leak pitfall: with ``hx-swap="outerHTML"`` the old span
  // detaches from the document, but a per-element ``setInterval``
  // keeps firing on the detached node — leaking +1 timer every 10 s
  // for the duration of the pending phase. A single global interval
  // is unaffected by swaps; ``querySelectorAll`` simply returns the
  // current set of elements after each swap.
  function tickAll() {
    const els = document.querySelectorAll(
      ".js-countdown[data-deadline-iso]"
    );
    for (const el of els) {
      const deadlineMs = new Date(el.dataset.deadlineIso).getTime();
      const remainingMs = Math.max(0, deadlineMs - Date.now());
      const totalSec = Math.floor(remainingMs / 1000);
      const h = Math.floor(totalSec / 3600);
      const m = Math.floor((totalSec % 3600) / 60);
      const s = totalSec % 60;
      // Multi-hour deadlines render as ``Hh Mm Ss``; sub-hour as
      // ``M:SS`` so a 5-minute window stays visually compact.
      el.textContent = h > 0
        ? h + "h " + m + "m " + s + "s"
        : m + ":" + String(s).padStart(2, "0");
    }
  }
  // Single timer, started once on page load. No per-element timers,
  // no rebinding on htmx:afterSwap, no chance of duplication.
  document.addEventListener("DOMContentLoaded", () => {
    tickAll();
    setInterval(tickAll, 1000);
  });
})();
```

The IIFE keeps helpers off the global namespace. The single global
interval handles all `.js-countdown` elements currently in the DOM —
HTMX swaps replace elements in place, the interval keeps ticking and
naturally picks up the new ones.

### Route handler change (`routes/ui.py`)

Inside `ui_tournament_detail`, after the existing `partial == "live"`
branch:

```python
if partial == "pending-banner":
    # Wrapper template needs only the path id, not the full ORM row —
    # see "Wrapper context contract" note below.
    ctx = {"tournament_id": t.id, **_pending_banner_context(t)}
    response = templates.TemplateResponse(
        request=request,
        name="ui/partials/pending_banner_wrapper.html",
        context=ctx,
    )
    response.headers["Cache-Control"] = "no-store"
    return response
```

**Wrapper context contract:** the wrapper template references
`tournament_id` (not `tournament.id`) deliberately — it is the minimal
input the wrapper needs (path id for HTMX URL construction). Both host
routes must put `tournament_id` in their context:

- `ui_tournament_live` already has the path arg `tournament_id` in
  scope (and currently in context as `"tournament_id": tournament_id`).
- `ui_tournament_detail` must add the same key explicitly:
  `"tournament_id": tournament_id`.

The narrow contract means the partial endpoint can be served without
loading the full Tournament row into the template context.

In both `ui_tournament_detail` and `ui_tournament_live` full-page
branches, spread `**_pending_banner_context(t)` into the existing
context dict. The `match_detail.html` template is also rendered by
`/ui/matches/{match_id}` (the replay route at `routes/ui.py:652`),
which has no tournament context — for that caller the spread does NOT
happen, so the template must conditionally include the wrapper:

```jinja
{% if pending_banner_show is defined %}
  {% include "ui/partials/pending_banner_wrapper.html" %}
{% endif %}
```

`tournament_detail.html` is only rendered by `ui_tournament_detail`
which always supplies `pending_banner_show`; the conditional could be
omitted there but adding it costs nothing and keeps both includes
identical for grepability.

Add the include immediately before each page's existing `<h2>`.

In `base_ui.html`, add the script tag in `<head>`:

```html
<script src="/static/v2/js/pending_banner.js" defer></script>
```

## Data flow

```
Initial page GET (pending tournament)
  ├─ visibility gate
  ├─ load Tournament with participants (already done for scoreboard)
  ├─ helper computes pending_banner_show=True + 3 fields
  ├─ template renders wrapper (with hx-trigger) + banner body
  └─ browser parses HTML, JS pending_banner.js ticks countdown every 1 s

Every 10 s while pending:
  ├─ HTMX fires GET /ui/tournaments/{id}?partial=pending-banner
  ├─ route returns refreshed wrapper + body (counter updated)
  ├─ HTMX swaps outerHTML — wrapper element replaced
  └─ Single global JS interval picks up the new <span> on the next
     1 s tick via querySelectorAll — no rebinding logic needed

When status flips to active (between pulls):
  ├─ next pull returns wrapper with NO hx-trigger, empty body
  ├─ HTMX swap replaces the element — polling stops
  └─ JS finds no .js-countdown in the new element — no new interval

When deadline passes but status still pending (race window ≤10 s):
  └─ JS clamps remainingMs to 0 → countdown shows "0:00"
     (next pull will see status flipped and remove the banner)
```

## Edge cases & error handling

| Case | Behaviour |
|---|---|
| Status flips to `active` between pulls | Next pull returns empty wrapper, polling stops, banner gone |
| Status flips to `cancelled` (rare under shrink) | Same: empty wrapper, polling stops |
| Deadline expired, status still pending | JS clamps to 0:00; counter still updates; next pull removes banner |
| `Tournament.pending_deadline` is tz-naive | Helper coerces to UTC before `isoformat()` — see helper code above. Without this, JS parses the ISO as the user's local time and the countdown is wrong by the clock skew. |
| `Tournament.pending_deadline IS NULL` | Schema makes this impossible (`Mapped[datetime]`, `nullable=False`, `server_default=func.now()`). The helper does NOT add a defensive `is not None` guard — that would be dead code per pyrefly and would mask a future schema break that should fail loudly instead. |
| Viewer's clock is wrong | Countdown is off by the clock skew; counter is server-authoritative; eventually the next pull replaces wrapper with empty when status flips |
| Tournament not found at partial endpoint | 404 (same as `?partial=live`) |
| Non-default tenant | `pending_banner_show=False`, empty wrapper rendered |
| Lazy-load on `tournament.participants` | Hard prerequisite — see "Hard prerequisite" note next to the helper. Both route handlers MUST eager-load participants. The fix for `ui_tournament_live` is non-optional: without `selectinload(Tournament.participants)` the helper raises `MissingGreenlet` (HTTP 500). No fallback is added in the helper because that would mask deploy bugs. |
| HTMX swap | Single global interval queries the DOM each tick — no per-element timers, no leak. Old detached spans are garbage-collected normally. |
| Multiple viewers polling simultaneously | Each viewer hits the route independently. Counter is `len(participants)` — cheap. No new caching needed. |
| Private tournament + non-allowed viewer | Parent page 404 (detail route) → partial endpoint also 404, since the partial branch is on the detail route which runs the visibility gate first. The live route renders an in-page placeholder (`not_found=True` flag) for invisible tournaments rather than raising 404, but the partial endpoint lives on the detail route and inherits the detail route's strict 404. The host live page therefore never even reaches the banner-rendering branch when the tournament is invisible — `pending_banner_show=False` triggers the empty-wrapper case. |
| Multi-hour pending phase (>60 min) | Countdown formats as `Hh Mm Ss` instead of overflowing minutes (e.g. `2h 5m 30s`, not `125:30`). Sub-hour stays compact `M:SS`. |

## Performance

- Counter is `O(participants)` in Python (no DB hit beyond what scoreboard already loads).
- 10 s HTMX poll × few admin viewers × pending phase (typically minutes) = negligible load.
- No caching needed in v1. The partial endpoint emits
  `Cache-Control: no-store` to prevent browser back/forward cache from
  serving a stale counter.
- JS interval is exactly one global `setInterval` per page, started
  once on `DOMContentLoaded` and never cleared. HTMX swaps don't
  affect it.

## Security

- No auth changes; banner inherits parent page's visibility gate.
- `pending_deadline_iso` is a server-generated ISO8601 string — no XSS surface.
- Jinja autoescape handles all template variables.
- Rate limit (120/min) inherited from existing UI routes via `@limiter.limit`.

## Testing

### Unit (`tests/unit/dashboard/test_pending_banner_context.py`)

- `pending` + DEFAULT_TENANT_ID → returns all 4 fields; counter equals number of un-released participants.
- `active` / `completed` / `cancelled` → returns `{"pending_banner_show": False}` only.
- Non-default tenant → `{"pending_banner_show": False}`.
- Released participants (released_at set) excluded from counter.
- **Timezone:** when `Tournament.pending_deadline` is tz-naive (the
  default per the schema), the returned `pending_deadline_iso` ends in
  `+00:00`. When already tz-aware, it is returned unchanged.

### Integration (`tests/integration/dashboard/test_pending_banner.py`)

**Status gate:**

- Pending tournament: parent page contains `data-deadline-iso` and
  `Registered: N`. Also assert the ISO string has a UTC marker —
  `assert "+00:00" in r.text or "Z" in r.text` — otherwise the
  timezone fix has regressed.
- Active / completed / cancelled: parent page contains the wrapper
  `<div id="pending-banner">` but with no `hx-trigger` and no body.
- 404 on `?partial=pending-banner` for non-existent tournament.

**Status flip between pulls (race window):**

- Seed a pending tournament. Fetch `?partial=pending-banner`, assert
  `hx-trigger` present.
- Flip the tournament to `active`. Fetch `?partial=pending-banner`
  again, assert response body has the wrapper element with NO
  `hx-trigger` and an empty interior. This is the swap that gracefully
  retires the banner.

**Non-default tenant — parent page integration:**

- Render `/ui/tournaments/{id}` for a tournament with
  `tenant_id != DEFAULT_TENANT_ID`. Assert the wrapper renders empty
  (no `hx-trigger`, no body). The unit-level helper test covers the
  helper's return value; this integration test covers the wrapper's
  template-level conditional and ensures the tenant filter is wired
  correctly all the way through.

**Wrapper URL parity (regression for v3 fix):**

- For a pending tournament, fetch both `/ui/tournaments/{id}` and
  `/ui/tournaments/{id}/live`. Assert each contains
  `hx-get="/ui/tournaments/{id}?partial=pending-banner"` literally —
  not `/ui/tournaments/?partial=...` (the bug shape if `tournament_id`
  is missing from the live route's context).

**Counter accuracy:**

- 0 participants → `Registered: 0 / num_players`.
- N participants, M released → `Registered: (N-M) / num_players`.
- Add a participant, re-fetch partial → counter incremented.

**Partial endpoint:**

- Pending: response contains `hx-trigger="every 10s"` on wrapper.
- Active: response contains the wrapper but no `hx-trigger`.

**Render parity:**

- Detail page and live page both contain wrapper + identical
  `hx-get="/ui/tournaments/{id}?partial=pending-banner"`.

**Visibility:**

- Private tournament + non-member viewer → parent 404, partial endpoint
  404.

**Cache headers:**

- The partial endpoint emits exactly `Cache-Control: no-store`. Assert
  the value, not just presence.

### JS-tick

No JS unit tests in v1 (the project has no JS test stack). Integration
tests verify the contract (`data-deadline-iso` + `class="js-countdown"`
present on the right element). Manual smoke on staging confirms the
seconds tick.

### Coverage target

≥80 % on the new Python helper and partial branch.

## Migration plan

None. No DB changes, no schema bumps. Single deploy.

## Open questions

- **Color cues** — out of scope for v1 by user choice. Easy to add later
  with a CSS class toggled on `remainingMs < 60_000`.
- **SSE push for counter** — not in v1. If admins want sub-10-s counter
  updates, replace the HTMX poll with an SSE subscriber on the existing
  tournament event bus.
- **/live page for pending tournaments** — currently the El Farol cards
  UI shows an empty grid for pending tournaments. The banner improves
  this without changing the cards UI itself; a follow-up could add a
  proper "Waiting for round 1" placeholder.
- **Inline styles in `pending_banner.html`** — colors and layout are
  inline for v1. A follow-up can extract a `.pending-banner` CSS class
  to `static/v2/css/ui.css` once the visual is locked in.

## References

- `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py:766` (`ui_tournament_live`)
- `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html`
  (existing `?partial=live` HTMX poll on `#live-content`)
- `packages/atp-dashboard/atp/dashboard/tournament/models.py:97-120`
  (`num_players`, `pending_deadline` columns)
- `docs/superpowers/specs/2026-05-01-tournament-shrink-design.md` (the
  shrink-on-deadline behaviour that replaces auto-cancel)
