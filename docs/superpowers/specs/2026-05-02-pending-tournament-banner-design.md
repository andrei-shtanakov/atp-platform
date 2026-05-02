# Pending Tournament Banner — Design

**Status:** Draft
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

The banner inherits the parent page's visibility rule. Private
tournaments (`join_token IS NOT NULL`) are visible only to admin /
creator / participants — same as today; if the parent page 404s for a
viewer, the partial endpoint also 404s (route order: visibility check
first, then `partial` branch).

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
def _pending_banner_context(tournament: Tournament) -> dict[str, Any]:
    """Return template context fragments for the pending banner.

    Returns ``{"pending_banner_show": False}`` when the banner is not
    applicable; otherwise returns the full set of four keys.
    """
    show = (
        tournament.status == TournamentStatus.PENDING
        and tournament.tenant_id == DEFAULT_TENANT_ID
    )
    if not show:
        return {"pending_banner_show": False}
    registered = sum(
        1 for p in tournament.participants if p.released_at is None
    )
    return {
        "pending_banner_show": True,
        "pending_deadline_iso": tournament.pending_deadline.isoformat(),
        "pending_registered_count": registered,
        "pending_planned_count": tournament.num_players,
    }
```

The helper is called from both `ui_tournament_detail` and
`ui_tournament_live` after the visibility gate has admitted the viewer.
Spread into the existing context dict via `**_pending_banner_context(t)`.

### `pending_banner_wrapper.html`

```jinja
<div id="pending-banner"
     {% if pending_banner_show %}
     hx-get="/ui/tournaments/{{ tournament.id }}?partial=pending-banner"
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
  function tickAll(root) {
    const els = (root || document).querySelectorAll(
      ".js-countdown[data-deadline-iso]"
    );
    for (const el of els) {
      if (el._intervalId) {
        clearInterval(el._intervalId);
      }
      const deadline = new Date(el.dataset.deadlineIso).getTime();
      function tick() {
        const remainingMs = Math.max(0, deadline - Date.now());
        const totalSec = Math.floor(remainingMs / 1000);
        const m = Math.floor(totalSec / 60);
        const s = totalSec % 60;
        el.textContent = m + ":" + String(s).padStart(2, "0");
      }
      tick();
      el._intervalId = setInterval(tick, 1000);
    }
  }
  document.addEventListener("DOMContentLoaded", () => tickAll());
  document.addEventListener("htmx:afterSwap", (e) => tickAll(e.target));
})();
```

The IIFE keeps helpers off the global namespace. Re-init on
`htmx:afterSwap` ensures a swapped wrapper element gets a new interval;
the `clearInterval` guard prevents duplicate timers piling up.

### Route handler change (`routes/ui.py`)

Inside `ui_tournament_detail`, after the existing `partial == "live"`
branch:

```python
if partial == "pending-banner":
    ctx = {"tournament": t, **_pending_banner_context(t)}
    return templates.TemplateResponse(
        request=request,
        name="ui/partials/pending_banner_wrapper.html",
        context=ctx,
    )
```

In both `ui_tournament_detail` and `ui_tournament_live` full-page
branches, spread `**_pending_banner_context(t)` into the existing
context dict.

In both host templates (`tournament_detail.html`, `match_detail.html`),
add `{% include "ui/partials/pending_banner_wrapper.html" %}` immediately
before the existing page `<h2>`.

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
  └─ htmx:afterSwap → JS re-binds countdown to the new <span>

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
| `pending_deadline IS NULL` (defensive) | Helper would still emit ISO from `None` → AttributeError; helper returns `pending_banner_show=False` defensively if `pending_deadline is None`. Add explicit check |
| Viewer's clock is wrong | Countdown is off by the clock skew; counter is server-authoritative; eventually the next pull replaces wrapper with empty when status flips |
| Tournament not found at partial endpoint | 404 (same as `?partial=live`) |
| Non-default tenant | `pending_banner_show=False`, empty wrapper rendered |
| Lazy-load on `tournament.participants` | Both route handlers must `selectinload(Tournament.participants)` before calling helper. The detail route already does this for the scoreboard; verify the live route too. If lazy-load is a risk, helper returns 0 silently rather than raising — but the right fix is eager load. |
| HTMX swap re-init runs multiple times | `clearInterval(el._intervalId)` before setting new id — no timer accumulation |
| Multiple viewers polling simultaneously | Each viewer hits the route independently. Counter is `len(participants)` — cheap. No new caching needed. |
| Private tournament + non-allowed viewer | Parent page 404 → partial endpoint also 404 (same visibility gate fires before `partial` branch) |

### Defensive helper update

```python
def _pending_banner_context(tournament: Tournament) -> dict[str, Any]:
    show = (
        tournament.status == TournamentStatus.PENDING
        and tournament.tenant_id == DEFAULT_TENANT_ID
        and tournament.pending_deadline is not None
    )
    if not show:
        return {"pending_banner_show": False}
    ...
```

## Performance

- Counter is `O(participants)` in Python (no DB hit beyond what scoreboard already loads).
- 10 s HTMX poll × few admin viewers × pending phase (typically minutes) = negligible load.
- No caching needed in v1.
- JS interval is one `setInterval` per page; cleared and reset on swap.

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
- `pending_deadline IS None` → `{"pending_banner_show": False}`.
- Released participants (released_at set) excluded from counter.

### Integration (`tests/integration/dashboard/test_pending_banner.py`)

**Status gate:**

- Pending tournament: parent page contains `data-deadline-iso` and
  `Registered: N`.
- Active / completed / cancelled: parent page contains the wrapper
  `<div id="pending-banner">` but with no `hx-trigger` and no body.
- 404 on `?partial=pending-banner` for non-existent tournament.

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

- The partial endpoint does NOT set `Cache-Control: public` — counter
  data must not be served from a CDN cache.

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

## References

- `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py:766` (`ui_tournament_live`)
- `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html`
  (existing `?partial=live` HTMX poll on `#live-content`)
- `packages/atp-dashboard/atp/dashboard/tournament/models.py:97-120`
  (`num_players`, `pending_deadline` columns)
- `docs/superpowers/specs/2026-05-01-tournament-shrink-design.md` (the
  shrink-on-deadline behaviour that replaces auto-cancel)
