# Admin Tournament GUI (El Farol first)

**Date:** 2026-04-20
**Status:** Draft (pending user review)
**Scope:** A (admin management UI) + B (live bot-activity monitoring) for El Farol tournaments. Companion feature C (long-lived bot MCP sessions) is out of scope and will get its own spec.

## Overview

Today admins can only interact with tournaments through scattered REST endpoints and a read-only public detail page at `/ui/tournaments/{id}`. There is no place to *create* a tournament through the UI, no way to watch bot-submission activity as it happens, and no way to audit activity after a tournament completes without running SQL.

This spec adds a dedicated admin section under `/ui/admin/tournaments/*` that covers the full tournament lifecycle for El Farol: create → monitor live → cancel → review post-mortem. It also lifts the admin session TTL so a multi-hour monitoring session does not expire mid-tournament.

In-scope admin actions (from brainstorming):

- **a · Create tournament** (must) — form for `game_type=el_farol`, `num_players`, `total_rounds`, `round_deadline_s`, and El-Farol-specific `capacity_threshold`.
- **b · Live monitoring view** (must) — HTMX-polled dashboard: current round, per-participant status table, cross-round activity heatmap.
- **c · Cancel tournament** (must) — UI button wrapping existing `POST /api/v1/tournaments/{id}/cancel`.
- **d · Post-mortem activity log** (must) — reuses the same detail page for `status=completed/cancelled` tournaments; live block becomes static audit view.
- **e · Kick participant** (nice-to-have) — new admin endpoint; sets `Participant.released_at` and creates a `TIMEOUT_DEFAULT` action if kick happens mid-round.

Out of scope (tracked in TODO.md):

- **f** force-advance round · **g** extend deadline — not needed operationally yet.
- **h** MCP SSE connection status — needs a new connection registry; deferred.

## Tech stack

Matches existing dashboard frontend conventions (no new dependencies):

- **HTMX** — `hx-get` + `hx-trigger="every 2s"` for polling; `hx-swap="outerHTML"` for partial fragments.
- **Jinja2 + Pico CSS** — templates under `packages/atp-dashboard/atp/dashboard/v2/templates/ui/admin/`.
- **FastAPI** — new routes live in a new module `packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py`, registered from `factory.py`.
- **SQLAlchemy** — no new tables; existing `Tournament`, `Round`, `Action`, `Participant` are enough.
- **JWT auth** — new env var `ATP_ADMIN_TOKEN_EXPIRE_MINUTES` (default 720 = 12 h) applied at token issuance when `User.is_admin` is true; non-admins keep `ATP_TOKEN_EXPIRE_MINUTES` (default 60).

## Pages and routes

### Admin UI routes (new)

| Route | Method | Page | Notes |
|-------|--------|------|-------|
| `/ui/admin` | GET | Admin landing — links to tournaments, users, invites | Thin index |
| `/ui/admin/tournaments` | GET | Admin list of all tournaments (all statuses, all owners) | Reuses existing list endpoint with `is_admin=True` serializer |
| `/ui/admin/tournaments/new` | GET | Create-tournament form | El Farol only for MVP |
| `/ui/admin/tournaments/new` | POST | Submit form → create → redirect to detail | Wraps existing `POST /api/v1/tournaments` |
| `/ui/admin/tournaments/{id}` | GET | Admin detail (live view when in_progress, post-mortem otherwise) | Two-column layout B |
| `/ui/admin/tournaments/{id}/activity` | GET | HTML fragment with live table + heatmap | Polled every 2 s from the detail page |

All routes gated via a new `require_admin_user` dependency (reuses `User.is_admin` flag; non-admins get 403).

### Admin REST endpoints (new)

| Endpoint | Method | Purpose |
|---|---|---|
| `DELETE /api/v1/tournaments/{id}/participants/{participant_id}` | DELETE | Kick a participant. Sets `released_at`; if kick happens mid-round, inserts a `TIMEOUT_DEFAULT` action so round resolution does not deadlock. Admin-only. |

No new endpoints for the rest — Cancel already exists; Create reuses existing `POST /api/v1/tournaments`.

## Data flow

### Create

```
Admin fills form at /ui/admin/tournaments/new
  → POST /ui/admin/tournaments/new
    → TournamentService.create() (existing)
      → returns Tournament
  → redirect 303 to /ui/admin/tournaments/{id}
```

### Live monitor (current round)

```
Detail page /ui/admin/tournaments/{id} (GET)
  → renders shell (header, cancel button, two-column frame)
  → hx-get="/ui/admin/tournaments/{id}/activity" hx-trigger="every 2s"
    → query:
        SELECT Participant.*, latest Action per round
        FROM participants LEFT JOIN actions
        WHERE tournament_id=:id
      + current Round with deadline
    → renders activity.html.j2 fragment
  → fragment swapped into #activity-block on every tick
```

### Post-mortem

Same detail page, but when `tournament.status IN ('completed', 'cancelled')` the template renders a static variant of the activity block (no timer, no Cancel button, no HTMX polling — just the final heatmap + score table + summary counts).

### Kick (nice-to-have)

```
Admin clicks Kick on a row
  → hx-delete="/api/v1/tournaments/{id}/participants/{pid}"
  → TournamentService.kick_participant(pid) (new method)
      - sets Participant.released_at = now
      - if current round is in_progress and no Action yet for this pid, insert Action(source=TIMEOUT_DEFAULT, action=game.default_action_on_timeout())
      - commits
  → returns 204; HTMX removes the row
```

The existing partial unique index `uq_participant_user_active` (on `user_id WHERE released_at IS NULL`) enforces the "one active tournament per user" invariant — kicking frees the slot immediately.

## Component design

### Route module `admin_ui.py`

```
routes/admin_ui.py
├── router = APIRouter(prefix="/ui/admin")
├── require_admin_user(...)          # dependency: 403 if not is_admin
├── admin_home()                     # GET /ui/admin
├── admin_tournaments_list()         # GET /ui/admin/tournaments
├── admin_tournament_new_form()      # GET /ui/admin/tournaments/new
├── admin_tournament_new_submit()    # POST /ui/admin/tournaments/new
├── admin_tournament_detail()        # GET /ui/admin/tournaments/{id}
└── admin_tournament_activity()      # GET /ui/admin/tournaments/{id}/activity
```

### Templates

```
templates/ui/admin/
├── index.html                       # /ui/admin landing
├── tournaments_list.html            # /ui/admin/tournaments
├── tournament_new.html              # create form
├── tournament_detail.html           # full page shell (live OR post-mortem)
└── _activity_block.html             # HTMX fragment: table + heatmap + timer
```

`_activity_block.html` is the only template polled at 2 Hz. It accepts a single context object `activity` with:

```
activity = {
    "tournament_id": int,
    "status": str,
    "current_round": int | None,
    "total_rounds": int,
    "deadline_remaining_s": int | None,   # None if no round in progress
    "participants": [
        {
            "id": int,
            "agent_name": str,
            "released_at": datetime | None,
            "total_score": float,
            "current_round_status": "submitted" | "waiting" | "timeout" | "released",
            "current_round_submitted_at": datetime | None,
            "row_per_round": ["submitted" | "timeout" | "waiting"] * total_rounds,
        },
        ...
    ],
    "submitted_this_round": int,
    "total_this_round": int,
}
```

The `row_per_round` list drives the heatmap; admin UI renders it as one span per round with CSS classes `submitted` / `timeout` / `waiting`.

### Service-layer additions

`TournamentService`:

- `get_admin_activity(tournament_id: int) -> ActivitySnapshot` — single query path that returns the struct above. Joins `Round` + `Participant` + `Action`, aggregates to the heatmap.
- `kick_participant(tournament_id: int, participant_id: int) -> None` — nice-to-have. Sets `released_at`, inserts `TIMEOUT_DEFAULT` action if mid-round.

### Auth change

`packages/atp-dashboard/atp/dashboard/auth/__init__.py`:

```python
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ATP_TOKEN_EXPIRE_MINUTES", "60"))
ADMIN_TOKEN_EXPIRE_MINUTES = int(os.getenv("ATP_ADMIN_TOKEN_EXPIRE_MINUTES", "720"))
```

In `create_access_token(...)`, pick the appropriate TTL based on the `User.is_admin` flag at issuance time. If an admin's flag is later revoked, their existing long-lived token remains valid until expiry — acceptable for MVP; the admin-revocation security posture is already best-effort (tokens cannot be invalidated early without a denylist, which is also future work).

Downstream cap in `TournamentService.create`: the existing safety check at `service.py:239-247` is `(ATP_TOKEN_EXPIRE_MINUTES − 10) × 60 = budget_s`. This check is about **bot** session budget, so it stays wired to `ATP_TOKEN_EXPIRE_MINUTES`, not the admin TTL. Nothing changes in that path.

## Layout (B — two columns)

Two-column desktop layout for the live/post-mortem detail page. Layout selection rationale recorded in brainstorming session (2026-04-20). Key elements:

```
┌─────────────────────────────────────────────────────────────┐
│ Tournament #12 · El Farol · round 12/30 · in_progress       │
│ created by admin · 6 participants            [Cancel]       │
├───────────────────────────────────┬─────────────────────────┤
│ Current round · 4/6 submitted     │ Activity heatmap        │
│ ┌───────────┬──────────┬────────┐ │ (participants × rounds) │
│ │ Agent     │ Status   │ Score  │ │                         │
│ │ ...       │ ...      │ ...    │ │ bot_alpha ■ ■ ■ ■ ...   │
│ └───────────┴──────────┴────────┘ │ bot_beta  ■ ■ ✕ ■ ...   │
│                                   │ ...                     │
│ (~60% width)                      │ ┌─────────┐             │
│                                   │ │  00:18  │ deadline    │
│                                   │ └─────────┘             │
│                                   │ (~40% width)            │
└───────────────────────────────────┴─────────────────────────┘
```

Below 640 px viewport, Pico's default responsive behaviour collapses the two columns into one (`<article>` elements stack), heatmap becomes scrollable horizontally. No mobile-specific design work in this spec.

## Error handling

| Condition | Behaviour |
|---|---|
| Non-admin hits `/ui/admin/*` | Dependency returns `HTTPException 403`; Jinja renders standard error page. |
| Admin hits `/ui/admin/tournaments/{id}` with unknown id | 404 template. |
| Activity polling fetch fails (DB blip) | Fragment endpoint returns 503; HTMX `hx-swap-oob` inserts a small error banner above the block, polling continues. |
| Tournament completes while admin watches | Next poll returns the post-mortem variant of the fragment; Cancel button replaced by a "View summary" note. |
| Tournament cancelled by another admin | Same as above; banner "cancelled by {user} at {time}". |
| Admin's token expires mid-session | 401 returned on next polling request; HTMX emits `htmx:responseError` → tiny global JS (already in shell) redirects to `/ui/login`. |
| Kick on participant that already has `released_at` | Endpoint returns 409 Conflict; frontend shows toast "already released". |

## Testing strategy

Test pyramid aligned with existing dashboard tests:

**Unit (`tests/unit/dashboard/ui/test_admin_ui.py`)** — new:

- Template rendering of `_activity_block.html` given a mocked `ActivitySnapshot` (happy path, empty tournament, mid-round with mixed statuses, post-mortem read).
- Heatmap cell class selection (ok / timeout / waiting / released).
- Admin token TTL selection: is_admin=True → 720 min, is_admin=False → 60 min.

**Integration (`tests/integration/dashboard/test_admin_tournament_ui.py`)** — new:

- GET `/ui/admin/tournaments` as non-admin → 403.
- GET `/ui/admin/tournaments` as admin → 200 with list.
- Full create flow: POST `/ui/admin/tournaments/new` → redirect → detail page renders.
- Activity fragment polling: seed tournament with known Actions → GET `/activity` → assert fragment contains expected participant rows + heatmap cells.
- Cancel button path: POST cancel → detail page shows cancelled state on next fetch.
- Kick (if nice-to-have included): DELETE participant; if mid-round, verify `TIMEOUT_DEFAULT` action was inserted.

**E2E** — skip for MVP. No Playwright, no browser driver. HTMX polling behaviour is covered by fragment integration tests; the JS payload is minimal (HTMX 2.0 from CDN, no custom JS).

**Regression** — existing `/ui/tournaments/{id}` tests must keep passing (public view is not changed).

## Scope fence

**Explicitly not in this spec:**

- All games other than El Farol — the create form has a `game_type` dropdown but only exposes `el_farol` in MVP. Adding other games = one enum entry + per-game config fieldset. Post-MVP.
- Force-advance round, extend deadline (f, g) — require new service methods and safety analysis.
- Live SSE connection status (h) — requires a new connection registry; deferred.
- Bot session longevity (C) — own spec, not this one.
- Admin-facing audit log across the platform (who-cancelled-what-when) — the detail page surfaces tournament-scoped events, but a global admin audit trail is out of scope.

## Rollout

One PR per MVP slice is workable; the whole spec should fit in a single PR of ~800–1000 LOC (counting templates + tests). If it runs long during implementation planning, the natural split is:

1. **Phase 1** — admin auth gating + `/ui/admin` shell + list + create + cancel (a, c). No live view yet; detail page is the existing public page.
2. **Phase 2** — `_activity_block.html` + polling fragment + heatmap (b, d).
3. **Phase 3** — kick participant (e).

Phase 1 is shippable on its own; phases 2 and 3 layer cleanly. Writing-plans step will decide whether to keep one big plan or three small ones.

## Open questions

None at design time. Implementation plan may surface specifics (exact column widths, exact CSS for heatmap cells, exact SQL for the activity aggregation) — those are implementation details, not design choices.

## References

- Brainstorming session: this conversation, 2026-04-20
- Tournament data model: `packages/atp-dashboard/atp/dashboard/tournament/models.py`
- Existing public detail page: `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py:604`
- Existing REST API: `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py`
- Dashboard frontend stack precedent: `docs/superpowers/specs/2026-04-03-dashboard-frontend-phase1-design.md`
- El Farol rules (for the create form's capacity_threshold defaults): `docs/games/rules/el-farol-bar.en.md`
