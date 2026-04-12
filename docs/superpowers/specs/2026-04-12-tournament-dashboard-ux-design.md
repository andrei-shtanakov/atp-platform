# Tournament Dashboard UX — Design Spec

**Goal:** Add tournament list and detail pages to the ATP dashboard so operators and players can see tournament names, participants, scores, round-by-round history, and cancellation reasons — replacing the current minimal 5-column table on `/ui/games`.

**Scope:** Tournament pages only (Plan 2b follow-up #22). Benchmark pages are out of scope.

## Architecture

New routes `/ui/tournaments` (list) and `/ui/tournaments/{id}` (detail) as a separate nav section. The existing `/ui/games` page keeps the game registry table but removes its tournament table (replaced by a link to `/ui/tournaments`). Server-rendered HTMX + Jinja2 + Pico CSS, consistent with the existing dashboard stack.

No new REST API endpoints needed — UI routes query the DB directly via SQLAlchemy (same pattern as existing `/ui/` routes). The existing REST API at `/api/v1/tournaments` is linked from the detail page for raw JSON access.

## Pages

### 1. Tournament List (`/ui/tournaments`)

Single unified table for all users (no admin/player split on this page).

**Columns:**

| Column | Source | Notes |
|--------|--------|-------|
| Tournament | `config["name"]` + `game_type` | Name as purple link → detail page. Game type as small gray text below. |
| Players | `COUNT(participants)` / `num_players` | e.g. "2 / 2". For pending/active: count active participants (`released_at IS NULL`). For completed/cancelled: `COUNT(*)` — all who ever participated. |
| Rounds | `COUNT(completed rounds)` / `total_rounds` | e.g. "30 / 30". "— / 30" for pending. |
| Scores | Per-participant `total_score` | "90 : 90" for 2-player completed/active. If one score is NULL: "45 : —". "—" if both NULL (pending/no rounds). |
| Status | `status` enum | Badge with color: completed=green, active=blue, pending=yellow, cancelled=red. Cancelled rows add human-readable reason below badge. |
| Created by | `users.username` via `created_by` FK | |
| Created | `created_at` | Formatted "Mon DD, HH:MM". |

**Cancel reason mapping** (inline below badge, uses `CancelReason` StrEnum from `atp.dashboard.tournament.reasons`):
- `CancelReason.PENDING_TIMEOUT` ("pending_timeout") → "Expired before full roster"
- `CancelReason.ADMIN_ACTION` ("admin_action") → "Cancelled by admin"
- `CancelReason.ABANDONED` ("abandoned") → "All participants left"

**Pagination:** 50 per page, HTMX-driven (same pattern as `/ui/benchmarks`). Offset-based is acceptable for current scale (tens-hundreds of tournaments).

**Ordering:** Newest first (`id DESC`).

**Scores formatting for 2-player games:** Join participants ordered by `id`, display "score1 : score2". NULL scores render as "—" (e.g. "45 : —" if one participant left early). For N>2 player games (future): show top scorer's value only, e.g. "42 (best)".

**404 handling:** If tournament not found, return `templates/ui/error.html` with 404 status (existing template).

### 2. Tournament Detail (`/ui/tournaments/{id}`)

Three sections visible to all users, one admin-only section.

**404 handling:** If tournament ID doesn't exist, return `templates/ui/error.html` with 404 status.

#### 2.1 Header

- Tournament name (`config["name"]`) as h1
- Meta line: status badge, game_type, "by {username}", time range
  - Completed: "Apr 11, 17:16 — 17:19" (`starts_at` — `ends_at`)
  - Active: "Started Apr 11, 17:16" (`starts_at`)
  - Pending: "Created Apr 11, 16:12" (`created_at`); `starts_at` may be NULL
  - Cancelled: "Created Apr 11, 16:12" (`created_at`)

#### 2.2 Stat Cards (4 cards, horizontal flex row)

| Card | Value | Source |
|------|-------|--------|
| Players | `{joined} / {num_players}` | Active participant count / `tournament.num_players` |
| Rounds | `{played} / {total_rounds}` | Completed round count / `tournament.total_rounds` |
| Duration | `{Xm Ys}` | `ends_at - starts_at` for completed (both non-NULL). "In progress" for active. "—" for pending/cancelled. |
| Round Deadline | `{N}s` | `tournament.round_deadline_s` |

#### 2.3 Scoreboard

Ranked table of ALL participants (including released — they played rounds and have scores):

| Column | Source |
|--------|--------|
| # (rank) | Ranked by `total_score` DESC. Ties share rank. Top 3 get medal emoji. NULL scores sort last. |
| Agent | `participant.agent_name` |
| Score | `participant.total_score` (bold, purple accent). NULL → "—". |
| Avg / Round | `total_score / rounds_played`. NULL if no score. |

Not shown for cancelled tournaments with 0 rounds played.

#### 2.4 Round History

Table of all rounds, **newest first**:

| Column | Source |
|--------|--------|
| Round | `round.round_number` |
| {agent_name} columns | One column per participant. Value: action choice ("cooperate" / "defect") from `action.action_data["choice"]`. Color: green for cooperate, red for defect. Admin sees "(timeout)" suffix for `source=TIMEOUT_DEFAULT` actions. |
| Payoff | Per-participant payoff from `action.payoff`, formatted "3 : 3". NULL payoff → "—". |
| Status | `round.status` |

**Data query:** Eagerly load rounds → actions → participant for the tournament. Single query with `selectinload`.

Not shown for cancelled/pending tournaments with 0 rounds.

#### 2.5 Event Timeline (admin only)

Vertical timeline reconstructed from DB state (events are fire-and-forget on the in-memory bus, not persisted):

- `tournament_created` — `tournament.created_at`
- `participant_joined` — `participant.joined_at`, one per participant
- `round_started` — `round.started_at`, one per round
- `tournament_completed` — `tournament.ends_at` (if completed and non-NULL)
- `tournament_cancelled` — `tournament.cancelled_at` (if cancelled), include reason and cancelled_by username

Sorted chronologically (newest at top — consistent with round history).

Show as a vertical line with dots and event labels. Collapse middle events with "... N more events ..." if > 10 events.

**Admin detection:** The detail route handler loads `User` via `session.get(User, request.state.user_id)` once, then checks `user.is_admin`. This is a single query already needed for the "Created by" username — no extra DB round-trip. The `is_admin` flag is passed to the template context; the template uses `{% if is_admin %}` to show/hide the timeline section.

#### 2.6 Cancelled Tournament View

When status is `cancelled`:
- Header + stat cards shown normally
- Red box replaces scoreboard/rounds:
  - Human-readable cancel reason (bold, red)
  - `CancelReason.PENDING_TIMEOUT`: "Tournament was pending for {duration} without reaching the required {num_players} players ({joined} joined). Auto-cancelled by deadline worker at {cancelled_at}."
  - `CancelReason.ADMIN_ACTION`: "Cancelled by {username} at {cancelled_at}. Reason: {cancelled_reason_detail}"
  - `CancelReason.ABANDONED`: "All participants left. Last departure at {last_released_at}."
- If some rounds WERE played before cancellation, scoreboard and rounds still show below the cancel box.

#### 2.7 JSON Link

Below round history: link to `/api/v1/tournaments/{id}` labeled "View raw JSON →".

#### 2.8 HTMX Polling (active tournaments)

For tournaments with `status=active`:
- Single combined partial polls every 10s: `hx-trigger="every 10s"`, `hx-get="/ui/tournaments/{id}?partial=live"`
- The `?partial=live` response returns stat cards + scoreboard + round history as one HTML fragment, swapped into a single `#live-content` target div.
- This avoids 3 parallel requests per interval (rate limit concern: `ATP_RATE_LIMIT_DEFAULT=60/minute` would be hit with 3×6=18 req/min per viewer).

Same `?partial=` pattern used by `/ui/leaderboard`.

**Rate limiting:** The `/ui/tournaments/{id}?partial=live` endpoint uses the default `@limiter.limit("120/minute")` already applied to UI routes — sufficient for polling at 6 req/min per viewer.

## File Changes

### New files

| File | Purpose |
|------|---------|
| `templates/ui/tournaments.html` | Tournament list page |
| `templates/ui/tournament_detail.html` | Tournament detail page |
| `templates/ui/partials/tournament_list_table.html` | HTMX partial for list pagination |
| `templates/ui/partials/tournament_live.html` | HTMX partial for active tournament polling (stats + scoreboard + rounds combined) |

### Modified files

| File | Change |
|------|--------|
| `routes/ui.py` | Add `ui_tournaments()` list route and `ui_tournament_detail()` detail route |
| `templates/ui/base_ui.html` | Add "Tournaments" nav item after "Games" |
| `templates/ui/games.html` | Remove tournament table, add link to `/ui/tournaments` |
| `static/css/ui.css` | Add styles for tournament-specific elements (scoreboard, round colors, timeline, cancel box) |

### Not modified

- `tournament_api.py` — no new REST endpoints needed
- `service.py` — no new service methods needed; all data accessible via ORM queries in ui.py
- `models.py` — no schema changes

## CSS Additions

New classes in `ui.css`:

- `.cooperate` — green text for cooperate actions
- `.defect` — red text for defect actions
- `.cancel-box` — red-bordered box for cancelled tournament detail
- `.timeline` / `.timeline-item` — vertical event timeline (admin section)
- `.score-pair` — monospace styling for payoff display
- `.admin-badge` — small purple label for admin-only sections

All styles follow existing Pico CSS conventions (no new framework, no Tailwind).

## Data Queries

### Tournament list query

```python
# Base query with participant count and round count
tournaments = await session.execute(
    select(Tournament)
    .options(
        selectinload(Tournament.participants),
        selectinload(Tournament.rounds),
    )
    .order_by(Tournament.id.desc())
    .limit(50)
    .offset((page - 1) * 50)
)
```

Creator username: batch-load via `select(User).where(User.id.in_(creator_ids))` for the page.

### Tournament detail query

```python
tournament = await session.execute(
    select(Tournament)
    .where(Tournament.id == tournament_id)
    .options(
        selectinload(Tournament.participants),
        selectinload(Tournament.rounds).selectinload(Round.actions),
    )
)
```

Single query loads the full object graph. Round history sorted in Python (`sorted(rounds, key=lambda r: r.round_number, reverse=True)`).

Admin detection: `user = await session.get(User, request.state.user_id)` — reuses the same session, one query.

## Nav Structure

"Tournaments" added as a new sidebar item after "Games". The Games page (`/ui/games`) retains the game environment registry table (Available Game Environments: 8 games) and replaces the tournament table with a "View Tournaments →" link. Games page still serves a purpose as the reference for available game types.

## Non-goals

- No tournament creation UI (admin creates via REST API or CLI)
- No tournament cancellation button in UI (use REST API)
- No leaderboard across tournaments (future Plan 2c scope)
- No benchmark page changes
- No responsive/mobile layout
- No SSE/WebSocket (HTMX polling is sufficient)
