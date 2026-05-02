# El Farol Winners Dashboard — Design

**Status:** Draft (revision 3 — addresses follow-up reviews 2026-05-02)
**Date:** 2026-05-02
**Owner:** prosto.andrey.g@gmail.com

## Summary

Add two read-only views surfacing El Farol tournament results:

1. **Per-tournament winners** — `/ui/tournaments/{id}/winners`: a "poster"
   page showing tournament parameters in the header and a ranked table of
   participants (bot, description, owner, score, optional LLM telemetry).
2. **Hall of Fame** — `/ui/leaderboard/el-farol`: cross-tournament leaderboard
   ranking *logical* user-owned agents by total score across all completed
   public El Farol tournaments. Backed by a JSON API at
   `/public/leaderboard/el-farol` (consistent with the existing
   `public_leaderboard.py` namespace).

Both views are public (no auth required) and limited to `join_token IS NULL`
tournaments. Builtin strategies are excluded from the Hall of Fame.

## Goals

- Provide a clean, shareable summary of El Farol tournament results.
- Aggregate user-bot performance across tournaments to surface top agents.
- Expose a stable, versioned JSON contract for the leaderboard so external
  tools can consume it without scraping HTML.

## Non-goals

- No filters/sorting on the Hall of Fame beyond rank-by-total-score
  pagination.
- No editing, no comments, no per-round drill-down.
- No private-tournament support. Private tournaments are excluded from
  both surfaces entirely.
- **Single-tenant only in v1.** Both queries filter by
  `tenant_id = DEFAULT_TENANT_ID`. Multi-tenant deployments will need an
  explicit roll-out (route-scoped tenant resolution) — out of scope here.
- No historical snapshot of agent metadata. Description and owner are
  shown "as of now"; the agent's *display name* on the per-tournament
  poster is taken from `Participant.agent_name` (historical at the time
  of joining), which is the only field where rename-after-the-fact would
  be misleading.

## Surfaces

| Surface | URL | Method | Response | Mounted via |
|---|---|---|---|---|
| Per-tournament UI | `/ui/tournaments/{id}/winners` | GET | Jinja `winners_tournament.html` | `winners_ui` router, included alongside `ui_router` (no prefix) in `factory.py` |
| Hall of Fame UI | `/ui/leaderboard/el-farol` | GET | Jinja `winners_hall_of_fame.html` | `winners_ui` router |
| Hall of Fame JSON | `/api/public/leaderboard/el-farol` | GET | `LeaderboardPayload` | `winners_api` router with `prefix="/public/leaderboard"`, attached to `api_router` (which is mounted under `/api` in `factory.py:245`) |

**Why two routers:** the existing wiring in `factory.py:244-252` mounts
`api_router` under `/api` and `ui_router` directly. A single combined
router would either force UI routes under `/api/ui/...` or force the
JSON endpoint to skip the `/api` prefix. Splitting keeps the conventions
already in use (compare `ui.py` vs `public_leaderboard.py`).

### Access gates

**Per-tournament:** 404 unless all of the following hold:

- Tournament exists.
- `tournament.tenant_id == DEFAULT_TENANT_ID`.
- `tournament.game_type == "el_farol"`.
- `tournament.status == "completed"`.
- `tournament.join_token IS NULL` (= public).

The 404 reason is intentionally generic ("Not found") to preserve the
enumeration guard already used by `TournamentService.get_tournament`.

**Hall of Fame:** always reachable. SQL filter:
`game_type='el_farol' AND status='completed' AND join_token IS NULL AND
tenant_id=DEFAULT_TENANT_ID`. Empty state rendered when no rows match.

### Caching

- Reuse `atp.dashboard.query_cache.QueryCache` (already used by
  `public_leaderboard.py`). TTL 60s for both per-tournament winners and
  the Hall of Fame.
- Set `Cache-Control: public, s-maxage=60` on all three responses.
- `QueryCache` keys are strings produced via `QueryCache._make_key(...)`.
  Cache keys (semantic):
  - Per-tournament: `_make_key("winners", tournament_id)`.
  - Hall of Fame: **must include pagination** —
    `_make_key("hall_of_fame", limit=limit, offset=offset)`. Without
    `(limit, offset)` in the key, page 1 would be served for every
    subsequent paginated request.
- Per-tournament data is immutable post-`completed`, so a longer TTL
  (e.g. `max-age=3600, immutable`) is theoretically optimal. v1 keeps
  60s deliberately so that a future mutable post-completion field
  (e.g. moderator-added annotation) does not require a header migration.
  The route handler must carry an inline comment explaining this choice
  so it doesn't get "optimised" away.

## Data model touchpoints

No schema migrations. Reads only.

- `Tournament` (`packages/atp-dashboard/atp/dashboard/tournament/models.py`)
  — `id`, `tenant_id`, `game_type`, `config`, `status`, `starts_at`,
  `ends_at`, `num_players`, `total_rounds`, `round_deadline_s`,
  `join_token`.
- `Participant` — `tournament_id`, `agent_id`, `agent_name`,
  `total_score`, `builtin_strategy`. **Canonical builtin signal:**
  `agent_id IS NULL` (CHECK constraint enforces XOR with
  `builtin_strategy`).
- `Action` — `participant_id`, `tokens_in`, `tokens_out`, `cost_usd`,
  `model_id`.
- `Agent` (`packages/atp-dashboard/atp/dashboard/models.py`) — `id`,
  `tenant_id`, `name`, `description`, `version`, `owner_id`,
  `deleted_at`.
- `User` — `id`, `username`.

### Header parameters (per-tournament)

`tournament.config` only stores `{"name": name}` (see
`packages/atp-dashboard/atp/dashboard/tournament/service.py:398`). Header
fields are derived from columns:

| Field | Source |
|---|---|
| Name | `(t.config or {}).get("name") or f"Tournament #{t.id}"` |
| Players | `t.num_players` |
| Days | `t.total_rounds` |
| Capacity | `max(1, int(_CAPACITY_RATIO * t.num_players))` — **import** the same `_CAPACITY_RATIO` constant from `el_farol_from_tournament.py:266`; do not inline `0.6`, otherwise winners and the live match engine drift apart silently if the ratio ever changes |
| Duration | `t.ends_at - t.starts_at`, formatted `Hh Mm Ss` (drop leading zero units). Only rendered when both are set; otherwise "—". The status gate (`completed`) guarantees `ends_at` is set in the happy path. |

## Components

### New files

```
packages/atp-dashboard/atp/dashboard/v2/
├── routes/
│   ├── winners_api.py               # JSON: /public/leaderboard/el-farol (under /api)
│   └── winners_ui.py                # HTML: /ui/tournaments/{id}/winners + /ui/leaderboard/el-farol
├── services/
│   └── winners.py                   # shared query helpers (_winners_query, _hall_of_fame_query) + Pydantic models
└── templates/ui/
    ├── winners_tournament.html      # per-tournament page
    └── winners_hall_of_fame.html    # Hall of Fame page

tests/unit/dashboard/winners/
└── test_aggregation.py              # _winners_query / _hall_of_fame_query

tests/integration/dashboard/winners/
├── test_winners_tournament_ui.py    # rendering + access gates
├── test_hall_of_fame_ui.py          # rendering + pagination
└── test_hall_of_fame_api.py         # JSON contract + rate limit + cache headers
```

The shared helpers live in `services/winners.py` so the API and UI
routers run identical SQL — guaranteeing semantic parity between the
JSON contract and what gets rendered server-side.

### Modified files

```
packages/atp-dashboard/atp/dashboard/v2/
├── routes/__init__.py               # +include winners_api router (lands under /api)
├── factory.py                       # +include winners_ui router alongside ui_router
└── templates/ui/
    ├── tournament_detail.html       # +"Winners →" link for completed/el_farol/public
    └── base_ui.html                 # +flat sidebar link "El Farol Hall of Fame"
```

The current sidebar in `base_ui.html` is a flat list. v1 adds one new
flat item; nested grouping is out of scope.

### Module shapes

```python
# services/winners.py — shared models + helpers
class WinnerEntry(BaseModel):
    rank: int
    agent_name: str            # from Participant.agent_name (historical)
    agent_description: str | None  # from Agent.description (as of now)
    owner_username: str            # from User.username (as of now), "—" or "system"
    score: float | None
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    model_id: str | None       # "mixed" if multi-model, None if all NULL

class HallEntry(BaseModel):
    rank: int
    owner_username: str
    agent_name: str            # from Agent.name (logical agent name)
    agent_description: str | None
    total_score: float
    tournaments_count: int

class LeaderboardPayload(BaseModel):
    schema_version: int = 1
    generated_at: datetime
    total: int
    limit: int
    offset: int
    entries: list[HallEntry]

async def _winners_query(session, tournament_id: int) -> list[WinnerEntry]:
    """Per-tournament winners, dense-ranked."""

async def _hall_of_fame_query(
    session, limit: int, offset: int
) -> tuple[int, list[HallEntry]]:
    """Cross-tournament logical-agent leaderboard. Returns (total_count, page)."""


# routes/winners_ui.py
ui_router = APIRouter(tags=["winners-ui"])

@ui_router.get("/ui/tournaments/{tournament_id}/winners", response_class=HTMLResponse)
async def get_winners_html(...): ...

@ui_router.get("/ui/leaderboard/el-farol", response_class=HTMLResponse)
async def get_hall_of_fame_html(...): ...


# routes/winners_api.py
api_router = APIRouter(prefix="/public/leaderboard", tags=["winners-api"])

@api_router.get("/el-farol", response_model=LeaderboardPayload)
async def get_hall_of_fame_json(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
): ...
```

## Data flow

### Per-tournament

```
GET /ui/tournaments/{id}/winners
  ├─ load Tournament, run access gates (404 on any failure)
  ├─ _winners_query(session, id):
  │     SELECT
  │       p.id AS participant_id,
  │       p.agent_name AS display_name,    -- historical, from Participant
  │       p.agent_id,
  │       p.builtin_strategy,
  │       p.total_score,
  │       a.description AS agent_description,
  │       a.deleted_at  AS agent_deleted_at,
  │       u.username    AS owner_username,
  │       SUM(act.tokens_in)              AS tokens_in,
  │       SUM(act.tokens_out)             AS tokens_out,
  │       SUM(act.cost_usd)               AS cost_usd,
  │       MIN(act.model_id)               AS sample_model,
  │       COUNT(DISTINCT act.model_id)    AS distinct_models
  │     FROM tournament_participants p
  │     LEFT JOIN agents a ON a.id = p.agent_id
  │     LEFT JOIN users  u ON u.id = a.owner_id
  │     LEFT JOIN tournament_actions act
  │            ON act.participant_id = p.id
  │     WHERE p.tournament_id = :id
  │     GROUP BY p.id, a.id, u.id
  │     ORDER BY p.total_score DESC NULLS LAST, p.id ASC
  ├─ post-process:
  │     model_id = sample_model if distinct_models <= 1 else "mixed"
  │     dense rank by score (ties share a rank, next jumps)
  │     for builtins (agent_id IS NULL):
  │       owner_username = "system"
  │       agent_description = "built-in strategy"
  │     for archived agents (agent_deleted_at IS NOT NULL):
  │       display_name suffixed " (archived)"
  └─ render winners_tournament.html
```

The `, p.id ASC` tiebreaker keeps tied scores in a deterministic order
across requests.

### Hall of Fame

```
GET /ui/leaderboard/el-farol?limit=50&offset=0
GET /public/leaderboard/el-farol?limit=50&offset=0
  ├─ Pydantic Query validates: 1 <= limit <= 200, offset >= 0 (422 otherwise — strict in both directions)
  ├─ _hall_of_fame_query(session, limit, offset):
  │     -- Step 1: aggregate score per logical agent
  │     WITH agg AS (
  │       SELECT
  │         a.tenant_id,
  │         a.owner_id,
  │         a.name,
  │         SUM(p.total_score)              AS total_score,
  │         COUNT(DISTINCT p.tournament_id) AS tournaments_count
  │       FROM tournament_participants p
  │       JOIN tournaments t ON t.id = p.tournament_id
  │       JOIN agents      a ON a.id = p.agent_id
  │       WHERE t.tenant_id  = :default_tenant
  │         AND a.tenant_id  = :default_tenant
  │         AND t.game_type  = 'el_farol'
  │         AND t.status     = 'completed'
  │         AND t.join_token IS NULL
  │         AND p.agent_id   IS NOT NULL          -- exclude builtins
  │         AND p.total_score IS NOT NULL
  │       GROUP BY a.tenant_id, a.owner_id, a.name
  │     )
  │     SELECT
  │       agg.owner_id,
  │       agg.name,
  │       agg.total_score,
  │       agg.tournaments_count,
  │       u.username,
  │       -- newest extant Agent row's description for the (owner, name) pair
  │       (SELECT description FROM agents a2
  │         WHERE a2.tenant_id = agg.tenant_id
  │           AND a2.owner_id  = agg.owner_id
  │           AND a2.name      = agg.name
  │           AND a2.deleted_at IS NULL
  │         ORDER BY a2.updated_at DESC LIMIT 1) AS description,
  │       -- explicit signal for "every version is soft-deleted"
  │       EXISTS (
  │         SELECT 1 FROM agents a3
  │         WHERE a3.tenant_id  = agg.tenant_id
  │           AND a3.owner_id   = agg.owner_id
  │           AND a3.name       = agg.name
  │           AND a3.deleted_at IS NULL
  │       ) AS has_extant
  │     FROM agg
  │     LEFT JOIN users u ON u.id = agg.owner_id
  │     ORDER BY agg.total_score DESC, agg.owner_id ASC, agg.name ASC
  │     LIMIT :limit OFFSET :offset
  │
  │  -- Total count: COUNT(*) FROM (SELECT 1 FROM ... GROUP BY ...) sub
  ├─ rank = offset + index + 1 (pagination index, NOT score-rank)
  └─ HTML or JSON
```

**Identity:** the leaderboard rows are *logical agents* keyed by
`(tenant_id, owner_id, name)`. Versions of the same agent (same owner +
same name, different `version`) are aggregated together — that is the
intent for a "hall of fame" where the agent evolves over time.

**Description selection:** show the latest non-deleted version's
description. The `has_extant` flag distinguishes the two NULL cases the
post-processor needs to render:

- `has_extant=true, description=NULL` → live agent simply has no
  description; render "—" with no archived badge.
- `has_extant=false` → every version has been soft-deleted; render "—"
  for description and append "(archived)" suffix to the agent name.

**Lineage identity caveat:** the `(tenant_id, owner_id, name)` key is a
*lineage* identity, not an immutable-entity identity. Concretely:

- Renaming an agent (`alfa` → `alfa_v2`) creates a new lineage in the
  Hall of Fame; the original `alfa` retains its accumulated score under
  the old name.
- Deleting all versions of `alfa` and later creating a fresh `alfa` for
  the same owner merges old participation rows with the new agent.

This is the explicit v1 design pick — see Open questions for the
follow-up if user feedback shows it's confusing.

**Pagination ranking caveat:** `rank` is the pagination index, not a
score-rank. Adjacent pages may have rows with equal scores split across
the boundary. The HTML page surfaces a small note ("rank is page-relative
ordering") and the JSON contract documents this behaviour. The
`(total_score DESC, owner_id ASC, name ASC)` tiebreaker keeps ordering
stable across requests.

## Edge cases & error handling

### Per-tournament

| Case | Behaviour |
|---|---|
| Tournament missing | 404 |
| Wrong tenant_id, game_type, status, or `join_token IS NOT NULL` | 404 ("Not found") — generic, enumeration parity |
| Empty participants list | Header renders, table shows "No final scores recorded" |
| `agent_id IS NOT NULL` and `Agent` deleted (`deleted_at IS NOT NULL`) | Show with "(archived)" suffix on display name; do not 500 |
| Builtin participant (`agent_id IS NULL`) | Owner = "system", description = "built-in strategy" |
| Owner missing (LEFT JOIN null) | `owner_username` rendered as "—" |
| `participant.total_score IS NULL` | Score rendered as "—"; row sorted last (NULLS LAST) |
| No telemetry recorded | Token / cost / model columns rendered as "—" |
| Multiple `model_id`s in the same participant's actions | Show "mixed" |
| Partial telemetry (some actions have `cost_usd`, others NULL) | SUM ignores NULLs and returns the partial total. Row shows the partial number; UI does not flag partialness in v1. |

### Hall of Fame

| Case | Behaviour |
|---|---|
| No qualifying tournaments | Empty page: "No completed El Farol tournaments yet." |
| `limit > 200` or `limit < 1` or `offset < 0` | 422 (Pydantic strict validation, no clamp) |
| Builtin participants | Excluded (`p.agent_id IS NOT NULL`) |
| Soft-deleted agent versions | Aggregated into the logical agent. Description falls back to the latest non-deleted version, or "—" if all are deleted; name suffix "(archived)" applied in the latter case. |
| User (owner) deleted | `owner_username` falls back via LEFT JOIN to "—" |
| Multiple Agent versions with same `(owner_id, name)` | Aggregated under one row by design (logical agent identity) |
| Two logical agents tied on score across page boundary | Stable order via `(owner_id ASC, name ASC)`; ranks are page indices |

### Security & abuse

- No auth required (matches existing `public_leaderboard.py`).
- Apply `ATP_RATE_LIMIT_API` (default 120/min) to all three routes.
- `QueryCache` (60s TTL) absorbs anonymous bursts; first request after
  cache miss runs the GROUP BY, subsequent ones are O(1).
- Jinja autoescape handles XSS.

## Performance

- Both queries use existing indices (`idx_participant_tournament`,
  `idx_tournaments_status`, `idx_user_tenant`, `idx_agent_owner`).
- Expected to be fast enough on production-scale data (~1k tournaments,
  ~10k participants, ~500k actions). EXPLAIN ANALYZE results to be
  recorded in the implementation PR description; if the per-tournament
  LLM-telemetry SUM dominates, switch to a two-query strategy
  (participants first, then a separate aggregate over actions). No
  premature optimisation in v1.
- 60s `QueryCache` TTL bounds DB pressure regardless of incoming traffic.

## UI specifics

### `winners_tournament.html`

```
┌────────────────────────────────────────────────┐
│ {{ name }}            [completed] el_farol      │
│ Players: 6 · Days: 30 · Capacity: 4 · 12m 4s   │
├────────────────────────────────────────────────┤
│ # │ Bot       │ Description    │ Owner │ Score │ Tokens in/out │ Cost  │ Model
│ 1 │ 🥇 alfa   │ greedy spammer │ alice │ 142.0 │ 12 345 / 9 80 │ 0.12  │ gpt-4o-mini
│ 2 │ 🥈 beta   │ —              │ bob   │ 121.5 │ —             │ —     │ —
│ 3 │ 🥉 random │ built-in       │ system│  90.0 │ —             │ —     │ —
└────────────────────────────────────────────────┘
[Cards replay →] [Back to tournament →]
```

Reuses styles from `tournament_live.html` (medals for top-3, score-pair
font, muted "—"). Header uses the existing `stat-cards` block.

### `winners_hall_of_fame.html`

```
┌──────────────────────────────────────────────────────┐
│ El Farol Hall of Fame                                │
│ Top user-built agents by total score across all      │
│ completed public El Farol tournaments.               │
│ Note: rank reflects pagination order; ties may split │
│ across pages.                                        │
├──────────────────────────────────────────────────────┤
│ # │ Bot     │ Description       │ Owner │ Total │ T# │
│ 1 │ alfa    │ greedy spammer    │ alice │ 1 420 │ 10 │
│ 2 │ gamma   │ uses RNG          │ bob   │   880 │  7 │
│ 3 │ delta   │ —                 │ carol │   720 │  5 │
└──────────────────────────────────────────────────────┘
[ ◀ prev ]  page 1 / 8  [ next ▶ ]
```

Pagination matches `tournaments.html` HTMX style.

### `tournament_detail.html` patch

```jinja
{% if t.game_type == "el_farol" and t.status == "completed"
      and not t.join_token %}
  <span><a href="/ui/tournaments/{{ t.id }}/winners">Winners →</a></span>
{% endif %}
```

### `base_ui.html` patch

One flat sidebar entry: "El Farol Hall of Fame" → `/ui/leaderboard/el-farol`.
Future game leaderboards add their own flat entries.

## Testing

### Unit (`tests/unit/dashboard/winners/test_aggregation.py`)

- `_winners_query`: SUM tokens / SUM cost across multiple actions.
- Single distinct `model_id` returned; multiple → "mixed"; all NULL → None.
- Builtin participant: owner="system", description="built-in strategy".
- Dense ranking: `[100, 100, 90]` → `[1, 1, 3]`.
- Tiebreaker `, p.id ASC` keeps order stable.
- `_hall_of_fame_query`:
  - SUM across N tournaments for the same `(owner_id, name)` aggregates
    versions together (this is the explicit design choice).
  - Builtins excluded.
  - `total_score IS NULL` rows excluded.
  - Latest non-deleted version's description chosen.
  - All versions deleted → `has_extant=false`, description NULL, name "(archived)".
  - Live agent without description → `has_extant=true`, description NULL,
    no "(archived)" suffix (distinct from the previous case).
  - Two logical agents tied on score → ordered by `(owner_id, name)`.
- Tenant filter: rows from non-default tenant are excluded.

### Integration (`tests/integration/dashboard/winners/`)

**`test_winners_tournament_ui.py`:**

- 200 on completed/public/el_farol/default-tenant; render contains header
  fields and at least one row.
- 404 on each of: missing tournament, pending, active, cancelled, private
  (`join_token` set), non-el_farol game type, non-default tenant.
- Soft-deleted agent rendered with "(archived)" suffix.
- Telemetry-free tournament renders "—" in token / cost / model columns.
- `Cache-Control: public, s-maxage=60` header present.

**`test_hall_of_fame_ui.py`:**

- 200 with empty body when no qualifying tournaments.
- Order: highest total_score first; ties broken by `(owner, name)`.
- Builtins absent from output.
- Private tournaments do not contribute to SUM.
- Multiple agent versions of same `(owner, name)` collapse into one row.
- Pagination: `?limit=2&offset=0` and `?limit=2&offset=2` differ.
- HTMX partial swap on `?partial=1`.

**`test_hall_of_fame_api.py`:**

- Schema matches `LeaderboardPayload`, includes `schema_version=1` and
  `generated_at`.
- `limit > 200`, `limit < 1`, `offset < 0` → 422 (no silent clamp).
- HTML and JSON return semantically identical entries (same helper).
- Rate-limit applies (one test resetting the in-memory rate-limit store).
- `Cache-Control` header present **and equal to** `public, s-maxage=60`
  (assert exact value, not just presence — TTL regressions must fail
  the test).
- Repeat calls inside 60s hit the cache (assert via call counter on the
  query helper).
- HoF cache key includes `(limit, offset)`: `?limit=50&offset=0` and
  `?limit=50&offset=50` must miss independently.

### Smoke

Extend `scripts/smoke_el_farol_prod.py` to fetch
`/ui/tournaments/{id}/winners` after the tournament completes and assert
the response is 200 and contains the agent name.

### Coverage target

≥80% on new code (CLAUDE.md). Use `anyio` for async tests, not
`asyncio`.

## Migration plan

None. No DB changes. Single deploy.

## Open questions

- **Multi-tenant roll-out** — out of scope for v1. When picked up, route
  must resolve tenant from auth context (admin) or query param.
- **MCP tool follow-up** — `get_leaderboard()` would be a natural
  extension of the existing tournament MCP server
  (`atp/dashboard/mcp/`). Not in v1.
- **Per-version drill-down** — not in v1. Logical-agent aggregation is
  the design pick; a per-version detail page can be added later.
- **Rename = new lineage** — accepted for v1. If user feedback shows
  this is confusing, switch identity to a stable surrogate (e.g.
  `Agent.id` of the first version of a `(owner_id, name)` pair). This is
  a non-trivial migration but doable.

## References

- `packages/atp-dashboard/atp/dashboard/v2/templates/ui/tournament_detail.html`
- `packages/atp-dashboard/atp/dashboard/v2/templates/ui/partials/tournament_live.html`
- `packages/atp-dashboard/atp/dashboard/tournament/models.py`
- `packages/atp-dashboard/atp/dashboard/tournament/service.py:1800-1870`
  (visibility rule: `join_token IS NULL` ⇒ public)
- `packages/atp-dashboard/atp/dashboard/v2/routes/public_leaderboard.py`
  (`QueryCache`, `/public/leaderboard` namespace, anonymous-read pattern)
- `packages/atp-dashboard/atp/dashboard/v2/routes/el_farol_from_tournament.py:266`
  (capacity formula)
- `packages/atp-dashboard/atp/dashboard/models.py:67-122` (Agent
  identity: `(tenant_id, owner_id, name, version)`)
- `docs/superpowers/specs/2026-04-15-el-farol-tournament-design.md`
