# Design: Tournament Agent Sandbox

**Status:** Accepted · **Date:** 2026-04-23 · **Epic:** TBD (Linear) · **ADR:** see ADR-005 for a neighbouring decision trail.

## Problem

Tournament agents talk to ATP via the MCP server at `/mcp` when a
tournament runs. Today a user who wants to develop a tournament agent
has no way to test it end-to-end before an admin-created public
tournament starts:

- no private tournament concept exists — all tournaments are
  publicly visible
- only admins can create tournaments
- MCP has no "sandbox" mode
- agent registration is a single flat pool with one quota
  (`ATP_MAX_AGENTS_PER_USER=10`), conflating benchmark-suite agents
  with tournament agents

Result: users ship untested agents to public tournaments, discover
bugs in front of other participants, and have no iteration loop
before real play.

## Goals

1. Users register up to **5 tournament-purpose agents** (distinct
   quota from existing 10 benchmark agents).
2. Users create **private test tournaments** — visible only to
   themselves and admins — through a self-service UI flow, without
   admin mediation.
3. Test tournaments run the **same game engine and the same MCP
   protocol** as real tournaments — no "simulation" shortcut. Users
   who pass their test round are confident their agent will work in
   production.
4. Users can **mix their own agents with builtin strategies**
   (traditionalist, contrarian, random, etc.) as sparring partners so
   even one tournament agent gives a playable roster.
5. Back-pressure: per-user limit on concurrent test tournaments so a
   single user can't grind the queue.

## Non-goals

- Scheduled test tournaments (run now, schedule later → separate
  feature).
- Cross-user private tournaments / invite-based sparring between
  teams.
- A separate "test MCP" endpoint. One MCP with visibility filtering
  on tournaments is simpler (see Architecture below).
- Differences in game mechanics between test and prod
  (time-compressed test mode, cheat-mode, etc.).
- Migration of existing tournaments from "no visibility" to
  explicit public — all existing rows default to `public`.

## Architecture

### Single MCP endpoint, visibility-gated tournaments

One `/mcp` endpoint serves every tournament-purpose agent. The MCP
tools (`list_tournaments`, `join_tournament`, `make_move`, etc.)
filter tournaments by:

```
visibility = 'public'
  OR (visibility = 'private' AND tournament.created_by = agent.owner_id)
  OR user.is_admin
```

This keeps the "test" vs "prod" distinction on the data side
(`Tournament.visibility`) not on the infrastructure side. A test
agent that works against a private tournament is a test agent that
will work against a public one — no "worked in test, broken in prod"
class of bug is possible at the MCP layer.

Same endpoint also means no new FastMCP instance, no duplicated
tools, no separate auth middleware wiring.

### Purpose separation at the agent level

New column `Agent.purpose: Literal['benchmark', 'tournament']`
(server_default `'benchmark'`). Existing rows become benchmark — zero
behavioural change for current users.

Token gating is implicit via the agent FK on `APIToken`:

- MCP middleware rejects tokens whose `agent.purpose != 'tournament'`
  with 403 "MCP is tournament-agents only".
- Benchmark API (`/api/v1/benchmarks/*`) symmetrically rejects
  tokens whose `agent.purpose != 'benchmark'`.

Quotas are additive:

- `ATP_MAX_BENCHMARK_AGENTS_PER_USER=10` (renamed from
  `ATP_MAX_AGENTS_PER_USER`; the old name stays as a read-fallback
  for one release)
- `ATP_MAX_TOURNAMENT_AGENTS_PER_USER=5`
- `ATP_MAX_CONCURRENT_TEST_TOURNAMENTS_PER_USER=3`

A user can hold up to 10 benchmark + 5 tournament = 15 agents total.

### Tournament visibility

New column `Tournament.visibility: Literal['public', 'private']`
(server_default `'public'`). All existing tournaments become public,
preserving current behaviour.

- Private tournaments require an authenticated user in the creator
  role. Public tournaments still require admin to create (status
  quo).
- Private tournament rules:
  - `created_by` is mandatory
  - Roster must include at least one of the creator's tournament
    agents (validator on POST)
  - Per-user concurrent cap: max 3 `pending`/`active` private
    tournaments per creator
- Listings:
  - Anonymous: public tournaments only
  - Authenticated user: `public OR (private AND created_by=me)`
  - Admin: everything
- Detail page / API for not-permitted private tournaments: **404**,
  not 403, to avoid leaking existence.

### Match → tournament linkage

New column `GameResult.tournament_id: int | None` (nullable FK →
`tournaments.id ON DELETE SET NULL`). The CLI game writer fills it
when the match was part of a tournament; CLI standalone runs leave
it NULL.

`/ui/matches` filters matches whose `tournament_id` points to a
private tournament, applying the same `owner|admin` rule as
tournament detail.

## Data Model Summary

### New columns

| Table | Column | Type | Server default | Nullable |
| --- | --- | --- | --- | --- |
| `agents` | `purpose` | `VARCHAR(20)` | `'benchmark'` | NO |
| `tournaments` | `visibility` | `VARCHAR(20)` | `'public'` | NO |
| `game_results` | `tournament_id` | `INTEGER` | — | YES |

### New indexes

- `idx_agents_owner_purpose` on `(owner_id, purpose)` — for "list
  tournament agents of user" quota query.
- `idx_tournaments_visibility_status` on `(visibility, status)` — for
  concurrent-cap query.
- `idx_game_results_tournament` on `(tournament_id)` — for
  match-by-tournament lookups.

### Alembic migration

Single revision `<hash>_agent_purpose_tournament_visibility.py`:
additive columns + indexes + FK on `game_results.tournament_id`. No
backfill needed — all existing rows land at defaults.

## API Changes

### `POST /api/v1/agents`

New optional body field:

```json
{ "name": "...", "purpose": "tournament" }
```

Default `"benchmark"`. Quota enforced by counting
`WHERE owner_id = :me AND purpose = :purpose AND deleted_at IS NULL`
against the appropriate env cap.

### `GET /api/v1/agents`

New query param `?purpose=tournament|benchmark`. Default — all.

### `POST /api/v1/tournaments`

New optional body field:

```json
{ "game_type": "el_farol", "visibility": "private", "participants": [...] }
```

- `visibility='public'` — retains whatever role gate the endpoint has
  today (admin-only in practice via the dashboard admin UI); this
  spec doesn't change that surface.
- `visibility='private'` — requires any authenticated user; extra
  validator: `participants` must include at least one agent whose
  `owner_id = current_user.id`; concurrent-cap checked.
- Default `'public'`.

### `GET /api/v1/tournaments`

Filter applied transparently by caller identity — anonymous sees
public-only, authenticated sees `public + own private`, admin sees
everything. No new query parameters.

### MCP tool behaviour

- `list_tournaments` — returns the visibility-filtered slice.
- `join_tournament`, `make_move`, `get_current_state`,
  `get_history` — 404 on any private tournament whose creator is
  not the agent's owner (the agent sees the same set as its owner).

## UI Changes

### `/ui/tournaments/new` (new)

Jinja + Pico + HTMX form with these fields:

- **Game** — `<select>` backed by `GameRegistry.list_games()`.
- **Visibility** — radio `Private` / `Public`. Public disabled for
  non-admin users (radio locked + tooltip).
- **Your agents** — multi-select of the user's
  `purpose='tournament'` agents, min 1, max 5.
- **Sparring builtins** — checkboxes for each builtin strategy
  available for the chosen game (pulled from
  `game_envs.strategies.<game>_strategies`).
- **Game config** — game-specific inline form
  (for El Farol: `num_rounds`, `num_slots`, `capacity_threshold`).
- **Round deadline (s)** — default 30.

Submit calls `POST /api/v1/tournaments`, redirects to
`/ui/tournaments/{id}`.

### `/ui/tournaments/{id}` (updated)

- `Private` badge next to the existing status pill when
  `visibility='private'`.
- **Cancel** button (creator + admin) visible while status in
  `pending` or `active`.
- Roster progress line — "3/5 participants joined · 2 builtins ready".

### `/ui/tournaments` (updated)

- "**New tournament**" primary button top-right — always visible to
  authenticated users; for non-admins it lands on the form with
  `visibility` locked to `private`.
- Listing filter applied transparently.

### `/ui/agents` (updated)

- **Purpose** column with `benchmark` / `tournament` badges.
- Two separate buttons: **Register benchmark agent**,
  **Register tournament agent**.
- Quota strip: "Tournament agents: 2/5 · Benchmark agents: 7/10".

### `/ui/matches` (updated)

Same visibility filter applied as tournaments, via JOIN on
`GameResult.tournament_id → tournaments.visibility`.

## Error Handling

| Condition | HTTP | Body |
| --- | --- | --- |
| Benchmark-purpose token hits `/mcp` | 403 | `"MCP is tournament-agents only; this token belongs to a benchmark agent"` |
| Tournament-purpose token hits `/api/v1/benchmarks/*` | 403 | `"benchmark API is benchmark-agents only"` |
| `POST /api/v1/agents` with 6th tournament agent | 429 | `"tournament agent quota exceeded (5/5)"` |
| `POST /api/v1/tournaments {visibility: private}` without own agent in roster | 400 | `"private tournament must include at least one of your tournament agents"` |
| 4th concurrent private tournament | 429 | `"concurrent private tournament limit exceeded (3/3)"` |
| `GET /ui/tournaments/{id}` of someone else's private | 404 | Standard 404 page |
| `GET /ui/matches/{id}` of match from someone else's private tournament | 404 | Same |
| Anonymous `POST /api/v1/tournaments` | 401 | `"authentication required"` |

## Testing

**Unit**

- `Agent.purpose` accepts `{benchmark, tournament}`, rejects others.
- `Tournament.visibility` accepts `{public, private}`, defaults to
  public.
- Quota counters: count scope per purpose, exclude `deleted_at`,
  exclude across-tenants.

**Integration**

- MCP `list_tournaments` with benchmark-purpose token → 403.
- MCP `list_tournaments` with tournament-purpose token, as owner of
  a private tournament → private tournament visible.
- MCP `list_tournaments` from user A, private tournament of user B →
  not visible in list, 404 on `join_tournament`.
- `POST /api/v1/tournaments {visibility: private, participants: []}`
  → 400 (no own agent).
- Concurrent-cap: 4th pending private tournament → 429.
- Full flow: user creates private → two of their tournament agents
  join → 2 builtin participants configured at create time → game
  runs → `GameResult` written with `tournament_id` → `/ui/matches`
  shows it to owner, 404 to non-owner.

**Alembic**

- Upgrade to head on fresh SQLite, downgrade, re-upgrade — clean.
- Upgrade on a DB with legacy rows (no new columns) — rows keep
  defaults.

## Rollout

The feature decomposes into four PRs, each shippable independently:

1. **Data model + migration** (small):
   `Agent.purpose`, `Tournament.visibility`, `GameResult.tournament_id`;
   Alembic migration; Pydantic schema updates; new env vars with
   fallback to old names.
2. **API + MCP auth gating** (medium): `POST/GET /api/v1/agents`
   with `purpose`; `POST/GET /api/v1/tournaments` with `visibility`;
   MCP middleware purpose-check and tool-level visibility filter.
3. **UI self-service form** (medium): `/ui/tournaments/new` Jinja
   page; `/ui/agents` quota strip and purpose column;
   private-badge and Cancel button in tournament detail.
4. **End-to-end + docs** (small): Playwright smoke "register
   tournament agent → create private tournament → join via MCP →
   watch match → see result at `/ui/matches`"; CLAUDE.md and
   user-facing docs updated.

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Users flood server with private tournaments | Medium | Medium | Per-user concurrent cap (3); builtin participants are cheap so compute is bounded by `cap × num_rounds × decide_ms` |
| Existing benchmark-agent tokens break after purpose gate | Low | High | Existing tokens stay benchmark-purpose; gate only rejects benchmark-tokens hitting `/mcp`, which no current user should be doing (MCP is new surface) |
| Admin-created public tournaments regress | Low | High | `visibility` defaults to `public`; existing admin API callers omit the field and land on the public branch |
| Private tournament results leaked via leaderboard | Medium | Medium | Leaderboard query must add the same visibility filter; explicit test case |
| GameResult `tournament_id` FK breaks CLI standalone runs | Low | Low | FK is nullable; `_build_game_result_kwargs` sets it only when `run_config.tournament_id` present |

## Out of Scope / Future Work

- Scheduled / deferred-start private tournaments.
- Multi-user private tournaments (invite codes like real tournaments
  have today).
- Replay / rerun of a private tournament with the same roster.
- Admin observability: a dashboard of "active private tournaments
  right now" for capacity planning.
- Agent "promotion" — a flow to convert a tested private agent
  into a production one (copy config, rewrite tokens).

## References

- `packages/atp-dashboard/atp/dashboard/mcp/tools.py` — MCP tool
  definitions
- `packages/atp-dashboard/atp/dashboard/mcp/__init__.py` —
  `MCPAuthMiddleware`
- `packages/atp-dashboard/atp/dashboard/models.py` — `Agent`,
  `GameResult`, `APIToken`
- `packages/atp-dashboard/atp/dashboard/tournament/models.py` —
  `Tournament`, `Participant`
- `atp/cli/commands/game.py::_build_game_result_kwargs` — existing
  match writer
- CLAUDE.md — platform overview
