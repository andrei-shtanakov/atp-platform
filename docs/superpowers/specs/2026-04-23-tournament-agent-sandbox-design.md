# Design: Tournament Agent Sandbox

**Status:** Accepted · **Date:** 2026-04-23 (revised after review) ·
**Epic:** TBD (Linear) · **Supersedes:** none · **Related:** ADR-005.

## Problem

Tournament agents talk to ATP via the MCP server at `/mcp` when a
tournament runs. Users building tournament agents today can't
effectively rehearse the full flow before a public tournament
because:

- **No purpose-specific quota or classification on agents.** A
  general per-user agent quota is already enforced in
  `packages/atp-dashboard/atp/dashboard/v2/routes/agent_management_api.py`
  (`create_agent_for_user` compares `COUNT(*)` to
  `cfg.max_agents_per_user`), but there is no way to say "this
  agent is for tournaments" vs "this agent is for benchmark
  suites" and therefore no way to set separate limits for the two
  purposes — a single flat pool conflates them.
- **No sparring opponents.** Tournaments require N live participants
  via MCP `join_tournament`. A user who only has one tournament
  agent has no way to test — the tournament never starts because
  the roster never fills. There is no notion of a builtin-strategy
  participant (`game-environments/game_envs/strategies/*` exists but
  isn't wired into the tournament runner).
- **No UI for self-service creation.** `POST /api/v1/tournaments`
  already allows any authenticated user to create a tournament,
  optionally `private=True` (which produces a secret invite
  `join_token`, shown once on creation, reusable by multiple
  joiners until the tournament starts), and the visibility filter in
  `tournament/service.py::list_tournaments` already hides private
  tournaments from non-participants. The UI surface is
  admin-only — there is no `/ui/tournaments/new` for regular users.
- **No match→tournament linkage in `GameResult`.** Tournaments write
  to `Round` and `Action` tables, not `GameResult`. The Cards
  dashboard at `/ui/matches/{match_id}` (LABS-102) reads
  `GameResult` rows and therefore cannot render a tournament match
  today.

Result: users can technically create private tournaments through
the API, but there is no sparring opponent, no agent classification,
no UI form, and no way to watch the resulting match in the Cards
dashboard.

## Goals

1. Users register up to **5 tournament-purpose agents** separately
   from their **10 benchmark-purpose agents** (first-time
   enforcement of both quotas).
2. Users start **private test tournaments** end-to-end via the UI —
   `/ui/tournaments/new` form, reusing the existing `join_token`
   machinery for visibility.
3. Users can add **builtin strategies as sparring participants** at
   tournament-creation time (traditionalist, contrarian, random,
   etc.). A user with one tournament agent still gets a playable
   roster.
4. Completed tournament matches appear in **`/ui/matches`** with the
   same visibility rules, closing the loop between the tournament
   engine and the Cards dashboard.
5. Back-pressure: per-user cap on concurrent `pending`/`active`
   private tournaments.

## Non-goals

- Scheduled / deferred-start tournaments.
- Replacing `join_token`. The existing private-tournament mechanism
  is left intact; this work adds agent-purpose classification and
  builtins on top of it, not a parallel visibility model.
- A second MCP endpoint (`/mcp-test`). One MCP server serves both
  public and private tournaments; visibility filtering already
  lives in the tool layer.
- Multi-tenant / invite-across-organisations flows (possible via the
  existing `join_token` already, but no UI for it).
- Time-compressed or relaxed-rules test mode (sandbox plays the same
  game engine the prod tournament plays).
- Backfill of legacy tournaments with a tournament_id FK on
  pre-existing `GameResult` rows (all existing game results stay
  un-linked).

## Preserved existing behaviour

- `POST /api/v1/tournaments` — any authenticated user can create,
  `private: bool` already works; this spec doesn't change the
  permission gate.
- `Tournament.join_token` — cryptographic (32-byte urlsafe) secret
  for private tournaments. Reusable by multiple joiners until the
  tournament fills — not single-use. Invite-by-link continues to
  work exactly as today; another authenticated user who has the
  token can still join the private tournament via MCP. This is a
  feature, not a bug to fix.
- `list_tournaments` / `get_tournament` visibility filter in
  `service.py` — reused verbatim. Admin sees everything; a regular
  user sees `join_token IS NULL OR created_by=me OR EXISTS
  participant row`.
- `can_view_reasoning` in `tournament/access.py` — reused verbatim.
- Expired-pending auto-cancel in `tournament/deadlines.py` — reused.

## Architecture

The design reuses the existing private-tournament machinery. The
**only new visibility concept** is the concurrent-private cap — the
rest of the filtering comes from `Tournament.join_token IS NULL`
which service-layer queries already key off.

Three new axes are added: **agent purpose**, **builtin participants**,
and **match→tournament linkage**. Each is isolated so the PRs stay
reviewable.

### Agent purpose classification

New column `Agent.purpose: Literal['benchmark','tournament']` with
a SQL `CHECK` constraint plus `server_default 'benchmark'`. Existing
rows become benchmark on migration — no behavioural change.

Purpose-based token gating:

- **Token claims carry `agent_id` and `agent_purpose`.** At token
  issuance time (`POST /api/v1/tokens` for agent-scoped `atp_a_*`
  tokens), the JWT payload — or the `APIToken` row for opaque
  tokens — records the `agent_id` + `agent.purpose` snapshot. No
  DB lookup on the hot path.
- **`JWTUserStateMiddleware`** is extended to decode the claims and
  write `scope["state"]["agent_id"]` / `agent_purpose` alongside
  `user_id`. User-level `atp_u_*` tokens and admin browser sessions
  leave these keys unset.
- **`MCPAuthMiddleware`** reads `agent_purpose` from state and
  rejects anything that isn't `'tournament'`. If `agent_purpose` is
  absent entirely (user-level token or admin browser session), also
  reject — MCP is strictly for agent tokens.
- **Benchmark API (`/api/v1/benchmarks/*`)** symmetrically rejects
  tokens whose `agent_purpose == 'tournament'`. Unset (user-level /
  admin) stays allowed.
- **Migration for existing tokens.** Tokens minted before this PR
  have no claims — when one hits the hot path we fall back to a
  lazy one-time lookup (`SELECT agent_id, purpose FROM api_tokens
  JOIN agents ON ...`) and cache the result in-process keyed by
  token hash. Once an operator rotates tokens, the lazy path stops
  triggering. Rotation note goes in the PR-3 CHANGELOG.

Quotas (all first-time enforcement):

- `ATP_MAX_BENCHMARK_AGENTS_PER_USER=10` — new env var
- `ATP_MAX_TOURNAMENT_AGENTS_PER_USER=5` — new env var
- `ATP_MAX_CONCURRENT_PRIVATE_TOURNAMENTS_PER_USER=3` — new env var

Quota enforcement lives in `POST /api/v1/agents`:
```
COUNT(*) FROM agents
WHERE owner_id = :me
  AND purpose = :purpose
  AND deleted_at IS NULL
```
checked against the per-purpose cap.

### Builtin participants

This is the biggest undesigned chunk and needs its own subsystem.

**Data model.** `Participant.agent_id` is already nullable in today's
schema. Add a new column:

```
Participant.builtin_strategy: VARCHAR(64) | None
```

Invariant enforced in application code (and by a `CHECK` constraint):
```
(agent_id IS NOT NULL AND builtin_strategy IS NULL)
  OR (agent_id IS NULL AND builtin_strategy IS NOT NULL)
```

I.e. every `Participant` row is either a real agent OR a builtin
strategy, never both and never neither.

**Runner.** The tournament runner today reads moves via
`await wait_for_make_move(...)` which blocks on the MCP tool call.
For builtin participants it builds an `Observation` (per
`game_envs.core.strategy.Strategy.choose_action(observation)` — the
real interface, not a hypothetical `decide(state)`) and calls the
strategy directly:

```
obs = build_observation(game_type, game_state, participant)
strategy = resolve_builtin(
    game_type, strategy_name,
    tournament_id=tournament.id, participant_id=participant.id,
)
action = strategy.choose_action(obs)
```

`resolve_builtin` is a dashboard-side wrapper that:

1. Looks up the class by `(game_type, strategy_name)` via
   `game_envs.strategies.registry.StrategyRegistry` (the existing
   real registry — not a fictional `REGISTRY` dict).
2. Inspects the constructor signature and passes a deterministic
   `seed` kwarg **only when the class accepts one**
   (`inspect.signature(cls).parameters`). Non-RNG-backed strategies
   like El Farol's `Traditionalist(window_size=6)` get plain default
   args; RNG-backed strategies like the `Random` family receive the
   seed.
3. Derives the seed from the tournament + participant identity using
   a **stable hash** (`hashlib.blake2b(f"{tournament_id}:{participant_id}".encode(), digest_size=8)`),
   **not** Python's built-in `hash()` — the built-in is randomised
   per-process by default (`PYTHONHASHSEED`) and would break the
   "two runs produce identical moves" guarantee.

Because strategies across games share short names (e.g.
`"traditionalist"` is El Farol, `"random"` exists in multiple
games), the dashboard registry wrapper is **per-game**: it imports
the game's strategies module to trigger `StrategyRegistry.register`
calls, then narrows to classes exported from that module. The
wire-level `builtin_strategy` value on `Participant` stays
namespaced (`el_farol/traditionalist`) — this is what the
`POST /api/v1/tournaments` roster submits and what the server
round-trips back.

The runner's "wait for move" state machine needs one branching
point:

```
if participant.builtin_strategy is not None:
    action = builtin.choose_action(obs)   # sync, deterministic
else:
    action = await wait_for_make_move(...)  # via MCP
```

**Observation construction** is game-specific. The runner already
knows how to build per-player views of game state for MCP
participants (same data the MCP `get_current_state` tool returns);
the builtin path reuses the same construction, converting the
per-player dict into the `Observation` dataclass the Strategy base
class expects.

**Cross-package boundary.** `GameRegistry` and the strategies live
in `game-environments`, which `atp-dashboard` does not currently
import. Pragma: accept the new import, add `game-environments` to
`atp-dashboard`'s runtime dependencies (it's already a peer package
in the uv workspace — no external dep). Alternative — hardcoding the
builtin list in dashboard — rejected, drifts with upstream.

**Creation API.** `POST /api/v1/tournaments` body gains an optional
`roster` field. Strategy names are namespaced by game (see
`GET /api/v1/games/{game_type}/builtins` below):

```json
{
  "game_type": "el_farol",
  "private": true,
  "roster": [
    {"builtin_strategy": "el_farol/traditionalist"},
    {"builtin_strategy": "el_farol/contrarian"}
  ],
  "config": {...}
}
```

Real agents still join through MCP `join_tournament` after creation
(status quo). Builtins are inserted into `Participant` rows
**immediately** at creation. The tournament engine's existing
"start when `participants.count() == num_players`" check keeps
working unchanged: builtins already count because they already
exist as Participant rows.

Roster semantics, explicit:

- `num_players = N`, `len(roster) = K` → engine waits for
  `N - K` MCP joiners before starting. Already existing behaviour.
- `K == N` → tournament starts immediately, no MCP joiners
  expected. Edge case worth testing — validates "pure sparring
  against builtins".
- `K > N` → 400 on POST, "builtin roster larger than num_players".
- `K == 0`, `private=true` → validator still requires at least one
  of the creator's tournament-purpose agents to be eligible (the
  creator must commit to joining); the agent joins via MCP after
  creation as today. 400 if creator has zero tournament-purpose
  agents.

### Concurrent-private cap

A new validator on `POST /api/v1/tournaments` when `private=True`:

```sql
SELECT count(*) FROM tournaments
WHERE created_by = :me
  AND join_token IS NOT NULL
  AND status IN ('pending','active')
  AND (status != 'pending' OR pending_deadline > NOW())
```

— if ≥ `ATP_MAX_CONCURRENT_PRIVATE_TOURNAMENTS_PER_USER`, return
429. The `pending_deadline > NOW()` guard excludes already-expired
pending tournaments so the auto-cancel helper in
`tournament/deadlines.py` doesn't get into a race with this cap.

### Match → tournament linkage

The central question: how does `/ui/matches` surface tournament
matches?

**Chosen approach: write a `GameResult` row at tournament
completion.** The tournament runner, on its terminal-state
transition (`status → completed`), calls a new writer that
materialises the tournament into the `GameResult` schema:

- `game_name`, `game_type` come from the game config
- `match_id` is **always a freshly-generated UUID**, never the
  tournament's `join_token`. The `join_token` is a 32-byte invite
  secret and must never appear in URLs, access logs, or browser
  history. `GameResult.tournament_id` carries the real linkage.
- `tournament_id` (new nullable FK) is set to `tournament.id`
- `actions_json`, `day_aggregates_json`, `round_payoffs_json`,
  `agents_json` are filled from `Round` / `Action` rows via a
  reshape — same shape the CLI writer produces today

This is **dual-write**, so idempotency matters. Rather than the
check-then-insert pattern (which is TOCTOU-vulnerable under
concurrent completion handlers), enforce uniqueness in the schema:

```sql
CREATE UNIQUE INDEX uq_game_results_tournament_id
    ON game_results(tournament_id)
    WHERE tournament_id IS NOT NULL;
```

The writer just attempts the INSERT and treats `IntegrityError` on
this constraint as "already written, drop it". Works on Postgres
and SQLite ≥3.8.

Trade-offs considered:

- UNION at the `/ui/matches` query level — rejected. Makes every
  matches query cross-join two tables, and the Pydantic schema at
  `/api/v1/games/{match_id}/dashboard` is already glued to
  `GameResult`.
- Tournament-only route `/ui/tournaments/{id}/results` — partial
  win. But the Cards dashboard we just shipped lives on
  `/ui/matches/{match_id}` and the user's mental model is "one
  place for all completed El Farol runs". Keep the surface unified.

### Visibility filtering for matches

`/ui/matches` JOINs `GameResult` ↔ `Tournament` on
`tournament_id` and applies:

```
tournament_id IS NULL
  OR tournament.join_token IS NULL
  OR tournament.created_by = :me
  OR EXISTS participant with user_id = :me
  OR user.is_admin
```

**Non-regression invariant.** Every `GameResult` row that exists
today has `tournament_id IS NULL` (the column is new with this
spec), so every current match passes the first clause and stays
visible to anonymous visitors just like today. No existing user-
facing match disappears. The filter only takes effect for
tournament matches written **after** PR-5 lands.

An integration test on the `/ui/matches` listing must include an
anonymous caller seeing a `tournament_id IS NULL` match — the
contract guarantee for legacy data.

## Data Model Summary

### New columns

| Table | Column | Type | Default | Nullable | Constraint |
| --- | --- | --- | --- | --- | --- |
| `agents` | `purpose` | `VARCHAR(20)` | `'benchmark'` | NO | `CHECK (purpose IN ('benchmark','tournament'))` |
| `participants` | `builtin_strategy` | `VARCHAR(64)` | — | YES | `CHECK ((agent_id IS NOT NULL) != (builtin_strategy IS NOT NULL))` |
| `game_results` | `tournament_id` | `INTEGER` | — | YES | `FK tournaments(id) ON DELETE SET NULL` |

No `Tournament.visibility` column — we reuse `join_token IS NULL` /
`IS NOT NULL` as the public/private marker (existing convention).

### New indexes

- `idx_agents_owner_purpose` on `(owner_id, purpose)` — quota count
- `idx_participants_builtin` on `(tournament_id, builtin_strategy)`
  where `builtin_strategy IS NOT NULL` — partial, for builtin
  roster snapshot
- `uq_game_results_tournament_id` — **UNIQUE partial index** on
  `(tournament_id) WHERE tournament_id IS NOT NULL` — enforces
  at-most-one `GameResult` per tournament, neutralising dual-write
  TOCTOU races (see Match → tournament linkage)

### Alembic migration

One revision. Pre-requisite: migration runs before the new quota
env vars are read (simple — env reads happen at request-time, not
import-time). Legacy rows: all existing agents land as benchmark;
existing participants keep `builtin_strategy=NULL` (they're all
agent-backed); existing game_results keep `tournament_id=NULL`.

## API Changes

### `POST /api/v1/agents`

```json
{ "name": "my-agent", "agent_type": "mcp", "purpose": "tournament" }
```
`purpose` optional, default `"benchmark"`. Quota enforced per-purpose.

### `GET /api/v1/agents?purpose=tournament`

New optional query param. Default lists all purposes for the user.

### `POST /api/v1/tournaments`

Request body gains one optional field on top of today's schema.
The snippet below shows **only the new / relevant fields** —
`name`, `num_players`, `total_rounds`, `round_deadline_s`, etc.
remain required exactly as today:

```json
{
  "game_type": "el_farol",
  "private": true,
  "roster": [{"builtin_strategy": "el_farol/traditionalist"}],
  "config": {...}
}
```
- `roster` is a list of builtin strategies by name (agent-backed
  participants still join via MCP after creation).
- `private: true` triggers the concurrent-cap check (new) plus
  existing `join_token` generation.
- Validator: if `private: true`, the roster OR the user's
  tournament-purpose agent count must be positive (non-empty
  creator commitment).

### MCP tools

- `list_tournaments` — unchanged in SQL (visibility filter already
  correct); new post-filter rejects benchmark-purpose token at the
  middleware layer before the tool is even called.
- `join_tournament` / `make_move` / `get_current_state` /
  `get_history` — purpose-gated at middleware.
- No new tools. Builtins never talk to MCP — they're resolved
  server-side.

### `GET /api/v1/games/{game_type}/builtins`

New endpoint backing the "Builtin sparring partners" widget on
`/ui/tournaments/new`. Returns the namespaced list of builtin
strategies available for a game:

```json
{
  "game_type": "el_farol",
  "builtins": [
    {"name": "el_farol/traditionalist", "description": "..."},
    {"name": "el_farol/contrarian", "description": "..."},
    {"name": "el_farol/random", "description": "..."}
  ]
}
```

The endpoint returns **namespaced** names (`{game}/{strategy}`) so
the client never has to prefix anything — the value in
`builtins[i].name` is the exact string to submit as
`roster[i].builtin_strategy` on `POST /api/v1/tournaments` and the
exact string stored in `Participant.builtin_strategy`. Short name
`"random"` exists in PD, auction, congestion, and blotto as
distinct classes; namespacing keeps the wire unambiguous
end-to-end. The validator on `POST /api/v1/tournaments` checks
`(game_type, strategy_name) ∈ registry.allowed_set()`.

## UI Changes

### `/ui/tournaments/new` (new)

Jinja + Pico + HTMX form, reachable from a primary button on
`/ui/tournaments` for authenticated users. Fields:

- **Game** — `<select>`. Options come from a small
  dashboard-owned whitelist (`el_farol`, `prisoners_dilemma`,
  etc.) mirrored from `game_envs.games.registry` at import time.
  Avoids cross-package runtime queries per request.
- **Visibility** — radio `Private` / `Public`. Public is enabled
  only for admins (today's permission baseline).
- **Your agents** — multi-select of the user's
  `purpose='tournament'` agents (0..5; zero is OK if the roster
  contains builtins).
- **Builtin sparring partners** — checkbox list of builtins
  available for the chosen game; submitted as `roster` field.
- **Config** — game-specific inline (El Farol: `num_rounds`,
  `num_slots`, `capacity_threshold`).
- **Round deadline (s)** — default 30.

On success the server **renders the tournament detail page directly
from the POST handler** — no redirect. The HTML response carries
the `join_token` in a dismissible copy-box at the top. A POST→GET
redirect would drop the token from the response; stuffing it into a
query param would leak it to browser history and access logs.
Refreshing the detail page later fetches it again via a normal GET
that does **not** return the token — the user sees the token once,
stays on the page, copies it out of band. This is the only
deviation from the rest of `/ui/*` where POST handlers redirect.

### `/ui/tournaments/{id}`

- `Private` badge next to status pill when `join_token IS NOT NULL`
- Cancel button for creator + admin while status in
  `pending`/`active`
- Roster progress: "3/5 MCP participants joined · 2 builtins
  ready"

### `/ui/agents`

- `Purpose` column with `benchmark`/`tournament` badges
- Two buttons: `Register benchmark agent`, `Register tournament
  agent`
- Quota strip: "Tournament agents: 2/5 · Benchmark agents: 7/10"

### `/ui/matches`

- Filter updates with the JOIN described above. Anonymous visitors
  see only matches from public tournaments plus CLI standalone
  runs.

## Error Handling

| Condition | HTTP | Body |
| --- | --- | --- |
| Benchmark-purpose token hits `/mcp` | 403 | `"MCP is tournament-agents only; this token belongs to a benchmark agent"` |
| User-level token (`atp_u_*`) or admin browser session hits `/mcp` | 403 | `"MCP requires an agent-scoped token (atp_a_*)"` |
| Tournament-purpose token hits `/api/v1/benchmarks/*` | 403 | `"benchmark API is benchmark-agents only"` |
| `POST /api/v1/agents` with 6th tournament agent | 429 | `"tournament agent quota exceeded (5/5)"` |
| `POST /api/v1/tournaments {private:true, roster:[]}` with no owned tournament agents | 400 | `"private tournament needs at least one participant (your agent or a builtin)"` |
| 4th concurrent private tournament | 429 | `"concurrent private tournament limit exceeded (3/3)"` |
| Unknown builtin strategy in roster | 400 | `"unknown builtin strategy 'xxx' for game 'el_farol'"` |
| `GET /ui/tournaments/{id}` of someone else's private without the token | 404 | Standard 404 page |
| `GET /ui/matches/{id}` of match from someone else's private tournament | 404 | Same |
| Anonymous `POST /api/v1/tournaments` | 401 | `"authentication required"` |

## Testing

**Unit**

- `Agent.purpose` accepts `{benchmark, tournament}`, rejects others
  (CHECK constraint + Pydantic).
- `Participant` invariant: exactly one of `agent_id` /
  `builtin_strategy` set.
- Quota counters — per-purpose, exclude `deleted_at`, per-tenant.
- Builtin registry lookup: known names resolve, unknown 400.

**Integration**

- MCP `list_tournaments` with benchmark-purpose token → 403.
- MCP `list_tournaments` with tournament-purpose token, private
  tournament I created → visible; same from another user → hidden
  unless they were given the `join_token`.
- `POST /api/v1/tournaments {private:true, roster:[]}` with zero
  tournament agents → 400.
- Concurrent-cap: 4th `pending`/`active` private → 429. 4th with
  expired `pending_deadline` → allowed (auto-cancel pending).
- Full flow: user creates private with 1 own agent + 2 builtins →
  agent joins via MCP → tournament starts → builtins play
  deterministically → completion writes `GameResult` row with
  `tournament_id` → `/ui/matches` shows it to owner, 404 to non-
  owner.
- Leaderboard (`/api/v1/leaderboard`, `/ui/leaderboard`) — private
  tournament runs don't surface. Explicit test on the SQL.

**Alembic**

- Upgrade on fresh SQLite + Postgres, downgrade, re-upgrade.
- Upgrade on populated DB — all existing agents remain benchmark;
  all existing participants keep `builtin_strategy=NULL`; all
  existing game_results keep `tournament_id=NULL`.

## Rollout

Five PRs, each independently shippable:

1. **Data model + migration** (small): the three columns + indexes
   + `CHECK` constraints + Alembic revision. Pydantic schema
   updates. No behaviour change on endpoints.
2. **Quota enforcement + purpose-based API** (small): actually
   enforce the two env vars in `POST /api/v1/agents`; add
   `purpose` field to request and response; `?purpose=` filter on
   list. **First time** the platform rejects an over-quota agent
   registration.
3. **MCP/benchmark auth gating** (medium): token → agent
   resolution in `JWTUserStateMiddleware`; `MCPAuthMiddleware`
   rejects non-tournament tokens; benchmark API rejects
   non-benchmark tokens. Requires careful rollout — any existing
   benchmark-purpose token still works everywhere it worked
   before.
4. **Builtin participants + tournament runner integration**
   (large): `Participant.builtin_strategy` machinery, tournament
   runner branching, `roster` field on `POST /tournaments`,
   concurrent-cap, cross-package `GameRegistry` import.
5. **UI self-service form + match linkage** (medium):
   `/ui/tournaments/new`, `/ui/agents` updates, private badge and
   cancel button on tournament detail, `GameResult` dual-write on
   tournament completion, visibility JOIN on `/ui/matches`. Ends
   with a Playwright smoke covering the full flow.

PR-4 is the heaviest (that's where the original spec undercounted
effort).

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Users flood server with private tournaments | Medium | Medium | Per-user concurrent cap of 3; builtins are cheap (synchronous decide) so compute is bounded by `cap × num_rounds × decide_ms` |
| MCP rejects existing benchmark-purpose tokens retroactively | Low | High | Pre-existing tokens default to benchmark purpose; MCP is a new surface — no one currently uses it with a benchmark token, but a test confirms the 403 path only catches the intended class |
| Token → agent DB lookup on every JWT-authenticated request | Medium | High | PR-3 puts `agent_id` / `agent_purpose` into the token claims at issuance so the hot path is decode-only. A one-time lazy lookup with in-process cache absorbs legacy tokens until operators rotate. p99 regression monitored on `/api/v1/benchmarks/next-task` and `/mcp` after PR-3 rolls out |
| Builtins produce non-deterministic moves across two runs of the same private tournament | Medium | High | Runner seeds every RNG-backed builtin with `int.from_bytes(hashlib.blake2b(f"{tournament.id}:{participant.id}".encode(), digest_size=8).digest(), "big")` — Python's built-in `hash()` is `PYTHONHASHSEED`-randomised per process and must not be used. Documented in Architecture → Builtin participants. Unit test: run the same tournament twice, assert identical action transcripts |
| Private tournament results leak into public leaderboard | Medium | Medium | Explicit test on the leaderboard SQL joins `game_results.tournament_id → tournaments.join_token IS NULL` |
| `GameResult` dual-write races with tournament completion | Low | High | Dual-write runs inside the same DB transaction that flips status to `completed`; idempotency guard on `tournament_id` unique lookup |
| Cross-package `game-environments` import breaks dashboard startup if registry misconfigured | Low | Medium | Import is at module-import time of `tournament/service.py`; any broken registry crashes the server at startup rather than runtime — easy to catch in CI |

## Out of Scope / Future Work

- Scheduled / deferred-start private tournaments.
- "Promote agent": convert a tested private-tournament agent into a
  production agent (copy config, rewrite tokens).
- Admin observability page — list of all active private
  tournaments for capacity planning.
- Replacing `join_token` with a visibility enum — possible future
  cleanup if the token mechanism limits us, but not needed today.

## References

- `packages/atp-dashboard/atp/dashboard/tournament/service.py:1020–1090`
  — existing visibility filter
- `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py:297`
  — existing authenticated-user creation gate
- `packages/atp-dashboard/atp/dashboard/tournament/models.py:211`
  — `Participant.agent_id` already nullable
- `packages/atp-dashboard/atp/dashboard/mcp/auth.py` — middleware
  lacking agent context (see Architecture → Agent purpose
  classification for the fix)
- `packages/atp-dashboard/atp/dashboard/tournament/deadlines.py` —
  expired-pending auto-cancel (interacts with concurrent-cap)
- `game-environments/game_envs/strategies/` — builtin strategies
  module
- ADR-005 (`docs/adr/005-el-farol-dashboard-stack.md`) — dashboard
  rendering stack decision
