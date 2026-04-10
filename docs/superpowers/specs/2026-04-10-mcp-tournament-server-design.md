# MCP Tournament Server Design

## Summary

Build an MCP-first tournament server on the ATP platform that lets any MCP-compliant client (Claude Desktop, Claude Code, Cursor, Anthropic Agent SDK, our in-house `MCPAdapter`, etc.) participate in live turn-based game-theoretic tournaments via a single remote SSE endpoint. v1 ships with Prisoner's Dilemma as the flagship game, N-player-capable service layer from day 1, and a read-only tournament dashboard.

## Context

### Status of prerequisites (2026-04-10)

- ✅ **Issue 1 (IDOR) fixed** — `e46a98b`. Benchmark run lifecycle endpoints enforce ownership; `Run.user_id` is NOT NULL with backfill. Phase 0 blocker of the original MCP plan is lifted.
- ✅ **Per-user rate limiting** — this session. `JWTUserStateMiddleware` populates `request.state.user_id` so slowapi keys per user, not per IP. Required for multi-participant public tournaments behind shared NAT.
- ✅ **SDK `on_token_expired` + `drain()`** — this session. Long-running clients can refresh tokens without losing in-flight state. Benefit for benchmark SDK and, by extension, any future Python participant path.
- ❌ **MCP server on the platform** — does not exist. `tournament/` has models and schemas but the REST endpoints are 501 stubs. No FastMCP import anywhere.
- ✅ **Tournament SQLAlchemy models** — `Tournament`, `Participant`, `Round`, `Action` exist from migration `4e902371a941_add_tournament_tables.py`. Usable with small additions.
- ✅ **`game-environments`** — 9 games implemented with a `GameRegistry`, including `prisoners_dilemma`, `el_farol`, `stag_hunt`, `battle_of_sexes`, etc. Core state/action/history primitives in `core/`.
- ✅ **In-house MCP client `MCPAdapter`** — supports both stdio and SSE transports (`packages/atp-adapters/atp/adapters/mcp/transport.py`, 997 lines). Reusable as e2e test harness and as "programmatic Python participant" path.

### Out of scope

- Benchmark MCP facade (separate parallel track, post-v1)
- Any new REST gameplay endpoints — the existing 501 stubs in `tournament_api.py` will be removed
- Multi-worker deployment / Redis event bus — single uvicorn worker is sufficient for MVP and the current production shape
- stdio transport for participants — remote SSE only (simpler distribution, no per-participant CLI install)

### Goal (definition of done)

Run a public Prisoner's Dilemma tournament at `https://atp.pr0sto.space/mcp/sse` with external participants joining via standard MCP clients, playing 100-round matches end-to-end through the MCP `notifications/message` push model, with server-enforced round deadlines and a read-only dashboard showing round-by-round history.

## Architectural decisions

### AD-1: SSE-only transport, no stdio for v1

**Decision.** Single remote endpoint `https://atp.pr0sto.space/mcp/sse`. Participants put ~10 lines of JSON in their MCP client config.

**Why.** The infrastructure that usually makes SSE hard (TLS, deploy pipeline, JWT middleware, per-user rate limits, CORS) is already solved on this platform as of this session. The remaining delta vs stdio is a single `app.mount("/mcp", mcp.sse_app())` call plus handshake auth. The UX delta in the other direction is large: SSE gives every participant a 10-line config on macOS/Windows/Linux identically; stdio forces each participant to `pip install atp-platform` with its Python dependency chain.

**Rejected alternatives.**
- stdio-only — raises the install barrier for external participants. Useful later as a local debug mode (backlog J).
- Both — doubles integration and test surface without v1 benefit.
- SSE primary + stdio fallback — no v1 user justifies the second transport.

### AD-2: MCP-first gameplay, REST read-only admin

**Decision.** All gameplay (`join`, `get_state`, `make_move`, `leave`, notifications) goes through MCP. The REST admin surface is strictly read-only (`list`, `get`, `rounds`) plus one admin write endpoint (`POST /api/v1/tournaments` for tournament creation). Admin authorization reuses the existing RBAC `User.is_admin` flag — no new permission model.

**Why.** Building both MCP and REST gameplay is double maintenance without a second audience. MCP is the authored-for-LLMs protocol with server-push notifications, which turn-based games need. REST read-only is enough for the dashboard UI and research scripts.

**Consequence.** The 501 stubs in `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py:74-117` are **deleted**, not completed.

### AD-3: N-player-capable service layer from day 1

**Decision.** `TournamentService.join` accepts `num_players` from the Tournament record, not hardcoded to 2. `Round` and per-player state formatting handle arbitrary participant counts. PD runs as `num_players=2` special case.

**Why.** The planned second game (El Farol Bar) is N-player. Generalizing from 2 to N in v1 costs almost nothing — the difference is `participants: list` instead of `player_a`/`player_b` and deferring per-player `format_state_for_player` to the game registry. Refactoring a 2-player-hardcoded service later is measurably harder.

### AD-4: In-process `asyncio.Queue` event bus, no Redis

**Decision.** `TournamentEventBus` is a module-level singleton with per-tournament `set[asyncio.Queue]` subscribers. Events are ephemeral and delivered best-effort to connected sessions.

**Why.** Production is a single uvicorn worker; inter-worker event fan-out isn't a v1 requirement. In-process pub/sub is trivially simple, has full access to the same connection pool, and adds zero deploy complexity.

**Escape hatch.** When load or multi-worker concerns arrive (trigger in backlog I), swap the `TournamentEventBus` implementation for a Redis-backed one without touching callers. The interface is designed to make this a drop-in replacement.

### AD-5: Round resolution is synchronous, inside the last `make_move`

**Decision.** When the last participant submits their action for a round, the `make_move` handler synchronously resolves the round, updates scores, creates the next round, and publishes events — all in the same DB transaction. No separate "round resolver" worker.

**Why.** Fewer moving parts. The resolution itself is a 2x2 matrix lookup (PD) or equivalent cheap computation. Atomicity in one transaction means there is no window where an action is recorded but the round is not resolved. The cost is that the last mover's `make_move` response includes round result data instead of just an ack — that's a feature, not a bug.

### AD-6: Deadline enforcement via single asyncio background task

**Decision.** One `asyncio.create_task` in the FastAPI `lifespan`, ticking every ~2 seconds. Each tick runs one `SELECT` for expired rounds and calls `force_resolve_round` on each, which fills missing actions with the game's default action and resolves the round.

**Why.** Matches the single-process model. No Celery, no rq, no separate worker deployment. Tick granularity of 2s is acceptable for 30s round deadlines.

**Race protection.** `force_resolve_round` does `UPDATE rounds SET status='resolving' WHERE id=? AND status='waiting_for_actions'`; if 0 rows affected, silently exits. This guards against races between the deadline worker and a last-second `make_move`.

### AD-7: No reconnect event replay in v1

**Decision.** If a participant's session drops, their pending notifications are lost. They can reconnect and call `get_current_state` to see where the tournament is now. Subsequent notifications flow normally.

**Why.** Event replay requires server-side per-player delivery tracking, which is significant complexity. The fallback (re-query state on reconnect) is good enough for MVP. Backlog C tracks this with an explicit trigger.

### AD-8: `make_move` is NOT idempotent — second call returns 409

**Decision.** The database-level `UniqueConstraint(round_id, participant_id)` on `Action` guarantees one action per player per round. A second `make_move` call returns 409 Conflict. SDK/MCP clients must handle this as a terminal state, not retry.

**Why.** Idempotent submit (return 200 with `already_submitted` marker) is simpler UX but requires a SELECT before INSERT or careful IntegrityError handling. v1 starts with the strict version; we relax it (backlog B) if observed retry behavior in the first tournament causes real confusion.

## Architecture

### Module layout

**New modules** under `packages/atp-dashboard/atp/dashboard/`:

```
tournament/
  service.py        — TournamentService, protocol-agnostic core
  events.py         — TournamentEventBus + TournamentEvent dataclass
  deadlines.py      — background deadline worker loop
  state.py          — RoundState formatter, delegates to game-environments
  errors.py         — ValidationError, ConflictError, NotFoundError
  (models.py, schemas.py — exist, modified per §Persistence)

mcp/
  __init__.py       — FastMCP instance, session storage
  tools.py          — MCP tool handlers (thin wrappers over TournamentService)
  notifications.py  — per-session event subscriber → notifications/message pump
  auth.py           — MCPAuthMiddleware (reads request.state.user_id)
```

**Modified modules:**

- `packages/atp-dashboard/atp/dashboard/v2/factory.py` — mount `mcp.sse_app()` under `/mcp`; start deadline worker in `lifespan`
- `packages/atp-dashboard/atp/dashboard/tournament/models.py` — additive schema changes (§Persistence)
- `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py` — delete 501 stubs, add 3 read-only endpoints plus 1 admin create
- `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` — add `/ui/tournaments/` list and `/ui/tournaments/{id}` detail pages with HTMX partials
- `game-environments/game_envs/games/prisoners_dilemma.py` — add `format_state_for_player(round, participant_idx) -> dict` method (or equivalent on the PD game class)
- `migrations/dashboard/versions/xxxx_tournament_mvp_schema.py` — new Alembic migration

**Reused as-is:**

- `game-environments/` core — scoring, action validation, game definitions
- `packages/atp-adapters/atp/adapters/mcp/` — in-house MCPClient used as e2e test participant and as the documented "programmatic Python participant path"
- `JWTUserStateMiddleware` (this session) — SSE handshake auth
- Existing JWT auth dependency `get_current_user`

### High-level flow (one PD tournament)

```
T+0   Admin:    POST /api/v1/tournaments
                → Tournament(id=7, status=PENDING, num_players=2, total_rounds=100, round_deadline_s=30)

T+0+  Player A: MCP join_tournament(7, "alice-tft")
                → Participant(7, alice, seat_index=0)
                → participants < num_players → tournament stays PENDING

T+0+  Player B: MCP join_tournament(7, "bob-random")
                → Participant(7, bob, seat_index=1)
                → participants == num_players → _start_tournament()
                → Tournament.status = ACTIVE
                → Round(1, status="waiting_for_actions", deadline=now+30s)
                → bus.publish(round_started, tournament_id=7, round_number=1)
                → MCP notification layer formats per-player RoundState and
                   pushes notifications/message to both subscribers

T+3   Player A: MCP make_move(7, {"choice": "cooperate"})
                → Action(round=1, alice, source=player) inserted
                → still waiting on B → return {"status": "waiting"}

T+7   Player B: MCP make_move(7, {"choice": "defect"})
                → Action(round=1, bob, source=player) inserted
                → all actions present → _resolve_round(1) SYNC in same txn:
                    * compute payoffs via game-environments PD scoring
                    * write Action.payoff for both actions
                    * Round(1).status = completed, resolved_at = now
                    * Round(2, ...) created with new deadline
                    * commit
                → bus.publish(round_ended, round=1, payoffs=...)
                → bus.publish(round_started, round=2, ...)
                → return {"status": "round_resolved", "your_payoff": 5, ...}

T+37  Deadline worker tick:
                → SELECT rounds WHERE status='waiting_for_actions' AND deadline<now
                → Round(2) expired because player A didn't move
                → force_resolve_round(2):
                    * atomic UPDATE status='resolving' (guard)
                    * fill missing Action for A with default="cooperate", source=timeout_default
                    * _resolve_round(2) — same code path as happy path

...100 rounds later...

T+END Service: _complete_tournament(7)
                → Tournament.status = COMPLETED, ends_at = now
                → Participant.total_score written for all
                → bus.publish(tournament_completed, final_scores=...)
                → both players receive final leaderboard notification
```

### Service layer API

```python
class TournamentService:
    def __init__(self, session: AsyncSession, bus: TournamentEventBus) -> None: ...

    # Admin
    async def create_tournament(self, admin, *, name, game_type, num_players,
                                 total_rounds, round_deadline_s) -> Tournament: ...
    async def cancel_tournament(self, admin, tournament_id) -> None: ...

    # Participant lifecycle
    async def join(self, tournament_id, user, agent_name) -> Participant: ...
    async def leave(self, tournament_id, user) -> None: ...

    # Gameplay
    async def get_state_for(self, tournament_id, user) -> RoundState: ...
    async def submit_action(self, tournament_id, user, action) -> SubmitResult: ...
    async def get_history(self, tournament_id, user, last_n=None) -> list[RoundRecord]: ...

    # Discovery
    async def list_tournaments(self, user, status=None) -> list[Tournament]: ...
    async def get_tournament(self, tournament_id) -> Tournament: ...

    # Internal, called by deadline worker and submit_action
    async def force_resolve_round(self, round_id) -> None: ...

    # Private
    async def _start_tournament(self, tournament_id) -> None: ...
    async def _resolve_round(self, round_id) -> None: ...
    async def _complete_tournament(self, tournament_id) -> None: ...
```

**Invariants:**

- Every public method takes a `User` and performs its own ownership/permission check.
- State-changing methods write to DB and event bus in one logical operation: bus publish happens **after** `session.commit()`, never before.
- The service never touches FastAPI or MCP concerns. It is unit-testable via direct calls with an in-memory session and a test event bus.

### Event bus

```python
@dataclass
class TournamentEvent:
    event_type: Literal["round_started", "round_ended",
                        "tournament_completed", "tournament_cancelled"]
    tournament_id: int
    round_number: int | None
    data: dict          # generic payload, per-player formatting is a subscriber concern
    timestamp: datetime

class TournamentEventBus:
    _subscribers: dict[int, set[asyncio.Queue[TournamentEvent]]]

    async def publish(self, event: TournamentEvent) -> None:
        """Fan-out to all subscribers of event.tournament_id.
        Best-effort: queue-full subscribers are dropped with a warning log.
        Never raises.
        """

    @asynccontextmanager
    async def subscribe(self, tournament_id: int) -> AsyncIterator[asyncio.Queue]:
        """Per-subscriber queue, maxsize=100.
        Registers on enter, removes on exit (including on task cancellation).
        """
```

**Key properties:**

- Events are **ephemeral**. No persistence. Missed events on disconnect are not replayed in v1.
- **Fan-out is per-subscriber, not shared.** Each subscriber has its own `asyncio.Queue` so a slow consumer cannot block others. `maxsize=100` handles the worst case (100 rounds × 2 events/round) with margin.
- **Per-player personalization happens in the subscriber.** The service publishes generic events; the MCP notification subscriber calls `service.get_state_for(user, ...)` to build the player-private RoundState before sending.

### MCP server

FastMCP mounted as a Starlette sub-app under `/mcp`.

**Tools (exposed to clients):**

| Tool | Purpose |
|---|---|
| `list_tournaments(status?)` | Discovery |
| `get_tournament(id)` | Details |
| `join_tournament(id, agent_name)` | Lifecycle + auto-subscribe to events |
| `leave_tournament(id)` | Cancel subscription + participant.leave |
| `get_current_state(id)` | Fetch player-private RoundState |
| `make_move(id, action)` | Submit an action |
| `get_history(id, last_n?)` | Retrieve past rounds for strategies with memory |

**Notifications (server → client):**

All use the standard MCP `notifications/message` method. Payload `data` contains:

| Event | Data |
|---|---|
| `round_started` | `{tournament_id, round_number, deadline_at, state: RoundState}` |
| `round_ended` | `{tournament_id, round_number, your_payoff, opponent_summary}` |
| `tournament_completed` | `{tournament_id, your_final_score, your_rank, leaderboard}` |
| `tournament_cancelled` | `{tournament_id, reason}` |

`deadline_warning` is **not** in v1 — it adds a second timer per round and complicates the deadline worker without clear value for the first tournament. Tracked in backlog as part of richer deadline UX.

**Authentication:**

- SSE handshake `GET /mcp/sse` carries `Authorization: Bearer <jwt>`.
- `JWTUserStateMiddleware` writes `request.state.user_id` on the handshake request (already wired this session).
- A small `MCPAuthMiddleware` in front of FastMCP rejects handshakes without `request.state.user_id` with 401.
- The user_id is stored in session context; tool handlers call `_user_from_context(ctx)` to retrieve the `User` object from the DB.
- No per-tool auth after handshake. No per-tool rate limit in v1 (one move per round is self-limiting).

**Notification delivery mechanism:**

On `join_tournament`, the tool handler spawns a background `asyncio.Task` per `(session_id, tournament_id)` pair:

```python
async def _forward_events_to_session(ctx, tournament_id, user):
    async with bus.subscribe(tournament_id) as queue:
        while True:
            event = await queue.get()
            notification = await _format_notification_for_user(event, user, tournament_id)
            if notification is not None:
                await ctx.session.send_notification(notification)
```

The task is cancelled on `leave_tournament` or session close. Task handles are tracked in a module-level `dict[session_id, dict[tournament_id, Task]]` (acknowledged as mutable global state; acceptable for MVP).

## Persistence

### Existing models

`tournament/models.py` already defines `Tournament`, `Participant`, `Round`, `Action` with relationships and basic indexes from migration `4e902371a941`. v1 adds columns, constraints, and an index; no new tables.

### Schema changes (Alembic migration `xxxx_tournament_mvp_schema.py`)

**Tournament (additive columns):**

| Column | Type | Notes |
|---|---|---|
| `name` | `String(200) NOT NULL` | server_default="" for backfill |
| `num_players` | `Integer NOT NULL` | server_default="2" |
| `total_rounds` | `Integer NOT NULL` | server_default="1" |
| `round_deadline_s` | `Integer NOT NULL` | server_default="30" |

Status enum is reused: `PENDING` = accepting joins, `ACTIVE` = game running, `COMPLETED` / `CANCELLED` unchanged. No rename.

**Participant:**

| Change | Details |
|---|---|
| `seat_index` | new `Integer NULLABLE` — position in tournament (0-indexed), assigned on `_start_tournament` |
| `user_id` | NULL → **NOT NULL** with backfill, matching Issue 1's `Run.user_id` fix for consistency |
| `UniqueConstraint(tournament_id, user_id)` | new — one JWT cannot join twice |

**Round:**

| Change | Details |
|---|---|
| `resolved_at` | new `DateTime NULLABLE` |
| `UniqueConstraint(tournament_id, round_number)` | new — round numbers unique per tournament |
| `Index(status, deadline)` | new — supports `WHERE status='waiting_for_actions' AND deadline<NOW()` for deadline worker |
| `status` values | reused field, expanded to `waiting_for_actions` / `resolving` / `completed` via new `RoundStatus(StrEnum)` |

**Action:**

| Change | Details |
|---|---|
| `source` | new `String(20) NOT NULL` — `player` or `timeout_default`, server_default="player" |
| `payoff` | new `Float NULLABLE` — denormalized per-player payoff, written during `_resolve_round` |
| `UniqueConstraint(round_id, participant_id)` | new — **critical**: one action per player per round. Primary defense against double-submit races. |

### Data integrity invariants guaranteed by the schema

1. A participant cannot join the same tournament twice — `uq_participant_tournament_user`.
2. A participant cannot submit two actions in the same round — `uq_action_round_participant`.
3. A round number is unique within a tournament — `uq_round_tournament_number`.
4. Every action has a non-null owning participant with a non-null user — audit trail for every move.

### Migration safety

- Backfill strategy mirrors `c8d5f2a91234_enforce_run_user_id_not_null.py`: set `server_default` values, copy legacy `user_id IS NULL` rows to a system/admin user or archive them.
- Idempotent — the migration can be re-run after a partial apply without duplicate constraint errors.
- SQLite-compatible (production is SQLite on Namecheap VPS). Alembic handles the rebuild-table dance internally for UNIQUE constraints.

## Error handling

Four error classes with explicit mapping:

| Class | Source | Service exception | MCP surface | REST surface |
|---|---|---|---|---|
| Validation | Invalid shape, missing fields, unknown game_type, action doesn't match game schema | `ValidationError` | `ToolError(422)` | `422` |
| State machine | Join in ACTIVE, make_move in completed round, double move, leave ACTIVE tournament | `ConflictError` | `ToolError(409)` | `409` |
| Not found / not owned | Non-existent resource, or existing resource belonging to another user | `NotFoundError` | `ToolError(404)` | `404` (independent of existence — enumeration guard per Issue 1 pattern) |
| Auth | Missing, expired, invalid JWT | `AuthError` | SSE handshake `401`, connection refused | `401` |

**Rules:**

1. Exceptions are the only error channel in the service. No tuple returns, no sentinels.
2. `IntegrityError` from unique constraints is caught at the service layer and translated to the appropriate `ConflictError`. It never leaks to FastAPI/FastMCP.
3. The deadline worker never raises upward. Per-round errors are logged and the loop continues — one broken round cannot kill enforcement for others.
4. Bus publish is always best-effort. Subscribers with full queues or crashed forwarders are logged, not propagated.
5. 404 for non-owned resources matches the enumeration-guard pattern established by the Issue 1 fix — clients cannot distinguish "doesn't exist" from "exists but not yours".

## Testing strategy

**Unit (~70% of coverage):**

- `test_tournament_service_join.py` — join happy path, full tournament 409, completed tournament 409, double join 409 via UniqueConstraint, auto `_start_tournament` on reaching `num_players`
- `test_tournament_service_actions.py` — valid submit, double submit 409, submit to completed round 409, non-participant 404, last-mover triggers `_resolve_round`, scoring via game-environments PD matrix
- `test_tournament_service_state.py` — `get_state_for` produces correct per-player private view, cumulative scores
- `test_tournament_event_bus.py` — single subscriber receives, multi-subscriber fan-out, queue-full drop + warning, unsubscribe via context manager exit, no-subscriber publish is noop
- `test_deadline_worker.py` — expired round resolves with `timeout_default`, multiple expired rounds all resolve, race guard against concurrent `submit_action` + `force_resolve_round`
- `test_mcp_tools.py` — handlers route to correct service methods with correct arguments; error translation to `ToolError`

**Integration (~20%):**

- `test_mcp_handshake.py` — handshake without / invalid / valid token (401, 401, 200)
- `test_mcp_flow.py` — FastMCP in TestClient end-to-end: `list_tournaments → get_tournament → join → get_current_state → make_move`, verify notifications received
- `test_rest_admin_readonly.py` — `list`, `get`, `rounds` respect ownership scoping
- `test_tournament_schema_migration.py` — applies cleanly on fresh SQLite; applies on a DB containing legacy pre-v1 tournament rows without data loss

**E2E (~10%) — acceptance test for the vertical slice:**

`tests/e2e/test_mcp_pd_tournament.py` spins up a real uvicorn instance on a random port with an ephemeral SQLite database, creates an admin user and 2 participant users, runs a 3-round PD tournament where two `MCPAdapter` bots connect over SSE and play a scripted strategy (alice always cooperates, bob always defects), verifies final scores match the expected payoff matrix, and asserts final DB state (`Tournament.status == COMPLETED`, 3 rounds persisted, all actions recorded). This test is the **acceptance criterion** for the MVP vertical slice — when it passes locally and in CI, the base game loop works through the real transport.

**Explicit non-goals for v1 testing:**

- Load testing (backlog L)
- Concurrent-tournament stress testing (>50 parallel tournaments)
- Adversarial MCP client behavior testing (malformed payloads, protocol violations) — basic validation only

## Rollout plan (vertical slice first)

The vertical slice is deliberately narrower than the full v1 scope to establish an early end-to-end feedback loop:

1. **Slice scope.** PD only, 2 players, 3 rounds (fixed), no deadlines, in-memory round state, MCP tools: `join_tournament`, `get_current_state`, `make_move`; notifications: `round_started`, `tournament_completed` only.
2. **Proof point.** The e2e test above (possibly with 3 rounds instead of 100) passing locally with two `MCPAdapter` bots.
3. **Expand to full v1.** Add deadlines, the full notification set, `leave_tournament`, `get_history`, `list_tournaments`, dashboard, admin REST, migration to persist per-round state.

This lets us validate the FastMCP + SSE + session-scoped subscription mechanics before committing to the full feature surface.

## Deferred / Post-v1 Backlog

Three tiers: **planned next** (known what and when), **deferred with trigger** (revisit when condition fires), **deferred without trigger** (recorded so we don't forget).

### Planned next (after PD is debugged)

| # | Feature | Estimate | Prerequisites |
|---|---|---|---|
| A | **El Farol Bar as second game_type** (N-player infra validation) | ~2-3 days | PD working end-to-end; N-player-ready service layer (free per AD-3) |

### Deferred with explicit trigger

| # | Feature | Revisit when... |
|---|---|---|
| B | Idempotent `make_move` (409 → 200 `already_submitted`) | First public tournament shows real confusion from retry-driven 409s |
| C | Reconnect with event replay (server remembers last N events per player) | >10% of participants miss rounds due to disconnects; metric = Actions with `source=timeout_default` for players who were online between rounds |
| D | Spectator mode + live SSE feed on dashboard | After first public tournament, if community asks to watch matches in real time |
| E | Step-by-step replay UI (scrubber) | Together with spectator mode, or by researcher request |
| F | Chain-of-thought capture per round (persisted `reasoning` on Action) | When analysis needs it, not preemptively |
| G | Charts: score trajectory, heatmap, cooperation rate | After 5+ tournaments, when metric utility is known |
| H | Head-to-head matrix + per-seat breakdown | After first N-player game ships (El Farol) and multi-pair tournaments exist |
| I | Redis-backed event bus + multi-worker uvicorn | p99 `make_move` latency > 500ms OR parallel active tournaments > 50 OR worker RSS > 500MB |
| J | stdio transport as local dev/debug mode | Developer friction during local iteration becomes real |
| K | Per-tool rate limit inside MCP session | Observed botnet-like flood from a single JWT |
| L | Load testing suite | First public tournament with 5+ participants raises performance concerns |
| U | `deadline_warning` notification (T-5s before round expires) | Players complain about missing the deadline with no heads-up; requires second timer per round |

### Deferred without trigger (idea parking lot)

| # | Feature |
|---|---|
| M | Remaining games from `game-environments`: Stag Hunt, Battle of Sexes (near-free, shape ≡ PD), Public Goods, Colonel Blotto, Auction, Congestion |
| N | Tournament scheduling (cron-created tournaments, round schedule visibility) |
| O | Brackets / ELO persistent ranking across tournaments |
| P | Benchmark MCP facade — MCP tools for `list/get/start_run/status/leaderboard` (parallel track, independent of tournaments) |
| Q | Custom payoff matrices set by admin at tournament creation |
| R | Team tournaments (players grouped into teams) |
| S | Cross-game meta-tournaments (composite scoring across game types) |
| T | Stream/demo-friendly large-display dashboard |

### Backlog hygiene rules

- This backlog is the source of truth, cross-referenced from `memory/project_mcp_backlog.md`.
- When a trigger fires, the item moves to an active work list with explicit trigger rationale captured.
- Completed items are marked ✅ DONE and kept in the file for audit trail.
- Review cadence: before every planned tournament, or every ~4 weeks, whichever comes first.

## References

- `docs/atp-mcp-server-for-games.md` — original architectural exploration (586 lines), superseded by this spec
- `docs/atp-issues-ownership-and-buffer.md` — Issue 1 (IDOR, ✅ done) and Issue 2 (SDK buffer loss, Fix A ✅ done)
- `docs/atp-auth-ratelimit-sdk.md` — current auth + rate-limiting architecture (recently updated for per-user rate limits)
- `packages/atp-dashboard/atp/dashboard/tournament/models.py` — existing SQLAlchemy models
- `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py:74-117` — 501 stubs to delete
- `packages/atp-adapters/atp/adapters/mcp/transport.py` — in-house MCP client (stdio + SSE)
- `game-environments/game_envs/games/prisoners_dilemma.py` — flagship game
- `migrations/dashboard/versions/4e902371a941_add_tournament_tables.py` — initial tournament schema
- `migrations/dashboard/versions/c8d5f2a91234_enforce_run_user_id_not_null.py` — pattern for NOT NULL backfill migrations
- FastMCP: https://github.com/jlowin/fastmcp
- MCP specification: https://spec.modelcontextprotocol.io
