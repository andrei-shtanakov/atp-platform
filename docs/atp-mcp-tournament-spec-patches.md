# Patches to `2026-04-10-mcp-tournament-server-design.md`

Конкретные точечные изменения к спеке MCP Tournament Server по итогам
review. Каждый патч можно применить независимо. Порядок — от
блокирующих (1-3) к важным (4-5) к честности оценок (6-7).

Применять в `docs/superpowers/specs/2026-04-10-mcp-tournament-server-design.md`.

---

## Patch 1 — AD-9: Token expiry policy (критично, блокер v1)

**Location**: Architectural decisions section, после AD-8 (после строки
89, перед "## Architecture").

**Insert**:

```markdown
### AD-9: Hard duration cap on tournaments to avoid mid-game JWT expiry

**Decision.** `create_tournament` enforces
`max_duration_s = total_rounds * round_deadline_s` and rejects the
creation request if `max_duration_s > (ATP_TOKEN_EXPIRE_MINUTES - 10) * 60`.
With defaults (`ATP_TOKEN_EXPIRE_MINUTES=60`, `round_deadline_s=30`),
this caps a tournament at `(60 - 10) * 60 / 30 = 100 rounds` worst case
— which matches the flagship PD configuration. No JWT refresh logic
inside the MCP session.

**Why.** JWTs are validated only on SSE handshake
(`JWTUserStateMiddleware` writes `request.state.user_id` once per
connection). Nothing re-validates the token after handshake. If the
token expires mid-tournament, the session context still has a valid
`user_id`, but (a) any future reconnect after drop will fail at
handshake with 401, and (b) per AD-7, dropped sessions lose in-flight
notifications with no replay. The combination is a latent footgun on
any tournament longer than ~50 minutes of wall clock.

A hard duration cap is the smallest change that makes the problem
impossible in v1: the tournament cannot outlive its participants'
tokens by construction. The 10-minute buffer absorbs clock skew,
late joins (token age when the session opens, not when it was issued),
and the time between `create_tournament` and actual gameplay start.

**Rejected alternatives.**

- **Bump `ATP_TOKEN_EXPIRE_MINUTES` globally to 240.** Weakens
  platform-wide security posture for a single feature's convenience.
- **Re-validate JWT on every tool call and return `ToolError(401)`
  near expiry.** Forces participants to reconnect mid-game, losing
  notifications per AD-7. Pushes complexity onto every client.
- **Tournament-scoped tokens with dedicated TTL via
  `POST /auth/tournament-token?tournament_id=X`.** Correct long-term
  solution but requires a new auth path, token revocation on
  cancel/leave, and client-side token swapping. Out of v1 scope —
  tracked as backlog item V with trigger "first request for
  tournaments > 50 minutes wall clock".

**Operational note.** The duration check happens at
`create_tournament` time and produces a `ValidationError`. Admin
creating a tournament sees an immediate 422 with an explicit message
(`"max duration 100 rounds at 30s deadline; got 200 rounds"`) rather
than discovering the problem when a participant's token expires on
round 95.
```

**Also add** to the backlog "Deferred with explicit trigger" table
(after current row U, line ~446):

```markdown
| V | Tournament-scoped JWT (`POST /auth/tournament-token`) with TTL >= tournament max duration | First request to run a tournament that exceeds the AD-9 hard cap (e.g. 200-round game or longer `round_deadline_s`) |
```

**Rationale.** This is the #1 blocker from review. Without this
decision, the first publicly-announced tournament has a non-trivial
chance of stalling on expired tokens, which would be a catastrophic
PR hit. The hard cap is the minimum-viable fix; the backlog item V
captures the path forward without committing to it now.

---

## Patch 2 — AD-10: Matchmaking model for v1 (критично, блокер v1)

**Location**: Architectural decisions section, после AD-9 (после
вставленного Patch 1).

**Insert**:

```markdown
### AD-10: Open-join matchmaking with optional join_token for private tournaments

**Decision.** v1 tournaments are **open-join by default**: any
authenticated user can call `join_tournament(id)` on any `PENDING`
tournament, and the first `num_players` joiners trigger game start.
To support research-group tournaments and controlled match-ups,
`Tournament` gains an optional `join_token: str | None` column; if
non-null, `join_tournament` requires the client to pass the matching
token as an extra parameter, and `ConflictError` is raised otherwise.
The admin receives the token once on creation and distributes it
out-of-band.

**Why.** Pure open-join is unsafe for any non-public tournament —
squatter clients can grab slots intended for specific participants,
or sit in multiple tournaments simultaneously and play only in some
(leaving others to timeout-default every round). A full RBAC/invite
model is overkill for MVP; a shared-secret token is a 30-minute
change that covers 90% of the real use cases: "I want my research
group to play this specific tournament."

**Schema impact.** One additive column on `Tournament`:

| Column | Type | Notes |
|---|---|---|
| `join_token` | `String(64) NULLABLE` | NULL = open-join; non-null = token required |

Token is generated server-side with `secrets.token_urlsafe(32)` on
create. Returned once in the `create_tournament` response body, never
listed in `get_tournament` or REST admin responses (admin who lost
it must cancel and recreate).

**Tool signature change.** `join_tournament(id, agent_name,
join_token: str | None = None)`.

**Rejected alternatives.**

- **Explicit invite list (admin specifies user_ids at create time).**
  Richer model, requires a new `TournamentInvite` table, participant
  lookup by user_id before token exists, and a flow to add/remove
  invitees. Deferred to backlog item W.
- **No access control in v1.** Unacceptable for the planned research
  tournaments where controlled match-ups are the whole point.

**Agent name uniqueness.** `agent_name` remains a free-form display
string; it is NOT unique within a tournament. Two participants may
join with the same `agent_name="tft"` if the underlying user_ids
differ. The tournament UI and leaderboard disambiguate by appending
a seat index or user suffix.
```

**Also add** to the backlog "Deferred without trigger" table (after
current row T, line ~459):

```markdown
| W | Explicit invite list / per-tournament RBAC (admin specifies user_ids at creation, invite accept flow) |
```

**Rationale.** This closes the second review concern. `join_token`
is small enough to fit in v1 but solves the realistic private-match
use case. Open-join is explicitly documented as the default behavior
rather than an implicit assumption discoverable only by reading the
flow diagram.

---

## Patch 3 — Phase 0 verification task: MCPAdapter notification support (критично, скрытый риск)

**Location**: Rollout plan section, перед подзаголовком "1. Slice
scope." (строка ~415).

**Insert**:

```markdown
### Phase 0 — pre-slice verification

Before committing to the vertical slice, run two verification tasks
that de-risk hidden assumptions:

1. **`MCPAdapter` notification capability check.** The in-house MCP
   client at `packages/atp-adapters/atp/adapters/mcp/` was built as
   a tool-calling client to test MCP-server agents, not as a
   subscription client that sits on an SSE connection and processes
   server-pushed `notifications/message` in a long-running loop.
   v1 e2e tests and the documented "programmatic Python participant"
   path both depend on this capability.

   **Verification**: write a throwaway integration test that spins
   up a FastMCP test server that sends a `notifications/message`
   once per second over SSE for 10 seconds, and assert that
   `MCPAdapter` in subscription mode receives and surfaces all 10
   notifications to a callback. If this test cannot be written
   against the current `MCPAdapter` API, expand `MCPAdapter` as
   part of fase 1 — add this as an explicit task in the service
   layer milestone, NOT buried in the e2e test milestone.

   **Failure mode if skipped**: e2e test phase (fase 5) discovers
   mid-sprint that the harness fundamentally can't observe
   notifications, forcing either a pivot to the third-party Python
   `mcp` package (new dep, new code path diverging from documented
   participant path) or a significant `MCPAdapter` rewrite not
   budgeted in any phase.

2. **FastMCP + Starlette mount auth integration.** Before writing
   `TournamentService`, verify that mounting `mcp.sse_app()` under
   `/mcp` in the existing FastAPI app correctly triggers
   `JWTUserStateMiddleware` on the SSE handshake request. FastMCP's
   SSE transport creates Starlette routes that may or may not
   respect the outer FastAPI middleware stack. A 15-minute
   verification: stand up a trivial FastMCP with one tool that
   reads `request.state.user_id`, call it via curl with a test
   JWT, confirm the user_id is populated.

   **Failure mode if skipped**: auth is built in fase 2, discovered
   not to work in fase 5 e2e, and debugging is hampered by the
   middleware/mount-order interaction.

These two checks are 2-4 hours total and gate the decision to
start fase 1 with the assumed architecture intact.
```

**Rationale.** This is the concrete reality-check from review concern
3. The spec currently assumes `MCPAdapter` can do both roles (test
harness + participant path) — that assumption is plausible but
unverified, and a failure mode is a week of rework deep in the
project. 4 hours of verification up front is cheap insurance.

---

## Patch 4 — Session sync on reconnect (fixes commit-publish gap)

**Location**: MCP server section, подраздел "Notification delivery
mechanism", после code-блока `_forward_events_to_session` (строки
~293-305).

**Insert after the code block**:

```markdown
**Automatic state sync on handshake / join.** Immediately after
`join_tournament` succeeds and the event forwarder task starts, the
tool handler sends a synthetic `notifications/message` with
`data = {"event": "session_sync", "state": <full RoundState>}`
before returning the tool result. The client thus receives a
guaranteed initial state snapshot as the first notification on the
session, and SHOULD treat it as the authoritative starting point —
any subsequent `round_started` / `round_ended` events are deltas on
top of it.

This closes a narrow but real gap: between the
`session.commit()` of a state-changing operation (e.g. deadline
worker resolving a round) and the `bus.publish()` call, a subscriber
that just reconnected may register AFTER commit but BEFORE publish,
missing the event entirely. Without `session_sync`, that subscriber
would see a `round_started` for round N+2 without ever having seen
`round_ended` for round N+1, and might hold stale `your_history` in
client-side memory. With `session_sync`, reconnect always produces a
consistent baseline from the database.

Clients SHOULD also fall back to `get_current_state` if they observe
a gap (`round_number` jumping by >1 between consecutive notifications)
or a period of no events longer than `2 * round_deadline_s`.
```

**Rationale.** One paragraph that patches review concern 4 (commit →
publish window) without architectural changes. The `session_sync`
mechanism is ~10 lines of code in the `join_tournament` handler and
makes the whole protocol robust against crash/reconnect without
requiring the full event-replay machinery (backlog C).

---

## Patch 5 — `force_resolve_round` contract clarification

**Location**: Service layer API section, Invariants subsection (строки
~215-218).

**Current text**:

```markdown
**Invariants:**

- Every public method takes a `User` and performs its own ownership/permission check.
- State-changing methods write to DB and event bus in one logical operation: bus publish happens **after** `session.commit()`, never before.
- The service never touches FastAPI or MCP concerns. It is unit-testable via direct calls with an in-memory session and a test event bus.
```

**Replace with**:

```markdown
**Invariants:**

- Every public method takes a `User` and performs its own ownership/permission check.
- State-changing methods write to DB and event bus in one logical operation: bus publish happens **after** `session.commit()`, never before.
- The service never touches FastAPI or MCP concerns. It is unit-testable via direct calls with an in-memory session and a test event bus.
- **Round resolution has exactly one implementation.** `force_resolve_round` is defined as `fill_missing_actions_with_default(round_id); await self._resolve_round(round_id)` — it reuses `_resolve_round` verbatim, never duplicates scoring logic. Both the happy path (triggered by the last participant's `submit_action`) and the deadline path (triggered by the background worker) converge on the same private resolver. Any change to payoff computation or round-state transitions happens in exactly one place.
```

**Rationale.** Review concern 7. Without this invariant, an implementer
might plausibly write two parallel resolvers (one "fast happy path",
one "deadline path with defaults") and introduce subtle scoring bugs
that only manifest in timeout scenarios and surface weeks later. One
sentence of contract prevents this.

---

## Patch 6 — Backlog item I: PostgreSQL honestly coupled to multi-worker

**Location**: Backlog "Deferred with explicit trigger" table, row I
(строка ~442).

**Current text**:

```markdown
| I | Redis-backed event bus + multi-worker uvicorn | p99 `make_move` latency > 500ms OR parallel active tournaments > 50 OR worker RSS > 500MB |
```

**Replace with**:

```markdown
| I | Redis-backed event bus + multi-worker uvicorn + **PostgreSQL migration** (SQLite cannot support concurrent writers across workers; all three changes are one logical rollout) | p99 `make_move` latency > 500ms OR parallel active tournaments > 50 OR worker RSS > 500MB |
```

**Rationale.** Review concern 6. Current item I implies "swap bus and
add workers" is a self-contained change. It isn't: production is
SQLite (confirmed in CLAUDE.md memory `reference_deployment.md`), and
SQLite's single-writer lock means adding a second uvicorn worker just
moves contention from Python-level to filesystem-level without
solving it. Being explicit now prevents a rude surprise when the
trigger fires.

---

## Patch 7 — New backlog item: notification personalization scales N²

**Location**: Backlog "Deferred with explicit trigger" table, в конце
существующего списка (после row U, перед вставленной из Patch 1 row V).

**Insert**:

```markdown
| X | Shared per-event memoization in `_format_notification_for_user` — cache round_number, deadline, game_type, payoff_matrix per event and only recompute per-player `your_history`/`opponent_history` in memory, cutting DB queries from `O(subscribers)` to `O(1)` per event | First game with `num_players >= 8` (current flagship is PD with 2; El Farol introduces 5-10 player tournaments per backlog A) |
```

**Rationale.** Review concern 5. For 2-player PD the N² cost is 4
DB queries per round and invisible. For 10-player El Farol it's 100
queries per round × 100 rounds × 2 events = 20 000 queries per
tournament just for notification formatting. The trigger ("first game
with num_players >= 8") ties the work to the moment it actually
starts mattering, not speculatively to v1.

---

## Application order

If applying interactively, apply in this order to minimize merge
friction:

1. **Patches 1 and 2** (AD-9, AD-10) — additive, go into the
   "Architectural decisions" section in sequence. Patch 1 first
   because Patch 2 references it implicitly (both use the same AD
   numbering convention).
2. **Patch 5** (`force_resolve_round` invariant) — smallest change,
   lowest risk of collision.
3. **Patch 4** (session sync paragraph) — inserted mid-section,
   touches the MCP server subsection only.
4. **Patch 3** (Phase 0 verification) — touches Rollout plan section,
   which is stable and unlikely to conflict.
5. **Patches 6 and 7** (backlog items) — same table, apply together
   to avoid two passes over the backlog section.

All seven patches together expand the spec by ~150 lines and do not
require rewriting any existing content except the small edit in
Patch 5 and Patch 6. They can be applied in one editing session in
15-30 minutes.

## What these patches deliberately do NOT address

The following minor review items are intentionally omitted from this
patch set — they can be applied opportunistically during the next
spec edit, but do not block v1:

- **Line 417 editorial** ("3 rounds instead of 100" phrasing) — pure
  copy-edit, no semantic content.
- **OpenTelemetry spans on `TournamentService`** — observability
  improvement, not a correctness issue. Add to backlog when there's a
  second round of backlog grooming.
- **Tournament cancellation cascade behavior** (what happens to
  ACTIVE subscribers, already-played rounds, running deadline
  checks) — 1-2 paragraphs in the "Architecture" section,
  straightforward to add but not a blocker.
- **`Tournament.expires_at` auto-cancel for stale PENDING tournaments**
  — quality-of-life, not a bug. Add as a backlog trigger item if real
  stale tournaments start accumulating.
- **Backlog item U deadline_warning rationale** — the spec claims it
  needs a "second timer per round" but a `warned_at` column on
  `Round` and one extra predicate in the existing deadline worker
  tick achieves the same thing. Minor correction, non-blocking.

These are worth fixing but not worth delaying the patch pass that
addresses the real risks.
