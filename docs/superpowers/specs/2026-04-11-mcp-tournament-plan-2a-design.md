# MCP Tournament Server — Plan 2a Design

**Status:** design ratified, ready for implementation plan.
**Date:** 2026-04-11.
**Baseline commit:** `2759613` (vertical slice merged).
**Source spec:** `docs/superpowers/specs/2026-04-10-mcp-tournament-server-design.md`.
**Supersedes:** `docs/atp-mcp-tournament-spec-patches.md` (all seven patches already merged into source spec; file retained as historical record only).

---

## Section 1 — Overview & Scope

### Baseline

The vertical slice merged as commit `2759613` (branch `feat/mcp-tournament-vertical-slice`). That slice shipped:

- A 3-round Prisoner's Dilemma tournament playable end-to-end via FastMCP SSE.
- `TournamentService` with `create_tournament`, `join`, `submit_action`, `get_state_for`, plus private `_start_tournament`, `_resolve_round`, `_complete_tournament`.
- Three MCP tools: `join_tournament`, `get_current_state`, `make_move`.
- Two notification types: `round_started`, `tournament_completed`.
- Additive schema columns only; zero schema constraints; zero deadline worker; zero idempotency; zero `session_sync`.
- A single e2e test with two `MCPAdapter` bots completing a PD tournament.
- Two SSE transport fixes in `packages/atp-adapters/atp/adapters/mcp/transport.py` (endpoint-frame parsing and response/notification queue split).

### Goal

Take the vertical slice to a state where **public launch is safe**: enforced schema invariants, automatic timeout-resolution of rounds, REST admin surface for ops and future dashboard, full MCP tool set, idempotent reconnect, and both safety decisions already designed in the source spec (AD-9 hard duration cap, AD-10 matchmaking + one-active-per-user) fully implemented.

### In scope

1. **Schema migration** — 7 DB invariants in one Alembic migration (5 base constraints from source spec §Persistence + 1 AD-10 partial unique index + 1 AD-10 cancel-consistency CHECK constraint) plus 8 additive columns: `Tournament.pending_deadline`, `Tournament.join_token`, `Tournament.cancelled_at`, `Tournament.cancelled_by`, `Tournament.cancelled_reason`, `Tournament.cancelled_reason_detail`, `Participant.released_at`, `Action.source`. Portable via `op.batch_alter_table` across SQLite 3.8+ and PostgreSQL. Verify-then-alter with a probe module (6 data integrity probes; the CHECK constraint does not require a probe — `alembic upgrade` either applies it or fails). Plan 2a also includes a minor refactor pass on vertical slice code: bare string literals for `Round.status` (service.py lines 148, 295, 431) and any equivalent occurrences in tests are replaced with `RoundStatus.<VALUE>` enum references. Scope: roughly 5–10 line changes across 2–3 files. Transparent to PR reviewers that the PR touches existing code for a stated cleanup reason.

2. **Deadline worker** — single asyncio task in the FastAPI lifespan, two-path scan per tick (`waiting_for_actions` rounds with expired deadline; `pending` tournaments past `pending_deadline`), default poll interval 5 seconds configurable via `ATP_DEADLINE_WORKER_POLL_INTERVAL_S`, `WEB_CONCURRENCY=1` startup assertion referencing backlog I, outer try/except around each iteration and inner try/except around each row, hard cancel on shutdown.

3. **Full MCP tool set** — five new tools on top of the three from the vertical slice: `leave_tournament`, `get_history`, `list_tournaments`, `get_tournament`, `cancel_tournament`. Total 8 tools on the MCP surface.

4. **REST admin** — six endpoints:
   - `GET /api/v1/tournaments` — list with filters.
   - `GET /api/v1/tournaments/{id}` — detail.
   - `GET /api/v1/tournaments/{id}/rounds` — rounds and actions.
   - `GET /api/v1/tournaments/{id}/participants` — roster.
   - `POST /api/v1/tournaments` — create; returns `join_token` once for private tournaments.
   - `POST /api/v1/tournaments/{id}/cancel` — cancel.

   **Default visibility policy.** Tournaments default to public open-join (`join_token IS NULL`) unless the creator explicitly passes a token at create time. The list endpoint filters results per caller: non-admin users see all public tournaments plus private tournaments they themselves created or already joined; admin users (`user.is_admin == True`) see all tournaments regardless of visibility. Private tournaments never leak the token value in any response — only a `has_join_token: bool` flag.

5. **Idempotent `join` plus `session_sync`** — indivisible pair, both fully new in Plan 2a. A second `join_tournament` call for the same `(user_id, tournament_id)` returns the existing Participant, re-installs the event subscription task, and sends a synthetic `session_sync` notification carrying the full current `RoundState` as the first event on the session — before the tool result returns. Closes the commit→publish window described in source spec §MCP server.

6. **New notification types** — `round_ended`, `tournament_cancelled`. Together with the existing `round_started`, `tournament_completed` and the new `session_sync`, Plan 2a ships the complete notification set from the source spec.

7. **AD-9 hard duration cap** — `TOURNAMENT_PENDING_MAX_WAIT_S = 300` module constant; new column `Tournament.pending_deadline` computed server-side on create as `now() + TOURNAMENT_PENDING_MAX_WAIT_S`; formula validated in `create_tournament`:
   ```
   TOURNAMENT_PENDING_MAX_WAIT_S + total_rounds × round_deadline_s
       ≤ (ATP_TOKEN_EXPIRE_MINUTES − 10) × 60
   ```
   `ValidationError` otherwise with explicit numbers in the message. The deadline worker's second path auto-cancels pending tournaments past their `pending_deadline` via `cancel_tournament_system(reason='pending_timeout')`. Flagship PD tops out at 90 rounds × 30 seconds with default `ATP_TOKEN_EXPIRE_MINUTES=60`.

8. **AD-10 matchmaking + 1-active-per-user** — new column `Tournament.join_token: String(64) NULLABLE` (NULL = open-join; non-null = clients must present matching token on `join_tournament`, creator receives value once in create response, never echoed in list/get). New column `Participant.released_at: DateTime NULL`. Partial unique index `uq_participant_user_active` as `UNIQUE(user_id) WHERE user_id IS NOT NULL AND released_at IS NULL`. **Universal constraint** — admin users are subject to the same 1-active limit. Multi-tournament testing uses dedicated service accounts. No `user.is_admin` exempt path. `released_at` is set inside the same transaction as the corresponding state change in `_complete_tournament`, `_cancel_impl`, and `leave`. Leave is terminal for that specific tournament by construction (via interaction with `uq_participant_tournament_user`).

### Out of scope — deferred to Plan 2b

- Dashboard UI (HTMX pages `/ui/tournaments/`, `/ui/tournaments/{id}`, admin controls in the web).
- Observability work: OpenTelemetry spans on `TournamentService`, leaderboard panels, tournament metrics in Prometheus.

### Out of scope — deferred to named backlog items

Source spec backlog items (existing letters) plus three new Plan 2a items:

- **C** — Reconnect with event replay. Trigger: more than 10% of participants miss rounds due to disconnects (measured via `Action.source='timeout_default'` for players known to be online between rounds).
- **D** — Spectator mode + live SSE feed on dashboard. Trigger: community asks to watch after the first public tournament.
- **I** — Redis-backed event bus + multi-worker uvicorn + PostgreSQL migration (one logical rollout, not separable). Trigger: p99 `make_move` > 500ms, or parallel active tournaments > 50, or worker RSS > 500MB.
- **L** — Load testing suite. Trigger: first public tournament with 5+ participants raises performance concerns.
- **U** — `deadline_warning` notification T−5s. Trigger: players complain about no heads-up before a deadline.
- **V** — Tournament-scoped JWT (`POST /auth/tournament-token`) with TTL ≥ tournament max duration. Trigger: request for a tournament longer than the AD-9 hard cap at default `round_deadline_s`.
- **W** — Explicit invite list / per-tournament RBAC. Trigger: `join_token` sharing proves insufficient.
- **X** — Shared per-event memoization in `_format_for_user` cutting DB queries from `O(subscribers)` to `O(1)`. Trigger: first game with `num_players ≥ 8`.
- **Y** — Lift `MAX_ACTIVE_TOURNAMENTS_PER_USER = 1` to a higher value or per-user override. Trigger: first credible power-user request.
- **Z** (new in Plan 2a) — Force-eject participant (`DELETE /api/v1/tournaments/{id}/participants/{pid}`). Trigger: first real incident where cancel-whole-tournament is too blunt AND the game-semantics questions (cascade on remaining rounds, `num_players` handling, leaderboard treatment) have been answered.
- **AA** (new in Plan 2a) — Manual round force-resolve (`POST /api/v1/tournaments/{id}/rounds/{n}/force-resolve`). Trigger: first real incident where the deadline worker is verifiably broken and no recovery path exists.
- **AB** (new in Plan 2a) — Poison-row detection in the deadline worker (N consecutive failures on the same row → mark as `failed`, skip in subsequent ticks, emit alert). Trigger: first observed production incident where a single corrupted row blocks deadline processing across multiple ticks.
- **AC** (new in Plan 2a) — Unify audit FK column naming in the tournament module with the dashboard-wide convention. Rename `Tournament.created_by` → `Tournament.created_by_id` (FK column) with a Python-side relationship attribute `created_by: Mapped["User | None"]`; do the same for Plan 2a's new `cancelled_by` column; align with the `SuiteDefinition.created_by_id + created_by` precedent at `packages/atp-dashboard/atp/dashboard/models.py:610`. One-shot convergence PR with migration, backfill, and test updates. Trigger: (a) a second audit FK with `_by` suffix added to any dashboard module, strengthening the precedent ratio from 1/N to 2/N, OR (b) developer frustration at maintaining two conventions surfaces in review comments, OR (c) a Plan 2c+ project adds another dashboard-wide naming refactor making it cheap to bundle.

### Known Limitations

- **90-round flagship cap.** With default `ATP_TOKEN_EXPIRE_MINUTES=60` and `TOURNAMENT_PENDING_MAX_WAIT_S=300`, the formula `(60 − 10) × 60 − 300 = 2700 s / 30 s per round = 90 rounds` caps the flagship PD configuration at 90 rounds × 30 seconds. Longer tournaments require either bumping `ATP_TOKEN_EXPIRE_MINUTES` globally (weakens security posture for one feature's convenience) or waiting for backlog V (tournament-scoped JWT). No mid-tournament token refresh in Plan 2a.

- **Alembic downgrade is best-effort, not tested.** Plan 2a's acceptance criteria cover fresh upgrade and model-schema parity only. The migration's `downgrade()` function is written but not exercised beyond a single round-trip test. Recovery path if a production migration needs reverting is restore-from-backup, not `alembic downgrade`.

- **Leave is terminal per tournament.** A user who calls `leave_tournament` on a specific tournament cannot rejoin it (interaction between `uq_participant_tournament_user` and the `released_at IS NOT NULL` rejection branch in `join`). They can join a different tournament via the released slot.

- **No admin exempt for 1-active limit.** Operators running multiple tournaments from a single human account will hit 409. Use dedicated service accounts (backlog Y trigger if this becomes painful).

- **Private tournaments are opaque pre-join.** Tournaments with non-NULL `join_token` are not visible via `get_tournament` / `list_tournaments` to users who are neither the creator nor an existing participant — even if those users hold the correct token. Pre-join config inspection is not supported; the client must trust the share mechanism. Clients that need to display tournament config to the user before committing should encode it in the share link (e.g. `atp://join?tournament_id=42&token=XYZ&num_players=4&rounds=30`) and verify it matches after `join_tournament` returns. Post-join, `get_tournament` works normally for that participant.

- **`join_token` is not recoverable.** The token is returned once in the 201 response body to `POST /api/v1/tournaments` and never echoed back in any subsequent `GET` response (the column is excluded from serialization; only `has_join_token: bool` is exposed). If the creator loses the token before distributing it, the only recovery is `POST /api/v1/tournaments/{id}/cancel` followed by creating a new tournament with a fresh token. Regeneration is explicitly out of scope — it would introduce token-rotation semantics that complicate the "generated once, distributed once" model without clear value for Plan 2a.

- **Cancel audit attribution under concurrent race.** When a user-initiated cancel (`cancel_tournament`) and a system cancel (`cancel_tournament_system` via deadline worker `pending_timeout` or `leave()` abandoned cascade) target the same tournament within the same tick window, both transactions see `status IN (PENDING, ACTIVE)` in their respective snapshots (SQLAlchemy's SQLite dialect silently drops `FOR UPDATE`; SQLite WAL serializes writes at commit time but does not provide row-level isolation for reads), both pass the step-2 idempotent guard in `_cancel_impl`, both reach step 4 and write audit fields to their session-local `Tournament` instance, and both commit — serialized at the WAL write lock. The status transition itself is deterministic (always `CANCELLED`) and the cascade operations on participants and rounds are idempotent bulk UPDATEs (second pass matches zero rows). Only the **audit attribution** fields (`cancelled_by`, `cancelled_reason`, `cancelled_reason_detail`) reflect whichever writer committed **last** — the second `UPDATE tournaments SET ... WHERE id = ?` overwrites the first without a version check. Both callers receive success, and both publish a `TournamentCancelEvent` — subscribers see two events with conflicting attribution for the same tournament and must treat the second as authoritative. Resolved under backlog item I (PostgreSQL migration), where `FOR UPDATE` becomes a real row-level lock and the second caller observes `status=CANCELLED` at step 2, returning `None` without publishing. Plan 2a accepts the race because: (a) the window is sub-second and requires exact coincidence of admin action and deadline expiry, (b) safety (status, cascades) is guaranteed, only audit attribution is non-deterministic, and (c) backlog I closes the race without a separate fix on SQLite.

### Deployment constraints

- Production must run uvicorn with `--workers 1`. Enforced via a FastAPI startup assertion on `WEB_CONCURRENCY` env var that crashes with a descriptive message referencing backlog I if violated.
- `docker-compose.yml` in the deploy repo must pin `command: uvicorn ... --workers 1` on the `platform` service.
- Production is SQLite with WAL mode. The partial unique index on `released_at` guarantees "1 active per user" correctness without relying on locking semantics; works identically under WAL mode and PostgreSQL.
- Pre-deploy checklist requires running `python -m atp.dashboard.migrations.probes.check_tournament_invariants` against a staging snapshot of production data **before** opening the maintenance window. The deploy playbook includes a probe-failure → resolution table for common cases, reproduced in the migration file's header docstring.

### Acceptance criteria

1. **Two `MCPAdapter` bots run a multi-round PD tournament end-to-end** without losing state across simulated connection drops. Mid-tournament reconnect triggers `session_sync` with the authoritative `RoundState` as the first notification on the reconnected session, and gameplay resumes correctly. CI gate uses **30 rounds** for wall-clock economy (~30–40 seconds with `round_deadline_s=1`). A **90-round** variant exists as a manual benchmark test runnable locally before release to exercise the full AD-9 flagship formula; it is not part of the CI e2e job to keep the e2e stage under its 10-minute budget.

2. **A tournament with one of two players not submitting** resolves via the deadline worker within `poll_interval + 1s` of deadline expiry, with correct payoffs and a `round_ended` notification carrying `source=timeout_default` for the missing action.

3. **AD-9 pending auto-cancel test.** Create a tournament with `num_players=4`, wait `TOURNAMENT_PENDING_MAX_WAIT_S + poll_interval + 1s`, assert tournament status transitions to `CANCELLED` and a `tournament_cancelled` notification is published with `reason='pending_timeout'`.

4. **AD-10 concurrent join test.** Two concurrent `join_tournament` calls for the same `user_id` targeting two *different* tournaments: exactly one succeeds, one raises `ConflictError` with HTTP 409 equivalent. Verifies `uq_participant_user_active` enforces under concurrency.

5. **Idempotent join test.** One user calls `join_tournament` twice in a row on the same tournament — the second call returns the same `Participant` row without error or duplicate, re-installs the forwarder task, and sends a fresh `session_sync` notification.

6. **Idempotent cancel test.** `cancel_tournament` called twice in a row on the same tournament — the second call is either a no-op (200 "already cancelled") or a 404 enumeration guard, never a 500, never a second state transition.

7. **Probe dry-run passes.** `python -m atp.dashboard.migrations.probes.check_tournament_invariants` runs clean against a staging snapshot of production DB immediately before the maintenance window.

8. **REST admin ops path works.** An operator can cancel a stuck tournament via `curl -X POST /api/v1/tournaments/{id}/cancel -H "Authorization: Bearer $TOKEN"` without any MCP client installed, receives 200 with empty body, and the `tournament_cancelled` event is visible in server logs.

9. **Alembic upgrade on fresh SQLite** produces 7 new constraints/indexes (5 base + 1 partial unique index + 1 CHECK constraint), 8 new columns, with zero schema drift from models. Downgrade is implemented but not tested in full (per Known Limitations) — only a single round-trip test ensures `downgrade()` is well-formed.

10. **Regression criterion.** All vertical slice acceptance tests from `tests/e2e/test_mcp_pd_tournament.py` and `tests/unit/dashboard/{tournament,mcp}/` continue to pass after Plan 2a merge.

**Phase 0 carry-over note.** Vertical slice Phase 0 verified that `MCPAdapter` in subscription mode correctly receives `round_started` and `tournament_completed` notifications over SSE (Task 0.1 passed at commit `87054c2`). Plan 2a adds three new notification types: `round_ended`, `tournament_cancelled`, `session_sync`. The e2e test in criterion 1 must exercise all five types and fail explicitly if any are not received by the client. If any new type fails delivery during implementation, it is a Plan 2a blocker — not a feature to defer to Plan 2b.

---

## Section 2 — Architecture & Components

### Module structure

**Files modified** (all exist from vertical slice):

| File | Changes |
|---|---|
| `packages/atp-dashboard/atp/dashboard/tournament/models.py` | Add `Tournament.pending_deadline`, `Tournament.join_token`, `Tournament.cancelled_at`, `Tournament.cancelled_by`, `Tournament.cancelled_reason`, `Tournament.cancelled_reason_detail`, `Participant.released_at`, `Action.source`. Add `uq_participant_user_active` partial unique index. Add `RoundStatus` and `ActionSource` StrEnums. |
| `packages/atp-dashboard/atp/dashboard/tournament/service.py` | Extend `TournamentService` — five new public methods, updated `join` and `create_tournament`, new private `_load_for_auth` and `_cancel_impl`. Refactor bare `Round.status` string literals to `RoundStatus.<VALUE>` references. |
| `packages/atp-dashboard/atp/dashboard/tournament/events.py` | Add `round_ended`, `tournament_cancelled` event types; add `TournamentCancelEvent` dataclass with `__post_init__` validator; add `session_sync` as wire-only type. |
| `packages/atp-dashboard/atp/dashboard/tournament/errors.py` | No new classes (reuse `ConflictError`, `NotFoundError`, `ValidationError`). |
| `packages/atp-dashboard/atp/dashboard/mcp/tools.py` | Add five new MCP tools: `leave_tournament`, `get_history`, `list_tournaments`, `get_tournament`, `cancel_tournament`. Update `join_tournament` for idempotency and `session_sync` emission. |
| `packages/atp-dashboard/atp/dashboard/mcp/notifications.py` | Add `_format_for_user(event, user)` dispatcher. Add formatters for `round_ended`, `tournament_cancelled`, `session_sync`. |
| `packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py` | Implement 6 REST admin endpoints. |
| `packages/atp-dashboard/atp/dashboard/v2/factory.py` | Start deadline worker in `lifespan`; add `WEB_CONCURRENCY` startup assertion. |

**New files:**

| File | Purpose |
|---|---|
| `packages/atp-dashboard/atp/dashboard/tournament/deadlines.py` | Deadline worker loop: `run_deadline_worker`, `_tick`, two-path scanner, shutdown handling. |
| `packages/atp-dashboard/atp/dashboard/tournament/reasons.py` | `CancelReason` StrEnum (`admin_action`, `pending_timeout`, `abandoned`). Single authoritative enum, imported by service, deadline worker, handlers, migration, tests. |
| `packages/atp-dashboard/atp/dashboard/migrations/probes/__init__.py` | Empty. |
| `packages/atp-dashboard/atp/dashboard/migrations/probes/check_tournament_invariants.py` | Probe module with `check_tournament_schema_ready(connection) -> list[str]` and `if __name__ == "__main__":` block. |
| `migrations/dashboard/versions/<hash>_tournament_plan_2a_constraints.py` | The Plan 2a Alembic migration. `down_revision = "028d8a9fdc46"`. Uses `op.batch_alter_table` exclusively for column operations. Calls `check_tournament_schema_ready` as the first step of `upgrade()`. |

### `TournamentService` — extended API

```python
class TournamentService:
    # ── Unchanged from vertical slice ───────────────────────────
    def __init__(self, session: AsyncSession, bus: TournamentEventBus) -> None: ...
    async def get_state_for(self, tournament_id: int, user: User) -> RoundState: ...
    async def submit_action(
        self, tournament_id: int, user: User, action: dict
    ) -> SubmitResult: ...

    # ── Updated signatures ──────────────────────────────────────
    async def create_tournament(
        self,
        creator: User,
        *,
        name: str,
        game_type: str,
        num_players: int,
        total_rounds: int,
        round_deadline_s: int,
        private: bool = False,
    ) -> tuple[Tournament, str | None]:
        """
        Creates a tournament on behalf of `creator` (any authenticated user —
        not admin-only).

        Does NOT auto-join the creator as a Participant. The creator may call
        join_tournament() separately, which is subject to the AD-10
        1-active-per-user constraint like any other join. Use case: "I create
        a tournament for my research group to play while I watch" — creator
        does not occupy an AD-10 slot by creating.

        Validates AD-9 duration cap:
            TOURNAMENT_PENDING_MAX_WAIT_S + total_rounds * round_deadline_s
            <= (ATP_TOKEN_EXPIRE_MINUTES - 10) * 60

        Writes Tournament.pending_deadline = now() + TOURNAMENT_PENDING_MAX_WAIT_S.
        If private=True, generates join_token via secrets.token_urlsafe(32),
        persists, and returns plaintext in the tuple (REST/MCP handler returns
        it once to caller).

        Raises ValidationError on cap violation with explicit numbers in the
        message.
        """

    async def join(
        self,
        tournament_id: int,
        user: User,
        agent_name: str,
        join_token: str | None = None,
    ) -> tuple[Participant, bool]:  # (participant, is_new)
        """
        Idempotent: if a Participant already exists with
        (tournament_id, user.id) AND released_at IS NULL, returns
        (existing, False) without raising.

        If released_at IS NOT NULL (user previously left this tournament),
        raises ConflictError per leave-is-terminal invariant.

        If Tournament.join_token is not NULL, requires matching join_token
        parameter; raises ConflictError on mismatch (constant-time comparison).

        Race semantics on INSERT IntegrityError. The idempotent pre-check
        (SELECT existing Participant) and the INSERT are not atomic under
        SQLite WAL — two concurrent join calls from the same user can both
        pass the pre-check and race on INSERT. The handler must inspect the
        constraint name on IntegrityError and resolve per case:

        - `uq_participant_tournament_user` violation: a concurrent idempotent
          re-join from the same user to the same tournament won the insert
          race. The losing caller re-reads the existing Participant row
          (now committed by the winner) and returns (existing, False). The
          idempotent contract holds under concurrency: both callers see the
          same (participant, False) outcome even though exactly one of them
          actually performed the INSERT.

        - `uq_participant_user_active` violation: the caller has an active
          participation in a different tournament. Raise
          ConflictError(409, 'user already has an active tournament').

        - Any other IntegrityError: re-raise unchanged — unknown constraint
          violations are not silently swallowed.

        DB is the single source of truth for concurrency; both constraints
        are enforced at the partial/unique index level and the service layer
        only interprets which class of race happened after the fact.
        """

    # ── New methods ─────────────────────────────────────────────
    async def leave(self, tournament_id: int, user: User) -> None:
        """
        Finds the caller's Participant. Sets released_at = now().

        Last-participant detection: after setting released_at, counts
        remaining Participants with released_at IS NULL in the same
        transaction. If count == 0 AND Tournament.status == ACTIVE, calls
        _cancel_impl(reason=CancelReason.ABANDONED) inside the same
        transaction before commit.

        Retry semantics: idempotent at the DB level. A caller that retries
        after a successful-but-unacknowledged first attempt receives
        NotFoundError on the retry (released_at != NULL filter). SDK retry
        layers MUST treat NotFoundError after leave() as a terminal success
        signal.
        """

    async def get_history(
        self,
        tournament_id: int,
        user: User,
        last_n: int | None = None,
    ) -> list[RoundRecord]:
        """
        Returns player-personalized rounds. Plan 2a implements PD
        (reveal-after-commit) semantics — all participants see all actions
        once a round resolves. Future game types with hidden-information
        models must implement per-game filtering via
        mcp/notifications.py::_format_for_user (same mechanism as notification
        personalization). No per-tool reveal logic lives here.

        Admins see all actions unconditionally.

        Pagination: last_n bounds to most recent N rounds, default None = all.
        Plan 2a hard-caps last_n at 100 regardless of input (enumeration
        protection).
        """

    async def list_tournaments(
        self,
        user: User,
        status: TournamentStatus | None = None,
    ) -> list[Tournament]:
        """
        Default visibility filter:
        - If user.is_admin: no filter — all tournaments returned.
        - Else: Tournament.join_token IS NULL
                OR Tournament.created_by = user.id
                OR EXISTS (SELECT 1 FROM tournament_participants
                           WHERE tournament_id = Tournament.id
                             AND user_id = user.id).

        status filter applied on top if provided.

        join_token column excluded from serialization everywhere; only
        has_join_token: bool is exposed in response.
        """

    async def get_tournament(self, tournament_id: int, user: User) -> Tournament:
        """
        Same visibility filter as list_tournaments. Not-visible → NotFoundError
        (404 enumeration guard).
        """

    async def cancel_tournament(
        self,
        user: User,
        tournament_id: int,
        reason_detail: str | None = None,
    ) -> None:
        """
        User-facing cancel entry point. Called by REST
        POST /api/v1/tournaments/{id}/cancel and MCP cancel_tournament tool.

        Authorization: caller is authorized if
            tournament.created_by IS NOT NULL AND tournament.created_by == user.id
            OR user.is_admin
        Else NotFoundError (enumeration guard).

        Delegates to _cancel_impl(tournament_id, CancelReason.ADMIN_ACTION,
        cancelled_by=user.id, reason_detail=reason_detail).
        """

    async def cancel_tournament_system(
        self,
        tournament_id: int,
        reason: CancelReason,
        reason_detail: str | None = None,
    ) -> None:
        """
        System-initiated cancel. Called ONLY by deadline worker
        (pending_timeout path) and by service.leave() (abandoned cascade).
        Not exposed via any HTTP or MCP surface.

        Code-review invariant: no handler file imports this method. Enforced
        by tests/unit/dashboard/tournament/test_static_guards.py grep test.

        No authorization check (caller is trusted).

        Delegates to _cancel_impl(tournament_id, reason, cancelled_by=None,
        reason_detail=reason_detail).
        """

    # ── Private ─────────────────────────────────────────────────
    async def _load_for_auth(
        self,
        tournament_id: int,
        user: User,
    ) -> Tournament:
        """
        Load tournament and verify that `user` is authorized to act on it.

        Authorization rule:
        - Admins (user.is_admin): always allowed.
        - Owners (tournament.created_by == user.id): allowed.
        - Legacy with no owner (tournament.created_by IS NULL): admin only.
        - Everyone else: denied.

        All denial cases raise NotFoundError — the same exception that
        "tournament does not exist" raises. This preserves the
        enumeration-guard invariant: an unauthorized caller cannot
        distinguish between "tournament doesn't exist" and "tournament
        exists but you're not allowed".

        Called BEFORE opening session.begin(). The load is an unlocked SELECT;
        if authorized, the caller opens its own transaction and re-acquires
        the row with with_for_update for the mutation.
        """

    async def _cancel_impl(
        self,
        tournament_id: int,
        reason: CancelReason,
        cancelled_by: int | None,
        reason_detail: str | None,
    ) -> TournamentCancelEvent | None:
        """
        Shared cancellation logic. Single source of truth.

        Mutates DB state but does NOT commit — caller owns the transaction.
        Does NOT publish to bus — returns the event for the caller to publish
        after its commit succeeds.

        Returns None if the tournament was already in a terminal state
        (idempotent no-op); returns a TournamentCancelEvent if the call
        caused a state transition.

        Caller contract:
        - Must have an open `async with self.session.begin()` block active.
        - Must publish the returned event to the bus AFTER the transaction
          commits, wrapped in try/except that logs at WARN and swallows
          (bus publish failures after commit do not fail the operation —
          session_sync on subscriber reconnect closes the gap).

        Steps:
        1. Lock+load tournament via session.get(..., with_for_update=True).
        2. Idempotent guard: if already CANCELLED or COMPLETED, return None.
        3. Compute final_rounds_played = COUNT(Round WHERE status=COMPLETED).
           MUST happen before step 5 — otherwise the bulk UPDATE there would
           inflate the count.
        4. Set Tournament.status, cancelled_at, cancelled_by, cancelled_reason,
           cancelled_reason_detail.
        5. Bulk UPDATE all Participants of this tournament with
           released_at IS NULL to released_at = now().
        6. Bulk UPDATE all Rounds of this tournament with
           status IN (WAITING_FOR_ACTIONS, IN_PROGRESS) to status = CANCELLED.
        7. Return TournamentCancelEvent built from the pre-step-5 snapshot.
        """
```

**Invariants** (additions to source spec §Service layer invariants):

- **Cancel has exactly two public entry points.** `cancel_tournament` (user-authenticated) and `cancel_tournament_system` (no auth, internal callers only). Both delegate to a single private `_cancel_impl`. A static test (`tests/unit/dashboard/tournament/test_static_guards.py`) greps `packages/atp-dashboard/atp/dashboard/{mcp,v2/routes}` for `cancel_tournament_system` and asserts zero matches.

- **Idempotent cancel.** `_cancel_impl` is a no-op if the tournament is already `CANCELLED` or `COMPLETED`. Does not raise, does not double-publish. Acceptance criterion 6 exercises this.

- **Leave is terminal per tournament.** The `(tournament_id, user_id)` UNIQUE constraint combined with the explicit rejection of rejoin-with-released_at in `join` makes rejoin impossible. Documented in the `leave_tournament` tool description for clients.

- **Single-`session.begin()` contract.** Every state-changing `TournamentService` method wraps its DB work in `async with self.session.begin():`. On normal exit this commits; on exception this rolls back. After the method returns (normally or via exception), the session is in a clean state ready for the next method call. This invariant enables the deadline worker to reuse one session across both scan paths per tick without explicit rollback handling.

- **Action creation sets `source` explicitly.** Every `session.add(Action(...))` call site must set `source`. Happy-path creation in `submit_action` sets `source=ActionSource.SUBMITTED`. Deadline-path creation in `force_resolve_round` (its `fill_missing_actions_with_default` step) sets `source=ActionSource.TIMEOUT_DEFAULT`. The column is `nullable=False` — SQLAlchemy raises `IntegrityError` if either call site forgets, and the test suite catches this before merge. No code path should rely on the server_default; that default exists purely for historical row backfill during migration.

### Event bus extensions

```python
class TournamentEventType(StrEnum):
    ROUND_STARTED = "round_started"            # existing
    TOURNAMENT_COMPLETED = "tournament_completed"  # existing
    ROUND_ENDED = "round_ended"                # new
    TOURNAMENT_CANCELLED = "tournament_cancelled"  # new
    SESSION_SYNC = "session_sync"              # new — wire-only
    # SESSION_SYNC is a wire-format notification type only. It is NOT routed
    # through TournamentEventBus. The join_tournament tool handler synthesizes
    # it directly via ctx.session.send_notification, bypassing the bus and
    # _format_for_user. Included in this enum so clients can pattern-match on
    # a single type space for all tournament notifications regardless of
    # routing path.
```

**Pre-wrapper payloads** (the `event` discriminator field is added by `_format_for_user` on the wire):

```python
# round_ended — published by _resolve_round after session.commit
{
    "round_number": int,
    "actions_by_participant": {
        participant_id: {
            **action_data,
            "source": "submitted" | "timeout_default",  # from Action.source
        },
    },
    "payoffs_by_participant": {participant_id: float},
    "total_scores_by_participant": {participant_id: float},
}

# tournament_cancelled — published by _cancel_impl caller after commit
#
# cancelled_by is set to None for non-admin recipients by _format_for_user
# (matches REST response privacy rules).
{
    "reason": "admin_action" | "pending_timeout" | "abandoned",
    "reason_detail": str | None,
    "cancelled_by": int | None,        # None for non-admin recipients
    "final_rounds_played": int,
}

# session_sync — synthesized directly by join_tournament tool handler;
# already has the `event` field because it bypasses _format_for_user.
{
    "event": "session_sync",
    "state": <full RoundState from get_state_for(tournament_id, user)>,
}
```

**`TournamentCancelEvent` dataclass with `__post_init__` validator:**

```python
from dataclasses import dataclass
from datetime import datetime

from atp.dashboard.tournament.models import TournamentStatus
from atp.dashboard.tournament.reasons import CancelReason


_SYSTEM_CANCEL_REASONS: frozenset[CancelReason] = frozenset({
    CancelReason.PENDING_TIMEOUT,
    CancelReason.ABANDONED,
})


@dataclass(frozen=True)
class TournamentCancelEvent:
    """Payload for `tournament_cancelled` bus event.

    Field invariant (enforced three ways — defense in depth):

    1. DB CHECK constraint `ck_tournament_cancel_consistency` on the
       `tournaments` table (Section 3 migration).
    2. `__post_init__` validator on this dataclass (raises ValueError on
       construction).
    3. Construction call site in `_cancel_impl` — always builds from
       consistent inputs by construction.

    Invariant:
        cancelled_by IS NULL  ⟺  cancelled_reason ∈ {PENDING_TIMEOUT, ABANDONED}
        cancelled_by NOT NULL ⟺  cancelled_reason == ADMIN_ACTION

    UI code can render "cancelled by system" vs "cancelled by user X" from
    `cancelled_by` alone without consulting `cancelled_reason`.
    """

    tournament_id: int
    cancelled_at: datetime
    cancelled_by: int | None
    cancelled_reason: CancelReason
    cancelled_reason_detail: str | None
    final_rounds_played: int
    final_status: TournamentStatus

    def __post_init__(self) -> None:
        is_system = self.cancelled_reason in _SYSTEM_CANCEL_REASONS
        has_actor = self.cancelled_by is not None

        if is_system and has_actor:
            raise ValueError(
                f"system cancel (reason={self.cancelled_reason.value}) "
                f"must have cancelled_by=None, got {self.cancelled_by}"
            )
        if not is_system and not has_actor:
            raise ValueError(
                f"user-initiated cancel (reason={self.cancelled_reason.value}) "
                f"must have cancelled_by set, got None"
            )
        if self.final_status != TournamentStatus.CANCELLED:
            raise ValueError(
                f"TournamentCancelEvent.final_status must be CANCELLED, "
                f"got {self.final_status.value}"
            )
```

### Notification personalization model

Bus events carry generic payloads — every subscriber sees the same bus-level event. Personalization happens at the wire layer in `packages/atp-dashboard/atp/dashboard/mcp/notifications.py::_format_for_user(event, user) -> dict | None`. The formatter takes the bus event and the recipient's `User`, and returns either the personalized `data` dict to send via `session.send_notification`, or `None` if the event is not deliverable to this user.

**Three responsibilities of `_format_for_user`:**

1. **Add discriminator.** Every outgoing payload gets `data["event"] = <TournamentEventType value>` as its first field, so clients can route on a single key. Bus events that carry the type separately don't include it in the raw payload schemas above.

2. **Game-specific reveal semantics.** For Plan 2a (PD only), `round_ended` is near-identity — all fields visible to all participants. Future games with hidden-information models override per-game reveal rules here.

3. **Privacy filtering for admin-only fields.** `cancelled_by` on `tournament_cancelled` events is set to `None` in the outgoing payload unless `user.is_admin`. Matches REST response serialization, which hides `cancelled_by` from non-admins.

`session_sync` bypasses the bus and therefore bypasses `_format_for_user`. The `join_tournament` tool handler calls `service.get_state_for(tournament_id, user)` — already personalized at the service layer — and sends that state directly.

### Authorization matrix

| Surface | Operation | Authorization rule | On denial |
|---|---|---|---|
| REST | `GET /api/v1/tournaments` | Authenticated user; visibility filter applied per `user.is_admin` | — (filter hides, never denies) |
| REST | `GET /api/v1/tournaments/{id}` | Authenticated user; same visibility filter | `404 NotFoundError` (enumeration guard) |
| REST | `GET /api/v1/tournaments/{id}/rounds` | Authenticated user; same visibility filter | `404 NotFoundError` |
| REST | `GET /api/v1/tournaments/{id}/participants` | Authenticated user; same visibility filter | `404 NotFoundError` |
| REST | `POST /api/v1/tournaments` | Authenticated user (any — not admin-only) | — (no service-level authz denial path; authentication is enforced by middleware before the handler, not by the service method) |
| REST | `POST /api/v1/tournaments/{id}/cancel` | `user.id == tournament.created_by` (NULL-safe) OR `user.is_admin` | `404 NotFoundError` (enumeration guard, NOT 403) |
| MCP | `join_tournament` | Authenticated user; `join_token` required if `tournament.join_token IS NOT NULL` | `ToolError(409_conflict)` on wrong token, `ToolError(409_conflict)` on AD-10 violation |
| MCP | `leave_tournament` | Participant must exist in tournament | `ToolError(404_not_found)` if not a participant |
| MCP | `get_history` | Same visibility rule as `get_tournament` | `ToolError(404_not_found)` |
| MCP | `list_tournaments` | Authenticated user; same filter as REST list | — |
| MCP | `get_tournament` | Same visibility rule | `ToolError(404_not_found)` |
| MCP | `cancel_tournament` | Same rule as REST cancel (symmetry invariant) | `ToolError(404_not_found)` |
| MCP | `get_current_state`, `make_move` | Unchanged from vertical slice — see source spec §MCP server. Plan 2a does not modify these tools. | — |

**Invariant:** the authorization rule for cancel is defined **once** in `TournamentService.cancel_tournament` (via `_load_for_auth`), and both REST and MCP handlers are thin wrappers that call it. Neither handler implements its own check. This guarantees REST and MCP can never drift in cancel semantics.

### REST admin handler contract

All six REST endpoints are thin wrappers. Handler responsibilities:

1. Parse request.
2. Call the service method.
3. Serialize the response.
4. Map service exceptions to HTTP status codes via existing `atp.dashboard.v2.routes.error_handlers`.

**Serialization rules (applied globally to every tournament-shaped response):**

- `join_token` — **never** serialized. Column excluded from Pydantic response models.
- `has_join_token` — computed `bool(tournament.join_token)`, always present.
- `released_at` on Participant — serialized as ISO8601 when the row belongs to the current user OR the caller is admin; omitted otherwise. Implementation: the list-of-participants serializer walks each row and includes `released_at` only when `row.user_id == current_user.id or current_user.is_admin`. This lets a participant see their own leave-time for client-side debugging while protecting opsec on other players' leave-times.
- `cancelled_reason` and `cancelled_reason_detail` — serialized for all visibility-eligible users.
- `cancelled_by` — serialized only for admins.

### Cancel twin-methods visual contract

```
 REST POST /tournaments/{id}/cancel          MCP cancel_tournament tool
          │                                           │
          └──────────────┬────────────────────────────┘
                         ▼
         TournamentService.cancel_tournament(user, ...)
                 │ _load_for_auth(tournament_id, user):
                 │   404 NotFoundError if unauthorized (enumeration guard)
                 ▼
         _cancel_impl(tournament_id, CancelReason.ADMIN_ACTION, user.id, ...)
                         ▲
                         │ (no auth, trusted callers only)
                         │
         ┌───────────────┴─────────────────────┐
         │                                     │
 deadline_worker._tick()              service.leave() last-participant cascade
 (pending_timeout path)               (abandoned cascade)
         │                                     │
         ▼                                     ▼
 cancel_tournament_system(              cancel_tournament_system(
     id,                                      id,
     reason=PENDING_TIMEOUT                   reason=ABANDONED
 )                                       )
```

Static invariant enforced by `tests/unit/dashboard/tournament/test_static_guards.py`: a grep over `packages/atp-dashboard/atp/dashboard/{mcp,v2/routes}/**/*.py` for `cancel_tournament_system` must return zero matches. Any handler-side usage fails CI in the unit stage.

### `leave_tournament` on last active participant

When the **last remaining active participant** calls `leave_tournament` on an `ACTIVE` tournament, the service detects the zero-remaining state and cascades to `_cancel_impl(reason=CancelReason.ABANDONED)` inside the same transaction as the `released_at` write.

**Rationale:**
- "Leave succeeds, tournament stays ACTIVE with zero players" would create a zombie state requiring a third scan path in the deadline worker and publishing wasted `round_ended` events to zero subscribers.
- "Leave refuses with `ConflictError` if caller is the last participant" is user-hostile — a participant's client crash or deliberate exit shouldn't leave the platform with no graceful escape.
- The chosen cascade is single-atomic-transition: the participant's `released_at` write and the tournament's `CANCELLED` transition commit together, `tournament_cancelled` with `reason=abandoned` publishes after commit, any subscribers watching via `get_current_state` see the correct terminal state on their next poll.

**Boundary.** The cascade fires **only** when the leaving participant is the last one with `released_at IS NULL` **and** the tournament is in `ACTIVE` status. Tournaments in `PENDING` status that reach zero participants do **not** cascade — they sit and wait for joiners until their `pending_deadline` fires via the deadline worker's Path 2 (`pending_timeout`).

**Race with concurrent `join_tournament`.** The last-participant detection performs `SELECT COUNT(*) ... WHERE released_at IS NULL` inside the same transaction as the `released_at = now()` write. If a concurrent `join_tournament()` from a new user commits **before** the leaving user's transaction, the count returns ≥ 1 and the cascade does not fire — the tournament survives with the new joiner. If the concurrent join commits **after** the leaving user's transaction, the cascade fires first (count = 0), the tournament transitions to `CANCELLED`, and the joining user's `join_tournament()` subsequently fails with `ConflictError`. **This race is intentional** — the last-leaver detection is a UX optimization, not a safety invariant. Either outcome is acceptable. No special handling, no retry loop, no sentinel state.

---

## Section 3 — Persistence: Migration, Probe Module, Schema Deltas

### Plan 2a invariants (7 total)

1. `uq_participant_tournament_user` (`Participant`) — `UNIQUE(tournament_id, user_id)`.
2. `uq_action_round_participant` (`Action`) — `UNIQUE(round_id, participant_id)`.
3. `uq_round_tournament_number` (`Round`) — `UNIQUE(tournament_id, round_number)`.
4. `idx_round_status_deadline` (`Round`) — composite index on `(status, deadline)`, supports deadline worker Path 1 scan.
5. `Participant.user_id NOT NULL` — flip from nullable.
6. `uq_participant_user_active` (`Participant`) — partial unique index on `(user_id) WHERE user_id IS NOT NULL AND released_at IS NULL`.
7. `ck_tournament_cancel_consistency` (`Tournament`) — CHECK constraint enforcing the `cancelled_by ⟺ cancelled_reason` invariant.

**8 additive columns:**

| Table | Column | Type | Purpose |
|---|---|---|---|
| `tournaments` | `pending_deadline` | `DateTime NOT NULL` | AD-9 pending auto-cancel deadline |
| `tournaments` | `join_token` | `String(64) NULL` | AD-10 private tournament token |
| `tournaments` | `cancelled_at` | `DateTime NULL` | Cancel audit |
| `tournaments` | `cancelled_by` | `Integer NULL FK users.id ON DELETE SET NULL` | Cancel audit |
| `tournaments` | `cancelled_reason` | `String(32) NULL` | Cancel audit (stores `CancelReason` value) |
| `tournaments` | `cancelled_reason_detail` | `String(512) NULL` | Free-text admin reason |
| `tournament_participants` | `released_at` | `DateTime NULL` | AD-10 slot release |
| `tournament_actions` | `source` | `String(32) NOT NULL DEFAULT 'submitted'` | Audit: submitted vs timeout_default |

### Model deltas

Applied to `packages/atp-dashboard/atp/dashboard/tournament/models.py`. Existing classes shown with `...` for unchanged fields.

```python
from enum import StrEnum

from sqlalchemy import (
    DateTime, ForeignKey, Index, Integer, JSON, String, UniqueConstraint, text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship


class TournamentStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RoundStatus(StrEnum):
    """Round lifecycle status.

    WAITING_FOR_ACTIONS, IN_PROGRESS, COMPLETED existed as bare string
    literals in vertical slice service.py. Plan 2a introduces this StrEnum
    for type safety and adds CANCELLED as a new value used by _cancel_impl
    step 6 to transition in-flight rounds when their tournament is
    cancelled.

    Stored as plain String(20) in the DB without a native enum type or
    CHECK constraint. Adding values is a Python-side-only change on SQLite
    today. Under backlog item I (PG migration), if that migration chooses
    to convert status columns to native PG ENUM types, the DDL must
    enumerate all four values.
    """
    WAITING_FOR_ACTIONS = "waiting_for_actions"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ActionSource(StrEnum):
    """Origin of an Action row.

    SUBMITTED — player sent make_move via MCP tool before deadline.
    TIMEOUT_DEFAULT — deadline worker force_resolve_round created a default
    action for a participant who did not submit before the round deadline.

    Stored as plain String(32) without a native enum type or CHECK
    constraint (matches RoundStatus pattern). StrEnum instances are
    transparently string-compatible in Python 3.11+, so direct assignment
    to the SQLAlchemy column works without .value.
    """
    SUBMITTED = "submitted"
    TIMEOUT_DEFAULT = "timeout_default"


class Tournament(Base):
    """A tournament definition for game-theoretic evaluation."""

    __tablename__ = "tournaments"

    # ... existing columns unchanged, INCLUDING:
    # - created_by: Mapped[int | None] (from vertical slice, line 60 of
    #   models.py at commit 2759613). Plan 2a relies on this column for
    #   cancel authorization but does NOT modify it. New Plan 2a tournaments
    #   always populate it via TournamentService.create_tournament(creator=...).

    # Audit FK columns. Note: this module deviates from the SuiteDefinition
    # precedent (created_by_id + created_by relationship) — vertical slice
    # used `created_by` as the FK column name directly. Plan 2a preserves
    # that and names the new cancelled column symmetrically. Future
    # unification with dashboard convention is tracked as backlog item AC.

    # NEW — AD-9
    pending_deadline: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # NEW — AD-10
    join_token: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # NEW — cancel audit
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cancelled_by: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    cancelled_reason: Mapped[CancelReason | None] = mapped_column(
        sa.Enum(CancelReason, native_enum=False, length=32),
        nullable=True,
    )
    cancelled_reason_detail: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )

    __table_args__ = (
        Index("idx_tournaments_status", "status"),
        Index("idx_tournaments_tenant", "tenant_id"),
        Index(
            "idx_tournaments_status_pending_deadline",
            "status", "pending_deadline",
        ),  # composite, supports deadline worker Path 2 scan
    )


class Participant(Base):
    __tablename__ = "tournament_participants"

    # ... existing columns unchanged, EXCEPT: ...

    # CHANGED from nullable=True to nullable=False
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )

    # NEW — AD-10
    released_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_participant_tournament", "tournament_id"),
        Index("idx_participant_user", "user_id"),
        UniqueConstraint(
            "tournament_id", "user_id",
            name="uq_participant_tournament_user",
        ),
        Index(
            # uq_ prefix: semantically a unique constraint, implemented as a
            # partial unique index because SQLAlchemy UniqueConstraint does
            # not accept WHERE clauses and neither SQLite nor PostgreSQL
            # support partial UNIQUE in CREATE TABLE syntax.
            "uq_participant_user_active",
            "user_id",
            unique=True,
            sqlite_where=text("user_id IS NOT NULL AND released_at IS NULL"),
            postgresql_where=text("user_id IS NOT NULL AND released_at IS NULL"),
        ),
    )


class Round(Base):
    __tablename__ = "tournament_rounds"

    # ... existing columns unchanged, INCLUDING:
    # - deadline: Mapped[datetime] (from vertical slice). Plan 2a uses
    #   this column in the deadline worker Path 1 scan and adds
    #   idx_round_status_deadline below to speed up that query. The
    #   column itself is not modified.

    __table_args__ = (
        Index("idx_round_tournament", "tournament_id"),
        UniqueConstraint(
            "tournament_id", "round_number",
            name="uq_round_tournament_number",
        ),
        Index("idx_round_status_deadline", "status", "deadline"),
    )


class Action(Base):
    __tablename__ = "tournament_actions"

    # ... existing columns unchanged ...

    # NEW — audit trail for timeout-default vs player-submitted
    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="submitted",  # historical vertical slice rows
    )

    __table_args__ = (
        Index("idx_action_round", "round_id"),
        UniqueConstraint(
            "round_id", "participant_id",
            name="uq_action_round_participant",
        ),
    )
```

**`packages/atp-dashboard/atp/dashboard/tournament/reasons.py`:**

```python
"""Cancellation reason enum. Single source of truth, imported by service,
deadline worker, handlers, models, and tests."""

from enum import StrEnum


class CancelReason(StrEnum):
    ADMIN_ACTION = "admin_action"
    PENDING_TIMEOUT = "pending_timeout"
    ABANDONED = "abandoned"
```

### Probe module

`packages/atp-dashboard/atp/dashboard/migrations/probes/check_tournament_invariants.py`:

```python
"""Pre-migration probe for Plan 2a tournament schema invariants.

Exposes `check_tournament_schema_ready(connection) -> list[str]` for use
from the Alembic upgrade() step and from the __main__ block below.

Returns a list of violation descriptions. Empty list = safe to migrate.
Any non-empty return = migration MUST abort (not silently continue).

Usage from CLI (pre-deploy staging check):

    python -m atp.dashboard.migrations.probes.check_tournament_invariants

Reads ATP_DATABASE_URL from env. Exits 0 if clean, exits 1 with violation
list if not.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

from sqlalchemy import Connection, create_engine, text


def _rows(conn: Connection, sql: str) -> Iterable[tuple]:
    return conn.execute(text(sql)).all()


def check_tournament_schema_ready(connection: Connection) -> list[str]:
    """Run all probes; return list of human-readable violation descriptions."""
    violations: list[str] = []

    # Probe 1: Participant.user_id NOT NULL precondition
    rows = _rows(
        connection,
        "SELECT COUNT(*) FROM tournament_participants WHERE user_id IS NULL",
    )
    null_user_id_count = rows[0][0]
    if null_user_id_count > 0:
        violations.append(
            f"P1: {null_user_id_count} tournament_participants rows have "
            f"user_id IS NULL. Plan 2a requires user_id NOT NULL. "
            f"Resolution: DELETE the anonymous rows or backfill them to "
            f"known user_ids before re-running upgrade."
        )

    # Probe 2: FK orphan check (SQLite default does not enforce FK).
    # Plan 2a applies the analogous NOT NULL invariant to
    # tournament_participants.user_id that c8d5f2a91234 applied to
    # benchmark_runs.user_id. tournament_participants is a sibling table
    # not touched by the IDOR migration; this probe verifies a fresh
    # invariant, not a re-check of earlier work.
    rows = _rows(
        connection,
        """
        SELECT COUNT(*) FROM tournament_participants p
        WHERE p.user_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM users u WHERE u.id = p.user_id)
        """,
    )
    orphan_count = rows[0][0]
    if orphan_count > 0:
        violations.append(
            f"P2: {orphan_count} tournament_participants rows reference a "
            f"user_id that does not exist in the users table. This is an "
            f"FK integrity violation that SQLite silently allows. "
            f"Resolution: DELETE FROM tournament_participants WHERE user_id "
            f"NOT IN (SELECT id FROM users)."
        )

    # Probe 3: uq_participant_tournament_user precondition
    rows = _rows(
        connection,
        """
        SELECT tournament_id, user_id, COUNT(*) as cnt
        FROM tournament_participants
        WHERE user_id IS NOT NULL
        GROUP BY tournament_id, user_id
        HAVING COUNT(*) > 1
        """,
    )
    dup_participant_rows = list(rows)
    if dup_participant_rows:
        examples = ", ".join(
            f"(tournament={t}, user={u}, count={c})"
            for t, u, c in dup_participant_rows[:5]
        )
        violations.append(
            f"P3: {len(dup_participant_rows)} (tournament_id, user_id) pairs "
            f"have duplicate participant rows. Examples: {examples}. "
            f"Plan 2a requires uq_participant_tournament_user. "
            f"Resolution: manually deduplicate, keeping the row with the "
            f"earliest joined_at."
        )

    # Probe 4: uq_action_round_participant precondition
    rows = _rows(
        connection,
        """
        SELECT round_id, participant_id, COUNT(*) as cnt
        FROM tournament_actions
        GROUP BY round_id, participant_id
        HAVING COUNT(*) > 1
        """,
    )
    dup_action_rows = list(rows)
    if dup_action_rows:
        examples = ", ".join(
            f"(round={r}, participant={p}, count={c})"
            for r, p, c in dup_action_rows[:5]
        )
        violations.append(
            f"P4: {len(dup_action_rows)} (round_id, participant_id) pairs "
            f"have duplicate action rows. Examples: {examples}. "
            f"Plan 2a requires uq_action_round_participant. "
            f"Resolution: manually deduplicate."
        )

    # Probe 5: uq_round_tournament_number precondition
    rows = _rows(
        connection,
        """
        SELECT tournament_id, round_number, COUNT(*) as cnt
        FROM tournament_rounds
        GROUP BY tournament_id, round_number
        HAVING COUNT(*) > 1
        """,
    )
    dup_round_rows = list(rows)
    if dup_round_rows:
        examples = ", ".join(
            f"(tournament={t}, round={r}, count={c})"
            for t, r, c in dup_round_rows[:5]
        )
        violations.append(
            f"P5: {len(dup_round_rows)} (tournament_id, round_number) pairs "
            f"have duplicate round rows. Examples: {examples}. "
            f"Plan 2a requires uq_round_tournament_number."
        )

    # Probe 6: uq_participant_user_active precondition.
    # Only participants in pending/active tournaments count toward the
    # invariant. Participants in completed/cancelled tournaments are
    # backfilled to released_at=CURRENT_TIMESTAMP by migration step 5a
    # before the partial unique index is created, so they do not need to
    # be probed.
    rows = _rows(
        connection,
        """
        SELECT p.user_id, COUNT(*) as cnt
        FROM tournament_participants p
        JOIN tournaments t ON p.tournament_id = t.id
        WHERE p.user_id IS NOT NULL
          AND t.status IN ('pending', 'active')
        GROUP BY p.user_id
        HAVING COUNT(*) > 1
        """,
    )
    dup_active_rows = list(rows)
    if dup_active_rows:
        examples = ", ".join(
            f"(user={u}, count={c})" for u, c in dup_active_rows[:5]
        )
        violations.append(
            f"P6: {len(dup_active_rows)} users are currently in more than "
            f"one pending/active tournament. Examples: {examples}. "
            f"Plan 2a enforces 1-active-per-user via "
            f"uq_participant_user_active. Resolution: transition stale "
            f"tournaments to completed/cancelled status directly via SQL "
            f"(the migration's backfill step will then handle released_at "
            f"automatically), or DELETE stale participant rows before "
            f"re-running upgrade. Do NOT attempt to set released_at "
            f"directly — the column does not exist at probe time."
        )

    return violations


def _main() -> int:
    # Fail-loud on unset env var: this is a pre-deploy safety tool. A
    # cwd-relative default (e.g. "sqlite:///./atp.db") could silently
    # open an empty database in a staging or CI environment, report
    # "OK", and give the operator false confidence that their check
    # passed. Require explicit configuration.
    db_url = os.environ.get("ATP_DATABASE_URL")
    if not db_url:
        print(
            "FAIL: ATP_DATABASE_URL environment variable is not set. "
            "Pre-deploy probe requires an explicit database URL — there "
            "is no default to prevent silent misconfiguration.",
            file=sys.stderr,
        )
        return 2

    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            violations = check_tournament_schema_ready(conn)
    finally:
        engine.dispose()

    if not violations:
        print("OK: all tournament schema invariants satisfied")
        return 0

    print(f"FAIL: {len(violations)} violations found:")
    for v in violations:
        print(f"  - {v}")
    print()
    print("See migration file header for the full probe->resolution playbook.")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
```

### Alembic migration

`migrations/dashboard/versions/<auto-generated-hash>_tournament_plan_2a_constraints.py`:

```python
"""tournament plan 2a — schema constraints, cancel audit, AD-9 + AD-10 columns

Revision ID: <auto-generated>
Revises: 028d8a9fdc46  (tournament_slice_columns — HEAD at Plan 2a start)
Create Date: 2026-04-11

## Precondition

Transitively follows c8d5f2a91234 (IDOR fix, enforce_run_user_id_not_null)
via 028d8a9fdc46. IDOR fix backfilled and constrained benchmark_runs; Plan
2a applies the analogous invariant to tournament_participants, a sibling
table not touched by the IDOR migration. The FK-orphan probe on
tournament_participants therefore verifies a fresh invariant, not a
re-check of earlier work.

## Probe-to-resolution playbook

If `check_tournament_schema_ready` reports a violation, look up the probe
ID below for a concrete resolution. Running the probe against a staging
snapshot of production BEFORE the maintenance window is a deploy-checklist
requirement.

| Probe | Violation | Resolution |
|-------|-----------|------------|
| P1 | Participant with user_id IS NULL | DELETE anonymous rows, or backfill user_id from agent_name lookup if meaningful. Do NOT assign a sentinel user_id. |
| P2 | FK orphan on user_id | DELETE orphan rows: DELETE FROM tournament_participants WHERE user_id NOT IN (SELECT id FROM users). If this occurs on production, investigate how the orphan was created before proceeding. |
| P3 | Duplicate (tournament_id, user_id) participant | Manually dedupe. Keep the row with earliest joined_at. Contact ops before running bulk DELETE. |
| P4 | Duplicate (round_id, participant_id) action | Manually dedupe. Keep the row with earliest submitted_at. These are likely corrupted/debug data — inspect action_data before removing. |
| P5 | Duplicate (tournament_id, round_number) round | Indicates a logic bug in vertical slice round creation. Inspect the tournament's history before deleting. Likely cause: re-submitted create-round call during a crash. |
| P6 | User with >1 participant row in pending/active tournaments | Transition stale tournaments to completed/cancelled status directly via SQL (e.g. UPDATE tournaments SET status='cancelled' WHERE id=N). The migration's step 5a backfill will then automatically set released_at for participants of those tournaments. Alternatively, DELETE stale participant rows directly. Do NOT attempt to set released_at via SQL at probe time — the column does not exist yet. |
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

from atp.dashboard.migrations.probes.check_tournament_invariants import (
    check_tournament_schema_ready,
)

revision = "<auto-generated>"
down_revision = "028d8a9fdc46"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()

    # Step 1: Run probes. Abort on any violation.
    violations = check_tournament_schema_ready(connection)
    if violations:
        message = "Plan 2a migration aborted by probe — resolve and re-run:\n"
        for v in violations:
            message += f"  - {v}\n"
        raise RuntimeError(message)

    # Step 2: Tournament columns.
    # pending_deadline is added NULLABLE first, backfilled, then flipped
    # NOT NULL. Three-step approach is the portable idiom: SQLite accepts
    # CURRENT_TIMESTAMP as a non-volatile default but PostgreSQL equivalent
    # requires a non-volatile default and would fail.
    with op.batch_alter_table("tournaments") as batch_op:
        batch_op.add_column(sa.Column("pending_deadline", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("join_token", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("cancelled_at", sa.DateTime(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "cancelled_by",
                sa.Integer(),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            )
        )
        batch_op.add_column(sa.Column("cancelled_reason", sa.String(32), nullable=True))
        batch_op.add_column(sa.Column("cancelled_reason_detail", sa.String(512), nullable=True))

    op.execute(
        text(
            "UPDATE tournaments SET pending_deadline = CURRENT_TIMESTAMP "
            "WHERE pending_deadline IS NULL"
        )
    )

    with op.batch_alter_table("tournaments") as batch_op:
        batch_op.alter_column(
            "pending_deadline", existing_type=sa.DateTime(), nullable=False
        )
        batch_op.create_index(
            "idx_tournaments_status_pending_deadline",
            ["status", "pending_deadline"],
            unique=False,
        )
        batch_op.create_check_constraint(
            "ck_tournament_cancel_consistency",
            """(
                (
                    cancelled_reason IS NULL
                    AND cancelled_by IS NULL
                    AND cancelled_at IS NULL
                ) OR (
                    cancelled_reason = 'admin_action'
                    AND cancelled_by IS NOT NULL
                    AND cancelled_at IS NOT NULL
                ) OR (
                    cancelled_reason IN ('pending_timeout', 'abandoned')
                    AND cancelled_by IS NULL
                    AND cancelled_at IS NOT NULL
                )
            )""",
        )

    # Step 3: Round — uq_round_tournament_number + idx_round_status_deadline
    with op.batch_alter_table("tournament_rounds") as batch_op:
        batch_op.create_unique_constraint(
            "uq_round_tournament_number", ["tournament_id", "round_number"]
        )
        batch_op.create_index(
            "idx_round_status_deadline", ["status", "deadline"], unique=False
        )

    # Step 4: Action — source column + uq_action_round_participant.
    # server_default="submitted" backfills historical vertical slice rows
    # (which had no deadline worker, so every action was player-submitted).
    with op.batch_alter_table("tournament_actions") as batch_op:
        batch_op.add_column(
            sa.Column(
                "source",
                sa.String(32),
                nullable=False,
                server_default="submitted",
            )
        )
        batch_op.create_unique_constraint(
            "uq_action_round_participant", ["round_id", "participant_id"]
        )

    # Step 5: Participant — released_at, NOT NULL flip, uq_participant_tournament_user.
    with op.batch_alter_table("tournament_participants") as batch_op:
        batch_op.add_column(sa.Column("released_at", sa.DateTime(), nullable=True))
        batch_op.alter_column(
            "user_id", existing_type=sa.Integer(), nullable=False
        )
        batch_op.create_unique_constraint(
            "uq_participant_tournament_user", ["tournament_id", "user_id"]
        )

    # Step 5a: Backfill released_at for participants in terminal-status
    # tournaments. Any user who ever played in a completed/cancelled
    # tournament would otherwise have multiple participant rows with
    # released_at=NULL after step 5, violating uq_participant_user_active
    # in step 6. Setting released_at to CURRENT_TIMESTAMP is semantically
    # "the slot was released as of migration time".
    op.execute(text("""
        UPDATE tournament_participants
        SET released_at = CURRENT_TIMESTAMP
        WHERE tournament_id IN (
            SELECT id FROM tournaments
            WHERE status IN ('completed', 'cancelled')
        )
    """))

    # Step 6: Partial unique index on (user_id) WHERE user_id IS NOT NULL
    # AND released_at IS NULL. Created outside batch_alter_table because
    # older Alembic versions propagate dialect-specific where clauses
    # inconsistently through batch_op.create_index.
    op.create_index(
        "uq_participant_user_active",
        "tournament_participants",
        ["user_id"],
        unique=True,
        sqlite_where=text("user_id IS NOT NULL AND released_at IS NULL"),
        postgresql_where=text("user_id IS NOT NULL AND released_at IS NULL"),
    )


def downgrade() -> None:
    """Plan 2a downgrade is best-effort, not tested (per Known Limitations).
    Restore from backup is the supported recovery path.
    """
    op.drop_index("uq_participant_user_active", "tournament_participants")

    with op.batch_alter_table("tournament_participants") as batch_op:
        batch_op.drop_constraint("uq_participant_tournament_user", type_="unique")
        batch_op.alter_column(
            "user_id", existing_type=sa.Integer(), nullable=True
        )
        batch_op.drop_column("released_at")

    with op.batch_alter_table("tournament_actions") as batch_op:
        batch_op.drop_constraint("uq_action_round_participant", type_="unique")
        batch_op.drop_column("source")

    with op.batch_alter_table("tournament_rounds") as batch_op:
        batch_op.drop_index("idx_round_status_deadline")
        batch_op.drop_constraint("uq_round_tournament_number", type_="unique")

    with op.batch_alter_table("tournaments") as batch_op:
        batch_op.drop_constraint(
            "ck_tournament_cancel_consistency", type_="check"
        )
        batch_op.drop_index("idx_tournaments_status_pending_deadline")
        batch_op.drop_column("cancelled_reason_detail")
        batch_op.drop_column("cancelled_reason")
        batch_op.drop_column("cancelled_by")
        batch_op.drop_column("cancelled_at")
        batch_op.drop_column("join_token")
        batch_op.drop_column("pending_deadline")
```

### Design notes

**Probes before any ALTER.** `check_tournament_schema_ready` runs as the first statement of `upgrade()`, reusing the exact function the CLI probe calls. If the operator ran the CLI probe against staging and saw 0 violations, they still get the same probe re-run inside the transaction at actual upgrade time — small extra cost (sub-second on any reasonable DB), large safety margin against drift between probe and upgrade.

**CHECK constraint on cancel consistency.** `ck_tournament_cancel_consistency` enforces at the DB level the invariant that `TournamentCancelEvent.__post_init__` enforces at the Python level. Both SQLite (3.3+) and PostgreSQL support CHECK constraints in `batch_alter_table`; the constraint text is identical across backends because it uses only standard SQL `IS NULL` / `IN` predicates. The constraint also enforces atomicity of the cancel-tuple: you cannot record `cancelled_by=42` without also setting `cancelled_at`, and you cannot set `cancelled_at` without a reason. This rules out half-populated cancel state from rogue scripts or faulty migrations. The probe module does not need a probe for this constraint: `alembic upgrade` either applies it (and subsequent writes are guarded) or fails loudly.

**`pending_deadline` added nullable-then-backfill-then-NOT NULL.** SQLite's `batch_alter_table` recreates the table when flipping nullability, which is expensive but correct. Adding `pending_deadline NOT NULL DEFAULT CURRENT_TIMESTAMP` directly would accept on SQLite but fail on PostgreSQL (non-volatile default required). Three-step approach is the portable idiom.

**Partial unique index created outside `batch_alter_table`.** Older Alembic versions propagate `sqlite_where=` / `postgresql_where=` inconsistently through `batch_op.create_index`. The reliable form is `op.create_index` directly on the table. SQLite doesn't require batch mode for new index creation — batch mode only matters for column-level changes on tables that need recreation.

**`Round.status` storage.** Vertical slice stores `Round.status` as `String(20)` without a DB-level CHECK or native enum type. Adding `RoundStatus.CANCELLED` is therefore a Python-side-only change on both SQLite and PostgreSQL — the migration does not ALTER the `status` column itself. Under backlog item I, if that migration chooses to convert `Round.status` to a native PG ENUM type, the DDL must include all four values (`waiting_for_actions`, `in_progress`, `completed`, `cancelled`). The same caveat applies to `Tournament.status`. Backlog item I scope must enumerate both.

**Migration is forward-only in practice.** `downgrade()` is implemented so the revision chain is well-formed and tools that walk the graph don't break, but it is exercised by exactly one round-trip test (not the full regression suite). Operators wanting to revert restore from backup.

**Alembic revision ID.** Auto-generated by `alembic revision` at implementation time. The important invariants are `down_revision = "028d8a9fdc46"` and descriptive filename suffix `_tournament_plan_2a_constraints.py`.

---

## Section 4 — Deadline Worker & Cancel Flows: Lifecycle and Ops

### Deadline worker lifecycle

Three phases: startup → running → shutdown.

#### Startup

```
FastAPI app initialization
    │
    ▼
combined_lifespan(app) entered
    │
    ▼
WEB_CONCURRENCY assertion
    ├── wc == 1: continue
    └── wc != 1: raise RuntimeError with descriptive message → process
                  exits before any HTTP port is bound. No partial state.
    │
    ▼
asyncio.Event() created → shutdown_event
    │
    ▼
asyncio.create_task(run_deadline_worker(
    session_factory=app.state.session_factory,
    bus=app.state.tournament_event_bus,
    shutdown_event=shutdown_event,
))
    │
    ▼
mcp_app.router.lifespan_context(app) entered
    → FastMCP sub-app lifespan initialises session manager etc.
    │
    ▼
yield → FastAPI ready to serve requests
    → worker_task starts running its first tick concurrently
    │
    ▼
First _tick() logs `deadline_worker.started` with poll_interval_s
```

**Startup invariants:**

- If `WEB_CONCURRENCY` is wrong, the process dies **before** any HTTP listener is bound — there is no window where the platform serves requests without a deadline worker.
- If `run_deadline_worker` raises synchronously before its first `await`, `asyncio.create_task` raises and the lifespan exits with the same exception before `yield` — FastAPI never marks itself ready.
- If the first `_tick()` raises, the outer try/except catches it and logs `deadline_worker.tick_failed`; the loop continues and FastAPI is already serving.

**Startup wiring in `factory.py` lifespan:**

```python
@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    wc = int(os.environ.get("WEB_CONCURRENCY", "1"))
    if wc != 1:
        raise RuntimeError(
            f"ATP Tournament deadline worker requires WEB_CONCURRENCY=1 "
            f"(got {wc}). Multiple workers would each run a deadline worker, "
            f"racing on force_resolve_round and wasting DB reads. "
            f"Multi-worker support is backlog item I "
            f"(Redis bus + PG migration)."
        )

    shutdown = asyncio.Event()
    worker_task = asyncio.create_task(
        run_deadline_worker(
            app.state.session_factory,
            app.state.bus,
            shutdown_event=shutdown,
        )
    )
    try:
        async with mcp_app.router.lifespan_context(app):
            yield
    finally:
        shutdown.set()
        worker_task.cancel()
        await asyncio.gather(worker_task, return_exceptions=True)
```

#### Running

```python
POLL_INTERVAL_S = float(os.environ.get("ATP_DEADLINE_WORKER_POLL_INTERVAL_S", "5"))


async def run_deadline_worker(
    session_factory: async_sessionmaker[AsyncSession],
    bus: TournamentEventBus,
    *,
    shutdown_event: asyncio.Event,
) -> None:
    """Single asyncio task. Runs inside FastAPI lifespan."""
    log = structlog.get_logger("tournament.deadlines")
    log.info("deadline_worker.started", poll_interval_s=POLL_INTERVAL_S)

    while not shutdown_event.is_set():
        try:
            await _tick(session_factory, bus, log)   # OUTER guard
        except Exception:
            log.exception("deadline_worker.tick_failed")

        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=POLL_INTERVAL_S,
            )
        except TimeoutError:
            pass  # normal path — interval elapsed

    log.info("deadline_worker.shutting_down")


async def _tick(session_factory, bus, log) -> None:
    """One scan pass across both paths."""
    t_start = time.monotonic()
    async with session_factory() as session:
        service = TournamentService(session, bus)

        # Path 1: expired round deadlines
        expired_rounds = await session.execute(
            select(Round.id)
              .join(Tournament, Tournament.id == Round.tournament_id)
              .where(Round.status == RoundStatus.WAITING_FOR_ACTIONS)
              .where(Round.deadline < datetime.utcnow())
              .where(Tournament.status == TournamentStatus.ACTIVE)  # AD-6
        )
        round_ids = [r[0] for r in expired_rounds]
        for round_id in round_ids:
            try:
                await service.force_resolve_round(round_id)   # INNER guard
            except Exception:
                log.exception(
                    "deadline_worker.round_resolve_failed",
                    round_id=round_id,
                )

        # Path 2: expired PENDING tournaments (AD-9)
        expired_pending = await session.execute(
            select(Tournament.id)
              .where(Tournament.status == TournamentStatus.PENDING)
              .where(Tournament.pending_deadline < datetime.utcnow())
        )
        tournament_ids = [t[0] for t in expired_pending]
        for tournament_id in tournament_ids:
            try:
                await service.cancel_tournament_system(
                    tournament_id,
                    reason=CancelReason.PENDING_TIMEOUT,
                )
            except Exception:
                log.exception(
                    "deadline_worker.pending_cancel_failed",
                    tournament_id=tournament_id,
                )

    log.info(
        "deadline_worker.tick_complete",
        rounds_processed=len(round_ids),
        pending_cancelled=len(tournament_ids),
        elapsed_ms=int((time.monotonic() - t_start) * 1000),
    )
```

**AD-6 race guard.** Path 1 JOINs on `Tournament.status == ACTIVE` to exclude rounds belonging to tournaments cancelled between the round's creation and the current tick. Without this filter, the deadline worker could attempt to force-resolve a round whose tournament was cancelled by user action seconds earlier. The filter makes this impossible at query time rather than at service method time.

**Lock semantics by backend:**

- **SQLite in WAL mode** (Plan 2a production): `with_for_update` is silently ignored by the SQLAlchemy SQLite dialect — there is no row-level lock, but WAL mode's single-writer serialization provides the equivalent guarantee. Only one writer can be inside a `BEGIN IMMEDIATE` transaction at a time, which means `force_resolve_round` and a concurrent `submit_action` will serialize at the file lock level. The AD-6 status filter combined with this serialization prevents double-resolution: whichever transaction commits first flips the status, the second observes `WHERE status=waiting_for_actions` matching zero rows and is a no-op.

- **SQLite without WAL mode** (test fixtures only): same single-writer guarantee via the legacy `BEGIN IMMEDIATE` lock, but connection contention surfaces as `SQLITE_BUSY` errors instead of clean waits. Plan 2a's `conftest.py` fixture sets `PRAGMA journal_mode=WAL` on the test engine non-negotiably.

- **PostgreSQL** (backlog item I): `with_for_update` becomes a real row-level lock. The deadline worker acquires `FOR UPDATE` on the round row and blocks any concurrent `submit_action` attempting the same, until the deadline worker commits. The AD-6 status filter becomes belt-and-suspenders.

**Invariant under all three backends:** `force_resolve_round` and `submit_action` on the same `round_id` can never both commit a resolution. At least one of them will observe `status != WAITING_FOR_ACTIONS` when it re-reads under lock and will no-op.

**Why Path 1 runs before Path 2 each tick.** If a tournament is about to expire its pending deadline but also has a round whose deadline passed (impossible in the normal state machine — a PENDING tournament has no rounds — but possible during data corruption), Path 1 running first guarantees any round-level work completes before its tournament is cancelled. In the normal case the ordering doesn't matter because the sets are disjoint.

**Session reuse.** One `async with session_factory()` wraps both paths per tick. Each service method opens its own `async with self.session.begin():` internally. The inner try/except guarantees a raised exception in one row does not propagate to the enclosing `async with` — it is caught and logged, leaving the next `session.begin()` to open a fresh transaction.

#### Shutdown

```
FastAPI shutdown signal (SIGTERM, uvicorn stop)
    │
    ▼
combined_lifespan yield returns
    │
    ▼
finally: shutdown.set()
    → loop condition `not shutdown_event.is_set()` becomes False
    │
    ▼
worker_task.cancel()
    → CancelledError thrown into worker_task at next await
    → SQLAlchemy async context manager catches in __aexit__, rolls back
      open transaction, closes session
    → run_deadline_worker returns (or raises CancelledError)
    │
    ▼
await asyncio.gather(worker_task, return_exceptions=True)
    → soaks CancelledError
```

**Correctness of hard cancel during mid-transaction work:**

- During SQL execution: `session.execute(...)` is an `await` point. CancelledError unwinds through SQLAlchemy's async machinery, which always rolls back the open transaction. DB state is exactly pre-`session.begin()`.
- During commit: `COMMIT` is an `await`. SQLite's single-file atomicity guarantees the commit is either fully applied or fully rolled back — there is no half-commit state.
- Between commit and bus publish: if cancel hits after `session.commit()` but before `bus.publish(...)`, DB is in the new state but the event is lost. The next `session_sync` on any client's reconnect replays the authoritative state from DB.
- No `asyncio.shield`: every await is a potential cancel point, and every such point has a correctness-preserving unwind via either transaction rollback or session_sync replay.

### `_cancel_impl` step-by-step

```python
async def _cancel_impl(
    self,
    tournament_id: int,
    reason: CancelReason,
    cancelled_by: int | None,
    reason_detail: str | None,
) -> TournamentCancelEvent | None:
    """Mutates DB state for cancellation. Does NOT commit — caller is
    responsible. Does NOT publish to bus — returns the event for the
    caller to publish after its commit succeeds.
    """

    # Step 1: Lock + load tournament
    tournament = await self.session.get(
        Tournament, tournament_id, with_for_update=True
    )
    if tournament is None:
        raise NotFoundError(f"tournament {tournament_id}")

    # Step 2: Idempotent guard
    if tournament.status in (
        TournamentStatus.CANCELLED,
        TournamentStatus.COMPLETED,
    ):
        return None

    # Step 3: Compute final_rounds_played BEFORE step 6's bulk UPDATE.
    # Otherwise the count would include in-flight rounds that step 6
    # transitions to CANCELLED.
    final_rounds_played = await self.session.scalar(
        select(func.count())
          .select_from(Round)
          .where(Round.tournament_id == tournament_id)
          .where(Round.status == RoundStatus.COMPLETED)
    ) or 0

    # Step 4: Write tournament audit fields
    now = datetime.utcnow()
    tournament.status = TournamentStatus.CANCELLED
    tournament.cancelled_at = now
    tournament.cancelled_by = cancelled_by
    tournament.cancelled_reason = reason
    tournament.cancelled_reason_detail = reason_detail

    # Step 5: Release all unreleased participants (bulk UPDATE)
    await self.session.execute(
        update(Participant)
          .where(Participant.tournament_id == tournament_id)
          .where(Participant.released_at.is_(None))
          .values(released_at=now)
    )

    # Step 6: Cancel all in-flight rounds (bulk UPDATE)
    await self.session.execute(
        update(Round)
          .where(Round.tournament_id == tournament_id)
          .where(Round.status.in_([
              RoundStatus.WAITING_FOR_ACTIONS,
              RoundStatus.IN_PROGRESS,
          ]))
          .values(status=RoundStatus.CANCELLED)
    )

    # Step 7: Build event (caller publishes post-commit)
    return TournamentCancelEvent(
        tournament_id=tournament_id,
        cancelled_at=now,
        cancelled_by=cancelled_by,
        cancelled_reason=reason,
        cancelled_reason_detail=reason_detail,
        final_rounds_played=final_rounds_played,
        final_status=TournamentStatus.CANCELLED,
    )
```

**Critical design choices:**

- **`_cancel_impl` does not commit and does not publish.** It mutates session state and returns an event. This lets it be called from inside an existing transaction (the `leave()` cascade path) without nested-transaction semantics and without double-publishing.
- **Idempotent via status guard.** Second cancel on already-CANCELLED tournament returns None silently.
- **`with_for_update=True` on SELECT.** On PostgreSQL acquires a real row lock preventing concurrent cancel mutations. On SQLite the SQLAlchemy dialect silently drops the clause; WAL mode's single-writer serialization at commit time provides atomicity but not row-level isolation — concurrent cancels from different transactions can both reach step 2 with `status IN (PENDING, ACTIVE)` in their respective snapshots and both pass the idempotent guard. Plan 2a accepts this limitation for audit attribution only (status transition and cascade effects remain safe); see Known Limitations "Cancel audit attribution under concurrent race". Resolved under backlog I when `FOR UPDATE` becomes a real lock.
- **Bulk UPDATEs** for participants and rounds. Steps 5 and 6 use `UPDATE ... WHERE` rather than iterating over ORM objects.
- **Bus publish failures after commit report success.** Once `session.commit()` returns, the cancellation is fact. If `bus.publish(event)` subsequently raises, the caller's response is still 200/success; the failure is logged at WARN with `tournament.cancel.publish_failed` and `exc_info`. Subscribers who missed the event recover their state via the next `session_sync` on reconnect. DB is the single source of truth, bus is a best-effort push layer.

### Cancel call sites

#### Path A — User-initiated cancel (REST or MCP)

```python
async def cancel_tournament(
    self,
    user: User,
    tournament_id: int,
    reason_detail: str | None = None,
) -> None:
    # Pre-transaction: enumeration-guarded auth check
    await self._load_for_auth(tournament_id, user)

    # Transaction: call _cancel_impl, commit, then publish
    async with self.session.begin():
        event = await self._cancel_impl(
            tournament_id,
            reason=CancelReason.ADMIN_ACTION,
            cancelled_by=user.id,
            reason_detail=reason_detail,
        )
    # Outside transaction
    if event is not None:
        try:
            await self.bus.publish(event)
        except Exception:
            log.warning(
                "tournament.cancel.publish_failed",
                tournament_id=tournament_id,
                exc_info=True,
            )
```

Authorization runs against an unlocked SELECT so unauthorized callers never acquire the row lock.

#### Path B — Deadline worker pending_timeout

```python
async def cancel_tournament_system(
    self,
    tournament_id: int,
    reason: CancelReason,
    reason_detail: str | None = None,
) -> None:
    async with self.session.begin():
        event = await self._cancel_impl(
            tournament_id,
            reason=reason,
            cancelled_by=None,
            reason_detail=reason_detail,
        )
    if event is not None:
        try:
            await self.bus.publish(event)
        except Exception:
            log.warning(
                "tournament.cancel.publish_failed",
                tournament_id=tournament_id,
                exc_info=True,
            )
```

No authorization. Caller is trusted (deadline worker or `leave()` cascade).

#### Path C — Last-participant abandoned cascade inside `leave()`

```python
async def leave(self, tournament_id: int, user: User) -> None:
    cancel_event = None
    async with self.session.begin():
        # Find participant
        participant = await self.session.scalar(
            select(Participant)
              .where(Participant.tournament_id == tournament_id)
              .where(Participant.user_id == user.id)
              .where(Participant.released_at.is_(None))
        )
        if participant is None:
            raise NotFoundError(
                f"user {user.id} is not active in tournament {tournament_id}"
            )

        # Mark released
        participant.released_at = datetime.utcnow()
        await self.session.flush()

        # Last-participant detection
        remaining = await self.session.scalar(
            select(func.count())
              .select_from(Participant)
              .where(Participant.tournament_id == tournament_id)
              .where(Participant.released_at.is_(None))
        )
        tournament = await self.session.get(Tournament, tournament_id)

        if remaining == 0 and tournament.status == TournamentStatus.ACTIVE:
            log.info(
                "tournament.leave.abandoned_cascade",
                tournament_id=tournament_id,
                leaving_user_id=user.id,
            )
            cancel_event = await self._cancel_impl(
                tournament_id,
                reason=CancelReason.ABANDONED,
                cancelled_by=None,
                reason_detail=None,
            )
    # Outside transaction
    if cancel_event is not None:
        try:
            await self.bus.publish(cancel_event)
        except Exception:
            log.warning(
                "tournament.cancel.publish_failed",
                tournament_id=tournament_id,
                exc_info=True,
            )
```

**Single transaction for leave + cascade.** `leave()` opens one `session.begin()` that contains both the `released_at` write and the `_cancel_impl` delegation. Commit is atomic: either both the participant release and the tournament cancellation persist, or neither.

### Cancel cascade — complete effect list

Single `_cancel_impl` call produces the following state transitions, all in one transaction:

| Entity | Before | After | Filter |
|---|---|---|---|
| `Tournament` | `status=PENDING\|ACTIVE` | `status=CANCELLED`, `cancelled_at`, `cancelled_by`, `cancelled_reason`, `cancelled_reason_detail` populated | one row by id |
| `Participant` | `released_at=NULL` | `released_at=NOW` | all for this tournament with NULL released_at |
| `Round` | `status IN (waiting_for_actions, in_progress)` | `status=CANCELLED` | all for this tournament in those two states |
| `Action` | — | — | untouched; historical actions preserved for audit |
| Bus | — | `tournament_cancelled` published after commit | exactly one event |

**What is deliberately not cascaded:**
- **Completed rounds** stay `COMPLETED`. Audit trail for "how many rounds were played before abandonment" is preserved.
- **Payoffs** on completed actions are not cleared.
- **Leaderboard results** computed up to the cancel point remain queryable; `list_tournaments` / `get_tournament` serializers include `cancelled_reason` so downstream consumers know to exclude cancelled tournaments from "active rankings" views (Plan 2b responsibility).

**AD-6 race guard effect after cancel:** once `Round.status == CANCELLED`, the deadline worker's Path 1 WHERE clause excludes these rounds on every subsequent tick. No extra cleanup code.

### Error handling matrix

| Failure point | User cancel (Path A) | Deadline pending_timeout (Path B) | leave() abandoned (Path C) |
|---|---|---|---|
| Pre-transaction auth check raises | 404 to caller (enum guard) | N/A — no auth check | N/A — not authorized at this level |
| `session.begin()` cannot acquire connection | 500 to caller, nothing mutated | caught by tick outer guard, logged, loop continues | 500 to caller, nothing mutated |
| `with_for_update` SELECT raises | 500 | caught and logged, next tick retries | 500 |
| Tournament already terminal at step 2 | silent 200 (idempotent; no event published) | silent no-op | `leave()` still commits the `released_at` write; only the cascade is a no-op |
| UPDATE on participants / rounds fails | transaction rolls back, 500 | caught, logged, next tick retries | full rollback — leave() also rolls back |
| `session.commit()` fails | rollback, 500 | caught, logged, next tick retries | rollback, leave() reports failure |
| `bus.publish()` fails AFTER commit | **200 OK to caller** — logged WARN `tournament.cancel.publish_failed`. Subscribers recover via `session_sync` on reconnect. No retry. | Caught by inner try/except, logged, next tick naturally skips already-terminal tournament. No recovery action needed. | `leave_tournament` returns success — release AND cascade committed. WARN log. Subscribers recover via session_sync. |
| Process shutdown mid-transaction | CancelledError → rollback, caller sees 500 or connection reset | CancelledError → rollback, next restart picks up expired tournaments via fresh tick | CancelledError → rollback, caller's request never completes; post-restart state unchanged |
| Process shutdown between commit and publish | DB is in CANCELLED state, event lost, caller may see 500. Recovery via session_sync. | Same — next restart doesn't re-publish but idempotent status guard prevents double-cancel | Same |
| Retry of `leave()` after successful first call | N/A | N/A | Second call returns `NotFoundError` because `released_at IS NULL` filter excludes the already-released row. **SDK retry layers MUST treat NotFoundError after leave_tournament as a terminal success signal**. |
| Concurrent cancel from another path on the same tournament (SQLite WAL) | Both commit (WAL serializes writes), audit attribution = last writer wins, two `TournamentCancelEvent`s published to bus with conflicting `cancelled_by` / `cancelled_reason`. Status and cascade effects are safe (deterministic `CANCELLED`, idempotent bulk UPDATEs). See Known Limitations "Cancel audit attribution under concurrent race". Resolved under backlog I (PG `FOR UPDATE`). | Same. | Same. |

**Invariant.** DB state is either "fully cancelled" or "fully pre-cancel state", never partial. Bus state can lag DB state (published event missing), and `session_sync` is the universal catchup path.

### Observability — structured log fields

Logs use `structlog` with JSON output in production.

**Deadline worker logs:**

| Event | Fields | When |
|---|---|---|
| `deadline_worker.started` | `poll_interval_s` | Startup |
| `deadline_worker.tick_complete` | `rounds_processed`, `pending_cancelled`, `elapsed_ms` | End of every successful tick |
| `deadline_worker.tick_failed` | `exc_info` | Outer try/except caught exception in `_tick` |
| `deadline_worker.round_resolve_failed` | `round_id`, `exc_info` | Inner try/except caught `force_resolve_round` exception |
| `deadline_worker.pending_cancel_failed` | `tournament_id`, `exc_info` | Inner try/except caught `cancel_tournament_system` exception |
| `deadline_worker.shutting_down` | — | Loop exit |

**Cancel flow logs:**

| Event | Fields | When |
|---|---|---|
| `tournament.cancel.user_initiated` | `tournament_id`, `user_id`, `is_admin`, `reason_detail` | `cancel_tournament` called, before `_cancel_impl` |
| `tournament.cancel.system_initiated` | `tournament_id`, `reason`, `reason_detail` | `cancel_tournament_system` called |
| `tournament.cancel.already_terminal` | `tournament_id`, `current_status`, `via_path` | `_cancel_impl` step 2 guard returned None |
| `tournament.cancel.committed` | `tournament_id`, `reason`, `participants_released`, `rounds_cancelled`, `final_rounds_played` | After `session.begin()` exits successfully |
| `tournament.cancel.published` | `tournament_id`, `subscriber_count` | After `bus.publish(event)` returns |
| `tournament.cancel.publish_failed` | `tournament_id`, `exc_info` | Bus publish raised — DB is committed but event is lost |
| `tournament.leave.abandoned_cascade` | `tournament_id`, `leaving_user_id` | Inside `leave()` when last-participant detection fires |

All fields are bounded cardinality. `tournament_id` / `user_id` / `round_id` are integer FKs. `reason` is a 3-value enum. The one free-text field `reason_detail` is bounded to 512 chars by the DB column.

**Ops playbook expectation** (future dashboard observability):
- `deadline_worker.tick_failed` count > 0 per 5-minute window → page.
- `deadline_worker.round_resolve_failed` count > threshold → investigate poison row (triggers backlog AB).
- `tournament.cancel.publish_failed` count > 0 → bus health check.
- `deadline_worker.tick_complete.elapsed_ms` p99 > 500 → triggers backlog I threshold review.

---

## Section 5 — Testing Strategy

### Test pyramid

Plan 2a adds roughly 1540 LOC of production code across eight modules. Target test-to-prod ratio 1.5:1 as an orientation (service.py higher, probe module lower, overall sum ~2300 LOC of tests). Distribution across layers:

| Layer | Share | LOC estimate | Coverage focus | Fixture base |
|---|---|---|---|---|
| Unit (mock session, mock bus) | ~48% | ~1100 | service control flow, `_cancel_impl` branching, deadline tick isolation, `TournamentCancelEvent.__post_init__`, static architectural guards | `MagicMock(spec=AsyncSession)`, `TestEventBus` stub |
| Integration (SQLite WAL, real service + bus) | ~32% | ~750 | probe module (6 probes × positive/negative), Alembic upgrade/downgrade round-trip, partial unique index under concurrency, CHECK constraint rejection, AD-6 race guard, cancel cascade (3 paths), session_sync delivery | `tournament_db` fixture |
| E2E (FastMCP + FastAPI + deadline worker in lifespan) | ~20% | ~450 | full lifecycle over MCP SSE, reconnect → session_sync, REST admin curl path, AD-9/AD-10 enforcement | `mcp_test_client`, `tournament_app` |

**Pyramid invariant.** Each Section 1 acceptance criterion (SC-1..SC-10 verbatim) is covered by at least one test in at least two layers when applicable (some SCs are inherently single-layer — SC-7 probe dry-run is integration-only; SC-10 regression is a CI gate). The mapping in §5.6 is **the** coverage document, not line %.

### Unit layer

**Path:** `tests/unit/dashboard/tournament/`

No file I/O, no DB, no network. Sub-second iteration loop.

#### Service layer tests

**`test_load_for_auth.py`** — seven-row table-driven test for the enumeration guard invariant:

```python
@pytest.mark.parametrize("scenario,user_kind,tournament_created_by,expect", [
    ("admin_normal_owner",   "admin",   42,       "ok"),
    ("admin_other_owner",    "admin",   99,       "ok"),
    ("admin_legacy_null",    "admin",   None,     "ok"),
    ("owner_match",          "regular", "self",   "ok"),
    ("owner_mismatch",       "regular", 99,       NotFoundError),
    ("non_admin_legacy",     "regular", None,     NotFoundError),
    ("tournament_missing",   "regular", "no_row", NotFoundError),
])
async def test_load_for_auth_enumeration_guard(...): ...
```

Critical invariant: all four denial paths raise `NotFoundError` (same class as "doesn't exist"), not `PermissionError`.

**`test_cancel_impl_logic.py`** — `_cancel_impl` with `MagicMock(spec=AsyncSession)`:

- Happy path (PENDING → CANCELLED) — tournament.status is set, all four cancel audit fields set, return value is a `TournamentCancelEvent` with matching fields.
- Happy path (ACTIVE → CANCELLED).
- Idempotent on CANCELLED — return None, no UPDATE calls (only the initial `session.get` SELECT executes).
- Idempotent on COMPLETED — same.
- `final_rounds_played` ordering regression test: mock round set (2 COMPLETED, 1 WAITING_FOR_ACTIONS, 1 IN_PROGRESS) → event has `final_rounds_played == 2`, not 4. Anyone refactoring step 3 below step 6 fails this test instantly. Uses the `frozen_clock` fixture so `cancelled_at` in the returned event is deterministic for assertion (`assert event.cancelled_at == datetime(2026, 4, 15, 10, 0, 0)`).
- In-flight round transitions use one bulk UPDATE.
- Participant release uses one bulk UPDATE.
- Event built after mutations (verified via mock call ordering).

**`test_tournament_cancel_event_post_init.py`** — validates the `__post_init__` validator:

```python
@pytest.mark.parametrize("reason,cancelled_by,final_status,should_raise", [
    (CancelReason.ADMIN_ACTION,    42,   TournamentStatus.CANCELLED, False),
    (CancelReason.PENDING_TIMEOUT, None, TournamentStatus.CANCELLED, False),
    (CancelReason.ABANDONED,       None, TournamentStatus.CANCELLED, False),
    (CancelReason.PENDING_TIMEOUT, 42,   TournamentStatus.CANCELLED, True),
    (CancelReason.ABANDONED,       42,   TournamentStatus.CANCELLED, True),
    (CancelReason.ADMIN_ACTION,    None, TournamentStatus.CANCELLED, True),
    (CancelReason.ADMIN_ACTION,    42,   TournamentStatus.ACTIVE,    True),
])
def test_cancel_event_invariant(...): ...
```

Validates the Python layer of the defense-in-depth trinity. DB layer gets its own integration test; `_cancel_impl` construction layer is the happy path.

#### Deadline worker tick tests

**`test_deadline_tick_isolation.py`** — main invariant: one poisoned row does not kill the tick.

```python
service_mock.force_resolve_round = AsyncMock(
    side_effect=[None, ServiceError("simulated"), None]
)

await _tick(mock_session_factory, captured_bus, log)

assert service_mock.force_resolve_round.call_count == 3
failed_logs = [r for r in caplog.records
               if r.event == "deadline_worker.round_resolve_failed"]
assert len(failed_logs) == 1
outer_logs = [r for r in caplog.records
              if r.event == "deadline_worker.tick_failed"]
assert len(outer_logs) == 0
```

**`test_deadline_tick_outer_failure.py`** — session_factory itself raises → outer guard catches, loop survives.

**`test_deadline_tick_two_paths.py`** — Path 1 runs before Path 2 deterministically; empty-path case logs `tick_complete` with zero counts.

**`test_deadline_tick_shutdown_hard_cancel.py`** — correctly encodes hard-cancel semantics:

```python
async def test_shutdown_hard_cancels_in_flight_tick(mock_session_factory):
    """shutdown.set() immediately followed by task.cancel(). CancelledError
    propagates into the in-flight tick's await, SQLAlchemy async context
    manager rolls back, no partial DB state."""
    service_mock.force_resolve_round = AsyncMock(
        side_effect=lambda rid: asyncio.sleep(5)
    )
    # ...
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_deadline_worker(mock_session_factory, captured_bus,
                            shutdown_event=shutdown)
    )
    await asyncio.sleep(0.1)

    shutdown.set()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert task.done()
    mock_session.__aexit__.assert_called()
    assert mock_session.__aexit__.call_args[0][0] is asyncio.CancelledError
```

Key assertion: task completes in <1 second despite the mock's 5-second sleep.

#### Static architectural guard tests

**`test_static_guards.py`** — grep-based tests enforcing architectural contracts. Run in the unit stage for sub-30-second feedback.

```python
def test_cancel_tournament_system_not_called_from_handlers():
    """Twin-methods invariant: system method must not be reachable from
    any REST or MCP handler. Called only from deadline worker."""
    matches = grep_pattern(
        r"cancel_tournament_system\b",
        paths=[
            "packages/atp-dashboard/atp/dashboard/mcp",
            "packages/atp-dashboard/atp/dashboard/v2/routes",
        ],
    )
    assert matches == [], (
        f"cancel_tournament_system called from handler files "
        f"(must be deadline_worker-only): {matches}"
    )


def test_no_bare_string_round_status_comparisons():
    """Plan 2a refactor invariant: all Round.status comparisons must use
    RoundStatus enum."""
    matches = grep_pattern(
        r'Round\.status\s*[=!]=\s*["\']',
        paths=["packages/atp-dashboard/atp/dashboard/tournament"],
    )
    assert matches == [], (
        f"Bare string literal comparison on Round.status: {matches}. "
        f"Use RoundStatus enum from models.py."
    )


def test_no_direct_cancel_field_writes_outside_cancel_impl():
    """All writes to cancelled_by/cancelled_at/cancelled_reason/
    cancelled_reason_detail must go through _cancel_impl."""
    matches = grep_pattern(
        r"\.(cancelled_by|cancelled_at|cancelled_reason|cancelled_reason_detail)\s*=",
        paths=["packages/atp-dashboard/atp/dashboard/tournament"],
    )
    allowed_file = "service.py"
    bad = [m for m in matches if Path(m.path).name != allowed_file]
    assert bad == [], (
        f"Direct cancel-field writes outside service.py _cancel_impl: {bad}"
    )
```

Helper `grep_pattern` lives in `tests/unit/dashboard/tournament/_grep_helper.py` — ~20-line wrapper around `subprocess.run(["git", "grep", "-n", pattern, *paths])`.

### Integration layer

**Path:** `tests/integration/dashboard/tournament/`

Shared fixture `tournament_db` provides a SQLite WAL-mode database with full `alembic upgrade head` applied.

#### Migration probe tests

Probes correspond exactly to Section 3 probe module (P1–P6). Each gets positive (clean DB) + negative (seeded violation) tests, plus edge cases where applicable.

| Probe | Covers | Positive | Negative | Edge case |
|---|---|---|---|---|
| P1 | `Participant.user_id IS NULL` | Clean DB → `[]` | Seed 1 anonymous participant → returns "P1:" | — |
| P2 | FK orphans on `Participant.user_id` | Clean DB | `PRAGMA foreign_keys=OFF`, seed participant with user_id pointing to non-existent user, `PRAGMA foreign_keys=ON`, run probe → "P2:". The fixture default enables FK enforcement, so the negative test must toggle it explicitly to seed the orphan — this mirrors production exposure where FK=OFF is the silent-failure mode P2 defends against. | — |
| P3 | Duplicate `(tournament_id, user_id)` in participants | Clean DB | Seed 2 rows with same pair → "P3:" | — |
| P4 | Duplicate `(round_id, participant_id)` in actions | Clean DB | Seed 2 action rows for same pair → "P4:" | — |
| P5 | Duplicate `(tournament_id, round_number)` in rounds | Clean DB | Seed 2 rounds with same pair → "P5:" | — |
| P6 | Multi-active users (relaxed via JOIN) | Clean DB | Seed 1 user with 2 participants both in pending/active tournaments → "P6:" | **Critical edge case:** seed 1 user with 1 active participant in ACTIVE tournament + 3 participants in COMPLETED tournaments → probe must return `[]`. Regression-tests the relaxed JOIN — anyone "optimizing" by removing the JOIN turns this test red. |

**`test_migration_probe_cli_entrypoint.py`** — exercises the `_main()` block:

```python
def test_probe_cli_exit_zero_on_clean_db(tmp_path, tournament_db_path):
    result = subprocess.run(
        [sys.executable, "-m",
         "atp.dashboard.migrations.probes.check_tournament_invariants"],
        env={**os.environ, "ATP_DATABASE_URL": f"sqlite:///{tournament_db_path}"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "OK:" in result.stdout


def test_probe_cli_exit_one_on_violations(seeded_dup_participants_db):
    result = subprocess.run(
        [sys.executable, "-m",
         "atp.dashboard.migrations.probes.check_tournament_invariants"],
        env={**os.environ, "ATP_DATABASE_URL": f"sqlite:///{seeded_dup_participants_db}"},
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "P3:" in result.stdout
```

**`test_migration_upgrade_downgrade_round_trip.py`** — full round trip from `028d8a9fdc46` baseline: upgrade → all 7 invariants + 8 columns exist → downgrade → schema returns to baseline → upgrade again (idempotent).

#### Schema invariant tests

**`test_partial_unique_index.py`** — `uq_participant_user_active` under direct writes:

- Insert one active participant → OK.
- Insert second active participant (same user, different tournament) → `IntegrityError` with "uq_participant_user_active" in the message.
- Release from first tournament → second insert now succeeds.
- Insert 100 released rows for one user across 100 different tournaments → all OK (partial index ignores released).

**`test_cancel_check_constraint.py`** — exercises `ck_tournament_cancel_consistency`:

```python
@pytest.mark.parametrize("fields,should_fail", [
    # Valid: not cancelled
    ({"status": "active", "cancelled_reason": None, "cancelled_by": None, "cancelled_at": None}, False),
    # Valid: admin_action
    ({"status": "cancelled", "cancelled_reason": "admin_action", "cancelled_by": 42, "cancelled_at": now()}, False),
    # Valid: pending_timeout
    ({"status": "cancelled", "cancelled_reason": "pending_timeout", "cancelled_by": None, "cancelled_at": now()}, False),
    # Valid: abandoned
    ({"status": "cancelled", "cancelled_reason": "abandoned", "cancelled_by": None, "cancelled_at": now()}, False),
    # Invalid: admin_action without actor
    ({"status": "cancelled", "cancelled_reason": "admin_action", "cancelled_by": None, "cancelled_at": now()}, True),
    # Invalid: system reason with actor
    ({"status": "cancelled", "cancelled_reason": "pending_timeout", "cancelled_by": 42, "cancelled_at": now()}, True),
    ({"status": "cancelled", "cancelled_reason": "abandoned", "cancelled_by": 42, "cancelled_at": now()}, True),
    # Invalid: reason set without cancelled_at
    ({"status": "cancelled", "cancelled_reason": "admin_action", "cancelled_by": 42, "cancelled_at": None}, True),
    # Invalid: cancelled_by set without reason
    ({"status": "cancelled", "cancelled_reason": None, "cancelled_by": 42, "cancelled_at": None}, True),
])
async def test_check_constraint_enforces_cancel_tuple(
    tournament_db, fixture_tournament, fields, should_fail
): ...
```

Bypasses `TournamentService` entirely and writes directly via `session.execute(update(...))` — the path the CHECK constraint defends against.

#### Concurrent access tests

All concurrent tests use **separate sessions per concurrent actor** from a shared factory. A single `AsyncSession` is not safe for parallel operations in SQLAlchemy async.

**`test_idempotent_join_concurrent.py`** — two tests, one per SC. SC-5 verifies idempotent re-join from the same user to the **same** tournament under race; SC-4 verifies the AD-10 partial unique index across **different** tournaments.

```python
async def test_sc5_five_concurrent_joins_same_tournament_all_idempotent(
    session_factory, factory, bus
):
    """SC-5: five concurrent join calls from the same user to the SAME
    tournament must all succeed idempotently. Exactly one row ends up in
    the DB; all callers receive the same Participant. This exercises the
    uq_participant_tournament_user race path in the join() contract."""
    t = await factory.tournament(num_players=4)
    user = await factory.user()

    async def one_join():
        async with session_factory() as session:
            svc = TournamentService(session, bus)
            return await svc.join(
                tournament_id=t.id, user=user, agent_name="bot"
            )

    results = await asyncio.gather(*[one_join() for _ in range(5)])

    # All five calls returned successfully (no ConflictError raised)
    assert len(results) == 5
    for participant, is_new in results:
        assert participant is not None

    # Exactly one call reported is_new=True; the other four resolved to
    # (existing, False) via the uq_participant_tournament_user race path
    # in join()
    new_flags = [is_new for _, is_new in results]
    assert new_flags.count(True) == 1
    assert new_flags.count(False) == 4

    # All five Participant references point at the same row
    participant_ids = {p.id for p, _ in results}
    assert len(participant_ids) == 1

    # DB has exactly one row
    async with session_factory() as verify:
        count = await verify.scalar(
            select(func.count()).select_from(Participant)
              .where(Participant.tournament_id == t.id)
              .where(Participant.user_id == user.id)
        )
        assert count == 1


async def test_sc4_two_concurrent_joins_different_tournaments_one_wins(
    session_factory, factory, bus
):
    """SC-4: two concurrent join calls from the same user to two DIFFERENT
    tournaments must resolve to exactly one success and one ConflictError.
    This exercises the uq_participant_user_active partial unique index."""
    t_a = await factory.tournament(name="a", num_players=4)
    t_b = await factory.tournament(name="b", num_players=4)
    user = await factory.user()

    async def one_join(tournament_id):
        async with session_factory() as session:
            svc = TournamentService(session, bus)
            try:
                return await svc.join(
                    tournament_id=tournament_id, user=user, agent_name="bot"
                )
            except ConflictError as e:
                return e

    results = await asyncio.gather(one_join(t_a.id), one_join(t_b.id))
    successes = [r for r in results if not isinstance(r, Exception)]
    conflicts = [r for r in results if isinstance(r, ConflictError)]

    assert len(successes) == 1
    assert len(conflicts) == 1
    assert "already has an active tournament" in str(conflicts[0])

    async with session_factory() as verify:
        total = await verify.scalar(
            select(func.count()).select_from(Participant)
              .where(Participant.user_id == user.id)
              .where(Participant.released_at.is_(None))
        )
        assert total == 1  # exactly one active slot across all tournaments
```

**`test_deadline_worker_race.py`** — AD-6 race guard:

```python
async def test_force_resolve_vs_submit_action(session_factory, factory, bus):
    t, round_ = await factory.setup_two_player_round_with_one_pending_action()
    user2 = await factory.get_pending_user(round_)

    async def deadline_path():
        async with session_factory() as session:
            svc = TournamentService(session, bus)
            try:
                await svc.force_resolve_round(round_.id)
                return "deadline_won"
            except ConflictError:
                return "deadline_noop"

    async def submit_path():
        async with session_factory() as session:
            svc = TournamentService(session, bus)
            try:
                await svc.submit_action(user2, round_.id, {"move": "cooperate"})
                return "submit_won"
            except ConflictError:
                return "submit_noop"

    results = await asyncio.gather(deadline_path(), submit_path())
    wins = [r for r in results if r.endswith("_won")]
    assert len(wins) == 1

    async with session_factory() as s:
        r = await s.get(Round, round_.id)
        assert r.status == RoundStatus.COMPLETED

    round_events = [e for e in bus.captured if e.round_id == round_.id]
    assert len(round_events) == 1
```

Relies on SQLite WAL single-writer serialization. The `tournament_db` fixture enforces `PRAGMA journal_mode=WAL` non-negotiably.

#### Cancel cascade tests

**`test_cancel_cascade_complete.py`** — parametrized across all three cancel paths:

```python
@pytest.mark.parametrize(
    "cancel_path", ["user_initiated", "pending_timeout", "abandoned"]
)
async def test_cascade_full_effect(session_factory, factory, bus, cancel_path):
    t = await factory.tournament_with_one_completed_round()

    async with session_factory() as session:
        svc = TournamentService(session, bus)
        if cancel_path == "user_initiated":
            await svc.cancel_tournament(user=factory.admin, tournament_id=t.id)
        elif cancel_path == "pending_timeout":
            await svc.cancel_tournament_system(
                tournament_id=t.id, reason=CancelReason.PENDING_TIMEOUT
            )
        else:  # abandoned
            await svc.leave(tournament_id=t.id, user=factory.last_participant)

    async with session_factory() as verify:
        refreshed = await verify.get(Tournament, t.id)
        assert refreshed.status == TournamentStatus.CANCELLED
        assert refreshed.cancelled_at is not None

        expected_reason = {
            "user_initiated": CancelReason.ADMIN_ACTION,
            "pending_timeout": CancelReason.PENDING_TIMEOUT,
            "abandoned": CancelReason.ABANDONED,
        }[cancel_path]
        assert refreshed.cancelled_reason == expected_reason

        if cancel_path == "user_initiated":
            assert refreshed.cancelled_by == factory.admin.id
        else:
            assert refreshed.cancelled_by is None

        completed_count = await verify.scalar(
            select(func.count()).select_from(Round)
              .where(Round.tournament_id == t.id)
              .where(Round.status == RoundStatus.COMPLETED)
        )
        assert completed_count == 1

        unreleased = await verify.scalar(
            select(func.count()).select_from(Participant)
              .where(Participant.tournament_id == t.id)
              .where(Participant.released_at.is_(None))
        )
        assert unreleased == 0

    cancel_events = [
        e for e in bus.captured
        if isinstance(e, TournamentCancelEvent) and e.tournament_id == t.id
    ]
    assert len(cancel_events) == 1
    assert cancel_events[0].final_rounds_played == 1
```

**`test_cancel_idempotent.py`** — second cancel on already-CANCELLED: zero UPDATE statements (verified via `event.listens_for(session, "after_execute")`), zero additional events published.

**`test_cancel_publish_failure_returns_success.py`**:

```python
async def test_cancel_succeeds_when_bus_publish_raises(
    tournament_db, factory, caplog
):
    failing_bus = MagicMock()
    failing_bus.publish = AsyncMock(side_effect=ConnectionError("bus down"))

    async with session_factory() as session:
        svc = TournamentService(session, failing_bus)
        # Must NOT raise
        await svc.cancel_tournament(user=factory.admin, tournament_id=t.id)

    async with session_factory() as verify:
        refreshed = await verify.get(Tournament, t.id)
        assert refreshed.status == TournamentStatus.CANCELLED

    warn_records = [
        r for r in caplog.records
        if r.event == "tournament.cancel.publish_failed"
        and r.levelname == "WARNING"
    ]
    assert len(warn_records) == 1
```

#### Session sync tests

**`test_session_sync_on_join.py`** — `session_sync` is the first notification delivered to a new subscriber.

**`test_session_sync_closes_commit_publish_gap.py`** — the narrow race from source spec §MCP server: monkeypatch `bus.publish` to sleep 100ms, reconnect during that window, assert `session_sync` carries post-commit state.

### E2E layer

**Path:** `tests/e2e/dashboard/tournament/`

**Hard prerequisite:** vertical slice Phase 0 Tasks 0.1 and 0.2 (both verified).

#### E2E fixtures

- `tournament_app` — full FastAPI instance with MCP sub-app mounted under `/mcp`, `JWTUserStateMiddleware` wired, deadline worker launched in lifespan, SQLite WAL DB with `alembic upgrade head` applied.
- `mcp_test_client(tournament_app, jwt_token)` — async context manager returning an `MCPAdapter` connected over real SSE (not in-memory ASGI).
- `test_jwt` — factory for signed JWTs with configurable `user_id` and TTL.

#### Lifecycle tests

**`test_e2e_30_round_pd_with_reconnect.py`** — SC-1 CI gate.

1. Admin creates a PD tournament (30 rounds, 2 players, `round_deadline_s=1` for test wall-clock economy).
2. Two `MCPAdapter` clients connect via MCP SSE and call `join_tournament`.
3. Loop for 30 rounds: both clients await `round_started`, call `make_move(cooperate)`, await `round_ended`.
4. **Mid-tournament reconnect** at round 15: client 1 drops SSE transport and reconnects with a fresh session. First notification after reconnect must be `session_sync` with `round_number=15`.
5. Tournament continues to round 30.
6. Both clients receive `tournament_completed`.
7. Final payoffs match `cooperate-cooperate × 30`.

Target wall-clock budget: ~30–40 seconds at `round_deadline_s=1` + reconnect sequence + setup/teardown.

**`test_e2e_90_round_pd_benchmark.py`** — AD-9 flagship formula verification, **not in CI**.

Manual benchmark variant: 90 rounds × 30 seconds (per AD-9 default formula), mid-tournament reconnect at round 45. Designed to be run locally before release to confirm the full flagship path holds. Marked `@pytest.mark.benchmark` and skipped by default (`pytest --benchmark` or equivalent marker selection to include). Expected runtime ~45 minutes — unsuitable for CI e2e budget.

**`test_e2e_deadline_timeout_default.py`** — SC-2:

1. Admin creates 3-round PD, `round_deadline_s=2`.
2. Two clients join.
3. First round: only client 1 submits.
4. Wait `POLL_INTERVAL_S + 2 + 1` seconds.
5. Both clients receive `round_ended` with `actions_by_participant[client_2_id]["source"] == "timeout_default"`.
6. DB inspection: `Action.source == ActionSource.TIMEOUT_DEFAULT.value`.

**`test_e2e_user_cancel_via_mcp.py`** — part of SC-8 (MCP half):

1. Admin creates, two clients join, play 1 round.
2. Admin calls `cancel_tournament` MCP tool.
3. Both clients receive `tournament_cancelled` with `reason=admin_action`, `cancelled_by=<admin_id>`, `final_rounds_played=1`.
4. Subsequent `make_move` → `ToolError(409_conflict, "tournament is cancelled")`.

**`test_e2e_rest_admin_curl_path.py`** — the REST half of SC-8:

```python
async def test_rest_cancel_via_bearer_token(tournament_app, test_jwt, test_tournament):
    jwt = test_jwt(user_id=admin.id, ttl_minutes=60)
    async with httpx.AsyncClient(
        app=tournament_app, base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/v1/tournaments/{test_tournament.id}/cancel",
            headers={"Authorization": f"Bearer {jwt}"},
        )
    assert response.status_code == 200

    async with session_factory() as s:
        refreshed = await s.get(Tournament, test_tournament.id)
        assert refreshed.status == TournamentStatus.CANCELLED
        assert refreshed.cancelled_reason == CancelReason.ADMIN_ACTION
```

**`test_e2e_pending_timeout_autocancel.py`** — SC-3 (uses monkeypatch, not per-tournament parameter):

```python
async def test_pending_tournament_autocancels_after_timeout(
    tournament_app, monkeypatch
):
    monkeypatch.setattr(
        "atp.dashboard.tournament.deadlines.TOURNAMENT_PENDING_MAX_WAIT_S", 2
    )
    t = await create_tournament_via_rest(num_players=4)

    await asyncio.sleep(2 + POLL_INTERVAL_S + 1)

    async with session_factory() as s:
        refreshed = await s.get(Tournament, t.id)
        assert refreshed.status == TournamentStatus.CANCELLED
        assert refreshed.cancelled_reason == CancelReason.PENDING_TIMEOUT
        assert refreshed.cancelled_by is None
```

#### Reconnect / recovery tests

**`test_e2e_reconnect_session_sync.py`** — focused reconnect test (companion to the 90-round test):

1. Tournament advanced to round 2 of 3.
2. Client connected, received `round_started` for round 2.
3. Drop SSE transport.
4. Reconnect with new MCP session, call `join_tournament` (idempotent re-join).
5. First notification after reconnect is `session_sync` with `round_number=2`.
6. `make_move` in round 2 still works, round resolves normally.

#### AD-9 / AD-10 enforcement end-to-end

**`test_e2e_ad9_duration_cap_validation.py`** — SC-3 complement:

```python
async def test_create_tournament_rejects_over_cap(tournament_app, test_jwt):
    jwt = test_jwt(user_id=admin.id)
    async with httpx.AsyncClient(
        app=tournament_app, base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/tournaments",
            headers={"Authorization": f"Bearer {jwt}"},
            json={
                "game_type": "prisoners_dilemma",
                "num_players": 2,
                "total_rounds": 200,
                "round_deadline_s": 30,
            },
        )
    assert response.status_code == 422
    assert "max duration" in response.json()["detail"].lower()
```

**`test_e2e_ad10_join_token_enforcement.py`** — SC-4 complement: private tournament with `join_token`, clients without/wrong/correct token, `get_tournament` for non-participants never exposes the token.

**`test_e2e_ad10_one_active_per_user.py`** — SC-4 main path: user joins tournament A; second join to tournament B fails; A completes; third join to B succeeds.

### Fixtures and factories

**Path:** `tests/dashboard/tournament/conftest.py`

#### Database fixture

```python
@pytest.fixture
async def tournament_db(tmp_path):
    """SQLite WAL-mode DB with full Plan 2a schema applied.

    WAL mode is NON-NEGOTIABLE — the deadline worker race tests rely on
    WAL's single-writer serialization for deterministic outcomes.
    """
    db_path = tmp_path / "plan2a_test.db"

    sync_engine = create_engine(f"sqlite:///{db_path}")
    with sync_engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        # FK enforcement ON by default — matches production configuration
        # and protects most tests from accidentally seeding orphan rows.
        # The P2 probe negative test explicitly toggles FK=OFF to seed an
        # orphan row, because P2 exists to catch orphans that SQLite-without-
        # FK-enforcement would silently accept on production.
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.commit()
    sync_engine.dispose()

    env = {**os.environ, "ATP_DATABASE_URL": f"sqlite:///{db_path}"}
    result = subprocess.run(
        ["uv", "run", "alembic", "-c",
         "migrations/dashboard/alembic.ini", "upgrade", "head"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"alembic upgrade failed: {result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    yield engine
    await engine.dispose()
```

#### Event bus test double

```python
# tests/dashboard/tournament/test_bus.py
class TestEventBus(TournamentEventBus):
    """In-memory bus subclass that records all publishes for assertion."""
    def __init__(self) -> None:
        super().__init__()
        self.captured: list[TournamentEvent] = []

    async def publish(self, event: TournamentEvent) -> None:
        self.captured.append(event)
        await super().publish(event)
```

#### Frozen clock

```python
@pytest.fixture
def frozen_clock():
    with freezegun.freeze_time("2026-04-15 10:00:00") as clock:
        yield clock
```

`freezegun` added to `pyproject.toml` `[dependency-groups.dev]` as part of Plan 2a implementation.

#### Deadline worker fixture for e2e

```python
@pytest.fixture
async def deadline_worker_task(tournament_db, captured_bus):
    """Spawn the production deadline worker as a background asyncio task."""
    shutdown = asyncio.Event()
    session_factory = async_sessionmaker(tournament_db)

    task = asyncio.create_task(
        run_deadline_worker(
            session_factory, captured_bus, shutdown_event=shutdown
        )
    )
    yield task
    shutdown.set()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)
```

### Acceptance criteria → test mapping

Mirror of Section 1 SC numbering. Every SC is covered by tests in at least one layer; most by tests in at least two layers.

| SC # | Section 1 criterion | Unit | Integration | E2E |
|---|---|---|---|---|
| **SC-1** | Two `MCPAdapter` bots run a multi-round PD tournament end-to-end with mid-tournament reconnect → `session_sync` restores baseline. 30-round variant in CI; 90-round variant as manual pre-release benchmark. | `test_tournament_cancel_event_post_init` (event shape invariant) | `test_session_sync_on_join`, `test_session_sync_closes_commit_publish_gap` | `test_e2e_30_round_pd_with_reconnect` (CI), `test_e2e_reconnect_session_sync`, `test_e2e_90_round_pd_benchmark` (manual) |
| **SC-2** | Tournament with one non-submitting player resolves via deadline worker with `source=timeout_default` | `test_deadline_tick_isolation`, `test_deadline_tick_two_paths` | `test_deadline_worker_race` | `test_e2e_deadline_timeout_default` |
| **SC-3** | AD-9 pending auto-cancel | `test_deadline_tick_two_paths` (Path 2) | `test_cancel_cascade_complete[pending_timeout]` | `test_e2e_pending_timeout_autocancel`, `test_e2e_ad9_duration_cap_validation` |
| **SC-4** | AD-10 concurrent join — exactly one success, one ConflictError | — | `test_idempotent_join_concurrent`, `test_partial_unique_index` | `test_e2e_ad10_one_active_per_user`, `test_e2e_ad10_join_token_enforcement` |
| **SC-5** | Idempotent join — same Participant row, fresh session_sync | `test_cancel_impl_logic` (idempotency pattern) | `test_session_sync_on_join` (verifies session_sync on both first join and re-join) | `test_e2e_reconnect_session_sync` |
| **SC-6** | Idempotent cancel — noop or 404, never 500 | `test_cancel_impl_logic` (already-CANCELLED case), `test_load_for_auth` | `test_cancel_idempotent`, `test_cancel_publish_failure_returns_success` | `test_e2e_user_cancel_via_mcp` (second call path) |
| **SC-7** | Probe dry-run via `python -m ...` returns exit 0 on clean | — | `test_migration_probe_cli_entrypoint`, 6 probe positive/negative tests | — |
| **SC-8** | Operator cancels via curl REST, 200, event in logs | `test_static_guards::test_cancel_tournament_system_not_called_from_handlers` | `test_cancel_cascade_complete[user_initiated]`, `test_cancel_publish_failure_returns_success` | `test_e2e_rest_admin_curl_path`, `test_e2e_user_cancel_via_mcp` |
| **SC-9** | Fresh Alembic upgrade produces 7 new constraints/indexes + 8 new columns, zero drift | `test_static_guards::test_no_bare_string_round_status_comparisons` | `test_migration_upgrade_downgrade_round_trip`, `test_cancel_check_constraint`, `test_partial_unique_index` | — |
| **SC-10** | All vertical slice tests continue to pass | CI gate | CI gate | CI gate |

**Coverage gate:** every SC row must have at least one green test across its declared layers on every PR touching `packages/atp-dashboard/atp/dashboard/tournament/` or the migration. Missing test-to-SC declaration for any new test in these directories → CI warning.

### Coverage targets

| Module | Line coverage target | Rationale |
|---|---|---|
| `tournament/service.py` (new + updated methods) | ≥ 95% | Critical path; every branch of `_cancel_impl`, `_load_for_auth`, and three cancel call sites must be exercised. |
| `tournament/deadlines.py` | ≥ 90% | Worker isolation logic + two paths. |
| `migrations/probes/check_tournament_invariants.py` | **100%** | Each of the 6 probes gets positive + negative test. |
| `tournament/reasons.py` | 100% | 3-value StrEnum; trivially 100% via any import. |
| `tournament/events.py` (new types + `TournamentCancelEvent`) | ≥ 95% | `__post_init__` validator has a dedicated parametrized test. |
| `tournament/models.py` (new columns + enums) | n/a — declarative | Verified via integration tests. |
| `mcp/tools.py` (new tools) | ≥ 85% | Happy paths + auth failures; unhappy paths via e2e. |
| `v2/routes/tournament_api.py` | ≥ 85% | Thin wrappers; primary value in service layer tests. |

**Global floor:** `coverage report --fail-under=85` on the `atp.dashboard.tournament` package.

### CI integration

```yaml
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: uv sync --group dev
      - name: Unit tests + static guards (target <30s wall)
        run: uv run pytest tests/unit/dashboard/tournament -v --cov=atp.dashboard.tournament --cov-report=xml

  integration:
    runs-on: ubuntu-latest
    needs: unit
    steps:
      - uses: actions/checkout@v4
      - run: uv sync --group dev
      - name: Integration tests (target <5min wall)
        run: uv run pytest tests/integration/dashboard/tournament -v --cov=atp.dashboard.tournament --cov-append --cov-report=xml

  e2e:
    runs-on: ubuntu-latest
    needs: integration
    if: github.event_name == 'push' || contains(github.event.pull_request.labels.*.name, 'tournament-e2e')
    steps:
      - uses: actions/checkout@v4
      - run: uv sync --group dev
      - name: E2E tests (target <10min wall)
        run: uv run pytest tests/e2e/dashboard/tournament -v --cov=atp.dashboard.tournament --cov-append --cov-report=xml

  coverage-gate:
    runs-on: ubuntu-latest
    needs: [unit, integration, e2e]
    steps:
      - name: Enforce 85% floor
        run: uv run coverage report --fail-under=85
```

**Why static guards run in the unit stage:** they use `git grep`, need no DB, no fixtures, and must fail before any slower test burns CI minutes.

**Why e2e is conditional:** e2e tests cost real wall clock (~10 minutes for the 90-round test alone) and require the Phase 0 verified stack. Small refactor PRs that don't touch tournament code should not pay this cost on every push. The `tournament-e2e` label opts a PR in; main branch push always runs e2e.

### Out of scope for testing

| Area | Deferred why | Trigger to add |
|---|---|---|
| Load testing (100+ concurrent tournaments) | Single-worker SQLite scale | Backlog L |
| Bus replay / event sourcing | `session_sync` is the recovery mechanism | Backlog C |
| MCP protocol fuzzing | FastMCP responsibility | Vendor CVE |
| N² notification formatter perf | Only matters at `num_players ≥ 8` | Backlog X |
| Tournament-scoped JWT (AD-9 alternative) | AD-9 hard cap sufficient | Backlog V |
| Multi-worker deadline coordination | WEB_CONCURRENCY=1 enforced | Backlog I |
| PostgreSQL migration correctness | Plan 2a ships SQLite-only | Backlog I |
| Dashboard UI interaction tests | Plan 2b scope | Plan 2b |

---

## Document trailer

**Authoritative naming:** `created_by` (legacy from vertical slice), `cancelled_by` (new in Plan 2a, symmetric with `created_by`). Audit FK convergence with dashboard-wide `_by_id` precedent tracked as backlog item AC.

**Source spec:** `docs/superpowers/specs/2026-04-10-mcp-tournament-server-design.md` — all seven patches from `docs/atp-mcp-tournament-spec-patches.md` already merged into the source. Plan 2a references source spec §sections directly, not patch numbers.

**Revision chain:** Plan 2a migration sets `down_revision = "028d8a9fdc46"` (tournament slice columns). Transitively follows `c8d5f2a91234` (IDOR fix).

**Next step:** implementation plan written via `superpowers:writing-plans` skill, saved to `docs/superpowers/plans/2026-04-11-mcp-tournament-plan-2a.md`.
