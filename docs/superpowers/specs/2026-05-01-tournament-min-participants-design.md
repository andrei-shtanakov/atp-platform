# Tournament Autostart with No-Show Placeholders — Design

**Status:** Draft (rev 3 — second-round review incorporated)
**Date:** 2026-05-01
**Owner:** ATP platform
**Scope:** `el_farol`, `public_goods` tournament types only

## Problem

Today a tournament transitions `PENDING → ACTIVE` only when `count(participants) == num_players`
(`packages/atp-dashboard/atp/dashboard/tournament/service.py:600`). If the configured
`pending_deadline` (`ATP_TOURNAMENT_PENDING_MAX_WAIT_S`, default 300 s) elapses before that
equality holds, the deadline worker cancels the tournament with
`cancelled_reason = PENDING_TIMEOUT`
(`packages/atp-dashboard/atp/dashboard/tournament/deadlines.py:107-146`).

This is brittle for multi-player games. A tournament configured for 20 El Farol bots fails
entirely when even one participant has a transient connectivity problem during the join window.
The remaining 19 successfully-connected participants get nothing, and the operator must recreate
the tournament from scratch.

## Goal

Allow the creator to opt into a "minimum live participants" threshold below `num_players`. If at
least that many participants have joined when `pending_deadline` expires, the tournament starts
with the missing slots filled by deterministic placeholder participants who play a stable no-op
strategy each round.

Out of scope:

- 2-player games (`prisoners_dilemma`, `stag_hunt`, `battle_of_sexes`) — the math is degenerate
  and a 1-real-vs-1-placeholder match is not a tournament.
- Mid-game leave handling — if a real participant leaves an `ACTIVE` tournament and drops the
  live count below `min_participants`, the existing `ABANDONED` cascade still applies. Adding
  mid-game no-show conversion is a possible follow-up.
- Re-running cancelled tournaments — operators recreate via the existing API.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Apply only to `el_farol` and `public_goods` | Other game types are exactly-2-players where threshold relaxation is degenerate. |
| 2 | Reuse existing `Participant.builtin_strategy` infrastructure | Placeholder = special builtin (`<game>/no_show`). No engine, runner, or evaluator changes. |
| 3 | Per-tournament `min_participants` field, default NULL = `num_players` | Explicit per-tournament control beats a global magic constant. NULL preserves legacy behavior. |
| 4 | Hard floor `min_participants >= 2` | A 1-real-player game is meaningless even in El Farol. |
| 5 | No-show actions: `el_farol → stay_home`, `public_goods → contribute=0` | Deterministic stable action that maps intuitively to "didn't show up". Welfare implications are game-specific (free-riding for PG; for EF they may even *help* surviving players when the bar would otherwise overflow) and explicitly documented, not framed as a penalty signal. |
| 6 | Derived `was_no_show` / `kind` field in API responses, no new column on `Participant` | UI can render three categories (user / builtin / no-show) without doubling the source of truth. |
| 7 | Fill logic lives in the existing `deadline_worker` tick | Reuses the already-running scan; no new background job. |
| 8 | `agent_name = "missed-N"` is a convention, not a reserved namespace | Programmatic disambiguation MUST use the `kind` field. We do not validate against real users naming an agent `missed-1`; the data fields (`user_id`, `agent_id`) and derived `kind` already make rows unambiguous. |
| 9 | Single deadline-worker assumption made explicit | Today's deploy runs one container with one deadline-worker task in lifespan. Defensive `SELECT … FOR UPDATE` on the Tournament row is added as cheap insurance against future multi-replica deploys. |

## Architecture

```
deadline_worker tick (deadlines.py)
  └─ for tournament_id in tournaments where pending_deadline < now():
       ├─ open per-tournament session
       ├─ if game_type in {el_farol, public_goods}:
       │     service.try_autostart_or_cancel(tournament_id)   # NEW
       │       (internally: lock + re-check + fill OR cancel)
       └─ else:
             service.cancel_tournament_system(t.id, PENDING_TIMEOUT)   # existing
```

`try_autostart_or_cancel(tournament_id)` (new method on `TournamentService`):

1. `tournament = await session.get(Tournament, tournament_id, with_for_update=True)` and
   `if tournament is None: raise NotFoundError(...)`. Matches the existing codebase pattern
   (e.g. `service.py:1884`). The `with_for_update=True` flag locks the row for the rest of
   the transaction, closing the read-then-write race against any concurrent `join()` (which
   inserts a `Participant` and may call `_start_tournament` itself when
   `count == num_players`).
2. Re-check `tournament.status == PENDING`. If not (concurrent join already started it, or a
   user cancel ran), return — nothing to do.
3. `live_count = SELECT count(*) FROM tournament_participants WHERE tournament_id = ? AND
   released_at IS NULL` — **MUST filter `released_at IS NULL`**. `leave()` (`service.py:1636`)
   sets `released_at = now()` without removing the row, so a join→leave inside the PENDING
   window leaves a phantom that would otherwise inflate the count.
4. `threshold = tournament.min_participants OR tournament.num_players`.
5. Branch:
   - `live_count >= threshold` → call `_fill_no_shows_and_start(tournament, missing=num_players − live_count)`.
   - `live_count < threshold` → call `cancel_tournament_system(tournament_id, PENDING_TIMEOUT)`.
6. Commit semantics:
   - **Cancel branch:** `cancel_tournament_system` does not commit (matches existing
     `deadlines.py:141` pattern). The outer `await session.commit()` in the worker
     completes the transaction.
   - **Fill branch:** `_fill_no_shows_and_start` calls `_start_tournament`, which already
     commits internally at `service.py:729` *before* publishing `round_started` (the LABS-74
     invariant — subscribers must see the persisted state). The outer
     `await session.commit()` in the worker is a no-op for this branch but kept symmetrically
     for code clarity.

`_fill_no_shows_and_start(tournament, missing)` (private helper):

1. Insert `missing` `Participant` rows with:
   - `user_id = NULL`
   - `agent_id = NULL`
   - `agent_name = f"missed-{i}"` for `i = 1..missing`, monotonic by insertion order
   - `builtin_strategy = f"{game_type}/no_show"`
2. Call existing `_start_tournament(tournament, no_show_fill_count=missing)` (new kwarg —
   see Event Payload section). It flips `PENDING → ACTIVE`, creates round 1, commits, and
   publishes `round_started` augmented with no-show metadata.

The lock acquired in step 1 of `try_autostart_or_cancel` is held until the outer commit, so the
sequence count→insert→start is atomic against any concurrent `join()` on the same tournament.

## Data Model

**New column** `tournaments.min_participants INTEGER NULL`:

- `NULL` = legacy and default. Treated as `num_players` everywhere → preserves
  "all-or-nothing" behavior for unmodified clients.
- Set on create; immutable afterward (no PATCH).
- Alembic migration required (per project convention: `create_all()` does not `ALTER`).

**No changes** to `Participant` — `builtin_strategy` already exists and is what the no-show
fill writes into. The `kind` / `was_no_show` API fields are derived at serialization time. The
existing `ck_participants_agent_xor_builtin` CHECK constraint
(`packages/atp-dashboard/atp/dashboard/tournament/models.py:280`) already permits
`(agent_id IS NULL, builtin_strategy IS NOT NULL)`. The partial unique
`uq_participant_tournament_agent WHERE agent_id IS NOT NULL`
(`models.py:251-256`) does not constrain no-show rows, which is what we want — multiple
no-shows can coexist in one tournament.

## Builtin Registration

Add to `atp-games/atp_games/`:

- `el_farol/no_show` — strategy class returning `stay_home` (action=0) every round.
- `public_goods/no_show` — strategy class returning `contribute=0` every round.

Register via the existing `BUILTIN_REGISTRY` mechanism. The resolver at
`packages/atp-dashboard/atp/dashboard/tournament/service.py:302` (`resolve_builtin`) picks
them up unchanged. Treat them like any other builtin for the purpose of `roster` validation —
they are valid `<game>/<name>` entries and can in principle be specified in `roster=[...]`
explicitly (no need to special-case prevent that).

## API Surface

### Request

`CreateTournamentRequest`
(`packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py:109`) gains:

```python
min_participants: int | None = Field(default=None, ge=2)
```

UI form (`templates/ui/tournament_new.html` rendered from
`packages/atp-dashboard/atp/dashboard/v2/routes/ui.py:1066-1100`) gains a corresponding optional
input, surfaced only when `game_type` is `el_farol` or `public_goods`. Help text:
`"default = num_players (all-or-nothing)"`.

A small inline `<script>` clears the `min_participants` input value when the user switches
`game_type` away from `el_farol`/`public_goods`. The backend keeps strict 422 validation;
mercy-mode (silently dropping the field for unsupported game types) is rejected because it
hides the user's intent and produces a tournament behaviorally different from what the form
showed.

### Validation (in `service.create_tournament`)

```python
if min_participants is not None:
    if game_type not in ("el_farol", "public_goods"):
        raise ValidationError(
            "min_participants is only supported for el_farol and public_goods"
        )
    if not (2 <= min_participants <= num_players):
        raise ValidationError(
            f"min_participants must satisfy 2 <= min_participants <= num_players "
            f"({num_players}); got {min_participants}"
        )
```

Surfaces as HTTP 422 via the existing `ValidationError → HTTPException` mapping in the API
layer.

### Response

`_serialize` (`tournament_api.py:129`) gains:

```python
"min_participants": t.min_participants,   # int | None
```

`participants` endpoint per-row dict (`tournament_api.py` around line 296) gains:

```python
row["kind"] = (
    "no_show" if (p.builtin_strategy or "").endswith("/no_show")
    else "builtin" if p.builtin_strategy
    else "user"
)
row["was_no_show"] = row["kind"] == "no_show"
```

Both additions are additive; legacy clients reading `id` / `user_id` / `agent_name` are
unaffected.

## Event Payload

`round_started` is published from **two** sites, not one:

- `_start_tournament` for round 1 (`service.py:730`).
- `_resolve_round` for the next-round transition after each round resolution
  (`service.py:1438`).

Today both publish `data={"total_rounds": tournament.total_rounds}`. There is **no separate
`tournament_started` event** in `EventType` (`events.py:28-33`); creating one would force
every SSE/MCP subscriber to handle a new type for marginal benefit. Instead, augment the
existing round-started payload via a shared helper applied at *both* sites — so subscribers
can read fields directly without `event.data.get(..., default)` defensiveness.

New helper on `TournamentService`:

```python
@staticmethod
def _round_started_payload(
    tournament: Tournament,
    *,
    no_show_fill_count: int = 0,
) -> dict[str, Any]:
    return {
        "total_rounds": tournament.total_rounds,
        "had_no_show_fill": no_show_fill_count > 0,
        "no_show_count": no_show_fill_count,
    }
```

`_start_tournament(tournament)` gains a kwarg:

```python
async def _start_tournament(
    self,
    tournament: Tournament,
    *,
    no_show_fill_count: int = 0,
) -> None:
    ...
    data = self._round_started_payload(
        tournament, no_show_fill_count=no_show_fill_count
    )
```

`_resolve_round` next-round publish (`service.py:1436-1444`) switches to the helper with
default `no_show_fill_count=0`. Rounds 2+ thus always carry `had_no_show_fill=False`,
`no_show_count=0`. Subscribers (`tournament_live.py:221` already ignores `data`;
`notifications.py:140` only reads `total_rounds`) remain compatible — the additions are
strictly additive within `data`.

The deadline worker also emits `log.info("deadline_worker.no_show_fill", extra={
"tournament_id": ..., "missing": ..., "threshold": ..., "live_count": ...})` for ops
visibility distinct from the existing `deadline_worker.tick_complete` summary line.

## Edge Cases & Invariants

| Case | Behavior |
|---|---|
| `min_participants is None` (default) | Identical to today: fill never fires; expired pending → cancel. |
| `count >= num_players` before deadline | Existing `_start_tournament()` in `join()` fires; deadline worker never visits. `had_no_show_fill=False`. |
| `count < min_participants` at deadline | `cancel_tournament_system(PENDING_TIMEOUT)` — current behavior. |
| `count == min_participants == num_players − k` at deadline (k ≥ 1) | Fill inserts k no-shows, transitions to ACTIVE, `had_no_show_fill=True`, `no_show_count=k`. |
| All slots pre-filled by `roster` at create | `create_tournament` already starts the tournament inline (`service.py:399`); deadline worker never runs the fill path. |
| **Partial roster, `len(roster) >= min_participants`, no live joins arrive** | At `pending_deadline`, `live_count = len(roster) >= min_participants` → fill `num_players − len(roster)` no-shows, start. (For private tournaments this case is already blocked by the creator-commit check at `service.py:319-329`; it is reachable only for public tournaments.) |
| `count < min_participants` because of `join → leave` in PENDING | `released_at IS NULL` filter excludes the abandoned row, so the threshold sees only truly-live joins. Tournament cancels, not starts. |
| Private tournament + low `min_participants` | Existing creator-commit check (`service.py:308-329`) is unchanged: it gates *creation*, not *start*. Creator must still own a tournament-purpose agent OR fill the entire `roster` at create time. |
| `mid-game leave()` drops live below `min_participants` | Out of scope for this iteration. Existing `ABANDONED` cascade applies as today. |
| AD-9 duration cap (`service.py:268-279`) | No change — `max_wall_clock = TOURNAMENT_PENDING_MAX_WAIT_S + total_rounds * round_deadline_s` already accounts for the full pending window; fill consumes that window without lengthening it. |
| Concurrent `join()` racing the deadline tick | `SELECT … FOR UPDATE` on the Tournament row inside `try_autostart_or_cancel` serializes against `join()`. After the lock, status and live_count are re-read; if join already flipped status to ACTIVE, the worker returns without action. |
| `_release_participants` runs at `tournament_completed` | Sets `released_at` on every Participant including no-shows. Expected and tested. |
| User registers a real agent named `missed-1` | Allowed. The Participant row has `user_id != NULL`, `agent_id != NULL`, derived `kind="user"`. No data ambiguity; only cosmetic overlap in `agent_name`. Documented in API docs. |

## Concurrency Model

This design assumes **a single deadline-worker instance** (one `asyncio` task in the FastAPI
lifespan, today running on a single VPS container — see `deadlines.py:1-9`). Operations that
remain safe under this assumption:

- `try_autostart_or_cancel` per tournament_id is serial within a tick (loop iterates).
- Re-entry safety across consecutive ticks is guaranteed by the `status == PENDING`
  re-check after the row lock — a tournament that started or cancelled in a prior tick is
  skipped.

`SELECT … FOR UPDATE` is included as **defensive insurance** for a possible future
multi-replica deploy. It is not strictly required today but adds negligible cost and turns a
silent correctness bug into a clean serialization on the row. If we ever scale out, the
contract for the worker becomes: at most one fill+start transaction per Tournament at a time.

### `join()` race against the new ACTIVE-flip path

`join()` reads the Tournament row at `service.py:440` via `session.get(...)` **without** a
lock and validates `tournament.status == PENDING` at `service.py:499` from that cached
object. The Participant `INSERT` runs later (`service.py:592-594`). The INSERT triggers an
implicit FK lock (`FOR KEY SHARE` on the parent Tournament row in Postgres), which serializes
against the worker's `FOR UPDATE`.

Pre-rev-3 this race was unreachable because the worker only `cancel`led — the late INSERT
into a CANCELLED tournament was rejected by the existing `status == PENDING` check, and
the FK lock contention never produced a wrong outcome. Rev 3 introduces the new active-flip
path: the worker can transition `PENDING → ACTIVE`, commit, and release the lock. A
concurrent `join()` whose status check passed *before* the worker took the lock will, after
the lock releases, INSERT a Participant into an ACTIVE tournament — bypassing the cached
status check.

This is reachable. The race window is narrow (single asyncio loop, same process), but
**ignoring it would let the new feature ship a known correctness bug**. The fix lives on the
join side because the root cause is join's stale cached read, not the worker's design:

```python
# In TournamentService.join, AFTER the successful flush() at service.py:594,
# BEFORE the count == num_players check at service.py:600.
await self._session.refresh(tournament)
if tournament.status != TournamentStatus.PENDING:
    await self._session.rollback()
    raise ConflictError(
        f"tournament {tournament_id} flipped to {tournament.status!r} "
        "concurrently — try again or accept the autostart"
    )
```

Mechanically: `refresh()` re-reads the locked-and-released row; if the worker flipped status
under the lock, join() now sees ACTIVE and rolls back its own INSERT. Same exception type as
the existing pre-INSERT status check at `service.py:499-503`, so callers (REST + MCP) need
no new handling — they already retry/surface 409 cleanly.

This fix is **in scope for this rev** because the feature opens the race; deferring it would
ship a regression. It is also the only place we touch the existing `join()` flow.

## Test Plan

**Unit (`tests/unit/dashboard/tournament/`)**

- `test_service_create.py`:
  - `min_participants is None` → row stored as NULL, behavior unchanged.
  - `min_participants` set on `prisoners_dilemma` → 422.
  - `min_participants = 1` → 422 (below floor).
  - `min_participants > num_players` → 422.
  - `min_participants = num_players` → ok, equivalent to default.
- `test_service_fill.py` (new):
  - Inserts exactly `num_players − live_count` rows.
  - `agent_name` is `missed-1..missed-N` in insertion order.
  - `builtin_strategy` is the correct `<game>/no_show` value.
  - `user_id IS NULL`, `agent_id IS NULL`.
  - Tournament status flips PENDING → ACTIVE.
  - Round 1 is created.
  - `round_started` event published with `had_no_show_fill=True`,
    `no_show_count=missing`.
  - Filter `released_at IS NULL`: pre-seed a Participant with `released_at` set, verify it is
    not counted toward `live_count`.
- `test_service_start_event.py` (new):
  - Regression guard for default event payload — `_start_tournament` called WITHOUT
    `no_show_fill_count` publishes `round_started` with `had_no_show_fill=False`,
    `no_show_count=0`. Catches future payload migrations that drop the defaults.
  - **Same guard for `_resolve_round` next-round publish** — round 2+ `round_started` events
    must always carry `had_no_show_fill=False`, `no_show_count=0` via the shared
    `_round_started_payload` helper. Direct dict access (no `.get(..., default)`) is the
    contract.
- `test_deadlines.py` (extend):
  - Expired pending with `live_count >= min_participants` AND game in {el_farol,
    public_goods} → fill + start, no cancel.
  - Expired pending with `live_count < min_participants` → cancel as today.
  - Expired pending with `min_participants is None` → cancel as today (regression guard).
  - Expired pending for `prisoners_dilemma` → cancel as today (scope guard).
  - **Concurrent `join()` during fill** (skipped on SQLite due to no real `FOR UPDATE`
    semantics; runs on Postgres in CI). Documents the lock contract: with both transactions
    in flight, the late `join()` either (a) acquired the FK lock first → worker's
    `try_autostart_or_cancel` re-reads status and skips, no fill happens, OR (b) the worker
    won → `join()`'s defensive `refresh()` (see Concurrency Model) sees ACTIVE and raises
    `ConflictError`, rolling back its INSERT. **`count > num_players` is unreachable in
    either ordering.**
  - **`join()` defensive refresh regression guard** (new test in `test_service_join.py`):
    simulate a flip of `tournament.status` to ACTIVE between `flush()` and the post-flush
    check; verify `join()` raises `ConflictError` and rolls back the participant INSERT.
- `test_api_serialization.py` (extend):
  - `kind="no_show"` and `was_no_show=True` for a Participant whose
    `builtin_strategy="el_farol/no_show"`.
  - `kind="builtin"` for a Participant whose `builtin_strategy="el_farol/grim"` (no
    `/no_show` suffix).
  - `kind="user"` for a Participant with non-null `user_id`.

**Builtin tests (`atp-games/tests/`)**

- `test_no_show_strategies.py`:
  - `el_farol/no_show` returns `stay_home` for many rounds, irrespective of observation.
  - `public_goods/no_show` returns `contribute=0` for many rounds.
  - Both resolve through `resolve_builtin(...)` and round-trip through `BUILTIN_REGISTRY`.

**Integration (`tests/integration/dashboard/`)**

- E2E: create `el_farol` with `num_players=5 min_participants=4`, join 4 agents, advance the
  clock past `pending_deadline`, assert tournament transitioned to ACTIVE, the 5th
  participant exists with `kind="no_show"`, `agent_name="missed-1"`, and the played actions
  across all rounds are `stay_home`.
- E2E completion: same fixture, run the tournament to completion, assert the no-show
  Participant has `released_at` set after `tournament_completed` (mirrors
  `_release_participants` behavior).

**Migration test**

- Alembic upgrade then downgrade on:
  - empty database
  - database with existing tournaments (verify `min_participants IS NULL` for all
    pre-existing rows after upgrade; downgrade drops the column cleanly).

## Documentation Updates

- `TOURNAMENT_CREATION_API.md`: document `min_participants` field, validation rules, and
  no-show fill behavior. Include an example response showing `kind` / `was_no_show` for a
  participant. Note that `agent_name="missed-N"` is a display convention and not a reserved
  identifier.
- `CLAUDE.md`: no env-var change. Add a one-line note under the tournament service
  description pointing at the new behavior.
- `docs/maestro-integration.md`: explicitly grep for `count == num_players` and
  `num_players` to find any lifecycle-invariant assertions; update if it claims this
  equality holds when `status == ACTIVE`.
- Participant kit READMEs (e.g. `packages/.../participant-kit-el-farol-en/`): add a brief
  note that no-show participants may appear in the tournament with `agent_name="missed-N"`
  and `kind="no_show"`. Wire contract is unchanged.

## Migration Safety

- Single additive column with default NULL → safe online ALTER on Postgres.
- No backfill required: NULL means "legacy behavior" everywhere.
- Downgrade simply drops the column; any tournaments created with non-NULL values revert to
  legacy behavior, which is `min_participants = num_players` — strictly stricter than what
  the operator opted into. No data loss, just a degraded UX during the downgrade window.

## Rollout

1. Land migration + service changes behind no flag — additive and default-off.
2. Update docs.
3. Smoke a real El Farol tournament on production with `num_players=5 min_participants=4`,
   intentionally have one participant skip the join window, verify autostart with one
   `missed-1` placeholder and `had_no_show_fill=True` in the round-1 event payload.

## Open Questions (deferred)

- **Mid-game leave → no-show conversion:** if a real participant leaves an `ACTIVE`
  tournament and live count drops below `min_participants`, do we convert their slot to a
  no-show and continue? Requires additional engine-level reasoning per game and is not
  required by the immediate use case.
- **Leaderboard handling of no-shows:** should they be excluded from rankings or shown with
  a marker? Likely a separate UI-only iteration once we have production data.
- **Existing latent bug in `join()` count:** `service.py:595-598` counts Participants
  without filtering `released_at IS NULL`. In the current code path the partial unique
  index on `agent_id` masks the symptom for the same-agent case, but a different agent
  joining after a leave could in theory miscount. Out of scope here (the new fill path is
  defended properly); flagged for a separate fix.
