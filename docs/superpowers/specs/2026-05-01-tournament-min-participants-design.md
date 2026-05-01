# Tournament Autostart with No-Show Placeholders — Design

**Status:** Draft
**Date:** 2026-05-01
**Owner:** ATP platform
**Scope:** `el_farol`, `public_goods` tournament types only

## Problem

Today a tournament transitions `PENDING → ACTIVE` only when `count(participants) == num_players`
(`packages/atp-dashboard/atp/dashboard/tournament/service.py:600`). If the configured
`pending_deadline` (`ATP_TOURNAMENT_PENDING_MAX_WAIT_S`, default 300 s) elapses before that
equality holds, `deadline_worker` cancels the tournament with `cancelled_reason =
PENDING_TIMEOUT` (`deadlines.py:107-136`).

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

- 2-player games (`prisoners_dilemma`, `stag_hunt`, `battle_of_sexes`) — the math is
  degenerate and a 1-real-vs-1-placeholder match is not a tournament.
- Mid-game leave handling — if a real participant leaves an `ACTIVE` tournament and drops
  the live count below `min_participants`, the existing `ABANDONED` cascade still applies.
  Adding mid-game no-show conversion is a possible follow-up.
- Re-running cancelled tournaments — operators recreate via the existing API.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Apply only to `el_farol` and `public_goods` | Other game types are exactly-2-players where 80%-style thresholds are degenerate. |
| 2 | Reuse existing `Participant.builtin_strategy` infrastructure | Placeholder = special builtin (`<game>/no_show`). No engine, runner, or evaluator changes. |
| 3 | Per-tournament `min_participants` field, default = `num_players` | Explicit per-tournament control beats a global magic constant. NULL preserves legacy behavior. |
| 4 | Hard floor `min_participants >= 2` | A 1-real-player game is meaningless even in El Farol. |
| 5 | No-show actions: `el_farol → stay_home`, `public_goods → contribute=0` | "Didn't show up" maps cleanly to "didn't participate". Honestly degrades welfare for survivors, which is the right signal. |
| 6 | Derived `was_no_show` / `kind` field in API responses, no new column on `Participant` | UI can render three categories (user / builtin / no-show) without doubling the source of truth. |
| 7 | Fill logic lives in the existing `deadline_worker` tick | Reuses the already-running scan; no new background job. |

## Architecture

```
deadline_worker tick (deadlines.py)
  └─ for tournament_id in tournaments where pending_deadline < now():
       ├─ live_count = count(Participant where tournament_id=t.id)
       ├─ threshold  = t.min_participants OR t.num_players
       ├─ if game_type in {el_farol, public_goods} AND live_count >= threshold:
       │     service.fill_no_shows_and_start(t)   # NEW
       └─ else:
             service.cancel_tournament_system(t.id, PENDING_TIMEOUT)   # existing
```

`fill_no_shows_and_start()` (new method on `TournamentService`):

1. Compute `missing = num_players - live_count`.
2. Insert `missing` `Participant` rows with:
   - `user_id = NULL`
   - `agent_id = NULL`
   - `agent_name = f"missed-{i}"` (i = 1..missing, monotonic by insertion order)
   - `builtin_strategy = f"{game_type}/no_show"`
3. Call existing `_start_tournament(t)` to flip `PENDING → ACTIVE`, create round 1, and publish
   the `tournament_started` event.

Whole operation is in one session; event publish happens after commit (mirrors current
`_start_tournament` pattern).

## Data Model

**New column** `tournaments.min_participants INTEGER NULL`:

- `NULL` = legacy, treated as `num_players` everywhere → preserves "all-or-nothing" behavior.
- Set on create; immutable afterward.
- Alembic migration required (per project convention: `create_all()` does not `ALTER`).

**No changes** to `Participant` — `builtin_strategy` already exists and is what the no-show
fill writes into. The `kind` / `was_no_show` API fields are derived at serialization time.

## Builtin Registration

Add to `atp-games/atp_games/`:

- `el_farol/no_show` — strategy class returning `stay_home` (action=0) every round.
- `public_goods/no_show` — strategy class returning `contribute=0` every round.

Register via the existing `BUILTIN_REGISTRY` mechanism. The resolver in
`service.py:302` (`resolve_builtin`) picks them up unchanged. Treat them like any other builtin
for the purpose of `roster` validation — they are valid `<game>/<name>` entries and could in
principle be specified explicitly in a `roster=[...]` (no need to special-case prevent that).

## API Surface

### Request

`CreateTournamentRequest` (`tournament_api.py:109`) gains:

```python
min_participants: int | None = Field(default=None, ge=2)
```

UI form (`templates/ui/tournament_new.html` via `ui.py:1100`) gains a corresponding optional
input, surfaced only when `game_type` is `el_farol` or `public_goods`, with help text
`"default = num_players (all-or-nothing)"`.

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

Surfaces as HTTP 422 via existing `ValidationError → HTTPException` mapping in the API layer.

### Response

`_serialize` (`tournament_api.py:129`) gains:

```python
"min_participants": t.min_participants,   # int | None
```

`participants` endpoint per-row dict (`tournament_api.py:307` area) gains:

```python
row["kind"] = (
    "no_show" if (p.builtin_strategy or "").endswith("/no_show")
    else "builtin" if p.builtin_strategy
    else "user"
)
row["was_no_show"] = row["kind"] == "no_show"
```

Both additions are additive; legacy clients reading `id`/`user_id`/`agent_name` are unaffected.

## Edge Cases & Invariants

| Case | Behavior |
|---|---|
| `min_participants is None` (default) | Identical to today: fill never fires; expired pending → cancel. |
| `count >= num_players` before deadline | Existing `_start_tournament()` in `join()` fires; deadline worker never visits. |
| `count < min_participants` at deadline | `cancel_tournament_system(PENDING_TIMEOUT)` — current behavior. |
| `count == min_participants == num_players − k` at deadline (k ≥ 1) | Fill inserts k no-shows, transitions to ACTIVE. |
| All slots pre-filled by `roster` at create | `create_tournament` already starts the tournament inline (`service.py:399`); deadline worker never runs the fill path. |
| Private tournament + low `min_participants` | Existing creator-commit check (`service.py:308-329`) is unchanged: it gates *creation*, not *start*. Creator must still own a tournament-purpose agent OR fill the entire `roster` at create time. |
| `mid-game leave()` drops live below `min_participants` | Out of scope for this iteration. Existing ABANDONED cascade applies as today. |
| AD-9 duration cap (`service.py:268-279`) | No change — `max_wall_clock = TOURNAMENT_PENDING_MAX_WAIT_S + total_rounds * round_deadline_s` already accounts for the full pending window; fill consumes that window without lengthening it. |
| Concurrent join wins race vs deadline tick | Race-safe by re-reading `tournament.status` inside the worker session before fill. If join already flipped status to ACTIVE, worker skips. |

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
  - `tournament_started` event published.
- `test_deadlines.py` (extend):
  - Expired pending with `live_count >= min_participants` AND game in {el_farol, public_goods}
    → fill + start, no cancel.
  - Expired pending with `live_count < min_participants` → cancel as today.
  - Expired pending with `min_participants is None` → cancel as today (regression guard).
  - Expired pending for `prisoners_dilemma` → cancel as today (scope guard).

**Builtin tests (`atp-games/tests/`)**

- `test_no_show_strategies.py`:
  - `el_farol/no_show` returns `stay_home` for many rounds, irrespective of observation.
  - `public_goods/no_show` returns `contribute=0` for many rounds.
  - Both resolve through `resolve_builtin(...)` and round-trip through `BUILTIN_REGISTRY`.

**Integration (`tests/integration/dashboard/`)**

- E2E: create `el_farol` with `num_players=5 min_participants=4`, join 4 agents, advance the
  clock past `pending_deadline`, assert tournament transitioned to ACTIVE, the 5th participant
  exists with `kind="no_show"`, `agent_name="missed-1"`, and the played actions across all rounds
  are `stay_home`.

**Migration test**

- Alembic upgrade then downgrade on:
  - empty database
  - database with existing tournaments (verify `min_participants IS NULL` for all pre-existing
    rows after upgrade; downgrade drops the column cleanly).

## Documentation Updates

- `TOURNAMENT_CREATION_API.md`: document `min_participants` field, validation rules, and
  no-show fill behavior. Include an example response showing `kind` / `was_no_show` for a
  participant.
- `CLAUDE.md`: no env-var change. Add a one-line note under tournament service description
  pointing at the new behavior.
- `docs/maestro-integration.md`: scan and update only if it currently asserts `count ==
  num_players` as an invariant for ACTIVE tournaments.
- Participant kit READMEs (`packages/.../participant-kit-el-farol-en/`): add a brief note that
  no-show participants may appear in the tournament with `agent_name="missed-N"`. Wire
  contract is unchanged.

## Migration Safety

- Single additive column with default NULL → safe online ALTER on Postgres.
- No backfill required: NULL means "legacy behavior" everywhere.
- Downgrade simply drops the column; any tournaments created with non-NULL values revert to
  legacy behavior, which is `min_participants = num_players` — strictly stricter than what
  the operator opted into. No data loss, just a degraded UX during downgrade window.

## Rollout

1. Land migration + service changes behind no flag — additive and default-off.
2. Update docs.
3. Smoke a real El Farol tournament on production with `num_players=5 min_participants=4`,
   intentionally have one participant skip the join window, verify autostart with one
   `missed-1` placeholder.

## Open Questions (deferred)

- **Mid-game leave → no-show conversion:** if a real participant leaves an `ACTIVE`
  tournament and live count drops below `min_participants`, do we convert their slot to a
  no-show and continue? Requires additional engine-level reasoning per game and is not
  required by the immediate use case.
- **Leaderboard handling of no-shows:** should they be excluded from rankings or shown with a
  marker? Likely a separate UI-only iteration once we have production data.
