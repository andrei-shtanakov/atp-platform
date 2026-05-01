# Tournament Shrink-on-Deadline — Design

**Status:** Approved (retroactively documented; implementation already in working tree)
**Date:** 2026-05-01
**Owner:** ATP platform
**Scope:** `el_farol`, `public_goods` tournament types only

## Context

This spec **supersedes** the `2026-05-01-tournament-min-participants-design.md` (no-show
placeholder fill) feature. That feature was implemented end-to-end on
`feat/tournament-min-participants` (21 commits, all tests green) but **never merged** because
the user revised the requirements during the final-review phase.

The new requirement is fundamentally different: instead of filling missing seats with
no-show placeholders to preserve `num_players`, the tournament **shrinks** to whatever live
roster size exists at deadline. The `num_players` field becomes mutable: if a tournament was
created for 100 bots and only 67 register by the pending deadline, `num_players` is updated
to 67 in the database and the tournament starts as a 67-player game.

The old `feat/tournament-min-participants` branch is abandoned (left in place as historical
reference; not merged).

## Problem

Today (`main` baseline) a tournament transitions `PENDING → ACTIVE` only when
`count(participants) == num_players`. If `pending_deadline`
(`ATP_TOURNAMENT_PENDING_MAX_WAIT_S`, default 300 s) elapses before that equality holds, the
tournament is cancelled with `cancelled_reason = PENDING_TIMEOUT`.

For multi-player games (`el_farol`, `public_goods`, both bound to 2..20 players), losing the
entire tournament because one or more bots had connectivity issues is unnecessarily brittle.
The platform is in debug mode with no production users; the right default is to start with
whoever showed up.

## Goal

For `el_farol` and `public_goods`: when `pending_deadline` expires, count live participants
(unreleased rows in `tournament_participants`). If at least 2, mutate `Tournament.num_players`
to that count and start. If 0 or 1, cancel with `PENDING_TIMEOUT` (the existing behavior). For
all other game types (`prisoners_dilemma`, `stag_hunt`, `battle_of_sexes`), keep the existing
unconditional cancel — these games are exactly-2-player and shrinking is meaningless.

Out of scope:

- 2-player games. They are exactly-2-player by engine constraint; shrinking from 2 to 1 is
  degenerate.
- Configurable per-tournament floor. Hardcoded floor of 2 is sufficient for the current
  use case. Adding a `min_participants` field is YAGNI; can be added later if real users
  ever need a higher floor.
- Preservation of the original requested `num_players`. The mutation overwrites it. Audit is
  via the structured log line `deadline_worker.tournament_shrunken` which carries both
  `original_num_players` and `actual_num_players`.
- Mid-game leave handling. The existing `ABANDONED` cascade still applies if all live
  participants leave an `ACTIVE` tournament.
- Wire-contract changes. Subscribers (SSE, MCP) discover the new `num_players` by re-reading
  the tournament; no event payload addition.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Apply only to `el_farol` and `public_goods` | Other game types are exactly-2-player; shrink is degenerate. |
| 2 | Hardcoded floor of 2 (no configurable `min_participants` field) | YAGNI. Engine bound is also 2; both al-or-nothing fail-safe. |
| 3 | Always-on shrink for el_farol/public_goods (no opt-in) | Platform is in debug mode; behavior change is acceptable per user. |
| 4 | Overwrite `Tournament.num_players` in place; do not preserve original | User explicitly rejected an audit column. Audit lives in the structured log. |
| 5 | `live_count` includes roster builtins + user joins where `released_at IS NULL` | Roster builtins are pre-committed lobby members and count as "registered". |
| 6 | `_play_roster_participants` helper returns participants with `released_at IS NULL OR has_played_action` | Closes a pre-existing edge case: a user who played round 1 then was kicked at round 2 must remain in the engine roster across all rounds (engine indexes by participant_id). |
| 7 | Defensive `session.refresh(tournament)` in `join()` after `flush()` | Closes the race opened by the new active-flip path: a concurrent `join()` whose status check passed before the worker took the FOR UPDATE lock could otherwise INSERT a Participant into an already-ACTIVE tournament. |
| 8 | `SELECT … FOR UPDATE` on Tournament row inside `try_shrink_and_start_or_cancel` | Serializes against `join()`'s implicit FK lock. Defensive insurance for future multi-replica deploys; today's single-worker assumption already serializes. |
| 9 | No new database columns | Minimal surface change. The mutation is on the existing `num_players` column. |

## Architecture

```
deadline_worker tick (deadlines.py)
  └─ for (tournament_id, game_type) in expired pending tournaments:
       ├─ open per-tournament session
       ├─ if game_type in {el_farol, public_goods}:
       │     service.try_shrink_and_start_or_cancel(tournament_id)
       └─ else:
             service.cancel_tournament_system(tournament_id, PENDING_TIMEOUT)
       └─ outer commit
```

`try_shrink_and_start_or_cancel(tournament_id)`:

1. `tournament = await session.get(Tournament, tournament_id, with_for_update=True)`. Lock
   row for the rest of the transaction. Closes the read-then-write race against
   any concurrent `join()`.
2. Re-check `tournament.status == PENDING`. If not, return — concurrent join/cancel raced
   ahead.
3. `live_count = COUNT(*) FROM tournament_participants WHERE tournament_id = ? AND
   released_at IS NULL`. Includes both roster builtins and user joins; excludes phantoms
   from `join → leave` in the PENDING window.
4. If `live_count < 2`: call `cancel_tournament_system(tournament_id,
   reason=CancelReason.PENDING_TIMEOUT)`. Existing cancel path; outer commit completes the
   transaction.
5. If `live_count != tournament.num_players`: mutate `tournament.num_players = live_count`
   and emit `log.info("deadline_worker.tournament_shrunken", ...)` with both original and
   actual values.
6. Call `_start_tournament(tournament)`. Existing helper flips PENDING → ACTIVE, creates
   round 1, commits internally before publishing `round_started` (LABS-74 invariant).

The outer commit in the deadline worker is **mandatory for the cancel branch** (the cancel
path doesn't commit) and **a defensive no-op for the shrink-start branch** (`_start_tournament`
already committed internally).

## Data Model

**Zero new columns.** `Tournament.num_players` becomes mutable in practice — no code path
mutated it post-create before this change; the new service method now does. The
existing migration is unchanged.

**No new columns on `Participant` either.** Existing `released_at` is the source of truth for
"in lobby vs left/kicked/timed-out".

## Service-layer changes

### `_live_participant_count(tournament_id) -> int`

Pure helper:

```python
async def _live_participant_count(self, tournament_id: int) -> int:
    count = await self._session.scalar(
        select(func.count(Participant.id))
        .where(Participant.tournament_id == tournament_id)
        .where(Participant.released_at.is_(None))
    )
    return int(count or 0)
```

Used by `try_shrink_and_start_or_cancel` and by the existing `count == num_players` autostart
check inside `join()`.

### `_play_roster_participants(tournament_id) -> list[Participant]`

```python
async def _play_roster_participants(
    self, tournament_id: int
) -> list[Participant]:
    played_any_round = (
        select(Action.id).where(Action.participant_id == Participant.id).exists()
    )
    return list(
        (await self._session.execute(
            select(Participant)
            .where(Participant.tournament_id == tournament_id)
            .where(Participant.released_at.is_(None) | played_any_round)
            .order_by(Participant.id)
        )).scalars()
    )
```

Returns the participants the engine should see in any given round: anyone currently in the
lobby (`released_at IS NULL`) plus anyone who has played at least one Action in this
tournament (so a player kicked at round 5 still appears in rounds 5+ for index stability).

Replaces three scattered places that previously did
`select(Participant).where(tournament_id == ...).order_by(id)`:
- `get_state_for` (state snapshot for a player)
- `_ensure_builtin_actions` (auto-fill builtin actions per round)
- `_resolve_round` (resolve round → compute payoffs)
- `_release_participants` final-completion step

Without this fix, a player kicked mid-tournament would silently disappear from subsequent
round states and shift the participant indices, corrupting payoffs.

### `try_shrink_and_start_or_cancel(tournament_id)`

New public method on `TournamentService`:

```python
async def try_shrink_and_start_or_cancel(self, tournament_id: int) -> None:
    tournament = await self._session.get(
        Tournament, tournament_id, with_for_update=True
    )
    if tournament is None:
        raise NotFoundError(f"tournament {tournament_id}")
    if tournament.status != TournamentStatus.PENDING:
        return

    live_count = await self._live_participant_count(tournament_id)
    if live_count < 2:
        await self.cancel_tournament_system(
            tournament_id, reason=CancelReason.PENDING_TIMEOUT
        )
        return

    if live_count != tournament.num_players:
        original = tournament.num_players
        tournament.num_players = live_count
        logger.info(
            "deadline_worker.tournament_shrunken",
            extra={
                "tournament_id": tournament_id,
                "original_num_players": original,
                "actual_num_players": live_count,
            },
        )

    await self._start_tournament(tournament)
```

### `join()` changes

Two modifications, both inside the post-flush block:

1. **Replace** `count = scalar(select(func.count(Participant.id)).where(tournament_id == ?))`
   (no released_at filter) with `live_count = self._live_participant_count(tournament_id)`.
   This makes the existing autostart check (`count == num_players → _start_tournament`)
   correctly exclude leavers. Pre-rev-3 the bug was unreachable because `leave()` in PENDING
   was rare; the new shrink path makes it reachable.
2. **Defensive `await self._session.refresh(tournament)` after `flush()`**, then check
   `if tournament.status != TournamentStatus.PENDING: rollback + raise ConflictError("not
   accepting joins")`. Closes the race where the worker flipped status to ACTIVE under
   FOR UPDATE between the cached read at the start of `join()` and the post-flush autostart
   check.

The existing idempotency lookups for `Participant` matching `tournament_id` + `agent_id`/`user_id`
also gain `released_at IS NULL` filters so a previously-released agent can rejoin the SAME
tournament if needed (this matches the documented intent of `released_at`).

## Deadline worker

`packages/atp-dashboard/atp/dashboard/tournament/deadlines.py` Path 2 changes:

```python
expired_pending_result = await scan_session.execute(
    select(Tournament.id, Tournament.game_type)
    .where(Tournament.status == TournamentStatus.PENDING)
    .where(Tournament.pending_deadline < _utc_now())
)
pending_tournaments = [(row[0], row[1]) for row in expired_pending_result]

# Path 2
for tournament_id, game_type in pending_tournaments:
    try:
        async with session_factory() as session:
            service = TournamentService(session, bus)
            if game_type in {"el_farol", "public_goods"}:
                await service.try_shrink_and_start_or_cancel(tournament_id)
            else:
                await service.cancel_tournament_system(
                    tournament_id, reason=CancelReason.PENDING_TIMEOUT
                )
            await session.commit()
    except Exception:
        log.exception(
            "deadline_worker.pending_transition_failed",
            extra={"tournament_id": tournament_id, "game_type": game_type},
        )

log.info(
    "deadline_worker.tick_complete rounds=%d pending_processed=%d elapsed_ms=%d",
    len(round_ids),
    len(pending_tournaments),
    int((time.monotonic() - t_start) * 1000),
)
```

Path 1 (expired round deadlines) is unchanged.

## Edge cases & invariants

| Case | Behavior |
|---|---|
| el_farol/PG, expired pending, `live_count >= 2` and `< num_players` | Shrink + start. `num_players` mutated; `tournament_shrunken` log emitted. |
| el_farol/PG, expired pending, `live_count == num_players` | No mutation; `_start_tournament` runs. (Theoretically unreachable because `join()` would have already started; defensive branch.) |
| el_farol/PG, expired pending, `live_count < 2` | Cancel with `PENDING_TIMEOUT`. Existing path. |
| `prisoners_dilemma`/`stag_hunt`/`battle_of_sexes`, expired pending | Cancel with `PENDING_TIMEOUT`. Unchanged. |
| Roster-only tournament (e.g. `roster=2`, no live joins), el_farol | `live_count = 2` → start with `num_players = 2`. Unchanged from existing roster-fills-everything inline-start path. |
| `join → leave` in PENDING window | `released_at IS NULL` filter excludes the phantom row. `live_count` reflects only live agents. |
| Concurrent `join()` racing the deadline tick | `FOR UPDATE` on Tournament row serializes the worker's transaction. Late `join()` after the lock release sees ACTIVE via the post-flush `refresh()` and raises `ConflictError`. |
| Player kicked mid-tournament | `_play_roster_participants` keeps them in the gameplay roster because they have ≥1 Action row, preserving participant indices for the engine. |
| Engine cache (`_el_farol_for(num_players)`, `_pg_for(num_players)`) | Keyed by num_players; shrink mutates num_players before any engine call (engine is built lazily at first round resolution), so the engine is constructed with the new size. No invalidation needed. |
| Pre-existing `count == num_players` autostart in `join()` (now uses `_live_participant_count`) | Correctly excludes leavers from the autostart trigger. |

## Test plan

**Unit (`tests/unit/dashboard/tournament/test_service_shrink.py`, new):**

- `test_shrinks_when_below_num_players` — create with 5, join 3, expect ACTIVE and num_players=3.
- `test_no_shrink_when_full` — create with 3 + roster of 3, expect ACTIVE and num_players unchanged.
- `test_cancels_when_below_floor` — 1 live → CANCELLED + PENDING_TIMEOUT.
- `test_cancels_when_zero_live` — 0 live → CANCELLED.
- `test_skips_when_status_not_pending` — pre-flipped to ACTIVE, no-op.
- `test_released_participants_excluded_from_count` — join+leave in PENDING; shrink uses
  remaining live count.
- `test_roster_counts_toward_live` — roster=2, no live joins, expect ACTIVE with num_players=2.

**Unit (`tests/unit/dashboard/tournament/test_service_join.py`, extended):**

- `test_join_rolls_back_if_refresh_sees_concurrent_start` — race regression for the
  defensive refresh.

**Unit (`tests/unit/dashboard/tournament/test_deadline_worker.py`, extended):**

- `test_tick_routes_pending_expiry_by_game_type` — el_farol → shrink, public_goods → shrink,
  prisoners_dilemma → cancel.

**Unit (`tests/unit/dashboard/tournament/conftest.py`, fixture hygiene):**

- `_clear_el_farol_cache` extended to also clear `_pg_for.cache_clear()` for symmetry.

**Integration (`tests/integration/dashboard/tournament/test_el_farol_flow.py` and
`test_public_goods_flow.py`, extended):**

- `test_el_farol_pending_timeout_shrinks_then_completes` — full E2E: create with 5, join 3,
  force pending_deadline into the past, run `_tick`, verify ACTIVE with num_players=3, then
  drive 2 rounds to completion and assert all 3 participants get scores.
- `test_public_goods_pending_timeout_shrinks_then_completes` — same shape for PG.

## What this design does NOT include

- Tests for the existing `count == num_players` autostart in `join()` after the formula
  switch to `_live_participant_count`. The existing flow tests cover this implicitly. A
  targeted unit test for the leaver-excluded autostart edge case would be a worthwhile
  follow-up.
- API or UI changes. The mutation is invisible at the contract level except that consumers
  reading `tournament.num_players` will see the post-shrink value after the deadline.
- Documentation updates beyond this spec. The CLAUDE.md autostart pointer added in the no_show
  branch is no longer applicable; if the no_show branch is ever merged separately, it will
  need a different one.
