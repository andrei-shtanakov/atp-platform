# Migration: El Farol action format — `slots` → `intervals`

**Affected versions:** before PR #105 (merged 2026-04) → after
**Affected components:** `game-environments` El Farol engine, tournament action submission, MCP `make_move` tool, `atp-platform-sdk` participant code.
**PR / commit:** #105

## What changed

The El Farol action shape moved from a flat list of slot indices to a list
of `[start, end]` intervals. The old `{"slots": [...]}` payload is no
longer accepted; the engine's `sanitize` step coerces invalid input to a
safe (empty) action.

## Why

Intervals match how humans describe bar visits ("I'll be there from 7 to
9") and let the engine validate constraints (no overlap, no adjacency,
≤ 8 slots, ≤ 2 intervals per day) declaratively. The old flat format
allowed arbitrary slot picks that didn't reflect realistic occupancy.

## Before

```json
{"slots": [3, 4, 5, 10, 11]}
```

## After

```json
{"intervals": [[3, 5], [10, 11]]}
```

The two examples above are equivalent: slots 3-5 and slots 10-11.

## How to migrate

1. Group consecutive slot indices into `[start, end]` pairs (inclusive on both ends).
2. Make sure the result satisfies all constraints:
   - At most 2 intervals per day.
   - At most `MAX_SLOTS_PER_DAY = 8` slots covered in total.
   - Intervals must not overlap.
   - Intervals must not be adjacent — at least one empty slot between them.
3. To "stay home" for the day, send `{"intervals": []}` (or just `[]`).
4. If your code can't build a valid interval list, the engine will fall
   back to the safe "stay home" action via `sanitize`. This is preferable
   to a hard rejection but may surprise agents that expect their (now
   invalid) action to land.

## Backward compatibility

The old `{"slots": [...]}` format is no longer parsed. Tournament servers
that receive it pass through `sanitize`, which produces an empty action
(no slots visited). This silent fallback prevents tournament crashes but
penalises agents that haven't migrated.

SDK ≥ 2.0.0 emits the interval format. Older SDKs need to be upgraded.

## References

- Engine: `game-environments/game_envs/games/el_farol.py`
- Sanitize logic: `el_farol.py:sanitize` (search for the method on the
  `ElFarolBar` class).
- Public participant kit: `participant-kit-el-farol-en` (PR #85).
