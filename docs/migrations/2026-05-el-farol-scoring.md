# Migration: El Farol scoring default — `happy_minus_crowded` → `happy_only`

**Affected versions:** before PR #121 (merged 2026-05) → after
**Affected components:** `game-environments` El Farol engine, tournament
score aggregation, dashboard winners view, anyone interpreting El Farol
scores out of band.
**PR / commit:** #121

## What changed

The default `scoring_mode` for El Farol flipped from `happy_minus_crowded`
(a ratio that penalised crowded-slot visits) to `happy_only` (a raw count
of happy-slot visits, no penalty for crowded slots).

## Why

The ratio formula `t_happy / max(t_crowded, 0.1)` made scores
non-monotonic and hard to interpret — visiting one extra happy slot could
*lower* a player's score if the agent also visited a crowded slot the
same day. Tournament rankings became sensitive to the ε floor (`0.1`).
The new default is monotone, additive, and matches how players intuit
El Farol payoffs ("just count my happy slots").

## Before

Per-day payoff: `happy − crowded`.
Final per-player score: `t_happy / max(t_crowded, 0.1)` (gated by
`min_total_hours`).
Per-day payoff sign: can be negative.
Tournament `total_score`: the final ratio.

## After

Per-day payoff: number of happy slots that day (≥ 0).
Final per-player score: `t_happy` (gated by `min_total_hours`).
Per-day payoff sign: never negative.
Tournament `total_score`: sum of per-day payoffs = `t_happy`.

## How to migrate

1. **Tournament-API users:** no action needed. Tournaments now use
   `happy_only` by default. Existing leaderboards get rebuilt under the
   new scoring on the next tournament cut.
2. **Score-comparing analysis code:** if you previously compared scores
   across El Farol tournaments, do not mix pre- and post-PR-#121 scores
   — they are not on the same scale. Rebuild any cross-tournament
   leaderboards from raw `Action.payoff` rows.
3. **Test code that asserts on legacy ratio behaviour:** opt in
   explicitly:
   ```python
   from game_envs.games.el_farol import ElFarolConfig, ElFarolBar
   config = ElFarolConfig(scoring_mode="happy_minus_crowded", ...)
   game = ElFarolBar(config=config)
   ```
   The legacy mode is preserved for tests and atp-games standalone
   scenarios. It is **not** exposed through the tournament API — opt-in
   only via direct engine construction.
4. **Player observation:** `your_t_crowded_slots` is still surfaced in
   the per-player observation, regardless of `scoring_mode`. Agents can
   still use it as a behaviour signal even though it no longer subtracts
   from their score.

## Backward compatibility

- Legacy mode (`scoring_mode="happy_minus_crowded"`) remains available
  via direct `ElFarolConfig` construction.
- Tournament API does not accept a `scoring_mode` parameter (deliberate
  — production tournaments always use the default).
- The disqualification rule on `min_total_hours` applies identically in
  both modes.

## References

- Engine: `game-environments/game_envs/games/el_farol.py` — search for
  `scoring_mode`.
- Public copy: `atp/dashboard/v2/game_copy.py` — `GAME_COPY["el_farol"]`
  (rules + payoff_formula now describe `happy_only`).
- Public participant kit: `participant-kit-el-farol-en` (PR #85).
