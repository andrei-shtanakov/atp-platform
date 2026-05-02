# El Farol Scoring Modes — Design

**Status:** Draft
**Date:** 2026-05-02
**Owner:** prosto.andrey.g@gmail.com

## Summary

Add a configurable `scoring_mode` to the El Farol Bar game with two
values:

- **`happy_only`** (new default) — each happy slot scores +1, each
  crowded slot scores 0. No penalties. Final score = sum of happy
  slots across all rounds.
- **`happy_minus_crowded`** (legacy) — each happy slot scores +1, each
  crowded slot scores −1. Final score = `t_happy / max(t_crowded, 0.1)`.

The new default simplifies scoring semantics for new tournaments. The
legacy mode stays available as opt-in for backwards compatibility with
existing tests and any tournament that explicitly needs it.

## Goals

- Make the dominant playing experience reward attendance of happy
  slots without penalising mistakes.
- Preserve backwards compatibility for existing tests and any
  consumer that depends on the legacy formula.
- Keep observation telemetry (`your_t_happy_slots`,
  `your_t_crowded_slots`) identical across modes — bots can still see
  how often they were caught in the crowd, regardless of scoring.

## Non-goals

- Migrating historical tournaments. Past `participant.total_score`
  values stay as-recorded; they are snapshots, not formulas.
- Per-tournament UI affordance for switching mode. Mode is a config
  field set at tournament creation; switching post-create is not
  supported.
- Removing `_t_crowded` accumulation. Both modes track it for
  observation parity.

## Architecture

### Config field

`game-environments/game_envs/games/el_farol.py::ElFarolConfig`:

```python
scoring_mode: Literal["happy_only", "happy_minus_crowded"] = "happy_only"
```

`__post_init__` validates that the value is one of the two literals;
unknown values raise `ValueError`.

### Mode behaviour

| | `happy_only` (default) | `happy_minus_crowded` (legacy) |
|---|---|---|
| Per-slot | +1 if happy, 0 if crowded | +1 if happy, −1 if crowded |
| Per-day payoff (`step()` and `compute_round_payoffs()`) | `happy` | `happy − crowded` |
| Final payoff (`get_payoffs()`) | `t_happy` | `t_happy / max(t_crowded, 0.1)` |
| `_t_crowded` accumulation | yes (telemetry only) | yes (telemetry + formula) |
| `min_total_hours` disqualification | applied | applied |

### Code touchpoints

Three methods branch on `c.scoring_mode`:

1. **`compute_round_payoffs(actions)`** — returns per-round payoffs
   for the round runner. Branch:
   ```python
   if c.scoring_mode == "happy_only":
       payoffs[p_idx] = float(happy)
   else:  # happy_minus_crowded
       payoffs[p_idx] = float(happy - crowded_count)
   ```

2. **`step(actions)`** — accumulates `_t_happy` / `_t_crowded` (in
   both modes) and computes the round's payoff. Same branch on the
   payoff computation.

3. **`get_payoffs()`** — final score per player. Both modes apply
   `min_total_hours` disqualification first; then:
   ```python
   if c.scoring_mode == "happy_only":
       result[pid] = th
   else:  # happy_minus_crowded
       result[pid] = th / max(tc, 0.1)
   ```

`_t_crowded` accumulation is unconditional — both modes update it on
every round so observation payloads stay consistent.

### Dashboard "About this game" copy

`packages/atp-dashboard/atp/dashboard/v2/game_copy.py` defines static
`GameCopy` records consumed by the dashboard's "About" / "Games"
pages. It is NOT a per-round MCP tool description — bots receive game
state through structured MCP responses, not free-form prose from this
file.

Update `GameCopy["el_farol"]` text in place to describe the new
default `happy_only` mode:

- `tagline`: replace "and −1 once attendance reaches or exceeds it" →
  "and 0 if attendance reaches or exceeds it (no penalty)".
- `rules` entries that mention "+1 happy / −1 crowded" → "+1 happy /
  0 crowded". Replace "Your round payoff is (happy slots) − (crowded
  slots)" → "Your round payoff is the number of happy slots".
- `payoff_formula`: replace "Round total = (happy slots) − (crowded
  slots)" → "Round total = number of happy slots. Final score = sum
  of happy slots across all rounds."

A tournament running in legacy `happy_minus_crowded` mode will show
the dashboard's `happy_only` text on the about page — a small
documentation drift, but the legacy mode is opt-in for tests and the
about page is not authoritative on per-tournament rules.

### Demo agent comment

`demo-game/agents/el_farol_agent.py` is a local test agent. Its
explanatory comment is updated to describe the new default
(`happy_only`) without conditional logic. Same drift consideration as
above — acceptable for a demo.

### Documentation

Both rules docs (`docs/games/rules/el-farol-bar.ru.md` and
`el-farol-bar.en.md`) get a new "Scoring modes" section that replaces
the existing "Per-round payoff" + "Final payoffs" sections. Structure
shown in the brainstorming spec — mirror in both languages.

The `ElFarolConfig` parameter table gains a `scoring_mode` row.

## Data flow

```
Tournament creation (REST or admin UI)
  └─ config dict may set "scoring_mode": "happy_minus_crowded"
     (defaults to "happy_only" if absent)

ElFarolConfig.__post_init__
  └─ validates scoring_mode ∈ {"happy_only", "happy_minus_crowded"}

Game step(actions)
  ├─ compute happy[pid], crowded[pid] per slot (unchanged)
  ├─ accumulate _t_happy[pid] += happy, _t_crowded[pid] += crowded (unchanged)
  └─ payoff = (happy if happy_only else happy - crowded)

Game get_payoffs()  (called at terminal)
  ├─ if total_hours < min_total_hours: result = 0  (unchanged)
  └─ else: result = (t_happy if happy_only else t_happy / max(t_crowded, 0.1))

Tournament service writes participant.total_score = result[pid]
Dashboard "About" page:
  └─ reads static GameCopy["el_farol"] (no config awareness; describes
     the new default mode unconditionally)
```

## Edge cases & error handling

| Case | Behaviour |
|---|---|
| `scoring_mode` absent in config | Defaults to `"happy_only"` via dataclass default |
| `scoring_mode="bogus"` | `ValueError` in `__post_init__` |
| Empty intervals `{"intervals": []}` | `happy = 0`, `crowded = 0`. Per-day payoff = 0 in both modes. Counter unchanged. |
| All players stay home | All payoffs = 0; `t_happy` / `t_crowded` stay 0; final per-mode: `happy_only` → 0; `happy_minus_crowded` → `0 / 0.1 = 0`. |
| `min_total_hours > 0`, viewer attends only 1 hour | Final = 0 regardless of mode (disqualification check applied first) |
| `happy_minus_crowded` with `t_crowded == 0` | Denominator clamped to `max(0, 0.1) = 0.1` (existing behaviour preserved) |
| Tournament created before this change | `scoring_mode` field absent → default `"happy_only"` applied on next play. **Note:** if a *running* (active) tournament exists at deploy time, the resumed game will use the new default. This is acceptable — no production tournaments are mid-game expected at deploy time, and the formula change is the explicit intent. |
| Snapshot scores in DB | `participant.total_score` rows from past tournaments are not recomputed. The Hall of Fame leaderboard mixes scores from both formulas; the dashboard does not surface the mode used per row in v1. Acceptable for pre-release. |

## Performance

No measurable impact. The branch on `scoring_mode` is a single string
compare per round; `_t_crowded` accumulation is unchanged.

## Security

None. Config field is server-only; no user-supplied path here.

## Testing

### Unit (`game-environments/tests/test_el_farol.py`)

**Renamed/relabelled (legacy regression):**
- `test_compute_round_payoffs_happy_minus_crowded` →
  `test_compute_round_payoffs_legacy_mode_happy_minus_crowded`. Add
  `scoring_mode="happy_minus_crowded"` to the config; assertions stay.
- Any other tests asserting `t_happy / t_crowded` or `happy − crowded`
  formulas: tag the config with the legacy mode.

**New tests for `happy_only` (default):**

1. `test_compute_round_payoffs_happy_only_default` — 3 players,
   threshold=3, slot 1 crowded:
   - p0 attends slot 0 (happy) + slot 1 (crowded) → payoff = 1
   - p1 attends slot 1 (crowded) + slot 2 (happy) → payoff = 1
   - p2 attends slot 0 (happy) + slot 1 (crowded) → payoff = 1

2. `test_step_happy_only_accumulates_t_happy_and_t_crowded` — both
   counters increment in `happy_only` mode, even though `_t_crowded`
   is not in the formula. Verifies observation telemetry parity.

3. `test_get_payoffs_happy_only_returns_t_happy_sum` — seed
   `_t_happy=5, _t_crowded=3`; final score = 5.0 (not ratio).

4. `test_get_payoffs_happy_only_disqualification_still_applies` —
   `min_total_hours=10`, attendance below threshold → final = 0
   regardless of mode.

5. `test_scoring_mode_validation_rejects_unknown` — config with
   `scoring_mode="bogus"` raises `ValueError`.

6. `test_observation_includes_t_crowded_in_both_modes` —
   `observe(player_id)` returns `your_t_crowded_slots` regardless of
   the active mode.

7. `test_scoring_mode_default_is_happy_only` — `ElFarolConfig()`
   without explicit field has `scoring_mode == "happy_only"`.

### Cross-module tests

- `atp-games/tests/test_round_payoffs.py` — assertions referencing
  `happy − crowded` formulas: tag the fixture's config with
  `scoring_mode="happy_minus_crowded"` so behaviour is preserved.
  Optionally add a `happy_only`-mode counterpart that verifies the new
  payoff shape.
- `atp-games/tests/test_el_farol_interval_agents.py`,
  `test_runner_action_records.py` — same approach if any assertion
  depends on the formula.

### Dashboard copy tests

In `tests/unit/dashboard/` (or wherever `game_copy.py` is tested
today; verify and add accordingly):

8. `test_game_copy_el_farol_describes_happy_only` —
   `GAME_COPY["el_farol"].payoff_formula` contains "happy slots" and
   "no penalty" (or equivalent). Does NOT contain "−1".

### Coverage target

≥80% on new code (`scoring_mode` field + branches in three methods).
The project's CLAUDE.md sets this as a hard gate.

## Migration plan

No data migration. The `scoring_mode` config field is a new dataclass
attribute with a default; absence in stored configs is fine. Tests
updated as part of the same commit chain.

## Open questions

- **Per-row scoring-mode badge on Hall of Fame** — out of scope for
  v1. If users start running mixed-mode tournaments and the
  leaderboard becomes confusing, surface the mode as a column. For now
  the spec accepts the mix as documented behaviour.
- **Bot-facing prompt: bilingual** — the existing English prompt is
  the only one. Russian is not surfaced to bots; the rules doc has
  both languages but bots see English.

## References

- `game-environments/game_envs/games/el_farol.py` — game class
- `packages/atp-dashboard/atp/dashboard/v2/game_copy.py` — LLM prompt
  builder
- `docs/games/rules/el-farol-bar.ru.md`,
  `docs/games/rules/el-farol-bar.en.md` — rules
- `demo-game/agents/el_farol_agent.py` — local demo agent
