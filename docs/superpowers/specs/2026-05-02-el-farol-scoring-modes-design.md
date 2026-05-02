# El Farol Scoring Modes — Design

**Status:** Draft (revision 3 — final-review polish)
**Date:** 2026-05-02
**Owner:** prosto.andrey.g@gmail.com

## Summary

Add a configurable `scoring_mode` to the El Farol Bar engine
(`game-environments/game_envs/games/el_farol.py`) with two values:

- **`happy_only`** (new default) — each happy slot scores +1, each
  crowded slot scores 0. No penalties. Standalone final via
  `get_payoffs()` = `t_happy` (unless disqualified by
  `min_total_hours`). Tournament `total_score` = sum of per-round
  `happy` over all resolved rounds.
- **`happy_minus_crowded`** (legacy, opt-in) — each happy slot scores
  +1, each crowded slot scores −1. The engine's standalone
  `get_payoffs()` final formula remains `t_happy / max(t_crowded, 0.1)`.

The change is **engine-level**. Tournaments inherit the new default
automatically because the dashboard's `_el_farol_for(num_players)`
factory constructs `ElFarolConfig(num_players=...)` without supplying
`scoring_mode` (`packages/atp-dashboard/atp/dashboard/tournament/service.py:120-128`).

Tournament `total_score` is computed as `sum(Action.payoff)` over all
resolved rounds (`tournament/service.py:1578-1582`); the engine's
`get_payoffs()` ratio formula is **never invoked** by the tournament
runtime. So switching the engine default from `happy − crowded` to
`happy` per round changes tournament totals from
"sum(happy − crowded) per round" to "sum of happy per round" — exactly
the user-requested rule.

## Goals

- Default scoring across the platform becomes "+1 per happy slot, 0
  per crowded slot, no penalties".
- Preserve backwards compatibility for tests and any standalone
  consumer that still wants the old per-round and ratio formulas, via
  explicit `scoring_mode="happy_minus_crowded"` opt-in.
- Engine and dashboard docs accurately describe one canonical scoring
  rule (the new default), with the legacy mode documented but not
  surfaced as a tournament option.

## Non-goals

- **Per-tournament configurability of `scoring_mode`.** The dashboard
  `create_tournament()` API does not accept it, the cache key in
  `_el_farol_for()` does not include it, and admin UI does not expose
  it. Adding this plumbing is out of scope for v1. Tournaments always
  use the engine default (`happy_only`) until a future PR threads
  the field through the API layer.
- Migrating historical tournaments. `participant.total_score` rows
  written under the old per-round formula stay as-recorded; they are
  snapshots, not formulas.
- Backfilling a `scoring_mode_used` denormalised column on
  `tournament_participants`. Since v1 only ships `happy_only` to
  tournaments, there is no per-row mode mix to disambiguate.

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
| Per-round payoff (`step()` and `compute_round_payoffs()`) | `happy` | `happy − crowded` |
| Standalone-runner final (`get_payoffs()`) | `t_happy` | `t_happy / max(t_crowded, 0.1)` |
| Tournament `total_score` | `sum(per-round happy)` = `t_happy` (never invokes `get_payoffs`) | unreachable — tournaments don't plumb this mode in v1 |
| `_t_crowded` accumulation | yes (telemetry only) | yes (telemetry + formula) |
| `min_total_hours` disqualification | applied by `get_payoffs` | applied by `get_payoffs` |

The "tournament `total_score` for legacy mode is unreachable" row is
the deliberate v1 trade-off: the legacy mode is reachable only from
direct `ElFarolConfig(scoring_mode="happy_minus_crowded")` calls
(tests, atp-games standalone runner, YAML scenarios).

### Engine code touchpoints

Three methods branch on `c.scoring_mode`:

1. **`compute_round_payoffs(actions)`** — used by tournaments.
   ```python
   if c.scoring_mode == "happy_only":
       payoffs[p_idx] = float(happy)
   else:  # happy_minus_crowded
       payoffs[p_idx] = float(happy - crowded_count)
   ```

2. **`step(actions)`** — used by standalone runner. Same branch on
   the per-round payoff. `_t_happy` and `_t_crowded` accumulate
   unconditionally.

3. **`get_payoffs()`** — used by standalone runner only. Both modes
   apply `min_total_hours` disqualification first; then:
   ```python
   if c.scoring_mode == "happy_only":
       result[pid] = th
   else:  # happy_minus_crowded
       result[pid] = th / max(tc, 0.1)
   ```

### Engine documentation touchpoints (in addition to the rules docs)

The engine itself carries scoring text in multiple places. All must
be updated to describe the new default; the legacy mode is documented
inline as opt-in.

- **Module docstring** (`el_farol.py:1-25`) — replace the
  per-round/final formulas in the file header.
- **`compute_round_payoffs` docstring** (`el_farol.py:512`) — describe
  the new branch and reference both modes.
- **`get_payoffs` docstring** (`el_farol.py:677`) — describe both
  modes, with `happy_only` as the default.
- **`to_prompt()`** (`el_farol.py:744`) — generates the human-readable
  game description used in standalone scenarios. **Must be
  mode-aware**: branch on `c.scoring_mode` and return the matching
  description (same shape as the three-method branches above). Treating
  this as conditional-if-cheap creates tech debt for the future
  tournament-level config work — the branch is the same five lines
  regardless.

### Dashboard "About this game" copy

`packages/atp-dashboard/atp/dashboard/v2/game_copy.py` defines static
`GameCopy` records consumed by the dashboard's "About" / "Games"
pages. It is **not** a per-round MCP tool description — bots receive
game state through structured MCP responses, not free-form prose
from this file.

Update `GameCopy["el_farol"]` text in place to describe the new
default `happy_only` mode. The legacy mode is not mentioned on the
dashboard since v1 doesn't expose it as a tournament option.

- `tagline`: replace "and −1 once attendance reaches or exceeds it"
  → "and 0 if attendance reaches or exceeds it (no penalty)".
- `rules` entry mentioning "+1 happy / −1 crowded" → "+1 happy / 0
  crowded". Replace "Your round payoff is (happy slots) − (crowded
  slots)" → "Your round payoff is the number of happy slots".
- `payoff_formula`: replace "Round total = (happy slots) − (crowded
  slots)" → "Round total = number of happy slots. In tournament play,
  the displayed score is the sum of round totals across all rounds."
  (The `/ui/games/{game}` page is a generic public game-detail page,
  not tournament-specific, so the wording must accommodate both
  standalone and tournament framings without overpromising.)

### Demo agent comment

`demo-game/agents/el_farol_agent.py` is a local test agent. Its
explanatory comment is updated to describe the new default
(`happy_only`) without conditional logic. Acceptable for a demo.

### Rules documentation

Both rules docs (`docs/games/rules/el-farol-bar.ru.md` and
`el-farol-bar.en.md`) get a new "Scoring modes" section that
replaces the existing "Per-round payoff" + "Final payoffs" sections.
Structure:

```md
## Scoring modes (`scoring_mode`)

The engine supports two scoring modes. The default is `happy_only`.
Tournaments always use the default — there is no per-tournament
opt-in to the legacy mode in v1.

### `happy_only` (default) — simple accumulation
- Each happy slot: +1.
- Each crowded slot: 0 (no penalty).
- Per-round payoff: number of happy slots that round.
- Standalone final via `get_payoffs()`: `t_happy` (unless
  disqualified by `min_total_hours`).
- Tournament `total_score`: sum of per-round happy values.

### `happy_minus_crowded` (legacy, opt-in)
- Reachable only from direct `ElFarolConfig(scoring_mode=...)`
  construction (tests, atp-games standalone scenarios). Not exposed
  in tournament APIs.
- Each happy slot: +1; each crowded slot: −1.
- Per-round payoff: `happy − crowded`.
- Standalone final via `get_payoffs()`: `t_happy / max(t_crowded, 0.1)`.

In both modes: `_t_crowded` is accumulated and surfaced in
observation (`your_t_crowded_slots`); `min_total_hours`
disqualification applies in `get_payoffs()`.
```

The `ElFarolConfig` parameter table gains a `scoring_mode` row.

## Data flow

```
Tournament path (always happy_only in v1):
  ├─ create_tournament(...): no scoring_mode passed
  ├─ _el_farol_for(num_players) → ElFarolConfig(num_players=...)
  │     scoring_mode defaults to "happy_only" via dataclass
  ├─ resolve_round → game.compute_round_payoffs(actions)
  │     branches on scoring_mode → returns per-round happy values
  ├─ Action.payoff = happy (per row)
  └─ participant.total_score = sum(Action.payoff) = sum(happy) = t_happy

Standalone runner path (atp-games scenarios, tests):
  ├─ scenario YAML / fixture sets ElFarolConfig(scoring_mode=...)
  ├─ game.step(actions) → per-round payoff (happy or happy − crowded)
  └─ game.get_payoffs() (terminal):
       happy_only → result[pid] = t_happy
       happy_minus_crowded → result[pid] = t_happy / max(t_crowded, 0.1)
       both apply min_total_hours disqualification first

Dashboard "About" page:
  └─ static GameCopy["el_farol"] (no config awareness; describes the
     new default mode unconditionally)
```

## Edge cases & error handling

| Case | Behaviour |
|---|---|
| `scoring_mode` absent in config | Defaults to `"happy_only"` via dataclass default |
| `scoring_mode="bogus"` | `ValueError` in `__post_init__` |
| Empty intervals `{"intervals": []}` | `happy = 0`, `crowded = 0`. Per-round payoff = 0 in both modes. Counters unchanged. |
| All players stay home | All payoffs = 0; `t_happy` / `t_crowded` stay 0; final per-mode: `happy_only` → 0; `happy_minus_crowded` standalone → `0 / 0.1 = 0`. |
| `min_total_hours > 0`, viewer attends only 1 hour | Standalone final = 0 regardless of mode (disqualification check applied first). Tournament `total_score` does not invoke `get_payoffs`, so disqualification has no effect on tournaments. This is a pre-existing engine asymmetry, not introduced by this change. |
| `happy_minus_crowded` with `t_crowded == 0` | Denominator clamped to `max(0, 0.1) = 0.1` (existing behaviour preserved) |
| Active (in-flight) tournament resumed after deploy | Engine default flips to `happy_only`. Future `compute_round_payoffs` calls return `happy` instead of `happy − crowded`. Past-round `Action.payoff` rows are not recomputed. The `total_score` aggregate at completion will mix old per-round values (computed under legacy) with new per-round values (computed under happy_only). Acceptable for pre-release; explicitly documented here so it is not surprising. |
| Standalone YAML scenario without `scoring_mode` | Defaults to `happy_only`. Existing scenarios that depend on the legacy formula must be updated to set `scoring_mode: happy_minus_crowded` explicitly. Examples: `examples/test_suites/13_game_el_farol.yaml` if any of its scoring expectations are baked in. Implementation must audit all such fixtures. |

## Performance

No measurable impact. The branch on `scoring_mode` is a single string
compare per round; `_t_crowded` accumulation is unchanged.

## Security

None. Config field is server-only; no user-supplied path here.

## Consumer impact

### participant-kit-el-farol-en (semi-public surface)

PR #85 (2026-04-24) made the El Farol wire contract semi-public. The
**wire format** (action schema + state shape) is unchanged. The
**score values** returned to bots in `your_cumulative_score` will
change scale: bots that were tuned to maximise `happy − crowded` will
still trend in the right direction (both formulas correlate
positively with happy attendance), but the per-round magnitude and
the absence of negative gradients changes the optimisation surface.

Required:
- Add a CHANGELOG entry in `participant-kit-el-farol-en/README.md`
  noting the default scoring change effective at the deploy date.
- Bump the participant-kit version (minor — behaviour change without
  wire-format break) per its own versioning convention.

### atp-games standalone runner / YAML scenarios

Behaviour changes globally for any scenario that constructs
`ElFarolConfig()` without an explicit `scoring_mode`.

**Concrete audit list** (from `grep -rln "happy_minus_crowded\|happy
- crowded\|t_happy / max" --include='*.py' --include='*.yaml'
--include='*.md'`):

- `game-environments/game_envs/games/el_farol.py` — engine itself
  (the file we're modifying).
- `game-environments/tests/test_el_farol.py` — engine unit tests.
- `atp-games/tests/test_round_payoffs.py` — atp-games runner tests.
- `tests/unit/games/test_el_farol_state_format.py` — state-format
  test (verify whether it asserts on payoffs).
- `examples/test_suites/13_game_el_farol.yaml` — example YAML suite.
  Audit for any baked-in payoff expectations.
- `docs/games/rules/el-farol-bar.ru.md`,
  `docs/games/rules/el-farol-bar.en.md` — rules docs (we're rewriting
  these).

Each file is either:
- Tagged with `scoring_mode: "happy_minus_crowded"` to preserve old
  behaviour as a regression test, OR
- Updated for the new default's expected payoff values.

### Existing tests

All assertions on the form `payoff == happy − crowded` or
`final_score == t_happy / max(t_crowded, 0.1)` will fail under the
new default. Each test is updated either to:
- Add `scoring_mode="happy_minus_crowded"` to its config (preserves
  old assertions as a regression test for the legacy mode), OR
- Rewrite assertions for the `happy_only` semantics.

The implementation pass must enumerate every such test; the
acceptance gate is "all engine + atp-games + dashboard tests green".

## Testing

### Unit (`game-environments/tests/test_el_farol.py`)

**Renamed/relabelled (legacy regression):**
- `test_compute_round_payoffs_happy_minus_crowded` →
  `test_compute_round_payoffs_legacy_mode_happy_minus_crowded`. Add
  `scoring_mode="happy_minus_crowded"` to the config.
- All other tests asserting `t_happy / t_crowded` or `happy − crowded`:
  tag config with the legacy mode.

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

8. `test_happy_only_per_round_payoff_is_non_negative` — property
   test (1000 random action draws): under `happy_only`, every
   per-round payoff ≥ 0. (Under legacy this can be negative.)

9. `test_scoring_mode_dataclass_round_trip` — `ElFarolConfig(
   scoring_mode="happy_minus_crowded")` survives `dataclasses.asdict`
   → re-construct → equal config.

### atp-games tests

- `atp-games/tests/test_round_payoffs.py` — assertions on
  `happy − crowded`: tag the fixture's config with
  `scoring_mode="happy_minus_crowded"`. Optionally add a
  happy_only-mode counterpart.
- `atp-games/tests/test_el_farol_interval_agents.py`,
  `test_runner_action_records.py` — same approach if any assertion
  depends on the formula.

### Dashboard tests

- `packages/atp-dashboard/...` — `test_game_copy_el_farol_describes_happy_only`
  (new): `GAME_COPY["el_farol"].payoff_formula` contains "happy
  slots" and is free of "−1". Add to whichever existing test file
  covers `game_copy.py` (or create a new one if absent).

### Coverage target

≥80% on new code (`scoring_mode` field + branches in three methods).

### Acceptance gates (CLAUDE.md hard requirements)

- All engine + atp-games + dashboard tests green.
- `uv run pyrefly check` clean across all modified files.
- `uv run ruff check` clean.

## Migration plan

No data migration. The `scoring_mode` config field is a new
dataclass attribute with a default; absence in stored configs is
fine.

The participant-kit changelog + version bump is documented but not
enforced by the engine — it is a courtesy notice for external bot
authors. The kit lives in `participant-kit-el-farol-en/` and is
published independently.

### Rollout precondition (procedural, not code)

**Before deploying this change to production**, run on the prod DB:

```sql
SELECT id, status, game_type, num_players, created_at
FROM tournaments
WHERE status IN ('pending', 'active') AND game_type = 'el_farol';
```

- **0 rows** → safe to deploy.
- **≥1 row** → either wait for those tournaments to complete /
  cancel them via `POST /api/v1/tournaments/{id}/cancel`, OR
  explicitly accept that mid-flight rounds will use the new formula
  and `total_score` will mix old and new per-round values (see Edge
  cases). The acceptance must be documented in the deploy commit /
  PR description.

This precondition is operational, not enforced by code, but skipping
it is the only way to produce surprising mid-tournament behaviour.

## Open questions

- **Tournament-level opt-in to legacy mode** — out of scope for v1.
  If a real demand emerges, the work is: extend `create_tournament()`
  to accept `scoring_mode`, store it in `Tournament.config`, plumb
  through `_el_farol_for()` factory (and update its cache key from
  `(num_players,)` to `(num_players, scoring_mode)`), surface in
  admin UI / REST schema. **Future-tournament legacy semantics
  (frozen here so the v2 plan doesn't have to re-litigate it):**
  per-round payoff = `happy − crowded` (already the engine
  branch); tournament `total_score = sum(happy − crowded)` (which
  matches what the existing dashboard would compute via
  `sum(Action.payoff)`). The standalone-runner ratio formula
  (`t_happy / max(t_crowded, 0.1)`) stays standalone-only and is
  NOT introduced into tournaments. Estimate: separate ~2-3 task plan.
- **Bot self-awareness of mode** — observation does not currently
  carry `scoring_mode`. Since v1 ships `happy_only` everywhere
  (tournaments + dashboard copy + rules docs), bots can safely
  assume `happy_only` until tournament-level configurability is
  introduced. When that lands, observation must also include
  `scoring_mode`.
- **About-page mode awareness** — see non-goals; intentionally
  unconditional in v1.
- **`min_total_hours` engine-vs-tournament asymmetry** — pre-existing
  before this change. `get_payoffs()` enforces the disqualification;
  tournament `total_score = sum(Action.payoff)` does NOT. Real
  sandbagging risk: a bot that stays home 99% of rounds but catches
  rare happy slots passes in tournaments while being disqualified
  standalone. Out of scope here, but worth filing a follow-up
  Linear / GitHub issue so it's not forgotten.

## References

- `game-environments/game_envs/games/el_farol.py` — game class
- `packages/atp-dashboard/atp/dashboard/tournament/service.py:120-128`
  (factory) and `1578-1582` (total_score = sum(Action.payoff))
- `packages/atp-dashboard/atp/dashboard/v2/game_copy.py` — about copy
- `docs/games/rules/el-farol-bar.ru.md`,
  `docs/games/rules/el-farol-bar.en.md` — rules
- `demo-game/agents/el_farol_agent.py` — local demo agent
- `docs/superpowers/specs/2026-04-15-el-farol-tournament-design.md`
  — original El Farol tournament design (sets the
  `total_score = sum(Action.payoff)` contract)
- `participant-kit-el-farol-en/README.md` — semi-public bot kit
