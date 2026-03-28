# Game Theory Coverage Report — Phase 0 Audit

**Date:** 2026-03-28
**Scope:** `game-environments/` library and `atp-games/` plugin
**Purpose:** Document the current state of game-theoretic evaluation infrastructure and identify gaps for Phase 1.

---

## 1. Implemented Games

Seven games are implemented in `game-environments/game_envs/games/`.

| Game | File | Players | Action Space | Nash Equilibrium | Test Status |
|------|------|---------|--------------|-----------------|-------------|
| Prisoner's Dilemma | `prisoners_dilemma.py` | 2 | Discrete {cooperate, defect} | (Defect, Defect) — strict dominant strategy | Correctness + edge cases |
| Sealed-Bid Auction | `auction.py` | 2+ | Continuous [min_bid, max_bid] | First-price: shade bid; Second-price: truthful bidding | Correctness + edge cases |
| Public Goods Game | `public_goods.py` | 2–20 | Continuous [0, endowment] | Full free-riding (Nash), full contribution (social optimum) | Correctness + edge cases |
| Colonel Blotto | `colonel_blotto.py` | 2 | Structured (allocation vector) | Mixed strategy NE (no pure strategy NE) | Correctness + edge cases |
| Congestion Game | `congestion.py` | 2–50 | Discrete (route name) | Wardrop equilibrium (pure strategy NE always exists) | Correctness + edge cases |
| Stag Hunt | `stag_hunt.py` | 2 | Discrete {stag, hare} | Two pure NE: (Stag, Stag) and (Hare, Hare) | Correctness + edge cases |
| Battle of the Sexes | `battle_of_sexes.py` | 2 | Discrete {A, B} | Two pure NE: (A, A) and (B, B) + one mixed NE | Correctness + edge cases |

All seven games support repeated play via `num_rounds` and a `discount_factor` parameter. All use `@register_game` for registry lookup.

---

## 2. Strategies Implemented

Strategies live in `game-environments/game_envs/strategies/`.

| Game | Strategy Classes | Count |
|------|-----------------|-------|
| Prisoner's Dilemma | `AlwaysCooperate`, `AlwaysDefect`, `TitForTat`, `GrimTrigger`, `Pavlov`, `RandomStrategy` | 6 |
| Sealed-Bid Auction | `TruthfulBidder`, `ShadeBidder`, `RandomBidder` | 3 |
| Colonel Blotto | `UniformAllocation`, `ConcentratedAllocation`, `NashMixed` | 3 |
| Congestion Game | `SelfishRouter`, `SocialOptimum`, `EpsilonGreedy` | 3 |
| Public Goods Game | `FullContributor`, `FreeRider`, `ConditionalCooperator`, `Punisher` | 4 |
| Stag Hunt | `AlwaysStag`, `AlwaysHare`, `StagTitForTat` | 3 |
| Battle of the Sexes | `AlwaysA`, `AlwaysB`, `Alternating` | 3 |

**Total: 25 strategies across 7 games.**

Each strategy implements the `Strategy` protocol: `name: str` property and `choose_action(observation: Observation) -> Any`. Stateful strategies expose a `reset()` method.

---

## 3. Analysis Modules

Analysis modules live in `game-environments/game_envs/analysis/`.

| Module | File | Capabilities |
|--------|------|-------------|
| Nash Solver | `nash_solver.py` | Support enumeration (all NE, 2-player), Lemke-Howson (single NE, 2-player), fictitious play (approximate, n-player), replicator dynamics (evolutionary) |
| Cooperation Analysis | `cooperation.py` | Cooperation rates, reciprocity, social welfare metrics |
| Exploitability | `exploitability.py` | Best-response computation, exploitability gap measurement |
| Fairness | `fairness.py` | Gini coefficient, payoff distribution analysis, bias detection |
| Population Dynamics | `population.py` | Evolutionary dynamics, strategy fitness tracking |
| Models | `models.py` | `NashEquilibrium` and related data models for analysis output |

---

## 4. Tournament Infrastructure

Tournament infrastructure lives in `atp-games/atp_games/suites/`.

| Component | File | Description |
|-----------|------|-------------|
| Round-Robin Tournament | `tournament.py` | All-pairs matching with `Standing` tracking (wins/losses/draws/points) |
| Single Elimination | `tournament.py` | Bracket-based single elimination |
| Double Elimination | `tournament.py` | Bracket-based double elimination with losers bracket |
| Cross-Play Suite | `cross_play.py` | Mixed strategy vs strategy experiments |
| Alympics Benchmark | `alympics.py` | Composite scoring across all 5 games (strategic 30%, cooperation 25%, fairness 25%, robustness 20%) |
| Stress Tests | `stress_test.py` | Best-response generation and exploitability measurement |
| Game Suite Loader | `game_suite_loader.py` | YAML-based suite definition loading |

Built-in YAML suites: `prisoners_dilemma.yaml`, `auction_battery.yaml`, `public_goods.yaml`, `alympics_lite.yaml`.

### 4.1 Evaluators

Five evaluators are implemented in `atp-games/atp_games/evaluators/`:

| Evaluator | File | What It Measures |
|-----------|------|-----------------|
| `CooperationEvaluator` | `cooperation_evaluator.py` | Cooperation rate, reciprocity across rounds |
| `EquilibriumEvaluator` | `equilibrium_evaluator.py` | Deviation from Nash equilibrium actions |
| `ExploitabilityEvaluator` | `exploitability_evaluator.py` | Exploitability gap vs best-response opponent |
| `FairnessEvaluator` | `fairness_evaluator.py` | Payoff Gini coefficient, bias detection by player attribute |
| `PayoffEvaluator` | `payoff_evaluator.py` | Absolute and relative payoff scores |

---

## 5. Testing Coverage

### 5.1 game-environments tests (`game-environments/tests/`)

| Test File | Type | What It Covers |
|-----------|------|---------------|
| `test_action_properties.py` | Property-based (Hypothesis, 100–200 examples per property) | Action space sampling, validation invariants for all 5 games |
| `test_game.py` | Unit | Core `Game` base class lifecycle |
| `test_prisoners_dilemma.py` | Unit | PD payoff matrix, repeated play |
| `test_auction.py` | Unit | First-price and second-price mechanics |
| `test_colonel_blotto.py` | Unit | Battlefield scoring, allocation validation |
| `test_congestion.py` | Unit | Latency formula, Wardrop equilibrium properties |
| `test_public_goods.py` | Unit | Payoff formula, punishment stage |
| `test_strategies/` | Unit | Per-game strategy tests (5 files) |
| `test_nash_solver.py` | Unit | All four NE solver algorithms |
| `test_cooperation.py` | Unit | Cooperation rate and reciprocity metrics |
| `test_exploitability.py` | Unit | Exploitability gap calculation |
| `test_fairness.py` | Unit | Gini coefficient, bias detection |
| `test_population.py` | Unit | Replicator dynamics and strategy fitness |
| `test_action.py`, `test_state.py`, `test_history.py` | Unit | Core data model tests |
| `test_communication.py` | Unit | `InformationSet` messaging model |
| `test_registry.py` | Unit | Game and strategy registry lookup |

### 5.2 atp-platform tests (`tests/unit/`)

| Test File | Type | What It Covers |
|-----------|------|---------------|
| `test_game_correctness.py` | Correctness (new, Phase 0) | All 5 games: verifies payoff formulas against theoretical definitions, dominant strategies, tie-breaking rules |
| `test_game_edge_cases.py` | Edge cases (new, Phase 0) | All 5 games: boundary inputs, terminal state re-step, reset behavior, invalid config detection |

---

## 6. Edge Case Findings

The following surprising behaviors were documented in Task 6 (`test_game_edge_cases.py`):

### 6.1 Public Goods Game: No Action Range Validation

`PublicGoodsGame.step()` does not validate contribution values. Negative contributions and contributions exceeding the endowment are silently accepted and the payoff formula is applied as-is. This is documented as expected behavior in the tests.

- `test_negative_contribution_accepted_formula_applies`: contribution of `-5.0` is processed normally.
- `test_contribution_exceeding_endowment_accepted`: contribution of `endowment + 10.0` is processed normally.

**Risk:** LLM agents may produce out-of-range bids/contributions, silently yielding economically nonsensical results (e.g., negative payoffs that do not reflect game theory).

### 6.2 Auction: No Action Range Validation

`Auction.step()` does not validate that bids fall within `[min_bid, max_bid]`. A bid of `200.0` with `max_bid=100.0` is accepted and produces a valid (though negative) payoff.

- `test_bid_above_max_not_validated`: bid of `200.0` produces payoff `value - 200 = -120`.

### 6.3 Congestion Game: KeyError for Unknown Route (Not ValueError)

When a player provides an unknown route name, `CongestionGame.step()` raises `KeyError` rather than `ValueError`. This is an inconsistency: all other games raise `ValueError` for invalid actions.

- `test_invalid_route_name_raises_key_error`: `pytest.raises(KeyError)` confirms the current behavior.
- This behavior stems from dict-lookup on the route map without a guard.

---

## 7. Resolved Gaps (Implemented in Phase 1)

### 7.1 Games Added

| Game | Status | Implementation |
|------|--------|----------------|
| Stag Hunt | DONE | `game-environments/game_envs/games/stag_hunt.py` with 3 strategies (`AlwaysStag`, `AlwaysHare`, `StagTitForTat`) |
| Battle of the Sexes | DONE | `game-environments/game_envs/games/battle_of_sexes.py` with 3 strategies (`AlwaysA`, `AlwaysB`, `Alternating`) |

### 7.2 Infrastructure Features Added

| Feature | Status | Implementation |
|---------|--------|----------------|
| Elo rating system | DONE | `atp-games/atp_games/rating/elo.py` — `EloCalculator` with configurable K-factor, integrated into `run_round_robin` as optional parameter |

### 7.3 Remaining Gaps

| Feature | Priority | Rationale |
|---------|----------|-----------|
| Result caching | MEDIUM | Game results are recomputed on every tournament run. A cache keyed on (game_id, strategy_a, strategy_b, seed) would eliminate redundant computation in large round-robin tournaments. |
| LLM reproducibility via `temperature=0` | MEDIUM | The `demo-game` OpenAI agent uses `temperature=0.3`. For deterministic evaluation runs, `temperature=0` is required. This is not enforced or documented in the ATP game runner configuration. |

### 7.3 Validation Gaps

| Gap | Priority | Recommended Fix |
|-----|----------|----------------|
| Public Goods: no bid range validation in `step()` | MEDIUM | Add clamp or raise `ValueError` for contributions outside `[0, endowment]`. |
| Auction: no bid range validation in `step()` | MEDIUM | Add `ValueError` for bids outside `[min_bid, max_bid]` in `step()`. |
| Congestion: `KeyError` instead of `ValueError` | LOW | Wrap route lookup with explicit `ValueError` message. |

---

## 8. Remaining Recommendations

1. **Fix Congestion `KeyError`** in `game-environments/game_envs/games/congestion.py` — guard the route dict lookup with a clear `ValueError` before the `KeyError` can propagate.

2. **Add input validation to Auction and Public Goods `step()`** — decide whether to clamp (lenient) or raise (strict). Document the chosen behavior explicitly.

3. **Enforce `temperature=0` in ATP game runner** for evaluation suites. Expose it as a config parameter in `GameRunConfig` (currently absent) so reproducibility is explicit rather than agent-specific.

4. **Add result caching** for tournament runs to avoid redundant computation in large round-robin brackets.

---

## Appendix: File Inventory

```
game-environments/
  game_envs/
    games/           7 game files + registry.py
    strategies/      7 strategy files + registry.py
    analysis/        nash_solver.py, cooperation.py, exploitability.py,
                     fairness.py, population.py, models.py
  tests/             ~20 test files (unit + property-based)

atp-games/
  atp_games/
    suites/          tournament.py, alympics.py, cross_play.py,
                     stress_test.py, game_suite_loader.py
                     builtin/ (4 YAML suites)
    evaluators/      5 evaluator files

tests/unit/
  test_game_correctness.py   (Phase 0, new)
  test_game_edge_cases.py    (Phase 0, new)
```
