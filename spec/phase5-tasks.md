# Phase 5: Game-Theoretic Evaluation — Tasks

> Implementation tasks for game-environments and atp-games packages (Q1–Q2 2026)
> Per ADR-002, Phase 5 Requirements, Phase 5 Design

## Legend

**Priority:**
| Emoji | Code | Description |
|-------|------|-------------|
| 🔴 | P0 | Critical — blocks release |
| 🟠 | P1 | High — needed for full functionality |
| 🟡 | P2 | Medium — improves experience |
| 🟢 | P3 | Low — nice to have |

**Status:**
| Emoji | Status | Description |
|-------|--------|-------------|
| ⬜ | TODO | Not started |
| 🔄 | IN PROGRESS | In work |
| ✅ | DONE | Completed |
| ⏸️ | BLOCKED | Waiting on dependency |

---

## Milestone 9: game-environments Core

### TASK-901: Core Abstractions & Models
🔴 P0 | ✅ DONE | Est: 6-8h

**Description:**
Create the `game-environments` package with core abstractions: `Game`, `GameState`, `Observation`, `ActionSpace`, `Strategy`, `History`, and all supporting data models.

**Checklist:**
- [x] Initialize `game-environments/` package with `pyproject.toml` (no ATP deps)
- [x] Implement `Game` abstract base class (`reset`, `step`, `get_payoffs`, `is_terminal`, `observe`, `action_space`)
- [x] Implement `GameConfig` (frozen dataclass: `num_players`, `num_rounds`, `discount_factor`, `noise`, `communication`, `seed`)
- [x] Implement `GameState`, `Observation`, `RoundResult`, `StepResult` data models
- [x] Implement `ActionSpace` ABC + `DiscreteActionSpace`, `ContinuousActionSpace`, `StructuredActionSpace`
- [x] Implement `Strategy` ABC with `choose_action(observation)` and `reset()`
- [x] Implement `GameHistory` with per-player filtering and serialization
- [x] Implement `Observation.to_prompt()` for LLM-friendly text output
- [x] Implement `Observation.to_dict()` for JSON serialization
- [x] Add type hints throughout, Pydantic validation where needed
- [x] Write unit tests for all models (serialization roundtrip, validation, edge cases)
- [x] Write property-based tests (Hypothesis) for ActionSpace.contains invariants

**Traces to:** [GE-FR-001], [GE-FR-002], [GE-FR-003]
**Depends on:** —
**Blocks:** [TASK-902], [TASK-903], [TASK-904..908], [TASK-909]

---

### TASK-902: Game Registry & Factory
🟠 P1 | ✅ DONE | Est: 2-3h

**Description:**
Create a game registry that maps game names to classes, with factory method for instantiation from config dicts / YAML.

**Checklist:**
- [x] Implement `GameRegistry` (singleton) with `register(name, cls)` and `create(name, config) → Game`
- [x] Auto-registration of built-in games via decorator `@register_game("prisoners_dilemma")`
- [x] Factory accepts dict config (for YAML deserialization)
- [x] `list_games()` returns available game names with metadata
- [x] `game_info(name)` returns description, action spaces, config schema
- [x] Write unit tests for registry and factory

**Traces to:** [GE-FR-001]
**Depends on:** [TASK-901]
**Blocks:** [TASK-914]

---

### TASK-903: Communication Channel & Information Sets
🟡 P2 | ✅ DONE | Est: 3-4h

**Description:**
Implement optional communication channel between agents and information set management for partial observability.

**Checklist:**
- [x] Implement `Message` data model (`sender`, `content`, `round`, `timestamp`)
- [x] Implement `CommunicationChannel` with modes: `no_communication`, `pre_action`, `post_action`, `free`
- [x] Implement `InformationSet` for partial observability (visible state subset)
- [x] Integrate communication into `Game` base class (optional, via config)
- [x] Integrate into `Observation` model (messages field)
- [x] Write tests: PD with pre-action communication, sealed-bid with hidden information

**Traces to:** [GE-FR-003], [GE-FR-004]
**Depends on:** [TASK-901]
**Blocks:** —

---

### TASK-904: Prisoner's Dilemma Implementation
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Implement Prisoner's Dilemma game: one-shot and repeated, with configurable payoff matrix, noise, and optional communication.

**Checklist:**
- [x] Implement `PDConfig(GameConfig)` with R/S/T/P payoff params
- [x] Implement `PrisonersDilemma(Game)` — one-shot mode
- [x] Extend to repeated mode with `num_rounds > 1`
- [x] Implement noise (action flip with probability `noise`)
- [x] Implement `observe()` with history of past rounds
- [x] Implement `to_prompt()` for LLM-friendly description
- [x] Verify payoff matrix analytically (T > R > P > S, 2R > T + S)
- [x] Write tests: one-shot all 4 outcomes, repeated 10 rounds, noise flip rate
- [x] Register in GameRegistry

**Traces to:** [GE-FR-005]
**Depends on:** [TASK-901]
**Blocks:** [TASK-909]

---

### TASK-905: Public Goods Game Implementation
🔴 P0 | ✅ DONE | Est: 3-4h

**Description:**
Implement Public Goods Game with configurable multiplier, endowment, and optional punishment mechanism.

**Checklist:**
- [x] Implement `PGConfig(GameConfig)` with `endowment`, `multiplier`, `punishment_cost`, `punishment_effect`
- [x] Implement `PublicGoodsGame(Game)` — continuous contribution [0, endowment]
- [x] Implement payoff: `payoff_i = endowment - contribution_i + multiplier * sum(contributions) / n`
- [x] Implement punishment variant (2-stage: contribute → punish)
- [x] Support n-player (2 to 20)
- [x] Implement `to_prompt()` describing the public goods scenario
- [x] Verify: dominant strategy in one-shot = free ride; social optimum = full contribute
- [x] Write tests: free riding payoff > cooperation payoff (one-shot), n-player sum checks
- [x] Register in GameRegistry

**Traces to:** [GE-FR-005]
**Depends on:** [TASK-901]
**Blocks:** [TASK-909]

---

### TASK-906: Auction Implementation
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Implement First-Price and Second-Price (Vickrey) sealed-bid auctions with private values.

**Checklist:**
- [x] Implement `AuctionConfig(GameConfig)` with `auction_type`, `min_bid`, `max_bid`, `reserve_price`, `value_distribution`
- [x] Implement `Auction(Game)` — sealed-bid with `ContinuousActionSpace`
- [x] Implement private value draw (uniform, normal) during `reset()`
- [x] Implement `observe()` showing own value but NOT others' (partial observability)
- [x] First-price: winner pays own bid
- [x] Second-price: winner pays second-highest bid
- [x] Handle reserve price (no winner if all bids below)
- [x] Implement `to_prompt()` describing auction scenario
- [x] Verify: truthful bidding dominant in second-price (Vickrey)
- [x] Verify: optimal shade in first-price with uniform values = (n-1)/n * value
- [x] Write tests: winner selection, payment computation, reserve price, ties
- [x] Register in GameRegistry

**Traces to:** [GE-FR-005]
**Depends on:** [TASK-901]
**Blocks:** [TASK-909]

---

### TASK-907: Colonel Blotto Implementation
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement Colonel Blotto game with configurable number of battlefields and troops.

**Checklist:**
- [x] Implement `BlottoConfig(GameConfig)` with `num_battlefields`, `total_troops`
- [x] Implement `Blotto(Game)` — `StructuredActionSpace` (allocation vector)
- [x] Validate: allocation sums to `total_troops`, all non-negative
- [x] Payoff: player wins battlefield if more troops; tie → split; payoff = fraction of battlefields won
- [x] Implement `to_prompt()` describing battlefield allocation
- [x] Verify: no pure Nash for symmetric Blotto (known result)
- [x] Write tests: allocation validation, payoff computation, ties
- [x] Register in GameRegistry

**Traces to:** [GE-FR-005]
**Depends on:** [TASK-901]
**Blocks:** [TASK-909]

---

### TASK-908: Congestion Game Implementation
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement Congestion (routing) game with configurable network and latency functions.

**Checklist:**
- [x] Implement `CongestionConfig(GameConfig)` with `routes` (list of route definitions), `latency_functions`
- [x] Implement `CongestionGame(Game)` — choose route, cost depends on congestion
- [x] Latency function: `latency(route) = base_cost + coefficient * num_users_on_route`
- [x] Payoff = negative latency (lower is better)
- [x] Support n-player (2 to 50)
- [x] Implement `to_prompt()` describing routing scenario
- [x] Verify: Braess's paradox example (adding route worsens outcomes)
- [x] Write tests: latency computation, Nash flow vs social optimum
- [x] Register in GameRegistry

**Traces to:** [GE-FR-005]
**Depends on:** [TASK-901]
**Blocks:** [TASK-909]

---

### TASK-909: Baseline Strategies
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Implement built-in baseline strategies for all 5 canonical games.

**Checklist:**
- [x] PD strategies: `TitForTat`, `AlwaysCooperate`, `AlwaysDefect`, `GrimTrigger`, `Pavlov`, `RandomStrategy`
- [x] Public Goods strategies: `FullContributor`, `FreeRider`, `ConditionalCooperator`, `Punisher`
- [x] Auction strategies: `TruthfulBidder`, `ShadeBidder(factor)`, `RandomBidder`
- [x] Blotto strategies: `UniformAllocation`, `ConcentratedAllocation`, `NashMixed` (approximate)
- [x] Congestion strategies: `SelfishRouter`, `SocialOptimum`, `EpsilonGreedy`
- [x] All strategies extend `Strategy` ABC with `choose_action(observation)`
- [x] Strategy `reset()` clears internal state between episodes
- [x] Write tests: TFT behavior verified round-by-round, truthful bidding in Vickrey wins, AllC/AllD payoffs match theory
- [x] Strategy registry (`StrategyRegistry`) with lookup by name

**Traces to:** [GE-FR-006]
**Depends on:** [TASK-901], [TASK-904..908]
**Blocks:** [TASK-912]

---

## Milestone 10: game-environments Analysis

### TASK-910: Nash Solver & Exploitability
🔴 P0 | ✅ DONE | Est: 6-8h

**Description:**
Implement Nash equilibrium solver and exploitability calculator.

**Checklist:**
- [x] Implement `NashEquilibrium` data model (strategies, payoffs, support, epsilon)
- [x] Implement support enumeration for 2-player bimatrix games
- [x] Implement Lemke-Howson for efficient single NE computation (2-player)
- [x] Implement fictitious play for n-player approximate NE
- [x] Implement replicator dynamics for evolutionary equilibrium
- [x] Implement `EmpiricalStrategy.from_history()` — extract strategy from game log
- [x] Implement `compute_exploitability()` — best response payoff gap
- [x] Implement best response oracle for discrete action games
- [x] Verify on known games:
  - [ ] PD: (Defect, Defect) is unique NE in one-shot
  - [ ] Matching Pennies: (0.5, 0.5) mixed NE
  - [ ] Battle of the Sexes: 2 pure + 1 mixed NE
  - [ ] Second-price auction: truthful bidding NE
- [x] Write tests: solver convergence, exploitability = 0 for NE strategy, exploitability > 0 for dominated strategy
- [x] Performance: solve 10x10 bimatrix < 1s, 100x100 < 10s

**Traces to:** [GE-FR-007], [GE-FR-008]
**Depends on:** [TASK-901]
**Blocks:** [TASK-915], [TASK-916]

---

### TASK-911: Cooperation, Fairness & Population Analysis
🟠 P1 | ✅ DONE | Est: 5-6h

**Description:**
Implement cooperation metrics, fairness analysis, and population dynamics simulation.

**Checklist:**
- [x] Cooperation metrics:
  - [ ] `cooperation_rate(history, player)` — fraction of cooperative actions
  - [ ] `conditional_cooperation(history, player)` — P(C|C) vs P(C|D)
  - [ ] `reciprocity_index(history)` — correlation of cooperation between players
- [x] Fairness metrics:
  - [ ] `gini_coefficient(payoffs)` — inequality measure
  - [ ] `envy_freeness(allocations)` — check if any player envies another
  - [ ] `proportionality(payoffs, entitlements)` — proportional fairness
  - [ ] `utilitarian_welfare(payoffs)` — sum of payoffs
- [x] Population dynamics:
  - [ ] `ReplicatorDynamics` — continuous-time evolution of strategy frequencies
  - [ ] `MoranProcess` — stochastic finite-population model
  - [ ] `is_ess(strategy, game)` — evolutionary stable strategy check
  - [ ] `PopulationSimulator` — run populations over generations with mutation
- [x] Output: strategy frequency timeseries, convergence detection, ESS classification
- [x] Write tests:
  - [ ] TFT cooperation rate ≈ 1.0 vs TFT, ≈ 0.5 vs Random
  - [ ] Gini = 0 for equal payoffs, Gini > 0 for unequal
  - [ ] Replicator dynamics converges to AllD in one-shot PD
  - [ ] TFT is ESS in repeated PD with sufficient discount factor

**Traces to:** [GE-FR-009], [GE-FR-010]
**Depends on:** [TASK-901], [TASK-904]
**Blocks:** [TASK-917]

---

## Milestone 11: atp-games Plugin Core

### TASK-912: GameRunner & Protocol Mapping
🔴 P0 | ✅ DONE | Est: 6-8h

**Description:**
Implement `GameRunner` extending ATP Runner with game loop orchestration, and ATP protocol mapping for observations/actions.

**Checklist:**
- [x] Initialize `atp-games/` package with `pyproject.toml` (depends on `atp-platform`, `game-environments`)
- [x] Implement `ObservationMapper.to_atp_request()` — `GameObservation` → `ATPRequest`
- [x] Implement `ActionMapper.from_atp_response()` — `ATPResponse` → `GameAction`
- [x] Implement `GameRunner(BaseRunner)` with `run_game(game, agents, config) → GameResult`
- [x] Implement game loop: reset → observe → map → send → validate → step → repeat
- [x] Implement `_parallel_moves()` for simultaneous move games
- [x] Implement `_sequential_moves()` for sequential move games
- [x] Implement `ActionValidator` with retry logic (max 3 attempts + default)
- [x] Implement `BuiltinAdapter` wrapping `Strategy` into ATP adapter interface
- [x] Implement `GameResult`, `EpisodeResult` data models
- [x] Write unit tests: mapper roundtrip, validator retry, runner with mock agents
- [x] Write integration test: full PD game with 2 builtin strategies

**Traces to:** [AG-FR-001], [AG-FR-002], [AG-FR-003]
**Depends on:** [TASK-901], [TASK-909]
**Blocks:** [TASK-914], [TASK-915..917], [TASK-918]

---

### TASK-913: Multi-Episode & Concurrency
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement multi-episode execution with statistical aggregation and parallel episode support.

**Checklist:**
- [x] Implement episode-level parallelism (configurable `--parallel=N`)
- [x] Implement seed management: each episode gets deterministic seed `base_seed + episode`
- [x] Implement aggregation: mean payoff, std, 95% CI per player per metric
- [x] Implement Welch's t-test for comparing agents (reuse ATP statistical module)
- [x] Implement Bonferroni correction for multiple comparisons
- [x] Implement progress reporting (episode N/M, ETA)
- [x] Write tests: 50 episodes aggregation, CI coverage, parallel vs sequential same results (deterministic seeds)

**Traces to:** [AG-FR-001], [AG-NFR-002], [AG-NFR-003]
**Depends on:** [TASK-912]
**Blocks:** [TASK-918]

---

### TASK-914: Game Suite YAML Loader
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Implement YAML parser for game suite definitions and CLI integration.

**Checklist:**
- [x] Define JSON Schema for `game_suite` YAML format
- [x] Implement `GameSuiteLoader` — parse YAML, validate against schema
- [x] Resolve `game.type` via `GameRegistry`, apply `game.config`
- [x] Resolve `agents` — create ATP adapters or wrap builtin strategies
- [x] Resolve `evaluation` — instantiate evaluators with weights/thresholds
- [x] Support variable substitution (`${AGENT_ENDPOINT}`) for CI
- [x] Support suite inheritance (`extends: base_pd.yaml`)
- [x] Register loader in ATP plugin system (`type: game_suite`)
- [x] Implement CLI: `atp test --suite=game:prisoners_dilemma.yaml`
- [x] Create 3 builtin suite YAMLs:
  - [ ] `prisoners_dilemma.yaml` — basic repeated PD evaluation
  - [ ] `auction_battery.yaml` — first-price + second-price with baselines
  - [ ] `alympics_lite.yaml` — battery of all 5 games
- [x] Write tests: YAML parsing, validation errors, variable substitution

**Traces to:** [AG-FR-004]
**Depends on:** [TASK-902], [TASK-912]
**Blocks:** [TASK-918]

---

### TASK-915: PayoffEvaluator & ExploitabilityEvaluator
🔴 P0 | ✅ DONE | Est: 4-5h

**Description:**
Implement game-theoretic evaluators for payoff analysis and exploitability measurement.

**Checklist:**
- [x] Implement `PayoffEvaluator`:
  - [ ] Average payoff per player per episode
  - [ ] Payoff distribution (min, max, percentiles)
  - [ ] Pareto efficiency check (is outcome Pareto optimal?)
  - [ ] Social welfare (sum of payoffs)
  - [ ] Configurable thresholds and weights
- [x] Implement `ExploitabilityEvaluator`:
  - [ ] Extract empirical strategy from game history
  - [ ] Compute best response using `game_envs.analysis.exploitability`
  - [ ] Report exploitability score per player and aggregate
  - [ ] Configurable epsilon threshold for "pass" (default: 0.15)
- [x] Register both in ATP evaluator registry
- [x] Output compatible with ATP scoring system
- [x] Write tests: payoff evaluator on known outcomes, exploitability = 0 for NE play, > 0 for AllC

**Traces to:** [AG-FR-005]
**Depends on:** [TASK-910], [TASK-912]
**Blocks:** [TASK-920]

---

### TASK-916: CooperationEvaluator & EquilibriumEvaluator
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Implement evaluators for cooperation dynamics and equilibrium convergence analysis.

**Checklist:**
- [x] Implement `CooperationEvaluator`:
  - [ ] Cooperation rate (overall and per-round trend)
  - [ ] Conditional cooperation P(C|C), P(C|D)
  - [ ] Reciprocity index
  - [ ] Cooperation stability (variance over episodes)
  - [ ] Thresholds: min cooperation rate, max defection streak
- [x] Implement `EquilibriumEvaluator`:
  - [ ] Classify convergence type: Nash, correlated, cyclic, no convergence
  - [ ] Convergence speed (rounds to stabilize)
  - [ ] Distance to known equilibria
  - [ ] Strategy entropy over time (decreasing → converging)
- [x] Register both in ATP evaluator registry
- [x] Write tests: TFT vs TFT → cooperation ≈ 1.0, AllD vs AllC → cooperation = 0.5, convergence detection

**Traces to:** [AG-FR-005]
**Depends on:** [TASK-911], [TASK-912]
**Blocks:** [TASK-920]

---

### TASK-917: FairnessEvaluator (FAIRGAME)
🟡 P2 | ✅ DONE | Est: 4-5h

**Description:**
Implement bias detection evaluator inspired by FAIRGAME methodology.

**Checklist:**
- [x] Implement `FairnessEvaluator`:
  - [ ] Gini coefficient of payoff distribution across agents
  - [ ] Envy-freeness check for allocation games
  - [ ] Proportionality check
- [x] Implement FAIRGAME-style bias detection:
  - [ ] Demographic variation: same game but opponent descriptions vary (gender, ethnicity, etc.)
  - [ ] Strategy shift measurement: does agent change behavior based on irrelevant opponent attributes?
  - [ ] Discrimination score: max difference in cooperation/aggression across groups
  - [ ] Statistical significance test (chi-squared / Fisher's exact)
- [x] Bias report generation: which attributes cause behavioral shifts, magnitude, p-values
- [x] Register in ATP evaluator registry
- [x] Write tests: no bias when opponent description doesn't vary, synthetic bias injection detected

**Traces to:** [AG-FR-009]
**Depends on:** [TASK-911], [TASK-912]
**Blocks:** [TASK-920]

---

## Milestone 12: Advanced Scenarios & Reporting

### TASK-918: Tournament, Cross-Play & Stress-Test
🟠 P1 | ✅ DONE | Est: 6-8h

**Description:**
Implement tournament mode, cross-play matrix, and adversarial stress-test.

**Checklist:**
- [x] Tournament — Round-Robin:
  - [ ] For each (agent_i, agent_j) pair: run N episodes
  - [ ] Compute standings: win/loss/draw, total payoff, average metrics
  - [ ] Handle byes for odd number of agents
- [x] Tournament — Elimination:
  - [ ] Single elimination bracket
  - [ ] Double elimination bracket
  - [ ] Seeding by round-robin results
- [x] Cross-Play Matrix:
  - [ ] Run every agent vs every agent (including self-play)
  - [ ] Output: payoff heatmap, dominance relationships
  - [ ] Pareto frontier identification
  - [ ] Cluster analysis (which agents play similarly?)
- [x] Adversarial Stress-Test:
  - [ ] Compute empirical strategy of agent under test
  - [ ] Generate best-response oracle
  - [ ] Run agent vs best-response, measure exploitability in practice
  - [ ] Optional: iterative (recompute BR after agent adapts)
- [x] CLI integration:
  - [ ] `atp tournament --suite=tournament.yaml`
  - [ ] `atp crossplay --suite=crossplay.yaml`
- [x] Write tests: 4-agent round-robin, elimination bracket correctness, cross-play matrix symmetry

**Traces to:** [AG-FR-006], [AG-FR-007], [AG-FR-008]
**Depends on:** [TASK-912], [TASK-913], [TASK-914]
**Blocks:** [TASK-920]

---

### TASK-919: Alympics-Style Benchmark Suite
🟡 P2 | ✅ DONE | Est: 3-4h

**Description:**
Create standardized benchmark battery covering all 5 games, inspired by Alympics paper.

**Checklist:**
- [x] Design benchmark suite: all 5 games, standardized configs, fixed episodes
- [x] Define composite score: weighted aggregate across games
- [x] Categories: strategic reasoning, cooperation, fairness, robustness
- [x] Implement `alympics_lite.yaml` suite with all games + baselines
- [x] Implement `atp benchmark --suite=alympics` shortcut command
- [x] Create scoring rubric documentation
- [x] Write integration test: run full benchmark with builtin strategies, verify scores
- [x] Example output: agent X scored 72/100 (strategic: 85, cooperation: 60, fairness: 78, robustness: 65)

**Traces to:** [AG-FR-006]
**Depends on:** [TASK-914], [TASK-915..917]
**Blocks:** [TASK-920]

---

### TASK-920: Dashboard & Reporting Integration
🟠 P1 | ✅ DONE | Est: 5-7h

**Description:**
Integrate game results into ATP Dashboard and reporting pipeline.

**Checklist:**
- [x] Game Reporter (extends ATP reporter):
  - [ ] JSON output with game-specific fields (payoff matrix, strategy profiles)
  - [ ] HTML report with game visualizations
- [x] Dashboard routes:
  - [ ] `/games/` — list of game evaluation results
  - [ ] `/games/{id}` — detailed result with all metrics
  - [ ] `/tournaments/{id}` — tournament standings and bracket
  - [ ] `/crossplay/{id}` — cross-play heatmap
- [x] Visualizations:
  - [ ] Payoff matrix table (color-coded)
  - [ ] Strategy distribution timeline (line chart over rounds)
  - [ ] Cooperation dynamics chart
  - [ ] Cross-play heatmap (Chart.js or D3-inline)
  - [ ] Tournament bracket (SVG)
- [x] Export: CSV + JSON for Jupyter analysis
- [x] Navigation: add "Games" section to dashboard sidebar
- [x] Write tests: reporter output format, dashboard routes return 200

**Traces to:** [AG-FR-010]
**Depends on:** [TASK-915..918]
**Blocks:** —

---

### TASK-921: Documentation & Examples
🟠 P1 | ✅ DONE | Est: 4-5h

**Description:**
Complete documentation for game-environments and atp-games.

**Checklist:**
- [x] `game-environments` docs:
  - [x] API reference (inline in README)
  - [x] Game development guide: how to add a new game
  - [x] Strategy development guide
  - [x] Analysis tools guide (Nash solver, exploitability, population)
- [x] `atp-games` docs:
  - [x] Installation and quick start
  - [x] Game suite YAML reference
  - [x] Evaluator configuration guide
  - [x] Tournament setup guide
- [x] Examples:
  - [x] `basic_usage.py` — run PD, strategies, tournaments
  - [x] `custom_game.py` — create new game from scratch
  - [x] `llm_agent_eval.py` — evaluate agent on game battery
  - [x] `population_dynamics.py` — evolutionary simulation
- [x] Update ATP main README with Phase 5 section
- [x] Update `docs/05-evaluators.md` with game-theoretic evaluators

**Traces to:** All Phase 5 requirements
**Depends on:** [TASK-901..920]
**Blocks:** —

---

### TASK-922: Package Publishing & CI
🟠 P1 | ✅ DONE | Est: 3-4h

**Description:**
Set up CI/CD and publish both packages.

**Checklist:**
- [x] `game-environments`:
  - [ ] GitHub repo setup
  - [ ] CI: pytest + coverage + linting (ruff)
  - [ ] Coverage gate: ≥ 90%
  - [ ] PyPI publishing workflow
  - [ ] Version: 0.1.0
- [x] `atp-games`:
  - [ ] GitHub repo (or monorepo with ATP)
  - [ ] CI: pytest + coverage + linting
  - [ ] Coverage gate: ≥ 80%
  - [ ] PyPI publishing workflow
  - [ ] Version: 0.1.0
- [x] Integration CI: test atp-games with latest game-environments
- [x] Dependabot / Renovate for dependency updates

**Traces to:** [GE-NFR-003], [AG-NFR-001]
**Depends on:** [TASK-901..921]
**Blocks:** —

---

## Dependency Graph

```
TASK-901 (Core Abstractions)
    │
    ├──► TASK-902 (Registry) ──► TASK-914 (YAML Loader) ──┐
    │                                                       │
    ├──► TASK-903 (Communication) [independent]             │
    │                                                       │
    ├──► TASK-904 (PD) ──┐                                 │
    ├──► TASK-905 (PG) ──┤                                 │
    ├──► TASK-906 (Auc) ─┤                                 │
    ├──► TASK-907 (Blotto)┤                                │
    ├──► TASK-908 (Cong) ─┤                                │
    │                     │                                 │
    │                     └──► TASK-909 (Baselines) ──┐    │
    │                                                  │    │
    ├──► TASK-910 (Nash/Exploit) ──► TASK-915 (Eval) ──┤   │
    │                                                   │   │
    └──► TASK-911 (Coop/Fair/Pop) ──► TASK-916 (Eval) ─┤   │
                                  └──► TASK-917 (Fair) ─┤   │
                                                        │   │
                          TASK-912 (GameRunner) ◄───────┘   │
                              │                             │
                              ├──► TASK-913 (Concurrency)   │
                              │                             │
                              └──► TASK-918 (Tournament) ◄──┘
                                       │
                                       ▼
                                  TASK-919 (Alympics)
                                       │
                                       ▼
                                  TASK-920 (Dashboard)
                                       │
                                       ▼
                                  TASK-921 (Docs)
                                       │
                                       ▼
                                  TASK-922 (Publish)
```

---

## Summary

| Milestone | Tasks | Total Est. Hours |
|-----------|-------|------------------|
| M9: game-environments Core | TASK-901..909 | ~28-37h |
| M10: game-environments Analysis | TASK-910..911 | ~11-14h |
| M11: atp-games Plugin Core | TASK-912..917 | ~24-31h |
| M12: Advanced & Reporting | TASK-918..922 | ~21-28h |
| **Total** | **22 tasks** | **~84-110h (~3-4 months)** |

---

## Critical Path

```
TASK-901 → TASK-904 → TASK-909 → TASK-912 → TASK-914 → TASK-918 → TASK-920 → TASK-921 → TASK-922
   8h        4h         5h         8h         5h         8h         7h         5h         4h
                                                                                    Total: ~54h
```

**Minimum duration with parallelization**: ~10-12 weeks (one developer), ~6-8 weeks (two developers)

---

## Recommended Execution Order

### Phase 5.1 (Weeks 1-4): Foundation
1. **Week 1**: TASK-901 (Core abstractions) — критический блокер
2. **Week 2**: TASK-904, TASK-905 (PD + PG) в параллели
3. **Week 3**: TASK-906, TASK-907, TASK-908 (Auction, Blotto, Congestion)
4. **Week 3-4**: TASK-909 (Baselines) + TASK-902 (Registry)

### Phase 5.2 (Weeks 5-8): Analysis & Plugin
5. **Week 5**: TASK-910 (Nash solver) + TASK-903 (Communication) в параллели
6. **Week 6**: TASK-911 (Cooperation/Fairness/Population)
7. **Week 7**: TASK-912 (GameRunner) — второй критический блокер
8. **Week 8**: TASK-913 (Concurrency) + TASK-914 (YAML loader)

### Phase 5.3 (Weeks 9-12): Evaluators & Advanced
9. **Week 9**: TASK-915, TASK-916 (Core evaluators)
10. **Week 10**: TASK-917 (Fairness/FAIRGAME) + TASK-918 (Tournament)
11. **Week 11**: TASK-919 (Alympics) + TASK-920 (Dashboard)
12. **Week 12**: TASK-921 (Docs) + TASK-922 (Publish)
