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
🟠 P1 | ⏸️ BLOCKED | Est: 2-3h

**Description:**
Create a game registry that maps game names to classes, with factory method for instantiation from config dicts / YAML.

**Checklist:**
- [ ] Implement `GameRegistry` (singleton) with `register(name, cls)` and `create(name, config) → Game`
- [ ] Auto-registration of built-in games via decorator `@register_game("prisoners_dilemma")`
- [ ] Factory accepts dict config (for YAML deserialization)
- [ ] `list_games()` returns available game names with metadata
- [ ] `game_info(name)` returns description, action spaces, config schema
- [ ] Write unit tests for registry and factory

**Traces to:** [GE-FR-001]
**Depends on:** [TASK-901]
**Blocks:** [TASK-914]

---

### TASK-903: Communication Channel & Information Sets
🟡 P2 | ⏸️ BLOCKED | Est: 3-4h

**Description:**
Implement optional communication channel between agents and information set management for partial observability.

**Checklist:**
- [ ] Implement `Message` data model (`sender`, `content`, `round`, `timestamp`)
- [ ] Implement `CommunicationChannel` with modes: `no_communication`, `pre_action`, `post_action`, `free`
- [ ] Implement `InformationSet` for partial observability (visible state subset)
- [ ] Integrate communication into `Game` base class (optional, via config)
- [ ] Integrate into `Observation` model (messages field)
- [ ] Write tests: PD with pre-action communication, sealed-bid with hidden information

**Traces to:** [GE-FR-003], [GE-FR-004]
**Depends on:** [TASK-901]
**Blocks:** —

---

### TASK-904: Prisoner's Dilemma Implementation
🔴 P0 | ⏸️ BLOCKED | Est: 3-4h

**Description:**
Implement Prisoner's Dilemma game: one-shot and repeated, with configurable payoff matrix, noise, and optional communication.

**Checklist:**
- [ ] Implement `PDConfig(GameConfig)` with R/S/T/P payoff params
- [ ] Implement `PrisonersDilemma(Game)` — one-shot mode
- [ ] Extend to repeated mode with `num_rounds > 1`
- [ ] Implement noise (action flip with probability `noise`)
- [ ] Implement `observe()` with history of past rounds
- [ ] Implement `to_prompt()` for LLM-friendly description
- [ ] Verify payoff matrix analytically (T > R > P > S, 2R > T + S)
- [ ] Write tests: one-shot all 4 outcomes, repeated 10 rounds, noise flip rate
- [ ] Register in GameRegistry

**Traces to:** [GE-FR-005]
**Depends on:** [TASK-901]
**Blocks:** [TASK-909]

---

### TASK-905: Public Goods Game Implementation
🔴 P0 | ⏸️ BLOCKED | Est: 3-4h

**Description:**
Implement Public Goods Game with configurable multiplier, endowment, and optional punishment mechanism.

**Checklist:**
- [ ] Implement `PGConfig(GameConfig)` with `endowment`, `multiplier`, `punishment_cost`, `punishment_effect`
- [ ] Implement `PublicGoodsGame(Game)` — continuous contribution [0, endowment]
- [ ] Implement payoff: `payoff_i = endowment - contribution_i + multiplier * sum(contributions) / n`
- [ ] Implement punishment variant (2-stage: contribute → punish)
- [ ] Support n-player (2 to 20)
- [ ] Implement `to_prompt()` describing the public goods scenario
- [ ] Verify: dominant strategy in one-shot = free ride; social optimum = full contribute
- [ ] Write tests: free riding payoff > cooperation payoff (one-shot), n-player sum checks
- [ ] Register in GameRegistry

**Traces to:** [GE-FR-005]
**Depends on:** [TASK-901]
**Blocks:** [TASK-909]

---

### TASK-906: Auction Implementation
🔴 P0 | ⏸️ BLOCKED | Est: 4-5h

**Description:**
Implement First-Price and Second-Price (Vickrey) sealed-bid auctions with private values.

**Checklist:**
- [ ] Implement `AuctionConfig(GameConfig)` with `auction_type`, `min_bid`, `max_bid`, `reserve_price`, `value_distribution`
- [ ] Implement `Auction(Game)` — sealed-bid with `ContinuousActionSpace`
- [ ] Implement private value draw (uniform, normal) during `reset()`
- [ ] Implement `observe()` showing own value but NOT others' (partial observability)
- [ ] First-price: winner pays own bid
- [ ] Second-price: winner pays second-highest bid
- [ ] Handle reserve price (no winner if all bids below)
- [ ] Implement `to_prompt()` describing auction scenario
- [ ] Verify: truthful bidding dominant in second-price (Vickrey)
- [ ] Verify: optimal shade in first-price with uniform values = (n-1)/n * value
- [ ] Write tests: winner selection, payment computation, reserve price, ties
- [ ] Register in GameRegistry

**Traces to:** [GE-FR-005]
**Depends on:** [TASK-901]
**Blocks:** [TASK-909]

---

### TASK-907: Colonel Blotto Implementation
🟠 P1 | ⏸️ BLOCKED | Est: 3-4h

**Description:**
Implement Colonel Blotto game with configurable number of battlefields and troops.

**Checklist:**
- [ ] Implement `BlottoConfig(GameConfig)` with `num_battlefields`, `total_troops`
- [ ] Implement `Blotto(Game)` — `StructuredActionSpace` (allocation vector)
- [ ] Validate: allocation sums to `total_troops`, all non-negative
- [ ] Payoff: player wins battlefield if more troops; tie → split; payoff = fraction of battlefields won
- [ ] Implement `to_prompt()` describing battlefield allocation
- [ ] Verify: no pure Nash for symmetric Blotto (known result)
- [ ] Write tests: allocation validation, payoff computation, ties
- [ ] Register in GameRegistry

**Traces to:** [GE-FR-005]
**Depends on:** [TASK-901]
**Blocks:** [TASK-909]

---

### TASK-908: Congestion Game Implementation
🟠 P1 | ⏸️ BLOCKED | Est: 3-4h

**Description:**
Implement Congestion (routing) game with configurable network and latency functions.

**Checklist:**
- [ ] Implement `CongestionConfig(GameConfig)` with `routes` (list of route definitions), `latency_functions`
- [ ] Implement `CongestionGame(Game)` — choose route, cost depends on congestion
- [ ] Latency function: `latency(route) = base_cost + coefficient * num_users_on_route`
- [ ] Payoff = negative latency (lower is better)
- [ ] Support n-player (2 to 50)
- [ ] Implement `to_prompt()` describing routing scenario
- [ ] Verify: Braess's paradox example (adding route worsens outcomes)
- [ ] Write tests: latency computation, Nash flow vs social optimum
- [ ] Register in GameRegistry

**Traces to:** [GE-FR-005]
**Depends on:** [TASK-901]
**Blocks:** [TASK-909]

---

### TASK-909: Baseline Strategies
🔴 P0 | ⏸️ BLOCKED | Est: 4-5h

**Description:**
Implement built-in baseline strategies for all 5 canonical games.

**Checklist:**
- [ ] PD strategies: `TitForTat`, `AlwaysCooperate`, `AlwaysDefect`, `GrimTrigger`, `Pavlov`, `RandomStrategy`
- [ ] Public Goods strategies: `FullContributor`, `FreeRider`, `ConditionalCooperator`, `Punisher`
- [ ] Auction strategies: `TruthfulBidder`, `ShadeBidder(factor)`, `RandomBidder`
- [ ] Blotto strategies: `UniformAllocation`, `ConcentratedAllocation`, `NashMixed` (approximate)
- [ ] Congestion strategies: `SelfishRouter`, `SocialOptimum`, `EpsilonGreedy`
- [ ] All strategies extend `Strategy` ABC with `choose_action(observation)`
- [ ] Strategy `reset()` clears internal state between episodes
- [ ] Write tests: TFT behavior verified round-by-round, truthful bidding in Vickrey wins, AllC/AllD payoffs match theory
- [ ] Strategy registry (`StrategyRegistry`) with lookup by name

**Traces to:** [GE-FR-006]
**Depends on:** [TASK-901], [TASK-904..908]
**Blocks:** [TASK-912]

---

## Milestone 10: game-environments Analysis

### TASK-910: Nash Solver & Exploitability
🔴 P0 | ⏸️ BLOCKED | Est: 6-8h

**Description:**
Implement Nash equilibrium solver and exploitability calculator.

**Checklist:**
- [ ] Implement `NashEquilibrium` data model (strategies, payoffs, support, epsilon)
- [ ] Implement support enumeration for 2-player bimatrix games
- [ ] Implement Lemke-Howson for efficient single NE computation (2-player)
- [ ] Implement fictitious play for n-player approximate NE
- [ ] Implement replicator dynamics for evolutionary equilibrium
- [ ] Implement `EmpiricalStrategy.from_history()` — extract strategy from game log
- [ ] Implement `compute_exploitability()` — best response payoff gap
- [ ] Implement best response oracle for discrete action games
- [ ] Verify on known games:
  - [ ] PD: (Defect, Defect) is unique NE in one-shot
  - [ ] Matching Pennies: (0.5, 0.5) mixed NE
  - [ ] Battle of the Sexes: 2 pure + 1 mixed NE
  - [ ] Second-price auction: truthful bidding NE
- [ ] Write tests: solver convergence, exploitability = 0 for NE strategy, exploitability > 0 for dominated strategy
- [ ] Performance: solve 10x10 bimatrix < 1s, 100x100 < 10s

**Traces to:** [GE-FR-007], [GE-FR-008]
**Depends on:** [TASK-901]
**Blocks:** [TASK-915], [TASK-916]

---

### TASK-911: Cooperation, Fairness & Population Analysis
🟠 P1 | ⬜ TODO | Est: 5-6h

**Description:**
Implement cooperation metrics, fairness analysis, and population dynamics simulation.

**Checklist:**
- [ ] Cooperation metrics:
  - [ ] `cooperation_rate(history, player)` — fraction of cooperative actions
  - [ ] `conditional_cooperation(history, player)` — P(C|C) vs P(C|D)
  - [ ] `reciprocity_index(history)` — correlation of cooperation between players
- [ ] Fairness metrics:
  - [ ] `gini_coefficient(payoffs)` — inequality measure
  - [ ] `envy_freeness(allocations)` — check if any player envies another
  - [ ] `proportionality(payoffs, entitlements)` — proportional fairness
  - [ ] `utilitarian_welfare(payoffs)` — sum of payoffs
- [ ] Population dynamics:
  - [ ] `ReplicatorDynamics` — continuous-time evolution of strategy frequencies
  - [ ] `MoranProcess` — stochastic finite-population model
  - [ ] `is_ess(strategy, game)` — evolutionary stable strategy check
  - [ ] `PopulationSimulator` — run populations over generations with mutation
- [ ] Output: strategy frequency timeseries, convergence detection, ESS classification
- [ ] Write tests:
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
🔴 P0 | ⬜ TODO | Est: 6-8h

**Description:**
Implement `GameRunner` extending ATP Runner with game loop orchestration, and ATP protocol mapping for observations/actions.

**Checklist:**
- [ ] Initialize `atp-games/` package with `pyproject.toml` (depends on `atp-platform`, `game-environments`)
- [ ] Implement `ObservationMapper.to_atp_request()` — `GameObservation` → `ATPRequest`
- [ ] Implement `ActionMapper.from_atp_response()` — `ATPResponse` → `GameAction`
- [ ] Implement `GameRunner(BaseRunner)` with `run_game(game, agents, config) → GameResult`
- [ ] Implement game loop: reset → observe → map → send → validate → step → repeat
- [ ] Implement `_parallel_moves()` for simultaneous move games
- [ ] Implement `_sequential_moves()` for sequential move games
- [ ] Implement `ActionValidator` with retry logic (max 3 attempts + default)
- [ ] Implement `BuiltinAdapter` wrapping `Strategy` into ATP adapter interface
- [ ] Implement `GameResult`, `EpisodeResult` data models
- [ ] Write unit tests: mapper roundtrip, validator retry, runner with mock agents
- [ ] Write integration test: full PD game with 2 builtin strategies

**Traces to:** [AG-FR-001], [AG-FR-002], [AG-FR-003]
**Depends on:** [TASK-901], [TASK-909]
**Blocks:** [TASK-914], [TASK-915..917], [TASK-918]

---

### TASK-913: Multi-Episode & Concurrency
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Implement multi-episode execution with statistical aggregation and parallel episode support.

**Checklist:**
- [ ] Implement episode-level parallelism (configurable `--parallel=N`)
- [ ] Implement seed management: each episode gets deterministic seed `base_seed + episode`
- [ ] Implement aggregation: mean payoff, std, 95% CI per player per metric
- [ ] Implement Welch's t-test for comparing agents (reuse ATP statistical module)
- [ ] Implement Bonferroni correction for multiple comparisons
- [ ] Implement progress reporting (episode N/M, ETA)
- [ ] Write tests: 50 episodes aggregation, CI coverage, parallel vs sequential same results (deterministic seeds)

**Traces to:** [AG-FR-001], [AG-NFR-002], [AG-NFR-003]
**Depends on:** [TASK-912]
**Blocks:** [TASK-918]

---

### TASK-914: Game Suite YAML Loader
🔴 P0 | ⬜ TODO | Est: 4-5h

**Description:**
Implement YAML parser for game suite definitions and CLI integration.

**Checklist:**
- [ ] Define JSON Schema for `game_suite` YAML format
- [ ] Implement `GameSuiteLoader` — parse YAML, validate against schema
- [ ] Resolve `game.type` via `GameRegistry`, apply `game.config`
- [ ] Resolve `agents` — create ATP adapters or wrap builtin strategies
- [ ] Resolve `evaluation` — instantiate evaluators with weights/thresholds
- [ ] Support variable substitution (`${AGENT_ENDPOINT}`) for CI
- [ ] Support suite inheritance (`extends: base_pd.yaml`)
- [ ] Register loader in ATP plugin system (`type: game_suite`)
- [ ] Implement CLI: `atp test --suite=game:prisoners_dilemma.yaml`
- [ ] Create 3 builtin suite YAMLs:
  - [ ] `prisoners_dilemma.yaml` — basic repeated PD evaluation
  - [ ] `auction_battery.yaml` — first-price + second-price with baselines
  - [ ] `alympics_lite.yaml` — battery of all 5 games
- [ ] Write tests: YAML parsing, validation errors, variable substitution

**Traces to:** [AG-FR-004]
**Depends on:** [TASK-902], [TASK-912]
**Blocks:** [TASK-918]

---

### TASK-915: PayoffEvaluator & ExploitabilityEvaluator
🔴 P0 | ⬜ TODO | Est: 4-5h

**Description:**
Implement game-theoretic evaluators for payoff analysis and exploitability measurement.

**Checklist:**
- [ ] Implement `PayoffEvaluator`:
  - [ ] Average payoff per player per episode
  - [ ] Payoff distribution (min, max, percentiles)
  - [ ] Pareto efficiency check (is outcome Pareto optimal?)
  - [ ] Social welfare (sum of payoffs)
  - [ ] Configurable thresholds and weights
- [ ] Implement `ExploitabilityEvaluator`:
  - [ ] Extract empirical strategy from game history
  - [ ] Compute best response using `game_envs.analysis.exploitability`
  - [ ] Report exploitability score per player and aggregate
  - [ ] Configurable epsilon threshold for "pass" (default: 0.15)
- [ ] Register both in ATP evaluator registry
- [ ] Output compatible with ATP scoring system
- [ ] Write tests: payoff evaluator on known outcomes, exploitability = 0 for NE play, > 0 for AllC

**Traces to:** [AG-FR-005]
**Depends on:** [TASK-910], [TASK-912]
**Blocks:** [TASK-920]

---

### TASK-916: CooperationEvaluator & EquilibriumEvaluator
🟠 P1 | ⬜ TODO | Est: 3-4h

**Description:**
Implement evaluators for cooperation dynamics and equilibrium convergence analysis.

**Checklist:**
- [ ] Implement `CooperationEvaluator`:
  - [ ] Cooperation rate (overall and per-round trend)
  - [ ] Conditional cooperation P(C|C), P(C|D)
  - [ ] Reciprocity index
  - [ ] Cooperation stability (variance over episodes)
  - [ ] Thresholds: min cooperation rate, max defection streak
- [ ] Implement `EquilibriumEvaluator`:
  - [ ] Classify convergence type: Nash, correlated, cyclic, no convergence
  - [ ] Convergence speed (rounds to stabilize)
  - [ ] Distance to known equilibria
  - [ ] Strategy entropy over time (decreasing → converging)
- [ ] Register both in ATP evaluator registry
- [ ] Write tests: TFT vs TFT → cooperation ≈ 1.0, AllD vs AllC → cooperation = 0.5, convergence detection

**Traces to:** [AG-FR-005]
**Depends on:** [TASK-911], [TASK-912]
**Blocks:** [TASK-920]

---

### TASK-917: FairnessEvaluator (FAIRGAME)
🟡 P2 | ⬜ TODO | Est: 4-5h

**Description:**
Implement bias detection evaluator inspired by FAIRGAME methodology.

**Checklist:**
- [ ] Implement `FairnessEvaluator`:
  - [ ] Gini coefficient of payoff distribution across agents
  - [ ] Envy-freeness check for allocation games
  - [ ] Proportionality check
- [ ] Implement FAIRGAME-style bias detection:
  - [ ] Demographic variation: same game but opponent descriptions vary (gender, ethnicity, etc.)
  - [ ] Strategy shift measurement: does agent change behavior based on irrelevant opponent attributes?
  - [ ] Discrimination score: max difference in cooperation/aggression across groups
  - [ ] Statistical significance test (chi-squared / Fisher's exact)
- [ ] Bias report generation: which attributes cause behavioral shifts, magnitude, p-values
- [ ] Register in ATP evaluator registry
- [ ] Write tests: no bias when opponent description doesn't vary, synthetic bias injection detected

**Traces to:** [AG-FR-009]
**Depends on:** [TASK-911], [TASK-912]
**Blocks:** [TASK-920]

---

## Milestone 12: Advanced Scenarios & Reporting

### TASK-918: Tournament, Cross-Play & Stress-Test
🟠 P1 | ⬜ TODO | Est: 6-8h

**Description:**
Implement tournament mode, cross-play matrix, and adversarial stress-test.

**Checklist:**
- [ ] Tournament — Round-Robin:
  - [ ] For each (agent_i, agent_j) pair: run N episodes
  - [ ] Compute standings: win/loss/draw, total payoff, average metrics
  - [ ] Handle byes for odd number of agents
- [ ] Tournament — Elimination:
  - [ ] Single elimination bracket
  - [ ] Double elimination bracket
  - [ ] Seeding by round-robin results
- [ ] Cross-Play Matrix:
  - [ ] Run every agent vs every agent (including self-play)
  - [ ] Output: payoff heatmap, dominance relationships
  - [ ] Pareto frontier identification
  - [ ] Cluster analysis (which agents play similarly?)
- [ ] Adversarial Stress-Test:
  - [ ] Compute empirical strategy of agent under test
  - [ ] Generate best-response oracle
  - [ ] Run agent vs best-response, measure exploitability in practice
  - [ ] Optional: iterative (recompute BR after agent adapts)
- [ ] CLI integration:
  - [ ] `atp tournament --suite=tournament.yaml`
  - [ ] `atp crossplay --suite=crossplay.yaml`
- [ ] Write tests: 4-agent round-robin, elimination bracket correctness, cross-play matrix symmetry

**Traces to:** [AG-FR-006], [AG-FR-007], [AG-FR-008]
**Depends on:** [TASK-912], [TASK-913], [TASK-914]
**Blocks:** [TASK-920]

---

### TASK-919: Alympics-Style Benchmark Suite
🟡 P2 | ⬜ TODO | Est: 3-4h

**Description:**
Create standardized benchmark battery covering all 5 games, inspired by Alympics paper.

**Checklist:**
- [ ] Design benchmark suite: all 5 games, standardized configs, fixed episodes
- [ ] Define composite score: weighted aggregate across games
- [ ] Categories: strategic reasoning, cooperation, fairness, robustness
- [ ] Implement `alympics_lite.yaml` suite with all games + baselines
- [ ] Implement `atp benchmark --suite=alympics` shortcut command
- [ ] Create scoring rubric documentation
- [ ] Write integration test: run full benchmark with builtin strategies, verify scores
- [ ] Example output: agent X scored 72/100 (strategic: 85, cooperation: 60, fairness: 78, robustness: 65)

**Traces to:** [AG-FR-006]
**Depends on:** [TASK-914], [TASK-915..917]
**Blocks:** [TASK-920]

---

### TASK-920: Dashboard & Reporting Integration
🟠 P1 | ⏸️ BLOCKED | Est: 5-7h

**Description:**
Integrate game results into ATP Dashboard and reporting pipeline.

**Checklist:**
- [ ] Game Reporter (extends ATP reporter):
  - [ ] JSON output with game-specific fields (payoff matrix, strategy profiles)
  - [ ] HTML report with game visualizations
- [ ] Dashboard routes:
  - [ ] `/games/` — list of game evaluation results
  - [ ] `/games/{id}` — detailed result with all metrics
  - [ ] `/tournaments/{id}` — tournament standings and bracket
  - [ ] `/crossplay/{id}` — cross-play heatmap
- [ ] Visualizations:
  - [ ] Payoff matrix table (color-coded)
  - [ ] Strategy distribution timeline (line chart over rounds)
  - [ ] Cooperation dynamics chart
  - [ ] Cross-play heatmap (Chart.js or D3-inline)
  - [ ] Tournament bracket (SVG)
- [ ] Export: CSV + JSON for Jupyter analysis
- [ ] Navigation: add "Games" section to dashboard sidebar
- [ ] Write tests: reporter output format, dashboard routes return 200

**Traces to:** [AG-FR-010]
**Depends on:** [TASK-915..918]
**Blocks:** —

---

### TASK-921: Documentation & Examples
🟠 P1 | ⏸️ BLOCKED | Est: 4-5h

**Description:**
Complete documentation for game-environments and atp-games.

**Checklist:**
- [ ] `game-environments` docs:
  - [ ] API reference (auto-generated from docstrings)
  - [ ] Game development guide: how to add a new game
  - [ ] Strategy development guide
  - [ ] Analysis tools guide (Nash solver, exploitability, population)
- [ ] `atp-games` docs:
  - [ ] Installation and quick start
  - [ ] Game suite YAML reference
  - [ ] Evaluator configuration guide
  - [ ] Tournament setup guide
- [ ] Examples:
  - [ ] `basic_usage.ipynb` — run PD in Jupyter, plot results
  - [ ] `custom_game.ipynb` — create new game from scratch
  - [ ] `llm_agent_eval.ipynb` — evaluate LLM agent on game battery
  - [ ] `population_dynamics.ipynb` — evolutionary simulation
- [ ] Update ATP main README with Phase 5 section
- [ ] Update `docs/05-evaluators.md` with game-theoretic evaluators

**Traces to:** All Phase 5 requirements
**Depends on:** [TASK-901..920]
**Blocks:** —

---

### TASK-922: Package Publishing & CI
🟠 P1 | ⏸️ BLOCKED | Est: 3-4h

**Description:**
Set up CI/CD and publish both packages.

**Checklist:**
- [ ] `game-environments`:
  - [ ] GitHub repo setup
  - [ ] CI: pytest + coverage + linting (ruff)
  - [ ] Coverage gate: ≥ 90%
  - [ ] PyPI publishing workflow
  - [ ] Version: 0.1.0
- [ ] `atp-games`:
  - [ ] GitHub repo (or monorepo with ATP)
  - [ ] CI: pytest + coverage + linting
  - [ ] Coverage gate: ≥ 80%
  - [ ] PyPI publishing workflow
  - [ ] Version: 0.1.0
- [ ] Integration CI: test atp-games with latest game-environments
- [ ] Dependabot / Renovate for dependency updates

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
