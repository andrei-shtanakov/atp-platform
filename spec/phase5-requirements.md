# Phase 5: Game-Theoretic Evaluation — Requirements

> Functional and Non-Functional Requirements for Game-Theoretic Agent Evaluation
> Per ADR-002: Two-package architecture (game-environments + atp-games)

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | Draft |
| Created | 2026-02-07 |
| Related | ADR-002, Phase 5 Roadmap |

---

## Scope

### In Scope

- Standalone game environments library (`game-environments`)
- ATP plugin for game-theoretic evaluation (`atp-games`)
- 5 canonical games with baseline strategies
- Game-theoretic metrics and evaluators
- Multi-agent game loop orchestration
- Tournament and cross-play modes
- Dashboard integration for game results

### Out of Scope

- Agent training / MARL training loop (это отдельный проект)
- Real-time multiplayer (все ходы через ATP protocol)
- GUI для создания игр (только YAML/Python API)
- Blockchain-based verification / smart contracts
- Integration с PettingZoo / OpenSpiel напрямую (свой lightweight interface)

---

## 1. Functional Requirements: game-environments

### GE-FR-001: Game Abstraction

| Aspect | Detail |
|--------|--------|
| Description | Единый интерфейс `Game` для всех типов игр: normal-form, extensive-form, repeated, stochastic |
| Interface | `reset() → GameState`, `step(actions) → GameState`, `get_payoffs() → dict`, `is_terminal → bool` |
| Acceptance | Все 5 канонических игр реализованы через один интерфейс; интерфейс не содержит ATP-зависимостей |

**Traces to:** TASK-901, TASK-902

---

### GE-FR-002: Action Spaces

| Aspect | Detail |
|--------|--------|
| Description | Поддержка дискретных, непрерывных и структурированных action spaces |
| Types | `DiscreteActionSpace(n)`, `ContinuousActionSpace(low, high)`, `StructuredActionSpace(schema)` |
| Validation | `action_space.contains(action) → bool`, валидация до выполнения step |
| Acceptance | Аукцион использует continuous (bid amount), PD использует discrete (C/D), Blotto использует structured (allocation vector) |

**Traces to:** TASK-901

---

### GE-FR-003: Game State & Observability

| Aspect | Detail |
|--------|--------|
| Description | Модели состояния игры с поддержкой полной и частичной наблюдаемости |
| Models | `GameState`, `Observation` (per-player view), `InformationSet` |
| Features | Скрытие приватной информации (карты в покере, ставки в sealed-bid auction), history filtering |
| Acceptance | В sealed-bid auction агент не видит ставки соперников до раунда reveal |

**Traces to:** TASK-901, TASK-903

---

### GE-FR-004: Communication Channel

| Aspect | Detail |
|--------|--------|
| Description | Опциональный канал для обмена сообщениями между агентами (cheap talk, negotiation) |
| Modes | `no_communication`, `pre_action` (сообщения до хода), `post_action`, `free` (любой момент) |
| Format | Text-based (для LLM-агентов), structured (для coded agents) |
| Acceptance | В repeated PD с communication agents могут обмениваться сообщениями перед каждым раундом |

**Traces to:** TASK-903

---

### GE-FR-005: Canonical Games Library

| Aspect | Detail |
|--------|--------|
| Description | 5 канонических игр из теории игр с полной реализацией |
| Games | Prisoner's Dilemma, Public Goods Game, First/Second-Price Auction, Colonel Blotto, Congestion Game |
| Variants | Каждая игра: one-shot + repeated; с/без noise; с/без communication |
| Parametrization | Payoff matrix customizable, число игроков configurable (где применимо) |
| Acceptance | Каждая игра имеет ≥3 baseline strategy, payoff verification через аналитические решения |

**Traces to:** TASK-904, TASK-905, TASK-906, TASK-907, TASK-908

---

### GE-FR-006: Baseline Strategies

| Aspect | Detail |
|--------|--------|
| Description | Встроенные стратегии для каждой игры, используемые как baselines и для exploitability computation |
| Per Game | PD: TFT, AllC, AllD, Grim, Pavlov, Random; PG: Full, Free, Conditional, Punisher; Auction: Truthful, Shade-50%, Random; Blotto: Uniform, Concentrated, Nash-mixed; Congestion: Selfish, Social, ε-greedy |
| Interface | `Strategy(ABC)` с `choose_action(observation) → action` |
| Acceptance | Baseline strategies воспроизводят известные теоретические результаты (AllD dominates one-shot PD, Truthful optimal in second-price) |

**Traces to:** TASK-909

---

### GE-FR-007: Nash Equilibrium Solver

| Aspect | Detail |
|--------|--------|
| Description | Вычисление равновесия Nash для normal-form games |
| Methods | Support / Lemke-Howson для 2-player, fictitious play / replicator dynamics для n-player |
| Output | Mixed strategy profiles, expected payoffs, support sets |
| Limitations | Approximate для >2 player general-sum; exact для 2-player zero-sum |
| Acceptance | Solver находит все Nash equilibria для 2x2 games; для larger games convergence within ε=0.01 |

**Traces to:** TASK-910

---

### GE-FR-008: Exploitability Calculator

| Aspect | Detail |
|--------|--------|
| Description | Вычисление exploitability стратегии — насколько она уязвима к best response |
| Method | Для каждого игрока: compute best response to opponent's empirical strategy, measure payoff gap |
| Output | Exploitability score per player, aggregate exploitability |
| Acceptance | Для Nash strategy exploitability ≈ 0; для AllC в one-shot PD exploitability = max |

**Traces to:** TASK-910

---

### GE-FR-009: Cooperation & Fairness Analysis

| Aspect | Detail |
|--------|--------|
| Description | Метрики кооперации и справедливости |
| Cooperation | Cooperation rate, conditional cooperation (P(C|C) vs P(C|D)), reciprocity index |
| Fairness | Envy-freeness, proportionality, Gini coefficient of payoffs, utilitarian welfare |
| Evolutionary | Strategy frequency over time, evolutionary stability, invasion resistance |
| Acceptance | TFT в repeated PD показывает conditional cooperation > 0.9; free rider detection в Public Goods |

**Traces to:** TASK-911

---

### GE-FR-010: Population Dynamics

| Aspect | Detail |
|--------|--------|
| Description | Симуляция эволюционной динамики в популяции агентов |
| Methods | Replicator dynamics, Moran process, evolutionary stable strategy (ESS) check |
| Features | Population of heterogeneous strategies, fitness-proportional selection, mutation |
| Output | Strategy distribution over generations, convergence point, ESS classification |
| Acceptance | TFT доминирует в population-based repeated PD с noise < 0.1 |

**Traces to:** TASK-911

---

## 2. Functional Requirements: atp-games Plugin

### AG-FR-001: Game Runner

| Aspect | Detail |
|--------|--------|
| Description | Расширение ATP Runner для orchestration multi-agent game loops |
| Modes | `single_game` (one game instance), `tournament` (round-robin), `population` (evolutionary), `cross_play` (all vs all) |
| Game Loop | Init game → for each round: get observations → send to agents (ATP Request) → collect actions (ATP Response) → step game → repeat until terminal |
| Concurrency | Simultaneous moves: parallel requests to agents; Sequential: ordered |
| Acceptance | Repeated PD с 2 LLM-агентами через HTTP adapter runs to completion |

**Traces to:** TASK-912, TASK-913

---

### AG-FR-002: ATP Protocol Mapping

| Aspect | Detail |
|--------|--------|
| Description | Маппинг игровых данных на ATP Protocol |
| Request | `GameObservation` → `ATPRequest.context` с полями: `game_state`, `available_actions`, `history`, `message` |
| Response | `ATPResponse.artifacts` → `GameAction` с полями: `action`, `message`, `reasoning` |
| Metadata | Game type, round number, total rounds → `ATPRequest.metadata` |
| Acceptance | Любой существующий ATP adapter (HTTP, Docker, CLI) работает с game requests без модификации |

**Traces to:** TASK-912

---

### AG-FR-003: Action Validation & Retry

| Aspect | Detail |
|--------|--------|
| Description | Валидация действий агентов и retry для LLM-агентов |
| Validation | Action checked against `game.action_space`; illegal actions rejected |
| Retry | LLM agents get retry with error message (max 3 attempts), then default action |
| Default | Configurable per game: random legal action, or specific safe action |
| Logging | All validation failures and retries logged for analysis |
| Acceptance | LLM agent submitting invalid JSON gets retry prompt and succeeds; after 3 failures, default action applied |

**Traces to:** TASK-912

---

### AG-FR-004: Game Suite YAML

| Aspect | Detail |
|--------|--------|
| Description | Расширение ATP YAML формата для описания game test suites |
| Format | `type: game_suite`, `game:` (type, variant, config), `agents:` (list with adapters), `evaluation:` (metrics, thresholds) |
| Features | Built-in strategy reference (`adapter: builtin, strategy: tit_for_tat`), parametric sweeps, inheritance from base suites |
| Acceptance | `atp test --suite=game:prisoners_dilemma.yaml` loads and runs game suite |

**Traces to:** TASK-914

---

### AG-FR-005: Game-Theoretic Evaluators

| Aspect | Detail |
|--------|--------|
| Description | Набор evaluators для оценки поведения агентов в играх |
| Evaluators | `PayoffEvaluator`, `ExploitabilityEvaluator`, `CooperationEvaluator`, `FairnessEvaluator`, `EquilibriumEvaluator` |
| Integration | Регистрируются в ATP evaluator registry; используют scoring system (weights, thresholds) |
| Output | Scores compatible с ATP reporting pipeline; composite game score |
| Acceptance | Evaluator results appear in standard ATP JSON/HTML reports |

**Traces to:** TASK-915, TASK-916, TASK-917

---

### AG-FR-006: Tournament Mode

| Aspect | Detail |
|--------|--------|
| Description | Round-robin и elimination tournaments между агентами |
| Round-Robin | Каждая пара агентов играет N эпизодов; агрегация payoff/metrics |
| Elimination | Single/double elimination bracket |
| Output | Tournament standings, win/loss/draw matrix, per-matchup detailed stats |
| Acceptance | 4 agents in round-robin PD tournament; standings reflect theoretical predictions (TFT > AllD > AllC in repeated) |

**Traces to:** TASK-918

---

### AG-FR-007: Cross-Play Matrix

| Aspect | Detail |
|--------|--------|
| Description | Матрица результатов всех агентов против всех |
| Computation | For each (agent_i, agent_j) pair: run N episodes, compute average payoffs |
| Analysis | Dominance relationships, Pareto frontier, cluster analysis |
| Visualization | Heatmap of payoffs, dominance graph |
| Acceptance | Cross-play matrix for 5 agents shows expected dominance patterns |

**Traces to:** TASK-918

---

### AG-FR-008: Adversarial Stress-Test

| Aspect | Detail |
|--------|--------|
| Description | Тестирование агента против best-response oracle |
| Method | Compute empirical strategy of agent → compute best response → play agent vs best response → measure exploitability |
| Iterative | Optional: iterative best response (agent learns, recompute BR) |
| Acceptance | Agent with low exploitability (<0.15) passes stress test; agent with obvious exploit fails |

**Traces to:** TASK-918

---

### AG-FR-009: Bias Detection (FAIRGAME)

| Aspect | Detail |
|--------|--------|
| Description | Выявление систематических смещений в стратегиях LLM-агентов |
| Methodology | Per FAIRGAME [arxiv:2504.14325]: vary demographics/descriptions of opponents, measure strategy shifts |
| Metrics | Discrimination score, aggression index, rule-following rate, cooperation variance across groups |
| Output | Bias report with statistical significance tests |
| Acceptance | Detect if agent cooperates differently based on opponent description |

**Traces to:** TASK-919

---

### AG-FR-010: Dashboard Integration

| Aspect | Detail |
|--------|--------|
| Description | Визуализация game results в ATP Dashboard |
| Views | Payoff matrix view, strategy distribution timeline, cooperation dynamics, cross-play heatmap, tournament leaderboard |
| Export | CSV, JSON для Jupyter; PNG для reports |
| Acceptance | Game test results accessible through existing dashboard URL with new navigation section |

**Traces to:** TASK-920

---

## 3. Non-Functional Requirements

### GE-NFR-001: Zero ATP Dependency

| Aspect | Requirement |
|--------|-------------|
| Description | `game-environments` не имеет зависимости от ATP |
| Enforcement | Separate package, separate pyproject.toml, no imports from atp.* |
| Dependencies | numpy (optional), no ML frameworks required |
| Acceptance | `pip install game-environments` works without atp-platform installed |

---

### GE-NFR-002: Performance

| Aspect | Requirement |
|--------|-------------|
| Game step | < 1ms for normal-form games |
| Nash solver (2-player) | < 1s for games up to 100 actions |
| Nash solver (n-player) | < 30s approximate for up to 10 players, 10 actions each |
| Exploitability | < 5s for empirical strategy over 1000 rounds |
| Population sim | < 10s for 100 agents, 1000 generations |

---

### GE-NFR-003: Test Coverage

| Aspect | Requirement |
|--------|-------------|
| Unit tests | ≥ 90% coverage for core and games modules |
| Analytical verification | Each game verified against known equilibria |
| Property-based | Hypothesis tests for invariants (payoffs sum, action space containment) |

---

### AG-NFR-001: Backward Compatibility

| Aspect | Requirement |
|--------|-------------|
| Existing tests | All existing ATP test suites work unchanged |
| Existing adapters | HTTP, Docker, CLI adapters work for game requests |
| CLI | New commands additive: `atp game`, `atp tournament` |
| Dashboard | New views don't break existing pages |

---

### AG-NFR-002: LLM Agent Latency Tolerance

| Aspect | Requirement |
|--------|-------------|
| Timeout | Per-move timeout configurable (default 30s for LLM agents, 1s for coded agents) |
| Parallelism | Simultaneous moves sent in parallel |
| Total game time | 100-round PD with 2 LLM agents < 30 minutes |
| Episode parallelism | Multiple episodes run in parallel (configurable --parallel) |

---

### AG-NFR-003: Statistical Validity

| Aspect | Requirement |
|--------|-------------|
| Min episodes | Default 50 per matchup for statistical significance |
| Confidence intervals | 95% CI reported for all metrics |
| Significance testing | Welch's t-test for agent comparison; Bonferroni correction for multiple comparisons |
| Reproducibility | Random seed support for deterministic replay |

---

## 4. Dependencies & Constraints

### Package Dependencies

| Package | Used By | Purpose |
|---------|---------|---------|
| numpy | game-environments | Payoff matrices, Nash solver numerics |
| scipy | game-environments (optional) | Optimization for Nash solver |
| pydantic | both | Data models, validation |
| atp-platform ≥ 4.0 | atp-games | Plugin host |

### Constraints

1. Python ≥ 3.11 (consistent with ATP)
2. `game-environments` must be usable standalone in Jupyter
3. All game rules must be deterministic given a seed
4. No GPU requirements

---

## 5. Acceptance Criteria Summary

### Phase 5 Exit Criteria

| Criterion | Metric |
|-----------|--------|
| Games implemented | 5 canonical + extensible |
| Baselines per game | ≥ 3 |
| Equilibria verified | Analytical for all 2-player games |
| ATP integration | Full CLI + dashboard |
| Test coverage | ≥ 90% game-environments, ≥ 80% atp-games |
| Documentation | API ref + game dev guide + eval methodology |
| Example | LLM agent evaluated on repeated PD, exploitability < 0.15 |
| Package published | Both packages on PyPI |

---

## Appendix: Traceability Matrix

| Requirement | Tasks | Milestone |
|-------------|-------|-----------|
| GE-FR-001 | TASK-901, TASK-902 | M9 |
| GE-FR-002 | TASK-901 | M9 |
| GE-FR-003 | TASK-901, TASK-903 | M9 |
| GE-FR-004 | TASK-903 | M9 |
| GE-FR-005 | TASK-904..908 | M9 |
| GE-FR-006 | TASK-909 | M9 |
| GE-FR-007 | TASK-910 | M10 |
| GE-FR-008 | TASK-910 | M10 |
| GE-FR-009 | TASK-911 | M10 |
| GE-FR-010 | TASK-911 | M10 |
| AG-FR-001 | TASK-912, TASK-913 | M11 |
| AG-FR-002 | TASK-912 | M11 |
| AG-FR-003 | TASK-912 | M11 |
| AG-FR-004 | TASK-914 | M11 |
| AG-FR-005 | TASK-915..917 | M11 |
| AG-FR-006 | TASK-918 | M12 |
| AG-FR-007 | TASK-918 | M12 |
| AG-FR-008 | TASK-918 | M12 |
| AG-FR-009 | TASK-919 | M12 |
| AG-FR-010 | TASK-920 | M12 |
