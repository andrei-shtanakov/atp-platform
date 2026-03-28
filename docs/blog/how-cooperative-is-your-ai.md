# How Cooperative Is Your AI? Game-Theoretic Evaluation of LLM Agents

When you deploy an AI agent, you test outputs. But do you test *how it decides*?

Most agent evaluation today is a one-shot affair: give the agent a task, check whether the result is correct, move on. That works fine if your agents live in isolation. But increasingly, agents operate alongside other agents -- negotiating, sharing resources, deciding whether to cooperate or defect. In those settings, *strategy* matters as much as accuracy. And traditional eval frameworks have nothing to say about it.

## The Problem with Input-Output Evaluation

Standard agent benchmarks measure: did the agent produce the right artifact? Did it call the right tools? Did the LLM-judge score it favorably?

None of that tells you whether your agent cooperates when cooperation is costly, coordinates under conflicting preferences, adapts based on opponent history, or is exploitable by a defector. These are behavioral properties that only emerge when agents interact. You can't observe them from a single task trace.

## Game Theory Meets AI Evaluation

Game theory studies strategic interaction -- how rational actors decide when outcomes depend on others' choices. The canonical scenarios are deliberately simple, which makes them powerful diagnostics. Cooperation games reveal whether an agent sacrifices individual gain for mutual benefit. Coordination games test whether agents align without explicit communication. Resource competition games show how agents behave under scarcity.

Running an AI agent through these games is a behavioral stress test. The games don't care what framework the agent was built with -- only how it behaves under strategic pressure.

## How ATP Approaches This

[ATP Platform](https://github.com/your-org/atp-platform) treats agent evaluation as a black-box protocol problem. An agent receives a structured observation and returns an action. The platform wraps agents in adapters, runs them through tournaments, tracks Elo ratings, and surfaces behavioral metrics.

The game-theoretic evaluation pipeline looks like this:

```
Agent A (adapter) ─┐
                   ├─► Game Environment ─► Round Results ─► Elo + Stats
Agent B (adapter) ─┘
```

Each agent is isolated -- it sees only the game state, not the opponent's internal logic. After repeated episodes, the platform aggregates payoffs, computes win/loss/draw standings, and updates Elo ratings using a standard chess-style formula.

The Elo system is particularly useful: it provides a relative measure of strategic strength that compounds across many matchups, making it easy to rank strategies and track improvement over time.

## The Seven Games

ATP's `game-environments` library ships seven games covering the major classes of strategic interaction:

| Game | Type | What It Tests |
|---|---|---|
| Prisoner's Dilemma | 2-player, repeated | Cooperation vs. defection; trust under temptation |
| Stag Hunt | 2-player, repeated | Coordination; willingness to take social risk |
| Battle of the Sexes | 2-player, repeated | Coordination under conflicting preferences |
| Public Goods Game | N-player, repeated | Group cooperation; free-rider detection |
| Sealed-Bid Auction | 2-player, one-shot/repeated | Truthful bidding; strategic value revelation |
| Colonel Blotto | 2-player, repeated | Resource allocation under adversarial conditions |
| Congestion Game | N-player, repeated | Routing under shared resource constraints |

Each game comes with a configurable environment, a set of builtin reference strategies, and evaluators that measure payoff, exploitability, and distance from Nash equilibrium.

## Experiment Results: Baseline Strategies

To ground the framework in real numbers, here are results from a 50-episode round-robin tournament across three core games (seed=42, reproducible):

### Prisoner's Dilemma

| Rank | Strategy | W | L | D | Total Payoff | Elo |
|---|---|---|---|---|---|---|
| 1 | always_defect | 2 | 0 | 0 | 10.0 | 1531 |
| 2 | always_cooperate | 0 | 1 | 1 | 3.0 | 1485 |
| 3 | tit_for_tat | 0 | 1 | 1 | 3.0 | 1484 |

**What this tells us:** In a small tournament with only three strategies, unconditional defection wins -- as theory predicts when there is no reciprocity pressure from adaptive opponents. Notably, Tit-for-Tat draws with Always Cooperate (both match each other move-for-move) but cannot overcome the payoff deficit from the defector matchup. In larger tournaments with more adaptive strategies, Tit-for-Tat typically recovers -- this illustrates why tournament composition matters.

### Stag Hunt

| Rank | Strategy | W | L | D | Total Payoff | Elo |
|---|---|---|---|---|---|---|
| 1 | always_hare | 2 | 0 | 0 | 6.0 | 1531 |
| 2 | always_stag | 0 | 1 | 1 | 4.0 | 1485 |
| 3 | stag_tit_for_tat | 0 | 1 | 1 | 4.0 | 1484 |

**What this tells us:** Stag Hunt has two Nash equilibria -- both cooperate (stag/stag) and both defect to safety (hare/hare). Always Hare wins here because it never risks the miscoordination penalty. But stag/stag yields higher joint payoff (4+4 vs 3+3). This makes Stag Hunt a useful test of an agent's *risk appetite* for high-value coordination.

### Battle of the Sexes

| Rank | Strategy | W | L | D | Total Payoff | Elo |
|---|---|---|---|---|---|---|
| 1 | always_a | 1 | 0 | 1 | 3.0 | 1516 |
| 2 | always_b | 0 | 0 | 2 | 0.0 | 1499 |
| 3 | alternating | 0 | 1 | 1 | 2.0 | 1485 |

**What this tells us:** Coordination under conflicting preferences is hard. Always B earns zero payoff here -- it never achieves coordination with Always A, and the Alternating strategy manages partial coordination. This game is a strong test of an agent's ability to signal and respond to asymmetric preferences.

These are deterministic baseline strategies. The interesting part -- running LLM agents through the same tournament and measuring whether GPT-4o behaves like Tit-for-Tat or Always Defect -- is exactly what the platform is built for.

## Try It Yourself

**Run the baseline experiment:**

```bash
git clone https://github.com/your-org/atp-platform
cd atp-platform
uv sync

# Run 50-episode tournament across 3 games
uv run python examples/experiments/run_experiment.py \
    --episodes 50 \
    --seed 42 \
    --output-dir results/my-experiment
```

**Evaluate two strategies head-to-head:**

```python
import asyncio
from game_envs.games.registry import GameRegistry
from game_envs.strategies.pd_strategies import TitForTat, AlwaysDefect
from atp_games.models import GameRunConfig
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.suites.tournament import run_round_robin
from atp_games.rating.elo import EloCalculator

import game_envs.games  # register games

async def main():
    game = GameRegistry.create("prisoners_dilemma")
    agents = {
        "tit_for_tat": BuiltinAdapter(TitForTat()),
        "always_defect": BuiltinAdapter(AlwaysDefect()),
    }
    result = await run_round_robin(
        game=game,
        agents=agents,
        config=GameRunConfig(episodes=100, base_seed=42),
        elo_calculator=EloCalculator(),
    )
    for s in result.standings:
        elo = result.elo_ratings[s.agent].rating
        print(f"{s.agent}: payoff={s.total_payoff:.1f}, elo={elo:.0f}")

asyncio.run(main())
```

## What's Next

The baseline results above use deterministic rule-based strategies. The next step is plugging in LLM agents and running the same tournaments -- asking whether a GPT-4o agent cooperates in Prisoner's Dilemma, coordinates in Stag Hunt, and adapts its strategy based on opponent history.

Early experiments with GPT-4o-mini (see `demo-game/`) show LLM agents can be wired into the same tournament infrastructure with minimal adapter code. The behavioral results are often surprising: LLMs tend to over-cooperate in early rounds, then shift strategy inconsistently. Measuring this formally, across games and models, is exactly the benchmark we are building.

If you work on multi-agent systems and want to contribute:

- Add a new game environment to `game-environments/`
- Implement a strategy for an existing game
- Wire an LLM or agent framework as an adapter
- Share results from your own agent evaluations

The platform is framework-agnostic by design. If your agent can read a JSON observation and return an action, it can play.

---

**Links**

- GitHub: `atp-platform` (monorepo with game environments, atp-games plugin, dashboard)
- Documentation: `docs/` -- architecture, protocol spec, quickstart guides
- PyPI: `atp-games` package (coming soon)

*Reproducible experiment data: seed=42, 50 episodes/match, 3 games, 9 strategies.*
