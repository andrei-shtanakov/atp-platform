# Game-Theoretic Evaluation Examples

Example scripts demonstrating game-theoretic evaluation of AI agents using `game-environments` and `atp-games`.

**No API keys required** -- all examples use built-in strategies as stand-in agents.

## Examples

### basic_usage.py

Introductory examples for the `game-environments` library:

- One-shot Prisoner's Dilemma with manual actions
- Repeated PD with Tit-for-Tat vs Always Defect
- Round-robin strategy tournament (6 strategies)
- Game registry usage (creating games by name)
- Trembling-hand noise demonstration

```bash
cd game-environments
uv run python ../examples/games/basic_usage.py
```

### custom_game.py

Create a new game from scratch by implementing the `Game` ABC:

- Rock-Paper-Scissors game implementation
- Custom strategies (AlwaysRock, CounterStrategy, FrequencyCounter)
- Running matches and inspecting game history

```bash
cd game-environments
uv run python ../examples/games/custom_game.py
```

### llm_agent_eval.py

Evaluate agents using the `atp-games` evaluation framework:

- Running a game battery (TitForTat against multiple opponents)
- Using `PayoffEvaluator` and `CooperationEvaluator`
- Statistical comparison with Welch's t-test
- Multi-episode evaluation with confidence intervals

```bash
cd atp-games
uv run python ../examples/games/llm_agent_eval.py
```

### population_dynamics.py

Evolutionary simulation with population dynamics:

- Replicator dynamics (continuous-time evolution)
- ESS (evolutionarily stable strategy) checks
- Moran process (stochastic finite-population evolution)
- Full population simulation with mutation

```bash
cd game-environments
uv run python ../examples/games/population_dynamics.py
```

## Prerequisites

Install the relevant package before running examples:

```bash
# For basic_usage.py, custom_game.py, population_dynamics.py
cd game-environments && uv sync

# For llm_agent_eval.py
cd atp-games && uv sync
```

## Related Documentation

- [game-environments README](../../game-environments/README.md) -- core library API and game development guide
- [atp-games README](../../atp-games/README.md) -- ATP plugin, YAML suites, evaluators, tournaments
- [Evaluation System](../../docs/05-evaluators.md) -- full evaluator reference including game-theoretic evaluators
