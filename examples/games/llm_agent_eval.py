"""LLM agent evaluation on game battery.

Demonstrates how to evaluate an LLM agent (or mock agent) across
multiple games using the atp-games framework. Uses BuiltinAdapter
to simulate agents without requiring API keys.

Usage:
    cd atp-games
    uv run python ../examples/games/llm_agent_eval.py
"""

from __future__ import annotations

import asyncio

from atp_games import (
    BuiltinAdapter,
    GameRunConfig,
    GameRunner,
)
from atp_games.evaluators.cooperation_evaluator import (
    CooperationEvaluator,
)
from atp_games.evaluators.payoff_evaluator import (
    PayoffConfig,
    PayoffEvaluator,
)
from game_envs import (
    AlwaysDefect,
    PDConfig,
    PrisonersDilemma,
    TitForTat,
)


async def evaluate_strategy_as_agent(
    strategy_name: str,
    strategy_adapter: BuiltinAdapter,
    baseline_adapter: BuiltinAdapter,
    baseline_name: str,
) -> None:
    """Evaluate a strategy against a baseline."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {strategy_name} vs {baseline_name}")
    print("=" * 60)

    game = PrisonersDilemma(PDConfig(num_rounds=50))
    agents = {
        "player_0": strategy_adapter,
        "player_1": baseline_adapter,
    }

    runner = GameRunner()
    result = await runner.run_game(
        game=game,
        agents=agents,
        config=GameRunConfig(episodes=20, base_seed=42),
    )

    # Print statistics
    print(f"\nEpisodes: {result.num_episodes}")
    print(f"Average payoffs: {result.average_payoffs}")

    for stat in result.player_statistics():
        pid_name = strategy_name if stat.player_id == "player_0" else baseline_name
        print(
            f"  {pid_name} ({stat.player_id}): "
            f"mean={stat.mean:.2f}, std={stat.std:.2f}, "
            f"95% CI=[{stat.ci_lower:.2f}, {stat.ci_upper:.2f}]"
        )

    # Run evaluators
    print("\nPayoff evaluation:")
    payoff_eval = PayoffEvaluator(
        PayoffConfig(
            min_payoff={"player_0": 50.0},
            min_social_welfare=100.0,
        )
    )
    payoff_result = payoff_eval.evaluate_game(result)
    for check in payoff_result.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: {check.message}")

    print("\nCooperation evaluation:")
    coop_eval = CooperationEvaluator()
    coop_result = coop_eval.evaluate_game(result)
    for check in coop_result.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: {check.message}")
        if check.details:
            for key, val in check.details.items():
                if isinstance(val, dict):
                    for k2, v2 in val.items():
                        if isinstance(v2, float):
                            print(f"    {k2}: {v2:.3f}")
                        else:
                            print(f"    {k2}: {v2}")

    # Agent comparison (Welch's t-test)
    comparisons = result.agent_comparisons()
    if comparisons:
        print("\nStatistical comparison:")
        for cmp in comparisons:
            sig = "SIGNIFICANT" if cmp.is_significant else "not significant"
            print(
                f"  {cmp.player_a} vs {cmp.player_b}: "
                f"diff={cmp.mean_a - cmp.mean_b:.2f}, "
                f"p={cmp.p_value:.4f} ({sig})"
            )


async def run_game_battery() -> None:
    """Run evaluation across multiple game configurations."""
    print("\n" + "#" * 60)
    print("GAME BATTERY: Testing TitForTat across opponents")
    print("#" * 60)

    tft = BuiltinAdapter(TitForTat())

    opponents = {
        "AlwaysDefect": BuiltinAdapter(AlwaysDefect()),
        "TitForTat (self-play)": BuiltinAdapter(TitForTat()),
    }

    for opp_name, opp_adapter in opponents.items():
        await evaluate_strategy_as_agent("TitForTat", tft, opp_adapter, opp_name)


async def main() -> None:
    """Run all evaluations."""
    await run_game_battery()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
