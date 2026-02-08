"""Basic game-environments usage examples.

Demonstrates how to run game-theoretic environments, use built-in
strategies, and analyze results. No API keys required.

Usage:
    cd game-environments
    uv run python ../examples/games/basic_usage.py
"""

from __future__ import annotations

from game_envs import (
    AlwaysCooperate,
    AlwaysDefect,
    GameRegistry,
    GrimTrigger,
    Pavlov,
    PDConfig,
    PrisonersDilemma,
    RandomStrategy,
    TitForTat,
)


def run_one_shot_pd() -> None:
    """Run a single round of Prisoner's Dilemma."""
    print("=" * 60)
    print("ONE-SHOT PRISONER'S DILEMMA")
    print("=" * 60)

    game = PrisonersDilemma(PDConfig(num_rounds=1))
    result = game.reset()

    # Show what each player sees
    obs = result.observations["player_0"]
    print("\nPlayer 0 observation:")
    print(obs.to_prompt())

    # Play the round
    result = game.step(
        {
            "player_0": "cooperate",
            "player_1": "defect",
        }
    )

    print(f"\nOutcome: {result.payoffs}")
    print(f"Terminal: {result.is_terminal}")
    print(f"Final payoffs: {game.get_payoffs()}")


def run_repeated_pd() -> None:
    """Run a repeated Prisoner's Dilemma with strategies."""
    print("\n" + "=" * 60)
    print("REPEATED PD: Tit-for-Tat vs Always Defect (10 rounds)")
    print("=" * 60)

    game = PrisonersDilemma(PDConfig(num_rounds=10, seed=42))
    tft = TitForTat()
    alld = AlwaysDefect()

    result = game.reset()
    round_num = 0

    while not result.is_terminal:
        obs_0 = result.observations["player_0"]
        obs_1 = result.observations["player_1"]

        action_0 = tft.choose_action(obs_0)
        action_1 = alld.choose_action(obs_1)

        result = game.step(
            {
                "player_0": action_0,
                "player_1": action_1,
            }
        )
        round_num += 1
        print(
            f"  Round {round_num}: "
            f"TFT={action_0}, AllD={action_1} "
            f"-> payoffs={result.payoffs}"
        )

    print(f"\nFinal payoffs: {game.get_payoffs()}")
    print("TFT cooperates round 1, then mirrors -> defects from round 2")


def strategy_tournament() -> None:
    """Round-robin tournament between PD strategies."""
    print("\n" + "=" * 60)
    print("STRATEGY TOURNAMENT (20 rounds each)")
    print("=" * 60)

    strategies = {
        "TitForTat": TitForTat,
        "AlwaysCooperate": AlwaysCooperate,
        "AlwaysDefect": AlwaysDefect,
        "GrimTrigger": GrimTrigger,
        "Pavlov": Pavlov,
        "Random": RandomStrategy,
    }

    totals: dict[str, float] = {name: 0.0 for name in strategies}
    names = list(strategies.keys())

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            game = PrisonersDilemma(PDConfig(num_rounds=20, seed=42))
            s_a = strategies[name_a]()
            s_b = strategies[name_b]()

            result = game.reset()
            while not result.is_terminal:
                obs_a = result.observations["player_0"]
                obs_b = result.observations["player_1"]
                result = game.step(
                    {
                        "player_0": s_a.choose_action(obs_a),
                        "player_1": s_b.choose_action(obs_b),
                    }
                )

            payoffs = game.get_payoffs()
            totals[name_a] += payoffs["player_0"]
            totals[name_b] += payoffs["player_1"]
            print(
                f"  {name_a} vs {name_b}: "
                f"{payoffs['player_0']:.0f} - {payoffs['player_1']:.0f}"
            )

    print("\nFinal standings:")
    for name, total in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {total:.0f}")


def use_game_registry() -> None:
    """Create games via the registry."""
    print("\n" + "=" * 60)
    print("GAME REGISTRY")
    print("=" * 60)

    print(f"\nAvailable games: {GameRegistry.list_games()}")

    # Create via registry
    game = GameRegistry.create(
        "prisoners_dilemma",
        {
            "num_rounds": 5,
            "noise": 0.1,
        },
    )
    print(f"\nCreated: {game.name}")
    print(f"Type: {game.game_type}")
    print(f"Move order: {game.move_order}")
    print(f"Players: {game.player_ids}")
    print(f"Action space: {game.action_space('player_0').to_description()}")


def noisy_game() -> None:
    """Demonstrate trembling-hand noise."""
    print("\n" + "=" * 60)
    print("NOISY PD (10% action flip probability)")
    print("=" * 60)

    game = PrisonersDilemma(
        PDConfig(
            num_rounds=20,
            noise=0.1,
            seed=42,
        )
    )
    allc = AlwaysCooperate()

    result = game.reset()

    while not result.is_terminal:
        obs = result.observations["player_0"]
        intended = allc.choose_action(obs)
        result = game.step(
            {
                "player_0": intended,
                "player_1": intended,
            }
        )

    print(f"Payoffs with noise: {game.get_payoffs()}")
    print("With 10% noise, ~2 of 20 actions may flip randomly")


if __name__ == "__main__":
    run_one_shot_pd()
    run_repeated_pd()
    strategy_tournament()
    use_game_registry()
    noisy_game()
    print("\nDone!")
