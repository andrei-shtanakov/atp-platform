"""Reproducible experiment runner for game-theoretic tournaments.

Runs round-robin tournaments for multiple games using builtin
strategies, tracks Elo ratings per game, and saves results as
JSON and CSV.

Usage::

    uv run python examples/experiments/run_experiment.py --episodes 10
    uv run python examples/experiments/run_experiment.py \\
        --episodes 100 --seed 42 --output-dir results/experiment
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Trigger game registry decorators before using GameRegistry.create()
import game_envs.games  # noqa: F401
from atp_games.models import GameRunConfig
from atp_games.rating.elo import EloCalculator
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.suites.tournament import TournamentResult, run_round_robin
from game_envs.games.registry import GameRegistry
from game_envs.strategies.bos_strategies import Alternating, AlwaysA, AlwaysB
from game_envs.strategies.pd_strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
)
from game_envs.strategies.stag_hunt_strategies import (
    AlwaysHare,
    AlwaysStag,
    StagTitForTat,
)


@dataclass
class GameSpec:
    """Specification for a single game tournament.

    Attributes:
        game_name: Registered game name in GameRegistry.
        agents: Mapping of strategy name to BuiltinAdapter.
    """

    game_name: str
    agents: dict[str, BuiltinAdapter]


def build_game_specs() -> list[GameSpec]:
    """Build the list of game specs with their strategies.

    Returns:
        List of GameSpec objects for PD, Stag Hunt, and BoS.
    """
    return [
        GameSpec(
            game_name="prisoners_dilemma",
            agents={
                "always_cooperate": BuiltinAdapter(AlwaysCooperate()),
                "always_defect": BuiltinAdapter(AlwaysDefect()),
                "tit_for_tat": BuiltinAdapter(TitForTat()),
            },
        ),
        GameSpec(
            game_name="stag_hunt",
            agents={
                "always_stag": BuiltinAdapter(AlwaysStag()),
                "always_hare": BuiltinAdapter(AlwaysHare()),
                "stag_tit_for_tat": BuiltinAdapter(StagTitForTat()),
            },
        ),
        GameSpec(
            game_name="battle_of_sexes",
            agents={
                "always_a": BuiltinAdapter(AlwaysA()),
                "always_b": BuiltinAdapter(AlwaysB()),
                "alternating": BuiltinAdapter(Alternating()),
            },
        ),
    ]


def print_standings(game_name: str, result: TournamentResult) -> None:
    """Print tournament standings to console.

    Args:
        game_name: Display name for the game.
        result: TournamentResult with standings and Elo data.
    """
    print(f"\n{'=' * 60}")
    print(f"  {game_name.upper().replace('_', ' ')}")
    print(f"{'=' * 60}")
    print(
        f"{'Rank':<5} {'Agent':<22} {'W':>3} {'L':>3} {'D':>3} "
        f"{'Pts':>6} {'Payoff':>8} {'Elo':>8}"
    )
    print("-" * 60)

    elo = result.elo_ratings or {}
    for rank, standing in enumerate(result.standings, 1):
        elo_val = elo.get(standing.agent)
        elo_str = f"{elo_val.rating:.1f}" if elo_val else "   N/A"
        print(
            f"{rank:<5} {standing.agent:<22} "
            f"{standing.wins:>3} {standing.losses:>3} {standing.draws:>3} "
            f"{standing.points:>6.0f} {standing.total_payoff:>8.3f} "
            f"{elo_str:>8}"
        )


def build_json_output(
    specs: list[GameSpec],
    results: list[TournamentResult],
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Build the full JSON output structure.

    Args:
        specs: Game specs used in the experiment.
        results: Tournament results, one per spec.
        episodes: Number of episodes per match.
        seed: Base seed used.

    Returns:
        Serializable dict with experiment metadata and results.
    """
    games_out: list[dict[str, Any]] = []
    for spec, result in zip(specs, results):
        games_out.append(
            {
                "game": spec.game_name,
                "tournament": result.to_dict(),
            }
        )
    return {
        "experiment": {
            "episodes_per_match": episodes,
            "seed": seed,
            "games": [s.game_name for s in specs],
        },
        "results": games_out,
    }


def build_csv_rows(
    specs: list[GameSpec],
    results: list[TournamentResult],
) -> list[dict[str, Any]]:
    """Build flat rows for CSV export.

    Args:
        specs: Game specs used.
        results: Tournament results.

    Returns:
        List of row dicts for csv.DictWriter.
    """
    rows: list[dict[str, Any]] = []
    for spec, result in zip(specs, results):
        elo = result.elo_ratings or {}
        for standing in result.standings:
            elo_val = elo.get(standing.agent)
            rows.append(
                {
                    "game": spec.game_name,
                    "agent": standing.agent,
                    "wins": standing.wins,
                    "losses": standing.losses,
                    "draws": standing.draws,
                    "points": standing.points,
                    "total_payoff": round(standing.total_payoff, 4),
                    "matches_played": standing.matches_played,
                    "elo_rating": round(elo_val.rating, 2) if elo_val else None,
                    "elo_games": elo_val.games_played if elo_val else None,
                }
            )
    return rows


async def run_experiment(
    episodes: int,
    seed: int,
    output_dir: Path,
) -> None:
    """Run the full experiment and save results.

    Args:
        episodes: Number of episodes per match.
        seed: Base seed for deterministic runs.
        output_dir: Directory to write JSON and CSV outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GameRunConfig(episodes=episodes, base_seed=seed)
    elo_calculator = EloCalculator()
    specs = build_game_specs()
    results: list[TournamentResult] = []

    for spec in specs:
        print(f"\nRunning tournament: {spec.game_name} ({episodes} episodes/match)...")
        game = GameRegistry.create(spec.game_name)
        result = await run_round_robin(
            game=game,
            agents=spec.agents,
            config=config,
            elo_calculator=elo_calculator,
        )
        results.append(result)
        print_standings(spec.game_name, result)

    # Save JSON
    json_path = output_dir / "experiment_results.json"
    json_data = build_json_output(specs, results, episodes, seed)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # Save CSV
    csv_path = output_dir / "experiment_summary.csv"
    rows = build_csv_rows(specs, results)
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"CSV saved:  {csv_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible game-theoretic experiments with builtin strategies."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per match (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiment",
        help="Directory for output files (default: results/experiment)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args and run the experiment."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    asyncio.run(run_experiment(args.episodes, args.seed, output_dir))


if __name__ == "__main__":
    # Ensure script can be run from project root
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    main()
