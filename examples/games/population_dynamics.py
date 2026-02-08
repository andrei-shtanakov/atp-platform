"""Population dynamics simulation example.

Demonstrates evolutionary dynamics with replicator dynamics,
Moran process, and ESS (evolutionarily stable strategy) checks.

Usage:
    cd game-environments
    uv run python ../examples/games/population_dynamics.py
"""

from __future__ import annotations

import numpy as np
from game_envs import (
    AlwaysDefect,
    PDConfig,
    PrisonersDilemma,
    TitForTat,
)
from game_envs.analysis.population import (
    MoranProcess,
    PopulationSimulator,
    ReplicatorDynamics,
    is_ess,
)


def replicator_dynamics_demo() -> None:
    """Simulate replicator dynamics in one-shot PD.

    In one-shot PD, Defect strictly dominates Cooperate.
    Replicator dynamics should drive cooperation to 0.
    """
    print("=" * 60)
    print("REPLICATOR DYNAMICS: One-shot Prisoner's Dilemma")
    print("=" * 60)

    # PD payoff matrix (row player perspective)
    # Actions: [Cooperate, Defect]
    # Payoffs: R=3, S=0, T=5, P=1
    payoff_matrix = np.array(
        [
            [3.0, 0.0],  # Cooperate vs [C, D]
            [5.0, 1.0],  # Defect vs [C, D]
        ]
    )

    rd = ReplicatorDynamics()
    result = rd.simulate(
        payoff_matrix=payoff_matrix,
        initial_freqs=np.array([0.5, 0.5]),
        generations=50,
    )

    print("\nStrategy frequencies over time:")
    print(f"  {'Gen':>4s}  {'Cooperate':>10s}  {'Defect':>10s}")
    for snap in result.snapshots[::10]:
        print(
            f"  {snap.generation:4d}  "
            f"{snap.frequencies[0]:10.4f}  "
            f"{snap.frequencies[1]:10.4f}"
        )

    final = result.snapshots[-1]
    print(
        f"\nFinal: Cooperate={final.frequencies[0]:.4f}, "
        f"Defect={final.frequencies[1]:.4f}"
    )
    print("As expected, Defect dominates in one-shot PD.")


def ess_check_demo() -> None:
    """Check which strategies are evolutionarily stable."""
    print("\n" + "=" * 60)
    print("ESS CHECK: Prisoner's Dilemma")
    print("=" * 60)

    payoff_matrix = np.array(
        [
            [3.0, 0.0],
            [5.0, 1.0],
        ]
    )

    strategies = ["Cooperate", "Defect"]
    for i, name in enumerate(strategies):
        stable = is_ess(i, payoff_matrix)
        status = "IS ESS" if stable else "is NOT ESS"
        print(f"  {name}: {status}")

    print("\nDefect is ESS because it's a strict Nash equilibrium.")
    print("Cooperate is not ESS because Defect can invade.")


def moran_process_demo() -> None:
    """Simulate Moran process (stochastic finite population).

    Unlike replicator dynamics, the Moran process is stochastic
    and models finite populations. Results may vary per run.
    """
    print("\n" + "=" * 60)
    print("MORAN PROCESS: Stochastic evolution (pop=30)")
    print("=" * 60)

    payoff_matrix = np.array(
        [
            [3.0, 0.0],
            [5.0, 1.0],
        ]
    )

    mp = MoranProcess(population_size=30)

    # Run 5 trials to show stochasticity
    print("\n5 independent trials:")
    for trial in range(5):
        result = mp.simulate(
            payoff_matrix=payoff_matrix,
            initial_counts=np.array([15, 15]),
            generations=100,
        )
        final = result.snapshots[-1]
        print(
            f"  Trial {trial + 1}: "
            f"Cooperators={final.frequencies[0]:.2f}, "
            f"Defectors={final.frequencies[1]:.2f}"
        )

    print("\nDefectors tend to fixate, but stochastic drift can occur.")


def population_simulator_demo() -> None:
    """Full population simulation with game-environments integration."""
    print("\n" + "=" * 60)
    print("POPULATION SIMULATOR: TFT vs AllD with mutation")
    print("=" * 60)

    game = PrisonersDilemma(PDConfig(num_rounds=10))
    strategies = [TitForTat(), AlwaysDefect()]

    sim = PopulationSimulator(
        game=game,
        strategies=strategies,
        population_size=50,
        mutation_rate=0.01,
    )

    result = sim.run(generations=100)

    print("\nEvolution of strategy frequencies:")
    print(f"  {'Gen':>4s}  {'TFT':>8s}  {'AllD':>8s}")
    for snap in result.snapshots[::20]:
        print(
            f"  {snap.generation:4d}  "
            f"{snap.frequencies[0]:8.4f}  "
            f"{snap.frequencies[1]:8.4f}"
        )

    final = result.snapshots[-1]
    print(f"\nFinal: TFT={final.frequencies[0]:.4f}, AllD={final.frequencies[1]:.4f}")

    if final.frequencies[0] > final.frequencies[1]:
        print("TFT dominates in repeated PD (cooperation sustained).")
    else:
        print("AllD dominates (cooperation not sustained).")


if __name__ == "__main__":
    replicator_dynamics_demo()
    ess_check_demo()
    moran_process_demo()
    population_simulator_demo()
    print("\nDone!")
