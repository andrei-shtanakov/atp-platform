"""Population dynamics simulation for evolutionary game theory.

Provides replicator dynamics, Moran process, ESS checking,
and a unified population simulator for studying strategy
evolution over generations.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from game_envs.core.game import Game
from game_envs.core.strategy import Strategy


@dataclass
class PopulationSnapshot:
    """State of a population at a generation.

    Attributes:
        generation: Generation number.
        frequencies: Strategy name to frequency mapping.
        fitness: Strategy name to average fitness.
    """

    generation: int
    frequencies: dict[str, float]
    fitness: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "generation": self.generation,
            "frequencies": dict(self.frequencies),
            "fitness": dict(self.fitness),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PopulationSnapshot:
        """Deserialize from dictionary."""
        return cls(
            generation=data["generation"],
            frequencies=data["frequencies"],
            fitness=data["fitness"],
        )


@dataclass
class PopulationResult:
    """Full result of a population dynamics simulation.

    Attributes:
        snapshots: List of population states over time.
        converged: Whether the population converged.
        convergence_generation: Generation at which convergence
            was detected, or None.
        dominant_strategy: Strategy with highest final
            frequency, or None.
        ess_strategies: Strategies classified as ESS.
    """

    snapshots: list[PopulationSnapshot]
    converged: bool = False
    convergence_generation: int | None = None
    dominant_strategy: str | None = None
    ess_strategies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "converged": self.converged,
            "convergence_generation": self.convergence_generation,
            "dominant_strategy": self.dominant_strategy,
            "ess_strategies": list(self.ess_strategies),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PopulationResult:
        """Deserialize from dictionary."""
        return cls(
            snapshots=[PopulationSnapshot.from_dict(s) for s in data["snapshots"]],
            converged=data.get("converged", False),
            convergence_generation=data.get("convergence_generation"),
            dominant_strategy=data.get("dominant_strategy"),
            ess_strategies=data.get("ess_strategies", []),
        )

    def frequency_timeseries(
        self,
    ) -> dict[str, list[float]]:
        """Get strategy frequency over time.

        Returns:
            Dict mapping strategy name to list of frequencies
            across all generations.
        """
        if not self.snapshots:
            return {}
        strategies = list(self.snapshots[0].frequencies.keys())
        return {
            s: [snap.frequencies.get(s, 0.0) for snap in self.snapshots]
            for s in strategies
        }


def _compute_payoff_matrix(
    game: Game,
    strategies: list[Strategy],
    num_rounds: int = 10,
) -> np.ndarray:
    """Compute average payoff matrix by simulating games.

    Args:
        game: The game to simulate.
        strategies: List of strategies.
        num_rounds: Rounds per game for repeated games.

    Returns:
        (n x n) payoff matrix where entry [i, j] is the
        average payoff to strategy i when playing against j.
    """
    n = len(strategies)
    payoff_matrix = np.zeros((n, n))
    players = game.player_ids[:2]

    for i in range(n):
        for j in range(n):
            strategies[i].reset()
            strategies[j].reset()
            game.reset()

            total_payoff = 0.0
            rounds_played = 0

            while not game.is_terminal:
                obs_0 = game.observe(players[0])
                obs_1 = game.observe(players[1])
                a0 = strategies[i].choose_action(obs_0)
                a1 = strategies[j].choose_action(obs_1)
                result = game.step({players[0]: a0, players[1]: a1})
                total_payoff += result.payoffs[players[0]]
                rounds_played += 1

            if rounds_played > 0:
                payoff_matrix[i, j] = total_payoff / rounds_played

    return payoff_matrix


class PopulationDynamics(ABC):
    """Abstract base for population dynamics models."""

    @abstractmethod
    def step(
        self,
        frequencies: np.ndarray,
        payoff_matrix: np.ndarray,
    ) -> np.ndarray:
        """Advance one generation.

        Args:
            frequencies: Current strategy frequencies.
            payoff_matrix: Payoff matrix between strategies.

        Returns:
            Updated strategy frequencies.
        """
        ...


class ReplicatorDynamics(PopulationDynamics):
    """Continuous-time replicator dynamics.

    Strategies grow proportional to their fitness advantage
    over the population average. Uses Euler integration.

    Args:
        dt: Time step for integration.
    """

    def __init__(self, dt: float = 0.1) -> None:
        self.dt = dt

    def step(
        self,
        frequencies: np.ndarray,
        payoff_matrix: np.ndarray,
    ) -> np.ndarray:
        """One step of replicator dynamics.

        dx_i/dt = x_i * (f_i - f_avg)

        Args:
            frequencies: Current strategy frequencies (sums to 1).
            payoff_matrix: (n x n) payoff matrix.

        Returns:
            Updated frequencies after one timestep.
        """
        fitness = payoff_matrix @ frequencies
        avg_fitness = float(frequencies @ fitness)
        delta = frequencies * (fitness - avg_fitness) * self.dt
        new_freq = frequencies + delta
        # Project onto simplex (handle numerical issues)
        new_freq = np.maximum(new_freq, 0.0)
        total = new_freq.sum()
        if total > 0:
            new_freq /= total
        return new_freq


class MoranProcess(PopulationDynamics):
    """Stochastic finite-population Moran process.

    In each step, one individual is chosen for reproduction
    (proportional to fitness) and one for death (uniform
    random). The offspring replaces the dead individual.

    Args:
        population_size: Number of individuals.
        intensity: Selection intensity. 0 = neutral drift,
            higher = stronger selection.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 100,
        intensity: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.population_size = population_size
        self.intensity = intensity
        self._rng = random.Random(seed)

    def step(
        self,
        frequencies: np.ndarray,
        payoff_matrix: np.ndarray,
    ) -> np.ndarray:
        """One step of the Moran process.

        Args:
            frequencies: Current strategy frequencies.
            payoff_matrix: (n x n) payoff matrix.

        Returns:
            Updated frequencies after one birth-death event.
        """
        n = len(frequencies)
        counts = np.round(frequencies * self.population_size).astype(int)
        # Fix rounding to match population size
        diff = self.population_size - counts.sum()
        if diff != 0:
            idx = int(np.argmax(counts))
            counts[idx] += diff

        fitness = payoff_matrix @ frequencies
        # Exponential fitness mapping
        exp_fitness = np.exp(self.intensity * fitness)
        weighted = counts * exp_fitness
        total_weighted = weighted.sum()

        if total_weighted == 0:
            return frequencies

        # Choose reproducer proportional to fitness
        birth_probs = weighted / total_weighted
        birth_cumulative = np.cumsum(birth_probs)
        r = self._rng.random()
        birth_idx = int(np.searchsorted(birth_cumulative, r))
        birth_idx = min(birth_idx, n - 1)

        # Choose individual to die uniformly
        death_cumulative = np.cumsum(counts / self.population_size)
        r = self._rng.random()
        death_idx = int(np.searchsorted(death_cumulative, r))
        death_idx = min(death_idx, n - 1)

        # Update counts
        counts[birth_idx] += 1
        counts[death_idx] -= 1
        counts = np.maximum(counts, 0)

        total = counts.sum()
        if total > 0:
            return counts.astype(float) / total
        return frequencies


def is_ess(
    strategy_idx: int,
    payoff_matrix: np.ndarray,
    epsilon: float = 1e-6,
) -> bool:
    """Check if a strategy is evolutionarily stable (ESS).

    A strategy s* is ESS if for all mutant strategies s != s*:
    1. E(s*, s*) > E(s, s*)  (strict NE condition), OR
    2. E(s*, s*) = E(s, s*) AND E(s*, s) > E(s, s)
       (stability condition)

    Args:
        strategy_idx: Index of the candidate ESS strategy.
        payoff_matrix: (n x n) payoff matrix.
        epsilon: Tolerance for equality comparison.

    Returns:
        True if the strategy is ESS.
    """
    n = payoff_matrix.shape[0]
    s_star = strategy_idx
    payoff_ss = payoff_matrix[s_star, s_star]

    for mutant in range(n):
        if mutant == s_star:
            continue

        payoff_ms = payoff_matrix[mutant, s_star]

        if payoff_ss > payoff_ms + epsilon:
            # Condition 1 satisfied: strict NE
            continue

        if abs(payoff_ss - payoff_ms) <= epsilon:
            # Check condition 2: stability
            payoff_sm = payoff_matrix[s_star, mutant]
            payoff_mm = payoff_matrix[mutant, mutant]
            if payoff_sm > payoff_mm + epsilon:
                continue
            # Failed stability condition
            return False
        else:
            # payoff_ms > payoff_ss: not even NE
            return False

    return True


def _detect_convergence(
    snapshots: list[PopulationSnapshot],
    window: int = 10,
    threshold: float = 1e-4,
) -> tuple[bool, int | None]:
    """Detect convergence in frequency timeseries.

    Checks if the max frequency change over a window of
    generations falls below the threshold.

    Args:
        snapshots: Population snapshots over time.
        window: Number of generations to look back.
        threshold: Maximum allowed frequency change.

    Returns:
        Tuple of (converged, generation at convergence).
    """
    if len(snapshots) < window + 1:
        return False, None

    strategies = list(snapshots[-1].frequencies.keys())

    for gen_idx in range(window, len(snapshots)):
        recent = snapshots[gen_idx - window : gen_idx + 1]
        max_change = 0.0
        for s in strategies:
            freqs = [snap.frequencies.get(s, 0.0) for snap in recent]
            max_change = max(max_change, max(freqs) - min(freqs))

        if max_change < threshold:
            return True, snapshots[gen_idx].generation

    return False, None


class PopulationSimulator:
    """Run population dynamics with strategies over generations.

    Supports mutation, convergence detection, and ESS
    classification.

    Args:
        game: The game to simulate.
        dynamics: Population dynamics model to use.
        mutation_rate: Probability of random mutation per
            generation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        game: Game,
        dynamics: PopulationDynamics | None = None,
        mutation_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.game = game
        self.dynamics = dynamics or ReplicatorDynamics()
        self.mutation_rate = mutation_rate
        self._rng = random.Random(seed)

    def run(
        self,
        strategies: list[Strategy],
        initial_frequencies: dict[str, float] | None = None,
        generations: int = 100,
        record_interval: int = 1,
    ) -> PopulationResult:
        """Run population simulation over generations.

        Args:
            strategies: List of strategies to compete.
            initial_frequencies: Strategy name to initial
                frequency. If None, uniform distribution.
            generations: Number of generations to simulate.
            record_interval: Record snapshot every N generations.

        Returns:
            PopulationResult with full timeseries.
        """
        n = len(strategies)
        names = [s.name for s in strategies]

        # Initialize frequencies
        if initial_frequencies is not None:
            freq = np.array([initial_frequencies.get(name, 0.0) for name in names])
            total = freq.sum()
            if total > 0:
                freq /= total
            else:
                freq = np.ones(n) / n
        else:
            freq = np.ones(n) / n

        # Compute payoff matrix
        payoff_matrix = _compute_payoff_matrix(self.game, strategies)

        snapshots: list[PopulationSnapshot] = []

        # Record initial state
        fitness = payoff_matrix @ freq
        snapshots.append(
            PopulationSnapshot(
                generation=0,
                frequencies=dict(zip(names, freq.tolist())),
                fitness=dict(zip(names, fitness.tolist())),
            )
        )

        for gen in range(1, generations + 1):
            freq = self.dynamics.step(freq, payoff_matrix)

            # Apply mutation
            if self.mutation_rate > 0:
                mutation = np.ones(n) / n
                freq = (1 - self.mutation_rate) * freq + self.mutation_rate * mutation

            # Record snapshot
            if gen % record_interval == 0:
                fitness = payoff_matrix @ freq
                snapshots.append(
                    PopulationSnapshot(
                        generation=gen,
                        frequencies=dict(zip(names, freq.tolist())),
                        fitness=dict(zip(names, fitness.tolist())),
                    )
                )

        # Detect convergence
        converged, conv_gen = _detect_convergence(snapshots)

        # Find dominant strategy
        final_freq = snapshots[-1].frequencies
        dominant = max(final_freq, key=lambda s: final_freq[s])

        # Check ESS for each strategy
        ess_list = [names[i] for i in range(n) if is_ess(i, payoff_matrix)]

        return PopulationResult(
            snapshots=snapshots,
            converged=converged,
            convergence_generation=conv_gen,
            dominant_strategy=dominant,
            ess_strategies=ess_list,
        )
