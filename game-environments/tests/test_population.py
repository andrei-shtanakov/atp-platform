"""Tests for population dynamics simulation."""

from __future__ import annotations

import numpy as np

from game_envs.analysis.population import (
    MoranProcess,
    PopulationResult,
    PopulationSimulator,
    PopulationSnapshot,
    ReplicatorDynamics,
    is_ess,
)
from game_envs.games.prisoners_dilemma import PDConfig, PrisonersDilemma
from game_envs.strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
)

# --- Prisoner's Dilemma payoff matrix ---
# Standard PD: T=5, R=3, P=1, S=0
# AllC vs AllC: 3, AllC vs AllD: 0, AllD vs AllC: 5, AllD vs AllD: 1
PD_PAYOFF = np.array(
    [
        [3.0, 0.0],  # AllC vs AllC, AllC vs AllD
        [5.0, 1.0],  # AllD vs AllC, AllD vs AllD
    ]
)


# --- PopulationSnapshot tests ---


class TestPopulationSnapshot:
    def test_creation(self):
        snap = PopulationSnapshot(
            generation=5,
            frequencies={"AllC": 0.4, "AllD": 0.6},
            fitness={"AllC": 1.2, "AllD": 2.3},
        )
        assert snap.generation == 5
        assert snap.frequencies["AllC"] == 0.4

    def test_serialization(self):
        snap = PopulationSnapshot(
            generation=10,
            frequencies={"AllC": 0.5, "AllD": 0.5},
            fitness={"AllC": 3.0, "AllD": 1.0},
        )
        data = snap.to_dict()
        restored = PopulationSnapshot.from_dict(data)
        assert restored.generation == 10
        assert restored.frequencies == snap.frequencies
        assert restored.fitness == snap.fitness


# --- PopulationResult tests ---


class TestPopulationResult:
    def test_frequency_timeseries(self):
        snaps = [
            PopulationSnapshot(
                generation=0,
                frequencies={"A": 0.5, "B": 0.5},
                fitness={"A": 1.0, "B": 1.0},
            ),
            PopulationSnapshot(
                generation=1,
                frequencies={"A": 0.6, "B": 0.4},
                fitness={"A": 1.0, "B": 1.0},
            ),
            PopulationSnapshot(
                generation=2,
                frequencies={"A": 0.7, "B": 0.3},
                fitness={"A": 1.0, "B": 1.0},
            ),
        ]
        result = PopulationResult(snapshots=snaps)
        ts = result.frequency_timeseries()
        assert ts["A"] == [0.5, 0.6, 0.7]
        assert ts["B"] == [0.5, 0.4, 0.3]

    def test_empty_timeseries(self):
        result = PopulationResult(snapshots=[])
        assert result.frequency_timeseries() == {}

    def test_serialization(self):
        result = PopulationResult(
            snapshots=[
                PopulationSnapshot(
                    generation=0,
                    frequencies={"A": 1.0},
                    fitness={"A": 2.0},
                )
            ],
            converged=True,
            convergence_generation=50,
            dominant_strategy="A",
            ess_strategies=["A"],
        )
        data = result.to_dict()
        restored = PopulationResult.from_dict(data)
        assert restored.converged is True
        assert restored.convergence_generation == 50
        assert restored.dominant_strategy == "A"
        assert restored.ess_strategies == ["A"]
        assert len(restored.snapshots) == 1


# --- ReplicatorDynamics tests ---


class TestReplicatorDynamics:
    def test_alld_dominates_in_one_shot_pd(self):
        """AllD should dominate in one-shot PD via replicator dynamics."""
        rd = ReplicatorDynamics(dt=0.1)
        freq = np.array([0.5, 0.5])  # 50/50 AllC/AllD

        for _ in range(500):
            freq = rd.step(freq, PD_PAYOFF)

        # AllD should dominate
        assert freq[1] > 0.95
        assert freq[0] < 0.05

    def test_pure_strategy_stable(self):
        """Pure AllD should be stable in PD."""
        rd = ReplicatorDynamics(dt=0.1)
        freq = np.array([0.01, 0.99])

        for _ in range(100):
            freq = rd.step(freq, PD_PAYOFF)

        assert freq[1] > 0.99

    def test_frequencies_sum_to_one(self):
        """Frequencies should always sum to 1."""
        rd = ReplicatorDynamics(dt=0.05)
        freq = np.array([0.3, 0.7])

        for _ in range(100):
            freq = rd.step(freq, PD_PAYOFF)
            assert abs(freq.sum() - 1.0) < 1e-10

    def test_three_strategies(self):
        """Replicator dynamics with 3 strategies (RPS-like)."""
        # Rock-Paper-Scissors payoff matrix
        rps = np.array(
            [
                [0.0, -1.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 1.0, 0.0],
            ]
        )
        rd = ReplicatorDynamics(dt=0.01)
        freq = np.array([0.33, 0.34, 0.33])

        for _ in range(100):
            freq = rd.step(freq, rps)

        # Should stay near uniform in RPS
        assert all(0.2 < f < 0.5 for f in freq)

    def test_custom_dt(self):
        """Smaller dt → slower convergence."""
        rd_fast = ReplicatorDynamics(dt=0.5)
        rd_slow = ReplicatorDynamics(dt=0.01)

        freq_fast = np.array([0.5, 0.5])
        freq_slow = np.array([0.5, 0.5])

        for _ in range(50):
            freq_fast = rd_fast.step(freq_fast, PD_PAYOFF)
            freq_slow = rd_slow.step(freq_slow, PD_PAYOFF)

        # Fast should be more converged
        assert freq_fast[1] > freq_slow[1]


# --- MoranProcess tests ---


class TestMoranProcess:
    def test_frequencies_valid(self):
        """Frequencies should stay non-negative and sum to ~1."""
        mp = MoranProcess(population_size=100, seed=42)
        freq = np.array([0.5, 0.5])

        for _ in range(50):
            freq = mp.step(freq, PD_PAYOFF)

        assert all(f >= 0.0 for f in freq)
        assert abs(freq.sum() - 1.0) < 0.02  # Allow small rounding

    def test_stochastic_variation(self):
        """Different seeds should produce different trajectories."""
        mp1 = MoranProcess(population_size=50, seed=1)
        mp2 = MoranProcess(population_size=50, seed=2)

        freq1 = np.array([0.5, 0.5])
        freq2 = np.array([0.5, 0.5])

        for _ in range(20):
            freq1 = mp1.step(freq1, PD_PAYOFF)
            freq2 = mp2.step(freq2, PD_PAYOFF)

        # Very unlikely to be identical
        assert not np.allclose(freq1, freq2)

    def test_strong_selection(self):
        """Strong selection → fitter strategy dominates faster."""
        mp = MoranProcess(population_size=100, intensity=5.0, seed=42)
        freq = np.array([0.3, 0.7])

        for _ in range(200):
            freq = mp.step(freq, PD_PAYOFF)

        # AllD should dominate with strong selection
        assert freq[1] > 0.5

    def test_neutral_drift(self):
        """With intensity=0, should be neutral drift."""
        mp = MoranProcess(population_size=100, intensity=0.0, seed=42)
        freq = np.array([0.5, 0.5])

        # Run many steps
        for _ in range(100):
            freq = mp.step(freq, PD_PAYOFF)

        # Should still be within bounds (can drift anywhere)
        assert all(0.0 <= f <= 1.0 for f in freq)


# --- is_ess tests ---


class TestIsESS:
    def test_alld_is_ess_in_pd(self):
        """AllD is ESS in one-shot PD."""
        assert is_ess(1, PD_PAYOFF) is True

    def test_allc_not_ess_in_pd(self):
        """AllC is NOT ESS in one-shot PD."""
        assert is_ess(0, PD_PAYOFF) is False

    def test_rps_no_pure_ess(self):
        """No pure strategy is ESS in RPS."""
        rps = np.array(
            [
                [0.0, -1.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 1.0, 0.0],
            ]
        )
        assert is_ess(0, rps) is False
        assert is_ess(1, rps) is False
        assert is_ess(2, rps) is False

    def test_dominant_strategy_is_ess(self):
        """A strictly dominant strategy is ESS."""
        # Strategy 0 dominates: always gets higher payoff
        payoff = np.array(
            [
                [5.0, 5.0],
                [1.0, 1.0],
            ]
        )
        assert is_ess(0, payoff) is True
        assert is_ess(1, payoff) is False

    def test_coordination_game(self):
        """Both pure strategies are ESS in coordination game."""
        coord = np.array(
            [
                [3.0, 0.0],
                [0.0, 3.0],
            ]
        )
        assert is_ess(0, coord) is True
        assert is_ess(1, coord) is True


# --- PopulationSimulator tests ---


class TestPopulationSimulator:
    def test_alld_dominates_allc_in_pd(self):
        """AllD dominates AllC in one-shot PD simulation."""
        config = PDConfig(num_rounds=1, seed=42)
        game = PrisonersDilemma(config)
        sim = PopulationSimulator(
            game=game,
            dynamics=ReplicatorDynamics(dt=0.1),
            seed=42,
        )

        result = sim.run(
            strategies=[AlwaysCooperate(), AlwaysDefect()],
            generations=200,
        )

        assert result.dominant_strategy == "always_defect"
        assert "always_defect" in result.ess_strategies
        assert "always_cooperate" not in result.ess_strategies

    def test_initial_frequencies(self):
        """Custom initial frequencies are respected."""
        config = PDConfig(num_rounds=1, seed=42)
        game = PrisonersDilemma(config)
        sim = PopulationSimulator(
            game=game,
            dynamics=ReplicatorDynamics(dt=0.1),
        )

        result = sim.run(
            strategies=[AlwaysCooperate(), AlwaysDefect()],
            initial_frequencies={
                "always_cooperate": 0.9,
                "always_defect": 0.1,
            },
            generations=10,
        )

        # First snapshot should reflect initial frequencies
        first = result.snapshots[0]
        assert abs(first.frequencies["always_cooperate"] - 0.9) < 0.01
        assert abs(first.frequencies["always_defect"] - 0.1) < 0.01

    def test_mutation_prevents_fixation(self):
        """Mutation keeps extinct strategies from vanishing."""
        config = PDConfig(num_rounds=1, seed=42)
        game = PrisonersDilemma(config)
        sim = PopulationSimulator(
            game=game,
            dynamics=ReplicatorDynamics(dt=0.1),
            mutation_rate=0.05,
        )

        result = sim.run(
            strategies=[AlwaysCooperate(), AlwaysDefect()],
            generations=200,
        )

        # With mutation, AllC shouldn't completely vanish
        final = result.snapshots[-1]
        assert final.frequencies["always_cooperate"] > 0.01

    def test_record_interval(self):
        """Record interval controls snapshot frequency."""
        config = PDConfig(num_rounds=1, seed=42)
        game = PrisonersDilemma(config)
        sim = PopulationSimulator(
            game=game,
            dynamics=ReplicatorDynamics(dt=0.1),
        )

        result = sim.run(
            strategies=[AlwaysCooperate(), AlwaysDefect()],
            generations=100,
            record_interval=10,
        )

        # 1 initial + 100/10 = 11 snapshots
        assert len(result.snapshots) == 11

    def test_convergence_detection(self):
        """Simulation should detect convergence."""
        config = PDConfig(num_rounds=1, seed=42)
        game = PrisonersDilemma(config)
        sim = PopulationSimulator(
            game=game,
            dynamics=ReplicatorDynamics(dt=0.1),
        )

        result = sim.run(
            strategies=[AlwaysCooperate(), AlwaysDefect()],
            generations=500,
        )

        # PD converges to AllD
        assert result.converged is True
        assert result.convergence_generation is not None

    def test_frequency_timeseries_output(self):
        """Result should provide strategy frequency timeseries."""
        config = PDConfig(num_rounds=1, seed=42)
        game = PrisonersDilemma(config)
        sim = PopulationSimulator(
            game=game,
            dynamics=ReplicatorDynamics(dt=0.1),
        )

        result = sim.run(
            strategies=[AlwaysCooperate(), AlwaysDefect()],
            generations=50,
        )

        ts = result.frequency_timeseries()
        assert "always_cooperate" in ts
        assert "always_defect" in ts
        assert len(ts["always_cooperate"]) == 51  # 0..50

    def test_tft_vs_alld_repeated_pd(self):
        """In repeated PD, TFT should be competitive."""
        config = PDConfig(num_rounds=10, seed=42)
        game = PrisonersDilemma(config)
        sim = PopulationSimulator(
            game=game,
            dynamics=ReplicatorDynamics(dt=0.1),
        )

        result = sim.run(
            strategies=[
                TitForTat(),
                AlwaysDefect(),
                AlwaysCooperate(),
            ],
            generations=200,
        )

        # TFT should not be eliminated
        final = result.snapshots[-1]
        tft_freq = final.frequencies.get("tit_for_tat", 0.0)
        # TFT is strong in repeated PD
        assert tft_freq > 0.1

    def test_moran_process_dynamics(self):
        """PopulationSimulator works with MoranProcess."""
        config = PDConfig(num_rounds=1, seed=42)
        game = PrisonersDilemma(config)
        sim = PopulationSimulator(
            game=game,
            dynamics=MoranProcess(population_size=50, seed=42),
        )

        result = sim.run(
            strategies=[AlwaysCooperate(), AlwaysDefect()],
            generations=100,
        )

        # Should produce valid snapshots
        assert len(result.snapshots) > 0
        for snap in result.snapshots:
            total = sum(snap.frequencies.values())
            assert abs(total - 1.0) < 0.05

    def test_result_serialization(self):
        """Full PopulationResult round-trips through dict."""
        config = PDConfig(num_rounds=1, seed=42)
        game = PrisonersDilemma(config)
        sim = PopulationSimulator(
            game=game,
            dynamics=ReplicatorDynamics(dt=0.1),
        )

        result = sim.run(
            strategies=[AlwaysCooperate(), AlwaysDefect()],
            generations=20,
        )

        data = result.to_dict()
        restored = PopulationResult.from_dict(data)
        assert restored.converged == result.converged
        assert restored.dominant_strategy == result.dominant_strategy
        assert len(restored.snapshots) == len(result.snapshots)
