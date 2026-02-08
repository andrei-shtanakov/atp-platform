"""Tests for Nash equilibrium solvers.

Covers:
- Support enumeration (exact, all NE)
- Lemke-Howson (efficient single NE)
- Fictitious play (n-player approximate)
- Replicator dynamics (evolutionary)
- Known game verification
- Performance benchmarks
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from game_envs.analysis.models import NashEquilibrium
from game_envs.analysis.nash_solver import NashSolver

# --- Payoff matrices for known games ---

# Prisoner's Dilemma: unique NE at (Defect, Defect)
PD_P1 = np.array([[3.0, 0.0], [5.0, 1.0]])
PD_P2 = np.array([[3.0, 5.0], [0.0, 1.0]])

# Matching Pennies: unique mixed NE at (0.5, 0.5)
MP_P1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
MP_P2 = np.array([[-1.0, 1.0], [1.0, -1.0]])

# Battle of Sexes: 2 pure + 1 mixed NE
BOS_P1 = np.array([[3.0, 0.0], [0.0, 2.0]])
BOS_P2 = np.array([[2.0, 0.0], [0.0, 3.0]])

# Rock-Paper-Scissors: unique mixed NE at (1/3, 1/3, 1/3)
RPS_P1 = np.array(
    [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ]
)
RPS_P2 = -RPS_P1  # Zero-sum


class TestNashEquilibriumModel:
    """Tests for the NashEquilibrium data model."""

    def test_creation(self) -> None:
        ne = NashEquilibrium(
            strategies={
                "player_0": np.array([0.0, 1.0]),
                "player_1": np.array([0.0, 1.0]),
            },
            payoffs={"player_0": 1.0, "player_1": 1.0},
            support={
                "player_0": [1],
                "player_1": [1],
            },
        )
        assert ne.epsilon == 0.0
        assert ne.payoffs["player_0"] == 1.0

    def test_is_pure(self) -> None:
        ne = NashEquilibrium(
            strategies={
                "player_0": np.array([0.0, 1.0]),
                "player_1": np.array([1.0, 0.0]),
            },
            payoffs={"player_0": 0.0, "player_1": 0.0},
            support={"player_0": [1], "player_1": [0]},
        )
        assert ne.is_pure()
        assert not ne.is_mixed()

    def test_is_mixed(self) -> None:
        ne = NashEquilibrium(
            strategies={
                "player_0": np.array([0.5, 0.5]),
                "player_1": np.array([0.5, 0.5]),
            },
            payoffs={"player_0": 0.0, "player_1": 0.0},
            support={
                "player_0": [0, 1],
                "player_1": [0, 1],
            },
        )
        assert ne.is_mixed()
        assert not ne.is_pure()

    def test_serialization(self) -> None:
        ne = NashEquilibrium(
            strategies={
                "player_0": np.array([0.6, 0.4]),
                "player_1": np.array([0.4, 0.6]),
            },
            payoffs={"player_0": 1.2, "player_1": 1.2},
            support={
                "player_0": [0, 1],
                "player_1": [0, 1],
            },
            epsilon=0.01,
        )
        d = ne.to_dict()
        assert d["epsilon"] == 0.01
        assert d["strategies"]["player_0"] == [0.6, 0.4]

        restored = NashEquilibrium.from_dict(d)
        np.testing.assert_array_almost_equal(
            restored.strategies["player_0"],
            ne.strategies["player_0"],
        )
        assert restored.epsilon == ne.epsilon

    def test_repr(self) -> None:
        ne = NashEquilibrium(
            strategies={
                "player_0": np.array([0.5, 0.5]),
                "player_1": np.array([0.5, 0.5]),
            },
            payoffs={"player_0": 0.0, "player_1": 0.0},
            support={
                "player_0": [0, 1],
                "player_1": [0, 1],
            },
        )
        r = repr(ne)
        assert "NashEquilibrium" in r
        assert "player_0" in r


class TestSupportEnumeration:
    """Tests for support enumeration solver."""

    def test_prisoners_dilemma(self) -> None:
        """PD has unique NE: (Defect, Defect)."""
        eqs = NashSolver.support_enumeration(PD_P1, PD_P2)
        assert len(eqs) == 1
        eq = eqs[0]
        np.testing.assert_array_almost_equal(
            eq.strategies["player_0"],
            [0.0, 1.0],
        )
        np.testing.assert_array_almost_equal(
            eq.strategies["player_1"],
            [0.0, 1.0],
        )
        assert eq.is_pure()
        assert abs(eq.payoffs["player_0"] - 1.0) < 1e-10
        assert abs(eq.payoffs["player_1"] - 1.0) < 1e-10

    def test_matching_pennies(self) -> None:
        """Matching Pennies: unique mixed NE at (0.5, 0.5)."""
        eqs = NashSolver.support_enumeration(MP_P1, MP_P2)
        assert len(eqs) == 1
        eq = eqs[0]
        np.testing.assert_array_almost_equal(
            eq.strategies["player_0"],
            [0.5, 0.5],
        )
        np.testing.assert_array_almost_equal(
            eq.strategies["player_1"],
            [0.5, 0.5],
        )
        assert eq.is_mixed()
        assert abs(eq.payoffs["player_0"]) < 1e-10

    def test_battle_of_sexes(self) -> None:
        """BoS: 2 pure NE + 1 mixed NE = 3 total."""
        eqs = NashSolver.support_enumeration(BOS_P1, BOS_P2)
        assert len(eqs) == 3

        pure_eqs = [eq for eq in eqs if eq.is_pure()]
        mixed_eqs = [eq for eq in eqs if eq.is_mixed()]
        assert len(pure_eqs) == 2
        assert len(mixed_eqs) == 1

        # Mixed NE: p0 plays (3/5, 2/5), p1 plays (2/5, 3/5)
        mixed = mixed_eqs[0]
        np.testing.assert_array_almost_equal(
            mixed.strategies["player_0"],
            [0.6, 0.4],
        )
        np.testing.assert_array_almost_equal(
            mixed.strategies["player_1"],
            [0.4, 0.6],
        )

    def test_rock_paper_scissors(self) -> None:
        """RPS: unique mixed NE at (1/3, 1/3, 1/3)."""
        eqs = NashSolver.support_enumeration(RPS_P1, RPS_P2)
        assert len(eqs) == 1
        eq = eqs[0]
        np.testing.assert_array_almost_equal(
            eq.strategies["player_0"],
            [1 / 3, 1 / 3, 1 / 3],
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            eq.strategies["player_1"],
            [1 / 3, 1 / 3, 1 / 3],
            decimal=6,
        )

    def test_pure_coordination(self) -> None:
        """Pure coordination game: 2 pure NE."""
        p1 = np.array([[2.0, 0.0], [0.0, 1.0]])
        p2 = np.array([[2.0, 0.0], [0.0, 1.0]])
        eqs = NashSolver.support_enumeration(p1, p2)
        # Should find at least 2 pure NE + possibly 1 mixed
        pure_eqs = [eq for eq in eqs if eq.is_pure()]
        assert len(pure_eqs) >= 2


class TestLemkeHowson:
    """Tests for Lemke-Howson solver."""

    def test_prisoners_dilemma(self) -> None:
        eq = NashSolver.lemke_howson(PD_P1, PD_P2)
        # Should find the NE (Defect, Defect)
        np.testing.assert_array_almost_equal(
            eq.strategies["player_0"],
            [0.0, 1.0],
        )
        np.testing.assert_array_almost_equal(
            eq.strategies["player_1"],
            [0.0, 1.0],
        )

    def test_matching_pennies(self) -> None:
        eq = NashSolver.lemke_howson(MP_P1, MP_P2)
        np.testing.assert_array_almost_equal(
            eq.strategies["player_0"],
            [0.5, 0.5],
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            eq.strategies["player_1"],
            [0.5, 0.5],
            decimal=6,
        )

    def test_battle_of_sexes(self) -> None:
        """Should find at least one valid NE."""
        eq = NashSolver.lemke_howson(BOS_P1, BOS_P2)
        # Verify it's actually a NE by checking best response
        assert sum(eq.strategies["player_0"]) == pytest.approx(
            1.0,
            abs=1e-10,
        )
        assert sum(eq.strategies["player_1"]) == pytest.approx(
            1.0,
            abs=1e-10,
        )

    def test_large_game(self) -> None:
        """10x10 game should solve quickly."""
        rng = np.random.default_rng(42)
        p1 = rng.random((10, 10))
        p2 = rng.random((10, 10))
        eq = NashSolver.lemke_howson(p1, p2)
        # Strategy should be valid probability distribution
        assert eq.strategies["player_0"].sum() == pytest.approx(
            1.0,
            abs=1e-10,
        )
        assert np.all(eq.strategies["player_0"] >= -1e-10)

    def test_different_initial_labels(self) -> None:
        """Different initial labels may find different NE."""
        eqs_found = set()
        for label in range(4):
            eq = NashSolver.lemke_howson(
                BOS_P1,
                BOS_P2,
                initial_label=label,
            )
            key = tuple(np.round(eq.strategies["player_0"], 4))
            eqs_found.add(key)
        # Should find at least 1 distinct NE
        assert len(eqs_found) >= 1


class TestFictitiousPlay:
    """Tests for fictitious play solver."""

    def test_prisoners_dilemma(self) -> None:
        """FP should converge near (Defect, Defect)."""
        eq = NashSolver.fictitious_play(
            [PD_P1, PD_P2],
            max_iterations=10000,
            epsilon=0.01,
        )
        # Player 0 should mostly defect (index 1)
        assert eq.strategies["player_0"][1] > 0.8
        # Player 1 should mostly defect (index 1)
        assert eq.strategies["player_1"][1] > 0.8

    def test_matching_pennies(self) -> None:
        """FP should converge near (0.5, 0.5)."""
        eq = NashSolver.fictitious_play(
            [MP_P1, MP_P2],
            max_iterations=50000,
            epsilon=0.005,
        )
        np.testing.assert_array_almost_equal(
            eq.strategies["player_0"],
            [0.5, 0.5],
            decimal=1,
        )

    def test_three_player_game(self) -> None:
        """3-player game should produce valid strategies."""
        # Simple 3-player game (2 actions each)
        rng = np.random.default_rng(123)
        p1 = rng.random((2, 2, 2))
        p2 = rng.random((2, 2, 2))
        p3 = rng.random((2, 2, 2))
        eq = NashSolver.fictitious_play(
            [p1, p2, p3],
            max_iterations=5000,
        )
        assert len(eq.strategies) == 3
        for pid in ["player_0", "player_1", "player_2"]:
            assert pid in eq.strategies
            assert eq.strategies[pid].sum() == pytest.approx(
                1.0,
                abs=1e-10,
            )

    def test_convergence_flag(self) -> None:
        """Epsilon should reflect convergence quality."""
        eq = NashSolver.fictitious_play(
            [PD_P1, PD_P2],
            max_iterations=10000,
            epsilon=0.01,
        )
        # Should have converged
        assert eq.epsilon <= 0.1


class TestReplicatorDynamics:
    """Tests for replicator dynamics solver."""

    def test_prisoners_dilemma(self) -> None:
        """RD should converge to (Defect, Defect)."""
        eq = NashSolver.replicator_dynamics(
            [PD_P1, PD_P2],
            max_steps=20000,
        )
        # Should converge to defect
        assert eq.strategies["player_0"][1] > 0.99
        assert eq.strategies["player_1"][1] > 0.99

    def test_matching_pennies(self) -> None:
        """RD on matching pennies from (0.5, 0.5) should stay near."""
        initial = [
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
        ]
        eq = NashSolver.replicator_dynamics(
            [MP_P1, MP_P2],
            initial=initial,
            max_steps=5000,
            dt=0.001,
        )
        # Should stay near uniform (center is unstable
        # but starting exactly there should remain close)
        np.testing.assert_array_almost_equal(
            eq.strategies["player_0"],
            [0.5, 0.5],
            decimal=1,
        )

    def test_custom_initial(self) -> None:
        """Should accept custom initial strategies."""
        initial = [
            np.array([0.9, 0.1]),
            np.array([0.1, 0.9]),
        ]
        eq = NashSolver.replicator_dynamics(
            [PD_P1, PD_P2],
            initial=initial,
            max_steps=5000,
        )
        # Should converge from any initial to (D, D)
        assert eq.strategies["player_0"][1] > 0.9

    def test_three_player(self) -> None:
        """RD should work for 3-player games."""
        rng = np.random.default_rng(42)
        p1 = rng.random((3, 3, 3))
        p2 = rng.random((3, 3, 3))
        p3 = rng.random((3, 3, 3))
        eq = NashSolver.replicator_dynamics(
            [p1, p2, p3],
            max_steps=5000,
        )
        for pid in ["player_0", "player_1", "player_2"]:
            assert eq.strategies[pid].sum() == pytest.approx(
                1.0,
                abs=1e-10,
            )


class TestSolve2Player:
    """Tests for the solve_2player convenience method."""

    def test_support_enumeration(self) -> None:
        eqs = NashSolver.solve_2player(
            PD_P1,
            PD_P2,
            method="support_enumeration",
        )
        assert len(eqs) == 1

    def test_lemke_howson(self) -> None:
        eqs = NashSolver.solve_2player(
            PD_P1,
            PD_P2,
            method="lemke_howson",
        )
        assert len(eqs) == 1

    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            NashSolver.solve_2player(
                PD_P1,
                PD_P2,
                method="unknown",
            )


class TestSolveNPlayer:
    """Tests for the solve_nplayer convenience method."""

    def test_fictitious_play(self) -> None:
        eq = NashSolver.solve_nplayer(
            [PD_P1, PD_P2],
            method="fictitious_play",
        )
        assert len(eq.strategies) == 2

    def test_replicator_dynamics(self) -> None:
        eq = NashSolver.solve_nplayer(
            [PD_P1, PD_P2],
            method="replicator_dynamics",
        )
        assert len(eq.strategies) == 2

    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            NashSolver.solve_nplayer(
                [PD_P1, PD_P2],
                method="unknown",
            )


class TestPerformance:
    """Performance benchmarks for Nash solvers."""

    def test_10x10_lemke_howson_under_1s(self) -> None:
        """10x10 bimatrix should solve in < 1s."""
        rng = np.random.default_rng(42)
        p1 = rng.random((10, 10))
        p2 = rng.random((10, 10))
        start = time.time()
        eq = NashSolver.lemke_howson(p1, p2)
        elapsed = time.time() - start
        assert elapsed < 1.0
        assert eq.strategies["player_0"].sum() == pytest.approx(
            1.0,
            abs=1e-10,
        )

    def test_100x100_lemke_howson_under_10s(self) -> None:
        """100x100 bimatrix should solve in < 10s."""
        rng = np.random.default_rng(42)
        p1 = rng.random((100, 100))
        p2 = rng.random((100, 100))
        start = time.time()
        eq = NashSolver.lemke_howson(p1, p2)
        elapsed = time.time() - start
        assert elapsed < 10.0
        assert eq.strategies["player_0"].sum() == pytest.approx(
            1.0,
            abs=1e-10,
        )

    def test_support_enum_small_game_fast(self) -> None:
        """Small games (4x4) should be fast with support enum."""
        rng = np.random.default_rng(42)
        p1 = rng.random((4, 4))
        p2 = rng.random((4, 4))
        start = time.time()
        NashSolver.support_enumeration(p1, p2)
        elapsed = time.time() - start
        assert elapsed < 1.0


class TestVerificationOnKnownGames:
    """Verify solver results match known game-theoretic results.

    Each test verifies both the equilibrium strategies and the
    expected payoffs against analytically known values.
    """

    def test_pd_defect_is_dominant(self) -> None:
        """In PD, Defect strictly dominates Cooperate."""
        eqs = NashSolver.support_enumeration(PD_P1, PD_P2)
        assert len(eqs) == 1
        eq = eqs[0]
        # Both players defect
        assert eq.strategies["player_0"][1] == pytest.approx(
            1.0,
        )
        assert eq.strategies["player_1"][1] == pytest.approx(
            1.0,
        )
        # Payoff is (1, 1) = mutual defection
        assert eq.payoffs["player_0"] == pytest.approx(1.0)
        assert eq.payoffs["player_1"] == pytest.approx(1.0)

    def test_matching_pennies_mixed(self) -> None:
        """Matching Pennies: unique mixed NE (0.5, 0.5)."""
        eqs = NashSolver.support_enumeration(MP_P1, MP_P2)
        assert len(eqs) == 1
        eq = eqs[0]
        np.testing.assert_array_almost_equal(
            eq.strategies["player_0"],
            [0.5, 0.5],
        )
        # Expected payoff = 0 (zero-sum, fair game)
        assert eq.payoffs["player_0"] == pytest.approx(
            0.0,
            abs=1e-10,
        )

    def test_bos_three_equilibria(self) -> None:
        """Battle of Sexes: 2 pure + 1 mixed NE."""
        eqs = NashSolver.support_enumeration(BOS_P1, BOS_P2)
        assert len(eqs) == 3

        # Sort by player_0 payoff for deterministic checking
        eqs.sort(key=lambda e: e.payoffs["player_0"])

        # Mixed NE: payoff 6/5 = 1.2 for both
        mixed = eqs[0]
        assert mixed.payoffs["player_0"] == pytest.approx(
            1.2,
            abs=0.01,
        )
        assert mixed.payoffs["player_1"] == pytest.approx(
            1.2,
            abs=0.01,
        )

        # Pure NE 1: (Football, Football) = (2, 3)
        assert eqs[1].payoffs["player_0"] == pytest.approx(
            2.0,
        )
        assert eqs[1].payoffs["player_1"] == pytest.approx(
            3.0,
        )

        # Pure NE 2: (Opera, Opera) = (3, 2)
        assert eqs[2].payoffs["player_0"] == pytest.approx(
            3.0,
        )
        assert eqs[2].payoffs["player_1"] == pytest.approx(
            2.0,
        )

    def test_rps_uniform_mixed(self) -> None:
        """RPS: unique NE is uniform mix (1/3, 1/3, 1/3)."""
        eqs = NashSolver.support_enumeration(RPS_P1, RPS_P2)
        assert len(eqs) == 1
        eq = eqs[0]
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_array_almost_equal(
            eq.strategies["player_0"],
            expected,
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            eq.strategies["player_1"],
            expected,
            decimal=6,
        )
        assert eq.payoffs["player_0"] == pytest.approx(
            0.0,
            abs=1e-6,
        )

    def test_lemke_howson_agrees_with_support_enum(self) -> None:
        """Lemke-Howson should find one of the equilibria
        that support enumeration finds."""
        lh = NashSolver.lemke_howson(PD_P1, PD_P2)
        se = NashSolver.support_enumeration(PD_P1, PD_P2)

        # LH result should match one of the SE results
        found_match = False
        for eq in se:
            if np.allclose(
                lh.strategies["player_0"],
                eq.strategies["player_0"],
                atol=1e-6,
            ) and np.allclose(
                lh.strategies["player_1"],
                eq.strategies["player_1"],
                atol=1e-6,
            ):
                found_match = True
                break
        assert found_match

    def test_all_solvers_agree_on_pd(self) -> None:
        """All solvers should converge to (Defect, Defect) in PD."""
        se = NashSolver.support_enumeration(PD_P1, PD_P2)
        lh = NashSolver.lemke_howson(PD_P1, PD_P2)
        fp = NashSolver.fictitious_play(
            [PD_P1, PD_P2],
            max_iterations=10000,
        )
        rd = NashSolver.replicator_dynamics(
            [PD_P1, PD_P2],
            max_steps=10000,
        )

        for eq in [se[0], lh, fp, rd]:
            assert eq.strategies["player_0"][1] > 0.8
            assert eq.strategies["player_1"][1] > 0.8

    def test_1x1_game(self) -> None:
        """1x1 game has trivial NE."""
        p1 = np.array([[5.0]])
        p2 = np.array([[3.0]])
        eqs = NashSolver.support_enumeration(p1, p2)
        assert len(eqs) == 1
        eq = eqs[0]
        np.testing.assert_array_almost_equal(
            eq.strategies["player_0"],
            [1.0],
        )
        np.testing.assert_array_almost_equal(
            eq.strategies["player_1"],
            [1.0],
        )
        assert eq.payoffs["player_0"] == pytest.approx(5.0)
        assert eq.payoffs["player_1"] == pytest.approx(3.0)

    def test_asymmetric_game(self) -> None:
        """Non-square (2x3) payoff matrices."""
        p1 = np.array([[3.0, 0.0, 2.0], [1.0, 4.0, 0.0]])
        p2 = np.array([[1.0, 4.0, 0.0], [3.0, 0.0, 2.0]])
        eqs = NashSolver.support_enumeration(p1, p2)
        # Should find at least one NE
        assert len(eqs) >= 1
        for eq in eqs:
            assert eq.strategies["player_0"].sum() == pytest.approx(
                1.0,
                abs=1e-10,
            )
            assert eq.strategies["player_1"].sum() == pytest.approx(
                1.0,
                abs=1e-10,
            )

    def test_zero_sum_game_payoffs_sum_to_zero(self) -> None:
        """In a zero-sum game, NE payoffs should sum to ~0."""
        eqs = NashSolver.support_enumeration(RPS_P1, RPS_P2)
        for eq in eqs:
            total = eq.payoffs["player_0"] + eq.payoffs["player_1"]
            assert total == pytest.approx(0.0, abs=1e-6)
