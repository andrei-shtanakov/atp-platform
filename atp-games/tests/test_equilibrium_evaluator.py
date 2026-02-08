"""Tests for EquilibriumEvaluator."""

import pytest

from atp_games.evaluators.equilibrium_evaluator import (
    EquilibriumConfig,
    EquilibriumEvaluator,
)
from atp_games.models import EpisodeResult, GameResult, GameRunConfig

# Standard PD payoff matrices (R=3, S=0, T=5, P=1)
# Actions: cooperate=0, defect=1
PD_PAYOFF_1 = [[3.0, 0.0], [5.0, 1.0]]
PD_PAYOFF_2 = [[3.0, 5.0], [0.0, 1.0]]
PD_ACTIONS = ["cooperate", "defect"]


def _make_result_with_actions(
    actions_per_episode: list[list[dict[str, str]]],
    payoffs_per_episode: list[dict[str, float]] | None = None,
) -> GameResult:
    """Create GameResult with action history."""
    episodes: list[EpisodeResult] = []
    for i, rounds in enumerate(actions_per_episode):
        actions_log = [
            {
                "round_number": j,
                "actions": actions,
                "payoffs": {},
            }
            for j, actions in enumerate(rounds)
        ]
        payoffs = (
            payoffs_per_episode[i]
            if payoffs_per_episode
            else {"player_0": 0.0, "player_1": 0.0}
        )
        episodes.append(
            EpisodeResult(
                episode=i,
                payoffs=payoffs,
                actions_log=actions_log,
            )
        )
    return GameResult(
        game_name="PD",
        config=GameRunConfig(episodes=len(episodes)),
        episodes=episodes,
        agent_names={
            "player_0": "agent_a",
            "player_1": "agent_b",
        },
    )


class TestEquilibriumEvaluatorBasic:
    def test_name(self) -> None:
        evaluator = EquilibriumEvaluator()
        assert evaluator.name == "equilibrium"

    def test_empty_episodes(self) -> None:
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
            ),
        )
        result = GameResult(
            game_name="Empty",
            config=GameRunConfig(),
        )
        eval_result = evaluator.evaluate_game(result)
        assert not eval_result.passed

    def test_missing_payoff_matrices(self) -> None:
        evaluator = EquilibriumEvaluator()
        result = _make_result_with_actions(
            [[{"player_0": "defect", "player_1": "defect"}]],
            [{"player_0": 1.0, "player_1": 1.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        assert not eval_result.passed
        assert "Payoff matrices required" in (eval_result.checks[0].message or "")

    def test_three_players_rejected(self) -> None:
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        result = GameResult(
            game_name="3-player",
            config=GameRunConfig(),
            episodes=[
                EpisodeResult(
                    episode=0,
                    payoffs={
                        "p0": 1.0,
                        "p1": 1.0,
                        "p2": 1.0,
                    },
                    actions_log=[
                        {
                            "round_number": 0,
                            "actions": {
                                "p0": "a",
                                "p1": "b",
                                "p2": "c",
                            },
                            "payoffs": {},
                        }
                    ],
                ),
            ],
        )
        eval_result = evaluator.evaluate_game(result)
        assert not eval_result.passed
        assert "2 players" in (eval_result.checks[0].message or "")


class TestNashDistanceAtEquilibrium:
    """AllD vs AllD is the NE of one-shot PD."""

    def test_at_ne_distance_zero(self) -> None:
        """Playing (Defect, Defect) should have NE distance ≈ 0."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        rounds = [{"player_0": "defect", "player_1": "defect"} for _ in range(20)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 20.0, "player_1": 20.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        nash_check = next(c for c in eval_result.checks if c.name == "nash_distance")
        assert nash_check.passed
        assert nash_check.details is not None
        assert nash_check.details["min_nash_distance"] == pytest.approx(0.0, abs=1e-6)

    def test_at_ne_with_threshold(self) -> None:
        """NE play should pass any distance threshold."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                max_nash_distance=0.1,
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        rounds = [{"player_0": "defect", "player_1": "defect"} for _ in range(20)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 20.0, "player_1": 20.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        nash_check = next(c for c in eval_result.checks if c.name == "nash_distance")
        assert nash_check.passed


class TestNashDistanceAwayFromEquilibrium:
    """AllC is far from NE in PD."""

    def test_allc_distance_positive(self) -> None:
        """AllC vs AllC: distance to NE (D,D) should be 2.0."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        rounds = [{"player_0": "cooperate", "player_1": "cooperate"} for _ in range(20)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 60.0, "player_1": 60.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        nash_check = next(c for c in eval_result.checks if c.name == "nash_distance")
        assert nash_check.details is not None
        # (1,0) vs (0,1) for each player → L1 = 1+1 = 2 per player
        # NE is (defect, defect) = ([0,1], [0,1])
        # AllC = ([1,0], [1,0])
        # Distance = |1-0|+|0-1| + |1-0|+|0-1| = 2 + 2 = 4
        assert nash_check.details["min_nash_distance"] == pytest.approx(4.0, abs=0.01)

    def test_allc_fails_tight_threshold(self) -> None:
        """AllC should fail a tight NE distance threshold."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                max_nash_distance=0.5,
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        rounds = [{"player_0": "cooperate", "player_1": "cooperate"} for _ in range(20)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 60.0, "player_1": 60.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        nash_check = next(c for c in eval_result.checks if c.name == "nash_distance")
        assert not nash_check.passed


class TestEquilibriumType:
    def test_pd_has_pure_equilibrium(self) -> None:
        """PD has one pure NE: (Defect, Defect)."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        rounds = [{"player_0": "defect", "player_1": "defect"} for _ in range(10)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 10.0, "player_1": 10.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        type_check = next(c for c in eval_result.checks if c.name == "equilibrium_type")
        assert type_check.passed
        assert type_check.details is not None
        assert type_check.details["pure_count"] >= 1

    def test_matching_pennies_mixed_equilibrium(self) -> None:
        """Matching Pennies has only mixed NE."""
        # Matching pennies payoff matrices
        mp_payoff_1 = [[1.0, -1.0], [-1.0, 1.0]]
        mp_payoff_2 = [[-1.0, 1.0], [1.0, -1.0]]

        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                payoff_matrix_1=mp_payoff_1,
                payoff_matrix_2=mp_payoff_2,
                action_names_1=["heads", "tails"],
                action_names_2=["heads", "tails"],
            ),
        )
        rounds = [
            {"player_0": "heads", "player_1": "heads"},
            {"player_0": "tails", "player_1": "tails"},
        ] * 10
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 0.0, "player_1": 0.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        type_check = next(c for c in eval_result.checks if c.name == "equilibrium_type")
        assert type_check.details is not None
        assert type_check.details["mixed_count"] >= 1


class TestConvergenceDetection:
    def test_converged_strategy(self) -> None:
        """Stable strategy should be detected as converged."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                convergence_window=20,
                convergence_threshold=0.1,
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        # All defect throughout → strategy is stable
        rounds = [{"player_0": "defect", "player_1": "defect"} for _ in range(30)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 30.0, "player_1": 30.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        conv_check = next(c for c in eval_result.checks if c.name == "convergence")
        assert conv_check.passed
        assert conv_check.details is not None
        assert conv_check.details["converged"] is True

    def test_not_converged_strategy(self) -> None:
        """Changing strategy should be detected as not converged."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                convergence_window=20,
                convergence_threshold=0.1,
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        # First half: all cooperate, second half: all defect
        rounds = [
            {"player_0": "cooperate", "player_1": "cooperate"} for _ in range(15)
        ] + [{"player_0": "defect", "player_1": "defect"} for _ in range(15)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 30.0, "player_1": 30.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        conv_check = next(c for c in eval_result.checks if c.name == "convergence")
        assert not conv_check.passed
        assert conv_check.details is not None
        assert conv_check.details["converged"] is False

    def test_insufficient_rounds_for_convergence(self) -> None:
        """Too few rounds should skip convergence check."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                convergence_window=20,
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        rounds = [{"player_0": "defect", "player_1": "defect"} for _ in range(2)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 2.0, "player_1": 2.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        conv_check = next(c for c in eval_result.checks if c.name == "convergence")
        assert conv_check.passed  # Skipped = pass
        assert "insufficient" in (conv_check.message or "").lower()


class TestConfigOverride:
    def test_override_threshold(self) -> None:
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                max_nash_distance=0.01,
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        rounds = [{"player_0": "cooperate", "player_1": "cooperate"} for _ in range(10)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 30.0, "player_1": 30.0}],
        )

        # With tight threshold: fails
        eval_result = evaluator.evaluate_game(result)
        nash_check = next(c for c in eval_result.checks if c.name == "nash_distance")
        assert not nash_check.passed

        # Override with generous threshold
        eval_result_2 = evaluator.evaluate_game(
            result,
            config={
                "max_nash_distance": 10.0,
                "payoff_matrix_1": PD_PAYOFF_1,
                "payoff_matrix_2": PD_PAYOFF_2,
                "action_names_1": PD_ACTIONS,
                "action_names_2": PD_ACTIONS,
            },
        )
        nash_check_2 = next(
            c for c in eval_result_2.checks if c.name == "nash_distance"
        )
        assert nash_check_2.passed


class TestEquilibriumScoring:
    def test_score_1_at_ne(self) -> None:
        """Score should be high at Nash equilibrium."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        rounds = [{"player_0": "defect", "player_1": "defect"} for _ in range(20)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 20.0, "player_1": 20.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        nash_check = next(c for c in eval_result.checks if c.name == "nash_distance")
        assert nash_check.score == pytest.approx(1.0)

    def test_all_scores_bounded(self) -> None:
        """All check scores are between 0 and 1."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        rounds = [
            {"player_0": "cooperate", "player_1": "defect"},
            {"player_0": "defect", "player_1": "cooperate"},
        ] * 10
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 30.0, "player_1": 30.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        for check in eval_result.checks:
            assert 0.0 <= check.score <= 1.0, (
                f"Check '{check.name}' score={check.score} out of bounds"
            )


class TestMultiEpisode:
    def test_aggregate_across_episodes(self) -> None:
        """Strategies extracted across all episodes."""
        evaluator = EquilibriumEvaluator(
            config=EquilibriumConfig(
                payoff_matrix_1=PD_PAYOFF_1,
                payoff_matrix_2=PD_PAYOFF_2,
                action_names_1=PD_ACTIONS,
                action_names_2=PD_ACTIONS,
            ),
        )
        result = _make_result_with_actions(
            [
                [{"player_0": "defect", "player_1": "defect"}] * 5,
                [{"player_0": "defect", "player_1": "defect"}] * 5,
            ],
            [
                {"player_0": 5.0, "player_1": 5.0},
                {"player_0": 5.0, "player_1": 5.0},
            ],
        )
        eval_result = evaluator.evaluate_game(result)

        nash_check = next(c for c in eval_result.checks if c.name == "nash_distance")
        assert nash_check.passed
        assert nash_check.details is not None
        assert nash_check.details["min_nash_distance"] == pytest.approx(0.0, abs=1e-6)


class TestPluginRegistration:
    def test_registers_in_registry(self) -> None:
        """Cooperation and equilibrium should be registered."""
        from atp.evaluators.registry import EvaluatorRegistry

        from atp_games.evaluators.cooperation_evaluator import (
            CooperationEvaluator,
        )
        from atp_games.evaluators.equilibrium_evaluator import (
            EquilibriumEvaluator,
        )

        registry = EvaluatorRegistry()
        registry.register("cooperation", CooperationEvaluator)
        registry.register("equilibrium", EquilibriumEvaluator)

        assert registry.is_registered("cooperation")
        assert registry.is_registered("equilibrium")

        coop_cls = registry.get_evaluator_class("cooperation")
        assert coop_cls is CooperationEvaluator

        eq_cls = registry.get_evaluator_class("equilibrium")
        assert eq_cls is EquilibriumEvaluator
