"""Tests for PayoffEvaluator."""

import pytest

from atp_games.evaluators.payoff_evaluator import (
    PayoffConfig,
    PayoffEvaluator,
)
from atp_games.models import EpisodeResult, GameResult, GameRunConfig


def _make_result(
    payoffs_list: list[dict[str, float]],
    game_name: str = "Test Game",
    agent_names: dict[str, str] | None = None,
) -> GameResult:
    """Helper to create a GameResult from a list of payoffs."""
    episodes = [EpisodeResult(episode=i, payoffs=p) for i, p in enumerate(payoffs_list)]
    return GameResult(
        game_name=game_name,
        config=GameRunConfig(episodes=len(payoffs_list)),
        episodes=episodes,
        agent_names=agent_names or {},
    )


class TestPayoffEvaluatorBasic:
    def test_name(self) -> None:
        evaluator = PayoffEvaluator()
        assert evaluator.name == "payoff"

    def test_empty_episodes(self) -> None:
        evaluator = PayoffEvaluator()
        result = GameResult(
            game_name="Empty",
            config=GameRunConfig(),
        )
        eval_result = evaluator.evaluate_game(result)
        assert not eval_result.passed
        assert eval_result.checks[0].name == "payoff_data"

    def test_single_episode(self) -> None:
        evaluator = PayoffEvaluator()
        result = _make_result([{"p0": 3.0, "p1": 3.0}])
        eval_result = evaluator.evaluate_game(result)
        assert eval_result.passed
        assert eval_result.evaluator == "payoff"


class TestPayoffEvaluatorAverages:
    def test_mutual_cooperation_pd(self) -> None:
        """Both cooperate in PD: payoff = (3, 3)."""
        evaluator = PayoffEvaluator()
        result = _make_result(
            [
                {"p0": 3.0, "p1": 3.0},
                {"p0": 3.0, "p1": 3.0},
            ]
        )
        eval_result = evaluator.evaluate_game(result)
        assert eval_result.passed

        avg_check = next(c for c in eval_result.checks if c.name == "average_payoff")
        assert avg_check.passed
        assert avg_check.details is not None
        assert avg_check.details["average_payoffs"]["p0"] == pytest.approx(3.0)

    def test_defector_wins_pd(self) -> None:
        """AllD vs AllC: defector gets 5, cooperator gets 0."""
        evaluator = PayoffEvaluator()
        result = _make_result(
            [
                {"p0": 5.0, "p1": 0.0},
                {"p0": 5.0, "p1": 0.0},
            ]
        )
        eval_result = evaluator.evaluate_game(result)
        assert eval_result.passed

        avg_check = next(c for c in eval_result.checks if c.name == "average_payoff")
        assert avg_check.details is not None
        assert avg_check.details["average_payoffs"]["p0"] == pytest.approx(5.0)
        assert avg_check.details["average_payoffs"]["p1"] == pytest.approx(0.0)

    def test_mutual_defection_pd(self) -> None:
        """Both defect: payoff = (1, 1)."""
        evaluator = PayoffEvaluator()
        result = _make_result(
            [
                {"p0": 1.0, "p1": 1.0},
            ]
        )
        eval_result = evaluator.evaluate_game(result)
        assert eval_result.passed


class TestPayoffThresholds:
    def test_min_payoff_pass(self) -> None:
        evaluator = PayoffEvaluator(
            config=PayoffConfig(
                min_payoff={"p0": 2.0, "p1": 2.0},
            ),
        )
        result = _make_result([{"p0": 3.0, "p1": 3.0}])
        eval_result = evaluator.evaluate_game(result)
        avg_check = next(c for c in eval_result.checks if c.name == "average_payoff")
        assert avg_check.passed

    def test_min_payoff_fail(self) -> None:
        evaluator = PayoffEvaluator(
            config=PayoffConfig(
                min_payoff={"p0": 4.0},
            ),
        )
        result = _make_result([{"p0": 2.0, "p1": 5.0}])
        eval_result = evaluator.evaluate_game(result)
        avg_check = next(c for c in eval_result.checks if c.name == "average_payoff")
        assert not avg_check.passed
        assert "p0" in (avg_check.message or "")

    def test_max_payoff_fail(self) -> None:
        evaluator = PayoffEvaluator(
            config=PayoffConfig(
                max_payoff={"p0": 2.0},
            ),
        )
        result = _make_result([{"p0": 5.0, "p1": 0.0}])
        eval_result = evaluator.evaluate_game(result)
        avg_check = next(c for c in eval_result.checks if c.name == "average_payoff")
        assert not avg_check.passed

    def test_config_override(self) -> None:
        """Override config via evaluate_game argument."""
        evaluator = PayoffEvaluator()
        result = _make_result([{"p0": 1.0, "p1": 1.0}])
        eval_result = evaluator.evaluate_game(
            result,
            config={"min_payoff": {"p0": 3.0}},
        )
        avg_check = next(c for c in eval_result.checks if c.name == "average_payoff")
        assert not avg_check.passed


class TestSocialWelfare:
    def test_social_welfare_computed(self) -> None:
        evaluator = PayoffEvaluator()
        result = _make_result([{"p0": 3.0, "p1": 3.0}])
        eval_result = evaluator.evaluate_game(result)
        welfare_check = next(
            c for c in eval_result.checks if c.name == "social_welfare"
        )
        assert welfare_check.passed
        assert welfare_check.details is not None
        assert welfare_check.details["social_welfare"] == pytest.approx(6.0)

    def test_social_welfare_threshold_pass(self) -> None:
        evaluator = PayoffEvaluator(
            config=PayoffConfig(min_social_welfare=4.0),
        )
        result = _make_result([{"p0": 3.0, "p1": 3.0}])
        eval_result = evaluator.evaluate_game(result)
        welfare_check = next(
            c for c in eval_result.checks if c.name == "social_welfare"
        )
        assert welfare_check.passed

    def test_social_welfare_threshold_fail(self) -> None:
        evaluator = PayoffEvaluator(
            config=PayoffConfig(min_social_welfare=10.0),
        )
        result = _make_result([{"p0": 1.0, "p1": 1.0}])
        eval_result = evaluator.evaluate_game(result)
        welfare_check = next(
            c for c in eval_result.checks if c.name == "social_welfare"
        )
        assert not welfare_check.passed


class TestPayoffDistribution:
    def test_distribution_stats(self) -> None:
        evaluator = PayoffEvaluator()
        result = _make_result(
            [
                {"p0": 1.0, "p1": 5.0},
                {"p0": 3.0, "p1": 3.0},
                {"p0": 5.0, "p1": 1.0},
                {"p0": 3.0, "p1": 3.0},
            ]
        )
        eval_result = evaluator.evaluate_game(result)
        dist_check = next(
            c for c in eval_result.checks if c.name == "payoff_distribution"
        )
        assert dist_check.passed
        assert dist_check.details is not None
        p0_stats = dist_check.details["per_player"]["p0"]
        assert p0_stats["min"] == pytest.approx(1.0)
        assert p0_stats["max"] == pytest.approx(5.0)
        assert p0_stats["mean"] == pytest.approx(3.0)


class TestParetoEfficiency:
    def test_pareto_not_checked_by_default(self) -> None:
        evaluator = PayoffEvaluator()
        result = _make_result(
            [
                {"p0": 1.0, "p1": 1.0},
                {"p0": 3.0, "p1": 3.0},
            ]
        )
        eval_result = evaluator.evaluate_game(result)
        pareto_checks = [c for c in eval_result.checks if c.name == "pareto_efficiency"]
        assert len(pareto_checks) == 0

    def test_pareto_dominated(self) -> None:
        evaluator = PayoffEvaluator(
            config=PayoffConfig(pareto_check=True),
        )
        # Average is (2, 2), but episode 1 has (3, 3) which dominates
        result = _make_result(
            [
                {"p0": 1.0, "p1": 1.0},
                {"p0": 3.0, "p1": 3.0},
            ]
        )
        eval_result = evaluator.evaluate_game(result)
        pareto_check = next(
            c for c in eval_result.checks if c.name == "pareto_efficiency"
        )
        assert not pareto_check.passed

    def test_pareto_efficient(self) -> None:
        evaluator = PayoffEvaluator(
            config=PayoffConfig(pareto_check=True),
        )
        # All episodes have same payoffs, avg is on frontier
        result = _make_result(
            [
                {"p0": 3.0, "p1": 3.0},
                {"p0": 3.0, "p1": 3.0},
            ]
        )
        eval_result = evaluator.evaluate_game(result)
        pareto_check = next(
            c for c in eval_result.checks if c.name == "pareto_efficiency"
        )
        assert pareto_check.passed

    def test_pareto_insufficient_episodes(self) -> None:
        evaluator = PayoffEvaluator(
            config=PayoffConfig(pareto_check=True),
        )
        result = _make_result([{"p0": 3.0, "p1": 3.0}])
        eval_result = evaluator.evaluate_game(result)
        pareto_check = next(
            c for c in eval_result.checks if c.name == "pareto_efficiency"
        )
        assert pareto_check.passed  # skipped, defaults to pass


class TestPayoffScoring:
    def test_score_zero_when_threshold_violated(self) -> None:
        evaluator = PayoffEvaluator(
            config=PayoffConfig(
                min_payoff={"p0": 5.0},
            ),
        )
        result = _make_result([{"p0": 2.5, "p1": 3.0}])
        eval_result = evaluator.evaluate_game(result)
        avg_check = next(c for c in eval_result.checks if c.name == "average_payoff")
        # Threshold violated → score = 0.0
        assert not avg_check.passed
        assert avg_check.score == pytest.approx(0.0)
        assert 0.0 <= avg_check.score <= 1.0

    def test_perfect_score(self) -> None:
        evaluator = PayoffEvaluator(
            config=PayoffConfig(
                min_payoff={"p0": 3.0},
            ),
        )
        result = _make_result([{"p0": 5.0, "p1": 3.0}])
        eval_result = evaluator.evaluate_game(result)
        avg_check = next(c for c in eval_result.checks if c.name == "average_payoff")
        assert avg_check.score == pytest.approx(1.0)

    def test_overall_score(self) -> None:
        evaluator = PayoffEvaluator()
        result = _make_result([{"p0": 3.0, "p1": 3.0}])
        eval_result = evaluator.evaluate_game(result)
        # All checks pass → score should be 1.0
        assert eval_result.score == pytest.approx(1.0)
