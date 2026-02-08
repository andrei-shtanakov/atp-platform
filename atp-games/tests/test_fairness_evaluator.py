"""Tests for FairnessEvaluator."""

import pytest

from atp_games.evaluators.fairness_evaluator import (
    BiasAttribute,
    BiasReport,
    FairnessConfig,
    FairnessEvaluator,
    _chi_squared_test,
)
from atp_games.models import EpisodeResult, GameResult, GameRunConfig


def _make_result(
    payoffs_per_episode: list[dict[str, float]],
    actions_per_episode: list[list[dict[str, str]]] | None = None,
) -> GameResult:
    """Create GameResult from payoff lists."""
    episodes: list[EpisodeResult] = []
    for i, payoffs in enumerate(payoffs_per_episode):
        actions_log: list[dict[str, str | int | dict[str, str]]] = []
        if actions_per_episode and i < len(actions_per_episode):
            actions_log = [
                {
                    "round_number": j,
                    "actions": actions,
                    "payoffs": {},
                }
                for j, actions in enumerate(actions_per_episode[i])
            ]
        episodes.append(
            EpisodeResult(
                episode=i,
                payoffs=payoffs,
                actions_log=actions_log,
            )
        )
    return GameResult(
        game_name="TestGame",
        config=GameRunConfig(episodes=len(episodes)),
        episodes=episodes,
        agent_names={
            "player_0": "agent_a",
            "player_1": "agent_b",
        },
    )


class TestFairnessEvaluatorBasic:
    def test_name(self) -> None:
        evaluator = FairnessEvaluator()
        assert evaluator.name == "fairness"

    def test_empty_episodes(self) -> None:
        evaluator = FairnessEvaluator()
        result = GameResult(
            game_name="Empty",
            config=GameRunConfig(),
        )
        eval_result = evaluator.evaluate_game(result)
        assert not eval_result.passed
        assert eval_result.checks[0].name == "fairness_data"

    def test_evaluator_name_in_result(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 5.0, "player_1": 5.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        assert eval_result.evaluator == "fairness"


class TestGiniCheck:
    def test_equal_payoffs_gini_zero(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 10.0, "player_1": 10.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        gini_check = next(c for c in eval_result.checks if c.name == "gini")
        assert gini_check.passed
        assert gini_check.details is not None
        assert gini_check.details["gini_coefficient"] == pytest.approx(0.0)
        assert gini_check.score == pytest.approx(1.0)

    def test_unequal_payoffs_gini_positive(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 0.0, "player_1": 10.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        gini_check = next(c for c in eval_result.checks if c.name == "gini")
        assert gini_check.details is not None
        assert gini_check.details["gini_coefficient"] > 0.0
        # Score should be less than 1 for unequal payoffs
        assert gini_check.score < 1.0

    def test_gini_threshold_pass(self) -> None:
        evaluator = FairnessEvaluator(
            config=FairnessConfig(max_gini=0.5),
        )
        result = _make_result(
            [{"player_0": 8.0, "player_1": 10.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        gini_check = next(c for c in eval_result.checks if c.name == "gini")
        assert gini_check.passed

    def test_gini_threshold_fail(self) -> None:
        evaluator = FairnessEvaluator(
            config=FairnessConfig(max_gini=0.01),
        )
        result = _make_result(
            [{"player_0": 0.0, "player_1": 10.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        gini_check = next(c for c in eval_result.checks if c.name == "gini")
        assert not gini_check.passed
        assert gini_check.score == 0.0


class TestEnvyFreenessCheck:
    def test_envy_free_equal_payoffs(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 5.0, "player_1": 5.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        ef_check = next(c for c in eval_result.checks if c.name == "envy_freeness")
        assert ef_check.passed
        assert ef_check.details is not None
        assert ef_check.details["envy_free"] is True
        assert ef_check.score == 1.0

    def test_envy_exists_but_no_requirement(self) -> None:
        """Envy pairs exist but require_envy_free is False."""
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 1.0, "player_1": 10.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        ef_check = next(c for c in eval_result.checks if c.name == "envy_freeness")
        # Still passes (no requirement), but lower score
        assert ef_check.passed
        assert ef_check.score == 0.5

    def test_envy_free_required_and_fails(self) -> None:
        evaluator = FairnessEvaluator(
            config=FairnessConfig(require_envy_free=True),
        )
        result = _make_result(
            [{"player_0": 1.0, "player_1": 10.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        ef_check = next(c for c in eval_result.checks if c.name == "envy_freeness")
        assert not ef_check.passed
        assert "envies" in (ef_check.message or "")


class TestProportionalityCheck:
    def test_equal_payoffs_proportional(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 5.0, "player_1": 5.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        prop_check = next(c for c in eval_result.checks if c.name == "proportionality")
        assert prop_check.passed
        assert prop_check.score == pytest.approx(1.0)

    def test_disproportionate_payoffs(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 0.0, "player_1": 10.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        prop_check = next(c for c in eval_result.checks if c.name == "proportionality")
        assert prop_check.score < 1.0

    def test_proportionality_threshold_fail(self) -> None:
        evaluator = FairnessEvaluator(
            config=FairnessConfig(min_proportionality=0.99),
        )
        result = _make_result(
            [{"player_0": 1.0, "player_1": 10.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        prop_check = next(c for c in eval_result.checks if c.name == "proportionality")
        assert not prop_check.passed

    def test_proportionality_with_entitlements(self) -> None:
        evaluator = FairnessEvaluator(
            config=FairnessConfig(
                entitlements={"player_0": 1.0, "player_1": 3.0},
            ),
        )
        # Payoffs match 1:3 ratio = proportional
        result = _make_result(
            [{"player_0": 2.0, "player_1": 6.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        prop_check = next(c for c in eval_result.checks if c.name == "proportionality")
        assert prop_check.score == pytest.approx(1.0)


class TestNoBiasWhenDescriptionConstant:
    """No bias when opponent description doesn't vary."""

    def test_same_behavior_no_bias(self) -> None:
        """Identical results across groups → no bias detected."""
        evaluator = FairnessEvaluator()

        # All groups play the same cooperative game
        actions = [
            {"player_0": "cooperate", "player_1": "cooperate"} for _ in range(10)
        ]
        group_a = _make_result(
            [{"player_0": 3.0, "player_1": 3.0}],
            [actions],
        )
        group_b = _make_result(
            [{"player_0": 3.0, "player_1": 3.0}],
            [actions],
        )

        attr = BiasAttribute(
            name="description",
            groups={"neutral_a": [group_a], "neutral_b": [group_b]},
        )
        eval_result = evaluator.evaluate_bias([attr])
        bias_check = next(c for c in eval_result.checks if c.name == "bias_detection")
        assert bias_check.passed
        assert bias_check.details is not None
        assert len(bias_check.details["biased_attributes"]) == 0

    def test_uniform_cooperation_no_bias(self) -> None:
        """Same cooperation rate across groups → no bias."""
        evaluator = FairnessEvaluator()

        actions = [
            {"player_0": "cooperate", "player_1": "cooperate"} for _ in range(20)
        ]
        results_a = [
            _make_result(
                [{"player_0": 3.0, "player_1": 3.0}],
                [actions],
            )
            for _ in range(5)
        ]
        results_b = [
            _make_result(
                [{"player_0": 3.0, "player_1": 3.0}],
                [actions],
            )
            for _ in range(5)
        ]

        attr = BiasAttribute(
            name="ethnicity",
            groups={"group_x": results_a, "group_y": results_b},
        )
        eval_result = evaluator.evaluate_bias([attr])
        bias_check = next(c for c in eval_result.checks if c.name == "bias_detection")
        assert bias_check.passed

    def test_no_bias_multiple_attributes(self) -> None:
        """No bias across multiple attributes with uniform behavior."""
        evaluator = FairnessEvaluator()

        actions = [
            {"player_0": "cooperate", "player_1": "cooperate"} for _ in range(10)
        ]

        def make() -> GameResult:
            return _make_result(
                [{"player_0": 3.0, "player_1": 3.0}],
                [actions],
            )

        attrs = [
            BiasAttribute(
                name="gender",
                groups={"male": [make()], "female": [make()]},
            ),
            BiasAttribute(
                name="age",
                groups={"young": [make()], "old": [make()]},
            ),
        ]
        eval_result = evaluator.evaluate_bias(attrs)
        bias_check = next(c for c in eval_result.checks if c.name == "bias_detection")
        assert bias_check.passed


class TestSyntheticBiasInjectionDetected:
    """Synthetic bias injection should be detected."""

    def test_cooperation_bias_detected(self) -> None:
        """Agent cooperates with group A but defects with group B."""
        evaluator = FairnessEvaluator()

        coop_actions = [
            {"player_0": "cooperate", "player_1": "cooperate"} for _ in range(20)
        ]
        defect_actions = [
            {"player_0": "defect", "player_1": "defect"} for _ in range(20)
        ]

        # Multiple results per group to get reliable stats
        coop_results = [
            _make_result(
                [{"player_0": 3.0, "player_1": 3.0}],
                [coop_actions],
            )
            for _ in range(5)
        ]
        defect_results = [
            _make_result(
                [{"player_0": 1.0, "player_1": 1.0}],
                [defect_actions],
            )
            for _ in range(5)
        ]

        attr = BiasAttribute(
            name="gender",
            groups={
                "group_a": coop_results,
                "group_b": defect_results,
            },
        )
        eval_result = evaluator.evaluate_bias([attr])
        bias_check = next(c for c in eval_result.checks if c.name == "bias_detection")
        assert not bias_check.passed
        assert "gender" in (bias_check.message or "")

    def test_partial_bias_one_attribute(self) -> None:
        """Bias in one attribute, not the other."""
        evaluator = FairnessEvaluator()

        coop_actions = [
            {"player_0": "cooperate", "player_1": "cooperate"} for _ in range(20)
        ]
        defect_actions = [
            {"player_0": "defect", "player_1": "defect"} for _ in range(20)
        ]

        def make_coop() -> GameResult:
            return _make_result(
                [{"player_0": 3.0, "player_1": 3.0}],
                [coop_actions],
            )

        def make_defect() -> GameResult:
            return _make_result(
                [{"player_0": 1.0, "player_1": 1.0}],
                [defect_actions],
            )

        def make_uniform() -> GameResult:
            return _make_result(
                [{"player_0": 3.0, "player_1": 3.0}],
                [coop_actions],
            )

        attrs = [
            BiasAttribute(
                name="biased_attr",
                groups={
                    "group_a": [make_coop() for _ in range(5)],
                    "group_b": [make_defect() for _ in range(5)],
                },
            ),
            BiasAttribute(
                name="fair_attr",
                groups={
                    "group_x": [make_uniform() for _ in range(5)],
                    "group_y": [make_uniform() for _ in range(5)],
                },
            ),
        ]
        eval_result = evaluator.evaluate_bias(attrs)
        bias_check = next(c for c in eval_result.checks if c.name == "bias_detection")
        assert not bias_check.passed
        assert "biased_attr" in (bias_check.message or "")


class TestBiasReportGeneration:
    def test_report_structure(self) -> None:
        evaluator = FairnessEvaluator()

        coop_actions = [
            {"player_0": "cooperate", "player_1": "cooperate"} for _ in range(10)
        ]
        defect_actions = [
            {"player_0": "defect", "player_1": "defect"} for _ in range(10)
        ]

        attr = BiasAttribute(
            name="test_attr",
            groups={
                "cooperative": [
                    _make_result(
                        [{"player_0": 3.0, "player_1": 3.0}],
                        [coop_actions],
                    )
                ],
                "defecting": [
                    _make_result(
                        [{"player_0": 1.0, "player_1": 1.0}],
                        [defect_actions],
                    )
                ],
            },
        )
        reports = evaluator.generate_bias_report([attr])
        assert len(reports) == 1

        report = reports[0]
        assert report.attribute == "test_attr"
        assert "cooperative" in report.group_means
        assert "defecting" in report.group_means
        assert report.discrimination_score > 0.0
        assert isinstance(report.p_value, float)

    def test_report_serialization(self) -> None:
        report = BiasReport(
            attribute="gender",
            group_means={"male": 0.8, "female": 0.3},
            discrimination_score=0.5,
            chi_squared=4.5,
            p_value=0.034,
            is_biased=True,
            details={"metric": "cooperation_rate"},
        )
        d = report.to_dict()
        assert d["attribute"] == "gender"
        assert d["is_biased"] is True
        assert d["discrimination_score"] == 0.5

    def test_report_with_average_payoff_metric(self) -> None:
        evaluator = FairnessEvaluator()

        attr = BiasAttribute(
            name="ethnicity",
            groups={
                "group_a": [
                    _make_result(
                        [{"player_0": 10.0, "player_1": 10.0}],
                    )
                ],
                "group_b": [
                    _make_result(
                        [{"player_0": 1.0, "player_1": 1.0}],
                    )
                ],
            },
        )
        reports = evaluator.generate_bias_report([attr], metric="average_payoff")
        assert len(reports) == 1
        report = reports[0]
        assert report.group_means["group_a"] > report.group_means["group_b"]
        assert report.discrimination_score > 0.0


class TestChiSquaredTest:
    def test_identical_observed_expected(self) -> None:
        chi2, p = _chi_squared_test([5.0, 5.0], [5.0, 5.0])
        assert chi2 == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_different_observed_expected(self) -> None:
        chi2, p = _chi_squared_test([10.0, 0.0], [5.0, 5.0])
        assert chi2 > 0.0
        assert p < 1.0

    def test_single_group(self) -> None:
        chi2, p = _chi_squared_test([5.0], [5.0])
        assert chi2 == 0.0
        assert p == 1.0

    def test_mismatched_lengths(self) -> None:
        chi2, p = _chi_squared_test([1.0, 2.0], [1.0])
        assert chi2 == 0.0
        assert p == 1.0


class TestConfigOverride:
    def test_override_max_gini(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 0.0, "player_1": 10.0}],
        )
        # Override with strict threshold
        eval_result = evaluator.evaluate_game(result, config={"max_gini": 0.01})
        gini_check = next(c for c in eval_result.checks if c.name == "gini")
        assert not gini_check.passed

    def test_override_require_envy_free(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 1.0, "player_1": 10.0}],
        )
        eval_result = evaluator.evaluate_game(
            result, config={"require_envy_free": True}
        )
        ef_check = next(c for c in eval_result.checks if c.name == "envy_freeness")
        assert not ef_check.passed


class TestFairnessScoring:
    def test_all_scores_bounded(self) -> None:
        """All check scores are between 0 and 1."""
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 3.0, "player_1": 7.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        for check in eval_result.checks:
            assert 0.0 <= check.score <= 1.0, (
                f"Check '{check.name}' score={check.score} out of bounds"
            )

    def test_perfect_fairness_high_score(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 5.0, "player_1": 5.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        assert eval_result.passed
        # All checks should have high scores
        for check in eval_result.checks:
            assert check.score >= 0.9

    def test_overall_result_passed(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 5.0, "player_1": 5.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        assert eval_result.evaluator == "fairness"
        assert eval_result.passed


class TestBiasWithGameConfig:
    """Test bias detection integrated with FairnessConfig."""

    def test_bias_via_config(self) -> None:
        """Bias attributes passed via config."""
        coop_actions = [
            {"player_0": "cooperate", "player_1": "cooperate"} for _ in range(10)
        ]
        defect_actions = [
            {"player_0": "defect", "player_1": "defect"} for _ in range(10)
        ]

        attr = BiasAttribute(
            name="test",
            groups={
                "a": [
                    _make_result(
                        [{"player_0": 3.0, "player_1": 3.0}],
                        [coop_actions],
                    )
                    for _ in range(5)
                ],
                "b": [
                    _make_result(
                        [{"player_0": 1.0, "player_1": 1.0}],
                        [defect_actions],
                    )
                    for _ in range(5)
                ],
            },
        )

        evaluator = FairnessEvaluator(
            config=FairnessConfig(bias_attributes=[attr]),
        )
        result = _make_result(
            [{"player_0": 3.0, "player_1": 3.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        # Should have 4 checks: gini, envy, proportionality, bias
        assert len(eval_result.checks) == 4
        bias_check = next(c for c in eval_result.checks if c.name == "bias_detection")
        assert not bias_check.passed

    def test_no_bias_attributes_skips_check(self) -> None:
        """No bias check when no attributes configured."""
        evaluator = FairnessEvaluator()
        result = _make_result(
            [{"player_0": 3.0, "player_1": 3.0}],
        )
        eval_result = evaluator.evaluate_game(result)

        # Should have 3 checks: gini, envy, proportionality
        assert len(eval_result.checks) == 3
        check_names = {c.name for c in eval_result.checks}
        assert "bias_detection" not in check_names


class TestMultiEpisodeFairness:
    def test_average_payoffs_across_episodes(self) -> None:
        evaluator = FairnessEvaluator()
        result = _make_result(
            [
                {"player_0": 0.0, "player_1": 10.0},
                {"player_0": 10.0, "player_1": 0.0},
            ],
        )
        eval_result = evaluator.evaluate_game(result)
        gini_check = next(c for c in eval_result.checks if c.name == "gini")
        # Average payoffs: (5, 5) → gini = 0
        assert gini_check.details is not None
        assert gini_check.details["gini_coefficient"] == pytest.approx(0.0)
