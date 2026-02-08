"""Tests for CooperationEvaluator."""

import pytest

from atp_games.evaluators.cooperation_evaluator import (
    CooperationConfig,
    CooperationEvaluator,
)
from atp_games.models import EpisodeResult, GameResult, GameRunConfig


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


class TestCooperationEvaluatorBasic:
    def test_name(self) -> None:
        evaluator = CooperationEvaluator()
        assert evaluator.name == "cooperation"

    def test_empty_episodes(self) -> None:
        evaluator = CooperationEvaluator()
        result = GameResult(
            game_name="Empty",
            config=GameRunConfig(),
        )
        eval_result = evaluator.evaluate_game(result)
        assert not eval_result.passed
        assert eval_result.checks[0].name == "cooperation_data"

    def test_no_action_history(self) -> None:
        evaluator = CooperationEvaluator()
        result = GameResult(
            game_name="PD",
            config=GameRunConfig(),
            episodes=[
                EpisodeResult(
                    episode=0,
                    payoffs={"player_0": 1.0, "player_1": 1.0},
                )
            ],
        )
        eval_result = evaluator.evaluate_game(result)
        assert not eval_result.passed

    def test_single_player_rejected(self) -> None:
        """Need at least 2 players for cooperation analysis."""
        evaluator = CooperationEvaluator()
        result = GameResult(
            game_name="Solo",
            config=GameRunConfig(),
            episodes=[
                EpisodeResult(
                    episode=0,
                    payoffs={"player_0": 1.0},
                    actions_log=[
                        {
                            "round_number": 0,
                            "actions": {"player_0": "cooperate"},
                            "payoffs": {},
                        }
                    ],
                )
            ],
        )
        eval_result = evaluator.evaluate_game(result)
        assert not eval_result.passed
        assert "2 players" in (eval_result.checks[0].message or "")


class TestTFTvsTFTCooperation:
    """TFT vs TFT should show cooperation ~ 1.0."""

    def test_full_cooperation(self) -> None:
        """Both cooperate every round → cooperation_rate = 1.0."""
        evaluator = CooperationEvaluator()
        rounds = [{"player_0": "cooperate", "player_1": "cooperate"} for _ in range(20)]
        result = _make_result_with_actions(
            [rounds],
            [{"player_0": 60.0, "player_1": 60.0}],
        )
        eval_result = evaluator.evaluate_game(result)
        assert eval_result.passed

        rate_check = next(c for c in eval_result.checks if c.name == "cooperation_rate")
        assert rate_check.passed
        assert rate_check.details is not None
        rates = rate_check.details["cooperation_rates"]
        assert rates["player_0"] == pytest.approx(1.0)
        assert rates["player_1"] == pytest.approx(1.0)

    def test_cooperation_rate_score(self) -> None:
        """Score should equal cooperation rate (avg of players)."""
        evaluator = CooperationEvaluator()
        rounds = [{"player_0": "cooperate", "player_1": "cooperate"} for _ in range(10)]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        rate_check = next(c for c in eval_result.checks if c.name == "cooperation_rate")
        assert rate_check.score == pytest.approx(1.0)

    def test_conditional_cooperation_tft(self) -> None:
        """TFT: P(C|C) ≈ 1.0, P(C|D) ≈ 0.0."""
        evaluator = CooperationEvaluator()
        rounds = [{"player_0": "cooperate", "player_1": "cooperate"} for _ in range(20)]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        cond_check = next(
            c for c in eval_result.checks if c.name == "conditional_cooperation"
        )
        assert cond_check.passed
        assert cond_check.details is not None
        p0_cond = cond_check.details["per_player"]["player_0"]
        assert p0_cond["prob_c_given_c"] == pytest.approx(1.0)

    def test_reciprocity_high(self) -> None:
        """Both always cooperating: zero variance means 0.0 reciprocity."""
        evaluator = CooperationEvaluator()
        rounds = [{"player_0": "cooperate", "player_1": "cooperate"} for _ in range(20)]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        recip_check = next(c for c in eval_result.checks if c.name == "reciprocity")
        # Zero variance yields 0.0 correlation
        assert recip_check.details is not None
        assert recip_check.details["reciprocity_index"] == pytest.approx(0.0)


class TestAllDvsAllCCooperation:
    """AllD vs AllC: cooperation = 0.5 (average of 0 and 1)."""

    def test_mixed_cooperation_rate(self) -> None:
        evaluator = CooperationEvaluator()
        rounds = [{"player_0": "defect", "player_1": "cooperate"} for _ in range(20)]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        rate_check = next(c for c in eval_result.checks if c.name == "cooperation_rate")
        assert rate_check.details is not None
        rates = rate_check.details["cooperation_rates"]
        assert rates["player_0"] == pytest.approx(0.0)
        assert rates["player_1"] == pytest.approx(1.0)

    def test_cooperation_score_half(self) -> None:
        """Score = avg cooperation rate = (0 + 1) / 2 = 0.5."""
        evaluator = CooperationEvaluator()
        rounds = [{"player_0": "defect", "player_1": "cooperate"} for _ in range(10)]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        rate_check = next(c for c in eval_result.checks if c.name == "cooperation_rate")
        assert rate_check.score == pytest.approx(0.5)


class TestCooperationThresholds:
    def test_min_cooperation_pass(self) -> None:
        evaluator = CooperationEvaluator(
            config=CooperationConfig(
                min_cooperation_rate={
                    "player_0": 0.8,
                    "player_1": 0.8,
                },
            ),
        )
        rounds = [{"player_0": "cooperate", "player_1": "cooperate"} for _ in range(10)]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        rate_check = next(c for c in eval_result.checks if c.name == "cooperation_rate")
        assert rate_check.passed

    def test_min_cooperation_fail(self) -> None:
        evaluator = CooperationEvaluator(
            config=CooperationConfig(
                min_cooperation_rate={"player_0": 0.8},
            ),
        )
        rounds = [{"player_0": "defect", "player_1": "cooperate"} for _ in range(10)]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        rate_check = next(c for c in eval_result.checks if c.name == "cooperation_rate")
        assert not rate_check.passed
        assert "player_0" in (rate_check.message or "")

    def test_max_cooperation_fail(self) -> None:
        evaluator = CooperationEvaluator(
            config=CooperationConfig(
                max_cooperation_rate={"player_1": 0.5},
            ),
        )
        rounds = [{"player_0": "defect", "player_1": "cooperate"} for _ in range(10)]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        rate_check = next(c for c in eval_result.checks if c.name == "cooperation_rate")
        assert not rate_check.passed

    def test_reciprocity_threshold_pass(self) -> None:
        evaluator = CooperationEvaluator(
            config=CooperationConfig(min_reciprocity=-0.5),
        )
        # Alternating pattern: some reciprocity
        rounds = [
            {"player_0": "cooperate", "player_1": "cooperate"},
            {"player_0": "defect", "player_1": "defect"},
        ] * 10
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        recip_check = next(c for c in eval_result.checks if c.name == "reciprocity")
        assert recip_check.passed

    def test_reciprocity_threshold_fail(self) -> None:
        evaluator = CooperationEvaluator(
            config=CooperationConfig(min_reciprocity=0.5),
        )
        # Anti-correlated: when one cooperates, other defects
        rounds = [
            {"player_0": "cooperate", "player_1": "defect"},
            {"player_0": "defect", "player_1": "cooperate"},
        ] * 10
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        recip_check = next(c for c in eval_result.checks if c.name == "reciprocity")
        assert not recip_check.passed


class TestConditionalCooperation:
    def test_tit_for_tat_pattern(self) -> None:
        """TFT: cooperate first, then copy opponent."""
        evaluator = CooperationEvaluator()
        # p0 plays TFT, p1 defects first then cooperates
        rounds = [
            {"player_0": "cooperate", "player_1": "defect"},
            {"player_0": "defect", "player_1": "cooperate"},
            {"player_0": "cooperate", "player_1": "defect"},
            {"player_0": "defect", "player_1": "cooperate"},
            {"player_0": "cooperate", "player_1": "defect"},
        ]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        cond_check = next(
            c for c in eval_result.checks if c.name == "conditional_cooperation"
        )
        assert cond_check.details is not None
        p0_cond = cond_check.details["per_player"]["player_0"]
        # P(C|C_opponent): when opponent cooperated (rounds 1,3),
        # player_0 cooperated (rounds 2,4) — both C
        assert p0_cond["prob_c_given_c"] is not None
        assert p0_cond["prob_c_given_d"] is not None

    def test_insufficient_rounds(self) -> None:
        """Single round: conditional cooperation returns None."""
        evaluator = CooperationEvaluator()
        rounds = [{"player_0": "cooperate", "player_1": "cooperate"}]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        cond_check = next(
            c for c in eval_result.checks if c.name == "conditional_cooperation"
        )
        assert cond_check.details is not None
        p0_cond = cond_check.details["per_player"]["player_0"]
        assert p0_cond["prob_c_given_c"] is None


class TestConfigOverride:
    def test_override_config(self) -> None:
        evaluator = CooperationEvaluator()
        rounds = [{"player_0": "defect", "player_1": "cooperate"} for _ in range(10)]
        result = _make_result_with_actions([rounds])

        # Override with strict min threshold
        eval_result = evaluator.evaluate_game(
            result,
            config={"min_cooperation_rate": {"player_0": 0.5}},
        )
        rate_check = next(c for c in eval_result.checks if c.name == "cooperation_rate")
        assert not rate_check.passed


class TestMultiEpisode:
    def test_cooperation_across_episodes(self) -> None:
        """History aggregated from all episodes."""
        evaluator = CooperationEvaluator()
        result = _make_result_with_actions(
            [
                [{"player_0": "cooperate", "player_1": "cooperate"}] * 5,
                [{"player_0": "defect", "player_1": "defect"}] * 5,
            ],
        )
        eval_result = evaluator.evaluate_game(result)

        rate_check = next(c for c in eval_result.checks if c.name == "cooperation_rate")
        assert rate_check.details is not None
        rates = rate_check.details["cooperation_rates"]
        # 5 coop + 5 defect = 50% cooperation
        assert rates["player_0"] == pytest.approx(0.5)
        assert rates["player_1"] == pytest.approx(0.5)


class TestCooperationScoring:
    def test_all_scores_bounded(self) -> None:
        """All check scores are between 0 and 1."""
        evaluator = CooperationEvaluator()
        rounds = [
            {"player_0": "cooperate", "player_1": "defect"},
            {"player_0": "defect", "player_1": "cooperate"},
        ] * 10
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)

        for check in eval_result.checks:
            assert 0.0 <= check.score <= 1.0, (
                f"Check '{check.name}' score={check.score} out of bounds"
            )

    def test_overall_result(self) -> None:
        evaluator = CooperationEvaluator()
        rounds = [{"player_0": "cooperate", "player_1": "cooperate"} for _ in range(10)]
        result = _make_result_with_actions([rounds])
        eval_result = evaluator.evaluate_game(result)
        assert eval_result.evaluator == "cooperation"
        assert eval_result.passed
