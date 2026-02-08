"""Tests for cooperation metrics analysis."""

from __future__ import annotations

import pytest

from game_envs.analysis.cooperation import (
    CooperationMetrics,
    conditional_cooperation,
    cooperation_rate,
    reciprocity_index,
)
from game_envs.core.state import RoundResult
from game_envs.games.prisoners_dilemma import PDConfig, PrisonersDilemma
from game_envs.strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    RandomStrategy,
    TitForTat,
)

# --- Helpers ---


def _make_history(
    actions_sequence: list[tuple[str, str]],
) -> list[RoundResult]:
    """Build history from a list of (p0_action, p1_action) tuples."""
    return [
        RoundResult(
            round_number=i + 1,
            actions={"player_0": a0, "player_1": a1},
            payoffs={"player_0": 0.0, "player_1": 0.0},
        )
        for i, (a0, a1) in enumerate(actions_sequence)
    ]


def _play_game(
    strategy_0,
    strategy_1,
    num_rounds: int = 100,
    seed: int = 42,
) -> list[RoundResult]:
    """Play a PD game between two strategies and return history."""
    config = PDConfig(num_rounds=num_rounds, seed=seed)
    game = PrisonersDilemma(config)
    game.reset()

    strategy_0.reset()
    strategy_1.reset()
    players = game.player_ids

    while not game.is_terminal:
        obs_0 = game.observe(players[0])
        obs_1 = game.observe(players[1])
        a0 = strategy_0.choose_action(obs_0)
        a1 = strategy_1.choose_action(obs_1)
        game.step({players[0]: a0, players[1]: a1})

    return game.history.rounds


# --- cooperation_rate tests ---


class TestCooperationRate:
    def test_always_cooperate(self):
        history = _make_history([("cooperate", "cooperate")] * 10)
        assert cooperation_rate(history, "player_0") == 1.0

    def test_always_defect(self):
        history = _make_history([("defect", "defect")] * 10)
        assert cooperation_rate(history, "player_0") == 0.0

    def test_mixed_actions(self):
        history = _make_history(
            [("cooperate", "defect")] * 5 + [("defect", "cooperate")] * 5
        )
        assert cooperation_rate(history, "player_0") == 0.5

    def test_tft_vs_tft(self):
        """TFT cooperation rate should be ~1.0 vs TFT."""
        history = _play_game(TitForTat(), TitForTat())
        rate = cooperation_rate(history, "player_0")
        assert rate >= 0.99

    def test_tft_vs_random(self):
        """TFT cooperation rate should be ~0.5 vs Random."""
        history = _play_game(TitForTat(), RandomStrategy(seed=42))
        rate = cooperation_rate(history, "player_0")
        assert 0.3 <= rate <= 0.7

    def test_empty_history_raises(self):
        with pytest.raises(ValueError, match="empty history"):
            cooperation_rate([], "player_0")

    def test_player_not_found_raises(self):
        history = _make_history([("cooperate", "defect")])
        with pytest.raises(ValueError, match="not found"):
            cooperation_rate(history, "player_99")


# --- conditional_cooperation tests ---


class TestConditionalCooperation:
    def test_tft_conditional_cooperation(self):
        """TFT should have P(C|C) = 1.0 when opponent always C."""
        # All cooperate → every transition is C→C
        history = _make_history([("cooperate", "cooperate")] * 10)
        result = conditional_cooperation(history, "player_0", "player_1")
        assert result["prob_c_given_c"] == 1.0
        # No defections → P(C|D) is None
        assert result["prob_c_given_d"] is None

    def test_tft_defects_after_opponent_defects(self):
        """After opponent defects, TFT-like player defects."""
        # Opponent defects every round, player defects in response
        history = _make_history([("cooperate", "defect")] + [("defect", "defect")] * 9)
        result = conditional_cooperation(history, "player_0", "player_1")
        assert result["prob_c_given_d"] == 0.0

    def test_always_cooperate_conditional(self):
        """AlwaysCooperate has P(C|C) = P(C|D) = 1.0."""
        history = _make_history(
            [("cooperate", "cooperate")] * 5 + [("cooperate", "defect")] * 5
        )
        result = conditional_cooperation(history, "player_0", "player_1")
        assert result["prob_c_given_c"] == 1.0
        assert result["prob_c_given_d"] == 1.0

    def test_auto_detect_opponent(self):
        history = _make_history([("cooperate", "cooperate")] * 5)
        result = conditional_cooperation(history, "player_0")
        assert result["prob_c_given_c"] == 1.0

    def test_insufficient_rounds_raises(self):
        history = _make_history([("cooperate", "defect")])
        with pytest.raises(ValueError, match="at least 2"):
            conditional_cooperation(history, "player_0")

    def test_none_when_no_observations(self):
        """If opponent never cooperated, P(C|C) is None."""
        history = _make_history([("cooperate", "defect")] * 5)
        result = conditional_cooperation(history, "player_0", "player_1")
        assert result["prob_c_given_c"] is None
        assert result["prob_c_given_d"] is not None


# --- reciprocity_index tests ---


class TestReciprocityIndex:
    def test_perfect_reciprocity(self):
        """Both players always cooperate → high reciprocity."""
        history = _make_history([("cooperate", "cooperate")] * 20)
        # Constant behavior → 0 (no variance)
        r = reciprocity_index(history)
        assert r == 0.0  # Both constant

    def test_correlated_cooperation(self):
        """Players cooperate/defect together → positive correlation."""
        history = _make_history(
            [("cooperate", "cooperate")] * 10 + [("defect", "defect")] * 10
        )
        r = reciprocity_index(history)
        assert r > 0.9

    def test_anti_correlated(self):
        """Players alternate: one cooperates when other defects."""
        history = _make_history(
            [("cooperate", "defect")] * 10 + [("defect", "cooperate")] * 10
        )
        r = reciprocity_index(history)
        assert r < -0.9

    def test_auto_detect_players(self):
        history = _make_history(
            [("cooperate", "cooperate")] * 10 + [("defect", "defect")] * 10
        )
        r = reciprocity_index(history)
        assert r > 0.9

    def test_empty_history_raises(self):
        with pytest.raises(ValueError, match="empty history"):
            reciprocity_index([])

    def test_single_player_raises(self):
        history = [
            RoundResult(
                round_number=1,
                actions={"player_0": "cooperate"},
                payoffs={"player_0": 1.0},
            )
        ]
        with pytest.raises(ValueError, match="at least 2 players"):
            reciprocity_index(history)


# --- CooperationMetrics serialization ---


class TestCooperationMetricsSerialization:
    def test_round_trip(self):
        metrics = CooperationMetrics(
            cooperation_rate=0.75,
            conditional_cooperation={
                "prob_c_given_c": 0.9,
                "prob_c_given_d": 0.3,
            },
            reciprocity_index=0.5,
        )
        data = metrics.to_dict()
        restored = CooperationMetrics.from_dict(data)
        assert restored.cooperation_rate == 0.75
        assert restored.conditional_cooperation == {
            "prob_c_given_c": 0.9,
            "prob_c_given_d": 0.3,
        }
        assert restored.reciprocity_index == 0.5

    def test_none_reciprocity(self):
        metrics = CooperationMetrics(
            cooperation_rate=1.0,
            conditional_cooperation={
                "prob_c_given_c": None,
                "prob_c_given_d": None,
            },
            reciprocity_index=None,
        )
        data = metrics.to_dict()
        restored = CooperationMetrics.from_dict(data)
        assert restored.reciprocity_index is None


# --- Integration with real game ---


class TestCooperationIntegration:
    def test_always_cooperate_vs_always_defect(self):
        """AllC vs AllD: AllC rate=1.0, AllD rate=0.0."""
        history = _play_game(AlwaysCooperate(), AlwaysDefect())
        assert cooperation_rate(history, "player_0") == 1.0
        assert cooperation_rate(history, "player_1") == 0.0

    def test_tft_reciprocity_vs_tft(self):
        """TFT vs TFT: perfect cooperation, zero variance → 0."""
        history = _play_game(TitForTat(), TitForTat())
        r = reciprocity_index(history)
        # Both always cooperate → constant → 0
        assert r == 0.0
