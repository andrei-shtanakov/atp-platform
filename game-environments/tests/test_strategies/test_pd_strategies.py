"""Tests for Prisoner's Dilemma strategies."""

from __future__ import annotations

import pytest

from game_envs.core.state import Observation, RoundResult
from game_envs.games.prisoners_dilemma import PDConfig, PrisonersDilemma
from game_envs.strategies.pd_strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    Pavlov,
    RandomStrategy,
    TitForTat,
)


def _make_obs(
    player_id: str = "player_0",
    history: list[RoundResult] | None = None,
) -> Observation:
    """Create a minimal PD observation for testing."""
    return Observation(
        player_id=player_id,
        game_state={
            "game": "Prisoner's Dilemma",
            "your_role": player_id,
            "payoff_matrix": {
                "mutual_cooperation": 3.0,
                "mutual_defection": 1.0,
                "temptation": 5.0,
                "sucker": 0.0,
            },
        },
        available_actions=["cooperate", "defect"],
        history=history or [],
        round_number=len(history) if history else 0,
        total_rounds=10,
    )


def _round(
    r: int,
    a0: str,
    a1: str,
    p0: float = 0.0,
    p1: float = 0.0,
) -> RoundResult:
    """Create a round result."""
    return RoundResult(
        round_number=r,
        actions={"player_0": a0, "player_1": a1},
        payoffs={"player_0": p0, "player_1": p1},
    )


class TestAlwaysCooperate:
    def test_always_cooperates(self) -> None:
        s = AlwaysCooperate()
        assert s.choose_action(_make_obs()) == "cooperate"

    def test_cooperates_after_defection(self) -> None:
        s = AlwaysCooperate()
        obs = _make_obs(history=[_round(1, "cooperate", "defect")])
        assert s.choose_action(obs) == "cooperate"

    def test_name(self) -> None:
        assert AlwaysCooperate().name == "always_cooperate"


class TestAlwaysDefect:
    def test_always_defects(self) -> None:
        s = AlwaysDefect()
        assert s.choose_action(_make_obs()) == "defect"

    def test_defects_after_cooperation(self) -> None:
        s = AlwaysDefect()
        obs = _make_obs(history=[_round(1, "defect", "cooperate")])
        assert s.choose_action(obs) == "defect"

    def test_name(self) -> None:
        assert AlwaysDefect().name == "always_defect"


class TestTitForTat:
    def test_cooperates_first(self) -> None:
        s = TitForTat()
        assert s.choose_action(_make_obs()) == "cooperate"

    def test_copies_cooperation(self) -> None:
        s = TitForTat()
        obs = _make_obs(history=[_round(1, "cooperate", "cooperate", 3.0, 3.0)])
        assert s.choose_action(obs) == "cooperate"

    def test_copies_defection(self) -> None:
        s = TitForTat()
        obs = _make_obs(history=[_round(1, "cooperate", "defect", 0.0, 5.0)])
        assert s.choose_action(obs) == "defect"

    def test_forgives_after_cooperation(self) -> None:
        s = TitForTat()
        obs = _make_obs(
            history=[
                _round(1, "cooperate", "defect", 0.0, 5.0),
                _round(2, "defect", "cooperate", 5.0, 0.0),
            ]
        )
        assert s.choose_action(obs) == "cooperate"

    def test_round_by_round_sequence(self) -> None:
        """Verify TFT behavior round-by-round against AllD."""
        game = PrisonersDilemma(PDConfig(num_rounds=5, seed=42))
        tft = TitForTat()
        alld = AlwaysDefect()

        game.reset()
        actions_log = []
        while not game.is_terminal:
            obs_0 = game.observe("player_0")
            obs_1 = game.observe("player_1")
            a0 = tft.choose_action(obs_0)
            a1 = alld.choose_action(obs_1)
            actions_log.append((a0, a1))
            game.step({"player_0": a0, "player_1": a1})

        # Round 1: TFT cooperates, AllD defects
        assert actions_log[0] == ("cooperate", "defect")
        # Round 2+: TFT retaliates with defect
        for a0, a1 in actions_log[1:]:
            assert a0 == "defect"
            assert a1 == "defect"

    def test_name(self) -> None:
        assert TitForTat().name == "tit_for_tat"


class TestGrimTrigger:
    def test_cooperates_first(self) -> None:
        s = GrimTrigger()
        assert s.choose_action(_make_obs()) == "cooperate"

    def test_cooperates_if_no_defection(self) -> None:
        s = GrimTrigger()
        obs = _make_obs(history=[_round(1, "cooperate", "cooperate")])
        assert s.choose_action(obs) == "cooperate"

    def test_defects_after_opponent_defection(self) -> None:
        s = GrimTrigger()
        obs = _make_obs(history=[_round(1, "cooperate", "defect")])
        assert s.choose_action(obs) == "defect"

    def test_never_forgives(self) -> None:
        s = GrimTrigger()
        history = [
            _round(1, "cooperate", "defect"),
            _round(2, "defect", "cooperate"),
            _round(3, "defect", "cooperate"),
        ]
        obs = _make_obs(history=history)
        assert s.choose_action(obs) == "defect"

    def test_reset_clears_trigger(self) -> None:
        s = GrimTrigger()
        obs = _make_obs(history=[_round(1, "cooperate", "defect")])
        s.choose_action(obs)
        assert s._triggered
        s.reset()
        assert not s._triggered
        assert s.choose_action(_make_obs()) == "cooperate"

    def test_name(self) -> None:
        assert GrimTrigger().name == "grim_trigger"


class TestPavlov:
    def test_cooperates_first(self) -> None:
        s = Pavlov()
        assert s.choose_action(_make_obs()) == "cooperate"

    def test_stays_on_mutual_cooperation(self) -> None:
        s = Pavlov()
        obs = _make_obs(history=[_round(1, "cooperate", "cooperate")])
        assert s.choose_action(obs) == "cooperate"

    def test_switches_on_sucker(self) -> None:
        s = Pavlov()
        obs = _make_obs(history=[_round(1, "cooperate", "defect")])
        assert s.choose_action(obs) == "defect"

    def test_switches_on_temptation(self) -> None:
        s = Pavlov()
        obs = _make_obs(history=[_round(1, "defect", "cooperate")])
        assert s.choose_action(obs) == "defect"

    def test_stays_on_mutual_defection(self) -> None:
        s = Pavlov()
        obs = _make_obs(history=[_round(1, "defect", "defect")])
        assert s.choose_action(obs) == "cooperate"

    def test_name(self) -> None:
        assert Pavlov().name == "pavlov"


class TestRandomStrategy:
    def test_produces_valid_actions(self) -> None:
        s = RandomStrategy(seed=42)
        obs = _make_obs()
        for _ in range(20):
            action = s.choose_action(obs)
            assert action in ("cooperate", "defect")

    def test_deterministic_with_seed(self) -> None:
        s1 = RandomStrategy(seed=123)
        s2 = RandomStrategy(seed=123)
        obs = _make_obs()
        for _ in range(10):
            assert s1.choose_action(obs) == s2.choose_action(obs)

    def test_name(self) -> None:
        assert RandomStrategy().name == "random"


class TestPDPayoffsMatchTheory:
    """Verify AllC/AllD payoffs match theoretical predictions."""

    def test_allc_vs_alld_one_shot(self) -> None:
        """AllC vs AllD: AllC gets sucker (0), AllD gets temptation (5)."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        game.reset()
        obs_0 = game.observe("player_0")
        obs_1 = game.observe("player_1")
        a0 = AlwaysCooperate().choose_action(obs_0)
        a1 = AlwaysDefect().choose_action(obs_1)
        result = game.step({"player_0": a0, "player_1": a1})
        assert result.payoffs["player_0"] == 0.0  # sucker
        assert result.payoffs["player_1"] == 5.0  # temptation

    def test_allc_vs_allc(self) -> None:
        """AllC vs AllC: both get reward (3)."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        game.reset()
        result = game.step({"player_0": "cooperate", "player_1": "cooperate"})
        assert result.payoffs["player_0"] == 3.0
        assert result.payoffs["player_1"] == 3.0

    def test_alld_vs_alld(self) -> None:
        """AllD vs AllD: both get punishment (1)."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        game.reset()
        result = game.step({"player_0": "defect", "player_1": "defect"})
        assert result.payoffs["player_0"] == 1.0
        assert result.payoffs["player_1"] == 1.0

    def test_alld_dominates_one_shot(self) -> None:
        """In one-shot PD, AllD should dominate AllC."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        game.reset()
        obs_0 = game.observe("player_0")
        obs_1 = game.observe("player_1")
        a0 = AlwaysCooperate().choose_action(obs_0)
        a1 = AlwaysDefect().choose_action(obs_1)
        result = game.step({"player_0": a0, "player_1": a1})
        # AllD payoff should exceed AllC payoff
        assert result.payoffs["player_1"] > result.payoffs["player_0"]

    def test_tft_vs_tft_all_cooperation(self) -> None:
        """TFT vs TFT: always mutual cooperation."""
        game = PrisonersDilemma(PDConfig(num_rounds=10, seed=42))
        tft0 = TitForTat()
        tft1 = TitForTat()
        game.reset()
        while not game.is_terminal:
            obs_0 = game.observe("player_0")
            obs_1 = game.observe("player_1")
            game.step(
                {
                    "player_0": tft0.choose_action(obs_0),
                    "player_1": tft1.choose_action(obs_1),
                }
            )
        payoffs = game.get_payoffs()
        # Each round: 3.0 for mutual cooperation
        assert payoffs["player_0"] == pytest.approx(30.0)
        assert payoffs["player_1"] == pytest.approx(30.0)
