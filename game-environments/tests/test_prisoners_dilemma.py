"""Tests for Prisoner's Dilemma game implementation."""

from __future__ import annotations

import pytest

from game_envs.core.game import GameType, MoveOrder
from game_envs.games.prisoners_dilemma import (
    COOPERATE,
    DEFECT,
    PDConfig,
    PrisonersDilemma,
)


class TestPDConfig:
    """Tests for PDConfig validation."""

    def test_defaults(self) -> None:
        c = PDConfig()
        assert c.reward == 3.0
        assert c.sucker == 0.0
        assert c.temptation == 5.0
        assert c.punishment == 1.0
        assert c.num_players == 2
        assert c.num_rounds == 1

    def test_frozen(self) -> None:
        c = PDConfig()
        with pytest.raises(AttributeError):
            c.reward = 10.0  # type: ignore[misc]

    def test_inherits_game_config_validation(self) -> None:
        with pytest.raises(ValueError, match="num_rounds"):
            PDConfig(num_rounds=0)

    def test_t_greater_than_r(self) -> None:
        with pytest.raises(ValueError, match="T > R"):
            PDConfig(temptation=2.0, reward=3.0)

    def test_r_greater_than_p(self) -> None:
        with pytest.raises(ValueError, match="R > P"):
            PDConfig(reward=1.0, punishment=2.0)

    def test_p_greater_than_s(self) -> None:
        with pytest.raises(ValueError, match="P > S"):
            PDConfig(punishment=0.0, sucker=1.0)

    def test_2r_greater_than_t_plus_s(self) -> None:
        # T=5, R=3 (T>R), P=1 (R>P), S=-1 (P>S)
        # but 2R=6, T+S=4, so 2R > T+S holds
        # We need 2R <= T+S: e.g. T=10, R=5, P=2, S=1
        # 2R=10, T+S=11 -> fails
        with pytest.raises(ValueError, match="2R > T \\+ S"):
            PDConfig(temptation=10.0, reward=5.0, punishment=2.0, sucker=1.0)

    def test_valid_custom_config(self) -> None:
        c = PDConfig(
            reward=4.0,
            sucker=0.0,
            temptation=6.0,
            punishment=2.0,
        )
        assert c.reward == 4.0
        assert c.temptation == 6.0

    def test_payoff_matrix_analytical(self) -> None:
        """Verify payoff constraints analytically."""
        c = PDConfig()
        assert c.temptation > c.reward > c.punishment > c.sucker
        assert 2 * c.reward > c.temptation + c.sucker


class TestPrisonersDilemmaOneShot:
    """Tests for one-shot Prisoner's Dilemma."""

    def test_properties(self) -> None:
        game = PrisonersDilemma()
        assert game.name == "Prisoner's Dilemma"
        assert game.game_type == GameType.NORMAL_FORM
        assert game.move_order == MoveOrder.SIMULTANEOUS
        assert game.player_ids == ["player_0", "player_1"]

    def test_action_space(self) -> None:
        game = PrisonersDilemma()
        space = game.action_space("player_0")
        assert space.contains(COOPERATE)
        assert space.contains(DEFECT)
        assert not space.contains("invalid")
        assert space.to_list() == [COOPERATE, DEFECT]

    def test_reset(self) -> None:
        game = PrisonersDilemma()
        result = game.reset()
        assert not result.is_terminal
        assert result.payoffs == {"player_0": 0.0, "player_1": 0.0}
        assert "player_0" in result.observations
        assert "player_1" in result.observations
        assert game.current_round == 0

    def test_mutual_cooperation(self) -> None:
        """(C, C) -> (R, R) = (3, 3)."""
        game = PrisonersDilemma()
        game.reset()
        result = game.step({"player_0": COOPERATE, "player_1": COOPERATE})
        assert result.payoffs == {"player_0": 3.0, "player_1": 3.0}
        assert result.is_terminal
        assert game.get_payoffs() == {
            "player_0": 3.0,
            "player_1": 3.0,
        }

    def test_mutual_defection(self) -> None:
        """(D, D) -> (P, P) = (1, 1)."""
        game = PrisonersDilemma()
        game.reset()
        result = game.step({"player_0": DEFECT, "player_1": DEFECT})
        assert result.payoffs == {"player_0": 1.0, "player_1": 1.0}
        assert result.is_terminal

    def test_temptation_and_sucker(self) -> None:
        """(D, C) -> (T, S) = (5, 0)."""
        game = PrisonersDilemma()
        game.reset()
        result = game.step({"player_0": DEFECT, "player_1": COOPERATE})
        assert result.payoffs == {"player_0": 5.0, "player_1": 0.0}

    def test_sucker_and_temptation(self) -> None:
        """(C, D) -> (S, T) = (0, 5)."""
        game = PrisonersDilemma()
        game.reset()
        result = game.step({"player_0": COOPERATE, "player_1": DEFECT})
        assert result.payoffs == {"player_0": 0.0, "player_1": 5.0}

    def test_invalid_action_raises(self) -> None:
        game = PrisonersDilemma()
        game.reset()
        with pytest.raises(ValueError, match="Invalid action for player_0"):
            game.step({"player_0": "invalid", "player_1": COOPERATE})
        game.reset()
        with pytest.raises(ValueError, match="Invalid action for player_1"):
            game.step({"player_0": COOPERATE, "player_1": "invalid"})

    def test_step_after_terminal_raises(self) -> None:
        game = PrisonersDilemma()
        game.reset()
        game.step({"player_0": COOPERATE, "player_1": COOPERATE})
        with pytest.raises(RuntimeError, match="terminal"):
            game.step({"player_0": COOPERATE, "player_1": COOPERATE})

    def test_custom_payoffs(self) -> None:
        config = PDConfig(
            reward=4.0,
            sucker=0.0,
            temptation=6.0,
            punishment=2.0,
        )
        game = PrisonersDilemma(config)
        game.reset()
        result = game.step({"player_0": COOPERATE, "player_1": COOPERATE})
        assert result.payoffs == {"player_0": 4.0, "player_1": 4.0}


class TestPrisonersDilemmaRepeated:
    """Tests for repeated Prisoner's Dilemma."""

    def test_repeated_name(self) -> None:
        config = PDConfig(num_rounds=10)
        game = PrisonersDilemma(config)
        assert game.name == "Prisoner's Dilemma (repeated x10)"
        assert game.game_type == GameType.REPEATED

    def test_ten_rounds_mutual_cooperation(self) -> None:
        config = PDConfig(num_rounds=10, seed=42)
        game = PrisonersDilemma(config)
        game.reset()

        for i in range(10):
            result = game.step({"player_0": COOPERATE, "player_1": COOPERATE})
            if i < 9:
                assert not result.is_terminal
            else:
                assert result.is_terminal
            assert result.payoffs == {
                "player_0": 3.0,
                "player_1": 3.0,
            }

        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == 30.0
        assert payoffs["player_1"] == 30.0

    def test_history_grows_each_round(self) -> None:
        config = PDConfig(num_rounds=5)
        game = PrisonersDilemma(config)
        game.reset()

        for i in range(5):
            game.step({"player_0": COOPERATE, "player_1": DEFECT})
            assert len(game.history) == i + 1

    def test_observe_shows_history(self) -> None:
        config = PDConfig(num_rounds=3)
        game = PrisonersDilemma(config)
        game.reset()

        game.step({"player_0": COOPERATE, "player_1": DEFECT})
        obs = game.observe("player_0")

        assert obs.round_number == 1
        assert obs.total_rounds == 3
        assert len(obs.history) == 1
        assert obs.history[0].actions["player_0"] == COOPERATE
        assert obs.history[0].actions["player_1"] == DEFECT

    def test_discount_factor(self) -> None:
        config = PDConfig(
            num_rounds=3,
            discount_factor=0.5,
        )
        game = PrisonersDilemma(config)
        game.reset()

        # Round 0: discount = 0.5^0 = 1.0, payoff = 3*1 = 3
        game.step({"player_0": COOPERATE, "player_1": COOPERATE})
        # Round 1: discount = 0.5^1 = 0.5, payoff = 3*0.5 = 1.5
        game.step({"player_0": COOPERATE, "player_1": COOPERATE})
        # Round 2: discount = 0.5^2 = 0.25, payoff = 3*0.25 = 0.75
        game.step({"player_0": COOPERATE, "player_1": COOPERATE})

        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(5.25)
        assert payoffs["player_1"] == pytest.approx(5.25)

    def test_reset_clears_state(self) -> None:
        config = PDConfig(num_rounds=3)
        game = PrisonersDilemma(config)
        game.reset()
        game.step({"player_0": COOPERATE, "player_1": COOPERATE})

        # Reset and verify clean state
        game.reset()
        assert game.current_round == 0
        assert not game.is_terminal
        assert len(game.history) == 0
        assert game.get_payoffs() == {
            "player_0": 0.0,
            "player_1": 0.0,
        }


class TestPrisonersDilemmaNoise:
    """Tests for noise (trembling hand) in PD."""

    def test_noise_zero_no_flip(self) -> None:
        config = PDConfig(noise=0.0, seed=42)
        game = PrisonersDilemma(config)
        game.reset()
        result = game.step({"player_0": COOPERATE, "player_1": COOPERATE})
        assert result.payoffs == {"player_0": 3.0, "player_1": 3.0}

    def test_noise_one_always_flips(self) -> None:
        config = PDConfig(noise=1.0, seed=42)
        game = PrisonersDilemma(config)
        game.reset()
        # Both cooperate, but noise=1.0 -> both flip to defect
        result = game.step({"player_0": COOPERATE, "player_1": COOPERATE})
        assert result.payoffs == {"player_0": 1.0, "player_1": 1.0}

    def test_noise_flip_rate_statistical(self) -> None:
        """Over many rounds, ~50% of actions should flip."""
        config = PDConfig(
            num_rounds=1000,
            noise=0.5,
            seed=123,
        )
        game = PrisonersDilemma(config)
        game.reset()

        flips_p0 = 0
        for _ in range(1000):
            game.step({"player_0": COOPERATE, "player_1": COOPERATE})

        # Count how many times player_0 ended up defecting
        # (i.e., action was flipped from cooperate)
        for rr in game.history.rounds:
            if rr.actions["player_0"] == DEFECT:
                flips_p0 += 1

        # With noise=0.5, expect ~500 flips out of 1000
        assert 400 < flips_p0 < 600

    def test_noise_records_actual_actions(self) -> None:
        """History should record noised actions, not original."""
        config = PDConfig(noise=1.0, seed=42)
        game = PrisonersDilemma(config)
        game.reset()
        game.step({"player_0": COOPERATE, "player_1": DEFECT})

        rr = game.history.rounds[0]
        # Cooperate flips to defect, defect flips to cooperate
        assert rr.actions["player_0"] == DEFECT
        assert rr.actions["player_1"] == COOPERATE


class TestPrisonersDilemmaObserve:
    """Tests for observe() and to_prompt()."""

    def test_observe_initial(self) -> None:
        game = PrisonersDilemma()
        game.reset()
        obs = game.observe("player_0")

        assert obs.player_id == "player_0"
        assert obs.round_number == 0
        assert obs.total_rounds == 1
        assert obs.available_actions == [COOPERATE, DEFECT]
        assert obs.history == []
        assert "payoff_matrix" in obs.game_state

    def test_observe_game_state_content(self) -> None:
        game = PrisonersDilemma()
        game.reset()
        obs = game.observe("player_0")

        gs = obs.game_state
        assert gs["your_role"] == "player_0"
        assert gs["payoff_matrix"]["mutual_cooperation"] == 3.0
        assert gs["payoff_matrix"]["mutual_defection"] == 1.0
        assert gs["payoff_matrix"]["temptation"] == 5.0
        assert gs["payoff_matrix"]["sucker"] == 0.0

    def test_to_prompt_contains_key_info(self) -> None:
        config = PDConfig(num_rounds=3)
        game = PrisonersDilemma(config)
        game.reset()
        game.step({"player_0": COOPERATE, "player_1": DEFECT})

        obs = game.observe("player_0")
        prompt = obs.to_prompt()

        assert "player_0" in prompt
        assert "cooperate" in prompt
        assert "defect" in prompt
        assert "Round 1 of 3" in prompt
        assert "History:" in prompt

    def test_observe_after_rounds(self) -> None:
        config = PDConfig(num_rounds=5)
        game = PrisonersDilemma(config)
        game.reset()

        game.step({"player_0": COOPERATE, "player_1": COOPERATE})
        game.step({"player_0": DEFECT, "player_1": COOPERATE})

        obs = game.observe("player_1")
        assert obs.round_number == 2
        assert len(obs.history) == 2
        assert obs.history[0].payoffs["player_1"] == 3.0
        assert obs.history[1].payoffs["player_1"] == 0.0


class TestPrisonersDilemmaSerialization:
    """Tests for observation serialization."""

    def test_observation_to_dict(self) -> None:
        game = PrisonersDilemma()
        game.reset()
        obs = game.observe("player_0")
        d = obs.to_dict()

        assert d["player_id"] == "player_0"
        assert d["round_number"] == 0
        assert d["total_rounds"] == 1
        assert d["available_actions"] == [COOPERATE, DEFECT]

    def test_observation_roundtrip(self) -> None:
        config = PDConfig(num_rounds=3)
        game = PrisonersDilemma(config)
        game.reset()
        game.step({"player_0": COOPERATE, "player_1": DEFECT})

        obs = game.observe("player_0")
        d = obs.to_dict()
        from game_envs.core.state import Observation

        restored = Observation.from_dict(d)
        assert restored.player_id == obs.player_id
        assert restored.round_number == obs.round_number
        assert len(restored.history) == len(obs.history)
