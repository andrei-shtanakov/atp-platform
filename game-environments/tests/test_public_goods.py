"""Tests for the Public Goods Game."""

from __future__ import annotations

import pytest

from game_envs.core.action import ContinuousActionSpace
from game_envs.core.game import GameType, MoveOrder
from game_envs.core.state import Observation
from game_envs.games.public_goods import PGConfig, PGStage, PublicGoodsGame
from game_envs.games.registry import GameRegistry


class TestPGConfig:
    """Tests for PGConfig validation."""

    def test_defaults(self) -> None:
        config = PGConfig()
        assert config.endowment == 20.0
        assert config.multiplier == 1.6
        assert config.punishment_cost == 0.0
        assert config.punishment_effect == 0.0
        assert config.num_players == 2

    def test_custom_values(self) -> None:
        config = PGConfig(
            num_players=4,
            endowment=100.0,
            multiplier=2.0,
            punishment_cost=1.0,
            punishment_effect=3.0,
        )
        assert config.num_players == 4
        assert config.endowment == 100.0
        assert config.multiplier == 2.0

    def test_endowment_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="endowment must be positive"):
            PGConfig(endowment=0.0)

    def test_endowment_negative(self) -> None:
        with pytest.raises(ValueError, match="endowment must be positive"):
            PGConfig(endowment=-5.0)

    def test_multiplier_must_exceed_one(self) -> None:
        with pytest.raises(ValueError, match="multiplier must be > 1"):
            PGConfig(multiplier=1.0)

    def test_multiplier_below_one(self) -> None:
        with pytest.raises(ValueError, match="multiplier must be > 1"):
            PGConfig(multiplier=0.5)

    def test_multiplier_must_be_less_than_num_players(self) -> None:
        with pytest.raises(ValueError, match="multiplier.*must be < num_players"):
            PGConfig(num_players=2, multiplier=2.0)

    def test_multiplier_equal_num_players(self) -> None:
        with pytest.raises(ValueError, match="multiplier.*must be < num_players"):
            PGConfig(num_players=3, multiplier=3.0)

    def test_punishment_cost_negative(self) -> None:
        with pytest.raises(ValueError, match="punishment_cost must be >= 0"):
            PGConfig(punishment_cost=-1.0)

    def test_punishment_effect_negative(self) -> None:
        with pytest.raises(ValueError, match="punishment_effect must be >= 0"):
            PGConfig(punishment_effect=-1.0)

    def test_inherits_gameconfig_validation(self) -> None:
        with pytest.raises(ValueError, match="num_rounds must be >= 1"):
            PGConfig(num_rounds=0)
        with pytest.raises(ValueError, match="discount_factor"):
            PGConfig(discount_factor=1.5)


class TestPublicGoodsGameProperties:
    """Tests for game properties."""

    def test_basic_properties(self) -> None:
        game = PublicGoodsGame()
        assert game.name == "Public Goods Game"
        assert game.game_type == GameType.NORMAL_FORM
        assert game.move_order == MoveOrder.SIMULTANEOUS
        assert game.player_ids == ["player_0", "player_1"]

    def test_repeated_name(self) -> None:
        config = PGConfig(num_rounds=5)
        game = PublicGoodsGame(config)
        assert "repeated x5" in game.name
        assert game.game_type == GameType.REPEATED

    def test_punishment_name(self) -> None:
        config = PGConfig(
            num_players=3,
            multiplier=1.5,
            punishment_cost=1.0,
            punishment_effect=3.0,
        )
        game = PublicGoodsGame(config)
        assert "with punishment" in game.name

    def test_punishment_repeated_name(self) -> None:
        config = PGConfig(
            num_players=3,
            multiplier=1.5,
            num_rounds=3,
            punishment_cost=1.0,
            punishment_effect=3.0,
        )
        game = PublicGoodsGame(config)
        assert "with punishment" in game.name
        assert "repeated x3" in game.name

    def test_n_player_ids(self) -> None:
        config = PGConfig(num_players=5, multiplier=4.0)
        game = PublicGoodsGame(config)
        assert len(game.player_ids) == 5
        assert game.player_ids == [
            "player_0",
            "player_1",
            "player_2",
            "player_3",
            "player_4",
        ]

    def test_max_players(self) -> None:
        config = PGConfig(num_players=20, multiplier=19.0)
        game = PublicGoodsGame(config)
        assert len(game.player_ids) == 20

    def test_too_many_players(self) -> None:
        config = PGConfig(num_players=21, multiplier=20.0)
        with pytest.raises(ValueError, match="between 2 and 20"):
            PublicGoodsGame(config)

    def test_too_few_players_config(self) -> None:
        """num_players=1 fails multiplier < num_players check."""
        with pytest.raises(ValueError):
            PGConfig(num_players=1, multiplier=1.5)

    def test_action_space_continuous(self) -> None:
        game = PublicGoodsGame()
        space = game.action_space("player_0")
        assert isinstance(space, ContinuousActionSpace)
        assert space.low == 0.0
        assert space.high == 20.0

    def test_action_space_contains(self) -> None:
        game = PublicGoodsGame()
        space = game.action_space("player_0")
        assert space.contains(0.0)
        assert space.contains(10.0)
        assert space.contains(20.0)
        assert not space.contains(-1.0)
        assert not space.contains(21.0)
        assert not space.contains("invalid")


class TestPublicGoodsGameOneShot:
    """Tests for one-shot public goods game payoffs."""

    def test_all_contribute_full(self) -> None:
        """Full contribution: each gets endowment * multiplier / n.

        With default n=2, e=20, m=1.6:
        payoff = 20 - 20 + 1.6 * 40 / 2 = 0 + 32 = 32.
        """
        game = PublicGoodsGame()
        game.reset()
        result = game.step({"player_0": 20.0, "player_1": 20.0})
        assert result.payoffs["player_0"] == pytest.approx(32.0)
        assert result.payoffs["player_1"] == pytest.approx(32.0)
        assert result.is_terminal

    def test_all_free_ride(self) -> None:
        """Zero contribution: each keeps endowment, pool = 0.

        payoff = 20 - 0 + 1.6 * 0 / 2 = 20.
        """
        game = PublicGoodsGame()
        game.reset()
        result = game.step({"player_0": 0.0, "player_1": 0.0})
        assert result.payoffs["player_0"] == pytest.approx(20.0)
        assert result.payoffs["player_1"] == pytest.approx(20.0)

    def test_one_contributes_one_free_rides(self) -> None:
        """Free rider exploits contributor.

        Player 0 contributes 20, Player 1 contributes 0:
        pool = 20, share = 1.6 * 20 / 2 = 16.
        p0 = 20 - 20 + 16 = 16.
        p1 = 20 - 0 + 16 = 36.
        """
        game = PublicGoodsGame()
        game.reset()
        result = game.step({"player_0": 20.0, "player_1": 0.0})
        assert result.payoffs["player_0"] == pytest.approx(16.0)
        assert result.payoffs["player_1"] == pytest.approx(36.0)

    def test_free_riding_dominates_one_shot(self) -> None:
        """Dominant strategy: free riding > cooperation.

        Regardless of other's action, contributing 0 always
        gives a higher individual payoff.
        """
        game = PublicGoodsGame()

        # Case 1: other cooperates
        game.reset()
        coop = game.step({"player_0": 20.0, "player_1": 20.0})
        game.reset()
        defect = game.step({"player_0": 0.0, "player_1": 20.0})
        assert defect.payoffs["player_0"] > coop.payoffs["player_0"]

        # Case 2: other free rides
        game.reset()
        coop2 = game.step({"player_0": 20.0, "player_1": 0.0})
        game.reset()
        defect2 = game.step({"player_0": 0.0, "player_1": 0.0})
        assert defect2.payoffs["player_0"] > coop2.payoffs["player_0"]

    def test_social_optimum_is_full_contribution(self) -> None:
        """Social optimum: total payoff maximized at full contribution.

        Full: total = 2 * 32 = 64.
        Zero: total = 2 * 20 = 40.
        """
        game = PublicGoodsGame()

        game.reset()
        full = game.step({"player_0": 20.0, "player_1": 20.0})
        total_full = sum(full.payoffs.values())

        game.reset()
        zero = game.step({"player_0": 0.0, "player_1": 0.0})
        total_zero = sum(zero.payoffs.values())

        assert total_full > total_zero

    def test_partial_contribution(self) -> None:
        """Partial contributions produce expected payoffs.

        p0=10, p1=5: pool=15, share=1.6*15/2=12.
        p0 = 20 - 10 + 12 = 22.
        p1 = 20 - 5 + 12 = 27.
        """
        game = PublicGoodsGame()
        game.reset()
        result = game.step({"player_0": 10.0, "player_1": 5.0})
        assert result.payoffs["player_0"] == pytest.approx(22.0)
        assert result.payoffs["player_1"] == pytest.approx(27.0)

    def test_payoff_sum_check(self) -> None:
        """Total payoff = n*e + (m-1)*sum(contributions).

        For n=2, e=20, m=1.6, contributions=(10, 5):
        total = 2*20 + (1.6-1)*15 = 40 + 9 = 49.
        """
        config = PGConfig(num_players=2, multiplier=1.6)
        game = PublicGoodsGame(config)
        game.reset()
        result = game.step({"player_0": 10.0, "player_1": 5.0})
        total = sum(result.payoffs.values())
        expected = 2 * 20 + (1.6 - 1) * 15
        assert total == pytest.approx(expected)


class TestPublicGoodsGameNPlayer:
    """Tests for n-player public goods game."""

    def test_four_players(self) -> None:
        """4-player game with equal contributions.

        n=4, e=20, m=2.0, each contributes 10.
        pool = 40, share = 2.0 * 40 / 4 = 20.
        payoff = 20 - 10 + 20 = 30 for each.
        """
        config = PGConfig(num_players=4, multiplier=2.0)
        game = PublicGoodsGame(config)
        game.reset()
        actions = {f"player_{i}": 10.0 for i in range(4)}
        result = game.step(actions)
        for pid in game.player_ids:
            assert result.payoffs[pid] == pytest.approx(30.0)

    def test_n_player_free_rider_advantage(self) -> None:
        """In n-player game, free rider gets more than cooperators."""
        config = PGConfig(num_players=4, multiplier=2.0)
        game = PublicGoodsGame(config)
        game.reset()
        actions = {
            "player_0": 0.0,  # free rider
            "player_1": 20.0,
            "player_2": 20.0,
            "player_3": 20.0,
        }
        result = game.step(actions)
        # Free rider gets endowment + share
        assert result.payoffs["player_0"] > result.payoffs["player_1"]

    def test_n_player_sum_check(self) -> None:
        """Total payoff = n*e + (m-1)*sum(contributions).

        n=5, e=20, m=3.0, each contributes 15.
        total_contributions = 75.
        total = 5*20 + (3-1)*75 = 100 + 150 = 250.
        """
        config = PGConfig(num_players=5, multiplier=3.0)
        game = PublicGoodsGame(config)
        game.reset()
        actions = {f"player_{i}": 15.0 for i in range(5)}
        result = game.step(actions)
        total = sum(result.payoffs.values())
        expected = 5 * 20 + (3.0 - 1) * 75
        assert total == pytest.approx(expected)

    def test_ten_players(self) -> None:
        """10-player game functions correctly."""
        config = PGConfig(num_players=10, multiplier=5.0)
        game = PublicGoodsGame(config)
        game.reset()
        actions = {f"player_{i}": 10.0 for i in range(10)}
        result = game.step(actions)
        # pool=100, share=5*100/10=50
        # payoff = 20 - 10 + 50 = 60
        for pid in game.player_ids:
            assert result.payoffs[pid] == pytest.approx(60.0)


class TestPublicGoodsGameRepeated:
    """Tests for repeated public goods game."""

    def test_multi_round(self) -> None:
        config = PGConfig(num_rounds=3, seed=42)
        game = PublicGoodsGame(config)
        game.reset()

        for i in range(3):
            result = game.step({"player_0": 10.0, "player_1": 10.0})
            if i < 2:
                assert not result.is_terminal
            else:
                assert result.is_terminal

    def test_step_after_terminal_raises(self) -> None:
        game = PublicGoodsGame()
        game.reset()
        game.step({"player_0": 10.0, "player_1": 10.0})
        with pytest.raises(RuntimeError, match="terminal"):
            game.step({"player_0": 10.0, "player_1": 10.0})

    def test_cumulative_payoffs(self) -> None:
        """Cumulative payoffs sum correctly over rounds."""
        config = PGConfig(num_rounds=3)
        game = PublicGoodsGame(config)
        game.reset()

        for _ in range(3):
            game.step({"player_0": 10.0, "player_1": 10.0})

        # Each round: share = 1.6 * 20 / 2 = 16
        # payoff = 20 - 10 + 16 = 26 per round
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(78.0)  # 26*3
        assert payoffs["player_1"] == pytest.approx(78.0)

    def test_discount_factor(self) -> None:
        """Discount factor reduces future payoffs."""
        config = PGConfig(num_rounds=3, discount_factor=0.5)
        game = PublicGoodsGame(config)
        game.reset()

        for _ in range(3):
            game.step({"player_0": 10.0, "player_1": 10.0})

        # Round payoff = 26 each time
        # Round 0: 26 * 0.5^0 = 26
        # Round 1: 26 * 0.5^1 = 13
        # Round 2: 26 * 0.5^2 = 6.5
        # Total = 45.5
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(45.5)

    def test_history_tracking(self) -> None:
        config = PGConfig(num_rounds=2)
        game = PublicGoodsGame(config)
        game.reset()

        game.step({"player_0": 5.0, "player_1": 15.0})
        game.step({"player_0": 10.0, "player_1": 10.0})

        history = game.history.for_player("player_0")
        assert len(history) == 2
        assert history[0].round_number == 1
        assert history[0].actions["player_0"] == 5.0
        assert history[1].round_number == 2

    def test_reset_clears_state(self) -> None:
        game = PublicGoodsGame()
        game.reset()
        game.step({"player_0": 10.0, "player_1": 10.0})

        game.reset()
        assert game.current_round == 0
        assert not game.is_terminal
        assert game.get_payoffs() == {
            "player_0": 0.0,
            "player_1": 0.0,
        }
        assert len(game.history.for_player("player_0")) == 0


class TestPublicGoodsGamePunishment:
    """Tests for the punishment variant (2-stage)."""

    def _make_game(self, num_players: int = 3, num_rounds: int = 1) -> PublicGoodsGame:
        config = PGConfig(
            num_players=num_players,
            multiplier=1.5,
            punishment_cost=1.0,
            punishment_effect=3.0,
            num_rounds=num_rounds,
        )
        return PublicGoodsGame(config)

    def test_two_stages(self) -> None:
        """Punishment variant has contribute then punish stage."""
        game = self._make_game()
        game.reset()

        # Stage 1: contribute
        result1 = game.step({"player_0": 10.0, "player_1": 10.0, "player_2": 10.0})
        assert not result1.is_terminal
        assert result1.info.get("stage") == PGStage.PUNISH

        # Stage 2: punish (no punishment)
        result2 = game.step({"player_0": 0.0, "player_1": 0.0, "player_2": 0.0})
        assert result2.is_terminal

    def test_no_punishment_payoff(self) -> None:
        """Without punishment, payoffs are the same as basic.

        n=3, e=20, m=1.5, each contributes 10.
        pool = 30, share = 1.5 * 30 / 3 = 15.
        payoff = 20 - 10 + 15 = 25 each.
        """
        game = self._make_game()
        game.reset()
        game.step({"player_0": 10.0, "player_1": 10.0, "player_2": 10.0})
        result = game.step({"player_0": 0.0, "player_1": 0.0, "player_2": 0.0})
        for pid in game.player_ids:
            assert result.payoffs[pid] == pytest.approx(25.0)

    def test_punishment_reduces_payoff(self) -> None:
        """Punishment costs the punisher and hurts the target.

        p0 spends 6 on punishment, distributed to p1 and p2.
        punishment_cost=1, punishment_effect=3.
        p0: base=25, cost=6*1=6, received=0 -> 19.
        p1: base=25, cost=0, received=3*3=9 -> 16.
        p2: base=25, cost=0, received=3*3=9 -> 16.
        """
        game = self._make_game()
        game.reset()
        game.step({"player_0": 10.0, "player_1": 10.0, "player_2": 10.0})
        result = game.step({"player_0": 6.0, "player_1": 0.0, "player_2": 0.0})
        assert result.payoffs["player_0"] == pytest.approx(19.0)
        assert result.payoffs["player_1"] == pytest.approx(16.0)
        assert result.payoffs["player_2"] == pytest.approx(16.0)

    def test_mutual_punishment(self) -> None:
        """All players punish each other."""
        game = self._make_game()
        game.reset()
        game.step({"player_0": 10.0, "player_1": 10.0, "player_2": 10.0})
        # Each spends 4 on punishment, distributed to 2 others
        result = game.step({"player_0": 4.0, "player_1": 4.0, "player_2": 4.0})
        # base = 25
        # cost = 4 * 1 = 4
        # received = 2 * (4/2) * 3 = 2 * 2 * 3 = 12
        # payoff = 25 - 4 - 12 = 9
        for pid in game.player_ids:
            assert result.payoffs[pid] == pytest.approx(9.0)

    def test_punishment_combined_actions_in_history(self) -> None:
        """History records both contribution and punishment."""
        game = self._make_game()
        game.reset()
        game.step({"player_0": 10.0, "player_1": 10.0, "player_2": 10.0})
        game.step({"player_0": 2.0, "player_1": 0.0, "player_2": 0.0})
        history = game.history.for_player("player_0")
        assert len(history) == 1
        actions = history[0].actions["player_0"]
        assert actions["contribution"] == 10.0
        assert actions["punishment_spent"] == 2.0

    def test_punishment_multi_round(self) -> None:
        """Punishment variant works over multiple rounds."""
        game = self._make_game(num_rounds=2)
        game.reset()

        # Round 1
        game.step({"player_0": 10.0, "player_1": 10.0, "player_2": 10.0})
        result1 = game.step({"player_0": 0.0, "player_1": 0.0, "player_2": 0.0})
        assert not result1.is_terminal

        # Round 2
        game.step({"player_0": 10.0, "player_1": 10.0, "player_2": 10.0})
        result2 = game.step({"player_0": 0.0, "player_1": 0.0, "player_2": 0.0})
        assert result2.is_terminal


class TestPublicGoodsGameObserve:
    """Tests for observe and to_prompt."""

    def test_observe_basic(self) -> None:
        game = PublicGoodsGame()
        game.reset()
        obs = game.observe("player_0")
        assert isinstance(obs, Observation)
        assert obs.player_id == "player_0"
        assert obs.round_number == 0
        assert obs.total_rounds == 1
        assert obs.game_state["num_players"] == 2
        assert obs.game_state["endowment"] == 20.0
        assert obs.game_state["multiplier"] == 1.6

    def test_observe_has_payoff_formula(self) -> None:
        game = PublicGoodsGame()
        game.reset()
        obs = game.observe("player_0")
        assert "payoff_formula" in obs.game_state
        assert "endowment - contribution" in obs.game_state["payoff_formula"]

    def test_to_prompt_content(self) -> None:
        game = PublicGoodsGame()
        game.reset()
        obs = game.observe("player_0")
        prompt = obs.to_prompt()
        assert "player_0" in prompt
        assert "Round 0 of 1" in prompt
        assert "[0.0, 20.0]" in prompt

    def test_to_prompt_with_history(self) -> None:
        config = PGConfig(num_rounds=3)
        game = PublicGoodsGame(config)
        game.reset()
        game.step({"player_0": 10.0, "player_1": 5.0})

        obs = game.observe("player_0")
        prompt = obs.to_prompt()
        assert "History:" in prompt
        assert "Round 1" in prompt

    def test_observe_punishment_shows_stage(self) -> None:
        config = PGConfig(
            num_players=3,
            multiplier=1.5,
            punishment_cost=1.0,
            punishment_effect=3.0,
        )
        game = PublicGoodsGame(config)
        game.reset()
        obs = game.observe("player_0")
        assert obs.game_state["stage"] == PGStage.CONTRIBUTE

    def test_observe_punishment_stage_shows_contributions(
        self,
    ) -> None:
        config = PGConfig(
            num_players=3,
            multiplier=1.5,
            punishment_cost=1.0,
            punishment_effect=3.0,
        )
        game = PublicGoodsGame(config)
        game.reset()
        game.step({"player_0": 10.0, "player_1": 5.0, "player_2": 15.0})
        obs = game.observe("player_0")
        assert obs.game_state["stage"] == PGStage.PUNISH
        assert obs.game_state["contributions"]["player_0"] == 10.0

    def test_serialization_roundtrip(self) -> None:
        game = PublicGoodsGame()
        game.reset()
        obs = game.observe("player_0")
        d = obs.to_dict()
        restored = Observation.from_dict(d)
        assert restored.player_id == obs.player_id
        assert restored.round_number == obs.round_number
        assert restored.total_rounds == obs.total_rounds
        assert restored.game_state == obs.game_state


class TestPublicGoodsGameRegistry:
    """Tests for game registry integration."""

    def setup_method(self) -> None:
        self._saved = dict(GameRegistry._registry)

    def teardown_method(self) -> None:
        GameRegistry._registry = self._saved

    def test_registered(self) -> None:
        assert "public_goods" in GameRegistry.list_games()

    def test_create_from_registry(self) -> None:
        game = GameRegistry.create("public_goods")
        assert isinstance(game, PublicGoodsGame)
        assert game.name == "Public Goods Game"

    def test_create_with_config(self) -> None:
        config = PGConfig(num_players=4, multiplier=2.0)
        game = GameRegistry.create("public_goods", config=config)
        assert isinstance(game, PublicGoodsGame)
        assert len(game.player_ids) == 4
