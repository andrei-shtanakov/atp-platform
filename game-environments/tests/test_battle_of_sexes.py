"""Tests for Battle of the Sexes game implementation."""

from __future__ import annotations

import pytest

from game_envs.games.battle_of_sexes import A, B, BattleOfSexes, BoSConfig
from game_envs.games.registry import GameRegistry


class TestBoSConfig:
    """Tests for BoSConfig defaults and validation."""

    def test_defaults(self) -> None:
        c = BoSConfig()
        assert c.preferred_a == 3.0
        assert c.other_a == 2.0
        assert c.preferred_b == 3.0
        assert c.other_b == 2.0
        assert c.mismatch == 0.0
        assert c.num_players == 2
        assert c.num_rounds == 1


class TestBattleOfSexesPayoffs:
    """Tests for Battle of the Sexes payoff matrix."""

    def test_both_choose_A(self) -> None:
        """Both A -> player_0 gets preferred_a=3.0, player_1 gets other_a=2.0."""
        game = BattleOfSexes()
        game.reset()
        result = game.step({"player_0": A, "player_1": A})
        assert result.payoffs["player_0"] == pytest.approx(3.0)
        assert result.payoffs["player_1"] == pytest.approx(2.0)

    def test_both_choose_B(self) -> None:
        """Both B -> player_0 gets other_b=2.0, player_1 gets preferred_b=3.0."""
        game = BattleOfSexes()
        game.reset()
        result = game.step({"player_0": B, "player_1": B})
        assert result.payoffs["player_0"] == pytest.approx(2.0)
        assert result.payoffs["player_1"] == pytest.approx(3.0)

    def test_mismatch_AB(self) -> None:
        """A/B mismatch -> (0.0, 0.0)."""
        game = BattleOfSexes()
        game.reset()
        result = game.step({"player_0": A, "player_1": B})
        assert result.payoffs["player_0"] == pytest.approx(0.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)

    def test_mismatch_BA(self) -> None:
        """B/A mismatch -> (0.0, 0.0)."""
        game = BattleOfSexes()
        game.reset()
        result = game.step({"player_0": B, "player_1": A})
        assert result.payoffs["player_0"] == pytest.approx(0.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)

    def test_payoff_ordering(self) -> None:
        """preferred > other > mismatch for each player."""
        c = BoSConfig()
        assert c.preferred_a > c.other_a > c.mismatch
        assert c.preferred_b > c.other_b > c.mismatch


class TestBattleOfSexesRepeated:
    """Tests for repeated game mechanics."""

    def test_multi_round(self) -> None:
        """3 rounds of both A: player_0 accumulates 9.0, player_1 accumulates 6.0."""
        game = BattleOfSexes(BoSConfig(num_rounds=3))
        game.reset()
        for _ in range(3):
            game.step({"player_0": A, "player_1": A})
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(9.0)
        assert payoffs["player_1"] == pytest.approx(6.0)

    def test_single_round_terminal(self) -> None:
        """Single round game terminates after 1 step."""
        game = BattleOfSexes()
        game.reset()
        result = game.step({"player_0": A, "player_1": A})
        assert result.is_terminal is True
        assert game.is_terminal is True


class TestBattleOfSexesRegistry:
    """Tests for game registry integration."""

    def test_in_registry(self) -> None:
        """'battle_of_sexes' appears in GameRegistry.list_games()."""
        assert "battle_of_sexes" in GameRegistry.list_games()

    def test_registry_create(self) -> None:
        """GameRegistry.create('battle_of_sexes') returns a BattleOfSexes instance."""
        game = GameRegistry.create("battle_of_sexes")
        assert isinstance(game, BattleOfSexes)
