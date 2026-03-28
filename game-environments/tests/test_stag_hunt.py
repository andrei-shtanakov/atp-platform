"""Tests for Stag Hunt game implementation."""

from __future__ import annotations

import pytest

from game_envs.games.registry import GameRegistry
from game_envs.games.stag_hunt import HARE, STAG, SHConfig, StagHunt


class TestSHConfig:
    """Tests for SHConfig validation."""

    def test_defaults(self) -> None:
        c = SHConfig()
        assert c.mutual_stag == 4.0
        assert c.mutual_hare == 3.0
        assert c.hare == 3.0
        assert c.sucker == 0.0
        assert c.num_players == 2
        assert c.num_rounds == 1

    def test_mutual_stag_greater_than_hare(self) -> None:
        with pytest.raises(ValueError, match="mutual_stag > hare"):
            SHConfig(mutual_stag=2.0, hare=3.0)

    def test_hare_greater_than_sucker(self) -> None:
        with pytest.raises(ValueError, match="hare > sucker"):
            SHConfig(hare=0.0, sucker=1.0)

    def test_mutual_stag_greater_than_mutual_hare(self) -> None:
        # mutual_stag=5.0 > hare=4.0 > sucker=0.0, but mutual_stag(5) < mutual_hare(6)
        with pytest.raises(ValueError, match="mutual_stag > mutual_hare"):
            SHConfig(mutual_stag=5.0, mutual_hare=6.0, hare=4.0, sucker=0.0)


class TestStagHuntPayoffs:
    """Tests for Stag Hunt payoff matrix."""

    def test_mutual_stag(self) -> None:
        """Both stag -> (4.0, 4.0)."""
        game = StagHunt()
        game.reset()
        result = game.step({"player_0": STAG, "player_1": STAG})
        assert result.payoffs["player_0"] == pytest.approx(4.0)
        assert result.payoffs["player_1"] == pytest.approx(4.0)

    def test_mutual_hare(self) -> None:
        """Both hare -> (3.0, 3.0)."""
        game = StagHunt()
        game.reset()
        result = game.step({"player_0": HARE, "player_1": HARE})
        assert result.payoffs["player_0"] == pytest.approx(3.0)
        assert result.payoffs["player_1"] == pytest.approx(3.0)

    def test_stag_vs_hare(self) -> None:
        """Stag/hare -> (0.0, 3.0)."""
        game = StagHunt()
        game.reset()
        result = game.step({"player_0": STAG, "player_1": HARE})
        assert result.payoffs["player_0"] == pytest.approx(0.0)
        assert result.payoffs["player_1"] == pytest.approx(3.0)

    def test_hare_vs_stag(self) -> None:
        """Hare/stag -> (3.0, 0.0)."""
        game = StagHunt()
        game.reset()
        result = game.step({"player_0": HARE, "player_1": STAG})
        assert result.payoffs["player_0"] == pytest.approx(3.0)
        assert result.payoffs["player_1"] == pytest.approx(0.0)

    def test_payoff_ordering(self) -> None:
        """mutual_stag > hare > sucker, mutual_stag > mutual_hare."""
        c = SHConfig()
        assert c.mutual_stag > c.hare
        assert c.hare > c.sucker
        assert c.mutual_stag > c.mutual_hare


class TestStagHuntNashEquilibria:
    """Tests for Nash equilibria properties."""

    def test_two_pure_nash_equilibria(self) -> None:
        """Both (stag, stag) and (hare, hare) are Nash equilibria."""
        c = SHConfig()
        # (stag, stag) is NE: deviating to hare gives hare(3) < mutual_stag(4)
        assert c.hare < c.mutual_stag, "Deviating from stag/stag should not help"
        # (hare, hare) is NE: deviating to stag gives sucker(0) < mutual_hare(3)
        assert c.sucker < c.mutual_hare, "Deviating from hare/hare should not help"


class TestStagHuntRepeated:
    """Tests for repeated game mechanics."""

    def test_multi_round_accumulates(self) -> None:
        """3 rounds of mutual stag = 12.0 cumulative."""
        game = StagHunt(SHConfig(num_rounds=3))
        game.reset()
        for _ in range(3):
            game.step({"player_0": STAG, "player_1": STAG})
        payoffs = game.get_payoffs()
        assert payoffs["player_0"] == pytest.approx(12.0)
        assert payoffs["player_1"] == pytest.approx(12.0)

    def test_single_round_terminal(self) -> None:
        """Single round game terminates after 1 step."""
        game = StagHunt()
        game.reset()
        result = game.step({"player_0": STAG, "player_1": STAG})
        assert result.is_terminal is True
        assert game.is_terminal is True

    def test_step_after_terminal_raises(self) -> None:
        """Stepping after terminal raises RuntimeError."""
        game = StagHunt()
        game.reset()
        game.step({"player_0": STAG, "player_1": STAG})
        with pytest.raises(RuntimeError, match="terminal"):
            game.step({"player_0": STAG, "player_1": STAG})


class TestStagHuntRegistry:
    """Tests for game registry integration."""

    def test_in_registry(self) -> None:
        """'stag_hunt' appears in GameRegistry.list_games()."""
        assert "stag_hunt" in GameRegistry.list_games()

    def test_registry_create(self) -> None:
        """GameRegistry.create('stag_hunt') returns a StagHunt instance."""
        game = GameRegistry.create("stag_hunt")
        assert isinstance(game, StagHunt)
