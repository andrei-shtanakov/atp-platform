"""Tests for Game ABC and GameConfig."""

from __future__ import annotations

import pytest

from game_envs.core.game import GameConfig, GameType, MoveOrder


class TestGameConfig:
    def test_defaults(self) -> None:
        c = GameConfig()
        assert c.num_players == 2
        assert c.num_rounds == 1
        assert c.discount_factor == 1.0
        assert c.noise == 0.0
        assert c.communication is False
        assert c.seed is None

    def test_custom(self) -> None:
        c = GameConfig(
            num_players=4,
            num_rounds=10,
            discount_factor=0.95,
            noise=0.05,
            communication=True,
            seed=42,
        )
        assert c.num_players == 4
        assert c.num_rounds == 10
        assert c.discount_factor == 0.95
        assert c.noise == 0.05
        assert c.communication is True
        assert c.seed == 42

    def test_frozen(self) -> None:
        c = GameConfig()
        with pytest.raises(AttributeError):
            c.num_players = 5  # type: ignore[misc]

    def test_invalid_num_players(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            GameConfig(num_players=0)
        with pytest.raises(ValueError, match="num_players"):
            GameConfig(num_players=-1)

    def test_invalid_num_rounds(self) -> None:
        with pytest.raises(ValueError, match="num_rounds"):
            GameConfig(num_rounds=0)

    def test_invalid_discount_factor(self) -> None:
        with pytest.raises(ValueError, match="discount_factor"):
            GameConfig(discount_factor=-0.1)
        with pytest.raises(ValueError, match="discount_factor"):
            GameConfig(discount_factor=1.1)

    def test_invalid_noise(self) -> None:
        with pytest.raises(ValueError, match="noise"):
            GameConfig(noise=-0.1)
        with pytest.raises(ValueError, match="noise"):
            GameConfig(noise=1.1)

    def test_boundary_values(self) -> None:
        # These should all be valid
        GameConfig(discount_factor=0.0)
        GameConfig(discount_factor=1.0)
        GameConfig(noise=0.0)
        GameConfig(noise=1.0)
        GameConfig(num_players=1)
        GameConfig(num_rounds=1)


class TestGameType:
    def test_values(self) -> None:
        assert GameType.NORMAL_FORM == "normal_form"
        assert GameType.EXTENSIVE_FORM == "extensive_form"
        assert GameType.REPEATED == "repeated"
        assert GameType.STOCHASTIC == "stochastic"


class TestMoveOrder:
    def test_values(self) -> None:
        assert MoveOrder.SIMULTANEOUS == "simultaneous"
        assert MoveOrder.SEQUENTIAL == "sequential"


class TestStubGame:
    def test_properties(self, stub_game) -> None:  # type: ignore[no-untyped-def]
        assert stub_game.name == "stub_game"
        assert stub_game.game_type == GameType.NORMAL_FORM
        assert stub_game.move_order == MoveOrder.SIMULTANEOUS
        assert len(stub_game.player_ids) == 2

    def test_reset(self, stub_game) -> None:  # type: ignore[no-untyped-def]
        result = stub_game.reset()
        assert not result.is_terminal
        assert stub_game.current_round == 0
        assert len(stub_game.history) == 0

    def test_step(self, stub_game) -> None:  # type: ignore[no-untyped-def]
        stub_game.reset()
        actions = {pid: "A" for pid in stub_game.player_ids}
        result = stub_game.step(actions)
        assert stub_game.current_round == 1
        assert result.payoffs[stub_game.player_ids[0]] == 1.0
        assert len(stub_game.history) == 1

    def test_full_game(self, stub_game) -> None:  # type: ignore[no-untyped-def]
        stub_game.reset()
        while not stub_game.is_terminal:
            actions = {pid: "A" for pid in stub_game.player_ids}
            stub_game.step(actions)
        payoffs = stub_game.get_payoffs()
        for pid in stub_game.player_ids:
            assert payoffs[pid] == 3.0  # 3 rounds * 1.0

    def test_observe(self, stub_game) -> None:  # type: ignore[no-untyped-def]
        stub_game.reset()
        pid = stub_game.player_ids[0]
        obs = stub_game.observe(pid)
        assert obs.player_id == pid
        assert obs.round_number == 0
        assert obs.total_rounds == 3

    def test_action_space(self, stub_game) -> None:  # type: ignore[no-untyped-def]
        pid = stub_game.player_ids[0]
        space = stub_game.action_space(pid)
        assert space.contains("A")
        assert space.contains("B")
        assert not space.contains("C")

    def test_reset_clears_state(self, stub_game) -> None:  # type: ignore[no-untyped-def]
        stub_game.reset()
        stub_game.step({pid: "A" for pid in stub_game.player_ids})
        assert len(stub_game.history) == 1
        stub_game.reset()
        assert len(stub_game.history) == 0
        assert stub_game.current_round == 0
