"""Tests for GameRegistry and register_game decorator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from game_envs.core.action import DiscreteActionSpace
from game_envs.core.game import Game, GameConfig, GameType, MoveOrder
from game_envs.core.state import GameState, StepResult
from game_envs.games.auction import Auction, AuctionConfig
from game_envs.games.colonel_blotto import BlottoConfig, ColonelBlotto
from game_envs.games.congestion import CongestionConfig, CongestionGame
from game_envs.games.prisoners_dilemma import (
    PDConfig,
    PrisonersDilemma,
)
from game_envs.games.public_goods import PGConfig, PublicGoodsGame
from game_envs.games.registry import GameRegistry, register_game


class _DummyGame(Game):
    """Minimal game for registry tests."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def game_type(self) -> GameType:
        return GameType.NORMAL_FORM

    @property
    def move_order(self) -> MoveOrder:
        return MoveOrder.SIMULTANEOUS

    @property
    def player_ids(self) -> list[str]:
        return ["p0"]

    def action_space(self, player_id: str) -> DiscreteActionSpace:
        return DiscreteActionSpace(["a"])

    def reset(self) -> StepResult:
        return StepResult(
            state=GameState(0, {}, {}),
            observations={},
            payoffs={},
            is_terminal=False,
        )

    def step(self, actions: dict[str, Any]) -> StepResult:
        return self.reset()

    def get_payoffs(self) -> dict[str, float]:
        return {}

    @property
    def is_terminal(self) -> bool:
        return False


@dataclass(frozen=True)
class _DummyConfig(GameConfig):
    """Config for dummy game with extra field."""

    difficulty: str = "easy"


class _ConfigurableDummy(_DummyGame):
    """Dummy game that uses _DummyConfig."""

    def __init__(self, config: _DummyConfig | None = None) -> None:
        super().__init__(config or _DummyConfig())


class TestGameRegistry:
    """Tests for the GameRegistry."""

    def setup_method(self) -> None:
        """Save and clear registry for each test."""
        self._saved_registry = dict(GameRegistry._registry)
        self._saved_configs = dict(GameRegistry._config_classes)
        GameRegistry.clear()

    def teardown_method(self) -> None:
        """Restore registry after each test."""
        GameRegistry._registry = self._saved_registry
        GameRegistry._config_classes = self._saved_configs

    def test_register_and_get(self) -> None:
        GameRegistry.register("dummy", _DummyGame)
        assert GameRegistry.get("dummy") is _DummyGame

    def test_register_duplicate_raises(self) -> None:
        GameRegistry.register("dummy", _DummyGame)
        with pytest.raises(ValueError, match="already registered"):
            GameRegistry.register("dummy", _DummyGame)

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown game"):
            GameRegistry.get("nonexistent")

    def test_get_unknown_lists_available(self) -> None:
        """Error message lists available games."""
        GameRegistry.register("alpha", _DummyGame)
        GameRegistry.register("beta", _DummyGame)
        with pytest.raises(KeyError, match="alpha, beta"):
            GameRegistry.get("missing")

    def test_create(self) -> None:
        GameRegistry.register("dummy", _DummyGame)
        game = GameRegistry.create("dummy")
        assert isinstance(game, _DummyGame)

    def test_create_with_config(self) -> None:
        GameRegistry.register("pd", PrisonersDilemma, PDConfig)
        config = PDConfig(num_rounds=5)
        game = GameRegistry.create("pd", config=config)
        assert isinstance(game, PrisonersDilemma)
        assert game.config.num_rounds == 5

    def test_create_with_dict_config(self) -> None:
        """Factory accepts dict config for YAML deserialization."""
        GameRegistry.register("pd", PrisonersDilemma, PDConfig)
        game = GameRegistry.create("pd", config={"num_rounds": 3})
        assert isinstance(game, PrisonersDilemma)
        assert game.config.num_rounds == 3

    def test_create_with_dict_config_custom_fields(self) -> None:
        """Dict config passes extra fields to config class."""
        GameRegistry.register(
            "configurable",
            _ConfigurableDummy,
            _DummyConfig,
        )
        game = GameRegistry.create(
            "configurable",
            config={"difficulty": "hard", "num_rounds": 2},
        )
        assert isinstance(game, _ConfigurableDummy)
        assert game.config.num_rounds == 2  # type: ignore[union-attr]

    def test_create_with_empty_dict_uses_defaults(self) -> None:
        """Empty dict config uses default values."""
        GameRegistry.register("pd", PrisonersDilemma, PDConfig)
        game = GameRegistry.create("pd", config={})
        assert isinstance(game, PrisonersDilemma)
        assert game.config.num_rounds == 1

    def test_create_with_none_config(self) -> None:
        """None config uses game's default."""
        GameRegistry.register("dummy", _DummyGame)
        game = GameRegistry.create("dummy", config=None)
        assert isinstance(game, _DummyGame)

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown game"):
            GameRegistry.create("nonexistent")

    def test_list_games(self) -> None:
        GameRegistry.register("b_game", _DummyGame)
        GameRegistry.register("a_game", _DummyGame)
        assert GameRegistry.list_games() == ["a_game", "b_game"]

    def test_list_games_empty(self) -> None:
        assert GameRegistry.list_games() == []

    def test_list_games_with_metadata(self) -> None:
        """list_games(with_metadata=True) returns dicts."""
        GameRegistry.register("dummy", _DummyGame)
        result = GameRegistry.list_games(with_metadata=True)
        assert len(result) == 1
        entry = result[0]
        assert entry["name"] == "dummy"
        assert "description" in entry
        assert entry["game_type"] == "normal_form"
        assert entry["move_order"] == "simultaneous"
        assert entry["num_players"] == 1

    def test_list_games_with_metadata_sorted(self) -> None:
        """Metadata list is sorted by name."""
        GameRegistry.register("zz_game", _DummyGame)
        GameRegistry.register("aa_game", _DummyGame)
        result = GameRegistry.list_games(with_metadata=True)
        names = [e["name"] for e in result]
        assert names == ["aa_game", "zz_game"]

    def test_list_games_with_metadata_empty(self) -> None:
        """Empty registry returns empty metadata list."""
        result = GameRegistry.list_games(with_metadata=True)
        assert result == []

    def test_clear(self) -> None:
        GameRegistry.register("dummy", _DummyGame, _DummyConfig)
        GameRegistry.clear()
        assert GameRegistry.list_games() == []
        assert GameRegistry._config_classes == {}

    def test_register_with_config_class(self) -> None:
        """Config class is stored for dict-based creation."""
        GameRegistry.register(
            "configurable",
            _ConfigurableDummy,
            _DummyConfig,
        )
        assert GameRegistry._config_classes["configurable"] is _DummyConfig

    def test_register_without_config_class_defaults(self) -> None:
        """Without config_class, defaults to GameConfig."""
        GameRegistry.register("dummy", _DummyGame)
        assert GameRegistry._config_classes["dummy"] is GameConfig


class TestGameInfo:
    """Tests for GameRegistry.game_info()."""

    def setup_method(self) -> None:
        self._saved_registry = dict(GameRegistry._registry)
        self._saved_configs = dict(GameRegistry._config_classes)
        GameRegistry.clear()

    def teardown_method(self) -> None:
        GameRegistry._registry = self._saved_registry
        GameRegistry._config_classes = self._saved_configs

    def test_game_info_basic(self) -> None:
        """game_info returns expected keys."""
        GameRegistry.register("dummy", _DummyGame)
        info = GameRegistry.game_info("dummy")
        assert info["name"] == "dummy"
        assert "description" in info
        assert info["game_type"] == "normal_form"
        assert info["move_order"] == "simultaneous"
        assert info["player_ids"] == ["p0"]
        assert "p0" in info["action_spaces"]
        assert "config_schema" in info

    def test_game_info_description(self) -> None:
        """Description comes from class docstring."""
        GameRegistry.register("dummy", _DummyGame)
        info = GameRegistry.game_info("dummy")
        assert info["description"] == ("Minimal game for registry tests.")

    def test_game_info_action_spaces(self) -> None:
        """Action space descriptions are populated."""
        GameRegistry.register("dummy", _DummyGame)
        info = GameRegistry.game_info("dummy")
        assert "Choose one of: a" in info["action_spaces"]["p0"]

    def test_game_info_config_schema(self) -> None:
        """Config schema includes fields with defaults."""
        GameRegistry.register(
            "configurable",
            _ConfigurableDummy,
            _DummyConfig,
        )
        info = GameRegistry.game_info("configurable")
        schema = info["config_schema"]
        assert "difficulty" in schema
        assert schema["difficulty"]["default"] == "easy"
        assert "num_players" in schema
        assert schema["num_players"]["default"] == 2

    def test_game_info_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown game"):
            GameRegistry.game_info("nonexistent")

    def test_game_info_prisoners_dilemma(self) -> None:
        """game_info works for real PD game."""
        GameRegistry.register("prisoners_dilemma", PrisonersDilemma, PDConfig)
        info = GameRegistry.game_info("prisoners_dilemma")
        assert info["name"] == "Prisoner's Dilemma"
        assert info["move_order"] == "simultaneous"
        assert len(info["player_ids"]) == 2
        assert "player_0" in info["action_spaces"]
        assert "player_1" in info["action_spaces"]
        schema = info["config_schema"]
        assert "reward" in schema
        assert "temptation" in schema

    def test_game_info_auction(self) -> None:
        """game_info works for Auction game."""
        GameRegistry.register("auction", Auction, AuctionConfig)
        info = GameRegistry.game_info("auction")
        assert "Auction" in info["name"]
        assert info["game_type"] == "normal_form"
        schema = info["config_schema"]
        assert "auction_type" in schema
        assert "min_bid" in schema

    def test_game_info_public_goods(self) -> None:
        """game_info works for Public Goods game."""
        GameRegistry.register("public_goods", PublicGoodsGame, PGConfig)
        info = GameRegistry.game_info("public_goods")
        assert "Public Goods" in info["name"]
        schema = info["config_schema"]
        assert "endowment" in schema
        assert "multiplier" in schema


class TestRegisterGameDecorator:
    """Tests for the @register_game decorator."""

    def setup_method(self) -> None:
        self._saved_registry = dict(GameRegistry._registry)
        self._saved_configs = dict(GameRegistry._config_classes)
        GameRegistry.clear()

    def teardown_method(self) -> None:
        GameRegistry._registry = self._saved_registry
        GameRegistry._config_classes = self._saved_configs

    def test_decorator_registers_game(self) -> None:
        """Decorator registers the class in the registry."""

        @register_game("test_game")
        class TestGame(_DummyGame):
            pass

        assert GameRegistry.get("test_game") is TestGame

    def test_decorator_returns_class_unchanged(self) -> None:
        """Decorator returns the original class."""

        @register_game("test_game")
        class TestGame(_DummyGame):
            pass

        game = TestGame()
        assert isinstance(game, _DummyGame)

    def test_decorator_with_config_class(self) -> None:
        """Decorator accepts config_class parameter."""

        @register_game("test_game", _DummyConfig)
        class TestGame(_ConfigurableDummy):
            pass

        assert GameRegistry._config_classes["test_game"] is _DummyConfig
        game = GameRegistry.create("test_game", config={"difficulty": "hard"})
        assert isinstance(game, TestGame)

    def test_decorator_without_config_class(self) -> None:
        """Decorator defaults config_class to GameConfig."""

        @register_game("test_game")
        class TestGame(_DummyGame):
            pass

        assert GameRegistry._config_classes["test_game"] is GameConfig

    def test_decorator_duplicate_raises(self) -> None:
        """Applying decorator twice with same name raises."""

        @register_game("test_game")
        class TestGame1(_DummyGame):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_game("test_game")
            class TestGame2(_DummyGame):
                pass

    def test_decorator_multiple_games(self) -> None:
        """Multiple games can be registered with decorator."""

        @register_game("game_a")
        class GameA(_DummyGame):
            pass

        @register_game("game_b")
        class GameB(_DummyGame):
            pass

        assert GameRegistry.list_games() == ["game_a", "game_b"]
        assert GameRegistry.get("game_a") is GameA
        assert GameRegistry.get("game_b") is GameB


class TestBuiltinRegistration:
    """Verify built-in games are registered on import."""

    def test_prisoners_dilemma_registered(self) -> None:
        import game_envs  # noqa: F401

        assert "prisoners_dilemma" in GameRegistry.list_games()
        assert GameRegistry.get("prisoners_dilemma") is PrisonersDilemma

    def test_auction_registered(self) -> None:
        import game_envs  # noqa: F401

        assert "auction" in GameRegistry.list_games()
        assert GameRegistry.get("auction") is Auction

    def test_public_goods_registered(self) -> None:
        import game_envs  # noqa: F401

        assert "public_goods" in GameRegistry.list_games()
        assert GameRegistry.get("public_goods") is PublicGoodsGame

    def test_colonel_blotto_registered(self) -> None:
        import game_envs  # noqa: F401

        assert "colonel_blotto" in GameRegistry.list_games()
        assert GameRegistry.get("colonel_blotto") is ColonelBlotto

    def test_congestion_registered(self) -> None:
        import game_envs  # noqa: F401

        assert "congestion" in GameRegistry.list_games()
        assert GameRegistry.get("congestion") is CongestionGame

    def test_all_builtins_listed(self) -> None:
        import game_envs  # noqa: F401

        games = GameRegistry.list_games()
        assert "auction" in games
        assert "colonel_blotto" in games
        assert "congestion" in games
        assert "prisoners_dilemma" in games
        assert "public_goods" in games

    def test_builtin_config_classes_registered(self) -> None:
        """Built-in games have their config classes registered."""
        import game_envs  # noqa: F401

        assert GameRegistry._config_classes["prisoners_dilemma"] is PDConfig
        assert GameRegistry._config_classes["auction"] is AuctionConfig
        assert GameRegistry._config_classes["public_goods"] is PGConfig
        assert GameRegistry._config_classes["colonel_blotto"] is BlottoConfig
        assert GameRegistry._config_classes["congestion"] is CongestionConfig

    def test_create_pd_from_dict(self) -> None:
        """Create PD game from dict config."""
        import game_envs  # noqa: F401

        game = GameRegistry.create(
            "prisoners_dilemma",
            config={"num_rounds": 10},
        )
        assert isinstance(game, PrisonersDilemma)
        assert game.config.num_rounds == 10

    def test_create_auction_from_dict(self) -> None:
        """Create Auction game from dict config."""
        import game_envs  # noqa: F401

        game = GameRegistry.create(
            "auction",
            config={
                "num_players": 3,
                "auction_type": "second_price",
            },
        )
        assert isinstance(game, Auction)
        assert game.config.num_players == 3

    def test_create_blotto_from_dict(self) -> None:
        """Create Colonel Blotto game from dict config."""
        import game_envs  # noqa: F401

        game = GameRegistry.create(
            "colonel_blotto",
            config={"num_battlefields": 5, "total_troops": 50},
        )
        assert isinstance(game, ColonelBlotto)
        assert game.config.num_battlefields == 5  # type: ignore[union-attr]

    def test_create_congestion_from_dict(self) -> None:
        """Create Congestion game from dict config."""
        import game_envs  # noqa: F401

        game = GameRegistry.create("congestion", config={})
        assert isinstance(game, CongestionGame)

    def test_list_builtins_with_metadata(self) -> None:
        """list_games(with_metadata=True) for builtins."""
        import game_envs  # noqa: F401

        result = GameRegistry.list_games(with_metadata=True)
        assert len(result) == 5
        names = [e["name"] for e in result]
        assert "prisoners_dilemma" in names
        assert "auction" in names
        for entry in result:
            assert "description" in entry
            assert "game_type" in entry
            assert "move_order" in entry
            assert "num_players" in entry

    def test_game_info_for_builtins(self) -> None:
        """game_info works for all built-in games."""
        import game_envs  # noqa: F401

        for name in GameRegistry.list_games():
            info = GameRegistry.game_info(name)
            assert info["name"]
            assert info["game_type"]
            assert info["player_ids"]
            assert info["config_schema"]
