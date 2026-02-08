"""Tests for data models."""

import pytest

from atp_games.models import (
    EpisodeResult,
    GameResult,
    GameRunConfig,
)


class TestGameRunConfig:
    def test_defaults(self) -> None:
        config = GameRunConfig()
        assert config.episodes == 1
        assert config.max_retries == 3
        assert config.move_timeout == 30.0

    def test_custom_values(self) -> None:
        config = GameRunConfig(episodes=10, max_retries=5, move_timeout=60.0)
        assert config.episodes == 10
        assert config.max_retries == 5
        assert config.move_timeout == 60.0

    def test_invalid_episodes(self) -> None:
        with pytest.raises(ValueError, match="episodes"):
            GameRunConfig(episodes=0)

    def test_invalid_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            GameRunConfig(max_retries=-1)

    def test_invalid_move_timeout(self) -> None:
        with pytest.raises(ValueError, match="move_timeout"):
            GameRunConfig(move_timeout=0)

    def test_frozen(self) -> None:
        config = GameRunConfig()
        with pytest.raises(AttributeError):
            config.episodes = 5  # type: ignore[misc]


class TestEpisodeResult:
    def test_basic(self) -> None:
        ep = EpisodeResult(
            episode=0,
            payoffs={"p0": 3.0, "p1": 3.0},
        )
        assert ep.episode == 0
        assert ep.payoffs == {"p0": 3.0, "p1": 3.0}

    def test_roundtrip(self) -> None:
        ep = EpisodeResult(
            episode=1,
            payoffs={"p0": 5.0, "p1": 0.0},
            actions_log=[{"p0": "cooperate", "p1": "defect"}],
        )
        data = ep.to_dict()
        restored = EpisodeResult.from_dict(data)
        assert restored.episode == ep.episode
        assert restored.payoffs == ep.payoffs
        assert restored.actions_log == ep.actions_log


class TestGameResult:
    def test_average_payoffs(self) -> None:
        config = GameRunConfig(episodes=2)
        result = GameResult(
            game_name="Test Game",
            config=config,
            episodes=[
                EpisodeResult(
                    episode=0,
                    payoffs={"p0": 2.0, "p1": 4.0},
                ),
                EpisodeResult(
                    episode=1,
                    payoffs={"p0": 4.0, "p1": 6.0},
                ),
            ],
        )
        avg = result.average_payoffs
        assert avg["p0"] == pytest.approx(3.0)
        assert avg["p1"] == pytest.approx(5.0)

    def test_average_payoffs_empty(self) -> None:
        result = GameResult(
            game_name="Empty",
            config=GameRunConfig(),
        )
        assert result.average_payoffs == {}

    def test_num_episodes(self) -> None:
        result = GameResult(
            game_name="Test",
            config=GameRunConfig(),
            episodes=[
                EpisodeResult(episode=0, payoffs={"p0": 1.0}),
            ],
        )
        assert result.num_episodes == 1

    def test_roundtrip(self) -> None:
        config = GameRunConfig(episodes=1, max_retries=2, move_timeout=15.0)
        result = GameResult(
            game_name="PD",
            config=config,
            agent_names={"p0": "tft", "p1": "alld"},
        )
        data = result.to_dict()
        restored = GameResult.from_dict(data)
        assert restored.game_name == "PD"
        assert restored.config.max_retries == 2
        assert restored.agent_names == {"p0": "tft", "p1": "alld"}
