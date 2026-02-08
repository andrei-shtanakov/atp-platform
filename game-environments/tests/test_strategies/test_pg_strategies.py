"""Tests for Public Goods Game strategies."""

from __future__ import annotations

import pytest

from game_envs.core.state import Observation, RoundResult
from game_envs.games.public_goods import PGConfig, PublicGoodsGame
from game_envs.strategies.pg_strategies import (
    ConditionalCooperator,
    FreeRider,
    FullContributor,
    Punisher,
)


def _make_obs(
    player_id: str = "player_0",
    endowment: float = 20.0,
    history: list[RoundResult] | None = None,
    stage: str | None = None,
    contributions: dict[str, float] | None = None,
) -> Observation:
    """Create a minimal PG observation."""
    game_state: dict = {
        "game": "Public Goods Game",
        "your_role": player_id,
        "num_players": 4,
        "endowment": endowment,
        "multiplier": 1.6,
        "payoff_formula": (
            "payoff = endowment - contribution + multiplier * total_contributions / 4"
        ),
    }
    if stage is not None:
        game_state["stage"] = stage
    if contributions is not None:
        game_state["contributions"] = contributions
    return Observation(
        player_id=player_id,
        game_state=game_state,
        available_actions=["[0.0, 20.0]"],
        history=history or [],
        round_number=len(history) if history else 0,
        total_rounds=10,
    )


class TestFullContributor:
    def test_contributes_full_endowment(self) -> None:
        s = FullContributor()
        obs = _make_obs(endowment=20.0)
        assert s.choose_action(obs) == 20.0

    def test_respects_endowment_value(self) -> None:
        s = FullContributor()
        obs = _make_obs(endowment=50.0)
        assert s.choose_action(obs) == 50.0

    def test_name(self) -> None:
        assert FullContributor().name == "full_contributor"


class TestFreeRider:
    def test_contributes_nothing(self) -> None:
        s = FreeRider()
        assert s.choose_action(_make_obs()) == 0.0

    def test_name(self) -> None:
        assert FreeRider().name == "free_rider"


class TestConditionalCooperator:
    def test_full_contribution_first_round(self) -> None:
        s = ConditionalCooperator()
        assert s.choose_action(_make_obs()) == 20.0

    def test_matches_average(self) -> None:
        s = ConditionalCooperator()
        history = [
            RoundResult(
                round_number=1,
                actions={
                    "player_0": 20.0,
                    "player_1": 10.0,
                    "player_2": 10.0,
                    "player_3": 0.0,
                },
                payoffs={
                    "player_0": 16.0,
                    "player_1": 26.0,
                    "player_2": 26.0,
                    "player_3": 36.0,
                },
            )
        ]
        obs = _make_obs(history=history)
        # Others contributed: 10, 10, 0 -> avg = 20/3 ≈ 6.67
        action = s.choose_action(obs)
        assert action == pytest.approx(20.0 / 3.0, abs=0.01)

    def test_clamps_to_endowment(self) -> None:
        s = ConditionalCooperator()
        history = [
            RoundResult(
                round_number=1,
                actions={
                    "player_0": 5.0,
                    "player_1": 20.0,
                    "player_2": 20.0,
                    "player_3": 20.0,
                },
                payoffs={
                    "player_0": 41.0,
                    "player_1": 26.0,
                    "player_2": 26.0,
                    "player_3": 26.0,
                },
            )
        ]
        obs = _make_obs(history=history)
        action = s.choose_action(obs)
        assert action == 20.0  # Clamped to endowment

    def test_name(self) -> None:
        assert ConditionalCooperator().name == "conditional_cooperator"


class TestPunisher:
    def test_contributes_fully(self) -> None:
        s = Punisher()
        assert s.choose_action(_make_obs()) == 20.0

    def test_punishes_free_riders(self) -> None:
        s = Punisher()
        obs = _make_obs(
            stage="punish",
            contributions={
                "player_0": 20.0,
                "player_1": 5.0,  # < 50% of 20
                "player_2": 20.0,
                "player_3": 0.0,  # < 50% of 20
            },
        )
        punishment = s.choose_action(obs)
        # player_1: 10 - 5 = 5, player_3: 10 - 0 = 10
        # total = 15
        assert punishment == 15.0

    def test_no_punishment_when_all_cooperate(self) -> None:
        s = Punisher()
        obs = _make_obs(
            stage="punish",
            contributions={
                "player_0": 20.0,
                "player_1": 15.0,  # >= 50%
                "player_2": 20.0,
                "player_3": 12.0,  # >= 50%
            },
        )
        assert s.choose_action(obs) == 0.0

    def test_contributes_in_contribute_stage(self) -> None:
        s = Punisher()
        obs = _make_obs(stage="contribute")
        assert s.choose_action(obs) == 20.0

    def test_name(self) -> None:
        assert Punisher().name == "punisher"


class TestPGIntegration:
    """Integration tests with actual PublicGoodsGame."""

    def test_free_rider_exploits_contributors(self) -> None:
        """Free rider gets higher payoff than full contributor."""
        game = PublicGoodsGame(PGConfig(num_players=4, num_rounds=1))
        game.reset()
        strategies = {
            "player_0": FullContributor(),
            "player_1": FullContributor(),
            "player_2": FullContributor(),
            "player_3": FreeRider(),
        }
        actions = {}
        for pid, strat in strategies.items():
            obs = game.observe(pid)
            actions[pid] = strat.choose_action(obs)

        result = game.step(actions)
        # Free rider keeps endowment + share of pool
        # Full contributors: 0 + 1.6 * 60 / 4 = 24
        # Free rider: 20 + 1.6 * 60 / 4 = 44
        assert result.payoffs["player_3"] > result.payoffs["player_0"]

    def test_all_free_riders_nash(self) -> None:
        """All free riders is Nash: each gets only endowment."""
        game = PublicGoodsGame(PGConfig(num_players=4, num_rounds=1))
        game.reset()
        actions = {}
        fr = FreeRider()
        for pid in game.player_ids:
            obs = game.observe(pid)
            actions[pid] = fr.choose_action(obs)

        result = game.step(actions)
        # All contribute 0 -> pool = 0 -> each gets 20
        for pid in game.player_ids:
            assert result.payoffs[pid] == pytest.approx(20.0)
