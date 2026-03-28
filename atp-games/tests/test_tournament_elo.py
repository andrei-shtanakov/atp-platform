"""Tests for Elo rating integration in tournament runner."""

import pytest
from game_envs.games.prisoners_dilemma import PDConfig, PrisonersDilemma
from game_envs.strategies.pd_strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
)

from atp_games.models import GameRunConfig
from atp_games.rating.elo import EloCalculator
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.suites.tournament import run_round_robin


class TestRoundRobinWithElo:
    """Tests for Elo rating integration in round-robin tournaments."""

    @pytest.mark.anyio
    async def test_elo_ratings_populated_when_calculator_provided(self) -> None:
        """elo_ratings is a dict when elo_calculator is provided."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1, base_seed=42)
        calc = EloCalculator()

        result = await run_round_robin(game, agents, config, elo_calculator=calc)

        assert result.elo_ratings is not None
        assert isinstance(result.elo_ratings, dict)

    @pytest.mark.anyio
    async def test_elo_ratings_none_without_calculator(self) -> None:
        """elo_ratings is None when no elo_calculator is passed (backward compat)."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1, base_seed=42)

        result = await run_round_robin(game, agents, config)

        assert result.elo_ratings is None

    @pytest.mark.anyio
    async def test_all_agents_have_elo_ratings(self) -> None:
        """Every agent participating has an Elo rating entry."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
        }
        config = GameRunConfig(episodes=2, base_seed=42)
        calc = EloCalculator()

        result = await run_round_robin(game, agents, config, elo_calculator=calc)

        assert result.elo_ratings is not None
        for agent_name in agents:
            assert agent_name in result.elo_ratings

    @pytest.mark.anyio
    async def test_elo_games_played_matches_matches_played(self) -> None:
        """Each agent's elo games_played equals matches_played in standings."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
        }
        config = GameRunConfig(episodes=2, base_seed=42)
        calc = EloCalculator()

        result = await run_round_robin(game, agents, config, elo_calculator=calc)

        assert result.elo_ratings is not None
        standings_by_name = {s.agent: s for s in result.standings}
        for agent_name, elo in result.elo_ratings.items():
            expected = standings_by_name[agent_name].matches_played
            assert elo.games_played == expected, (
                f"{agent_name}: elo.games_played={elo.games_played}"
                f" != matches_played={expected}"
            )

    @pytest.mark.anyio
    async def test_elo_ratings_differ_after_unequal_tournament(self) -> None:
        """Ratings diverge after a tournament where agents differ in skill."""
        game = PrisonersDilemma(PDConfig(num_rounds=10))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)
        calc = EloCalculator()

        result = await run_round_robin(game, agents, config, elo_calculator=calc)

        assert result.elo_ratings is not None
        # AlwaysDefect dominates AlwaysCooperate, so ratings should differ
        alld_rating = result.elo_ratings["alld"].rating
        allc_rating = result.elo_ratings["allc"].rating
        assert alld_rating != allc_rating

    @pytest.mark.anyio
    async def test_elo_serialization_includes_ratings(self) -> None:
        """to_dict() includes elo_ratings when calculator was provided."""
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1, base_seed=42)
        calc = EloCalculator()

        result = await run_round_robin(game, agents, config, elo_calculator=calc)
        d = result.to_dict()

        assert "elo_ratings" in d
        assert "allc" in d["elo_ratings"]
        assert "alld" in d["elo_ratings"]
        assert "rating" in d["elo_ratings"]["allc"]
        assert "games_played" in d["elo_ratings"]["allc"]

    @pytest.mark.anyio
    async def test_no_elo_in_serialization_without_calculator(self) -> None:
        """to_dict() does not include elo_ratings when no calculator provided."""
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1, base_seed=42)

        result = await run_round_robin(game, agents, config)
        d = result.to_dict()

        assert "elo_ratings" not in d

    @pytest.mark.anyio
    async def test_custom_k_factor_affects_ratings(self) -> None:
        """Custom K-factor results in larger rating changes."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents_high = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        agents_low = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=2, base_seed=42)

        result_high = await run_round_robin(
            game, agents_high, config, elo_calculator=EloCalculator(k_factor=64)
        )
        result_low = await run_round_robin(
            PrisonersDilemma(PDConfig(num_rounds=5)),
            agents_low,
            config,
            elo_calculator=EloCalculator(k_factor=16),
        )

        assert result_high.elo_ratings is not None
        assert result_low.elo_ratings is not None
        change_high = abs(result_high.elo_ratings["alld"].rating - 1500.0)
        change_low = abs(result_low.elo_ratings["alld"].rating - 1500.0)
        assert change_high > change_low
