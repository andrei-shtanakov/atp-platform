"""Tests for tournament modes: round-robin and elimination."""

import pytest
from game_envs.games.prisoners_dilemma import (
    PDConfig,
    PrisonersDilemma,
)
from game_envs.strategies.pd_strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
)

from atp_games.models import GameRunConfig
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.suites.tournament import (
    MatchResult,
    Standing,
    run_double_elimination,
    run_round_robin,
    run_single_elimination,
)


class TestRoundRobin:
    """Test round-robin tournament."""

    @pytest.mark.anyio
    async def test_two_agents(self) -> None:
        """Two agents play one match."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_round_robin(game, agents, config)

        assert result.mode == "round_robin"
        assert len(result.matches) == 1
        assert len(result.standings) == 2

        # AllD should beat AllC
        assert result.standings[0].agent == "alld"
        assert result.standings[0].wins == 1
        assert result.standings[0].losses == 0
        assert result.standings[1].agent == "allc"
        assert result.standings[1].wins == 0
        assert result.standings[1].losses == 1

    @pytest.mark.anyio
    async def test_four_agents_round_robin(self) -> None:
        """4-agent round-robin: 6 matches total (C(4,2))."""
        game = PrisonersDilemma(PDConfig(num_rounds=10))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
            "alld2": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_round_robin(game, agents, config)

        assert result.mode == "round_robin"
        assert len(result.matches) == 6  # C(4,2) = 6
        assert len(result.standings) == 4

        # All agents should have played 3 matches each
        for standing in result.standings:
            assert standing.matches_played == 3

    @pytest.mark.anyio
    async def test_three_agents_odd(self) -> None:
        """3 agents (odd number): 3 matches, no explicit bye needed."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
        }
        config = GameRunConfig(episodes=2, base_seed=42)

        result = await run_round_robin(game, agents, config)

        assert len(result.matches) == 3  # C(3,2) = 3
        for standing in result.standings:
            assert standing.matches_played == 2

    @pytest.mark.anyio
    async def test_standings_sorted_by_points(self) -> None:
        """Standings should be sorted by points then payoff."""
        game = PrisonersDilemma(PDConfig(num_rounds=10))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
        }
        config = GameRunConfig(episodes=5, base_seed=42)

        result = await run_round_robin(game, agents, config)

        # Standings are sorted by points descending
        for i in range(len(result.standings) - 1):
            s1 = result.standings[i]
            s2 = result.standings[i + 1]
            assert (s1.points, s1.total_payoff) >= (
                s2.points,
                s2.total_payoff,
            )

    @pytest.mark.anyio
    async def test_match_result_has_game_result(self) -> None:
        """Each match should contain full game result."""
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=2, base_seed=42)

        result = await run_round_robin(game, agents, config)

        match = result.matches[0]
        assert match.game_result is not None
        assert match.game_result.num_episodes == 2
        assert match.agent_a in {"allc", "alld"}
        assert match.agent_b in {"allc", "alld"}

    @pytest.mark.anyio
    async def test_serialization(self) -> None:
        """Tournament result should serialize to dict."""
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1, base_seed=42)

        result = await run_round_robin(game, agents, config)
        d = result.to_dict()

        assert d["mode"] == "round_robin"
        assert len(d["standings"]) == 2
        assert len(d["matches"]) == 1
        assert "wins" in d["standings"][0]
        assert "score_a" in d["matches"][0]


class TestSingleElimination:
    """Test single-elimination tournament."""

    @pytest.mark.anyio
    async def test_two_agents(self) -> None:
        """Two agents: one match, one winner."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_single_elimination(game, agents, config)

        assert result.mode == "single_elimination"
        assert len(result.matches) == 1
        assert result.bracket is not None
        assert len(result.bracket) == 1  # One round

    @pytest.mark.anyio
    async def test_four_agents_bracket(self) -> None:
        """4 agents: 2 rounds, 3 matches total."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
            "alld2": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_single_elimination(game, agents, config)

        assert result.mode == "single_elimination"
        assert len(result.matches) == 3
        assert result.bracket is not None
        assert len(result.bracket) == 2  # Two rounds

    @pytest.mark.anyio
    async def test_three_agents_with_bye(self) -> None:
        """3 agents: one gets a bye in first round."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_single_elimination(game, agents, config)

        assert result.mode == "single_elimination"
        # 3 agents padded to 4, but one bye pair → 2 matches total
        assert len(result.matches) == 2

    @pytest.mark.anyio
    async def test_seeding_order(self) -> None:
        """Seeding order affects bracket placement."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
            "alld2": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)

        # Custom seeding
        result = await run_single_elimination(
            game,
            agents,
            config,
            seeding=["alld", "allc", "tft", "alld2"],
        )

        assert result.mode == "single_elimination"
        # First match should be alld vs allc based on seeding
        first_match = result.bracket[0][0]  # type: ignore[index]
        assert {first_match.agent_a, first_match.agent_b} == {
            "alld",
            "allc",
        }

    @pytest.mark.anyio
    async def test_too_few_agents(self) -> None:
        """Raise error for fewer than 2 agents."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {"only": BuiltinAdapter(AlwaysCooperate())}

        with pytest.raises(ValueError, match="at least 2"):
            await run_single_elimination(game, agents)

    @pytest.mark.anyio
    async def test_bracket_serialization(self) -> None:
        """Bracket should serialize properly."""
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1, base_seed=42)

        result = await run_single_elimination(game, agents, config)
        d = result.to_dict()

        assert "bracket" in d
        assert d["bracket"] is not None
        assert len(d["bracket"]) >= 1


class TestDoubleElimination:
    """Test double-elimination tournament."""

    @pytest.mark.anyio
    async def test_two_agents(self) -> None:
        """Two agents: losers get second chance."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_double_elimination(game, agents, config)

        assert result.mode == "double_elimination"
        assert len(result.standings) == 2
        # At least 2 matches in double elimination
        assert len(result.matches) >= 2

    @pytest.mark.anyio
    async def test_four_agents(self) -> None:
        """4-agent double elimination."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
            "alld2": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=2, base_seed=42)

        result = await run_double_elimination(game, agents, config)

        assert result.mode == "double_elimination"
        assert len(result.standings) == 4
        assert result.bracket is not None

    @pytest.mark.anyio
    async def test_too_few_agents(self) -> None:
        """Raise error for fewer than 2 agents."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {"only": BuiltinAdapter(AlwaysCooperate())}

        with pytest.raises(ValueError, match="at least 2"):
            await run_double_elimination(game, agents)


class TestStanding:
    """Test Standing dataclass."""

    def test_points_calculation(self) -> None:
        """Points: 3*wins + 1*draws."""
        s = Standing(agent="test", wins=2, draws=1, losses=1)
        assert s.points == 7.0

    def test_zero_points(self) -> None:
        """No wins or draws gives 0 points."""
        s = Standing(agent="test", losses=3)
        assert s.points == 0.0

    def test_serialization(self) -> None:
        """Standing serializes correctly."""
        s = Standing(
            agent="test",
            wins=1,
            losses=2,
            draws=1,
            total_payoff=5.5,
            matches_played=4,
        )
        d = s.to_dict()
        assert d["agent"] == "test"
        assert d["wins"] == 1
        assert d["losses"] == 2
        assert d["draws"] == 1
        assert d["points"] == 4.0
        assert d["total_payoff"] == 5.5
        assert d["matches_played"] == 4


class TestMatchResult:
    """Test MatchResult serialization."""

    @pytest.mark.anyio
    async def test_to_dict(self) -> None:
        """MatchResult serializes correctly."""
        from atp_games.models import GameResult

        gr = GameResult(
            game_name="test",
            config=GameRunConfig(episodes=1),
        )
        mr = MatchResult(
            agent_a="a",
            agent_b="b",
            game_result=gr,
            winner="a",
            score_a=3.0,
            score_b=1.0,
        )
        d = mr.to_dict()
        assert d["agent_a"] == "a"
        assert d["agent_b"] == "b"
        assert d["winner"] == "a"
        assert d["score_a"] == 3.0
        assert d["score_b"] == 1.0
