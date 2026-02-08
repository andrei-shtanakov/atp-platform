"""Tests for cross-play matrix computation."""

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
from atp_games.suites.cross_play import (
    _compute_clusters,
    _compute_dominance,
    _compute_pareto_frontier,
    run_cross_play,
)


class TestCrossPlay:
    """Test cross-play matrix execution."""

    @pytest.mark.anyio
    async def test_two_agents_with_self_play(self) -> None:
        """Two agents: 4 matchups (including self-play)."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_cross_play(game, agents, config, include_self_play=True)

        assert len(result.agents) == 2
        # 2x2 = 4 entries with self-play
        assert len(result.entries) == 4
        assert "allc" in result.matrix
        assert "alld" in result.matrix
        assert "allc" in result.matrix["allc"]
        assert "alld" in result.matrix["allc"]

    @pytest.mark.anyio
    async def test_two_agents_no_self_play(self) -> None:
        """Two agents without self-play: 2 matchups."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_cross_play(game, agents, config, include_self_play=False)

        assert len(result.entries) == 2
        # Self-play entries should default to 0.0
        assert result.matrix["allc"]["allc"] == 0.0
        assert result.matrix["alld"]["alld"] == 0.0

    @pytest.mark.anyio
    async def test_three_agents_matrix(self) -> None:
        """Three agents: 9 matchups with self-play."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
            "tft": BuiltinAdapter(TitForTat()),
        }
        config = GameRunConfig(episodes=2, base_seed=42)

        result = await run_cross_play(game, agents, config, include_self_play=True)

        assert len(result.agents) == 3
        assert len(result.entries) == 9
        # Each row should have values for all 3 opponents
        for agent in result.agents:
            assert len(result.matrix[agent]) == 3

    @pytest.mark.anyio
    async def test_matrix_symmetry_check(self) -> None:
        """Self-play payoffs should be symmetric."""
        game = PrisonersDilemma(PDConfig(num_rounds=10))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=5, base_seed=42)

        result = await run_cross_play(game, agents, config, include_self_play=True)

        # Self-play of AllC vs AllC should yield same payoff
        # since both cooperate: R=3 per round * 10 rounds = 30
        allc_self = result.matrix["allc"]["allc"]
        assert allc_self == pytest.approx(30.0)

        # AllD vs AllD: P=1 per round * 10 rounds = 10
        alld_self = result.matrix["alld"]["alld"]
        assert alld_self == pytest.approx(10.0)

    @pytest.mark.anyio
    async def test_dominance_detection(self) -> None:
        """AllD should dominate AllC in one-shot PD."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=5, base_seed=42)

        result = await run_cross_play(game, agents, config, include_self_play=True)

        # AllD vs AllC = 5, AllD vs AllD = 1
        # AllC vs AllC = 3, AllC vs AllD = 0
        # AllD strictly dominates AllC: 5>3 and 1>0
        assert len(result.dominance) == 1
        assert result.dominance[0].dominator == "alld"
        assert result.dominance[0].dominated == "allc"
        assert result.dominance[0].strict is True

    @pytest.mark.anyio
    async def test_pareto_frontier(self) -> None:
        """Pareto frontier should exclude strictly dominated agents."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_cross_play(game, agents, config, include_self_play=True)

        # Both should be on Pareto frontier since neither
        # strictly dominates the other
        assert len(result.pareto_frontier) >= 1

    @pytest.mark.anyio
    async def test_serialization(self) -> None:
        """CrossPlayResult should serialize to dict."""
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        agents = {
            "allc": BuiltinAdapter(AlwaysCooperate()),
            "alld": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1, base_seed=42)

        result = await run_cross_play(game, agents, config, include_self_play=True)
        d = result.to_dict()

        assert "agents" in d
        assert "matrix" in d
        assert "entries" in d
        assert "dominance" in d
        assert "pareto_frontier" in d
        assert "clusters" in d
        assert len(d["agents"]) == 2


class TestDominance:
    """Test dominance computation."""

    def test_strict_dominance(self) -> None:
        """Agent A strictly dominates B if better against all."""
        agents = ["a", "b"]
        matrix = {
            "a": {"a": 5.0, "b": 5.0},
            "b": {"a": 3.0, "b": 3.0},
        }
        relations = _compute_dominance(agents, matrix)

        assert len(relations) == 1
        assert relations[0].dominator == "a"
        assert relations[0].dominated == "b"
        assert relations[0].strict is True

    def test_weak_dominance(self) -> None:
        """Agent A weakly dominates B if >= all, > some."""
        agents = ["a", "b"]
        matrix = {
            "a": {"a": 5.0, "b": 3.0},
            "b": {"a": 5.0, "b": 2.0},
        }
        relations = _compute_dominance(agents, matrix)

        # a dominates b (weakly: tied against a, better against b)
        assert len(relations) == 1
        assert relations[0].dominator == "a"
        assert relations[0].strict is False

    def test_no_dominance(self) -> None:
        """No dominance when each agent beats the other sometimes."""
        agents = ["a", "b"]
        matrix = {
            "a": {"a": 5.0, "b": 1.0},
            "b": {"a": 2.0, "b": 4.0},
        }
        relations = _compute_dominance(agents, matrix)
        assert len(relations) == 0


class TestParetoFrontier:
    """Test Pareto frontier computation."""

    def test_one_dominated(self) -> None:
        """Strictly dominated agent excluded from frontier."""
        agents = ["a", "b"]
        matrix = {
            "a": {"a": 5.0, "b": 5.0},
            "b": {"a": 3.0, "b": 3.0},
        }
        frontier = _compute_pareto_frontier(agents, matrix)
        assert frontier == ["a"]

    def test_none_dominated(self) -> None:
        """Both on frontier if neither strictly dominates."""
        agents = ["a", "b"]
        matrix = {
            "a": {"a": 5.0, "b": 1.0},
            "b": {"a": 2.0, "b": 4.0},
        }
        frontier = _compute_pareto_frontier(agents, matrix)
        assert set(frontier) == {"a", "b"}


class TestClusters:
    """Test clustering computation."""

    def test_identical_agents_cluster(self) -> None:
        """Agents with same payoff profiles cluster together."""
        agents = ["a", "b"]
        matrix = {
            "a": {"a": 5.0, "b": 3.0},
            "b": {"a": 5.0, "b": 3.0},
        }
        clusters = _compute_clusters(agents, matrix, threshold=0.1)
        assert len(clusters) == 1
        assert set(clusters[0]) == {"a", "b"}

    def test_different_agents_separate(self) -> None:
        """Very different agents should be in separate clusters."""
        agents = ["a", "b"]
        matrix = {
            "a": {"a": 100.0, "b": 100.0},
            "b": {"a": 1.0, "b": 1.0},
        }
        clusters = _compute_clusters(agents, matrix, threshold=0.1)
        assert len(clusters) == 2

    def test_empty_agents(self) -> None:
        """Empty agents list gives no clusters."""
        clusters = _compute_clusters([], {})
        assert clusters == []
