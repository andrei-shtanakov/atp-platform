"""Tests for multi-episode, concurrency, seeding, and statistics.

Covers:
- Episode-level parallelism (parallel vs sequential same results)
- Seed management (deterministic seeds)
- Aggregation: mean, std, 95% CI per player
- Welch's t-test and Bonferroni correction
- Progress reporting
- 50-episode aggregation
"""

from __future__ import annotations

import math

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

from atp_games.models import (
    AgentComparison,
    EpisodeResult,
    GameResult,
    GameRunConfig,
    _compute_player_stats,
    _t_critical_95,
    compare_agents,
    welchs_t_test,
)
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.runner.game_runner import (
    GameRunner,
    ProgressReporter,
    _make_game_for_episode,
)

# ------------------------------------------------------------------
# GameRunConfig: new fields
# ------------------------------------------------------------------


class TestGameRunConfigNewFields:
    """Test new parallel and base_seed fields."""

    def test_defaults(self) -> None:
        config = GameRunConfig()
        assert config.parallel == 1
        assert config.base_seed is None

    def test_custom_parallel(self) -> None:
        config = GameRunConfig(parallel=4)
        assert config.parallel == 4

    def test_invalid_parallel(self) -> None:
        with pytest.raises(ValueError, match="parallel"):
            GameRunConfig(parallel=0)

    def test_base_seed(self) -> None:
        config = GameRunConfig(base_seed=42)
        assert config.base_seed == 42

    def test_episode_seed_with_base(self) -> None:
        config = GameRunConfig(base_seed=100)
        assert config.episode_seed(0) == 100
        assert config.episode_seed(5) == 105
        assert config.episode_seed(49) == 149

    def test_episode_seed_without_base(self) -> None:
        config = GameRunConfig()
        assert config.episode_seed(0) is None
        assert config.episode_seed(10) is None


# ------------------------------------------------------------------
# EpisodeResult: seed field
# ------------------------------------------------------------------


class TestEpisodeResultSeed:
    """Test seed field on EpisodeResult."""

    def test_seed_stored(self) -> None:
        ep = EpisodeResult(
            episode=0,
            payoffs={"p0": 1.0},
            seed=42,
        )
        assert ep.seed == 42

    def test_seed_none_by_default(self) -> None:
        ep = EpisodeResult(episode=0, payoffs={"p0": 1.0})
        assert ep.seed is None

    def test_seed_serialization(self) -> None:
        ep = EpisodeResult(episode=0, payoffs={"p0": 1.0}, seed=42)
        data = ep.to_dict()
        assert data["seed"] == 42

    def test_seed_not_in_dict_when_none(self) -> None:
        ep = EpisodeResult(episode=0, payoffs={"p0": 1.0})
        data = ep.to_dict()
        assert "seed" not in data

    def test_seed_roundtrip(self) -> None:
        ep = EpisodeResult(episode=0, payoffs={"p0": 1.0}, seed=99)
        restored = EpisodeResult.from_dict(ep.to_dict())
        assert restored.seed == 99


# ------------------------------------------------------------------
# Seed management: deterministic results
# ------------------------------------------------------------------


class TestSeedManagement:
    """Test deterministic seeding across episodes."""

    def setup_method(self) -> None:
        self.runner = GameRunner()

    @pytest.mark.anyio
    async def test_same_seed_same_result(self) -> None:
        """Running with same base_seed gives identical results."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "player_0": BuiltinAdapter(TitForTat()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=10, base_seed=42)

        result1 = await self.runner.run_game(game, agents, config)
        result2 = await self.runner.run_game(game, agents, config)

        for ep1, ep2 in zip(result1.episodes, result2.episodes):
            assert ep1.payoffs == ep2.payoffs
            assert ep1.seed == ep2.seed

    @pytest.mark.anyio
    async def test_different_seeds_per_episode(self) -> None:
        """Each episode gets a unique seed."""
        config = GameRunConfig(episodes=5, base_seed=100)
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        result = await self.runner.run_game(game, agents, config)
        seeds = [ep.seed for ep in result.episodes]
        assert seeds == [100, 101, 102, 103, 104]


# ------------------------------------------------------------------
# Parallel vs Sequential: deterministic seeds give same results
# ------------------------------------------------------------------


class TestParallelVsSequential:
    """Verify parallel and sequential produce same results."""

    def setup_method(self) -> None:
        self.runner = GameRunner()

    @pytest.mark.anyio
    async def test_parallel_vs_sequential_deterministic(
        self,
    ) -> None:
        """Parallel and sequential with same seeds produce
        identical results."""
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        agents = {
            "player_0": BuiltinAdapter(TitForTat()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }

        seq_config = GameRunConfig(episodes=10, parallel=1, base_seed=42)
        par_config = GameRunConfig(episodes=10, parallel=4, base_seed=42)

        seq_result = await self.runner.run_game(game, agents, seq_config)
        par_result = await self.runner.run_game(game, agents, par_config)

        assert seq_result.num_episodes == par_result.num_episodes
        for seq_ep, par_ep in zip(seq_result.episodes, par_result.episodes):
            assert seq_ep.payoffs == par_ep.payoffs
            assert seq_ep.seed == par_ep.seed
            assert seq_ep.actions_log == par_ep.actions_log

    @pytest.mark.anyio
    async def test_parallel_execution(self) -> None:
        """Parallel execution completes all episodes."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysCooperate()),
        }
        config = GameRunConfig(episodes=20, parallel=5, base_seed=0)
        result = await self.runner.run_game(game, agents, config)
        assert result.num_episodes == 20
        # All episodes should have R=3 payoffs
        for ep in result.episodes:
            assert ep.payoffs["player_0"] == pytest.approx(3.0)


# ------------------------------------------------------------------
# 50-episode aggregation
# ------------------------------------------------------------------


class TestFiftyEpisodeAggregation:
    """Test aggregation with 50 episodes (default in spec)."""

    def setup_method(self) -> None:
        self.runner = GameRunner()

    @pytest.mark.anyio
    async def test_50_episodes_coop_vs_defect(self) -> None:
        """50 episodes of AllC vs AllD: deterministic payoffs."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=50, base_seed=0)
        result = await self.runner.run_game(game, agents, config)

        assert result.num_episodes == 50

        # AllC vs AllD: player_0 gets S=0, player_1 gets T=5
        avg = result.average_payoffs
        assert avg["player_0"] == pytest.approx(0.0)
        assert avg["player_1"] == pytest.approx(5.0)

        # Player statistics
        stats = result.player_statistics()
        assert "player_0" in stats
        assert "player_1" in stats

        # std should be 0 (deterministic game, no noise)
        assert stats["player_0"].std == pytest.approx(0.0)
        assert stats["player_1"].std == pytest.approx(0.0)

        # Mean matches expected
        assert stats["player_0"].mean == pytest.approx(0.0)
        assert stats["player_1"].mean == pytest.approx(5.0)

        # CI should be tight (zero variance)
        assert stats["player_0"].ci_lower == pytest.approx(0.0)
        assert stats["player_0"].ci_upper == pytest.approx(0.0)

    @pytest.mark.anyio
    async def test_50_episodes_with_noise(self) -> None:
        """50 episodes with noise: non-zero std, valid CI."""
        game = PrisonersDilemma(PDConfig(num_rounds=10, noise=0.1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=50, base_seed=123)
        result = await self.runner.run_game(game, agents, config)

        assert result.num_episodes == 50
        stats = result.player_statistics()

        # With noise, there should be some variance
        p0_stats = stats["player_0"]
        assert p0_stats.n_episodes == 50

        # CI should contain the mean
        assert p0_stats.ci_lower <= p0_stats.mean
        assert p0_stats.ci_upper >= p0_stats.mean

        # CI width should be reasonable
        ci_width = p0_stats.ci_upper - p0_stats.ci_lower
        assert ci_width >= 0


# ------------------------------------------------------------------
# PlayerStats / _compute_player_stats
# ------------------------------------------------------------------


class TestPlayerStats:
    """Test aggregation statistics computation."""

    def test_single_value(self) -> None:
        stats = _compute_player_stats("p0", [5.0])
        assert stats.mean == 5.0
        assert stats.std == 0.0
        assert stats.ci_lower == 5.0
        assert stats.ci_upper == 5.0
        assert stats.n_episodes == 1

    def test_two_values(self) -> None:
        stats = _compute_player_stats("p0", [2.0, 4.0])
        assert stats.mean == pytest.approx(3.0)
        assert stats.std > 0
        assert stats.ci_lower < stats.mean
        assert stats.ci_upper > stats.mean

    def test_constant_values(self) -> None:
        payoffs = [3.0] * 50
        stats = _compute_player_stats("p0", payoffs)
        assert stats.mean == pytest.approx(3.0)
        assert stats.std == pytest.approx(0.0)
        assert stats.min == pytest.approx(3.0)
        assert stats.max == pytest.approx(3.0)

    def test_known_statistics(self) -> None:
        """Test with known values for verification."""
        payoffs = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = _compute_player_stats("p0", payoffs)
        assert stats.mean == pytest.approx(3.0)
        expected_std = math.sqrt(sum((x - 3.0) ** 2 for x in payoffs) / 4)
        assert stats.std == pytest.approx(expected_std)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(5.0)
        assert stats.n_episodes == 5

    def test_ci_coverage(self) -> None:
        """95% CI should cover the true mean for known data."""
        # Generate data from a known distribution
        import random

        rng = random.Random(42)
        true_mean = 10.0
        payoffs = [true_mean + rng.gauss(0, 2) for _ in range(100)]
        stats = _compute_player_stats("p0", payoffs)
        # True mean should be within the CI
        # (with high probability for 100 samples)
        assert stats.ci_lower <= true_mean + 1.0
        assert stats.ci_upper >= true_mean - 1.0

    def test_serialization(self) -> None:
        stats = _compute_player_stats("p0", [1.0, 2.0, 3.0])
        data = stats.to_dict()
        assert data["player_id"] == "p0"
        assert "mean" in data
        assert "std" in data
        assert "ci_95" in data
        assert len(data["ci_95"]) == 2


# ------------------------------------------------------------------
# Welch's t-test
# ------------------------------------------------------------------


class TestWelchsTTest:
    """Test Welch's t-test implementation."""

    def test_identical_samples(self) -> None:
        """Identical means: t=0, p=1."""
        t, p = welchs_t_test(5.0, 1.0, 30, 5.0, 1.0, 30)
        assert t == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_very_different_means(self) -> None:
        """Very different means: small p-value."""
        t, p = welchs_t_test(10.0, 1.0, 50, 0.0, 1.0, 50)
        assert abs(t) > 2.0
        assert p < 0.05

    def test_small_sample(self) -> None:
        """n < 2: returns (0, 1)."""
        t, p = welchs_t_test(5.0, 1.0, 1, 3.0, 1.0, 30)
        assert t == 0.0
        assert p == 1.0

    def test_zero_variance(self) -> None:
        """Both stds = 0, same mean: no difference."""
        t, p = welchs_t_test(3.0, 0.0, 10, 3.0, 0.0, 10)
        assert t == 0.0
        assert p == 1.0

    def test_zero_variance_different_means(self) -> None:
        """Both stds = 0, different means: infinite t."""
        t, p = welchs_t_test(5.0, 0.0, 10, 3.0, 0.0, 10)
        assert t == float("inf")
        assert p == 0.0

    def test_unequal_sizes(self) -> None:
        """Unequal sample sizes should work."""
        t, p = welchs_t_test(5.0, 2.0, 10, 3.0, 1.0, 50)
        assert isinstance(t, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


# ------------------------------------------------------------------
# Bonferroni correction via compare_agents
# ------------------------------------------------------------------


class TestCompareAgents:
    """Test Welch's t-test with Bonferroni correction."""

    def _make_result(
        self,
        payoffs_list: list[dict[str, float]],
    ) -> GameResult:
        """Helper to create GameResult from payoff dicts."""
        episodes = [
            EpisodeResult(episode=i, payoffs=p) for i, p in enumerate(payoffs_list)
        ]
        return GameResult(
            game_name="Test",
            config=GameRunConfig(episodes=len(episodes)),
            episodes=episodes,
        )

    def test_two_agents_significant_difference(self) -> None:
        """Clear difference between agents is detected."""
        payoffs = [{"p0": 10.0, "p1": 0.0} for _ in range(50)]
        result = self._make_result(payoffs)
        comps = compare_agents(result)

        assert len(comps) == 1
        comp = comps[0]
        assert comp.player_a == "p0"
        assert comp.player_b == "p1"
        assert comp.mean_a == pytest.approx(10.0)
        assert comp.mean_b == pytest.approx(0.0)
        # With zero variance and diff means, should be significant
        assert comp.is_significant

    def test_two_agents_no_difference(self) -> None:
        """No difference is correctly detected as not significant."""
        payoffs = [{"p0": 3.0, "p1": 3.0} for _ in range(50)]
        result = self._make_result(payoffs)
        comps = compare_agents(result)

        assert len(comps) == 1
        assert not comps[0].is_significant

    def test_three_agents_bonferroni(self) -> None:
        """Three agents => 3 pairwise comparisons.

        Bonferroni correction should adjust p-values.
        """
        payoffs = [{"p0": 10.0, "p1": 5.0, "p2": 0.0} for _ in range(50)]
        result = self._make_result(payoffs)
        comps = compare_agents(result)

        # 3 choose 2 = 3 comparisons
        assert len(comps) == 3

        for comp in comps:
            # adjusted_p = p * n_comparisons (capped at 1.0)
            assert comp.adjusted_p_value >= comp.p_value
            assert comp.adjusted_p_value <= 1.0

    def test_empty_result(self) -> None:
        """No episodes => no comparisons."""
        result = GameResult(
            game_name="Empty",
            config=GameRunConfig(),
        )
        assert compare_agents(result) == []

    def test_single_player(self) -> None:
        """Single player => no pairwise comparisons."""
        payoffs = [{"p0": 5.0} for _ in range(10)]
        result = self._make_result(payoffs)
        assert compare_agents(result) == []

    def test_comparison_serialization(self) -> None:
        """AgentComparison serializes correctly."""
        comp = AgentComparison(
            player_a="p0",
            player_b="p1",
            metric="payoff",
            mean_a=10.0,
            mean_b=5.0,
            t_statistic=3.5,
            p_value=0.001,
            adjusted_p_value=0.003,
            is_significant=True,
        )
        data = comp.to_dict()
        assert data["player_a"] == "p0"
        assert data["is_significant"] is True
        assert data["adjusted_p_value"] == 0.003


# ------------------------------------------------------------------
# GameResult: player_statistics and agent_comparisons
# ------------------------------------------------------------------


class TestGameResultStatistics:
    """Test GameResult aggregation methods."""

    def test_player_statistics(self) -> None:
        result = GameResult(
            game_name="PD",
            config=GameRunConfig(episodes=5),
            episodes=[
                EpisodeResult(
                    episode=i,
                    payoffs={"p0": float(i), "p1": float(10 - i)},
                )
                for i in range(5)
            ],
        )
        stats = result.player_statistics()
        assert "p0" in stats
        assert "p1" in stats
        assert stats["p0"].n_episodes == 5
        assert stats["p0"].mean == pytest.approx(2.0)

    def test_player_payoffs(self) -> None:
        result = GameResult(
            game_name="PD",
            config=GameRunConfig(episodes=3),
            episodes=[
                EpisodeResult(episode=0, payoffs={"p0": 1.0, "p1": 2.0}),
                EpisodeResult(episode=1, payoffs={"p0": 3.0, "p1": 4.0}),
                EpisodeResult(episode=2, payoffs={"p0": 5.0, "p1": 6.0}),
            ],
        )
        assert result.player_payoffs("p0") == [1.0, 3.0, 5.0]
        assert result.player_payoffs("p1") == [2.0, 4.0, 6.0]

    def test_agent_comparisons(self) -> None:
        result = GameResult(
            game_name="PD",
            config=GameRunConfig(episodes=50),
            episodes=[
                EpisodeResult(
                    episode=i,
                    payoffs={"p0": 10.0, "p1": 0.0},
                )
                for i in range(50)
            ],
        )
        comps = result.agent_comparisons()
        assert len(comps) == 1
        assert comps[0].is_significant

    def test_to_dict_includes_statistics(self) -> None:
        result = GameResult(
            game_name="PD",
            config=GameRunConfig(episodes=3, base_seed=10),
            episodes=[
                EpisodeResult(
                    episode=i,
                    payoffs={"p0": 3.0, "p1": 1.0},
                    seed=10 + i,
                )
                for i in range(3)
            ],
        )
        data = result.to_dict()
        assert "player_statistics" in data
        assert "p0" in data["player_statistics"]
        assert data["config"]["base_seed"] == 10
        assert data["config"]["parallel"] == 1

    def test_roundtrip_with_new_fields(self) -> None:
        config = GameRunConfig(
            episodes=2,
            parallel=4,
            base_seed=42,
        )
        result = GameResult(
            game_name="PD",
            config=config,
            episodes=[
                EpisodeResult(
                    episode=0,
                    payoffs={"p0": 3.0},
                    seed=42,
                ),
                EpisodeResult(
                    episode=1,
                    payoffs={"p0": 1.0},
                    seed=43,
                ),
            ],
            agent_names={"p0": "tft"},
        )
        data = result.to_dict()
        restored = GameResult.from_dict(data)
        assert restored.config.parallel == 4
        assert restored.config.base_seed == 42
        assert restored.episodes[0].seed == 42


# ------------------------------------------------------------------
# t-critical lookup
# ------------------------------------------------------------------


class TestTCritical:
    """Test t-critical value lookup and interpolation."""

    def test_known_values(self) -> None:
        assert _t_critical_95(1) == pytest.approx(12.706)
        assert _t_critical_95(30) == pytest.approx(2.042)

    def test_large_df(self) -> None:
        assert _t_critical_95(100) == pytest.approx(1.96)
        assert _t_critical_95(1000) == pytest.approx(1.96)

    def test_zero_df(self) -> None:
        assert _t_critical_95(0) == float("inf")

    def test_interpolated_df(self) -> None:
        """Values between lookup keys should be interpolated."""
        t12 = _t_critical_95(12)
        assert 2.0 < t12 < 2.3  # Between df=10 and df=15


# ------------------------------------------------------------------
# ProgressReporter
# ------------------------------------------------------------------


class TestProgressReporter:
    """Test progress reporting."""

    @pytest.mark.anyio
    async def test_reports_all_episodes(self) -> None:
        reporter = ProgressReporter(total_episodes=5)
        for i in range(5):
            await reporter.report_complete(i)
        assert reporter.completed == 5

    @pytest.mark.anyio
    async def test_elapsed_time(self) -> None:
        reporter = ProgressReporter(total_episodes=1)
        assert reporter.elapsed >= 0.0
        await reporter.report_complete(0)
        assert reporter.elapsed >= 0.0


# ------------------------------------------------------------------
# _make_game_for_episode
# ------------------------------------------------------------------


class TestMakeGameForEpisode:
    """Test episode game creation with seeding."""

    def test_creates_independent_copy(self) -> None:
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        ep_game = _make_game_for_episode(game, seed=42)
        # Should be different objects
        assert ep_game is not game

    def test_applies_seed(self) -> None:
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        ep_game = _make_game_for_episode(game, seed=99)
        assert ep_game.config.seed == 99

    def test_none_seed_preserves_original(self) -> None:
        game = PrisonersDilemma(PDConfig(num_rounds=3, seed=42))
        ep_game = _make_game_for_episode(game, seed=None)
        # Original seed preserved via deep copy
        assert ep_game.config.seed == 42


# ------------------------------------------------------------------
# Integration: full pipeline with statistics
# ------------------------------------------------------------------


class TestIntegrationMultiEpisode:
    """End-to-end integration tests."""

    def setup_method(self) -> None:
        self.runner = GameRunner()

    @pytest.mark.anyio
    async def test_full_pipeline_50_episodes(self) -> None:
        """Run 50 episodes, compute stats, compare agents."""
        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(
            episodes=50,
            parallel=4,
            base_seed=42,
        )
        result = await self.runner.run_game(game, agents, config)

        # Basic assertions
        assert result.num_episodes == 50
        assert "Prisoner's Dilemma" in result.game_name

        # Statistics
        stats = result.player_statistics()
        assert stats["player_0"].mean == pytest.approx(0.0)
        assert stats["player_1"].mean == pytest.approx(25.0)

        # Agent comparison
        comps = result.agent_comparisons()
        assert len(comps) == 1
        assert comps[0].is_significant

        # Serialization
        data = result.to_dict()
        assert data["config"]["parallel"] == 4
        assert data["config"]["base_seed"] == 42

    @pytest.mark.anyio
    async def test_parallel_config_in_result(self) -> None:
        """Config is preserved in result."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysCooperate()),
        }
        config = GameRunConfig(
            episodes=5,
            parallel=2,
            base_seed=0,
        )
        result = await self.runner.run_game(game, agents, config)
        assert result.config.parallel == 2
        assert result.config.base_seed == 0
