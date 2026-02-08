"""Tests for the Alympics benchmark suite scoring module."""

from __future__ import annotations

import pytest

from atp_games.models import EpisodeResult, GameResult, GameRunConfig
from atp_games.suites.alympics import (
    CATEGORY_WEIGHTS,
    AlympicsResult,
    CategoryScore,
    compute_category_scores,
    compute_composite_score,
    load_alympics_config,
    normalise_payoff,
    score_benchmark,
)

# -------------------------------------------------------------------
# normalise_payoff
# -------------------------------------------------------------------


class TestNormalisePayoff:
    """Tests for payoff normalisation."""

    def test_midpoint_maps_to_50(self) -> None:
        """Midpoint of baseline range maps to 50."""
        score = normalise_payoff(2.0, "prisoners_dilemma")
        assert score == pytest.approx(50.0)

    def test_min_maps_to_0(self) -> None:
        """Minimum baseline maps to 0."""
        score = normalise_payoff(1.0, "prisoners_dilemma")
        assert score == pytest.approx(0.0)

    def test_max_maps_to_100(self) -> None:
        """Maximum baseline maps to 100."""
        score = normalise_payoff(3.0, "prisoners_dilemma")
        assert score == pytest.approx(100.0)

    def test_below_min_clipped_to_0(self) -> None:
        """Values below min baseline are clipped to 0."""
        score = normalise_payoff(0.0, "prisoners_dilemma")
        assert score == pytest.approx(0.0)

    def test_above_max_clipped_to_100(self) -> None:
        """Values above max baseline are clipped to 100."""
        score = normalise_payoff(5.0, "prisoners_dilemma")
        assert score == pytest.approx(100.0)

    def test_unknown_game_returns_50(self) -> None:
        """Unknown game type returns default 50."""
        score = normalise_payoff(42.0, "unknown_game")
        assert score == pytest.approx(50.0)

    def test_custom_baselines(self) -> None:
        """Custom baselines override defaults."""
        custom = {"test_game": {"min": 0.0, "max": 10.0}}
        score = normalise_payoff(5.0, "test_game", baselines=custom)
        assert score == pytest.approx(50.0)

    def test_equal_min_max_returns_50(self) -> None:
        """When min == max, returns 50."""
        custom = {"test_game": {"min": 5.0, "max": 5.0}}
        score = normalise_payoff(5.0, "test_game", baselines=custom)
        assert score == pytest.approx(50.0)

    def test_congestion_negative_range(self) -> None:
        """Congestion game has negative payoff range."""
        # min=-20, max=-1. Midpoint is -10.5
        score = normalise_payoff(-10.5, "congestion")
        assert score == pytest.approx(50.0)

        score_best = normalise_payoff(-1.0, "congestion")
        assert score_best == pytest.approx(100.0)

        score_worst = normalise_payoff(-20.0, "congestion")
        assert score_worst == pytest.approx(0.0)

    def test_auction_normalisation(self) -> None:
        """Auction payoff normalisation (0-50 range)."""
        score = normalise_payoff(25.0, "auction")
        assert score == pytest.approx(50.0)

    def test_colonel_blotto_normalisation(self) -> None:
        """Colonel Blotto uses 0-1 range."""
        score = normalise_payoff(0.5, "colonel_blotto")
        assert score == pytest.approx(50.0)


# -------------------------------------------------------------------
# compute_category_scores
# -------------------------------------------------------------------


class TestComputeCategoryScores:
    """Tests for category score computation."""

    def test_all_games_score_50(self) -> None:
        """When all games score 50, all categories score 50."""
        game_scores = {
            "prisoners_dilemma": 50.0,
            "public_goods": 50.0,
            "auction": 50.0,
            "colonel_blotto": 50.0,
            "congestion": 50.0,
        }
        categories = compute_category_scores(game_scores)
        for cat in categories.values():
            assert cat.score == pytest.approx(50.0)

    def test_strategic_category_weights(self) -> None:
        """Strategic category weights 4 games equally."""
        game_scores = {
            "prisoners_dilemma": 100.0,
            "auction": 0.0,
            "colonel_blotto": 100.0,
            "congestion": 0.0,
        }
        categories = compute_category_scores(game_scores)
        assert categories["strategic"].score == pytest.approx(50.0)

    def test_cooperation_category(self) -> None:
        """Cooperation category uses PD and PG."""
        game_scores = {
            "prisoners_dilemma": 80.0,
            "public_goods": 60.0,
        }
        categories = compute_category_scores(game_scores)
        # PD weight 0.5, PG weight 0.5
        expected = (80.0 * 0.5 + 60.0 * 0.5) / (0.5 + 0.5)
        assert categories["cooperation"].score == pytest.approx(expected)

    def test_missing_games_excluded(self) -> None:
        """Games not in game_scores are excluded from category."""
        game_scores = {"prisoners_dilemma": 100.0}
        categories = compute_category_scores(game_scores)
        # Strategic only has PD
        assert categories["strategic"].score == pytest.approx(100.0)
        # Cooperation has PD with weight 0.5
        assert categories["cooperation"].score == pytest.approx(100.0)

    def test_empty_scores(self) -> None:
        """Empty game scores produce 0 for all categories."""
        categories = compute_category_scores({})
        for cat in categories.values():
            assert cat.score == pytest.approx(0.0)

    def test_custom_weights(self) -> None:
        """Custom category weights are respected."""
        custom_weights = {"custom": 1.0}
        custom_map = {"custom": {"prisoners_dilemma": 1.0}}
        game_scores = {"prisoners_dilemma": 75.0}
        categories = compute_category_scores(
            game_scores,
            category_weights=custom_weights,
            category_game_map=custom_map,
        )
        assert "custom" in categories
        assert categories["custom"].score == pytest.approx(75.0)
        assert categories["custom"].weight == 1.0

    def test_game_scores_tracked_per_category(self) -> None:
        """Each category tracks which games contributed."""
        game_scores = {
            "prisoners_dilemma": 80.0,
            "auction": 60.0,
        }
        categories = compute_category_scores(game_scores)
        assert "prisoners_dilemma" in categories["strategic"].game_scores
        assert "auction" in categories["strategic"].game_scores

    def test_category_weights_match_defaults(self) -> None:
        """Default category weights sum to 1.0."""
        total = sum(CATEGORY_WEIGHTS.values())
        assert total == pytest.approx(1.0)


# -------------------------------------------------------------------
# compute_composite_score
# -------------------------------------------------------------------


class TestComputeCompositeScore:
    """Tests for composite score computation."""

    def test_uniform_scores(self) -> None:
        """All categories at same score gives that score."""
        categories = {
            "a": CategoryScore(name="a", score=75.0, weight=0.5),
            "b": CategoryScore(name="b", score=75.0, weight=0.5),
        }
        assert compute_composite_score(categories) == pytest.approx(75.0)

    def test_weighted_average(self) -> None:
        """Composite is weighted average of categories."""
        categories = {
            "a": CategoryScore(name="a", score=100.0, weight=0.3),
            "b": CategoryScore(name="b", score=0.0, weight=0.7),
        }
        expected = (100.0 * 0.3 + 0.0 * 0.7) / (0.3 + 0.7)
        assert compute_composite_score(categories) == pytest.approx(expected)

    def test_empty_categories(self) -> None:
        """Empty categories returns 0."""
        assert compute_composite_score({}) == pytest.approx(0.0)

    def test_zero_weights(self) -> None:
        """All zero weights returns 0."""
        categories = {
            "a": CategoryScore(name="a", score=100.0, weight=0.0),
        }
        assert compute_composite_score(categories) == pytest.approx(0.0)


# -------------------------------------------------------------------
# score_benchmark
# -------------------------------------------------------------------


def _make_game_result(
    game_name: str,
    payoffs: dict[str, float],
    num_episodes: int = 5,
) -> GameResult:
    """Create a GameResult with uniform payoffs across episodes."""
    config = GameRunConfig(episodes=num_episodes)
    episodes = [
        EpisodeResult(episode=i, payoffs=dict(payoffs)) for i in range(num_episodes)
    ]
    return GameResult(
        game_name=game_name,
        config=config,
        episodes=episodes,
        agent_names={pid: pid for pid in payoffs},
    )


class TestScoreBenchmark:
    """Tests for the full score_benchmark function."""

    def test_basic_scoring(self) -> None:
        """Score benchmark with results from all 5 games."""
        game_results = {
            "prisoners_dilemma": _make_game_result("PD", {"p0": 2.0, "p1": 2.0}),
            "public_goods": _make_game_result("PG", {"p0": 5.0, "p1": 5.0}),
            "auction": _make_game_result("Auction", {"p0": 25.0, "p1": 10.0}),
            "colonel_blotto": _make_game_result("Blotto", {"p0": 0.5, "p1": 0.5}),
            "congestion": _make_game_result("Cong", {"p0": -10.5, "p1": -10.5}),
        }

        result = score_benchmark(game_results, "test_agent")

        assert isinstance(result, AlympicsResult)
        assert result.agent_name == "test_agent"
        assert 0.0 <= result.composite_score <= 100.0
        assert len(result.categories) == 4
        assert "strategic" in result.categories
        assert "cooperation" in result.categories
        assert "fairness" in result.categories
        assert "robustness" in result.categories

    def test_midpoint_payoffs_give_50(self) -> None:
        """Midpoint payoffs across all games give ~50 composite."""
        game_results = {
            "prisoners_dilemma": _make_game_result("PD", {"p0": 2.0}),
            "public_goods": _make_game_result("PG", {"p0": 5.0}),
            "auction": _make_game_result("Auc", {"p0": 25.0}),
            "colonel_blotto": _make_game_result("Blotto", {"p0": 0.5}),
            "congestion": _make_game_result("Cong", {"p0": -10.5}),
        }

        result = score_benchmark(game_results, "agent")

        # All games at midpoint -> all categories at 50 -> composite 50
        assert result.composite_score == pytest.approx(50.0)
        for cat in result.categories.values():
            assert cat.score == pytest.approx(50.0)

    def test_player_id_selection(self) -> None:
        """Specific player_id is used for scoring."""
        game_results = {
            "prisoners_dilemma": _make_game_result("PD", {"p0": 1.0, "p1": 3.0}),
        }

        # p0 gets min (1.0) -> score 0
        result_p0 = score_benchmark(game_results, "agent", player_id="p0")
        # p1 gets max (3.0) -> score 100
        result_p1 = score_benchmark(game_results, "agent", player_id="p1")

        pd_score_p0 = result_p0.categories["strategic"].game_scores.get(
            "prisoners_dilemma", 0.0
        )
        pd_score_p1 = result_p1.categories["strategic"].game_scores.get(
            "prisoners_dilemma", 0.0
        )
        assert pd_score_p0 == pytest.approx(0.0)
        assert pd_score_p1 == pytest.approx(100.0)

    def test_empty_game_results(self) -> None:
        """Empty game results gives 0 composite."""
        result = score_benchmark({}, "agent")
        assert result.composite_score == pytest.approx(0.0)

    def test_summary_format(self) -> None:
        """Summary string follows expected format."""
        game_results = {
            "prisoners_dilemma": _make_game_result("PD", {"p0": 2.0}),
        }
        result = score_benchmark(game_results, "test_agent")
        summary = result.summary()

        assert "test_agent" in summary
        assert "scored" in summary
        assert "/100" in summary
        assert "strategic:" in summary
        assert "cooperation:" in summary

    def test_to_dict_serialization(self) -> None:
        """AlympicsResult serialises to dict correctly."""
        game_results = {
            "prisoners_dilemma": _make_game_result("PD", {"p0": 2.0}),
        }
        result = score_benchmark(game_results, "agent")
        data = result.to_dict()

        assert "agent_name" in data
        assert "composite_score" in data
        assert "categories" in data
        assert "game_results" in data
        assert data["agent_name"] == "agent"
        assert isinstance(data["composite_score"], float)

    def test_category_score_to_dict(self) -> None:
        """CategoryScore serialises correctly."""
        cat = CategoryScore(
            name="test",
            score=75.5,
            weight=0.3,
            game_scores={"pd": 80.0, "pg": 71.0},
        )
        data = cat.to_dict()
        assert data["name"] == "test"
        assert data["score"] == 75.5
        assert data["weight"] == 0.3
        assert data["game_scores"]["pd"] == 80.0


# -------------------------------------------------------------------
# load_alympics_config
# -------------------------------------------------------------------


class TestLoadAlympicsConfig:
    """Tests for loading the builtin YAML config."""

    def test_config_loads_successfully(self) -> None:
        """Builtin alympics_lite.yaml loads without error."""
        config = load_alympics_config()
        assert isinstance(config, dict)
        assert "games" in config

    def test_config_has_5_games(self) -> None:
        """Config defines all 5 canonical games."""
        config = load_alympics_config()
        games = config["games"]
        assert len(games) == 5

        game_types = {g["type"] for g in games}
        assert game_types == {
            "prisoners_dilemma",
            "public_goods",
            "auction",
            "colonel_blotto",
            "congestion",
        }

    def test_config_has_scoring_section(self) -> None:
        """Config includes scoring categories and baselines."""
        config = load_alympics_config()
        assert "scoring" in config
        scoring = config["scoring"]
        assert "categories" in scoring
        assert "baselines" in scoring

    def test_config_categories_match_module(self) -> None:
        """YAML categories match module constants."""
        config = load_alympics_config()
        yaml_cats = set(config["scoring"]["categories"].keys())
        module_cats = set(CATEGORY_WEIGHTS.keys())
        assert yaml_cats == module_cats

    def test_each_game_has_agents(self) -> None:
        """Each game spec has at least 2 agents."""
        config = load_alympics_config()
        for game in config["games"]:
            assert len(game.get("agents", [])) >= 2, (
                f"Game {game['type']} has < 2 agents"
            )

    def test_each_game_has_episodes(self) -> None:
        """Each game spec has an episode count."""
        config = load_alympics_config()
        for game in config["games"]:
            assert "episodes" in game, f"Game {game['type']} missing episodes"
            assert game["episodes"] > 0


# -------------------------------------------------------------------
# Integration: run_alympics
# -------------------------------------------------------------------


class TestRunAlympics:
    """Integration test for the full benchmark run."""

    @pytest.mark.anyio
    async def test_run_alympics_completes(self) -> None:
        """Full benchmark runs to completion with builtin strategies.

        Uses a small episode count for speed.
        """
        from atp_games.suites.alympics import run_alympics

        result = await run_alympics(
            agent_name="test_builtin",
            episodes_override=2,
        )

        assert isinstance(result, AlympicsResult)
        assert result.agent_name == "test_builtin"
        assert 0.0 <= result.composite_score <= 100.0
        assert len(result.categories) == 4
        assert len(result.game_results) == 5

        # Verify all 5 games ran
        assert set(result.game_results.keys()) == {
            "prisoners_dilemma",
            "public_goods",
            "auction",
            "colonel_blotto",
            "congestion",
        }

        # Each game should have results
        for game_type, game_result in result.game_results.items():
            assert game_result.num_episodes == 2, f"{game_type} expected 2 episodes"

    @pytest.mark.anyio
    async def test_run_alympics_scores_reasonable(self) -> None:
        """Benchmark scores are within reasonable bounds.

        With builtin strategies, scores should not be 0 or 100
        (those would indicate normalisation errors).
        """
        from atp_games.suites.alympics import run_alympics

        result = await run_alympics(
            agent_name="builtin",
            episodes_override=3,
        )

        # Composite should be somewhere in the middle
        assert result.composite_score > 0.0
        assert result.composite_score < 100.0

        # Summary should be well-formed
        summary = result.summary()
        assert "builtin scored" in summary

    @pytest.mark.anyio
    async def test_run_alympics_serialization(self) -> None:
        """Benchmark result serialises to dict and back."""
        from atp_games.suites.alympics import run_alympics

        result = await run_alympics(
            agent_name="test",
            episodes_override=2,
        )

        data = result.to_dict()
        assert isinstance(data, dict)
        assert "composite_score" in data
        assert "categories" in data
        assert len(data["categories"]) == 4
        assert len(data["game_results"]) == 5
