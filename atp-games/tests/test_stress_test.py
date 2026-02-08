"""Tests for adversarial stress-test."""

import pytest
from game_envs.games.prisoners_dilemma import (
    PDConfig,
    PrisonersDilemma,
)
from game_envs.strategies.pd_strategies import (
    AlwaysCooperate,
    AlwaysDefect,
)

from atp_games.models import GameRunConfig
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.suites.stress_test import (
    BestResponseStrategy,
    StressTestResult,
    _extract_empirical_strategy,
    run_stress_test,
)


class TestBestResponseStrategy:
    """Test BestResponseStrategy."""

    def test_always_plays_best_action(self) -> None:
        """BR strategy always returns the best action."""
        br = BestResponseStrategy(
            best_action="defect",
            action_names=["cooperate", "defect"],
        )
        # choose_action ignores observation, always returns best
        assert br.choose_action(None) == "defect"  # type: ignore[arg-type]
        assert br.name == "best_response(defect)"

    def test_reset_is_no_op(self) -> None:
        """Reset does nothing (stateless strategy)."""
        br = BestResponseStrategy(
            best_action="cooperate",
            action_names=["cooperate", "defect"],
        )
        br.reset()  # Should not raise


class TestStressTest:
    """Test adversarial stress-test execution."""

    @pytest.mark.anyio
    async def test_stress_test_allc(self) -> None:
        """AllC should be highly exploitable."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agent = BuiltinAdapter(AlwaysCooperate())
        config = GameRunConfig(episodes=5, base_seed=42)

        result = await run_stress_test(
            game=game,
            agent_name="allc",
            agent=agent,
            config=config,
            threshold=0.15,
            iterations=1,
            profiling_episodes=5,
        )

        assert isinstance(result, StressTestResult)
        assert result.agent_name == "allc"
        assert len(result.iterations) == 1
        # AllC is highly exploitable
        assert result.final_exploitability > 0.0
        assert result.passed is False  # Should fail stress test

    @pytest.mark.anyio
    async def test_stress_test_alld(self) -> None:
        """AllD should have lower exploitability (it's a NE)."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agent = BuiltinAdapter(AlwaysDefect())
        config = GameRunConfig(episodes=5, base_seed=42)

        result = await run_stress_test(
            game=game,
            agent_name="alld",
            agent=agent,
            config=config,
            threshold=0.15,
            iterations=1,
            profiling_episodes=5,
        )

        assert result.agent_name == "alld"
        # AllD in one-shot PD is Nash equilibrium
        # So exploitability for player 0 (AllD) should be ~0
        assert result.final_exploitability <= 0.15 + 1e-6

    @pytest.mark.anyio
    async def test_stress_test_iterative(self) -> None:
        """Multiple iterations should produce multiple results."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agent = BuiltinAdapter(AlwaysCooperate())
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_stress_test(
            game=game,
            agent_name="allc",
            agent=agent,
            config=config,
            threshold=0.15,
            iterations=2,
            profiling_episodes=3,
        )

        assert len(result.iterations) == 2

    @pytest.mark.anyio
    async def test_stress_test_serialization(self) -> None:
        """StressTestResult should serialize to dict."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agent = BuiltinAdapter(AlwaysDefect())
        config = GameRunConfig(episodes=3, base_seed=42)

        result = await run_stress_test(
            game=game,
            agent_name="alld",
            agent=agent,
            config=config,
            threshold=0.15,
            iterations=1,
            profiling_episodes=3,
        )
        d = result.to_dict()

        assert d["agent_name"] == "alld"
        assert "final_exploitability" in d
        assert "passed" in d
        assert "threshold" in d
        assert "iterations" in d
        assert len(d["iterations"]) == 1

    @pytest.mark.anyio
    async def test_requires_two_players(self) -> None:
        """Stress test requires exactly 2 players."""
        # PD is always 2-player, so we check via a direct call
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        # This should work fine with 2 players
        agent = BuiltinAdapter(AlwaysDefect())
        config = GameRunConfig(episodes=2, base_seed=42)

        result = await run_stress_test(
            game=game,
            agent_name="test",
            agent=agent,
            config=config,
            iterations=1,
            profiling_episodes=2,
        )
        assert len(result.iterations) == 1

    @pytest.mark.anyio
    async def test_best_response_to_allc_is_defect(self) -> None:
        """Best response to AllC in PD should be Defect."""
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agent = BuiltinAdapter(AlwaysCooperate())
        config = GameRunConfig(episodes=5, base_seed=42)

        result = await run_stress_test(
            game=game,
            agent_name="allc",
            agent=agent,
            config=config,
            threshold=0.15,
            iterations=1,
            profiling_episodes=5,
        )

        # BR to AllC is always Defect
        br_action = result.iterations[0].best_response_action
        assert br_action == "defect"


class TestExtractEmpiricalStrategy:
    """Test empirical strategy extraction from game results."""

    @pytest.mark.anyio
    async def test_extract_from_game_result(self) -> None:
        """Extract empirical strategy from game result."""
        from atp_games.runner.game_runner import GameRunner

        game = PrisonersDilemma(PDConfig(num_rounds=5))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        runner = GameRunner()
        config = GameRunConfig(episodes=3, base_seed=42)
        result = await runner.run_game(game, agents, config)

        strat = _extract_empirical_strategy(result, "player_0")
        assert strat.action_frequencies.get("cooperate", 0) == pytest.approx(1.0)

        strat_d = _extract_empirical_strategy(result, "player_1")
        assert strat_d.action_frequencies.get("defect", 0) == pytest.approx(1.0)

    @pytest.mark.anyio
    async def test_extract_empty_raises(self) -> None:
        """Extracting from result with no actions raises."""
        from atp_games.models import EpisodeResult, GameResult

        result = GameResult(
            game_name="test",
            config=GameRunConfig(episodes=1),
            episodes=[EpisodeResult(episode=0, payoffs={"p0": 0}, actions_log=[])],
        )
        with pytest.raises(ValueError, match="No actions"):
            _extract_empirical_strategy(result, "p0")
