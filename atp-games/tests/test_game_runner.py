"""Tests for GameRunner."""

import pytest
from atp.protocol.models import (
    ArtifactStructured,
    ATPResponse,
    ResponseStatus,
)
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
from atp_games.runner.game_runner import GameRunner


class TestGameRunnerWithBuiltinAgents:
    """Test GameRunner with built-in strategy adapters."""

    def setup_method(self) -> None:
        self.runner = GameRunner()

    @pytest.mark.anyio
    async def test_one_shot_pd(self) -> None:
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1)
        result = await self.runner.run_game(game, agents, config)

        assert result.game_name == "Prisoner's Dilemma"
        assert result.num_episodes == 1
        ep = result.episodes[0]
        # AllC vs AllD: C gets sucker (0), D gets temptation (5)
        assert ep.payoffs["player_0"] == pytest.approx(0.0)
        assert ep.payoffs["player_1"] == pytest.approx(5.0)

    @pytest.mark.anyio
    async def test_repeated_pd(self) -> None:
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysCooperate()),
        }
        config = GameRunConfig(episodes=1)
        result = await self.runner.run_game(game, agents, config)

        ep = result.episodes[0]
        # Mutual cooperation: R=3 per round, 3 rounds = 9
        assert ep.payoffs["player_0"] == pytest.approx(9.0)
        assert ep.payoffs["player_1"] == pytest.approx(9.0)

    @pytest.mark.anyio
    async def test_multiple_episodes(self) -> None:
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysDefect()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=3)
        result = await self.runner.run_game(game, agents, config)

        assert result.num_episodes == 3
        # Mutual defection: P=1 each episode
        for ep in result.episodes:
            assert ep.payoffs["player_0"] == pytest.approx(1.0)
            assert ep.payoffs["player_1"] == pytest.approx(1.0)

    @pytest.mark.anyio
    async def test_average_payoffs(self) -> None:
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysDefect()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=5)
        result = await self.runner.run_game(game, agents, config)

        avg = result.average_payoffs
        assert avg["player_0"] == pytest.approx(1.0)
        assert avg["player_1"] == pytest.approx(1.0)

    @pytest.mark.anyio
    async def test_agent_names(self) -> None:
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(TitForTat()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        result = await self.runner.run_game(game, agents)
        assert result.agent_names["player_0"] == "tit_for_tat"
        assert result.agent_names["player_1"] == "always_defect"

    @pytest.mark.anyio
    async def test_actions_log(self) -> None:
        game = PrisonersDilemma(PDConfig(num_rounds=2))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1)
        result = await self.runner.run_game(game, agents, config)

        ep = result.episodes[0]
        assert len(ep.actions_log) == 2
        for actions in ep.actions_log:
            assert actions["player_0"] == "cooperate"
            assert actions["player_1"] == "defect"

    @pytest.mark.anyio
    async def test_result_serialization(self) -> None:
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        result = await self.runner.run_game(game, agents)
        data = result.to_dict()

        assert data["game_name"] == "Prisoner's Dilemma"
        assert len(data["episodes"]) == 1
        assert "average_payoffs" in data


class TestGameRunnerWithMockAgent:
    """Test GameRunner with mock agents for validation/retry."""

    def setup_method(self) -> None:
        self.runner = GameRunner()

    @pytest.mark.anyio
    async def test_invalid_action_uses_default(self) -> None:
        """Agent returns invalid action, should fallback."""
        from unittest.mock import AsyncMock

        from atp.adapters.base import AgentAdapter

        mock_agent = AsyncMock(spec=AgentAdapter)
        mock_agent.adapter_type = "mock"

        # Always return an invalid action
        mock_agent.execute.return_value = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "invalid_action"},
                ),
            ],
        )

        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": mock_agent,
            "player_1": BuiltinAdapter(AlwaysCooperate()),
        }
        config = GameRunConfig(episodes=1, max_retries=2)
        result = await self.runner.run_game(game, agents, config)

        # Should complete (using default action after retries)
        assert result.num_episodes == 1
        ep = result.episodes[0]
        # Default action should be valid
        assert ep.actions_log[0]["player_0"] in [
            "cooperate",
            "defect",
        ]

    @pytest.mark.anyio
    async def test_retry_count(self) -> None:
        """Verify agent is called max_retries + 1 times."""
        from unittest.mock import AsyncMock

        from atp.adapters.base import AgentAdapter

        mock_agent = AsyncMock(spec=AgentAdapter)
        mock_agent.adapter_type = "mock"

        mock_agent.execute.return_value = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "bad"},
                ),
            ],
        )

        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": mock_agent,
            "player_1": BuiltinAdapter(AlwaysCooperate()),
        }
        config = GameRunConfig(episodes=1, max_retries=3)
        await self.runner.run_game(game, agents, config)

        # Should be called 4 times (1 initial + 3 retries)
        assert mock_agent.execute.call_count == 4

    @pytest.mark.anyio
    async def test_successful_retry(self) -> None:
        """Agent fails first, succeeds on retry."""
        from unittest.mock import AsyncMock

        from atp.adapters.base import AgentAdapter

        mock_agent = AsyncMock(spec=AgentAdapter)
        mock_agent.adapter_type = "mock"

        bad_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "invalid"},
                ),
            ],
        )
        good_response = ATPResponse(
            task_id="test",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "cooperate"},
                ),
            ],
        )
        mock_agent.execute.side_effect = [
            bad_response,
            good_response,
        ]

        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": mock_agent,
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1, max_retries=3)
        result = await self.runner.run_game(game, agents, config)

        ep = result.episodes[0]
        assert ep.actions_log[0]["player_0"] == "cooperate"
        assert mock_agent.execute.call_count == 2

    @pytest.mark.anyio
    async def test_agent_error_uses_default(self) -> None:
        """Agent returns FAILED status, should fallback."""
        from unittest.mock import AsyncMock

        from atp.adapters.base import AgentAdapter

        mock_agent = AsyncMock(spec=AgentAdapter)
        mock_agent.adapter_type = "mock"

        mock_agent.execute.return_value = ATPResponse(
            task_id="test",
            status=ResponseStatus.FAILED,
            error="Agent crashed",
        )

        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": mock_agent,
            "player_1": BuiltinAdapter(AlwaysCooperate()),
        }
        config = GameRunConfig(episodes=1, max_retries=2)
        result = await self.runner.run_game(game, agents, config)

        # Should still complete with default action
        assert result.num_episodes == 1
