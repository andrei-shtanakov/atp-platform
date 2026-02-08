"""Integration test: full PD game with 2 builtin strategies."""

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
from atp_games.runner.game_runner import GameRunner


class TestFullPDIntegration:
    """Full integration: repeated PD with builtin strategies."""

    @pytest.mark.anyio
    async def test_tft_vs_alld_repeated(self) -> None:
        """TFT vs AllD in repeated PD.

        TFT cooperates first, then copies opponent.
        AllD always defects.

        Round 1: TFT=C, AllD=D -> TFT gets S=0, AllD gets T=5
        Round 2+: TFT=D, AllD=D -> both get P=1

        Expected for 10 rounds:
          TFT:  0 + 9*1 = 9
          AllD: 5 + 9*1 = 14
        """
        game = PrisonersDilemma(PDConfig(num_rounds=10, seed=42))
        runner = GameRunner()
        agents = {
            "player_0": BuiltinAdapter(TitForTat()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        config = GameRunConfig(episodes=1)
        result = await runner.run_game(game, agents, config)

        ep = result.episodes[0]
        assert ep.payoffs["player_0"] == pytest.approx(9.0)
        assert ep.payoffs["player_1"] == pytest.approx(14.0)

        # Verify action sequence
        assert len(ep.actions_log) == 10
        assert ep.actions_log[0]["player_0"] == "cooperate"
        assert ep.actions_log[0]["player_1"] == "defect"
        for i in range(1, 10):
            assert ep.actions_log[i]["player_0"] == "defect"
            assert ep.actions_log[i]["player_1"] == "defect"

    @pytest.mark.anyio
    async def test_allc_vs_allc_repeated(self) -> None:
        """AllC vs AllC: mutual cooperation every round.

        Expected for 5 rounds: both get R=3 per round = 15.
        """
        game = PrisonersDilemma(PDConfig(num_rounds=5, seed=123))
        runner = GameRunner()
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysCooperate()),
        }
        config = GameRunConfig(episodes=1)
        result = await runner.run_game(game, agents, config)

        ep = result.episodes[0]
        assert ep.payoffs["player_0"] == pytest.approx(15.0)
        assert ep.payoffs["player_1"] == pytest.approx(15.0)

    @pytest.mark.anyio
    async def test_tft_vs_tft_repeated(self) -> None:
        """TFT vs TFT: mutual cooperation throughout.

        Both start with cooperation, then copy each other.
        Expected for 5 rounds: both get R=3 per round = 15.
        """
        game = PrisonersDilemma(PDConfig(num_rounds=5, seed=0))
        runner = GameRunner()
        agents = {
            "player_0": BuiltinAdapter(TitForTat()),
            "player_1": BuiltinAdapter(TitForTat()),
        }
        config = GameRunConfig(episodes=1)
        result = await runner.run_game(game, agents, config)

        ep = result.episodes[0]
        assert ep.payoffs["player_0"] == pytest.approx(15.0)
        assert ep.payoffs["player_1"] == pytest.approx(15.0)

    @pytest.mark.anyio
    async def test_multi_episode_consistency(self) -> None:
        """Run multiple episodes and verify consistency."""
        game = PrisonersDilemma(PDConfig(num_rounds=3, seed=99))
        runner = GameRunner()
        agents = {
            "player_0": BuiltinAdapter(AlwaysDefect()),
            "player_1": BuiltinAdapter(AlwaysCooperate()),
        }
        config = GameRunConfig(episodes=5)
        result = await runner.run_game(game, agents, config)

        assert result.num_episodes == 5
        for ep in result.episodes:
            # AllD vs AllC: D gets T=5 per round, C gets S=0
            assert ep.payoffs["player_0"] == pytest.approx(15.0)
            assert ep.payoffs["player_1"] == pytest.approx(0.0)

        avg = result.average_payoffs
        assert avg["player_0"] == pytest.approx(15.0)
        assert avg["player_1"] == pytest.approx(0.0)

    @pytest.mark.anyio
    async def test_game_result_serialization(self) -> None:
        """Full roundtrip: run game -> serialize -> verify."""
        game = PrisonersDilemma(PDConfig(num_rounds=2))
        runner = GameRunner()
        agents = {
            "player_0": BuiltinAdapter(TitForTat()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        result = await runner.run_game(game, agents)

        data = result.to_dict()
        assert "Prisoner's Dilemma" in data["game_name"]
        assert data["agent_names"]["player_0"] == "tit_for_tat"
        assert data["agent_names"]["player_1"] == "always_defect"
        assert len(data["episodes"]) == 1
        assert "average_payoffs" in data
