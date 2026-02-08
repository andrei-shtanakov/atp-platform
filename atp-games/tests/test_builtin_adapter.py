"""Tests for BuiltinAdapter."""

import pytest
from atp.protocol.models import ResponseStatus

# Import concrete strategies from game-environments
from game_envs.core.state import Observation
from game_envs.strategies.pd_strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
)

from atp_games.mapping.observation_mapper import ObservationMapper
from atp_games.runner.builtin_adapter import BuiltinAdapter


class TestBuiltinAdapter:
    def test_adapter_type(self) -> None:
        adapter = BuiltinAdapter(AlwaysCooperate())
        assert adapter.adapter_type == "builtin"

    @pytest.mark.anyio
    async def test_always_cooperate(self) -> None:
        adapter = BuiltinAdapter(AlwaysCooperate())
        mapper = ObservationMapper()

        obs = Observation(
            player_id="p0",
            game_state={},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        request = mapper.to_atp_request(obs, "PD", episode=0)
        response = await adapter.execute(request)

        assert response.status == ResponseStatus.COMPLETED
        assert len(response.artifacts) == 1
        assert response.artifacts[0].data["action"] == "cooperate"

    @pytest.mark.anyio
    async def test_always_defect(self) -> None:
        adapter = BuiltinAdapter(AlwaysDefect())
        mapper = ObservationMapper()

        obs = Observation(
            player_id="p1",
            game_state={},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        request = mapper.to_atp_request(obs, "PD", episode=0)
        response = await adapter.execute(request)

        assert response.artifacts[0].data["action"] == "defect"

    @pytest.mark.anyio
    async def test_tit_for_tat_first_move(self) -> None:
        adapter = BuiltinAdapter(TitForTat())

        obs = Observation(
            player_id="p0",
            game_state={},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=1,
            total_rounds=5,
        )
        mapper = ObservationMapper()
        request = mapper.to_atp_request(obs, "PD", episode=0)
        response = await adapter.execute(request)

        # TFT cooperates on first move
        assert response.artifacts[0].data["action"] == "cooperate"

    def test_strategy_property(self) -> None:
        strategy = AlwaysDefect()
        adapter = BuiltinAdapter(strategy)
        assert adapter.strategy is strategy

    def test_reset(self) -> None:
        adapter = BuiltinAdapter(TitForTat())
        # Should not raise
        adapter.reset()

    @pytest.mark.anyio
    async def test_stream_events(self) -> None:
        adapter = BuiltinAdapter(AlwaysCooperate())
        mapper = ObservationMapper()

        obs = Observation(
            player_id="p0",
            game_state={},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        request = mapper.to_atp_request(obs, "PD", episode=0)

        items = []
        async for item in adapter.stream_events(request):
            items.append(item)

        assert len(items) == 1
        assert items[0].status == ResponseStatus.COMPLETED

    @pytest.mark.anyio
    async def test_observation_roundtrip(self) -> None:
        """Verify observation is correctly reconstructed."""
        adapter = BuiltinAdapter(AlwaysCooperate())
        mapper = ObservationMapper()

        obs = Observation(
            player_id="player_0",
            game_state={"round": 2},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=2,
            total_rounds=10,
        )
        request = mapper.to_atp_request(obs, "PD", episode=1)

        # Verify _extract_observation reconstructs correctly
        extracted = adapter._extract_observation(request)
        assert extracted.player_id == "player_0"
        assert extracted.round_number == 2
        assert extracted.total_rounds == 10
