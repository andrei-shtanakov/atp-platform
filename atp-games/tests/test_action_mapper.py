"""Tests for ActionMapper."""

import pytest
from atp.protocol.models import (
    ArtifactFile,
    ArtifactStructured,
    ATPResponse,
    ResponseStatus,
)

from atp_games.mapping.action_mapper import ActionMapper


class TestActionMapper:
    def setup_method(self) -> None:
        self.mapper = ActionMapper()

    def test_structured_artifact(self) -> None:
        response = ATPResponse(
            task_id="test-1",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={
                        "action": "cooperate",
                        "reasoning": "I want peace",
                    },
                ),
            ],
        )
        action = self.mapper.from_atp_response(response)
        assert action.action == "cooperate"
        assert action.reasoning == "I want peace"

    def test_file_artifact_with_json_content(self) -> None:
        response = ATPResponse(
            task_id="test-2",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(
                    path="action.json",
                    content='{"action": "defect", "message": "hi"}',
                ),
            ],
        )
        action = self.mapper.from_atp_response(response)
        assert action.action == "defect"
        assert action.message == "hi"

    def test_json_in_text_content(self) -> None:
        response = ATPResponse(
            task_id="test-3",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(
                    path="response.txt",
                    content=('Some text {"action": "cooperate"} more'),
                ),
            ],
        )
        action = self.mapper.from_atp_response(response)
        assert action.action == "cooperate"

    def test_failed_response_raises(self) -> None:
        response = ATPResponse(
            task_id="test-4",
            status=ResponseStatus.FAILED,
            error="Agent crashed",
        )
        with pytest.raises(ValueError, match="Agent returned error"):
            self.mapper.from_atp_response(response)

    def test_no_action_in_artifacts(self) -> None:
        response = ATPResponse(
            task_id="test-5",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="other",
                    data={"foo": "bar"},
                ),
            ],
        )
        with pytest.raises(ValueError, match="No valid action"):
            self.mapper.from_atp_response(response)

    def test_empty_artifacts(self) -> None:
        response = ATPResponse(
            task_id="test-6",
            status=ResponseStatus.COMPLETED,
            artifacts=[],
        )
        with pytest.raises(ValueError, match="No valid action"):
            self.mapper.from_atp_response(response)

    def test_roundtrip_with_observation_mapper(self) -> None:
        """Verify mapper roundtrip: obs -> request -> response -> action."""
        from game_envs.core.state import Observation

        from atp_games.mapping.observation_mapper import (
            ObservationMapper,
        )

        obs_mapper = ObservationMapper()
        obs = Observation(
            player_id="p0",
            game_state={"x": 1},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=1,
            total_rounds=5,
        )
        request = obs_mapper.to_atp_request(obs, "PD", episode=0)

        # Simulate agent response
        response = ATPResponse(
            task_id=request.task_id,
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "cooperate"},
                ),
            ],
        )
        action = self.mapper.from_atp_response(response)
        assert action.action == "cooperate"
