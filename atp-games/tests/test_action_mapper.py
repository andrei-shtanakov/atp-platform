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


class TestActionMapperIntent:
    """Phase 3: Tier-1 agent self-report `intent` field extraction.

    The `intent` attribute is an optional free-text field on GameAction
    populated from the same structured artifact or parsed JSON payload
    that produces `action`. Missing/invalid intents must never fail the
    action extraction — they default to None.
    """

    def setup_method(self) -> None:
        self.mapper = ActionMapper()

    def test_intent_extracted_from_structured_artifact(self) -> None:
        # GIVEN a structured artifact carrying both action and intent
        response = ATPResponse(
            task_id="intent-1",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={
                        "action": [0, 1, 2],
                        "intent": "go morning",
                    },
                ),
            ],
        )
        # WHEN the mapper extracts the action
        action = self.mapper.from_atp_response(response)
        # THEN the intent propagates verbatim
        assert action.intent == "go morning"

    def test_intent_extracted_from_json_content(self) -> None:
        # GIVEN a file artifact with JSON content carrying intent
        response = ATPResponse(
            task_id="intent-2",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(
                    path="action.json",
                    content='{"action": [0,1,2], "intent": "avoid crowd"}',
                ),
            ],
        )
        # WHEN the mapper parses the JSON content
        action = self.mapper.from_atp_response(response)
        # THEN the intent is lifted out of the parsed payload
        assert action.intent == "avoid crowd"

    def test_intent_absent_defaults_to_none(self) -> None:
        # GIVEN a response with only an action field
        response = ATPResponse(
            task_id="intent-3",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "cooperate"},
                ),
            ],
        )
        # WHEN the mapper extracts the action
        action = self.mapper.from_atp_response(response)
        # THEN intent defaults to None (no failure)
        assert action.intent is None

    def test_intent_wrong_type_int_returns_none(self) -> None:
        # GIVEN an intent field with an integer value
        response = ATPResponse(
            task_id="intent-4",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "cooperate", "intent": 42},
                ),
            ],
        )
        # WHEN the mapper extracts the action
        # THEN it does not raise and intent falls back to None
        action = self.mapper.from_atp_response(response)
        assert action.intent is None
        assert action.action == "cooperate"

    def test_intent_wrong_type_list_returns_none(self) -> None:
        # GIVEN an intent field with a list value
        response = ATPResponse(
            task_id="intent-5",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "cooperate", "intent": ["a", "b"]},
                ),
            ],
        )
        # WHEN the mapper extracts the action
        # THEN it does not raise and intent falls back to None
        action = self.mapper.from_atp_response(response)
        assert action.intent is None
        assert action.action == "cooperate"

    def test_intent_whitespace_only_returns_none(self) -> None:
        # GIVEN an intent that is only whitespace characters
        response = ATPResponse(
            task_id="intent-6",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "cooperate", "intent": "   \n\t  "},
                ),
            ],
        )
        # WHEN the mapper extracts the action
        action = self.mapper.from_atp_response(response)
        # THEN whitespace-only intent is treated as absent
        assert action.intent is None

    def test_intent_strips_surrounding_whitespace(self) -> None:
        # GIVEN an intent padded with surrounding whitespace
        response = ATPResponse(
            task_id="intent-7",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "cooperate", "intent": "  go morning  "},
                ),
            ],
        )
        # WHEN the mapper extracts the action
        action = self.mapper.from_atp_response(response)
        # THEN surrounding whitespace is stripped, content preserved
        assert action.intent == "go morning"

    def test_intent_truncated_at_500_chars(self) -> None:
        # GIVEN an intent string of 600 characters
        long_intent = "x" * 600
        response = ATPResponse(
            task_id="intent-8",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "cooperate", "intent": long_intent},
                ),
            ],
        )
        # WHEN the mapper extracts the action
        action = self.mapper.from_atp_response(response)
        # THEN the intent is truncated to exactly 500 characters
        assert action.intent is not None
        assert len(action.intent) == 500
        assert action.intent == "x" * 500
        # AND the action itself remains valid (truncation, not rejection)
        assert action.action == "cooperate"

    def test_intent_empty_string_returns_none(self) -> None:
        # GIVEN an empty string intent
        response = ATPResponse(
            task_id="intent-9",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={"action": "cooperate", "intent": ""},
                ),
            ],
        )
        # WHEN the mapper extracts the action
        action = self.mapper.from_atp_response(response)
        # THEN empty-string intent is treated as absent
        assert action.intent is None

    def test_intent_preserves_existing_fields(self) -> None:
        # GIVEN a structured artifact with action, message, reasoning, intent
        response = ATPResponse(
            task_id="intent-10",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="game_action",
                    data={
                        "action": "cooperate",
                        "message": "hi",
                        "reasoning": "because",
                        "intent": "collab",
                    },
                ),
            ],
        )
        # WHEN the mapper extracts the action
        action = self.mapper.from_atp_response(response)
        # THEN adding intent does not disturb previously-populated fields
        assert action.action == "cooperate"
        assert action.message == "hi"
        assert action.reasoning == "because"
        assert action.intent == "collab"
