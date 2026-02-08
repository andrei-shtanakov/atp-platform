"""Tests for ObservationMapper."""

from game_envs.core.state import Message, Observation, RoundResult

from atp_games.mapping.observation_mapper import ObservationMapper


class TestObservationMapper:
    def setup_method(self) -> None:
        self.mapper = ObservationMapper()

    def test_basic_mapping(self) -> None:
        obs = Observation(
            player_id="player_0",
            game_state={"key": "value"},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=1,
            total_rounds=5,
        )
        request = self.mapper.to_atp_request(obs, "Prisoner's Dilemma", episode=0)

        assert request.task_id
        assert "Prisoner's Dilemma" in request.task.description
        assert "Round 1/5" in request.task.description
        assert request.context is not None
        assert request.context.environment is not None
        assert request.context.environment["player_id"] == "player_0"
        assert request.metadata is not None
        assert request.metadata["episode"] == 0
        assert request.metadata["game_type"] == "game_theoretic"

    def test_task_id_format(self) -> None:
        obs = Observation(
            player_id="p0",
            game_state={},
            available_actions=["a", "b"],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        request = self.mapper.to_atp_request(obs, "Test Game", episode=3)
        # task_id should only contain valid chars
        assert all(c.isalnum() or c in "_-" for c in request.task_id)

    def test_with_history(self) -> None:
        history = [
            RoundResult(
                round_number=0,
                actions={"p0": "cooperate", "p1": "defect"},
                payoffs={"p0": 0.0, "p1": 5.0},
            ),
        ]
        obs = Observation(
            player_id="p0",
            game_state={},
            available_actions=["cooperate", "defect"],
            history=history,
            round_number=1,
            total_rounds=5,
        )
        request = self.mapper.to_atp_request(obs, "PD", episode=0)
        assert request.metadata is not None
        assert len(request.metadata["history"]) == 1

    def test_with_messages(self) -> None:
        msgs = [
            Message(
                sender="p1",
                content="Let's cooperate",
                round_number=1,
            ),
        ]
        obs = Observation(
            player_id="p0",
            game_state={},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=1,
            total_rounds=3,
            messages=msgs,
        )
        request = self.mapper.to_atp_request(obs, "PD", episode=0)
        assert request.metadata is not None
        assert len(request.metadata["messages"]) == 1

    def test_metadata_contains_game_state(self) -> None:
        obs = Observation(
            player_id="p0",
            game_state={"score": 42},
            available_actions=["a"],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        request = self.mapper.to_atp_request(obs, "Test", episode=0)
        assert request.metadata is not None
        assert request.metadata["game_state"] == {"score": 42}
        assert request.metadata["available_actions"] == ["a"]

    def test_prompt_includes_response_format(self) -> None:
        obs = Observation(
            player_id="p0",
            game_state={},
            available_actions=["x"],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        request = self.mapper.to_atp_request(obs, "G", episode=0)
        assert '"action"' in request.task.description
        assert "JSON" in request.task.description
