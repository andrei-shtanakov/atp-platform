"""Tests for state data models."""

from __future__ import annotations

from game_envs.core.state import (
    GameState,
    Message,
    Observation,
    RoundResult,
    StepResult,
)


class TestMessage:
    def test_create(self) -> None:
        m = Message(sender="p1", content="hello", round_number=1)
        assert m.sender == "p1"
        assert m.content == "hello"
        assert m.round_number == 1

    def test_roundtrip(self) -> None:
        m = Message(sender="p1", content="hello", round_number=1)
        d = m.to_dict()
        m2 = Message.from_dict(d)
        assert m2.sender == m.sender
        assert m2.content == m.content
        assert m2.round_number == m.round_number

    def test_to_dict_keys(self) -> None:
        m = Message(sender="p1", content="hello", round_number=1)
        d = m.to_dict()
        assert set(d.keys()) == {
            "sender",
            "content",
            "round_number",
            "timestamp",
        }


class TestRoundResult:
    def test_create(self, sample_round_result: RoundResult) -> None:
        rr = sample_round_result
        assert rr.round_number == 1
        assert rr.actions == {
            "p1": "cooperate",
            "p2": "defect",
        }
        assert rr.payoffs == {"p1": 0.0, "p2": 5.0}
        assert len(rr.messages) == 1

    def test_roundtrip(self, sample_round_result: RoundResult) -> None:
        d = sample_round_result.to_dict()
        rr2 = RoundResult.from_dict(d)
        assert rr2.round_number == sample_round_result.round_number
        assert rr2.actions == sample_round_result.actions
        assert rr2.payoffs == sample_round_result.payoffs
        assert len(rr2.messages) == len(sample_round_result.messages)

    def test_default_messages(self) -> None:
        rr = RoundResult(
            round_number=1,
            actions={"p1": "A"},
            payoffs={"p1": 1.0},
        )
        assert rr.messages == []

    def test_from_dict_no_messages(self) -> None:
        d = {
            "round_number": 1,
            "actions": {"p1": "A"},
            "payoffs": {"p1": 1.0},
        }
        rr = RoundResult.from_dict(d)
        assert rr.messages == []


class TestGameState:
    def test_create(self) -> None:
        gs = GameState(
            round_number=3,
            player_states={"p1": {"score": 10}},
            public_state={"pot": 20},
        )
        assert gs.round_number == 3
        assert gs.is_terminal is False

    def test_terminal(self) -> None:
        gs = GameState(
            round_number=5,
            player_states={},
            public_state={},
            is_terminal=True,
        )
        assert gs.is_terminal is True

    def test_roundtrip(self) -> None:
        gs = GameState(
            round_number=2,
            player_states={
                "p1": {"score": 10},
                "p2": {"score": 5},
            },
            public_state={"pot": 15},
            is_terminal=False,
        )
        d = gs.to_dict()
        gs2 = GameState.from_dict(d)
        assert gs2.round_number == gs.round_number
        assert gs2.player_states == gs.player_states
        assert gs2.public_state == gs.public_state
        assert gs2.is_terminal == gs.is_terminal

    def test_from_dict_default_terminal(self) -> None:
        d = {
            "round_number": 1,
            "player_states": {},
            "public_state": {},
        }
        gs = GameState.from_dict(d)
        assert gs.is_terminal is False


class TestObservation:
    def test_create(self, sample_observation: Observation) -> None:
        obs = sample_observation
        assert obs.player_id == "p1"
        assert obs.round_number == 2
        assert obs.total_rounds == 5

    def test_to_prompt(self, sample_observation: Observation) -> None:
        prompt = sample_observation.to_prompt()
        assert "p1" in prompt
        assert "Round 2 of 5" in prompt
        assert "score" in prompt
        assert "cooperate" in prompt
        assert "defect" in prompt
        assert "History" in prompt
        assert "Messages" in prompt
        assert "hi" in prompt

    def test_to_prompt_no_history(self) -> None:
        obs = Observation(
            player_id="p1",
            game_state={"value": 42},
            available_actions=["bid"],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        prompt = obs.to_prompt()
        assert "History" not in prompt
        assert "value" in prompt
        assert "bid" in prompt

    def test_to_dict(self, sample_observation: Observation) -> None:
        d = sample_observation.to_dict()
        assert d["player_id"] == "p1"
        assert d["round_number"] == 2
        assert d["total_rounds"] == 5
        assert len(d["history"]) == 1
        assert len(d["messages"]) == 1
        assert "game_state" in d
        assert "available_actions" in d

    def test_roundtrip(self, sample_observation: Observation) -> None:
        d = sample_observation.to_dict()
        obs2 = Observation.from_dict(d)
        assert obs2.player_id == sample_observation.player_id
        assert obs2.round_number == sample_observation.round_number
        assert obs2.total_rounds == sample_observation.total_rounds
        assert obs2.game_state == sample_observation.game_state
        assert obs2.available_actions == sample_observation.available_actions
        assert len(obs2.history) == len(sample_observation.history)
        assert len(obs2.messages) == len(sample_observation.messages)

    def test_from_dict_defaults(self) -> None:
        d = {
            "player_id": "p1",
            "game_state": {},
            "available_actions": [],
            "round_number": 1,
            "total_rounds": 1,
        }
        obs = Observation.from_dict(d)
        assert obs.history == []
        assert obs.messages == []


class TestStepResult:
    def test_create(self) -> None:
        state = GameState(
            round_number=1,
            player_states={},
            public_state={},
        )
        obs = Observation(
            player_id="p1",
            game_state={},
            available_actions=["A"],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        sr = StepResult(
            state=state,
            observations={"p1": obs},
            payoffs={"p1": 1.0},
            is_terminal=False,
        )
        assert sr.payoffs == {"p1": 1.0}
        assert sr.is_terminal is False
        assert sr.info == {}

    def test_roundtrip(self) -> None:
        state = GameState(
            round_number=1,
            player_states={"p1": {"x": 1}},
            public_state={"y": 2},
        )
        obs = Observation(
            player_id="p1",
            game_state={"x": 1},
            available_actions=["A", "B"],
            history=[],
            round_number=1,
            total_rounds=3,
        )
        sr = StepResult(
            state=state,
            observations={"p1": obs},
            payoffs={"p1": 2.5},
            is_terminal=False,
            info={"extra": "data"},
        )
        d = sr.to_dict()
        sr2 = StepResult.from_dict(d)
        assert sr2.payoffs == sr.payoffs
        assert sr2.is_terminal == sr.is_terminal
        assert sr2.info == sr.info
        assert sr2.state.round_number == sr.state.round_number
