"""Tests for Strategy ABC."""

from __future__ import annotations

from game_envs.core.state import Observation


class TestStubStrategy:
    def test_name(self, stub_strategy) -> None:  # type: ignore[no-untyped-def]
        assert stub_strategy.name == "stub_strategy"

    def test_choose_action(self, stub_strategy) -> None:  # type: ignore[no-untyped-def]
        obs = Observation(
            player_id="p1",
            game_state={},
            available_actions=["cooperate", "defect"],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        action = stub_strategy.choose_action(obs)
        assert action == "cooperate"

    def test_choose_action_empty(self, stub_strategy) -> None:  # type: ignore[no-untyped-def]
        obs = Observation(
            player_id="p1",
            game_state={},
            available_actions=[],
            history=[],
            round_number=1,
            total_rounds=1,
        )
        action = stub_strategy.choose_action(obs)
        assert action is None

    def test_reset(self, stub_strategy) -> None:  # type: ignore[no-untyped-def]
        # Reset should not raise
        stub_strategy.reset()

    def test_repr(self, stub_strategy) -> None:  # type: ignore[no-untyped-def]
        r = repr(stub_strategy)
        assert "StubStrategy" in r
        assert "stub_strategy" in r
