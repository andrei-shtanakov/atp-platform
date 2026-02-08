"""Tests for ActionValidator."""

import random

from game_envs.core.action import (
    ContinuousActionSpace,
    DiscreteActionSpace,
    StructuredActionSpace,
)

from atp_games.mapping.action_mapper import GameAction
from atp_games.runner.action_validator import ActionValidator


class TestActionValidator:
    def setup_method(self) -> None:
        self.validator = ActionValidator(max_retries=3)

    def test_valid_discrete_action(self) -> None:
        space = DiscreteActionSpace(["cooperate", "defect"])
        action = GameAction(action="cooperate")
        result = self.validator.validate(action, space)
        assert result.valid
        assert result.attempts == 1
        assert not result.used_default

    def test_invalid_discrete_action(self) -> None:
        space = DiscreteActionSpace(["cooperate", "defect"])
        action = GameAction(action="surrender")
        result = self.validator.validate(action, space)
        assert not result.valid
        assert len(result.errors) == 1
        assert "surrender" in result.errors[0]

    def test_valid_continuous_action(self) -> None:
        space = ContinuousActionSpace(0.0, 100.0)
        action = GameAction(action=42.5)
        result = self.validator.validate(action, space)
        assert result.valid

    def test_invalid_continuous_action(self) -> None:
        space = ContinuousActionSpace(0.0, 100.0)
        action = GameAction(action=150.0)
        result = self.validator.validate(action, space)
        assert not result.valid

    def test_valid_structured_action(self) -> None:
        space = StructuredActionSpace(
            schema={
                "type": "allocation",
                "fields": ["a", "b"],
                "total": 10,
                "min_value": 0,
            }
        )
        action = GameAction(action={"a": 6.0, "b": 4.0})
        result = self.validator.validate(action, space)
        assert result.valid

    def test_invalid_structured_action(self) -> None:
        space = StructuredActionSpace(
            schema={
                "type": "allocation",
                "fields": ["a", "b"],
                "total": 10,
                "min_value": 0,
            }
        )
        action = GameAction(action={"a": 7.0, "b": 7.0})
        result = self.validator.validate(action, space)
        assert not result.valid

    def test_default_action(self) -> None:
        space = DiscreteActionSpace(["cooperate", "defect"])
        rng = random.Random(42)
        default = self.validator.get_default_action(space, rng)
        assert default.action in ["cooperate", "defect"]
        assert "Default" in (default.reasoning or "")

    def test_retry_prompt(self) -> None:
        space = DiscreteActionSpace(["a", "b", "c"])
        prompt = self.validator.build_retry_prompt("Invalid action 'x'", space)
        assert "invalid" in prompt.lower()
        assert "a, b, c" in prompt


class TestActionValidatorMaxRetries:
    def test_custom_max_retries(self) -> None:
        validator = ActionValidator(max_retries=5)
        assert validator.max_retries == 5

    def test_zero_retries(self) -> None:
        validator = ActionValidator(max_retries=0)
        assert validator.max_retries == 0
