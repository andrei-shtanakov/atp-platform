"""Tests for ActionSpace classes."""

from __future__ import annotations

import random

import pytest

from game_envs.core.action import (
    ContinuousActionSpace,
    DiscreteActionSpace,
    StructuredActionSpace,
)


class TestDiscreteActionSpace:
    def test_contains_valid(self, discrete_space: DiscreteActionSpace) -> None:
        assert discrete_space.contains("cooperate")
        assert discrete_space.contains("defect")

    def test_contains_invalid(self, discrete_space: DiscreteActionSpace) -> None:
        assert not discrete_space.contains("invalid")
        assert not discrete_space.contains(42)
        assert not discrete_space.contains(None)

    def test_sample_returns_valid(self, discrete_space: DiscreteActionSpace) -> None:
        rng = random.Random(0)
        for _ in range(50):
            action = discrete_space.sample(rng)
            assert discrete_space.contains(action)

    def test_sample_deterministic_with_seed(
        self, discrete_space: DiscreteActionSpace
    ) -> None:
        r1 = random.Random(42)
        r2 = random.Random(42)
        results1 = [discrete_space.sample(r1) for _ in range(10)]
        results2 = [discrete_space.sample(r2) for _ in range(10)]
        assert results1 == results2

    def test_to_list(self, discrete_space: DiscreteActionSpace) -> None:
        actions = discrete_space.to_list()
        assert actions == ["cooperate", "defect"]
        # Ensure it returns a copy
        actions.append("foo")
        assert discrete_space.to_list() == [
            "cooperate",
            "defect",
        ]

    def test_to_description(self, discrete_space: DiscreteActionSpace) -> None:
        desc = discrete_space.to_description()
        assert "cooperate" in desc
        assert "defect" in desc

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            DiscreteActionSpace([])

    def test_equality(self) -> None:
        a = DiscreteActionSpace(["x", "y"])
        b = DiscreteActionSpace(["x", "y"])
        c = DiscreteActionSpace(["y", "x"])
        assert a == b
        assert a != c

    def test_repr(self, discrete_space: DiscreteActionSpace) -> None:
        r = repr(discrete_space)
        assert "DiscreteActionSpace" in r
        assert "cooperate" in r


class TestContinuousActionSpace:
    def test_contains_valid(self, continuous_space: ContinuousActionSpace) -> None:
        assert continuous_space.contains(0.0)
        assert continuous_space.contains(50.0)
        assert continuous_space.contains(100.0)

    def test_contains_boundaries(self, continuous_space: ContinuousActionSpace) -> None:
        assert continuous_space.contains(0.0)
        assert continuous_space.contains(100.0)

    def test_contains_out_of_range(
        self, continuous_space: ContinuousActionSpace
    ) -> None:
        assert not continuous_space.contains(-0.1)
        assert not continuous_space.contains(100.1)

    def test_contains_invalid_type(
        self, continuous_space: ContinuousActionSpace
    ) -> None:
        assert not continuous_space.contains("not_a_number")
        assert not continuous_space.contains(None)
        assert not continuous_space.contains([1, 2])

    def test_contains_string_number(
        self, continuous_space: ContinuousActionSpace
    ) -> None:
        # Strings that can be converted to float
        assert continuous_space.contains("50.0")
        assert continuous_space.contains("0")

    def test_sample_in_range(self, continuous_space: ContinuousActionSpace) -> None:
        rng = random.Random(0)
        for _ in range(100):
            val = continuous_space.sample(rng)
            assert continuous_space.contains(val)

    def test_to_list(self, continuous_space: ContinuousActionSpace) -> None:
        result = continuous_space.to_list()
        assert len(result) == 1
        assert "[0.0, 100.0]" in result[0]

    def test_to_description_default(self) -> None:
        s = ContinuousActionSpace(0.0, 10.0)
        desc = s.to_description()
        assert "0.0" in desc
        assert "10.0" in desc

    def test_to_description_custom(self) -> None:
        s = ContinuousActionSpace(0.0, 10.0, description="Place your bid")
        assert s.to_description() == "Place your bid"

    def test_invalid_range(self) -> None:
        with pytest.raises(ValueError, match="less than"):
            ContinuousActionSpace(10.0, 5.0)
        with pytest.raises(ValueError, match="less than"):
            ContinuousActionSpace(5.0, 5.0)

    def test_equality(self) -> None:
        a = ContinuousActionSpace(0.0, 1.0)
        b = ContinuousActionSpace(0.0, 1.0)
        c = ContinuousActionSpace(0.0, 2.0)
        assert a == b
        assert a != c


class TestStructuredActionSpace:
    def test_contains_valid(self, structured_space: StructuredActionSpace) -> None:
        action = {
            "field_a": 30.0,
            "field_b": 40.0,
            "field_c": 30.0,
        }
        assert structured_space.contains(action)

    def test_contains_wrong_fields(
        self, structured_space: StructuredActionSpace
    ) -> None:
        action = {"field_a": 50.0, "field_b": 50.0}
        assert not structured_space.contains(action)

    def test_contains_wrong_sum(self, structured_space: StructuredActionSpace) -> None:
        action = {
            "field_a": 50.0,
            "field_b": 50.0,
            "field_c": 50.0,
        }
        assert not structured_space.contains(action)

    def test_contains_negative(self, structured_space: StructuredActionSpace) -> None:
        action = {
            "field_a": -10.0,
            "field_b": 60.0,
            "field_c": 50.0,
        }
        assert not structured_space.contains(action)

    def test_contains_not_dict(self, structured_space: StructuredActionSpace) -> None:
        assert not structured_space.contains("invalid")
        assert not structured_space.contains(42)
        assert not structured_space.contains(None)

    def test_contains_non_numeric_value(
        self, structured_space: StructuredActionSpace
    ) -> None:
        action = {
            "field_a": "abc",
            "field_b": 50.0,
            "field_c": 50.0,
        }
        assert not structured_space.contains(action)

    def test_sample_valid(self, structured_space: StructuredActionSpace) -> None:
        rng = random.Random(0)
        for _ in range(50):
            action = structured_space.sample(rng)
            assert isinstance(action, dict)
            assert set(action.keys()) == {
                "field_a",
                "field_b",
                "field_c",
            }
            assert abs(sum(action.values()) - 100) < 1e-6

    def test_sample_no_total(self) -> None:
        s = StructuredActionSpace(
            schema={
                "fields": ["x", "y"],
                "min_value": 0,
                "max_value": 10,
            }
        )
        rng = random.Random(0)
        action = s.sample(rng)
        assert "x" in action
        assert "y" in action

    def test_to_description(self, structured_space: StructuredActionSpace) -> None:
        desc = structured_space.to_description()
        assert "field_a" in desc
        assert "100" in desc

    def test_to_description_custom(self) -> None:
        s = StructuredActionSpace(
            schema={"fields": ["x"]},
            description="Allocate troops",
        )
        assert s.to_description() == "Allocate troops"

    def test_empty_schema_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            StructuredActionSpace(schema={})

    def test_max_value_constraint(self) -> None:
        s = StructuredActionSpace(
            schema={
                "fields": ["x"],
                "min_value": 0,
                "max_value": 10,
            }
        )
        assert s.contains({"x": 5.0})
        assert not s.contains({"x": 15.0})

    def test_equality(self) -> None:
        a = StructuredActionSpace(schema={"fields": ["x"], "total": 10})
        b = StructuredActionSpace(schema={"fields": ["x"], "total": 10})
        c = StructuredActionSpace(schema={"fields": ["y"], "total": 10})
        assert a == b
        assert a != c

    def test_to_list(self, structured_space: StructuredActionSpace) -> None:
        result = structured_space.to_list()
        assert len(result) >= 1
