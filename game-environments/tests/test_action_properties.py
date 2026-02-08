"""Property-based tests for ActionSpace using Hypothesis."""

from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from game_envs.core.action import (
    ContinuousActionSpace,
    DiscreteActionSpace,
    StructuredActionSpace,
)


class TestDiscreteActionSpaceProperties:
    @given(
        actions=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=10,
            unique=True,
        ),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=100)
    def test_sample_always_contained(self, actions: list[str], seed: int) -> None:
        """Sampled actions are always contained."""
        space = DiscreteActionSpace(actions)
        rng = random.Random(seed)
        for _ in range(10):
            action = space.sample(rng)
            assert space.contains(action)

    @given(
        actions=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=10,
            unique=True,
        ),
    )
    @settings(max_examples=50)
    def test_to_list_all_contained(self, actions: list[str]) -> None:
        """All listed actions are contained."""
        space = DiscreteActionSpace(actions)
        for action in space.to_list():
            assert space.contains(action)

    @given(
        actions=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=10,
            unique=True,
        ),
    )
    @settings(max_examples=50)
    def test_contains_iff_in_list(self, actions: list[str]) -> None:
        """Action contained iff it appears in to_list."""
        space = DiscreteActionSpace(actions)
        listed = set(space.to_list())
        for a in actions:
            assert space.contains(a) == (a in listed)

    @given(
        actions=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=10,
            unique=True,
        ),
        foreign=st.text(min_size=21, max_size=30),
    )
    @settings(max_examples=50)
    def test_foreign_not_contained(self, actions: list[str], foreign: str) -> None:
        """Actions outside the set are not contained."""
        space = DiscreteActionSpace(actions)
        # A string longer than max action can't be in set
        if foreign not in actions:
            assert not space.contains(foreign)


class TestContinuousActionSpaceProperties:
    @given(
        low=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        width=st.floats(
            min_value=0.001,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=100)
    def test_sample_always_contained(self, low: float, width: float, seed: int) -> None:
        """Sampled values are always contained."""
        high = low + width
        space = ContinuousActionSpace(low, high)
        rng = random.Random(seed)
        for _ in range(10):
            val = space.sample(rng)
            assert space.contains(val)

    @given(
        low=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        width=st.floats(
            min_value=0.001,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50)
    def test_boundaries_contained(self, low: float, width: float) -> None:
        """Boundary values are contained."""
        high = low + width
        space = ContinuousActionSpace(low, high)
        assert space.contains(low)
        assert space.contains(high)

    @given(
        low=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        width=st.floats(
            min_value=0.001,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        offset=st.floats(
            min_value=0.001,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50)
    def test_outside_not_contained(
        self, low: float, width: float, offset: float
    ) -> None:
        """Values outside the range are not contained."""
        high = low + width
        space = ContinuousActionSpace(low, high)
        assert not space.contains(low - offset)
        assert not space.contains(high + offset)


class TestStructuredActionSpaceProperties:
    @given(
        n_fields=st.integers(min_value=1, max_value=5),
        total=st.floats(
            min_value=1.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=100)
    def test_sample_sum_matches_total(
        self, n_fields: int, total: float, seed: int
    ) -> None:
        """Sampled allocations sum to total."""
        fields = [f"f_{i}" for i in range(n_fields)]
        space = StructuredActionSpace(
            schema={
                "fields": fields,
                "total": total,
                "min_value": 0,
            }
        )
        rng = random.Random(seed)
        action = space.sample(rng)
        assert abs(sum(action.values()) - total) < 1e-6

    @given(
        n_fields=st.integers(min_value=1, max_value=5),
        total=st.floats(
            min_value=10.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=100)
    def test_sample_always_contained(
        self, n_fields: int, total: float, seed: int
    ) -> None:
        """Sampled allocations are always valid."""
        fields = [f"f_{i}" for i in range(n_fields)]
        space = StructuredActionSpace(
            schema={
                "fields": fields,
                "total": total,
                "min_value": 0,
            }
        )
        rng = random.Random(seed)
        action = space.sample(rng)
        assert space.contains(action)

    @given(
        n_fields=st.integers(min_value=2, max_value=5),
        total=st.floats(
            min_value=10.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50)
    def test_wrong_sum_not_contained(self, n_fields: int, total: float) -> None:
        """Allocations with wrong sum are rejected."""
        fields = [f"f_{i}" for i in range(n_fields)]
        space = StructuredActionSpace(
            schema={
                "fields": fields,
                "total": total,
                "min_value": 0,
            }
        )
        # Create action with double the total
        action = {f: total for f in fields}
        assert not space.contains(action)
