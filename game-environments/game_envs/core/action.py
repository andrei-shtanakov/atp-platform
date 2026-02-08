"""Action space abstractions for game environments."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any


class ActionSpace(ABC):
    """Abstract base for action spaces."""

    @abstractmethod
    def contains(self, action: Any) -> bool:
        """Check if an action is valid in this space."""
        ...

    @abstractmethod
    def sample(self, rng: random.Random | None = None) -> Any:
        """Sample a random valid action."""
        ...

    @abstractmethod
    def to_list(self) -> list[Any]:
        """List all valid actions (or a description for continuous)."""
        ...

    @abstractmethod
    def to_description(self) -> str:
        """Human-readable description for LLM agents."""
        ...


class DiscreteActionSpace(ActionSpace):
    """Finite set of named actions (e.g., cooperate/defect)."""

    def __init__(self, actions: list[str]) -> None:
        if not actions:
            raise ValueError("actions must be non-empty")
        self.actions = list(actions)

    def contains(self, action: Any) -> bool:
        return action in self.actions

    def sample(self, rng: random.Random | None = None) -> str:
        r = rng or random.Random()
        return r.choice(self.actions)

    def to_list(self) -> list[str]:
        return list(self.actions)

    def to_description(self) -> str:
        return f"Choose one of: {', '.join(self.actions)}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiscreteActionSpace):
            return NotImplemented
        return self.actions == other.actions

    def __repr__(self) -> str:
        return f"DiscreteActionSpace({self.actions!r})"


class ContinuousActionSpace(ActionSpace):
    """Continuous range [low, high]."""

    def __init__(
        self,
        low: float,
        high: float,
        description: str = "",
    ) -> None:
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        self.low = low
        self.high = high
        self._description = description

    def contains(self, action: Any) -> bool:
        try:
            val = float(action)
        except (TypeError, ValueError):
            return False
        return self.low <= val <= self.high

    def sample(self, rng: random.Random | None = None) -> float:
        r = rng or random.Random()
        return r.uniform(self.low, self.high)

    def to_list(self) -> list[str]:
        return [f"[{self.low}, {self.high}]"]

    def to_description(self) -> str:
        if self._description:
            return self._description
        return f"Choose a number between {self.low} and {self.high}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContinuousActionSpace):
            return NotImplemented
        return self.low == other.low and self.high == other.high

    def __repr__(self) -> str:
        return f"ContinuousActionSpace(low={self.low}, high={self.high})"


class StructuredActionSpace(ActionSpace):
    """Structured action space (e.g., allocation vectors).

    Schema defines the structure:
        {
            "type": "allocation",
            "fields": ["field_1", "field_2", ...],
            "total": 100,           # optional: sum constraint
            "min_value": 0,         # optional: per-field minimum
            "max_value": 100,       # optional: per-field maximum
        }
    """

    def __init__(
        self,
        schema: dict[str, Any],
        description: str = "",
    ) -> None:
        if not schema:
            raise ValueError("schema must be non-empty")
        self.schema = dict(schema)
        self._description = description

    def contains(self, action: Any) -> bool:
        if not isinstance(action, dict):
            return False
        fields = self.schema.get("fields", [])
        if fields and set(action.keys()) != set(fields):
            return False
        min_val = self.schema.get("min_value")
        max_val = self.schema.get("max_value")
        for v in action.values():
            try:
                val = float(v)
            except (TypeError, ValueError):
                return False
            if min_val is not None and val < min_val:
                return False
            if max_val is not None and val > max_val:
                return False
        total = self.schema.get("total")
        if total is not None:
            actual = sum(float(v) for v in action.values())
            if abs(actual - total) > 1e-9:
                return False
        return True

    def sample(self, rng: random.Random | None = None) -> dict[str, float]:
        r = rng or random.Random()
        fields = self.schema.get("fields", ["value"])
        total = self.schema.get("total")
        min_val = self.schema.get("min_value", 0.0)
        max_val = self.schema.get("max_value")

        if total is not None:
            # Generate random allocation summing to total
            n = len(fields)
            weights = [r.random() for _ in range(n)]
            weight_sum = sum(weights)
            values = [max(min_val, w / weight_sum * total) for w in weights]
            # Adjust to ensure exact sum
            diff = total - sum(values)
            values[0] += diff
            return dict(zip(fields, values))

        # Independent random values
        result: dict[str, float] = {}
        for f in fields:
            lo = min_val
            hi = max_val if max_val is not None else 100.0
            result[f] = r.uniform(lo, hi)
        return result

    def to_list(self) -> list[str]:
        fields = self.schema.get("fields", [])
        total = self.schema.get("total")
        parts = [f"fields: {fields}"]
        if total is not None:
            parts.append(f"must sum to {total}")
        return parts

    def to_description(self) -> str:
        if self._description:
            return self._description
        fields = self.schema.get("fields", [])
        total = self.schema.get("total")
        desc = f"Provide values for: {', '.join(fields)}"
        if total is not None:
            desc += f" (must sum to {total})"
        return desc

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructuredActionSpace):
            return NotImplemented
        return self.schema == other.schema

    def __repr__(self) -> str:
        return f"StructuredActionSpace(schema={self.schema!r})"
