"""Fairness metrics for payoff distribution analysis.

Measures fairness of outcomes including inequality (Gini),
envy-freeness, proportionality, and utilitarian welfare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FairnessMetrics:
    """Results of fairness analysis.

    Attributes:
        gini: Gini coefficient [0, 1]. 0 = perfect equality.
        envy_free: Whether no player envies another.
        envy_pairs: Pairs (i, j) where player i envies player j.
        proportionality: Proportional fairness score [0, 1].
        utilitarian_welfare: Sum of all payoffs.
    """

    gini: float
    envy_free: bool
    envy_pairs: list[tuple[str, str]]
    proportionality: float
    utilitarian_welfare: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "gini": self.gini,
            "envy_free": self.envy_free,
            "envy_pairs": [list(p) for p in self.envy_pairs],
            "proportionality": self.proportionality,
            "utilitarian_welfare": self.utilitarian_welfare,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FairnessMetrics:
        """Deserialize from dictionary."""
        return cls(
            gini=data["gini"],
            envy_free=data["envy_free"],
            envy_pairs=[(p[0], p[1]) for p in data["envy_pairs"]],
            proportionality=data["proportionality"],
            utilitarian_welfare=data["utilitarian_welfare"],
        )


def gini_coefficient(payoffs: dict[str, float]) -> float:
    """Compute the Gini coefficient of payoff distribution.

    The Gini coefficient measures inequality: 0 means perfect
    equality, 1 means maximum inequality.

    Args:
        payoffs: Player ID to total payoff mapping.

    Returns:
        Gini coefficient in [0, 1].

    Raises:
        ValueError: If payoffs is empty.
    """
    if not payoffs:
        raise ValueError("Cannot compute Gini from empty payoffs")

    values = sorted(payoffs.values())
    n = len(values)

    if n == 1:
        return 0.0

    total = sum(values)
    if total == 0.0:
        return 0.0

    # Mean absolute difference formula
    numerator = sum(abs(values[i] - values[j]) for i in range(n) for j in range(n))
    denominator = 2 * n * total

    return numerator / denominator


def envy_freeness(
    allocations: dict[str, float],
) -> tuple[bool, list[tuple[str, str]]]:
    """Check if a payoff allocation is envy-free.

    An allocation is envy-free if no player prefers another
    player's payoff to their own.

    Args:
        allocations: Player ID to payoff mapping.

    Returns:
        Tuple of (is_envy_free, list of envy pairs (i, j)
        where player i envies player j).

    Raises:
        ValueError: If allocations is empty.
    """
    if not allocations:
        raise ValueError("Cannot check envy-freeness with empty allocations")

    envy_pairs: list[tuple[str, str]] = []
    players = list(allocations.keys())

    for i, p_i in enumerate(players):
        for j, p_j in enumerate(players):
            if i != j and allocations[p_i] < allocations[p_j]:
                envy_pairs.append((p_i, p_j))

    return len(envy_pairs) == 0, envy_pairs


def proportionality(
    payoffs: dict[str, float],
    entitlements: dict[str, float] | None = None,
) -> float:
    """Compute proportional fairness score.

    Measures how close the actual payoff distribution is to
    the entitled (ideal) distribution. Returns 1.0 when payoffs
    match entitlements perfectly.

    Args:
        payoffs: Player ID to actual payoff mapping.
        entitlements: Player ID to entitled share. If None,
            assumes equal entitlement (1/n each).

    Returns:
        Proportionality score in [0, 1].

    Raises:
        ValueError: If payoffs is empty or entitlements don't
            match payoff players.
    """
    if not payoffs:
        raise ValueError("Cannot compute proportionality from empty payoffs")

    players = list(payoffs.keys())
    n = len(players)

    if entitlements is None:
        entitlements = {p: 1.0 / n for p in players}
    else:
        missing = set(players) - set(entitlements.keys())
        if missing:
            raise ValueError(f"Entitlements missing for players: {missing}")

    total_payoff = sum(payoffs.values())
    total_entitlement = sum(entitlements[p] for p in players)

    if total_payoff == 0.0:
        # All payoffs zero — everyone gets equal (nothing)
        return 1.0
    if total_entitlement == 0.0:
        return 0.0

    # Compute actual shares vs entitled shares
    actual_shares = {p: payoffs[p] / total_payoff for p in players}
    entitled_shares = {p: entitlements[p] / total_entitlement for p in players}

    # 1 - normalized L1 distance between distributions
    l1_distance = sum(abs(actual_shares[p] - entitled_shares[p]) for p in players)
    # Max L1 distance between two distributions is 2
    return 1.0 - l1_distance / 2.0


def utilitarian_welfare(payoffs: dict[str, float]) -> float:
    """Compute utilitarian welfare (sum of payoffs).

    Args:
        payoffs: Player ID to payoff mapping.

    Returns:
        Total sum of all payoffs.

    Raises:
        ValueError: If payoffs is empty.
    """
    if not payoffs:
        raise ValueError("Cannot compute welfare from empty payoffs")

    return sum(payoffs.values())
