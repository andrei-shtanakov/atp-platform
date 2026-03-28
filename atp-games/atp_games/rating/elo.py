"""Elo rating system for agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EloRating:
    """Elo rating for a single agent.

    Attributes:
        agent: Agent identifier.
        rating: Current Elo rating (default 1500.0).
        games_played: Total number of games played.
        wins: Number of wins.
        losses: Number of losses.
        draws: Number of draws.
        history: List of rating values after each game.
    """

    agent: str
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    history: list[float] = field(default_factory=list)


class EloCalculator:
    """Computes Elo rating updates for pairwise agent matches.

    Uses standard Elo formula:
        E = 1 / (1 + 10^((Rb - Ra) / 400))
        R' = R + K * (S - E)

    where S is 1.0 for win, 0.0 for loss, 0.5 for draw.

    Attributes:
        k_factor: Sensitivity of rating changes per game.
        initial_rating: Default starting rating for new agents.
    """

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0) -> None:
        """Initialize calculator.

        Args:
            k_factor: K-factor controlling rating change magnitude.
            initial_rating: Starting rating for new agents.
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def create_rating(self, agent: str) -> EloRating:
        """Create a new EloRating with default values.

        Args:
            agent: Agent identifier.

        Returns:
            New EloRating at the initial rating.
        """
        return EloRating(agent=agent, rating=self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Compute expected score for agent A against agent B.

        Args:
            rating_a: Current rating of agent A.
            rating_b: Current rating of agent B.

        Returns:
            Expected score in [0, 1].
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        rating_a: EloRating,
        rating_b: EloRating,
        winner: str | None,
    ) -> None:
        """Update ratings in place after a match.

        Args:
            rating_a: Rating object for agent A (mutated).
            rating_b: Rating object for agent B (mutated).
            winner: Agent identifier of the winner, or None for draw.

        Raises:
            ValueError: If winner is not None, agent A, or agent B.
        """
        if winner is not None and winner not in (rating_a.agent, rating_b.agent):
            raise ValueError(
                f"winner '{winner}' is neither '{rating_a.agent}'"
                f" nor '{rating_b.agent}'"
            )

        ea = self.expected_score(rating_a.rating, rating_b.rating)
        eb = 1.0 - ea

        if winner is None:
            sa, sb = 0.5, 0.5
            rating_a.draws += 1
            rating_b.draws += 1
        elif winner == rating_a.agent:
            sa, sb = 1.0, 0.0
            rating_a.wins += 1
            rating_b.losses += 1
        else:
            sa, sb = 0.0, 1.0
            rating_b.wins += 1
            rating_a.losses += 1

        rating_a.rating += self.k_factor * (sa - ea)
        rating_b.rating += self.k_factor * (sb - eb)

        rating_a.games_played += 1
        rating_b.games_played += 1

        rating_a.history.append(rating_a.rating)
        rating_b.history.append(rating_b.rating)
