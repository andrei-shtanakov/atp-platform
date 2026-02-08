"""Alympics-style benchmark suite: composite scoring across all 5 games.

Inspired by the Alympics paper, this module defines a standardized benchmark
battery covering all 5 canonical games with composite scoring across four
categories: strategic reasoning, cooperation, fairness, and robustness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atp_games.models import GameResult, GameRunConfig

# ---------------------------------------------------------------------------
# Category definitions with default weights
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS: dict[str, float] = {
    "strategic": 0.30,
    "cooperation": 0.25,
    "fairness": 0.25,
    "robustness": 0.20,
}

# Which games contribute to which category, with per-game weight within that
# category (weights are normalised internally).
CATEGORY_GAME_MAP: dict[str, dict[str, float]] = {
    "strategic": {
        "prisoners_dilemma": 0.25,
        "auction": 0.25,
        "colonel_blotto": 0.25,
        "congestion": 0.25,
    },
    "cooperation": {
        "prisoners_dilemma": 0.50,
        "public_goods": 0.50,
    },
    "fairness": {
        "public_goods": 0.40,
        "auction": 0.30,
        "congestion": 0.30,
    },
    "robustness": {
        "prisoners_dilemma": 0.20,
        "public_goods": 0.20,
        "auction": 0.20,
        "colonel_blotto": 0.20,
        "congestion": 0.20,
    },
}

# Per-game baseline payoffs used to normalise raw scores to 0-100.
# These represent the range [worst_baseline, best_baseline] payoffs.
GAME_BASELINES: dict[str, dict[str, float]] = {
    "prisoners_dilemma": {"min": 1.0, "max": 3.0},
    "public_goods": {"min": 0.0, "max": 10.0},
    "auction": {"min": 0.0, "max": 50.0},
    "colonel_blotto": {"min": 0.0, "max": 1.0},
    "congestion": {"min": -20.0, "max": -1.0},
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CategoryScore:
    """Score for a single evaluation category.

    Attributes:
        name: Category name (strategic / cooperation / fairness / robustness).
        score: Normalised score 0-100.
        weight: Category weight in composite score.
        game_scores: Per-game normalised scores contributing to this category.
    """

    name: str
    score: float
    weight: float
    game_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "weight": self.weight,
            "game_scores": {k: round(v, 1) for k, v in self.game_scores.items()},
        }


@dataclass
class AlympicsResult:
    """Complete Alympics benchmark result.

    Attributes:
        agent_name: Name of the evaluated agent.
        composite_score: Weighted overall score 0-100.
        categories: Per-category scores.
        game_results: Raw GameResult per game type.
    """

    agent_name: str
    composite_score: float
    categories: dict[str, CategoryScore] = field(default_factory=dict)
    game_results: dict[str, GameResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_name": self.agent_name,
            "composite_score": round(self.composite_score, 1),
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "game_results": {k: v.to_dict() for k, v in self.game_results.items()},
        }

    def summary(self) -> str:
        """Human-readable one-line summary.

        Example:
            agent X scored 72/100 (strategic: 85, cooperation: 60,
            fairness: 78, robustness: 65)
        """
        parts = [f"{cat.name}: {cat.score:.0f}" for cat in self.categories.values()]
        return (
            f"{self.agent_name} scored {self.composite_score:.0f}/100 "
            f"({', '.join(parts)})"
        )


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def normalise_payoff(
    raw: float,
    game_type: str,
    baselines: dict[str, dict[str, float]] | None = None,
) -> float:
    """Normalise a raw average payoff to 0-100 using baseline bounds.

    Args:
        raw: Raw average payoff.
        game_type: Game type key (e.g. ``"prisoners_dilemma"``).
        baselines: Optional custom baselines; defaults to
            ``GAME_BASELINES``.

    Returns:
        Normalised score clipped to [0, 100].
    """
    baselines = baselines or GAME_BASELINES
    bounds = baselines.get(game_type)
    if bounds is None:
        return 50.0

    lo = bounds["min"]
    hi = bounds["max"]
    if hi <= lo:
        return 50.0

    normalised = (raw - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, normalised))


def compute_category_scores(
    game_scores: dict[str, float],
    category_weights: dict[str, float] | None = None,
    category_game_map: dict[str, dict[str, float]] | None = None,
) -> dict[str, CategoryScore]:
    """Compute per-category scores from per-game normalised scores.

    Args:
        game_scores: Mapping game_type -> normalised score (0-100).
        category_weights: Category name -> weight. Defaults to
            ``CATEGORY_WEIGHTS``.
        category_game_map: Category -> {game -> weight within category}.
            Defaults to ``CATEGORY_GAME_MAP``.

    Returns:
        Dict of category name -> ``CategoryScore``.
    """
    category_weights = category_weights or CATEGORY_WEIGHTS
    category_game_map = category_game_map or CATEGORY_GAME_MAP

    categories: dict[str, CategoryScore] = {}

    for cat_name, cat_weight in category_weights.items():
        game_map = category_game_map.get(cat_name, {})
        total_weight = 0.0
        weighted_sum = 0.0
        per_game: dict[str, float] = {}

        for game_type, game_weight in game_map.items():
            if game_type in game_scores:
                score = game_scores[game_type]
                weighted_sum += score * game_weight
                total_weight += game_weight
                per_game[game_type] = score

        cat_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        categories[cat_name] = CategoryScore(
            name=cat_name,
            score=cat_score,
            weight=cat_weight,
            game_scores=per_game,
        )

    return categories


def compute_composite_score(
    categories: dict[str, CategoryScore],
) -> float:
    """Compute weighted composite score from category scores.

    Args:
        categories: Per-category scores.

    Returns:
        Composite score 0-100.
    """
    total_weight = sum(c.weight for c in categories.values())
    if total_weight <= 0:
        return 0.0
    weighted_sum = sum(c.score * c.weight for c in categories.values())
    return weighted_sum / total_weight


def score_benchmark(
    game_results: dict[str, GameResult],
    agent_name: str,
    player_id: str | None = None,
    baselines: dict[str, dict[str, float]] | None = None,
    category_weights: dict[str, float] | None = None,
    category_game_map: dict[str, dict[str, float]] | None = None,
) -> AlympicsResult:
    """Score a complete Alympics benchmark run.

    Args:
        game_results: Mapping of game_type -> GameResult.
        agent_name: Name of the agent being evaluated.
        player_id: Which player_id in each GameResult belongs to the
            agent. If ``None``, uses the first player.
        baselines: Optional custom baseline bounds per game.
        category_weights: Optional custom category weights.
        category_game_map: Optional custom category-game mapping.

    Returns:
        ``AlympicsResult`` with composite + per-category scores.
    """
    # Compute normalised per-game scores
    game_scores: dict[str, float] = {}
    for game_type, result in game_results.items():
        avg = result.average_payoffs
        if not avg:
            game_scores[game_type] = 0.0
            continue

        if player_id and player_id in avg:
            raw = avg[player_id]
        else:
            # Use first player's payoff
            raw = next(iter(avg.values()))

        game_scores[game_type] = normalise_payoff(raw, game_type, baselines)

    # Compute category and composite
    categories = compute_category_scores(
        game_scores, category_weights, category_game_map
    )
    composite = compute_composite_score(categories)

    return AlympicsResult(
        agent_name=agent_name,
        composite_score=composite,
        categories=categories,
        game_results=game_results,
    )


# ---------------------------------------------------------------------------
# Suite YAML loading helpers
# ---------------------------------------------------------------------------

_SUITE_FILE = Path(__file__).parent / "builtin" / "alympics_lite.yaml"


def load_alympics_config() -> dict[str, Any]:
    """Load the builtin alympics_lite.yaml configuration.

    Returns:
        Parsed YAML data as a dict.
    """
    import yaml

    with open(_SUITE_FILE) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


async def run_alympics(
    agent_name: str = "builtin",
    episodes_override: int | None = None,
    verbose: bool = False,
) -> AlympicsResult:
    """Run the full Alympics benchmark battery.

    Loads the builtin ``alympics_lite.yaml``, runs all 5 games
    with their baseline strategies, and computes composite scores.

    Args:
        agent_name: Name label for the agent being evaluated.
        episodes_override: Override per-game episode count.
        verbose: Whether to log progress details.

    Returns:
        ``AlympicsResult`` with composite + per-category scores.
    """
    import logging

    from game_envs import GameRegistry, StrategyRegistry

    from atp_games.runner.builtin_adapter import BuiltinAdapter
    from atp_games.runner.game_runner import GameRunner

    logger = logging.getLogger(__name__)
    config_data = load_alympics_config()
    runner = GameRunner()

    game_results: dict[str, GameResult] = {}

    for game_spec in config_data.get("games", []):
        game_type = game_spec["type"]
        game_config = dict(game_spec.get("config", {}))

        if game_spec.get("variant") == "repeated":
            game_config.setdefault("num_rounds", 100)

        game = GameRegistry.create(game_type, game_config)

        # Build agents for this game
        agents: dict[str, Any] = {}
        agent_specs = game_spec.get("agents", [])
        player_ids = game.player_ids

        for i, agent_spec in enumerate(agent_specs):
            if i >= len(player_ids):
                break
            pid = player_ids[i]
            strategy = StrategyRegistry.create(agent_spec["strategy"])
            agents[pid] = BuiltinAdapter(strategy=strategy)

        episodes = episodes_override or game_spec.get("episodes", 20)
        run_config = GameRunConfig(episodes=episodes)

        if verbose:
            logger.info(
                "Running %s: %d episodes with %d agents",
                game_spec["name"],
                episodes,
                len(agents),
            )

        result = await runner.run_game(game, agents, run_config)
        game_results[game_type] = result

    # Use baseline bounds from config if present
    scoring_data = config_data.get("scoring", {})
    baselines = scoring_data.get("baselines")

    return score_benchmark(
        game_results=game_results,
        agent_name=agent_name,
        baselines=baselines,
    )
