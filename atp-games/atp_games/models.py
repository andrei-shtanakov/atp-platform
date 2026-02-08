"""Data models for game runner results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GameRunConfig:
    """Configuration for a game run.

    Attributes:
        episodes: Number of episodes to run.
        max_retries: Max retry attempts for invalid actions.
        move_timeout: Per-move timeout in seconds.
        parallel: Number of episodes to run in parallel.
            1 = sequential, >1 = concurrent execution.
        base_seed: Base seed for deterministic episode seeding.
            Each episode gets seed = base_seed + episode_index.
            None = non-deterministic.
    """

    episodes: int = 1
    max_retries: int = 3
    move_timeout: float = 30.0
    parallel: int = 1
    base_seed: int | None = None

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError(f"episodes must be >= 1, got {self.episodes}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.move_timeout <= 0:
            raise ValueError(f"move_timeout must be > 0, got {self.move_timeout}")
        if self.parallel < 1:
            raise ValueError(f"parallel must be >= 1, got {self.parallel}")

    def episode_seed(self, episode_index: int) -> int | None:
        """Compute deterministic seed for a given episode.

        Returns:
            base_seed + episode_index if base_seed is set,
            None otherwise.
        """
        if self.base_seed is None:
            return None
        return self.base_seed + episode_index


@dataclass
class EpisodeResult:
    """Result of a single game episode.

    Attributes:
        episode: Episode index.
        payoffs: Final cumulative payoffs per player.
        history: List of step result dicts from each round.
        actions_log: Per-round actions taken by each player.
        seed: Random seed used for this episode (if any).
    """

    episode: int
    payoffs: dict[str, float]
    history: list[dict[str, Any]] = field(default_factory=list)
    actions_log: list[dict[str, Any]] = field(default_factory=list)
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "episode": self.episode,
            "payoffs": dict(self.payoffs),
            "history": list(self.history),
            "actions_log": list(self.actions_log),
        }
        if self.seed is not None:
            result["seed"] = self.seed
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeResult:
        """Deserialize from dictionary."""
        return cls(
            episode=data["episode"],
            payoffs=data["payoffs"],
            history=data.get("history", []),
            actions_log=data.get("actions_log", []),
            seed=data.get("seed"),
        )


@dataclass
class PlayerStats:
    """Aggregated statistics for a single player.

    Attributes:
        player_id: Player identifier.
        mean: Mean payoff across episodes.
        std: Standard deviation of payoffs.
        ci_lower: Lower bound of 95% confidence interval.
        ci_upper: Upper bound of 95% confidence interval.
        min: Minimum payoff observed.
        max: Maximum payoff observed.
        n_episodes: Number of episodes.
    """

    player_id: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    min: float
    max: float
    n_episodes: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "player_id": self.player_id,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci_95": (round(self.ci_lower, 4), round(self.ci_upper, 4)),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "n_episodes": self.n_episodes,
        }


@dataclass
class AgentComparison:
    """Result of comparing two agents via Welch's t-test.

    Attributes:
        player_a: First player identifier.
        player_b: Second player identifier.
        metric: Name of the metric compared.
        mean_a: Mean value for player A.
        mean_b: Mean value for player B.
        t_statistic: Welch's t-test statistic.
        p_value: Raw p-value from test.
        adjusted_p_value: Bonferroni-corrected p-value.
        is_significant: Whether difference is significant
            after correction.
    """

    player_a: str
    player_b: str
    metric: str
    mean_a: float
    mean_b: float
    t_statistic: float
    p_value: float
    adjusted_p_value: float
    is_significant: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "player_a": self.player_a,
            "player_b": self.player_b,
            "metric": self.metric,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 6),
            "adjusted_p_value": round(self.adjusted_p_value, 6),
            "is_significant": self.is_significant,
        }


def _compute_player_stats(
    player_id: str,
    payoffs: list[float],
) -> PlayerStats:
    """Compute aggregated statistics for one player.

    Args:
        player_id: The player identifier.
        payoffs: List of payoffs across episodes.

    Returns:
        PlayerStats with mean, std, CI, min, max.
    """
    n = len(payoffs)
    mean = sum(payoffs) / n
    if n < 2:
        return PlayerStats(
            player_id=player_id,
            mean=mean,
            std=0.0,
            ci_lower=mean,
            ci_upper=mean,
            min=mean,
            max=mean,
            n_episodes=n,
        )
    variance = sum((x - mean) ** 2 for x in payoffs) / (n - 1)
    std = math.sqrt(variance)

    # 95% CI using t-distribution
    # For large n, use z=1.96; for small n, approximate with
    # lookup table.
    t_crit = _t_critical_95(n - 1)
    margin = t_crit * std / math.sqrt(n)

    return PlayerStats(
        player_id=player_id,
        mean=mean,
        std=std,
        ci_lower=mean - margin,
        ci_upper=mean + margin,
        min=min(payoffs),
        max=max(payoffs),
        n_episodes=n,
    )


# t-critical values for 95% two-tailed CI (df → t)
_T_CRIT_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def _t_critical_95(df: int) -> float:
    """Get t-critical value for 95% CI given degrees of freedom."""
    if df <= 0:
        return float("inf")
    if df > 30:
        return 1.96
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    # Interpolate between known values
    keys = sorted(_T_CRIT_95.keys())
    for i in range(len(keys) - 1):
        if keys[i] < df < keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            frac = (df - lo) / (hi - lo)
            return _T_CRIT_95[lo] + frac * (_T_CRIT_95[hi] - _T_CRIT_95[lo])
    return 1.96


def welchs_t_test(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
) -> tuple[float, float]:
    """Perform Welch's t-test for two independent samples.

    Args:
        mean1: Mean of sample 1.
        std1: Standard deviation of sample 1.
        n1: Size of sample 1.
        mean2: Mean of sample 2.
        std2: Standard deviation of sample 2.
        n2: Size of sample 2.

    Returns:
        Tuple of (t_statistic, p_value).
        Returns (0.0, 1.0) if calculation is not possible.
    """
    if n1 < 2 or n2 < 2:
        return (0.0, 1.0)

    se1 = std1**2 / n1
    se2 = std2**2 / n2
    se_sum = se1 + se2

    if se_sum < 1e-10:
        if abs(mean1 - mean2) < 1e-10:
            return (0.0, 1.0)
        return (float("inf"), 0.0)

    t_stat = (mean1 - mean2) / math.sqrt(se_sum)

    # Welch-Satterthwaite degrees of freedom
    numerator = se_sum**2
    denom = (se1**2 / (n1 - 1)) + (se2**2 / (n2 - 1))
    if denom < 1e-10:
        df = min(n1, n2) - 1
    else:
        df = numerator / denom

    p_value = _approx_p_value(abs(t_stat), df)
    return (t_stat, p_value)


def _approx_p_value(t_stat: float, df: float) -> float:
    """Approximate two-tailed p-value from t-statistic and df.

    Uses critical-value-based interpolation without scipy.
    """
    if t_stat <= 0:
        return 1.0
    if df < 1:
        return 1.0

    t_crit = _t_critical_95(int(df))

    if t_stat < t_crit * 0.5:
        return min(1.0, 1.0 - (t_stat / t_crit) * 0.5)
    elif t_stat < t_crit:
        ratio = t_stat / t_crit
        return max(0.05, 0.5 * (1.0 - ratio) + 0.05)
    elif t_stat < t_crit * 1.5:
        ratio = (t_stat - t_crit) / (t_crit * 0.5)
        return max(0.01, 0.05 * (1.0 - ratio))
    elif t_stat < t_crit * 2.5:
        ratio = (t_stat - t_crit * 1.5) / t_crit
        return max(0.001, 0.01 * (1.0 - ratio * 0.9))
    else:
        return 0.001


def compare_agents(
    result: GameResult,
    metric: str = "payoff",
    significance_level: float = 0.05,
) -> list[AgentComparison]:
    """Compare all pairs of agents using Welch's t-test.

    Applies Bonferroni correction for multiple comparisons.

    Args:
        result: GameResult with episode data.
        metric: Metric to compare (currently "payoff").
        significance_level: Base significance level.

    Returns:
        List of pairwise AgentComparison results.
    """
    if not result.episodes:
        return []

    player_ids = sorted(result.episodes[0].payoffs.keys())
    if len(player_ids) < 2:
        return []

    # Collect per-player payoff sequences
    payoff_sequences: dict[str, list[float]] = {pid: [] for pid in player_ids}
    for ep in result.episodes:
        for pid in player_ids:
            payoff_sequences[pid].append(ep.payoffs.get(pid, 0.0))

    # All pairwise comparisons
    pairs: list[tuple[str, str]] = []
    for i in range(len(player_ids)):
        for j in range(i + 1, len(player_ids)):
            pairs.append((player_ids[i], player_ids[j]))

    n_comparisons = len(pairs)
    corrected_level = significance_level / n_comparisons

    comparisons: list[AgentComparison] = []
    for pa, pb in pairs:
        vals_a = payoff_sequences[pa]
        vals_b = payoff_sequences[pb]
        n_a = len(vals_a)
        n_b = len(vals_b)
        mean_a = sum(vals_a) / n_a
        mean_b = sum(vals_b) / n_b
        std_a = (
            math.sqrt(sum((x - mean_a) ** 2 for x in vals_a) / (n_a - 1))
            if n_a > 1
            else 0.0
        )
        std_b = (
            math.sqrt(sum((x - mean_b) ** 2 for x in vals_b) / (n_b - 1))
            if n_b > 1
            else 0.0
        )

        t_stat, p_val = welchs_t_test(
            mean_a,
            std_a,
            n_a,
            mean_b,
            std_b,
            n_b,
        )
        adjusted_p = min(p_val * n_comparisons, 1.0)
        is_sig = p_val < corrected_level

        comparisons.append(
            AgentComparison(
                player_a=pa,
                player_b=pb,
                metric=metric,
                mean_a=mean_a,
                mean_b=mean_b,
                t_statistic=t_stat,
                p_value=p_val,
                adjusted_p_value=adjusted_p,
                is_significant=is_sig,
            )
        )

    return comparisons


@dataclass
class GameResult:
    """Aggregated result of a complete game run.

    Attributes:
        game_name: Name of the game played.
        config: Run configuration used.
        episodes: Per-episode results.
        agent_names: Mapping of player_id to agent name.
    """

    game_name: str
    config: GameRunConfig
    episodes: list[EpisodeResult] = field(default_factory=list)
    agent_names: dict[str, str] = field(default_factory=dict)

    @property
    def num_episodes(self) -> int:
        """Number of completed episodes."""
        return len(self.episodes)

    @property
    def average_payoffs(self) -> dict[str, float]:
        """Average payoff per player across episodes."""
        if not self.episodes:
            return {}
        totals: dict[str, float] = {}
        for ep in self.episodes:
            for pid, payoff in ep.payoffs.items():
                totals[pid] = totals.get(pid, 0.0) + payoff
        return {pid: total / len(self.episodes) for pid, total in totals.items()}

    def player_payoffs(self, player_id: str) -> list[float]:
        """Get payoff sequence for a player across episodes."""
        return [ep.payoffs.get(player_id, 0.0) for ep in self.episodes]

    def player_statistics(self) -> dict[str, PlayerStats]:
        """Compute aggregated statistics per player.

        Returns mean, std, 95% CI, min, max for each player.
        """
        if not self.episodes:
            return {}
        player_ids = sorted(self.episodes[0].payoffs.keys())
        return {
            pid: _compute_player_stats(pid, self.player_payoffs(pid))
            for pid in player_ids
        }

    def agent_comparisons(
        self,
        metric: str = "payoff",
        significance_level: float = 0.05,
    ) -> list[AgentComparison]:
        """Compare all agent pairs with Bonferroni correction.

        Args:
            metric: Metric to compare.
            significance_level: Base significance level.

        Returns:
            Pairwise comparison results.
        """
        return compare_agents(self, metric, significance_level)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "game_name": self.game_name,
            "config": {
                "episodes": self.config.episodes,
                "max_retries": self.config.max_retries,
                "move_timeout": self.config.move_timeout,
                "parallel": self.config.parallel,
            },
            "episodes": [e.to_dict() for e in self.episodes],
            "agent_names": dict(self.agent_names),
            "average_payoffs": self.average_payoffs,
        }
        if self.config.base_seed is not None:
            result["config"]["base_seed"] = self.config.base_seed
        if self.episodes:
            result["player_statistics"] = {
                pid: stats.to_dict() for pid, stats in self.player_statistics().items()
            }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameResult:
        """Deserialize from dictionary."""
        config_data = data.get("config", {})
        return cls(
            game_name=data["game_name"],
            config=GameRunConfig(
                episodes=config_data.get("episodes", 1),
                max_retries=config_data.get("max_retries", 3),
                move_timeout=config_data.get("move_timeout", 30.0),
                parallel=config_data.get("parallel", 1),
                base_seed=config_data.get("base_seed"),
            ),
            episodes=[EpisodeResult.from_dict(e) for e in data.get("episodes", [])],
            agent_names=data.get("agent_names", {}),
        )
