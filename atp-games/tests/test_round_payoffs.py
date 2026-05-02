"""Phase 4 TDD tests: GameRunner populates EpisodeResult.round_payoffs.

Phase 4 adds a new ``round_payoffs: list[dict[str, float]]`` field to
``EpisodeResult``. The runner appends one dict per resolved day, where
each dict maps ``player_id -> per_day_payoff`` and ``per_day_payoff``
equals ``step_result.payoffs[player_id]`` from the underlying game step.

Invariant note
--------------
The plan's invariant ``sum(round_payoffs[*][pid]) == ep.payoffs[pid]``
holds for games whose ``get_payoffs()`` is a cumulative sum of
per-day payoffs (e.g. Prisoner's Dilemma, El Farol in the new
``happy_only`` mode where final = t_happy = sum of per-day happy
counts). It does NOT hold for El Farol in legacy
``happy_minus_crowded`` mode, where ``get_payoffs()`` uses a
non-linear ``t_happy / max(t_crowded, 0.1)`` ratio. One test below
asserts the legacy divergence explicitly under
``scoring_mode="happy_minus_crowded"``.
"""

from __future__ import annotations

from typing import Any

import pytest
from game_envs.core.state import Observation
from game_envs.core.strategy import Strategy
from game_envs.games.el_farol import ElFarolBar, ElFarolConfig
from game_envs.games.prisoners_dilemma import PDConfig, PrisonersDilemma
from game_envs.strategies.pd_strategies import AlwaysCooperate, AlwaysDefect

from atp_games.models import EpisodeResult, GameRunConfig
from atp_games.runner.builtin_adapter import BuiltinAdapter
from atp_games.runner.game_runner import GameRunner

# ---------------------------------------------------------------------------
# Deterministic El Farol test strategies (duplicated from
# test_runner_action_records.py to keep this file self-contained — no
# cross-file imports from other test modules).
# ---------------------------------------------------------------------------


class MorningAttendee(Strategy):
    """Always attends the first three slots of the day."""

    @property
    def name(self) -> str:
        return "morning_attendee"

    def choose_action(self, observation: Observation) -> Any:
        return [[0, 2]]


class EveningAttendee(Strategy):
    """Always attends the last three slots of the day."""

    @property
    def name(self) -> str:
        return "evening_attendee"

    def choose_action(self, observation: Observation) -> Any:
        return [[13, 15]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_el_farol_match(
    *,
    num_rounds: int = 3,
    capacity_threshold: int = 2,
    scoring_mode: str = "happy_only",
) -> EpisodeResult:
    """Run a 2-agent El Farol match and return the first episode result."""
    config = ElFarolConfig(
        num_players=2,
        num_rounds=num_rounds,
        capacity_threshold=capacity_threshold,
        scoring_mode=scoring_mode,  # type: ignore[arg-type]
    )
    game = ElFarolBar(config)
    agents = {
        "player_0": BuiltinAdapter(MorningAttendee()),
        "player_1": BuiltinAdapter(EveningAttendee()),
    }
    runner = GameRunner()
    result = await runner.run_game(game, agents, GameRunConfig(episodes=1))
    return result.episodes[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEpisodeResultRoundPayoffsField:
    """The new ``round_payoffs`` field on EpisodeResult."""

    def test_round_payoffs_field_exists_and_empty_by_default(self) -> None:
        # GIVEN an EpisodeResult constructed with only the required args
        ep = EpisodeResult(episode=0, payoffs={"p0": 1.0})

        # THEN the new round_payoffs field is present and defaults to []
        assert ep.round_payoffs == []


class TestRunnerPopulatesRoundPayoffsForPD:
    """PD: per-day payoffs sum to ep.payoffs (linear / cumulative game)."""

    @pytest.mark.anyio
    async def test_pd_3round_sum_matches_cumulative_payoffs(self) -> None:
        # GIVEN a 3-round PD with AllC vs AllC (mutual cooperation, R=3/round)
        game = PrisonersDilemma(PDConfig(num_rounds=3))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysCooperate()),
        }
        runner = GameRunner()
        result = await runner.run_game(game, agents, GameRunConfig(episodes=1))
        ep = result.episodes[0]

        # THEN one entry per round, each a {player_id: payoff} dict
        assert len(ep.round_payoffs) == 3
        for rp in ep.round_payoffs:
            assert set(rp.keys()) == {"player_0", "player_1"}

        # AND PD is linear, so per-round payoffs sum to the cumulative total
        assert sum(rp["player_0"] for rp in ep.round_payoffs) == pytest.approx(
            ep.payoffs["player_0"]
        )
        assert sum(rp["player_1"] for rp in ep.round_payoffs) == pytest.approx(
            ep.payoffs["player_1"]
        )

    @pytest.mark.anyio
    async def test_pd_first_round_matches_step_payoff(self) -> None:
        # GIVEN a 1-round PD with AllC vs AllD
        game = PrisonersDilemma(PDConfig(num_rounds=1))
        agents = {
            "player_0": BuiltinAdapter(AlwaysCooperate()),
            "player_1": BuiltinAdapter(AlwaysDefect()),
        }
        runner = GameRunner()
        result = await runner.run_game(game, agents, GameRunConfig(episodes=1))
        ep = result.episodes[0]

        # THEN a single per-round entry reflects the step payoffs:
        # cooperator gets sucker (0), defector gets temptation (5)
        assert len(ep.round_payoffs) == 1
        assert ep.round_payoffs[0]["player_0"] == pytest.approx(0.0)
        assert ep.round_payoffs[0]["player_1"] == pytest.approx(5.0)


class TestRunnerPopulatesRoundPayoffsForElFarol:
    """El Farol: per-day payoffs are always captured. Cumulative
    ``ep.payoffs`` equals the sum of per-day payoffs in the new
    ``happy_only`` default; in legacy ``happy_minus_crowded`` mode the
    cumulative formula is non-linear and intentionally diverges from
    the sum."""

    @pytest.mark.anyio
    async def test_el_farol_round_payoffs_length_equals_num_days(self) -> None:
        # GIVEN a 3-day, 2-agent El Farol match
        ep = await _run_el_farol_match(num_rounds=3)

        # THEN round_payoffs has one entry per resolved day
        assert len(ep.round_payoffs) == 3

    @pytest.mark.anyio
    @pytest.mark.parametrize("scoring_mode", ["happy_only", "happy_minus_crowded"])
    async def test_el_farol_per_day_payoff_equals_happy_count_no_crowding(
        self, scoring_mode: str
    ) -> None:
        # GIVEN a 3-day, 2-agent El Farol match with disjoint contiguous picks
        # (morning [0,1,2] vs evening [13,14,15]) and capacity_threshold=2.
        # Each player fills 3 slots alone → happy=3, crowded=0 per day →
        # per-day step payoff is 3.0 under happy_only (the new default) and
        # also 3.0 under happy_minus_crowded since crowded=0 (happy − 0 =
        # happy). Mode-agnostic by design — runs against both modes to
        # lock in the equivalence.
        ep = await _run_el_farol_match(num_rounds=3, scoring_mode=scoring_mode)

        # THEN every day records a per-day payoff of 3.0 for both players
        for day_index in range(3):
            for pid in ("player_0", "player_1"):
                assert ep.round_payoffs[day_index][pid] == pytest.approx(3.0), (
                    f"per-day payoff for {pid} on day {day_index} should be "
                    f"3.0 under scoring_mode={scoring_mode!r}, got "
                    f"{ep.round_payoffs[day_index][pid]}"
                )

    @pytest.mark.anyio
    async def test_el_farol_legacy_mode_round_payoffs_sum_diverges_from_cumulative(
        self,
    ) -> None:
        # GIVEN the same 3-day El Farol match in LEGACY happy_minus_crowded
        # mode (morning vs evening, threshold=2, num_players=2, num_rounds=3).
        # Under happy_only this divergence does not exist: final = t_happy =
        # sum of per-day happy counts.
        ep = await _run_el_farol_match(num_rounds=3, scoring_mode="happy_minus_crowded")

        # THEN per-day payoffs sum to 9.0 (3 happy slots × 3 days) for each
        # player, but ep.payoffs uses the non-linear
        #    t_happy / max(t_crowded, 0.1)
        # formula — with t_crowded=0 across 3 days, t_happy=9, that yields
        # 9 / 0.1 = 90.0.  This divergence is intentional: El Farol's legacy
        # cumulative payoff is NOT a simple sum of per-day payoffs, so the
        # plan's invariant (sum == ep.payoffs) only applies to linear games
        # such as Prisoner's Dilemma (and to El Farol in happy_only mode).
        for pid in ("player_0", "player_1"):
            daily_sum = sum(rp[pid] for rp in ep.round_payoffs)
            assert daily_sum == pytest.approx(9.0), (
                f"expected per-day payoff sum for {pid} to equal 9.0 "
                f"(3 happy slots × 3 days), got {daily_sum}"
            )
            assert ep.payoffs[pid] == pytest.approx(90.0), (
                f"expected cumulative ep.payoffs[{pid}] to equal 90.0 "
                f"(t_happy=9 / max(t_crowded=0, 0.1) = 9 / 0.1 = 90.0 "
                f"across 3 days), got {ep.payoffs[pid]}"
            )
