"""Tests for Colonel Blotto strategies."""

from __future__ import annotations

import pytest

from game_envs.core.state import Observation
from game_envs.games.colonel_blotto import BlottoConfig, ColonelBlotto
from game_envs.strategies.blotto_strategies import (
    ConcentratedAllocation,
    NashMixed,
    UniformAllocation,
)

FIELDS = ["field_0", "field_1", "field_2", "field_3", "field_4"]
TOTAL = 100.0


def _make_obs(
    fields: list[str] | None = None,
    total: float = TOTAL,
) -> Observation:
    """Create a minimal Blotto observation."""
    if fields is None:
        fields = FIELDS
    return Observation(
        player_id="player_0",
        game_state={
            "game": "Colonel Blotto",
            "your_role": "player_0",
            "fields": fields,
            "total_units": total,
        },
        available_actions=[
            f"fields: {fields}",
            f"must sum to {total}",
        ],
        history=[],
        round_number=0,
        total_rounds=1,
    )


class TestUniformAllocation:
    def test_equal_distribution(self) -> None:
        s = UniformAllocation()
        action = s.choose_action(_make_obs())
        assert isinstance(action, dict)
        assert len(action) == 5
        for f in FIELDS:
            assert action[f] == pytest.approx(20.0)

    def test_sums_to_total(self) -> None:
        s = UniformAllocation()
        action = s.choose_action(_make_obs(total=60.0))
        assert sum(action.values()) == pytest.approx(60.0)

    def test_empty_fields(self) -> None:
        s = UniformAllocation()
        action = s.choose_action(_make_obs(fields=[]))
        assert action == {}

    def test_name(self) -> None:
        assert UniformAllocation().name == "uniform_allocation"


class TestConcentratedAllocation:
    def test_concentrates_on_majority(self) -> None:
        s = ConcentratedAllocation()
        action = s.choose_action(_make_obs())
        # 5 fields -> concentrate on 3 (5//2 + 1)
        nonzero = [f for f, v in action.items() if v > 0]
        zero = [f for f, v in action.items() if v == 0]
        assert len(nonzero) == 3
        assert len(zero) == 2

    def test_sums_to_total(self) -> None:
        s = ConcentratedAllocation()
        action = s.choose_action(_make_obs())
        assert sum(action.values()) == pytest.approx(TOTAL)

    def test_four_fields(self) -> None:
        s = ConcentratedAllocation()
        fields = ["a", "b", "c", "d"]
        action = s.choose_action(_make_obs(fields=fields))
        # 4 fields -> concentrate on 3 (4//2 + 1)
        nonzero = [f for f, v in action.items() if v > 0]
        assert len(nonzero) == 3
        assert sum(action.values()) == pytest.approx(TOTAL)

    def test_empty_fields(self) -> None:
        s = ConcentratedAllocation()
        action = s.choose_action(_make_obs(fields=[]))
        assert action == {}

    def test_name(self) -> None:
        assert ConcentratedAllocation().name == "concentrated_allocation"


class TestNashMixed:
    def test_sums_to_total(self) -> None:
        s = NashMixed(seed=42)
        action = s.choose_action(_make_obs())
        assert sum(action.values()) == pytest.approx(TOTAL)

    def test_all_positive(self) -> None:
        s = NashMixed(seed=42)
        action = s.choose_action(_make_obs())
        for v in action.values():
            assert v > 0

    def test_randomized(self) -> None:
        """Different calls produce different allocations."""
        s = NashMixed(seed=42)
        obs = _make_obs()
        a1 = s.choose_action(obs)
        a2 = s.choose_action(obs)
        # Very unlikely to be identical
        assert a1 != a2

    def test_deterministic_with_seed(self) -> None:
        s1 = NashMixed(seed=123)
        s2 = NashMixed(seed=123)
        obs = _make_obs()
        a1 = s1.choose_action(obs)
        a2 = s2.choose_action(obs)
        for f in FIELDS:
            assert a1[f] == pytest.approx(a2[f])

    def test_empty_fields(self) -> None:
        s = NashMixed(seed=42)
        action = s.choose_action(_make_obs(fields=[]))
        assert action == {}

    def test_name(self) -> None:
        assert NashMixed().name == "nash_mixed"


class TestBlottoIntegration:
    """Integration tests with actual ColonelBlotto game."""

    def test_uniform_vs_concentrated(self) -> None:
        """Uniform vs Concentrated: verify valid game play."""
        game = ColonelBlotto(BlottoConfig(num_rounds=1, seed=42))
        game.reset()

        uniform = UniformAllocation()
        conc = ConcentratedAllocation()

        obs_0 = game.observe("player_0")
        obs_1 = game.observe("player_1")

        a0 = uniform.choose_action(obs_0)
        a1 = conc.choose_action(obs_1)

        result = game.step({"player_0": a0, "player_1": a1})
        # Both payoffs should be valid fractions
        assert 0.0 <= result.payoffs["player_0"] <= 1.0
        assert 0.0 <= result.payoffs["player_1"] <= 1.0
        # Payoffs sum to 1.0 (zero-sum in terms of wins)
        total = result.payoffs["player_0"] + result.payoffs["player_1"]
        assert total == pytest.approx(1.0)

    def test_nash_mixed_produces_valid_actions(self) -> None:
        """NashMixed produces valid allocations for the game."""
        game = ColonelBlotto(BlottoConfig(num_battlefields=5, num_rounds=3, seed=42))
        game.reset()
        nash0 = NashMixed(seed=10)
        nash1 = NashMixed(seed=20)

        while not game.is_terminal:
            obs_0 = game.observe("player_0")
            obs_1 = game.observe("player_1")
            a0 = nash0.choose_action(obs_0)
            a1 = nash1.choose_action(obs_1)
            # Verify allocations are valid
            assert game.action_space("player_0").contains(a0)
            assert game.action_space("player_1").contains(a1)
            game.step({"player_0": a0, "player_1": a1})

        payoffs = game.get_payoffs()
        assert all(p >= 0.0 for p in payoffs.values())
