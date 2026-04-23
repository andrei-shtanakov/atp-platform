"""Tests for ``atp.cli.commands.game._build_game_result_kwargs``.

The writer helper builds the kwargs dict that is splatted into the ORM
``GameResult(...)`` constructor by ``_store_game_result``. These tests
drive the implementation via TDD: they exercise the pure, synchronous
helper in isolation (no DB, no runner, no I/O).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from atp_games.models import (
    ActionRecord,
    AgentRecord,
    EpisodeResult,
    GameRunConfig,
    IntervalPair,
)
from atp_games.models import (
    GameResult as RunGameResult,
)
from atp_games.suites.models import (
    GameAgentConfig,
    GameSuiteConfig,
)
from atp_games.suites.models import (
    GameConfig as SuiteGameConfig,
)
from game_envs.games.el_farol import ElFarolBar, ElFarolConfig

# The helper does not yet exist; the import is the first failing signal.
from atp.cli.commands.game import _build_game_result_kwargs  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interval_pair(
    first: tuple[int, int] | tuple[()] = (0, 0),
    second: tuple[int, int] | tuple[()] = (),
    *,
    num_slots: int = 16,
    max_total_slots: int = 8,
) -> IntervalPair:
    """Construct an IntervalPair with safe defaults."""
    return IntervalPair(
        first=first,
        second=second,
        num_slots=num_slots,
        max_total_slots=max_total_slots,
    )


def _make_action(
    *,
    match_id: str = "m-0",
    day: int = 1,
    agent_id: str = "p0",
    intervals: IntervalPair | None = None,
    picks: tuple[int, ...] = (0,),
    num_visits: int = 1,
    total_slots: int = 1,
    payoff: float = 1.0,
    num_under: int = 0,
    num_over: int = 0,
    submitted_at: datetime | None = None,
) -> ActionRecord:
    """Build an ActionRecord with minimal valid fields."""
    return ActionRecord(
        match_id=match_id,
        day=day,
        agent_id=agent_id,
        intervals=intervals if intervals is not None else _make_interval_pair(),
        picks=picks,
        num_visits=num_visits,
        total_slots=total_slots,
        payoff=payoff,
        num_under=num_under,
        num_over=num_over,
        submitted_at=submitted_at,
    )


def _make_episode(
    *,
    episode: int = 0,
    payoffs: dict[str, float] | None = None,
    actions: list[ActionRecord] | None = None,
    round_payoffs: list[dict[str, float]] | None = None,
    history: list[dict[str, Any]] | None = None,
    seed: int | None = None,
) -> EpisodeResult:
    """Build an EpisodeResult with sensible defaults."""
    return EpisodeResult(
        episode=episode,
        payoffs=payoffs if payoffs is not None else {"p0": 1.0},
        history=history if history is not None else [],
        actions_log=[],
        actions=actions if actions is not None else [],
        round_payoffs=round_payoffs if round_payoffs is not None else [],
        seed=seed,
    )


def _make_run_result(
    *,
    run_id: str = "run-xyz",
    game_name: str = "El Farol Bar (n=10, threshold=6, days=30)",
    agents: list[AgentRecord] | None = None,
    episodes: list[EpisodeResult] | None = None,
    agent_names: dict[str, str] | None = None,
) -> RunGameResult:
    """Build a RunGameResult with defaults: 1 episode, no agent roster."""
    return RunGameResult(
        game_name=game_name,
        config=GameRunConfig(episodes=1),
        episodes=episodes if episodes is not None else [_make_episode()],
        agent_names=agent_names if agent_names is not None else {"p0": "player0"},
        agents=agents if agents is not None else [],
        run_id=run_id,
    )


def _make_suite(version: str = "1.0") -> GameSuiteConfig:
    """Build a minimal GameSuiteConfig."""
    return GameSuiteConfig(
        name="test",
        version=version,
        game=SuiteGameConfig(type="el_farol", variant="repeated", config={}),
        agents=[
            GameAgentConfig(name="a0", adapter="builtin"),
            GameAgentConfig(name="a1", adapter="builtin"),
        ],
    )


def _make_el_farol_game(
    *,
    num_players: int = 10,
    num_rounds: int = 30,
    num_slots: int = 16,
    max_intervals: int = 2,
    max_total_slots: int = 8,
    capacity_ratio: float = 0.6,
) -> ElFarolBar:
    """Build an ElFarolBar instance with explicit knobs."""
    cfg = ElFarolConfig(
        num_players=num_players,
        num_rounds=num_rounds,
        num_slots=num_slots,
        max_intervals=max_intervals,
        max_total_slots=max_total_slots,
        capacity_ratio=capacity_ratio,
    )
    return ElFarolBar(cfg)


class _StubConfig:
    """Stub game config exposing only ``num_rounds``."""

    def __init__(self, num_rounds: int = 5) -> None:
        self.num_rounds = num_rounds


class _StubGame:
    """Stub game object without El Farol-specific config attrs."""

    def __init__(self, num_rounds: int = 5) -> None:
        self.config = _StubConfig(num_rounds=num_rounds)
        self.name = "stub-game"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_populates_existing_legacy_fields() -> None:
    # GIVEN a simple run result with 1 player, 1 episode, and an El Farol game
    result = _make_run_result(
        run_id="r1",
        episodes=[_make_episode(payoffs={"p0": 2.5})],
        agent_names={"p0": "tft"},
    )
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN the helper builds the kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN legacy fields are populated as before
    assert kwargs["game_name"] == result.game_name
    assert kwargs["game_type"] == "repeated"
    assert kwargs["num_players"] == 1
    assert kwargs["num_rounds"] == 30
    assert kwargs["num_episodes"] == 1
    assert kwargs["status"] == "completed"
    assert isinstance(kwargs["completed_at"], datetime)
    assert kwargs["players_json"] == [
        {
            "player_id": "p0",
            "name": "tft",
            "strategy": "tft",
            "average_payoff": 2.5,
        }
    ]
    assert kwargs["episodes_json"] == [
        {"episode": 0, "payoffs": {"p0": 2.5}, "seed": None}
    ]
    assert kwargs["metadata_json"]["suite_name"] == "test"


def test_match_id_uses_run_id() -> None:
    # GIVEN a run result with a specific run_id
    result = _make_run_result(run_id="abc-123-run")
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN match_id echoes the run_id
    assert kwargs["match_id"] == "abc-123-run"


def test_game_version_falls_back_to_suite_version() -> None:
    # GIVEN a suite with a custom version
    result = _make_run_result()
    suite = _make_suite(version="2.1")
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN game_version is read from the suite
    assert kwargs["game_version"] == "2.1"


def test_el_farol_config_columns_populated_from_game_config() -> None:
    # GIVEN an El Farol game with explicit config knobs
    result = _make_run_result()
    suite = _make_suite()
    game = _make_el_farol_game(
        num_players=10,
        num_rounds=30,
        num_slots=16,
        max_intervals=2,
        max_total_slots=8,
        capacity_ratio=0.6,
    )

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN the El Farol config columns come from the game config
    assert kwargs["num_days"] == 30
    assert kwargs["num_slots"] == 16
    assert kwargs["max_intervals"] == 2
    assert kwargs["max_total_slots"] == 8
    assert kwargs["capacity_ratio"] == 0.6
    # floor(0.6 * 10) == 6
    assert kwargs["capacity_threshold"] == 6


def test_non_el_farol_game_leaves_config_columns_absent_or_none() -> None:
    # GIVEN a game whose config has no El Farol attributes
    result = _make_run_result()
    suite = _make_suite()
    game = _StubGame(num_rounds=5)

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN num_days is present (from num_rounds) but El Farol-only keys are absent
    assert kwargs["num_days"] == 5
    assert "num_slots" not in kwargs
    assert "max_intervals" not in kwargs
    assert "max_total_slots" not in kwargs
    assert "capacity_ratio" not in kwargs
    assert "capacity_threshold" not in kwargs


def test_agents_json_serialised_from_agent_records() -> None:
    # GIVEN a run result with a typed agent roster
    agents = [
        AgentRecord(agent_id="p0", display_name="tft", user_id="u1"),
        AgentRecord(agent_id="p1", display_name="defector", user_id="u1"),
    ]
    result = _make_run_result(agents=agents)
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN agents_json contains serialised dicts with the expected keys
    serialised = kwargs["agents_json"]
    assert isinstance(serialised, list)
    assert len(serialised) == 2

    expected_keys = {
        "agent_id",
        "display_name",
        "user_id",
        "user_display",
        "family",
        "adapter_type",
        "model_id",
        "color",
    }
    for entry in serialised:
        assert expected_keys <= set(entry.keys())

    assert serialised[0]["agent_id"] == "p0"
    assert serialised[0]["display_name"] == "tft"
    assert serialised[1]["agent_id"] == "p1"
    assert serialised[1]["display_name"] == "defector"


def test_agents_json_absent_when_no_agent_records() -> None:
    # GIVEN a run result with no agent roster
    result = _make_run_result(agents=[])
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN agents_json key is omitted (matching other absence conventions)
    assert "agents_json" not in kwargs


def test_actions_json_flattens_records_across_episodes() -> None:
    # GIVEN two episodes, each with two ActionRecords
    ep0_actions = [
        _make_action(match_id="m#ep0", day=1, agent_id="p0"),
        _make_action(match_id="m#ep0", day=1, agent_id="p1"),
    ]
    ep1_actions = [
        _make_action(match_id="m#ep1", day=1, agent_id="p0"),
        _make_action(match_id="m#ep1", day=1, agent_id="p1"),
    ]
    episodes = [
        _make_episode(episode=0, actions=ep0_actions),
        _make_episode(episode=1, actions=ep1_actions),
    ]
    result = _make_run_result(episodes=episodes)
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN actions_json contains 4 dicts with the expected keys
    actions_json = kwargs["actions_json"]
    assert isinstance(actions_json, list)
    assert len(actions_json) == 4

    expected_keys = {
        "match_id",
        "day",
        "agent_id",
        "intervals",
        "picks",
        "num_visits",
        "total_slots",
        "payoff",
    }
    for entry in actions_json:
        assert expected_keys <= set(entry.keys())
        # intervals is the asdict'd IntervalPair: a dict, not an IntervalPair
        assert isinstance(entry["intervals"], dict)
        assert "first" in entry["intervals"]
        assert "second" in entry["intervals"]


def test_actions_json_serializes_submitted_at_as_iso_string() -> None:
    # GIVEN one action with a concrete datetime and one with None
    submitted = datetime(2025, 6, 1, 12, 30, 45)
    actions = [
        _make_action(agent_id="p0", submitted_at=submitted),
        _make_action(agent_id="p1", submitted_at=None),
    ]
    result = _make_run_result(
        episodes=[_make_episode(actions=actions)],
    )
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN datetime is serialised as ISO 8601 and None stays None
    actions_json = kwargs["actions_json"]
    by_agent = {a["agent_id"]: a for a in actions_json}
    assert by_agent["p0"]["submitted_at"] == "2025-06-01T12:30:45"
    assert by_agent["p1"]["submitted_at"] is None


def test_actions_json_absent_when_no_episode_has_actions() -> None:
    # GIVEN two episodes with empty actions lists
    episodes = [
        _make_episode(episode=0, actions=[]),
        _make_episode(episode=1, actions=[]),
    ]
    result = _make_run_result(episodes=episodes)
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN actions_json key is omitted
    assert "actions_json" not in kwargs


def test_round_payoffs_json_uses_first_episode() -> None:
    # GIVEN episode 0 with two rounds of payoffs, episode 1 differs
    ep0_payoffs = [{"p0": 1.0, "p1": 2.0}, {"p0": 0.5, "p1": 1.5}]
    ep1_payoffs = [{"p0": 9.0, "p1": 9.0}]
    episodes = [
        _make_episode(episode=0, round_payoffs=ep0_payoffs),
        _make_episode(episode=1, round_payoffs=ep1_payoffs),
    ]
    result = _make_run_result(episodes=episodes)
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN round_payoffs_json echoes episode 0's list only
    assert kwargs["round_payoffs_json"] == ep0_payoffs
    assert len(kwargs["round_payoffs_json"]) == 2


def test_round_payoffs_json_absent_when_episode_0_has_none() -> None:
    # GIVEN episode 0 has an empty round_payoffs list
    episodes = [_make_episode(episode=0, round_payoffs=[])]
    result = _make_run_result(episodes=episodes)
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN the key is omitted entirely
    assert "round_payoffs_json" not in kwargs


def test_day_aggregates_derived_from_episode_0_history() -> None:
    # GIVEN a game with capacity_threshold=2 (num_players=4, ratio=0.5)
    game = _make_el_farol_game(
        num_players=4,
        num_rounds=2,
        num_slots=4,
        max_intervals=2,
        max_total_slots=2,
        capacity_ratio=0.5,
    )
    assert game.config.capacity_threshold == 2

    # AND an episode 0 history with two day-dicts (synthetic step dicts)
    history = [
        {
            "state": {
                "public_state": {
                    "attendance_history": [[1, 0, 2, 1]],
                    "crowded_slots_today": [2],
                }
            },
            "payoffs": {},
            "is_terminal": False,
            "observations": {},
            "info": {},
        },
        {
            "state": {
                "public_state": {
                    "attendance_history": [[1, 0, 2, 1], [0, 3, 1, 0]],
                    "crowded_slots_today": [1],
                }
            },
            "payoffs": {},
            "is_terminal": True,
            "observations": {},
            "info": {},
        },
    ]
    # Populate actions so the El Farol gate opens.
    actions = [
        _make_action(
            match_id="m",
            day=1,
            agent_id="p0",
            intervals=_make_interval_pair(first=(0, 0), num_slots=4, max_total_slots=2),
        )
    ]
    run_id = "rZ"
    episodes = [_make_episode(episode=0, actions=actions, history=history)]
    result = _make_run_result(run_id=run_id, episodes=episodes)
    suite = _make_suite()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN day aggregates are derived from the per-day attendance_history
    expected_match_id = f"{game.name}#{run_id}#ep0"
    assert kwargs["day_aggregates_json"] == [
        {
            "match_id": expected_match_id,
            "day": 1,
            "slot_attendance": [1, 0, 2, 1],
            "over_slots": 1,
            "total_attendances": 4,
        },
        {
            "match_id": expected_match_id,
            "day": 2,
            "slot_attendance": [0, 3, 1, 0],
            "over_slots": 1,
            "total_attendances": 4,
        },
    ]


def test_day_aggregates_absent_when_actions_list_is_empty() -> None:
    # GIVEN history is present but episode 0 has no action records
    history = [
        {
            "state": {
                "public_state": {
                    "attendance_history": [[1, 0, 2, 1]],
                    "crowded_slots_today": [2],
                }
            },
            "payoffs": {},
            "is_terminal": True,
            "observations": {},
            "info": {},
        },
    ]
    episodes = [_make_episode(episode=0, actions=[], history=history)]
    result = _make_run_result(episodes=episodes)
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN day_aggregates_json is omitted because the El Farol gate is closed
    assert "day_aggregates_json" not in kwargs


def test_day_aggregates_absent_when_history_lacks_attendance() -> None:
    # GIVEN history entries with no attendance_history in public_state
    history = [
        {
            "state": {"public_state": {"crowded_slots_today": [2]}},
            "payoffs": {},
            "is_terminal": True,
            "observations": {},
            "info": {},
        },
    ]
    actions = [
        _make_action(
            match_id="m",
            day=1,
            agent_id="p0",
            intervals=_make_interval_pair(),
        )
    ]
    episodes = [_make_episode(episode=0, actions=actions, history=history)]
    result = _make_run_result(episodes=episodes)
    suite = _make_suite()
    game = _make_el_farol_game()

    # WHEN building kwargs
    kwargs = _build_game_result_kwargs(result, suite, game)

    # THEN day_aggregates_json is omitted because no attendance data was found
    assert "day_aggregates_json" not in kwargs


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
