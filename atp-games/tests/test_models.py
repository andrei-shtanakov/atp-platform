"""Tests for data models."""

import json
from datetime import datetime, timezone

import pytest

from atp_games.models import (
    ActionRecord,
    AgentRecord,
    EpisodeResult,
    GameResult,
    GameRunConfig,
    IntervalPair,
)


def _make_action_record(
    *,
    match_id: str = "match-1",
    day: int = 0,
    agent_id: str = "p0",
    intervals: IntervalPair | None = None,
    picks: tuple[int, ...] = (0, 1, 2),
    payoff: float = 1.5,
    num_under: int = 1,
    num_over: int = 0,
    intent: str | None = None,
) -> ActionRecord:
    """Build an ActionRecord with sane defaults for tests."""
    if intervals is None:
        intervals = IntervalPair(
            first=(0, 2),
            second=(),
            num_slots=16,
            max_total_slots=8,
        )
    return ActionRecord(
        match_id=match_id,
        day=day,
        agent_id=agent_id,
        intervals=intervals,
        picks=picks,
        num_visits=intervals.num_visits(),
        total_slots=intervals.total_slots(),
        payoff=payoff,
        num_under=num_under,
        num_over=num_over,
        intent=intent,
    )


class TestGameRunConfig:
    def test_defaults(self) -> None:
        config = GameRunConfig()
        assert config.episodes == 1
        assert config.max_retries == 3
        assert config.move_timeout == 30.0

    def test_custom_values(self) -> None:
        config = GameRunConfig(episodes=10, max_retries=5, move_timeout=60.0)
        assert config.episodes == 10
        assert config.max_retries == 5
        assert config.move_timeout == 60.0

    def test_invalid_episodes(self) -> None:
        with pytest.raises(ValueError, match="episodes"):
            GameRunConfig(episodes=0)

    def test_invalid_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            GameRunConfig(max_retries=-1)

    def test_invalid_move_timeout(self) -> None:
        with pytest.raises(ValueError, match="move_timeout"):
            GameRunConfig(move_timeout=0)

    def test_frozen(self) -> None:
        config = GameRunConfig()
        with pytest.raises(AttributeError):
            config.episodes = 5  # type: ignore[misc]


class TestEpisodeResult:
    def test_basic(self) -> None:
        ep = EpisodeResult(
            episode=0,
            payoffs={"p0": 3.0, "p1": 3.0},
        )
        assert ep.episode == 0
        assert ep.payoffs == {"p0": 3.0, "p1": 3.0}

    def test_roundtrip(self) -> None:
        ep = EpisodeResult(
            episode=1,
            payoffs={"p0": 5.0, "p1": 0.0},
            actions_log=[{"p0": "cooperate", "p1": "defect"}],
        )
        data = ep.to_dict()
        restored = EpisodeResult.from_dict(data)
        assert restored.episode == ep.episode
        assert restored.payoffs == ep.payoffs
        assert restored.actions_log == ep.actions_log


class TestGameResult:
    def test_average_payoffs(self) -> None:
        config = GameRunConfig(episodes=2)
        result = GameResult(
            game_name="Test Game",
            config=config,
            episodes=[
                EpisodeResult(
                    episode=0,
                    payoffs={"p0": 2.0, "p1": 4.0},
                ),
                EpisodeResult(
                    episode=1,
                    payoffs={"p0": 4.0, "p1": 6.0},
                ),
            ],
        )
        avg = result.average_payoffs
        assert avg["p0"] == pytest.approx(3.0)
        assert avg["p1"] == pytest.approx(5.0)

    def test_average_payoffs_empty(self) -> None:
        result = GameResult(
            game_name="Empty",
            config=GameRunConfig(),
        )
        assert result.average_payoffs == {}

    def test_num_episodes(self) -> None:
        result = GameResult(
            game_name="Test",
            config=GameRunConfig(),
            episodes=[
                EpisodeResult(episode=0, payoffs={"p0": 1.0}),
            ],
        )
        assert result.num_episodes == 1

    def test_roundtrip(self) -> None:
        config = GameRunConfig(episodes=1, max_retries=2, move_timeout=15.0)
        result = GameResult(
            game_name="PD",
            config=config,
            agent_names={"p0": "tft", "p1": "alld"},
        )
        data = result.to_dict()
        restored = GameResult.from_dict(data)
        assert restored.game_name == "PD"
        assert restored.config.max_retries == 2
        assert restored.agent_names == {"p0": "tft", "p1": "alld"}

    def test_game_result_to_dict_includes_run_id(self) -> None:
        # GIVEN a GameResult constructed with an explicit run_id
        result = GameResult(
            game_name="PD",
            config=GameRunConfig(),
            run_id="abc-123",
        )

        # THEN to_dict() surfaces the run_id verbatim
        assert result.to_dict()["run_id"] == "abc-123"

    def test_game_result_from_dict_roundtrips_run_id(self) -> None:
        # GIVEN a GameResult with an explicit run_id
        result = GameResult(
            game_name="PD",
            config=GameRunConfig(),
            run_id="abc-123",
        )

        # WHEN roundtripping through to_dict + from_dict
        restored = GameResult.from_dict(result.to_dict())

        # THEN run_id is preserved
        assert restored.run_id == "abc-123"

    def test_game_result_from_dict_without_run_id_generates_new(self) -> None:
        # GIVEN a serialized GameResult payload with NO "run_id" key
        data = {
            "game_name": "PD",
            "config": {
                "episodes": 1,
                "max_retries": 3,
                "move_timeout": 30.0,
                "parallel": 1,
            },
            "episodes": [],
            "agent_names": {},
        }
        assert "run_id" not in data

        # WHEN deserializing
        restored = GameResult.from_dict(data)

        # THEN a fresh non-empty run_id is generated
        assert isinstance(restored.run_id, str)
        assert restored.run_id, "expected a non-empty run_id to be generated"


class TestDashboardRoundtrip:
    """Regression tests for Tier-1 dashboard field preservation."""

    def test_episode_result_roundtrip_preserves_actions(self) -> None:
        # GIVEN an EpisodeResult with 2 ActionRecord entries
        intervals_a = IntervalPair(
            first=(0, 2),
            second=(5, 6),
            num_slots=16,
            max_total_slots=8,
        )
        intervals_b = IntervalPair(
            first=(1, 3),
            second=(),
            num_slots=16,
            max_total_slots=8,
        )
        action_a = _make_action_record(
            match_id="match-xyz",
            day=0,
            agent_id="p0",
            intervals=intervals_a,
            picks=(0, 1, 2, 5, 6),
            payoff=2.5,
            num_under=1,
            num_over=1,
            intent="foo",
        )
        action_b = _make_action_record(
            match_id="match-xyz",
            day=0,
            agent_id="p1",
            intervals=intervals_b,
            picks=(1, 2, 3),
            payoff=-0.5,
            num_under=0,
            num_over=2,
            intent="bar",
        )
        ep = EpisodeResult(
            episode=0,
            payoffs={"p0": 2.5, "p1": -0.5},
            actions=[action_a, action_b],
        )

        # WHEN roundtripping through to_dict/from_dict
        restored = EpisodeResult.from_dict(ep.to_dict())

        # THEN both action records are preserved verbatim
        assert len(restored.actions) == 2
        first = restored.actions[0]
        assert first.match_id == "match-xyz"
        assert first.day == 0
        assert first.agent_id == "p0"
        assert first.intervals.first == (0, 2)
        assert first.intervals.second == (5, 6)
        assert first.picks == (0, 1, 2, 5, 6)
        assert first.payoff == 2.5
        assert first.intent == "foo"

    def test_episode_result_roundtrip_preserves_round_payoffs(self) -> None:
        # GIVEN an EpisodeResult with per-round payoffs
        ep = EpisodeResult(
            episode=2,
            payoffs={"p0": 1.5, "p1": -0.5},
            round_payoffs=[
                {"p0": 1.0, "p1": -1.0},
                {"p0": 0.5, "p1": 0.5},
            ],
        )

        # WHEN roundtripping
        restored = EpisodeResult.from_dict(ep.to_dict())

        # THEN round_payoffs match exactly
        assert restored.round_payoffs == [
            {"p0": 1.0, "p1": -1.0},
            {"p0": 0.5, "p1": 0.5},
        ]

    def test_game_result_roundtrip_preserves_agents(self) -> None:
        # GIVEN a GameResult with 2 AgentRecord entries
        agent_a = AgentRecord(
            agent_id="p0",
            display_name="Alpha",
            user_id="user-1",
            user_display="Alice",
            family="gpt",
            adapter_type="openai",
            model_id="gpt-4o-mini",
            color="#ff0000",
        )
        agent_b = AgentRecord(
            agent_id="p1",
            display_name="Beta",
            user_id="user-2",
        )
        result = GameResult(
            game_name="El Farol",
            config=GameRunConfig(),
            agents=[agent_a, agent_b],
        )

        # WHEN roundtripping
        restored = GameResult.from_dict(result.to_dict())

        # THEN both agents are preserved with all fields
        assert len(restored.agents) == 2
        first = restored.agents[0]
        assert first.agent_id == "p0"
        assert first.display_name == "Alpha"
        assert first.user_id == "user-1"
        assert first.user_display == "Alice"
        assert first.family == "gpt"
        assert first.adapter_type == "openai"
        assert first.model_id == "gpt-4o-mini"
        assert first.color == "#ff0000"

    def test_game_result_roundtrip_full_fidelity(self) -> None:
        # GIVEN a fully-populated GameResult
        action = _make_action_record(
            match_id="match-full",
            day=0,
            agent_id="p0",
            intent="visit-early",
        )
        episode = EpisodeResult(
            episode=0,
            payoffs={"p0": 1.5, "p1": -0.5},
            actions=[action],
            round_payoffs=[{"p0": 1.5, "p1": -0.5}],
        )
        agents = [
            AgentRecord(
                agent_id="p0",
                display_name="Alpha",
                user_id="user-1",
            ),
            AgentRecord(
                agent_id="p1",
                display_name="Beta",
                user_id="user-2",
            ),
        ]
        result = GameResult(
            game_name="El Farol",
            config=GameRunConfig(),
            episodes=[episode],
            agents=agents,
            run_id="run-abc",
        )

        # WHEN roundtripping
        restored = GameResult.from_dict(result.to_dict())

        # THEN nothing is dropped
        assert restored.run_id == "run-abc"
        assert restored.game_name == "El Farol"
        assert len(restored.agents) == 2
        assert len(restored.episodes) == 1
        assert len(restored.episodes[0].actions) == 1
        assert restored.episodes[0].actions[0].intent == "visit-early"
        assert restored.episodes[0].round_payoffs == [{"p0": 1.5, "p1": -0.5}]

    def test_interval_pair_from_dict_accepts_list_form(self) -> None:
        # GIVEN the asdict/JSON form where tuples become lists
        data = {
            "first": [0, 2],
            "second": [],
            "num_slots": 16,
            "max_total_slots": 8,
        }

        # WHEN rebuilding via from_dict
        pair = IntervalPair.from_dict(data)

        # THEN tuples are restored and values match
        assert pair == IntervalPair(
            first=(0, 2),
            second=(),
            num_slots=16,
            max_total_slots=8,
        )
        assert pair.first == (0, 2)
        assert pair.second == ()

    def test_action_record_from_dict_handles_iso_submitted_at(self) -> None:
        # GIVEN an ActionRecord dict with an ISO-8601 submitted_at string
        data = {
            "match_id": "match-1",
            "day": 0,
            "agent_id": "p0",
            "intervals": {
                "first": [0, 1],
                "second": [],
                "num_slots": 16,
                "max_total_slots": 8,
            },
            "picks": [0, 1],
            "num_visits": 1,
            "total_slots": 2,
            "payoff": 1.0,
            "num_under": 0,
            "num_over": 0,
            "submitted_at": "2026-04-22T10:30:00",
        }

        # WHEN rebuilding
        record = ActionRecord.from_dict(data)

        # THEN submitted_at is parsed to a datetime with matching components
        assert isinstance(record.submitted_at, datetime)
        assert record.submitted_at.year == 2026
        assert record.submitted_at.month == 4
        assert record.submitted_at.day == 22
        assert record.submitted_at.hour == 10
        assert record.submitted_at.minute == 30
        assert record.submitted_at.second == 0

    def test_game_result_roundtrip_via_json(self) -> None:
        # GIVEN a GameResult with actions, agents, and round_payoffs
        action = _make_action_record(
            match_id="match-json",
            day=0,
            agent_id="p0",
            intent="probe",
        )
        episode = EpisodeResult(
            episode=0,
            payoffs={"p0": 1.5, "p1": -0.5},
            actions=[action],
            round_payoffs=[{"p0": 1.5, "p1": -0.5}],
        )
        agents = [
            AgentRecord(
                agent_id="p0",
                display_name="Alpha",
                user_id="user-1",
            ),
        ]
        result = GameResult(
            game_name="El Farol",
            config=GameRunConfig(),
            episodes=[episode],
            agents=agents,
            run_id="run-json",
        )

        # WHEN serializing via json.dumps and parsing back
        payload = json.loads(json.dumps(result.to_dict()))
        restored = GameResult.from_dict(payload)

        # THEN the JSON roundtrip preserves dashboard fields
        assert restored.run_id == "run-json"
        assert len(restored.agents) == 1
        assert restored.agents[0].agent_id == "p0"
        assert len(restored.episodes) == 1
        assert len(restored.episodes[0].actions) == 1
        assert restored.episodes[0].actions[0].intent == "probe"
        assert restored.episodes[0].actions[0].intervals.first == (0, 2)
        assert restored.episodes[0].round_payoffs == [{"p0": 1.5, "p1": -0.5}]

    def test_episode_result_to_dict_serializes_submitted_at_as_iso_string(
        self,
    ) -> None:
        # GIVEN an ActionRecord whose submitted_at is a naive datetime
        action = _make_action_record(
            match_id="match-iso",
            day=0,
            agent_id="p0",
        )
        action.submitted_at = datetime(2026, 4, 22, 10, 30, 0)
        ep = EpisodeResult(
            episode=0,
            payoffs={"p0": 1.0},
            actions=[action],
        )

        # WHEN serializing
        result = ep.to_dict()

        # THEN submitted_at is emitted as an ISO-8601 string
        assert isinstance(result["actions"][0]["submitted_at"], str)
        assert result["actions"][0]["submitted_at"] == "2026-04-22T10:30:00"

    def test_episode_result_to_dict_preserves_submitted_at_none(self) -> None:
        # GIVEN an ActionRecord with submitted_at left at its default (None)
        action = _make_action_record(
            match_id="match-none",
            day=0,
            agent_id="p0",
        )
        assert action.submitted_at is None
        ep = EpisodeResult(
            episode=0,
            payoffs={"p0": 1.0},
            actions=[action],
        )

        # WHEN serializing
        result = ep.to_dict()

        # THEN submitted_at remains None (or absent) — no stringification
        payload = result["actions"][0]
        assert payload.get("submitted_at") is None

    def test_game_result_to_dict_is_json_serializable_with_timestamps(self) -> None:
        # GIVEN a GameResult whose episode carries an action with a timestamp
        original_ts = datetime(2026, 4, 22, 10, 30, 0)
        action = _make_action_record(
            match_id="match-json-ts",
            day=0,
            agent_id="p0",
        )
        action.submitted_at = original_ts
        episode = EpisodeResult(
            episode=0,
            payoffs={"p0": 1.5},
            actions=[action],
        )
        result = GameResult(
            game_name="El Farol",
            config=GameRunConfig(),
            episodes=[episode],
            run_id="run-json-ts",
        )

        # WHEN dumping to JSON, parsing, and rebuilding
        # THEN json.dumps must NOT raise on the raw datetime
        payload = json.dumps(result.to_dict())
        restored = GameResult.from_dict(json.loads(payload))

        # THEN the roundtripped submitted_at is a matching datetime
        restored_action = restored.episodes[0].actions[0]
        assert isinstance(restored_action.submitted_at, datetime)
        assert restored_action.submitted_at == original_ts

    def test_action_record_roundtrip_preserves_submitted_at_across_json(
        self,
    ) -> None:
        # GIVEN an ActionRecord with a naive submitted_at datetime
        original_ts = datetime(2026, 4, 22, 10, 30, 0)
        action = _make_action_record(
            match_id="match-rt",
            day=0,
            agent_id="p0",
        )
        action.submitted_at = original_ts
        ep = EpisodeResult(
            episode=0,
            payoffs={"p0": 1.0},
            actions=[action],
        )

        # WHEN roundtripping EpisodeResult -> dict -> json -> dict -> EpisodeResult
        payload = json.loads(json.dumps(ep.to_dict()))
        restored = EpisodeResult.from_dict(payload)

        # THEN the ActionRecord's submitted_at is the exact original datetime
        assert restored.actions[0].submitted_at == original_ts

    def test_submitted_at_with_timezone_roundtrips(self) -> None:
        # GIVEN an ActionRecord with a timezone-aware submitted_at
        original_ts = datetime(2026, 4, 22, 10, 30, tzinfo=timezone.utc)
        action = _make_action_record(
            match_id="match-tz",
            day=0,
            agent_id="p0",
        )
        action.submitted_at = original_ts
        ep = EpisodeResult(
            episode=0,
            payoffs={"p0": 1.0},
            actions=[action],
        )

        # WHEN roundtripping via JSON
        payload = json.loads(json.dumps(ep.to_dict()))
        restored = EpisodeResult.from_dict(payload)

        # THEN the timezone-aware datetime is preserved
        restored_ts = restored.actions[0].submitted_at
        assert isinstance(restored_ts, datetime)
        assert restored_ts == original_ts
        assert restored_ts.tzinfo is not None
        assert restored_ts.utcoffset() == timezone.utc.utcoffset(None)

    def test_game_result_from_dict_legacy_no_new_fields(self) -> None:
        # GIVEN a minimal legacy payload with no agents/run_id
        data = {
            "game_name": "Legacy",
            "config": {},
            "episodes": [],
        }

        # WHEN deserializing
        restored = GameResult.from_dict(data)

        # THEN the result has an empty agents list and a generated run_id
        assert restored.game_name == "Legacy"
        assert restored.agents == []
        assert restored.episodes == []
        assert isinstance(restored.run_id, str)
        assert restored.run_id, "expected a non-empty generated run_id"
