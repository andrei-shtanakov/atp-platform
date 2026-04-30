"""Tests for Phase 6 El Farol dashboard data model: AgentRecord.

These tests cover the new ``AgentRecord`` dataclass being added to
``atp_games.models`` and the new ``agents: list[AgentRecord]`` field on
``GameResult``.

Tests are written TDD-first: the imports below target symbols that do not yet
exist, so this module is expected to fail on collection (``AgentRecord``
missing) until Phase 6 implementation lands. The ``GameResult`` tests will
also fail because the ``agents`` field is not yet declared.

Plan reference: docs/plans/el-farol-dashboard-data-model.md §8.2.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from atp_games.models import AgentRecord, GameResult, GameRunConfig

# ---------------------------------------------------------------------------
# AgentRecord — construction and defaults
# ---------------------------------------------------------------------------


class TestAgentRecord:
    def test_required_fields_only(self) -> None:
        # GIVEN the minimum required identity fields
        # WHEN constructing an AgentRecord
        record = AgentRecord(
            agent_id="p0",
            display_name="Tit for Tat",
            user_id="u-42",
        )

        # THEN required fields are stored
        assert record.agent_id == "p0"
        assert record.display_name == "Tit for Tat"
        assert record.user_id == "u-42"

        # AND optional fields default as documented
        assert record.user_display is None
        assert record.family is None
        assert record.adapter_type == "unknown"
        assert record.model_id is None
        assert record.color is None

    def test_all_fields_populated(self) -> None:
        # GIVEN values for every field (8 total)
        # WHEN constructing an AgentRecord
        record = AgentRecord(
            agent_id="p1",
            display_name="Calibrated Player",
            user_id="u-7",
            user_display="Alice",
            family="calibrated",
            adapter_type="MCP",
            model_id="gpt-4o-mini",
            color="#ff8800",
        )

        # THEN every field is readable
        assert record.agent_id == "p1"
        assert record.display_name == "Calibrated Player"
        assert record.user_id == "u-7"
        assert record.user_display == "Alice"
        assert record.family == "calibrated"
        assert record.adapter_type == "MCP"
        assert record.model_id == "gpt-4o-mini"
        assert record.color == "#ff8800"

    def test_agent_id_must_be_nonempty(self) -> None:
        # WHEN agent_id is the empty string
        # THEN __post_init__ rejects it
        with pytest.raises(ValueError):
            AgentRecord(
                agent_id="",
                display_name="tft",
                user_id="u1",
            )

    def test_user_id_must_be_nonempty(self) -> None:
        # WHEN user_id is the empty string
        # THEN __post_init__ rejects it
        with pytest.raises(ValueError):
            AgentRecord(
                agent_id="p0",
                display_name="tft",
                user_id="",
            )

    def test_display_name_must_be_nonempty(self) -> None:
        # WHEN display_name is the empty string
        # THEN __post_init__ rejects it
        with pytest.raises(ValueError):
            AgentRecord(
                agent_id="p0",
                display_name="",
                user_id="u1",
            )

    def test_from_legacy_sets_user_id_unknown(self) -> None:
        # GIVEN legacy data with only an agent_id and display_name
        # WHEN we use the compatibility helper
        record = AgentRecord.from_legacy(agent_id="p0", display_name="tft")

        # THEN it returns an AgentRecord with user_id="unknown"
        # and adapter_type defaulted to "unknown"
        assert isinstance(record, AgentRecord)
        assert record.agent_id == "p0"
        assert record.display_name == "tft"
        assert record.user_id == "unknown"
        assert record.adapter_type == "unknown"

    def test_agent_record_roundtrip_through_asdict(self) -> None:
        # GIVEN a fully-populated AgentRecord
        record = AgentRecord(
            agent_id="p0",
            display_name="tft",
            user_id="u1",
            user_display="Alice",
            family="calibrated",
            adapter_type="MCP",
            model_id="gpt-4o-mini",
            color="#abcdef",
        )

        # WHEN serialising with dataclasses.asdict
        data = asdict(record)

        # THEN every one of the 8 fields is present in the dict
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
        assert set(data.keys()) == expected_keys

        # AND values round-trip faithfully
        assert data["agent_id"] == "p0"
        assert data["display_name"] == "tft"
        assert data["user_id"] == "u1"
        assert data["user_display"] == "Alice"
        assert data["family"] == "calibrated"
        assert data["adapter_type"] == "MCP"
        assert data["model_id"] == "gpt-4o-mini"
        assert data["color"] == "#abcdef"


# ---------------------------------------------------------------------------
# GameResult.agents — new typed agent roster field
# ---------------------------------------------------------------------------


class TestGameResultAgents:
    def test_game_result_agents_field_defaults_empty(self) -> None:
        # GIVEN a GameResult constructed without an agents argument
        # WHEN we inspect result.agents
        result = GameResult(game_name="PD", config=GameRunConfig())

        # THEN it defaults to an empty list
        assert result.agents == []

    def test_game_result_accepts_agents_list(self) -> None:
        # GIVEN an AgentRecord in the roster
        record = AgentRecord(
            agent_id="p0",
            display_name="tft",
            user_id="u1",
        )

        # WHEN a GameResult is constructed with that roster
        result = GameResult(
            game_name="PD",
            config=GameRunConfig(),
            agents=[record],
        )

        # THEN the roster is stored and addressable
        assert len(result.agents) == 1
        assert result.agents[0].user_id == "u1"
        assert result.agents[0].agent_id == "p0"
        assert result.agents[0].display_name == "tft"

    def test_game_result_to_dict_includes_agents_when_present(self) -> None:
        # GIVEN a GameResult with one AgentRecord in the roster
        record = AgentRecord(
            agent_id="p0",
            display_name="tft",
            user_id="u1",
        )
        result = GameResult(
            game_name="PD",
            config=GameRunConfig(),
            agents=[record],
        )

        # WHEN we serialise to dict
        data = result.to_dict()

        # THEN agents is present as a list of dicts carrying identity fields
        assert "agents" in data
        agents_data = data["agents"]
        assert isinstance(agents_data, list)
        assert len(agents_data) == 1
        agent_dict = agents_data[0]
        assert isinstance(agent_dict, dict)
        assert agent_dict["agent_id"] == "p0"
        assert agent_dict["display_name"] == "tft"
        assert agent_dict["user_id"] == "u1"

        # AND when agents is empty, the key may be omitted (matching the
        # existing `if self.actions: result["actions"] = ...` pattern).
        empty_result = GameResult(game_name="PD", config=GameRunConfig())
        empty_data = empty_result.to_dict()
        if "agents" in empty_data:
            assert empty_data["agents"] == []
