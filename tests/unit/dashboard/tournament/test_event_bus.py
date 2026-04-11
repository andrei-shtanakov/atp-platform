"""Tests for TournamentEventBus and TournamentEvent."""

from __future__ import annotations

from datetime import UTC, datetime


def test_tournament_event_dataclass_holds_required_fields() -> None:
    from atp.dashboard.tournament.events import TournamentEvent

    event = TournamentEvent(
        event_type="round_started",
        tournament_id=7,
        round_number=1,
        data={"foo": "bar"},
        timestamp=datetime(2026, 4, 10, 12, 0, 0, tzinfo=UTC),
    )
    assert event.event_type == "round_started"
    assert event.tournament_id == 7
    assert event.round_number == 1
    assert event.data == {"foo": "bar"}
    assert event.timestamp.year == 2026
