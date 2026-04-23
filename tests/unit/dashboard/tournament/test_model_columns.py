"""Tests that model column declarations match Plan 2a schema deltas."""

from sqlalchemy import inspect

from atp.dashboard.tournament.models import (
    Action,
    Participant,
    Tournament,
)


def test_tournament_has_plan_2a_columns():
    columns = {c.name for c in inspect(Tournament).columns}
    required = {
        "pending_deadline",
        "join_token",
        "cancelled_at",
        "cancelled_by",
        "cancelled_reason",
        "cancelled_reason_detail",
    }
    assert required.issubset(columns), (
        f"Tournament missing columns: {required - columns}"
    )


def test_participant_has_released_at():
    columns = {c.name for c in inspect(Participant).columns}
    assert "released_at" in columns


def test_participant_user_id_nullable_for_builtins():
    # Plan 2a (IDOR fix) originally made user_id NOT NULL. LABS-TSA
    # PR-1 flipped it to nullable so builtin-strategy participants
    # (which have no backing User) can live in the same table.
    col = inspect(Participant).columns["user_id"]
    assert col.nullable is True


def test_action_has_source_column():
    columns = {c.name for c in inspect(Action).columns}
    assert "source" in columns


def test_action_source_has_server_default():
    col = inspect(Action).columns["source"]
    assert col.server_default is not None
