"""Tests that dashboard model column declarations match schema deltas."""


def test_suite_execution_has_agent_name_column() -> None:
    from atp.dashboard.models import SuiteExecution

    cols = {c.name: c for c in SuiteExecution.__table__.columns}
    assert "agent_name" in cols
    assert cols["agent_name"].nullable is False
    assert cols["agent_id"].nullable is True
