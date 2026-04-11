"""Static architectural guard tests. Run in the unit stage for sub-30s
feedback. They enforce architectural invariants via git grep."""

from pathlib import Path

from tests.unit.dashboard.tournament._grep_helper import grep_pattern


def test_cancel_tournament_system_not_called_from_handlers():
    """Twin-methods invariant: system method must not be reachable from
    any REST or MCP handler. Called only from the deadline worker and
    from service.leave() (same module)."""
    matches = grep_pattern(
        r"cancel_tournament_system\b",
        paths=[
            "packages/atp-dashboard/atp/dashboard/mcp",
            "packages/atp-dashboard/atp/dashboard/v2/routes",
        ],
    )
    assert matches == [], (
        f"cancel_tournament_system called from handler files "
        f"(must be deadline_worker/service-only): "
        f"{[m.path for m in matches]}"
    )


def test_no_bare_string_round_status_comparisons():
    """Plan 2a refactor invariant: all Round.status comparisons must use
    RoundStatus enum, not bare string literals."""
    matches = grep_pattern(
        r'Round\.status\s*[=!]=\s*["\x27]',
        paths=["packages/atp-dashboard/atp/dashboard/tournament"],
    )
    assert matches == [], (
        f"Bare string literal comparison on Round.status: "
        f"{[m.path + ':' + str(m.line_number) for m in matches]}. "
        f"Use RoundStatus enum from models.py."
    )


def test_no_direct_cancel_field_writes_outside_service():
    """All writes to cancelled_* fields must go through _cancel_impl
    (which lives in service.py)."""
    matches = grep_pattern(
        r"\.(cancelled_by|cancelled_at|cancelled_reason|cancelled_reason_detail)\s*=",
        paths=["packages/atp-dashboard/atp/dashboard/tournament"],
    )
    allowed_file = "service.py"
    bad = [m for m in matches if Path(m.path).name != allowed_file]
    assert bad == [], (
        f"Direct cancel-field writes outside service.py _cancel_impl: "
        f"{[m.path + ':' + str(m.line_number) for m in bad]}"
    )
