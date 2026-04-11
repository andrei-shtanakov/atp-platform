"""Integration tests for ck_tournament_cancel_consistency CHECK constraint."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


@pytest.mark.anyio
@pytest.mark.parametrize(
    "fields,should_fail",
    [
        (
            {
                "status": "active",
                "cancelled_reason": None,
                "cancelled_by": None,
                "cancelled_at": None,
            },
            False,
        ),
        (
            {
                "status": "cancelled",
                "cancelled_reason": "admin_action",
                "cancelled_by": 1,
                "cancelled_at": "2026-04-15T10:00:00",
            },
            False,
        ),
        (
            {
                "status": "cancelled",
                "cancelled_reason": "pending_timeout",
                "cancelled_by": None,
                "cancelled_at": "2026-04-15T10:00:00",
            },
            False,
        ),
        (
            {
                "status": "cancelled",
                "cancelled_reason": "abandoned",
                "cancelled_by": None,
                "cancelled_at": "2026-04-15T10:00:00",
            },
            False,
        ),
        (
            {
                "status": "cancelled",
                "cancelled_reason": "admin_action",
                "cancelled_by": None,
                "cancelled_at": "2026-04-15T10:00:00",
            },
            True,
        ),
        (
            {
                "status": "cancelled",
                "cancelled_reason": "pending_timeout",
                "cancelled_by": 1,
                "cancelled_at": "2026-04-15T10:00:00",
            },
            True,
        ),
        (
            {
                "status": "cancelled",
                "cancelled_reason": "abandoned",
                "cancelled_by": 1,
                "cancelled_at": "2026-04-15T10:00:00",
            },
            True,
        ),
        (
            {
                "status": "cancelled",
                "cancelled_reason": "admin_action",
                "cancelled_by": 1,
                "cancelled_at": None,
            },
            True,
        ),
        (
            {
                "status": "cancelled",
                "cancelled_reason": None,
                "cancelled_by": 1,
                "cancelled_at": None,
            },
            True,
        ),
    ],
)
async def test_check_constraint_enforces_cancel_tuple(
    session_factory, fields, should_fail
):
    async with session_factory() as s:
        await s.execute(
            text(
                "INSERT INTO users "
                "(id, tenant_id, username, email, hashed_password, "
                "is_active, is_admin, created_at, updated_at) "
                "VALUES (1, 'default', 'alice', 'alice@test.com', 'x', 1, 0, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        await s.execute(
            text(
                "INSERT INTO tournaments "
                "(id, tenant_id, game_type, config, rules, status, "
                "num_players, total_rounds, round_deadline_s, "
                "pending_deadline, created_by, created_at) "
                "VALUES (1, 'default', 'prisoners_dilemma', "
                "'{\"name\": \"t\"}', '{}', "
                "'active', 2, 3, 30, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP)"
            )
        )
        await s.commit()

    async def _attempt_update() -> None:
        async with session_factory() as s:
            await s.execute(
                text(
                    "UPDATE tournaments SET "
                    "status = :status, "
                    "cancelled_reason = :cancelled_reason, "
                    "cancelled_by = :cancelled_by, "
                    "cancelled_at = :cancelled_at "
                    "WHERE id = 1"
                ),
                fields,
            )
            await s.commit()

    if should_fail:
        with pytest.raises(IntegrityError) as exc_info:
            await _attempt_update()
        assert "ck_tournament_cancel_consistency" in str(exc_info.value)
    else:
        await _attempt_update()
