"""tournament plan 2a — schema constraints, cancel audit, AD-9 + AD-10 columns

Revision ID: c60b45e516be
Revises: 028d8a9fdc46
Create Date: 2026-04-11

## Precondition

Transitively follows c8d5f2a91234 (IDOR fix, enforce_run_user_id_not_null)
via 028d8a9fdc46. IDOR fix backfilled and constrained benchmark_runs; Plan
2a applies the analogous invariant to tournament_participants, a sibling
table not touched by the IDOR migration. The FK-orphan probe on
tournament_participants therefore verifies a fresh invariant, not a
re-check of earlier work.

## Probe-to-resolution playbook

| Probe | Violation | Resolution |
|-------|-----------|------------|
| P1 | Participant user_id IS NULL | DELETE anonymous rows, or backfill user_id from agent_name lookup if meaningful. Do NOT assign a sentinel user_id. |
| P2 | FK orphan on user_id | DELETE FROM tournament_participants WHERE user_id NOT IN (SELECT id FROM users). |
| P3 | Duplicate (tournament_id, user_id) | Manually dedupe, keep earliest joined_at. |
| P4 | Duplicate (round_id, participant_id) | Manually dedupe, keep earliest submitted_at. |
| P5 | Duplicate (tournament_id, round_number) | Inspect tournament history before deleting. Likely cause: re-submitted create-round during a crash. |
| P6 | User with >1 participant in pending/active tournaments | Transition stale tournaments to completed/cancelled status via SQL; migration step 5a backfill will set released_at automatically. Do NOT try to set released_at at probe time — column does not exist yet. |
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

from atp.dashboard.migrations.probes.check_tournament_invariants import (
    check_tournament_schema_ready,
)

# revision identifiers, used by Alembic.
revision: str = "c60b45e516be"
down_revision: str | Sequence[str] | None = "028d8a9fdc46"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply Plan 2a schema constraints and additive columns."""
    connection = op.get_bind()

    # Step 1: Run probes. Abort on any violation.
    violations = check_tournament_schema_ready(connection)
    if violations:
        message = "Plan 2a migration aborted by probe — resolve and re-run:\n"
        for v in violations:
            message += f"  - {v}\n"
        raise RuntimeError(message)

    # Step 2: Tournament columns — pending_deadline nullable first,
    # backfill, then flip to NOT NULL.
    with op.batch_alter_table("tournaments") as batch_op:
        batch_op.add_column(sa.Column("pending_deadline", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("join_token", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("cancelled_at", sa.DateTime(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "cancelled_by",
                sa.Integer(),
                sa.ForeignKey(
                    "users.id",
                    name="fk_tournaments_cancelled_by_users",
                    ondelete="SET NULL",
                ),
                nullable=True,
            )
        )
        batch_op.add_column(sa.Column("cancelled_reason", sa.String(32), nullable=True))
        batch_op.add_column(
            sa.Column("cancelled_reason_detail", sa.String(512), nullable=True)
        )

    op.execute(
        text(
            "UPDATE tournaments SET pending_deadline = CURRENT_TIMESTAMP "
            "WHERE pending_deadline IS NULL"
        )
    )

    with op.batch_alter_table("tournaments") as batch_op:
        batch_op.alter_column(
            "pending_deadline", existing_type=sa.DateTime(), nullable=False
        )
        batch_op.create_index(
            "idx_tournaments_status_pending_deadline",
            ["status", "pending_deadline"],
            unique=False,
        )
        batch_op.create_check_constraint(
            "ck_tournament_cancel_consistency",
            """(
                (
                    cancelled_reason IS NULL
                    AND cancelled_by IS NULL
                    AND cancelled_at IS NULL
                ) OR (
                    cancelled_reason = 'admin_action'
                    AND cancelled_by IS NOT NULL
                    AND cancelled_at IS NOT NULL
                ) OR (
                    cancelled_reason IN ('pending_timeout', 'abandoned')
                    AND cancelled_by IS NULL
                    AND cancelled_at IS NOT NULL
                )
            )""",
        )

    # Step 3: Round — unique constraint + composite index
    with op.batch_alter_table("tournament_rounds") as batch_op:
        batch_op.create_unique_constraint(
            "uq_round_tournament_number", ["tournament_id", "round_number"]
        )
        batch_op.create_index(
            "idx_round_status_deadline", ["status", "deadline"], unique=False
        )

    # Step 4: Action — source column + uq_action_round_participant
    with op.batch_alter_table("tournament_actions") as batch_op:
        batch_op.add_column(
            sa.Column(
                "source",
                sa.String(32),
                nullable=False,
                server_default="submitted",
            )
        )
        batch_op.create_unique_constraint(
            "uq_action_round_participant", ["round_id", "participant_id"]
        )

    # Step 5: Participant — released_at, NOT NULL flip,
    # uq_participant_tournament_user
    with op.batch_alter_table("tournament_participants") as batch_op:
        batch_op.add_column(sa.Column("released_at", sa.DateTime(), nullable=True))
        batch_op.alter_column("user_id", existing_type=sa.Integer(), nullable=False)
        batch_op.create_unique_constraint(
            "uq_participant_tournament_user", ["tournament_id", "user_id"]
        )

    # Step 5a: Backfill released_at for participants in terminal-status
    # tournaments.
    op.execute(
        text("""
        UPDATE tournament_participants
        SET released_at = CURRENT_TIMESTAMP
        WHERE tournament_id IN (
            SELECT id FROM tournaments
            WHERE status IN ('completed', 'cancelled')
        )
    """)
    )

    # Step 6: Partial unique index — created outside batch_alter_table.
    op.create_index(
        "uq_participant_user_active",
        "tournament_participants",
        ["user_id"],
        unique=True,
        sqlite_where=text("user_id IS NOT NULL AND released_at IS NULL"),
        postgresql_where=text("user_id IS NOT NULL AND released_at IS NULL"),
    )


def downgrade() -> None:
    """Reverse Plan 2a schema constraints and additive columns."""
    op.drop_index("uq_participant_user_active", "tournament_participants")

    with op.batch_alter_table("tournament_participants") as batch_op:
        batch_op.drop_constraint("uq_participant_tournament_user", type_="unique")
        batch_op.alter_column("user_id", existing_type=sa.Integer(), nullable=True)
        batch_op.drop_column("released_at")

    with op.batch_alter_table("tournament_actions") as batch_op:
        batch_op.drop_constraint("uq_action_round_participant", type_="unique")
        batch_op.drop_column("source")

    with op.batch_alter_table("tournament_rounds") as batch_op:
        batch_op.drop_index("idx_round_status_deadline")
        batch_op.drop_constraint("uq_round_tournament_number", type_="unique")

    with op.batch_alter_table("tournaments") as batch_op:
        batch_op.drop_constraint("ck_tournament_cancel_consistency", type_="check")
        batch_op.drop_index("idx_tournaments_status_pending_deadline")
        batch_op.drop_column("cancelled_reason_detail")
        batch_op.drop_column("cancelled_reason")
        batch_op.drop_constraint(
            "fk_tournaments_cancelled_by_users", type_="foreignkey"
        )
        batch_op.drop_column("cancelled_by")
        batch_op.drop_column("cancelled_at")
        batch_op.drop_column("join_token")
        batch_op.drop_column("pending_deadline")
