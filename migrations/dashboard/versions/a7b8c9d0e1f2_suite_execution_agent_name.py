"""Decouple suite_executions from agents via denormalized agent_name.

Adds suite_executions.agent_name (populated by backfill from agents.name)
and relaxes suite_executions.agent_id to nullable. This lets the CLI
write suite executions without requiring an Agent row, breaking the
ownerless-agent creation path through the upload flow (LABS-54 Phase 1).

Revision ID: a7b8c9d0e1f2
Revises: f1a2b3c4d5e6
Create Date: 2026-04-17 12:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("suite_executions") as batch_op:
        batch_op.add_column(
            sa.Column("agent_name", sa.String(length=100), nullable=True)
        )

    op.execute(
        sa.text(
            "UPDATE suite_executions "
            "SET agent_name = (SELECT name FROM agents "
            "WHERE agents.id = suite_executions.agent_id) "
            "WHERE agent_name IS NULL"
        )
    )

    null_name_count = (
        op.get_bind()
        .execute(
            sa.text("SELECT COUNT(*) FROM suite_executions WHERE agent_name IS NULL")
        )
        .scalar_one()
    )
    if null_name_count:
        raise RuntimeError(
            f"Backfill left {null_name_count} suite_executions rows with NULL "
            "agent_name. Every row must have a resolvable agent.name."
        )

    with op.batch_alter_table("suite_executions") as batch_op:
        batch_op.alter_column(
            "agent_name",
            existing_type=sa.String(length=100),
            nullable=False,
        )
        batch_op.alter_column(
            "agent_id",
            existing_type=sa.Integer(),
            nullable=True,
        )
        batch_op.create_index(
            "idx_suite_agent_name",
            ["agent_name"],
        )


def downgrade() -> None:
    # WARNING: Safe only if no suite_executions rows have agent_id IS NULL.
    # After the migration has been live for any CLI run, those rows exist.
    # SQLite silently coerces NULLs to 0 on the NOT NULL flip; Postgres
    # raises NotNullViolation. Run
    #   SELECT COUNT(*) FROM suite_executions WHERE agent_id IS NULL;
    # first and either delete those rows or assign owner agents before
    # downgrading in a real environment.
    with op.batch_alter_table("suite_executions") as batch_op:
        batch_op.drop_index("idx_suite_agent_name")
        batch_op.alter_column(
            "agent_id",
            existing_type=sa.Integer(),
            nullable=False,
        )
        batch_op.drop_column("agent_name")
