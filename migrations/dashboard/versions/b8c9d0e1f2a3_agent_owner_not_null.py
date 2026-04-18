"""Agent.owner_id NOT NULL + drop ownerless partial unique index.

Backfills any remaining NULL owner_id rows to the lowest-id admin user,
then flips the column to NOT NULL and drops the partial unique index
added by migration e1b2c3d4f5a6 (LABS-15 short-term guardrail).

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-04-18 06:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()

    null_count = bind.execute(
        sa.text("SELECT COUNT(*) FROM agents WHERE owner_id IS NULL")
    ).scalar_one()

    if null_count:
        admin_id = bind.execute(
            sa.text("SELECT id FROM users WHERE is_admin = 1 ORDER BY id LIMIT 1")
        ).scalar_one_or_none()
        if admin_id is None:
            raise RuntimeError(
                f"Cannot backfill agents.owner_id: {null_count} rows have "
                "NULL owner_id but no admin user exists. Create an admin "
                "first (set users.is_admin=1) and re-run."
            )
        bind.execute(
            sa.text("UPDATE agents SET owner_id = :uid WHERE owner_id IS NULL"),
            {"uid": admin_id},
        )

    op.drop_index("uq_agent_ownerless_tenant_name_version", "agents")

    with op.batch_alter_table("agents") as batch_op:
        batch_op.alter_column(
            "owner_id",
            existing_type=sa.Integer(),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("agents") as batch_op:
        batch_op.alter_column(
            "owner_id",
            existing_type=sa.Integer(),
            nullable=True,
        )

    op.create_index(
        "uq_agent_ownerless_tenant_name_version",
        "agents",
        ["tenant_id", "name", "version"],
        unique=True,
        sqlite_where=sa.text("owner_id IS NULL"),
        postgresql_where=sa.text("owner_id IS NULL"),
    )
