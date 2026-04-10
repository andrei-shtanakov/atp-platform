"""enforce benchmark_runs.user_id NOT NULL

Backfills any legacy rows with NULL user_id by assigning them to the
lowest-id admin user, then alters the column to NOT NULL.  This closes
the IDOR-vs-ownership gap where existing rows had no owner and the API
couldn't tell who started them.

If no admin user exists but legacy NULL rows are present, the migration
raises — the operator must create an admin before retrying.  On a fresh
database (no rows) the backfill is a no-op and only the ALTER runs.

Revision ID: c8d5f2a91234
Revises: b3a1f7c2d4e5
Create Date: 2026-04-10 13:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8d5f2a91234"
down_revision: str | Sequence[str] | None = "b3a1f7c2d4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()

    null_count = bind.execute(
        sa.text("SELECT COUNT(*) FROM benchmark_runs WHERE user_id IS NULL")
    ).scalar_one()

    if null_count:
        admin_id = bind.execute(
            sa.text("SELECT id FROM users WHERE is_admin = 1 ORDER BY id LIMIT 1")
        ).scalar_one_or_none()
        if admin_id is None:
            raise RuntimeError(
                f"Cannot backfill benchmark_runs.user_id: {null_count} rows "
                "have NULL user_id but no admin user exists. Create an admin "
                "first (set users.is_admin=1) and re-run the migration."
            )
        bind.execute(
            sa.text("UPDATE benchmark_runs SET user_id = :uid WHERE user_id IS NULL"),
            {"uid": admin_id},
        )

    with op.batch_alter_table("benchmark_runs") as batch_op:
        batch_op.alter_column(
            "user_id",
            existing_type=sa.Integer(),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("benchmark_runs") as batch_op:
        batch_op.alter_column(
            "user_id",
            existing_type=sa.Integer(),
            nullable=True,
        )
