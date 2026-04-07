"""add webhook_url and run events columns

Revision ID: b3a1f7c2d4e5
Revises: 4e902371a941
Create Date: 2026-04-07 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3a1f7c2d4e5"
down_revision: str | Sequence[str] | None = "4e902371a941"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add webhook_url to benchmarks
    with op.batch_alter_table("benchmarks") as batch_op:
        batch_op.add_column(
            sa.Column("webhook_url", sa.String(2048), nullable=True),
        )

    # Add events to benchmark_runs
    with op.batch_alter_table("benchmark_runs") as batch_op:
        batch_op.add_column(
            sa.Column("events", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    with op.batch_alter_table("benchmark_runs") as batch_op:
        batch_op.drop_column("events")

    with op.batch_alter_table("benchmarks") as batch_op:
        batch_op.drop_column("webhook_url")
