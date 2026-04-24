"""action_reasoning

Add nullable Text column ``reasoning`` to ``tournament_actions`` to
persist agent-supplied rationale per move (see ADR-004). No backfill;
existing rows keep NULL.

Revision ID: f1a2b3c4d5e6
Revises: e1b2c3d4f5a6
Create Date: 2026-04-16

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "e1b2c3d4f5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("tournament_actions") as batch_op:
        batch_op.add_column(sa.Column("reasoning", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("tournament_actions") as batch_op:
        batch_op.drop_column("reasoning")
