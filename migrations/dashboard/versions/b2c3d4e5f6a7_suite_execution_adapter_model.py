"""suite_execution_adapter_model

Add ``adapter`` and ``model`` columns to ``suite_executions`` so the
dashboard's run-history pages can tell runs of different models apart.

* ``adapter`` — the adapter type the suite ran under (e.g. "http"). Always
  known to the platform (the ``--adapter`` flag).
* ``model`` — an operator-supplied label (the ``--model`` flag). A black-box
  agent's model is opaque to ATP, so this is a tag, not an observed value.

Both nullable so existing rows remain valid with no backfill.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-06-11

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("suite_executions") as batch_op:
        batch_op.add_column(sa.Column("adapter", sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column("model", sa.String(length=100), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("suite_executions") as batch_op:
        batch_op.drop_column("model")
        batch_op.drop_column("adapter")
