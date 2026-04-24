"""action_tier2_fields

Add nullable tier-2 columns to ``tournament_actions`` so the dashboard's
DEBUG · OBSERVABILITY drawer panel can render real values when capture
paths land. All columns are nullable (``retry_count`` has a server
default of 0) so existing rows remain valid with no backfill.

Grouped by capture owner:

* runner-managed — ``retry_count``, ``validation_error``, ``decide_ms``
  populated by ``submit_action`` on validation retries and the round
  dispatcher on successful submit.
* agent-self-reported — ``model_id``, ``tokens_in``, ``tokens_out``,
  ``cost_usd`` populated from the optional ``telemetry`` sub-object on
  ``ElFarolAction`` / ``PDAction`` / etc.
* trace linkage — ``trace_id``, ``span_id`` extracted from the W3C
  ``traceparent`` header on the MCP request; enables deep-link to
  Langfuse from the drawer.

Revision ID: a1b2c3d4e5f6
Revises: f1a2b3c4d5e6
Create Date: 2026-04-24

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("tournament_actions") as batch_op:
        # runner-managed
        batch_op.add_column(
            sa.Column(
                "retry_count",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(sa.Column("validation_error", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("decide_ms", sa.Integer(), nullable=True))
        # agent-self-reported LLM telemetry
        batch_op.add_column(sa.Column("model_id", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("tokens_in", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("tokens_out", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("cost_usd", sa.Float(), nullable=True))
        # W3C traceparent linkage
        batch_op.add_column(sa.Column("trace_id", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("span_id", sa.String(length=32), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("tournament_actions") as batch_op:
        batch_op.drop_column("span_id")
        batch_op.drop_column("trace_id")
        batch_op.drop_column("cost_usd")
        batch_op.drop_column("tokens_out")
        batch_op.drop_column("tokens_in")
        batch_op.drop_column("model_id")
        batch_op.drop_column("decide_ms")
        batch_op.drop_column("validation_error")
        batch_op.drop_column("retry_count")
