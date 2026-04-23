"""el farol game_result columns

Revision ID: 7a1c3d9e4b02
Revises: b8c9d0e1f2a3
Create Date: 2026-04-21 12:00:00.000000

Adds El Farol dashboard data-model columns to ``game_results`` so the
dashboard can query per-day typed data without deserialising whole
episode blobs. All columns are nullable so legacy rows continue to be
readable via the JSON fallback path.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7a1c3d9e4b02"
down_revision: str | Sequence[str] | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: add El Farol columns to game_results."""
    with op.batch_alter_table("game_results", schema=None) as batch_op:
        batch_op.add_column(sa.Column("match_id", sa.String(length=255), nullable=True))
        batch_op.add_column(
            sa.Column("game_version", sa.String(length=50), nullable=True)
        )
        batch_op.add_column(sa.Column("num_days", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("num_slots", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("max_intervals", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("max_total_slots", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("capacity_ratio", sa.Float(), nullable=True))
        batch_op.add_column(
            sa.Column("capacity_threshold", sa.Integer(), nullable=True)
        )
        batch_op.add_column(sa.Column("actions_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("day_aggregates_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("round_payoffs_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("agents_json", sa.JSON(), nullable=True))
        batch_op.create_index("idx_game_result_match", ["match_id"], unique=False)
        batch_op.create_index(
            "idx_game_result_game_completed",
            ["game_name", "completed_at"],
            unique=False,
        )


def downgrade() -> None:
    """Downgrade schema: drop the El Farol columns."""
    with op.batch_alter_table("game_results", schema=None) as batch_op:
        batch_op.drop_index("idx_game_result_game_completed")
        batch_op.drop_index("idx_game_result_match")
        batch_op.drop_column("agents_json")
        batch_op.drop_column("round_payoffs_json")
        batch_op.drop_column("day_aggregates_json")
        batch_op.drop_column("actions_json")
        batch_op.drop_column("capacity_threshold")
        batch_op.drop_column("capacity_ratio")
        batch_op.drop_column("max_total_slots")
        batch_op.drop_column("max_intervals")
        batch_op.drop_column("num_slots")
        batch_op.drop_column("num_days")
        batch_op.drop_column("game_version")
        batch_op.drop_column("match_id")
