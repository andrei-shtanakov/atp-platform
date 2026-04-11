"""tournament_slice_columns

Revision ID: 028d8a9fdc46
Revises: c8d5f2a91234
Create Date: 2026-04-11 05:48:42.192462

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "028d8a9fdc46"
down_revision: str | Sequence[str] | None = "c8d5f2a91234"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tournaments",
        sa.Column("num_players", sa.Integer(), nullable=False, server_default="2"),
    )
    op.add_column(
        "tournaments",
        sa.Column("total_rounds", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "tournaments",
        sa.Column(
            "round_deadline_s",
            sa.Integer(),
            nullable=False,
            server_default="30",
        ),
    )
    op.add_column(
        "tournament_actions",
        sa.Column("payoff", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tournament_actions", "payoff")
    op.drop_column("tournaments", "round_deadline_s")
    op.drop_column("tournaments", "total_rounds")
    op.drop_column("tournaments", "num_players")
