"""tournament agent sandbox (LABS-TSA PR-1)

- Agent.purpose (VARCHAR(20) NOT NULL DEFAULT 'benchmark' CHECK IN ('benchmark','tournament'))
- Participant.user_id made nullable; Participant.builtin_strategy (VARCHAR(64))
  with agent-xor-builtin CHECK
- GameResult.tournament_id (nullable FK) + UNIQUE partial index
- Supporting indexes

Revision ID: 853688412c5b
Revises: 7a1c3d9e4b02
Create Date: 2026-04-23 17:33:52.371237
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "853688412c5b"
down_revision: str | Sequence[str] | None = "7a1c3d9e4b02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # agents.purpose + CHECK + index
    with op.batch_alter_table("agents", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "purpose",
                sa.String(length=20),
                nullable=False,
                server_default="benchmark",
            )
        )
        batch_op.create_check_constraint(
            "ck_agents_purpose",
            "purpose IN ('benchmark','tournament')",
        )
        batch_op.create_index(
            "idx_agents_owner_purpose",
            ["owner_id", "purpose"],
            unique=False,
        )

    # tournament_participants: user_id → nullable; builtin_strategy;
    # builtin-only partial index. The agent-xor-builtin CHECK is
    # deferred to the PR-4 migration, after TournamentService is
    # updated to populate agent_id on real-agent joins — otherwise
    # today's join flow (which inserts Participant rows without
    # agent_id) would violate the invariant.
    with op.batch_alter_table("tournament_participants", schema=None) as batch_op:
        batch_op.alter_column("user_id", existing_type=sa.Integer(), nullable=True)
        batch_op.add_column(
            sa.Column("builtin_strategy", sa.String(length=64), nullable=True)
        )
        batch_op.create_index(
            "idx_participants_builtin",
            ["tournament_id", "builtin_strategy"],
            unique=False,
            sqlite_where=sa.text("builtin_strategy IS NOT NULL"),
            postgresql_where=sa.text("builtin_strategy IS NOT NULL"),
        )

    # game_results.tournament_id + FK + UNIQUE partial index
    with op.batch_alter_table("game_results", schema=None) as batch_op:
        batch_op.add_column(sa.Column("tournament_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_game_results_tournament",
            "tournaments",
            ["tournament_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "idx_game_results_tournament",
            ["tournament_id"],
            unique=False,
        )
        batch_op.create_index(
            "uq_game_results_tournament_id",
            ["tournament_id"],
            unique=True,
            sqlite_where=sa.text("tournament_id IS NOT NULL"),
            postgresql_where=sa.text("tournament_id IS NOT NULL"),
        )


def downgrade() -> None:
    with op.batch_alter_table("game_results", schema=None) as batch_op:
        batch_op.drop_index("uq_game_results_tournament_id")
        batch_op.drop_index("idx_game_results_tournament")
        batch_op.drop_constraint("fk_game_results_tournament", type_="foreignkey")
        batch_op.drop_column("tournament_id")

    with op.batch_alter_table("tournament_participants", schema=None) as batch_op:
        batch_op.drop_index("idx_participants_builtin")
        batch_op.drop_column("builtin_strategy")
        batch_op.alter_column("user_id", existing_type=sa.Integer(), nullable=False)

    with op.batch_alter_table("agents", schema=None) as batch_op:
        batch_op.drop_index("idx_agents_owner_purpose")
        batch_op.drop_constraint("ck_agents_purpose", type_="check")
        batch_op.drop_column("purpose")
