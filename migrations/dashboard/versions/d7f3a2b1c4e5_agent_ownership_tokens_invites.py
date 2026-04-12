"""agent_ownership_tokens_invites

Revision ID: d7f3a2b1c4e5
Revises: c60b45e516be
Create Date: 2026-04-12

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d7f3a2b1c4e5"
down_revision: str | Sequence[str] | None = "c60b45e516be"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- New tables ---

    op.create_table(
        "api_tokens",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "tenant_id",
            sa.String(100),
            nullable=False,
            server_default="default",
        ),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("token_prefix", sa.String(12), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("scopes", sa.JSON(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_api_token_hash"),
    )
    op.create_index("idx_api_token_user", "api_tokens", ["user_id"])
    op.create_index("idx_api_token_hash", "api_tokens", ["token_hash"])

    op.create_table(
        "invites",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("code", sa.String(40), nullable=False, unique=True),
        sa.Column("created_by_id", sa.Integer(), nullable=False),
        sa.Column("used_by_id", sa.Integer(), nullable=True),
        sa.Column("used_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("max_uses", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("use_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["used_by_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_invite_code", "invites", ["code"])

    # --- Columns on existing tables (use batch mode for SQLite compatibility) ---

    with op.batch_alter_table("agents") as batch_op:
        batch_op.add_column(
            sa.Column("owner_id", sa.Integer(), nullable=True),
        )
        batch_op.add_column(
            sa.Column(
                "version", sa.String(50), nullable=False, server_default="latest"
            ),
        )
        batch_op.add_column(
            sa.Column("deleted_at", sa.DateTime(), nullable=True),
        )
        batch_op.create_foreign_key("fk_agent_owner_id", "users", ["owner_id"], ["id"])
        batch_op.create_index("idx_agent_owner", ["owner_id"])
        # Drop old constraint before creating the new one —
        # (tenant_id, name) would block multiple versions per agent.
        batch_op.drop_constraint("uq_agent_tenant_name", type_="unique")
        batch_op.create_unique_constraint(
            "uq_agent_tenant_owner_name_version",
            ["tenant_id", "owner_id", "name", "version"],
        )

    with op.batch_alter_table("tournament_participants") as batch_op:
        batch_op.add_column(
            sa.Column("agent_id", sa.Integer(), nullable=True),
        )
        batch_op.create_foreign_key(
            "fk_participant_agent_id", "agents", ["agent_id"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("tournament_participants") as batch_op:
        batch_op.drop_constraint("fk_participant_agent_id", type_="foreignkey")
        batch_op.drop_column("agent_id")

    with op.batch_alter_table("agents") as batch_op:
        batch_op.drop_constraint("uq_agent_tenant_owner_name_version", type_="unique")
        # Restore old constraint that was replaced in upgrade
        batch_op.create_unique_constraint("uq_agent_tenant_name", ["tenant_id", "name"])
        batch_op.drop_constraint("fk_agent_owner_id", type_="foreignkey")
        batch_op.drop_index("idx_agent_owner")
        batch_op.drop_column("deleted_at")
        batch_op.drop_column("version")
        batch_op.drop_column("owner_id")

    op.drop_table("invites")
    op.drop_index("idx_api_token_hash", table_name="api_tokens")
    op.drop_index("idx_api_token_user", table_name="api_tokens")
    op.drop_table("api_tokens")
