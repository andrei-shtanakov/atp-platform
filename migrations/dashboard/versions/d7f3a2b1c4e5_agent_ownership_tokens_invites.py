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

    # --- Columns on existing tables ---

    op.add_column(
        "agents",
        sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
    )
    op.add_column(
        "agents",
        sa.Column("version", sa.String(50), nullable=False, server_default="latest"),
    )
    op.add_column(
        "agents",
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_agent_owner", "agents", ["owner_id"])

    op.create_unique_constraint(
        "uq_agent_tenant_owner_name_version",
        "agents",
        ["tenant_id", "owner_id", "name", "version"],
    )

    op.add_column(
        "tournament_participants",
        sa.Column("agent_id", sa.Integer(), sa.ForeignKey("agents.id"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tournament_participants", "agent_id")
    op.drop_constraint("uq_agent_tenant_owner_name_version", "agents", type_="unique")
    op.drop_index("idx_agent_owner", table_name="agents")
    op.drop_column("agents", "deleted_at")
    op.drop_column("agents", "version")
    op.drop_column("agents", "owner_id")
    op.drop_table("invites")
    op.drop_index("idx_api_token_hash", table_name="api_tokens")
    op.drop_index("idx_api_token_user", table_name="api_tokens")
    op.drop_table("api_tokens")
