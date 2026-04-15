"""agent_ownerless_partial_unique

Adds a partial unique index on (tenant_id, name, version) WHERE
owner_id IS NULL. The existing uq_agent_tenant_owner_name_version
constraint does not enforce uniqueness for ownerless rows because
SQL treats NULL as distinct in unique constraints. This index closes
that gap for the legacy ownerless agent path (CLI upload,
routes/agents.py, AgentService.create_agent) — see LABS-15 / LABS-54.

If this migration fails with an IntegrityError, the prod DB already
has duplicate ownerless rows. Resolve them before re-running.

Revision ID: e1b2c3d4f5a6
Revises: d7f3a2b1c4e5
Create Date: 2026-04-15

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1b2c3d4f5a6"
down_revision: str | Sequence[str] | None = "d7f3a2b1c4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_agent_ownerless_tenant_name_version",
        "agents",
        ["tenant_id", "name", "version"],
        unique=True,
        sqlite_where=sa.text("owner_id IS NULL"),
        postgresql_where=sa.text("owner_id IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_agent_ownerless_tenant_name_version",
        table_name="agents",
    )
