"""APIToken.agent_purpose (LABS-TSA PR-3)

Adds a nullable snapshot of ``agents.purpose`` onto ``api_tokens`` so the
middleware auth hot path can read the purpose without a join. Legacy
tokens (issued before this revision) keep ``agent_purpose = NULL``; the
middleware has a lazy fallback for those.

Revision ID: 52987a83afb7
Revises: 853688412c5b
Create Date: 2026-04-23 19:38:15.638444
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "52987a83afb7"
down_revision: str | Sequence[str] | None = "853688412c5b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("api_tokens", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("agent_purpose", sa.String(length=20), nullable=True)
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("api_tokens", schema=None) as batch_op:
        batch_op.drop_column("agent_purpose")
