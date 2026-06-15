"""SP-4 task_type + language columns

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-06-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TEST_COLS = [
    ("task_type", sa.String(length=50)),
    ("language", sa.String(length=50)),
]
_SUITE_COLS = [
    ("language", sa.String(length=50)),
]


def _columns(insp: sa.Inspector, table: str) -> set[str]:
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    # Idempotent: `_add_missing_columns()` runs at app startup and may already
    # have added these columns on a live SQLite (boot-before-migrate). Skip any
    # that already exist so the revision still applies + stamps cleanly.
    insp = sa.inspect(op.get_bind())
    have = _columns(insp, "test_executions")
    for name, type_ in _TEST_COLS:
        if name not in have:
            op.add_column("test_executions", sa.Column(name, type_, nullable=True))
    have = _columns(insp, "suite_executions")
    for name, type_ in _SUITE_COLS:
        if name not in have:
            op.add_column("suite_executions", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    insp = sa.inspect(op.get_bind())
    have = _columns(insp, "suite_executions")
    for name, _ in _SUITE_COLS:
        if name in have:
            op.drop_column("suite_executions", name)
    have = _columns(insp, "test_executions")
    for name, _ in _TEST_COLS:
        if name in have:
            op.drop_column("test_executions", name)
