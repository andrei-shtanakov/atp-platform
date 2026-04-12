"""API token and invite ORM models + token generation helpers."""

import hashlib
import secrets
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from atp.dashboard.models import DEFAULT_TENANT_ID, Base


def generate_api_token(*, agent_scoped: bool) -> str:
    """Generate a new API token with appropriate prefix."""
    prefix = "atp_a_" if agent_scoped else "atp_u_"
    return prefix + secrets.token_hex(16)


def hash_token(token: str) -> str:
    """Compute SHA-256 hash of a token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def generate_invite_code() -> str:
    """Generate a new invite code."""
    return "atp_inv_" + secrets.token_hex(8)


class APIToken(Base):
    """API token for programmatic access."""

    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default=DEFAULT_TENANT_ID
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    agent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("agents.id"), nullable=True
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    token_prefix: Mapped[str] = mapped_column(String(12), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    scopes: Mapped[list[str]] = mapped_column(JSON, default=lambda: ["*"])
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    __table_args__ = (
        UniqueConstraint("token_hash", name="uq_api_token_hash"),
        Index("idx_api_token_user", "user_id"),
        Index("idx_api_token_hash", "token_hash"),
    )

    def __repr__(self) -> str:
        return (
            f"APIToken(id={self.id}, name={self.name!r}, prefix={self.token_prefix!r})"
        )


class Invite(Base):
    """Invite code for registration."""

    __tablename__ = "invites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    created_by_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )

    used_by_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    max_uses: Mapped[int] = mapped_column(Integer, default=1)
    use_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    __table_args__ = (Index("idx_invite_code", "code"),)

    def __repr__(self) -> str:
        return f"Invite(id={self.id}, code={self.code[:16]}...)"
