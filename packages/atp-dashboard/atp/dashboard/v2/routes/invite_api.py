"""Invite management API endpoints (admin only)."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select

from atp.dashboard.auth import require_user_level_token
from atp.dashboard.schemas import InviteCreate, InviteResponse
from atp.dashboard.tokens import Invite, generate_invite_code
from atp.dashboard.v2.dependencies import AdminUser, DBSession
from atp.dashboard.v2.rate_limit import limiter

router = APIRouter(
    prefix="/v1/invites",
    tags=["invites"],
    dependencies=[Depends(require_user_level_token)],
)


@router.post("", response_model=InviteResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_invite(
    request: Request,
    session: DBSession,
    admin: AdminUser,
    body: InviteCreate = InviteCreate(),
) -> InviteResponse:
    """Create a new invite code (admin only)."""
    expires_at = None
    if body.expires_in_days is not None:
        expires_at = datetime.now() + timedelta(days=body.expires_in_days)

    invite = Invite(
        code=generate_invite_code(),
        created_by_id=admin.id,
        expires_at=expires_at,
    )
    session.add(invite)
    await session.flush()
    await session.refresh(invite)
    return InviteResponse.model_validate(invite)


@router.get("", response_model=list[InviteResponse])
async def list_invites(
    session: DBSession,
    admin: AdminUser,
) -> list[InviteResponse]:
    """List all invite codes (admin only)."""
    result = await session.execute(select(Invite).order_by(Invite.created_at.desc()))
    return [InviteResponse.model_validate(i) for i in result.scalars().all()]


@router.delete("/{invite_id}", response_model=InviteResponse)
async def deactivate_invite(
    session: DBSession,
    admin: AdminUser,
    invite_id: int,
) -> InviteResponse:
    """Deactivate an invite code (admin only)."""
    invite = await session.get(Invite, invite_id)
    if invite is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found"
        )
    invite.max_uses = invite.use_count
    await session.flush()
    await session.refresh(invite)
    return InviteResponse.model_validate(invite)
