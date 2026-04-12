"""Token management API endpoints."""

from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import func, select

from atp.dashboard.models import Agent
from atp.dashboard.schemas import APITokenCreate, APITokenCreated, APITokenResponse
from atp.dashboard.tokens import APIToken, generate_api_token, hash_token
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.dependencies import DBSession, RequiredUser
from atp.dashboard.v2.rate_limit import limiter

router = APIRouter(prefix="/v1/tokens", tags=["tokens"])


@router.post("", response_model=APITokenCreated, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_token(
    request: Request,
    session: DBSession,
    user: RequiredUser,
    body: APITokenCreate,
) -> APITokenCreated:
    """Create a new API token."""
    config = get_config()

    if body.agent_id is not None:
        # Agent-scoped token: verify ownership
        agent = await session.get(Agent, body.agent_id)
        if agent is None or agent.owner_id != user.id or agent.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't own this agent",
            )
        # Check per-agent token limit
        count_result = await session.execute(
            select(func.count(APIToken.id)).where(
                APIToken.agent_id == body.agent_id,
                APIToken.revoked_at.is_(None),
            )
        )
        if count_result.scalar_one() >= config.max_tokens_per_agent:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Token limit reached for this agent "
                    f"(max {config.max_tokens_per_agent})"
                ),
            )
    else:
        # User-level token: check limit
        count_result = await session.execute(
            select(func.count(APIToken.id)).where(
                APIToken.user_id == user.id,
                APIToken.agent_id.is_(None),
                APIToken.revoked_at.is_(None),
            )
        )
        if count_result.scalar_one() >= config.max_user_tokens:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Token limit reached (max {config.max_user_tokens})",
            )

    # Compute expiry: None = use config default, 0 = never
    expires_in_days = body.expires_in_days
    if expires_in_days is None:
        expires_in_days = config.default_token_days

    expires_at = None
    if expires_in_days > 0:
        if config.max_token_days > 0 and expires_in_days > config.max_token_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Token expiry exceeds maximum ({config.max_token_days} days)",
            )
        expires_at = datetime.now() + timedelta(days=expires_in_days)

    raw_token = generate_api_token(agent_scoped=body.agent_id is not None)
    db_token = APIToken(
        user_id=user.id,
        agent_id=body.agent_id,
        name=body.name,
        token_prefix=raw_token[:12],
        token_hash=hash_token(raw_token),
        expires_at=expires_at,
    )
    session.add(db_token)
    await session.flush()
    await session.refresh(db_token)

    return APITokenCreated(
        id=db_token.id,
        name=db_token.name,
        token_prefix=db_token.token_prefix,
        agent_id=db_token.agent_id,
        scopes=db_token.scopes,
        expires_at=db_token.expires_at,
        last_used_at=db_token.last_used_at,
        revoked_at=db_token.revoked_at,
        created_at=db_token.created_at,
        token=raw_token,
    )


@router.get("", response_model=list[APITokenResponse])
async def list_tokens(
    session: DBSession,
    user: RequiredUser,
) -> list[APITokenResponse]:
    """List all tokens for the current user."""
    result = await session.execute(
        select(APIToken)
        .where(APIToken.user_id == user.id)
        .order_by(APIToken.created_at.desc())
    )
    tokens = result.scalars().all()
    return [APITokenResponse.model_validate(t) for t in tokens]


@router.delete("/{token_id}", response_model=APITokenResponse)
async def revoke_token(
    session: DBSession,
    user: RequiredUser,
    token_id: int,
) -> APITokenResponse:
    """Revoke an API token."""
    token = await session.get(APIToken, token_id)
    if token is None or (token.user_id != user.id and not user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Token not found",
        )
    if token.revoked_at is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Token already revoked",
        )
    token.revoked_at = datetime.now()
    await session.flush()
    await session.refresh(token)
    return APITokenResponse.model_validate(token)
