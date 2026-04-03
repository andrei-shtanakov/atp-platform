"""Device Flow authentication routes (RFC 8628).

Provides endpoints for CLI login via GitHub Device Flow:
- POST /auth/device     — initiate device flow, get user_code
- POST /auth/device/poll — poll for authorization result
"""

from datetime import timedelta

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from atp.dashboard.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
)
from atp.dashboard.auth.device_flow import (
    DeviceCodeExpiredError,
    DeviceCodeNotFoundError,
    DeviceCodePendingError,
    DeviceFlowError,
    DeviceFlowManager,
)
from atp.dashboard.auth.sso.oidc import (
    SSOUserInfo,
    provision_sso_user,
)
from atp.dashboard.schemas import Token
from atp.dashboard.v2.config import get_config
from atp.dashboard.v2.dependencies import DBSession

router = APIRouter(prefix="/auth", tags=["auth"])

# Module-level manager, lazily initialized
_manager: DeviceFlowManager | None = None


def _get_manager() -> DeviceFlowManager:
    """Get or create the DeviceFlowManager singleton."""
    global _manager
    config = get_config()
    if not config.github_client_id or not config.github_client_secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "GitHub OAuth is not configured. "
                "Set ATP_GITHUB_CLIENT_ID and ATP_GITHUB_CLIENT_SECRET."
            ),
        )
    if _manager is None:
        _manager = DeviceFlowManager(
            client_id=config.github_client_id,
            client_secret=config.github_client_secret,
        )
    return _manager


class DeviceInitResponse(BaseModel):
    """Response from POST /auth/device."""

    device_code: str = Field(..., description="Code for polling")
    user_code: str = Field(..., description="Code user enters at verification_uri")
    verification_uri: str = Field(..., description="URL where user enters user_code")
    expires_in: int = Field(..., description="Seconds until device_code expires")
    interval: int = Field(..., description="Minimum polling interval in seconds")


class DevicePollRequest(BaseModel):
    """Request body for POST /auth/device/poll."""

    device_code: str = Field(..., description="Device code from initiation")


@router.post("/device", response_model=DeviceInitResponse)
async def initiate_device_flow() -> DeviceInitResponse:
    """Initiate GitHub Device Flow for CLI login."""
    manager = _get_manager()
    result = await manager.initiate()
    return DeviceInitResponse(
        device_code=str(result["device_code"]),
        user_code=str(result["user_code"]),
        verification_uri=str(result["verification_uri"]),
        expires_in=int(result["expires_in"]),
        interval=int(result["interval"]),
    )


@router.post("/device/poll")
async def poll_device_flow(body: DevicePollRequest, session: DBSession) -> Token:
    """Poll for device flow authorization result."""
    manager = _get_manager()
    try:
        github_user = await manager.poll(body.device_code)
    except DeviceCodePendingError:
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail="authorization_pending",
        )
    except DeviceCodeExpiredError:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="expired_token",
        )
    except DeviceCodeNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device code not found",
        )
    except DeviceFlowError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    email = github_user.get("email") or ""
    login = github_user.get("login") or ""
    name = str(github_user.get("name") or login)

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub account has no verified email",
        )

    user_info = SSOUserInfo(
        sub=f"github:{github_user.get('id', '')}",
        email=str(email),
        email_verified=True,
        name=name,
        preferred_username=str(login),
    )

    user = await provision_sso_user(session=session, user_info=user_info)
    await session.commit()

    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=access_token)
