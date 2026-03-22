"""Authentication utilities for the AWS Bedrock adapter."""

from typing import Any

from atp.adapters.exceptions import AdapterConnectionError, AdapterError

from .models import BedrockAdapterConfig


def create_boto3_client(
    config: BedrockAdapterConfig,
    adapter_type: str = "bedrock",
) -> Any:
    """
    Create a boto3 client for Bedrock Agent Runtime.

    Args:
        config: Bedrock adapter configuration.
        adapter_type: Adapter type identifier for error messages.

    Returns:
        boto3 client for bedrock-agent-runtime.

    Raises:
        AdapterError: If boto3 is not installed.
        AdapterConnectionError: If client creation fails.
    """
    try:
        import boto3  # pyrefly: ignore[missing-import]
    except ImportError as e:
        raise AdapterError(
            "boto3 is required for Bedrock adapter. Install it with: uv add boto3",
            adapter_type=adapter_type,
        ) from e

    try:
        # Build session kwargs
        session_kwargs: dict[str, Any] = {
            "region_name": config.region,
        }

        if config.profile:
            session_kwargs["profile_name"] = config.profile

        # Create session
        session = boto3.Session(**session_kwargs)

        # Build client kwargs
        client_kwargs: dict[str, Any] = {}

        if config.access_key_id and config.secret_access_key:
            client_kwargs["aws_access_key_id"] = config.access_key_id
            client_kwargs["aws_secret_access_key"] = config.secret_access_key
            if config.session_token:
                client_kwargs["aws_session_token"] = config.session_token

        if config.endpoint_url:
            client_kwargs["endpoint_url"] = config.endpoint_url

        return session.client("bedrock-agent-runtime", **client_kwargs)

    except Exception as e:
        raise AdapterConnectionError(
            f"Failed to create Bedrock client: {e}",
            adapter_type=adapter_type,
            cause=e,
        ) from e
