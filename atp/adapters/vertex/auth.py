"""Authentication utilities for the Google Vertex AI adapter."""

from typing import Any

from atp.adapters.exceptions import (
    AdapterConnectionError,
    AdapterError,
)

from .models import VertexAdapterConfig


def get_vertexai_module() -> Any:
    """
    Import and return the vertexai module.

    Returns:
        The vertexai module.

    Raises:
        AdapterError: If google-cloud-aiplatform is not installed.
    """
    try:
        import vertexai  # pyrefly: ignore[missing-import]

        return vertexai
    except ImportError as e:
        raise AdapterError(
            "google-cloud-aiplatform is required for "
            "Vertex AI adapter. Install it with: "
            "uv add google-cloud-aiplatform",
            adapter_type="vertex",
        ) from e


def initialize_vertexai(
    config: VertexAdapterConfig,
    adapter_type: str = "vertex",
) -> None:
    """
    Initialize the Vertex AI SDK.

    Args:
        config: Vertex AI adapter configuration.
        adapter_type: Adapter type for error messages.

    Raises:
        AdapterConnectionError: If initialization fails.
    """
    vertexai = get_vertexai_module()

    try:
        init_kwargs: dict[str, Any] = {
            "project": config.project_id,
            "location": config.location,
        }

        # Handle credentials
        if config.credentials_path:
            from google.oauth2 import (  # pyrefly: ignore[missing-import]
                service_account,
            )

            credentials = service_account.Credentials.from_service_account_file(
                config.credentials_path
            )
            init_kwargs["credentials"] = credentials
        elif config.service_account_email:
            import google.auth  # pyrefly: ignore[missing-import]
            from google.auth import (  # pyrefly: ignore[missing-import]
                impersonated_credentials,
            )

            source_credentials, _ = google.auth.default()
            credentials = impersonated_credentials.Credentials(
                source_credentials=source_credentials,
                target_principal=(config.service_account_email),
                target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            init_kwargs["credentials"] = credentials

        vertexai.init(**init_kwargs)

    except Exception as e:
        raise AdapterConnectionError(
            f"Failed to initialize Vertex AI: {e}",
            adapter_type=adapter_type,
            cause=e,
        ) from e
