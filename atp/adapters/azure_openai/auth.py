"""Authentication utilities for the Azure OpenAI adapter."""

from typing import Any

from atp.adapters.exceptions import (
    AdapterConnectionError,
    AdapterError,
)

from .models import AzureOpenAIAdapterConfig


def get_openai_client(
    config: AzureOpenAIAdapterConfig,
    adapter_type: str = "azure_openai",
) -> Any:
    """
    Create an Azure OpenAI client.

    Args:
        config: Azure OpenAI adapter configuration.
        adapter_type: Adapter type for error messages.

    Returns:
        AzureOpenAI client instance.

    Raises:
        AdapterError: If openai is not installed.
        AdapterConnectionError: If client creation fails.
    """
    try:
        from openai import AzureOpenAI
    except ImportError as e:
        raise AdapterError(
            "openai is required for Azure OpenAI adapter. "
            "Install it with: uv add openai",
            adapter_type=adapter_type,
        ) from e

    try:
        client_kwargs: dict[str, Any] = {
            "azure_endpoint": config.endpoint,
            "api_version": config.api_version,
        }

        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        elif config.use_azure_ad:
            token_provider = get_azure_ad_token_provider(config, adapter_type)
            client_kwargs["azure_ad_token_provider"] = token_provider
        else:
            raise AdapterError(
                "No authentication method configured",
                adapter_type=adapter_type,
            )

        return AzureOpenAI(**client_kwargs)

    except Exception as e:
        if "openai" in str(type(e).__module__).lower():
            raise AdapterConnectionError(
                f"Failed to create Azure OpenAI client: {e}",
                adapter_type=adapter_type,
                cause=e,
            ) from e
        raise


def get_azure_ad_token_provider(
    config: AzureOpenAIAdapterConfig,
    adapter_type: str = "azure_openai",
) -> Any:
    """
    Get an Azure AD token provider function.

    Args:
        config: Azure OpenAI adapter configuration.
        adapter_type: Adapter type for error messages.

    Returns:
        Callable that returns Azure AD tokens.

    Raises:
        AdapterError: If azure-identity is not installed.
        AdapterConnectionError: If credential setup fails.
    """
    try:
        from azure.identity import (  # pyrefly: ignore[missing-import]
            ClientSecretCredential,
            DefaultAzureCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )
    except ImportError as e:
        raise AdapterError(
            "azure-identity is required for Azure AD "
            "authentication. "
            "Install it with: uv add azure-identity",
            adapter_type=adapter_type,
        ) from e

    try:
        # Service principal authentication
        if config.client_id and config.client_secret:
            if not config.tenant_id:
                raise AdapterError(
                    "tenant_id is required for service principal authentication",
                    adapter_type=adapter_type,
                )
            credential = ClientSecretCredential(
                tenant_id=config.tenant_id,
                client_id=config.client_id,
                client_secret=config.client_secret,
            )
        # Managed identity authentication
        elif config.managed_identity_client_id:
            credential = ManagedIdentityCredential(
                client_id=(config.managed_identity_client_id)
            )
        # Default Azure credential chain
        else:
            credential = DefaultAzureCredential()

        # Create token provider for Azure OpenAI
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )

        return token_provider

    except Exception as e:
        raise AdapterConnectionError(
            f"Failed to setup Azure AD authentication: {e}",
            adapter_type=adapter_type,
            cause=e,
        ) from e
