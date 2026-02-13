"""Azure OpenAI adapter for agent communication."""

from atp.adapters.azure_openai.adapter import (
    AzureOpenAIAdapter,
)
from atp.adapters.azure_openai.models import (
    AzureOpenAIAdapterConfig,
)

__all__ = [
    "AzureOpenAIAdapter",
    "AzureOpenAIAdapterConfig",
]
