"""AWS Bedrock Agents adapter for agent communication."""

from atp.adapters.bedrock.adapter import BedrockAdapter
from atp.adapters.bedrock.models import BedrockAdapterConfig

__all__ = [
    "BedrockAdapter",
    "BedrockAdapterConfig",
]
