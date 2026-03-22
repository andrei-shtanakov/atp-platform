"""Cost data models — pure dataclasses with no DB dependencies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CostEvent:
    """Event representing an LLM operation cost.

    Attributes:
        timestamp: When the operation occurred.
        provider: LLM provider (anthropic, openai, google, azure, bedrock).
        model: Model name (claude-3-sonnet, gpt-4, etc.).
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        test_id: Optional test ID for association.
        suite_id: Optional suite ID for association.
        agent_name: Optional agent name for association.
        metadata: Optional additional metadata.
    """

    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    test_id: str | None = None
    suite_id: str | None = None
    agent_name: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ModelPricing:
    """Pricing configuration for a specific model.

    Prices are in USD per 1,000 tokens.
    """

    input_per_1k: Decimal
    output_per_1k: Decimal
    name: str = ""

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Calculate cost for given token counts."""
        input_cost = (Decimal(input_tokens) / Decimal(1000)) * self.input_per_1k
        output_cost = (Decimal(output_tokens) / Decimal(1000)) * self.output_per_1k
        return input_cost + output_cost


@dataclass
class PricingConfig:
    """Pricing configuration for all models.

    Contains pricing for built-in models and custom configurations.
    """

    models: dict[str, ModelPricing] = field(default_factory=dict)
    provider_defaults: dict[str, ModelPricing] = field(default_factory=dict)

    @classmethod
    def default(cls) -> PricingConfig:
        """Create default pricing configuration with all major providers.

        Pricing is based on public pricing as of early 2026.
        Prices are in USD per 1,000 tokens.
        """
        models = {
            # Anthropic Claude models
            "claude-opus-4-5-20251101": ModelPricing(
                input_per_1k=Decimal("0.015"),
                output_per_1k=Decimal("0.075"),
                name="Claude Opus 4.5",
            ),
            "claude-sonnet-4-20250514": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Claude Sonnet 4",
            ),
            "claude-3-5-sonnet-20241022": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Claude 3.5 Sonnet",
            ),
            "claude-3-5-haiku-20241022": ModelPricing(
                input_per_1k=Decimal("0.0008"),
                output_per_1k=Decimal("0.004"),
                name="Claude 3.5 Haiku",
            ),
            "claude-3-opus-20240229": ModelPricing(
                input_per_1k=Decimal("0.015"),
                output_per_1k=Decimal("0.075"),
                name="Claude 3 Opus",
            ),
            "claude-3-sonnet-20240229": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Claude 3 Sonnet",
            ),
            "claude-3-haiku-20240307": ModelPricing(
                input_per_1k=Decimal("0.00025"),
                output_per_1k=Decimal("0.00125"),
                name="Claude 3 Haiku",
            ),
            # OpenAI GPT models
            "gpt-4o": ModelPricing(
                input_per_1k=Decimal("0.0025"),
                output_per_1k=Decimal("0.01"),
                name="GPT-4o",
            ),
            "gpt-4o-mini": ModelPricing(
                input_per_1k=Decimal("0.00015"),
                output_per_1k=Decimal("0.0006"),
                name="GPT-4o Mini",
            ),
            "gpt-4-turbo": ModelPricing(
                input_per_1k=Decimal("0.01"),
                output_per_1k=Decimal("0.03"),
                name="GPT-4 Turbo",
            ),
            "gpt-4": ModelPricing(
                input_per_1k=Decimal("0.03"),
                output_per_1k=Decimal("0.06"),
                name="GPT-4",
            ),
            "gpt-3.5-turbo": ModelPricing(
                input_per_1k=Decimal("0.0005"),
                output_per_1k=Decimal("0.0015"),
                name="GPT-3.5 Turbo",
            ),
            "o1": ModelPricing(
                input_per_1k=Decimal("0.015"),
                output_per_1k=Decimal("0.06"),
                name="O1",
            ),
            "o1-mini": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.012"),
                name="O1 Mini",
            ),
            "o1-pro": ModelPricing(
                input_per_1k=Decimal("0.15"),
                output_per_1k=Decimal("0.6"),
                name="O1 Pro",
            ),
            # Google Gemini models
            "gemini-1.5-pro": ModelPricing(
                input_per_1k=Decimal("0.00125"),
                output_per_1k=Decimal("0.005"),
                name="Gemini 1.5 Pro",
            ),
            "gemini-1.5-flash": ModelPricing(
                input_per_1k=Decimal("0.000075"),
                output_per_1k=Decimal("0.0003"),
                name="Gemini 1.5 Flash",
            ),
            "gemini-2.0-flash": ModelPricing(
                input_per_1k=Decimal("0.0001"),
                output_per_1k=Decimal("0.0004"),
                name="Gemini 2.0 Flash",
            ),
            "gemini-2.0-pro": ModelPricing(
                input_per_1k=Decimal("0.00125"),
                output_per_1k=Decimal("0.005"),
                name="Gemini 2.0 Pro",
            ),
            # AWS Bedrock - Claude models
            "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Claude 3.5 Sonnet (Bedrock)",
            ),
            "anthropic.claude-3-sonnet-20240229-v1:0": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Claude 3 Sonnet (Bedrock)",
            ),
            "anthropic.claude-3-haiku-20240307-v1:0": ModelPricing(
                input_per_1k=Decimal("0.00025"),
                output_per_1k=Decimal("0.00125"),
                name="Claude 3 Haiku (Bedrock)",
            ),
            "anthropic.claude-3-opus-20240229-v1:0": ModelPricing(
                input_per_1k=Decimal("0.015"),
                output_per_1k=Decimal("0.075"),
                name="Claude 3 Opus (Bedrock)",
            ),
            # AWS Bedrock - Titan models
            "amazon.titan-text-express-v1": ModelPricing(
                input_per_1k=Decimal("0.0002"),
                output_per_1k=Decimal("0.0006"),
                name="Titan Text Express",
            ),
            "amazon.titan-text-lite-v1": ModelPricing(
                input_per_1k=Decimal("0.00015"),
                output_per_1k=Decimal("0.0002"),
                name="Titan Text Lite",
            ),
            # Azure OpenAI
            "azure/gpt-4o": ModelPricing(
                input_per_1k=Decimal("0.0025"),
                output_per_1k=Decimal("0.01"),
                name="GPT-4o (Azure)",
            ),
            "azure/gpt-4o-mini": ModelPricing(
                input_per_1k=Decimal("0.00015"),
                output_per_1k=Decimal("0.0006"),
                name="GPT-4o Mini (Azure)",
            ),
            "azure/gpt-4-turbo": ModelPricing(
                input_per_1k=Decimal("0.01"),
                output_per_1k=Decimal("0.03"),
                name="GPT-4 Turbo (Azure)",
            ),
            "azure/gpt-4": ModelPricing(
                input_per_1k=Decimal("0.03"),
                output_per_1k=Decimal("0.06"),
                name="GPT-4 (Azure)",
            ),
        }

        # Provider defaults for unknown models
        provider_defaults = {
            "anthropic": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Anthropic Default",
            ),
            "openai": ModelPricing(
                input_per_1k=Decimal("0.0025"),
                output_per_1k=Decimal("0.01"),
                name="OpenAI Default",
            ),
            "google": ModelPricing(
                input_per_1k=Decimal("0.00125"),
                output_per_1k=Decimal("0.005"),
                name="Google Default",
            ),
            "azure": ModelPricing(
                input_per_1k=Decimal("0.0025"),
                output_per_1k=Decimal("0.01"),
                name="Azure Default",
            ),
            "bedrock": ModelPricing(
                input_per_1k=Decimal("0.003"),
                output_per_1k=Decimal("0.015"),
                name="Bedrock Default",
            ),
        }

        return cls(models=models, provider_defaults=provider_defaults)

    @classmethod
    def from_yaml(cls, path: Path) -> PricingConfig:
        """Load pricing configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Pricing configuration must be a dictionary")

        config = cls.default()

        if "models" in data:
            for model_name, pricing_data in data["models"].items():
                if not isinstance(pricing_data, dict):
                    raise ValueError(f"Invalid pricing for model {model_name}")
                config.models[model_name] = ModelPricing(
                    input_per_1k=Decimal(str(pricing_data.get("input_per_1k", 0))),
                    output_per_1k=Decimal(str(pricing_data.get("output_per_1k", 0))),
                    name=str(pricing_data.get("name", model_name)),
                )

        if "provider_defaults" in data:
            for provider, pricing_data in data["provider_defaults"].items():
                if not isinstance(pricing_data, dict):
                    raise ValueError(f"Invalid default pricing for provider {provider}")
                config.provider_defaults[provider] = ModelPricing(
                    input_per_1k=Decimal(str(pricing_data.get("input_per_1k", 0))),
                    output_per_1k=Decimal(str(pricing_data.get("output_per_1k", 0))),
                    name=pricing_data.get("name", f"{provider} Default"),
                )

        return config

    def calculate(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        """Calculate cost for an LLM operation."""
        if model in self.models:
            return self.models[model].calculate_cost(input_tokens, output_tokens)

        azure_key = f"azure/{model}"
        if azure_key in self.models:
            return self.models[azure_key].calculate_cost(input_tokens, output_tokens)

        provider_lower = provider.lower()
        if provider_lower in self.provider_defaults:
            logger.debug(
                f"Using default pricing for unknown model {model} "
                f"from provider {provider}"
            )
            return self.provider_defaults[provider_lower].calculate_cost(
                input_tokens, output_tokens
            )

        logger.warning(
            f"No pricing found for model {model} from provider {provider}. "
            "Using zero cost."
        )
        return Decimal("0")

    def get_model_pricing(self, model: str) -> ModelPricing | None:
        """Get pricing for a specific model."""
        return self.models.get(model)

    def add_custom_pricing(
        self,
        model: str,
        input_per_1k: Decimal | float | str,
        output_per_1k: Decimal | float | str,
        name: str | None = None,
    ) -> None:
        """Add custom pricing for a model."""
        self.models[model] = ModelPricing(
            input_per_1k=Decimal(str(input_per_1k)),
            output_per_1k=Decimal(str(output_per_1k)),
            name=name or model,
        )
