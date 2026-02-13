"""Pre-run cost estimation for ATP test suites.

Estimates LLM API costs before running a test suite based on prompt sizes,
model pricing, and number of runs.

Example usage:
    from atp.analytics.estimator import CostEstimator, CostEstimate

    estimator = CostEstimator()
    estimate = estimator.estimate_suite(suite, model="gpt-4o", runs_per_test=3)
    print(f"Estimated cost: ${estimate.total_min} - ${estimate.total_max}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from atp.analytics.cost import ModelPricing, PricingConfig
from atp.loader.loader import TestLoader
from atp.loader.models import TestDefinition, TestSuite

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ~ 4 characters for English text
CHARS_PER_TOKEN = 4

# Default expected output token ranges (min/max) by task complexity
# These are rough heuristics based on typical agent responses
OUTPUT_TOKENS_MIN = 100
OUTPUT_TOKENS_MAX = 500

# System prompt overhead tokens (adapter framing, instructions)
SYSTEM_PROMPT_OVERHEAD = 200


@dataclass
class TestEstimate:
    """Cost estimate for a single test.

    Attributes:
        test_id: Test identifier.
        test_name: Human-readable test name.
        input_tokens: Estimated input token count.
        output_tokens_min: Minimum estimated output tokens.
        output_tokens_max: Maximum estimated output tokens.
        cost_min: Minimum estimated cost in USD.
        cost_max: Maximum estimated cost in USD.
        runs: Number of runs for this test.
    """

    test_id: str
    test_name: str
    input_tokens: int
    output_tokens_min: int
    output_tokens_max: int
    cost_min: Decimal
    cost_max: Decimal
    runs: int


@dataclass
class CostEstimate:
    """Aggregate cost estimate for a test suite.

    Attributes:
        suite_name: Name of the test suite.
        model: Model used for estimation.
        provider: Provider name.
        total_tests: Number of tests in the suite.
        total_runs: Total number of runs across all tests.
        total_input_tokens: Total estimated input tokens.
        total_output_tokens_min: Minimum total estimated output tokens.
        total_output_tokens_max: Maximum total estimated output tokens.
        total_min: Minimum estimated total cost in USD.
        total_max: Maximum estimated total cost in USD.
        tests: Per-test estimates.
        pricing: Pricing used for estimation.
    """

    suite_name: str
    model: str
    provider: str
    total_tests: int
    total_runs: int
    total_input_tokens: int
    total_output_tokens_min: int
    total_output_tokens_max: int
    total_min: Decimal
    total_max: Decimal
    tests: list[TestEstimate] = field(default_factory=list)
    pricing: ModelPricing | None = None


class CostEstimator:
    """Estimates LLM API costs for test suites before execution.

    Uses test definitions (prompt sizes), model pricing, and run counts
    to produce min/max cost ranges.
    """

    def __init__(
        self,
        pricing_config: PricingConfig | None = None,
    ) -> None:
        """Initialize cost estimator.

        Args:
            pricing_config: Pricing configuration. Uses defaults if not provided.
        """
        self.pricing = pricing_config or PricingConfig.default()

    def estimate_suite(
        self,
        suite: TestSuite,
        model: str,
        provider: str = "",
        runs_per_test: int | None = None,
    ) -> CostEstimate:
        """Estimate costs for a test suite.

        Args:
            suite: Loaded test suite.
            model: Model name for pricing lookup.
            provider: Provider name (used as fallback for pricing).
            runs_per_test: Override runs per test. Uses suite defaults if None.

        Returns:
            CostEstimate with min/max cost ranges.
        """
        runs = runs_per_test or suite.defaults.runs_per_test
        model_pricing = self._get_pricing(model, provider)

        test_estimates: list[TestEstimate] = []
        total_input = 0
        total_output_min = 0
        total_output_max = 0
        total_cost_min = Decimal("0")
        total_cost_max = Decimal("0")

        for test in suite.tests:
            input_tokens = self._estimate_input_tokens(test)
            output_min = self._estimate_output_tokens_min(test)
            output_max = self._estimate_output_tokens_max(test)

            # Calculate cost for all runs of this test
            run_cost_min = Decimal("0")
            run_cost_max = Decimal("0")
            if model_pricing:
                run_cost_min = (
                    model_pricing.calculate_cost(input_tokens, output_min) * runs
                )
                run_cost_max = (
                    model_pricing.calculate_cost(input_tokens, output_max) * runs
                )

            test_estimates.append(
                TestEstimate(
                    test_id=test.id,
                    test_name=test.name,
                    input_tokens=input_tokens,
                    output_tokens_min=output_min,
                    output_tokens_max=output_max,
                    cost_min=run_cost_min,
                    cost_max=run_cost_max,
                    runs=runs,
                )
            )

            total_input += input_tokens * runs
            total_output_min += output_min * runs
            total_output_max += output_max * runs
            total_cost_min += run_cost_min
            total_cost_max += run_cost_max

        return CostEstimate(
            suite_name=suite.test_suite,
            model=model,
            provider=provider,
            total_tests=len(suite.tests),
            total_runs=len(suite.tests) * runs,
            total_input_tokens=total_input,
            total_output_tokens_min=total_output_min,
            total_output_tokens_max=total_output_max,
            total_min=total_cost_min,
            total_max=total_cost_max,
            tests=test_estimates,
            pricing=model_pricing,
        )

    def estimate_from_file(
        self,
        suite_path: str | Path,
        model: str,
        provider: str = "",
        runs_per_test: int | None = None,
    ) -> CostEstimate:
        """Estimate costs from a suite file path.

        Args:
            suite_path: Path to test suite YAML file.
            model: Model name for pricing lookup.
            provider: Provider name (used as fallback for pricing).
            runs_per_test: Override runs per test.

        Returns:
            CostEstimate with min/max cost ranges.
        """
        loader = TestLoader()
        suite = loader.load_file(suite_path)
        return self.estimate_suite(
            suite, model=model, provider=provider, runs_per_test=runs_per_test
        )

    def _get_pricing(self, model: str, provider: str) -> ModelPricing | None:
        """Look up pricing for a model.

        Args:
            model: Model name.
            provider: Provider name (fallback).

        Returns:
            ModelPricing if found, None otherwise.
        """
        pricing = self.pricing.get_model_pricing(model)
        if pricing:
            return pricing

        # Try provider default
        if provider:
            provider_lower = provider.lower()
            if provider_lower in self.pricing.provider_defaults:
                logger.debug(
                    "Using default pricing for %s from provider %s",
                    model,
                    provider,
                )
                return self.pricing.provider_defaults[provider_lower]

        logger.warning(
            "No pricing found for model %s (provider: %s). "
            "Cost estimates will be zero.",
            model,
            provider,
        )
        return None

    def _estimate_input_tokens(self, test: TestDefinition) -> int:
        """Estimate input tokens for a test.

        Counts characters in the task description and input data,
        then converts to approximate token count.

        Args:
            test: Test definition.

        Returns:
            Estimated input token count.
        """
        char_count = 0

        # Task description
        char_count += len(test.task.description)

        # Input data (serialized)
        if test.task.input_data:
            for key, value in test.task.input_data.items():
                char_count += len(str(key)) + len(str(value))

        # Expected artifacts (mentioned in prompt context)
        if test.task.expected_artifacts:
            for artifact in test.task.expected_artifacts:
                char_count += len(artifact)

        # Constraints contribute to system prompt
        if test.constraints:
            if test.constraints.allowed_tools:
                for tool in test.constraints.allowed_tools:
                    char_count += len(tool)

        # Convert chars to tokens + system overhead
        tokens = (char_count // CHARS_PER_TOKEN) + SYSTEM_PROMPT_OVERHEAD
        return max(tokens, SYSTEM_PROMPT_OVERHEAD)

    def _estimate_output_tokens_min(self, test: TestDefinition) -> int:
        """Estimate minimum output tokens for a test.

        Args:
            test: Test definition.

        Returns:
            Minimum estimated output token count.
        """
        # Base minimum
        tokens = OUTPUT_TOKENS_MIN

        # More steps allowed = potentially more output
        if test.constraints and test.constraints.max_steps:
            tokens = max(tokens, test.constraints.max_steps * 30)

        return tokens

    def _estimate_output_tokens_max(self, test: TestDefinition) -> int:
        """Estimate maximum output tokens for a test.

        Args:
            test: Test definition.

        Returns:
            Maximum estimated output token count.
        """
        tokens = OUTPUT_TOKENS_MAX

        # More steps = potentially much more output
        if test.constraints and test.constraints.max_steps:
            tokens = max(tokens, test.constraints.max_steps * 150)

        # If max_tokens is set, use it as upper bound
        if test.constraints and test.constraints.max_tokens:
            tokens = min(tokens, test.constraints.max_tokens)

        return tokens
