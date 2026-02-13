"""Tests for pre-run cost estimation."""

from decimal import Decimal
from pathlib import Path

import pytest
from click.testing import CliRunner

from atp.analytics.cost import ModelPricing, PricingConfig
from atp.analytics.estimator import (
    SYSTEM_PROMPT_OVERHEAD,
    CostEstimate,
    CostEstimator,
)
from atp.cli.main import cli
from atp.loader.models import (
    Constraints,
    TaskDefinition,
    TestDefaults,
    TestDefinition,
    TestSuite,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pricing_config() -> PricingConfig:
    """Return a minimal PricingConfig for tests."""
    return PricingConfig(
        models={
            "test-model": ModelPricing(
                input_per_1k=Decimal("0.01"),
                output_per_1k=Decimal("0.03"),
                name="Test Model",
            ),
        },
        provider_defaults={
            "test-provider": ModelPricing(
                input_per_1k=Decimal("0.005"),
                output_per_1k=Decimal("0.015"),
                name="Test Provider Default",
            ),
        },
    )


@pytest.fixture()
def simple_suite() -> TestSuite:
    """Return a minimal TestSuite for estimation tests."""
    return TestSuite(
        test_suite="estimate-test-suite",
        version="1.0",
        defaults=TestDefaults(runs_per_test=1),
        tests=[
            TestDefinition(
                id="test-001",
                name="Simple test",
                task=TaskDefinition(description="Echo hello"),
                constraints=Constraints(max_steps=1),
            ),
        ],
    )


@pytest.fixture()
def multi_test_suite() -> TestSuite:
    """Return a suite with multiple tests of varying complexity."""
    return TestSuite(
        test_suite="multi-test-suite",
        version="1.0",
        defaults=TestDefaults(runs_per_test=2),
        tests=[
            TestDefinition(
                id="test-001",
                name="Simple echo",
                task=TaskDefinition(description="Echo hello"),
                constraints=Constraints(max_steps=1),
            ),
            TestDefinition(
                id="test-002",
                name="Complex file task",
                task=TaskDefinition(
                    description="Create three files with specific content "
                    "and perform calculations on the data within them",
                    input_data={"key": "value", "data": "some input"},
                    expected_artifacts=["file1.txt", "file2.txt"],
                ),
                constraints=Constraints(max_steps=5),
            ),
            TestDefinition(
                id="test-003",
                name="Token-limited task",
                task=TaskDefinition(description="Short task"),
                constraints=Constraints(max_steps=3, max_tokens=200),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# CostEstimator unit tests
# ---------------------------------------------------------------------------


class TestCostEstimator:
    """Tests for CostEstimator class."""

    def test_estimate_suite_basic(
        self,
        pricing_config: PricingConfig,
        simple_suite: TestSuite,
    ) -> None:
        """Estimator returns correct structure for a simple suite."""
        estimator = CostEstimator(pricing_config=pricing_config)
        estimate = estimator.estimate_suite(simple_suite, model="test-model")

        assert isinstance(estimate, CostEstimate)
        assert estimate.suite_name == "estimate-test-suite"
        assert estimate.model == "test-model"
        assert estimate.total_tests == 1
        assert estimate.total_runs == 1
        assert len(estimate.tests) == 1
        assert estimate.pricing is not None

    def test_estimate_suite_costs_positive(
        self,
        pricing_config: PricingConfig,
        simple_suite: TestSuite,
    ) -> None:
        """Costs should be positive when pricing is found."""
        estimator = CostEstimator(pricing_config=pricing_config)
        estimate = estimator.estimate_suite(simple_suite, model="test-model")

        assert estimate.total_min > Decimal("0")
        assert estimate.total_max > Decimal("0")
        assert estimate.total_max >= estimate.total_min

    def test_estimate_suite_min_le_max(
        self,
        pricing_config: PricingConfig,
        multi_test_suite: TestSuite,
    ) -> None:
        """Min cost should always be <= max cost."""
        estimator = CostEstimator(pricing_config=pricing_config)
        estimate = estimator.estimate_suite(multi_test_suite, model="test-model")

        assert estimate.total_min <= estimate.total_max
        for t in estimate.tests:
            assert t.cost_min <= t.cost_max

    def test_estimate_suite_runs_multiplier(
        self,
        pricing_config: PricingConfig,
        simple_suite: TestSuite,
    ) -> None:
        """Costs should scale with the number of runs."""
        estimator = CostEstimator(pricing_config=pricing_config)

        est_1 = estimator.estimate_suite(
            simple_suite, model="test-model", runs_per_test=1
        )
        est_3 = estimator.estimate_suite(
            simple_suite, model="test-model", runs_per_test=3
        )

        assert est_3.total_min == est_1.total_min * 3
        assert est_3.total_max == est_1.total_max * 3
        assert est_3.total_runs == 3

    def test_estimate_suite_unknown_model(
        self,
        pricing_config: PricingConfig,
        simple_suite: TestSuite,
    ) -> None:
        """Unknown model without provider fallback yields zero costs."""
        estimator = CostEstimator(pricing_config=pricing_config)
        estimate = estimator.estimate_suite(simple_suite, model="unknown-model")

        assert estimate.total_min == Decimal("0")
        assert estimate.total_max == Decimal("0")
        assert estimate.pricing is None

    def test_estimate_suite_provider_fallback(
        self,
        pricing_config: PricingConfig,
        simple_suite: TestSuite,
    ) -> None:
        """Unknown model with a known provider uses provider defaults."""
        estimator = CostEstimator(pricing_config=pricing_config)
        estimate = estimator.estimate_suite(
            simple_suite,
            model="unknown-model",
            provider="test-provider",
        )

        assert estimate.total_min > Decimal("0")
        assert estimate.pricing is not None
        assert estimate.pricing.name == "Test Provider Default"

    def test_estimate_input_tokens_includes_overhead(
        self,
        pricing_config: PricingConfig,
    ) -> None:
        """Input token estimate includes system prompt overhead."""
        suite = TestSuite(
            test_suite="overhead-test",
            tests=[
                TestDefinition(
                    id="t1",
                    name="Tiny",
                    task=TaskDefinition(description="Hi"),
                    constraints=Constraints(),
                ),
            ],
        )
        estimator = CostEstimator(pricing_config=pricing_config)
        estimate = estimator.estimate_suite(suite, model="test-model")

        # Even with a tiny description, tokens >= overhead
        assert estimate.tests[0].input_tokens >= SYSTEM_PROMPT_OVERHEAD

    def test_estimate_input_tokens_scales_with_description(
        self,
        pricing_config: PricingConfig,
    ) -> None:
        """Longer descriptions produce more input tokens."""
        short_suite = TestSuite(
            test_suite="short",
            tests=[
                TestDefinition(
                    id="t1",
                    name="Short",
                    task=TaskDefinition(description="Do X"),
                ),
            ],
        )
        long_suite = TestSuite(
            test_suite="long",
            tests=[
                TestDefinition(
                    id="t1",
                    name="Long",
                    task=TaskDefinition(description="A" * 2000),
                ),
            ],
        )
        estimator = CostEstimator(pricing_config=pricing_config)
        short_est = estimator.estimate_suite(short_suite, model="test-model")
        long_est = estimator.estimate_suite(long_suite, model="test-model")

        assert long_est.tests[0].input_tokens > short_est.tests[0].input_tokens

    def test_estimate_output_tokens_respects_max_tokens(
        self,
        pricing_config: PricingConfig,
    ) -> None:
        """Output token max is capped by constraints.max_tokens."""
        suite = TestSuite(
            test_suite="capped",
            tests=[
                TestDefinition(
                    id="t1",
                    name="Capped",
                    task=TaskDefinition(description="Task"),
                    constraints=Constraints(max_steps=10, max_tokens=50),
                ),
            ],
        )
        estimator = CostEstimator(pricing_config=pricing_config)
        estimate = estimator.estimate_suite(suite, model="test-model")

        assert estimate.tests[0].output_tokens_max <= 50

    def test_estimate_multi_test_totals(
        self,
        pricing_config: PricingConfig,
        multi_test_suite: TestSuite,
    ) -> None:
        """Total costs equal sum of per-test costs."""
        estimator = CostEstimator(pricing_config=pricing_config)
        estimate = estimator.estimate_suite(multi_test_suite, model="test-model")

        sum_min = sum(t.cost_min for t in estimate.tests)
        sum_max = sum(t.cost_max for t in estimate.tests)

        assert estimate.total_min == sum_min
        assert estimate.total_max == sum_max

    def test_estimate_from_file(
        self,
        pricing_config: PricingConfig,
        tmp_path: Path,
    ) -> None:
        """estimate_from_file loads YAML and produces estimates."""
        suite_yaml = tmp_path / "suite.yaml"
        suite_yaml.write_text(
            """\
test_suite: file-test
version: "1.0"
tests:
  - id: t1
    name: Test One
    task:
      description: "Do something"
"""
        )

        estimator = CostEstimator(pricing_config=pricing_config)
        estimate = estimator.estimate_from_file(suite_yaml, model="test-model")

        assert estimate.suite_name == "file-test"
        assert estimate.total_tests == 1
        assert estimate.total_min > Decimal("0")

    def test_estimate_uses_suite_defaults_runs(
        self,
        pricing_config: PricingConfig,
    ) -> None:
        """Uses suite defaults.runs_per_test when not overridden."""
        suite = TestSuite(
            test_suite="defaults-runs",
            defaults=TestDefaults(runs_per_test=5),
            tests=[
                TestDefinition(
                    id="t1",
                    name="T",
                    task=TaskDefinition(description="X"),
                ),
            ],
        )
        estimator = CostEstimator(pricing_config=pricing_config)
        estimate = estimator.estimate_suite(suite, model="test-model")

        assert estimate.total_runs == 5
        assert estimate.tests[0].runs == 5


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------


class TestEstimateCLI:
    """Tests for the `atp estimate` CLI command."""

    @pytest.fixture()
    def suite_file(self, tmp_path: Path) -> Path:
        """Create a temporary test suite YAML file."""
        suite_yaml = tmp_path / "suite.yaml"
        suite_yaml.write_text(
            """\
test_suite: cli-test
version: "1.0"
defaults:
  runs_per_test: 1
tests:
  - id: t1
    name: CLI Test
    task:
      description: "Simple CLI test task"
    constraints:
      max_steps: 2
"""
        )
        return suite_yaml

    def test_estimate_console_output(self, suite_file: Path) -> None:
        """Console output includes key information."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["estimate", str(suite_file), "--model=gpt-4o"],
        )

        assert result.exit_code == 0
        assert "cli-test" in result.output
        assert "gpt-4o" in result.output
        assert "t1" in result.output

    def test_estimate_json_output(self, suite_file: Path) -> None:
        """JSON output is valid and contains required fields."""
        import json

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate",
                str(suite_file),
                "--model=gpt-4o",
                "--output=json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["suite_name"] == "cli-test"
        assert data["model"] == "gpt-4o"
        assert data["total_tests"] == 1
        assert "total_min_usd" in data
        assert "total_max_usd" in data
        assert len(data["tests"]) == 1

    def test_estimate_budget_check_pass(self, suite_file: Path) -> None:
        """Budget check passes when estimate is within budget."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate",
                str(suite_file),
                "--model=gpt-4o",
                "--budget-check=100.00",
            ],
        )

        assert result.exit_code == 0
        assert "Budget OK" in result.output

    def test_estimate_budget_check_fail(self, suite_file: Path) -> None:
        """Budget check fails when estimate exceeds budget."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate",
                str(suite_file),
                "--model=gpt-4o",
                "--budget-check=0.0000001",
            ],
        )

        assert result.exit_code == 1
        assert "BUDGET CHECK FAILED" in result.output

    def test_estimate_custom_runs(self, suite_file: Path) -> None:
        """Custom --runs flag affects output."""
        import json

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate",
                str(suite_file),
                "--model=gpt-4o",
                "--runs=5",
                "--output=json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_runs"] == 5
        assert data["tests"][0]["runs"] == 5

    def test_estimate_invalid_suite(self, tmp_path: Path) -> None:
        """Error for non-existent suite file."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate",
                str(tmp_path / "nonexistent.yaml"),
                "--model=gpt-4o",
            ],
        )

        assert result.exit_code != 0

    def test_estimate_missing_model_flag(self, suite_file: Path) -> None:
        """Error when --model is not provided."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["estimate", str(suite_file)],
        )

        assert result.exit_code != 0

    def test_estimate_with_provider_fallback(self, suite_file: Path) -> None:
        """Provider flag used for fallback pricing."""
        import json

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate",
                str(suite_file),
                "--model=some-unknown-model",
                "--provider=openai",
                "--output=json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        # Should have non-zero costs via provider default
        assert Decimal(data["total_max_usd"]) > Decimal("0")

    def test_estimate_with_tag_filter(self, tmp_path: Path) -> None:
        """Tag filtering works correctly."""
        import json

        suite_yaml = tmp_path / "tagged.yaml"
        suite_yaml.write_text(
            """\
test_suite: tagged-suite
version: "1.0"
tests:
  - id: t1
    name: Smoke Test
    tags: [smoke]
    task:
      description: "Smoke test"
  - id: t2
    name: Integration Test
    tags: [integration]
    task:
      description: "Integration test"
"""
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "estimate",
                str(suite_yaml),
                "--model=gpt-4o",
                "--tags=smoke",
                "--output=json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_tests"] == 1
        assert data["tests"][0]["test_id"] == "t1"
