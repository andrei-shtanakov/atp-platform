"""Tests for atp.sdk.compare multi-model comparison."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from atp.loader.models import (
    Constraints,
    TaskDefinition,
    TestDefinition,
    TestSuite,
)
from atp.protocol import ATPResponse, ResponseStatus
from atp.runner.models import RunResult, SuiteResult, TestResult
from atp.sdk.compare import (
    ComparisonResult,
    ModelConfig,
    _build_comparison,
    _resolve_configs,
    format_comparison_json,
    format_comparison_table,
)


@pytest.fixture
def sample_suite() -> TestSuite:
    """Create a sample test suite."""
    return TestSuite(
        test_suite="comparison-suite",
        tests=[
            TestDefinition(
                id="test-1",
                name="Test One",
                task=TaskDefinition(description="Do something"),
                constraints=Constraints(timeout_seconds=30),
            ),
            TestDefinition(
                id="test-2",
                name="Test Two",
                task=TaskDefinition(description="Do something else"),
                constraints=Constraints(timeout_seconds=30),
            ),
        ],
    )


@pytest.fixture
def model_a_result() -> SuiteResult:
    """Create a suite result for model A."""
    now = datetime.now()
    return SuiteResult(
        suite_name="comparison-suite",
        agent_name="model-a",
        tests=[
            TestResult(
                test=TestDefinition(
                    id="test-1",
                    name="Test One",
                    task=TaskDefinition(description="Do something"),
                ),
                runs=[
                    RunResult(
                        test_id="test-1",
                        run_number=1,
                        response=ATPResponse(
                            task_id="t1",
                            status=ResponseStatus.COMPLETED,
                        ),
                        start_time=now,
                        end_time=now,
                    )
                ],
                start_time=now,
                end_time=now,
            ),
            TestResult(
                test=TestDefinition(
                    id="test-2",
                    name="Test Two",
                    task=TaskDefinition(description="Do something else"),
                ),
                runs=[
                    RunResult(
                        test_id="test-2",
                        run_number=1,
                        response=ATPResponse(
                            task_id="t2",
                            status=ResponseStatus.COMPLETED,
                        ),
                        start_time=now,
                        end_time=now,
                    )
                ],
                start_time=now,
                end_time=now,
            ),
        ],
        start_time=now,
        end_time=now,
    )


@pytest.fixture
def model_b_result() -> SuiteResult:
    """Create a suite result for model B (one failure)."""
    now = datetime.now()
    return SuiteResult(
        suite_name="comparison-suite",
        agent_name="model-b",
        tests=[
            TestResult(
                test=TestDefinition(
                    id="test-1",
                    name="Test One",
                    task=TaskDefinition(description="Do something"),
                ),
                runs=[
                    RunResult(
                        test_id="test-1",
                        run_number=1,
                        response=ATPResponse(
                            task_id="t1",
                            status=ResponseStatus.COMPLETED,
                        ),
                        start_time=now,
                        end_time=now,
                    )
                ],
                start_time=now,
                end_time=now,
            ),
            TestResult(
                test=TestDefinition(
                    id="test-2",
                    name="Test Two",
                    task=TaskDefinition(description="Do something else"),
                ),
                runs=[
                    RunResult(
                        test_id="test-2",
                        run_number=1,
                        response=ATPResponse(
                            task_id="t2",
                            status=ResponseStatus.FAILED,
                            error="Something went wrong",
                        ),
                        start_time=now,
                        end_time=now,
                    )
                ],
                start_time=now,
                end_time=now,
            ),
        ],
        start_time=now,
        end_time=now,
    )


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_model_config_creation(self) -> None:
        """Test creating a ModelConfig."""
        mc = ModelConfig(
            name="gpt-4",
            adapter="http",
            config={"base_url": "http://localhost:8000"},
        )
        assert mc.name == "gpt-4"
        assert mc.adapter == "http"
        assert mc.config["base_url"] == "http://localhost:8000"

    def test_model_config_defaults(self) -> None:
        """Test ModelConfig with default empty config."""
        mc = ModelConfig(name="test", adapter="cli")
        assert mc.config == {}


class TestResolveConfigs:
    """Tests for _resolve_configs."""

    def test_resolve_model_config_objects(self) -> None:
        """Test resolving ModelConfig objects."""
        configs = [
            ModelConfig(name="a", adapter="http"),
            ModelConfig(name="b", adapter="cli"),
        ]
        result = _resolve_configs(configs)
        assert len(result) == 2
        assert result[0].name == "a"
        assert result[1].name == "b"

    def test_resolve_dict_configs(self) -> None:
        """Test resolving dict configs."""
        configs = [
            {"name": "a", "adapter": "http", "config": {"x": 1}},
            {"name": "b", "adapter": "cli"},
        ]
        result = _resolve_configs(configs)
        assert len(result) == 2
        assert isinstance(result[0], ModelConfig)
        assert result[0].config == {"x": 1}

    def test_resolve_mixed_configs(self) -> None:
        """Test resolving mixed ModelConfig and dict inputs."""
        configs = [
            ModelConfig(name="a", adapter="http"),
            {"name": "b", "adapter": "cli"},
        ]
        result = _resolve_configs(configs)
        assert len(result) == 2


class TestBuildComparison:
    """Tests for _build_comparison."""

    def test_build_comparison(
        self,
        model_a_result: SuiteResult,
        model_b_result: SuiteResult,
    ) -> None:
        """Test building a comparison from suite results."""
        result = _build_comparison(
            suite_name="comparison-suite",
            models=["model-a", "model-b"],
            suite_results={
                "model-a": model_a_result,
                "model-b": model_b_result,
            },
        )

        assert result.suite_name == "comparison-suite"
        assert result.models == ["model-a", "model-b"]
        assert len(result.tests) == 2

        # Test 1: both models passed
        tc1 = result.tests[0]
        assert tc1.test_id == "test-1"
        assert tc1.passed["model-a"] is True
        assert tc1.passed["model-b"] is True
        assert tc1.scores["model-a"] == 100.0
        assert tc1.scores["model-b"] == 100.0

        # Test 2: model-a passed, model-b failed
        tc2 = result.tests[1]
        assert tc2.test_id == "test-2"
        assert tc2.passed["model-a"] is True
        assert tc2.passed["model-b"] is False

        # Summary
        assert result.summary["model-a"]["success_rate"] == 1.0
        assert result.summary["model-b"]["success_rate"] == 0.5

    def test_best_model(
        self,
        model_a_result: SuiteResult,
        model_b_result: SuiteResult,
    ) -> None:
        """Test best_model property."""
        result = _build_comparison(
            suite_name="test",
            models=["model-a", "model-b"],
            suite_results={
                "model-a": model_a_result,
                "model-b": model_b_result,
            },
        )
        assert result.best_model == "model-a"

    def test_best_model_empty(self) -> None:
        """Test best_model with no data."""
        result = ComparisonResult(
            suite_name="test",
            models=[],
        )
        assert result.best_model is None


class TestFormatComparisonTable:
    """Tests for format_comparison_table."""

    def test_format_table(
        self,
        model_a_result: SuiteResult,
        model_b_result: SuiteResult,
    ) -> None:
        """Test table formatting."""
        comparison = _build_comparison(
            suite_name="test",
            models=["model-a", "model-b"],
            suite_results={
                "model-a": model_a_result,
                "model-b": model_b_result,
            },
        )
        table = format_comparison_table(comparison)
        assert "model-a" in table
        assert "model-b" in table
        assert "Test One" in table
        assert "Test Two" in table
        assert "PASS" in table
        assert "FAIL" in table
        assert "Best model: model-a" in table

    def test_format_empty_table(self) -> None:
        """Test formatting empty comparison."""
        result = ComparisonResult(
            suite_name="test",
            models=[],
        )
        table = format_comparison_table(result)
        assert "No comparison data" in table


class TestFormatComparisonJson:
    """Tests for format_comparison_json."""

    def test_format_json(
        self,
        model_a_result: SuiteResult,
        model_b_result: SuiteResult,
    ) -> None:
        """Test JSON formatting."""
        comparison = _build_comparison(
            suite_name="test-suite",
            models=["model-a", "model-b"],
            suite_results={
                "model-a": model_a_result,
                "model-b": model_b_result,
            },
        )
        data = format_comparison_json(comparison)

        assert data["suite_name"] == "test-suite"
        assert data["models"] == ["model-a", "model-b"]
        assert data["best_model"] == "model-a"
        assert len(data["tests"]) == 2
        assert "scores" in data["tests"][0]
        assert "passed" in data["tests"][0]


class TestAcompare:
    """Tests for acompare (async comparison)."""

    @pytest.mark.anyio
    async def test_acompare_runs_each_model(
        self,
        sample_suite: TestSuite,
        model_a_result: SuiteResult,
        model_b_result: SuiteResult,
    ) -> None:
        """Test that acompare runs the suite for each model config."""
        from atp.sdk.compare import acompare

        call_count = 0
        results = [model_a_result, model_b_result]

        async def mock_run_for_model(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            result = results[call_count]
            call_count += 1
            return result

        with (
            patch(
                "atp.sdk.compare._run_for_model",
                side_effect=mock_run_for_model,
            ),
            patch(
                "atp.sdk.compare._resolve_adapter",
                return_value=AsyncMock(),
            ),
        ):
            result = await acompare(
                suite=sample_suite,
                configs=[
                    ModelConfig(name="model-a", adapter="http"),
                    ModelConfig(name="model-b", adapter="http"),
                ],
            )

        assert isinstance(result, ComparisonResult)
        assert result.models == ["model-a", "model-b"]
        assert call_count == 2
