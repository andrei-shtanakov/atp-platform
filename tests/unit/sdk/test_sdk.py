"""Tests for atp.sdk public API."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.evaluators.base import EvalCheck, EvalResult
from atp.loader.models import (
    Assertion,
    Constraints,
    TaskDefinition,
    TestDefinition,
    TestSuite,
)
from atp.protocol import ATPResponse, ResponseStatus
from atp.runner.models import SuiteResult
from atp.scoring.models import ScoredTestResult


@pytest.fixture
def sample_test() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-1",
        name="Sample Test",
        task=TaskDefinition(description="Do something"),
        constraints=Constraints(timeout_seconds=30),
        assertions=[
            Assertion(
                type="artifact_exists",
                config={"artifact": "output.txt"},
            )
        ],
    )


@pytest.fixture
def sample_suite(sample_test: TestDefinition) -> TestSuite:
    """Create a sample test suite."""
    return TestSuite(
        test_suite="test-suite",
        tests=[sample_test],
    )


@pytest.fixture
def sample_response() -> ATPResponse:
    """Create a sample ATP response."""
    return ATPResponse(
        task_id="task-1",
        status=ResponseStatus.COMPLETED,
    )


@pytest.fixture
def sample_eval_result() -> EvalResult:
    """Create a sample evaluation result."""
    return EvalResult(
        evaluator="artifact",
        checks=[
            EvalCheck(
                name="artifact_exists",
                passed=True,
                score=1.0,
                message="Artifact found",
            )
        ],
    )


class TestLoadSuite:
    """Tests for load_suite and load_suite_string."""

    def test_load_suite_from_file(self, tmp_path: Path) -> None:
        """Test loading a suite from a YAML file."""
        import atp.sdk as sdk

        suite_yaml = tmp_path / "suite.yaml"
        suite_yaml.write_text(
            """
test_suite: my-suite
tests:
  - id: test-1
    name: Test One
    task:
      description: Do something
    assertions:
      - type: artifact_exists
        config:
          artifact: output.txt
"""
        )

        suite = sdk.load_suite(suite_yaml)
        assert suite.test_suite == "my-suite"
        assert len(suite.tests) == 1
        assert suite.tests[0].id == "test-1"

    def test_load_suite_string(self) -> None:
        """Test loading a suite from a YAML string."""
        import atp.sdk as sdk

        content = """
test_suite: string-suite
tests:
  - id: test-1
    name: Test One
    task:
      description: Do something
"""
        suite = sdk.load_suite_string(content)
        assert suite.test_suite == "string-suite"
        assert len(suite.tests) == 1

    def test_load_suite_with_env(self, tmp_path: Path) -> None:
        """Test loading a suite with custom environment."""
        import atp.sdk as sdk

        suite_yaml = tmp_path / "suite.yaml"
        suite_yaml.write_text(
            """
test_suite: env-suite
tests:
  - id: test-1
    name: Test One
    task:
      description: "${MY_VAR}"
"""
        )

        suite = sdk.load_suite(suite_yaml, env={"MY_VAR": "custom value"})
        assert suite.tests[0].task.description == "custom value"


class TestArun:
    """Tests for arun (async suite execution)."""

    @pytest.mark.anyio
    async def test_arun_with_suite_object(self, sample_suite: TestSuite) -> None:
        """Test arun with a TestSuite object."""
        import atp.sdk as sdk

        mock_adapter = AsyncMock(spec=AgentAdapter)
        mock_adapter.stream_events = MagicMock(
            return_value=_async_iter(
                [
                    ATPResponse(
                        task_id="t1",
                        status=ResponseStatus.COMPLETED,
                    )
                ]
            )
        )
        mock_adapter.cleanup = AsyncMock()

        result = await sdk.arun(
            suite=sample_suite,
            adapter=mock_adapter,
            agent_name="test-agent",
        )

        assert isinstance(result, SuiteResult)
        assert result.suite_name == "test-suite"
        assert result.agent_name == "test-agent"

    @pytest.mark.anyio
    async def test_arun_with_path(self, tmp_path: Path) -> None:
        """Test arun with a file path."""
        import atp.sdk as sdk

        suite_yaml = tmp_path / "suite.yaml"
        suite_yaml.write_text(
            """
test_suite: path-suite
tests:
  - id: test-1
    name: Test One
    task:
      description: Do something
"""
        )

        mock_adapter = AsyncMock(spec=AgentAdapter)
        mock_adapter.stream_events = MagicMock(
            return_value=_async_iter(
                [
                    ATPResponse(
                        task_id="t1",
                        status=ResponseStatus.COMPLETED,
                    )
                ]
            )
        )
        mock_adapter.cleanup = AsyncMock()

        result = await sdk.arun(
            suite=str(suite_yaml),
            adapter=mock_adapter,
            agent_name="test-agent",
        )

        assert isinstance(result, SuiteResult)
        assert result.suite_name == "path-suite"

    @pytest.mark.anyio
    async def test_arun_with_adapter_string(self, sample_suite: TestSuite) -> None:
        """Test arun with an adapter type string."""
        import atp.sdk as sdk

        mock_adapter_instance = AsyncMock()
        mock_adapter_instance.stream_events = MagicMock(
            return_value=_async_iter(
                [
                    ATPResponse(
                        task_id="t1",
                        status=ResponseStatus.COMPLETED,
                    )
                ]
            )
        )
        mock_adapter_instance.cleanup = AsyncMock()

        with patch(
            "atp.sdk.create_adapter",
            return_value=mock_adapter_instance,
        ) as mock_create:
            result = await sdk.arun(
                suite=sample_suite,
                adapter="http",
                adapter_config={"base_url": "http://localhost:8000"},
            )

            mock_create.assert_called_once_with(
                "http",
                {"base_url": "http://localhost:8000"},
            )
            assert isinstance(result, SuiteResult)

    @pytest.mark.anyio
    async def test_arun_with_tag_filter(self, sample_suite: TestSuite) -> None:
        """Test arun with tag filtering."""
        import atp.sdk as sdk

        sample_suite.tests[0].tags = ["smoke"]

        mock_adapter = AsyncMock(spec=AgentAdapter)
        mock_adapter.stream_events = MagicMock(
            return_value=_async_iter(
                [
                    ATPResponse(
                        task_id="t1",
                        status=ResponseStatus.COMPLETED,
                    )
                ]
            )
        )
        mock_adapter.cleanup = AsyncMock()

        result = await sdk.arun(
            suite=sample_suite,
            adapter=mock_adapter,
            tag_filter="smoke",
        )

        assert isinstance(result, SuiteResult)


class TestAevaluate:
    """Tests for aevaluate (async evaluation)."""

    @pytest.mark.anyio
    async def test_aevaluate_with_test_assertions(
        self,
        sample_test: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test aevaluate using assertions from test definition."""
        import atp.sdk as sdk

        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(
                        name="artifact_exists",
                        passed=True,
                        score=1.0,
                    )
                ],
            )
        )

        with patch("atp.sdk.get_evaluator_registry") as mock_registry_fn:
            mock_registry = MagicMock()
            mock_registry.create_for_assertion.return_value = mock_evaluator
            mock_registry_fn.return_value = mock_registry

            results = await sdk.aevaluate(
                response=sample_response,
                test=sample_test,
            )

            assert len(results) == 1
            assert results[0].passed

    @pytest.mark.anyio
    async def test_aevaluate_with_custom_assertions(
        self,
        sample_test: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Test aevaluate with explicitly provided assertions."""
        import atp.sdk as sdk

        custom_assertions = [
            Assertion(type="contains", config={"text": "hello"}),
            Assertion(type="artifact_exists", config={"artifact": "x"}),
        ]

        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=EvalResult(
                evaluator="artifact",
                checks=[
                    EvalCheck(
                        name="check",
                        passed=True,
                        score=1.0,
                    )
                ],
            )
        )

        with patch("atp.sdk.get_evaluator_registry") as mock_registry_fn:
            mock_registry = MagicMock()
            mock_registry.create_for_assertion.return_value = mock_evaluator
            mock_registry_fn.return_value = mock_registry

            results = await sdk.aevaluate(
                response=sample_response,
                test=sample_test,
                assertions=custom_assertions,
            )

            assert len(results) == 2

    @pytest.mark.anyio
    async def test_aevaluate_with_empty_assertions(
        self,
        sample_response: ATPResponse,
    ) -> None:
        """Test aevaluate with empty assertions list."""
        import atp.sdk as sdk

        test = TestDefinition(
            id="test-empty",
            name="Empty",
            task=TaskDefinition(description="No assertions"),
            assertions=[],
        )

        results = await sdk.aevaluate(
            response=sample_response,
            test=test,
        )

        assert results == []


class TestScore:
    """Tests for score function."""

    def test_score_basic(
        self,
        sample_eval_result: EvalResult,
        sample_response: ATPResponse,
    ) -> None:
        """Test basic scoring."""
        import atp.sdk as sdk

        result = sdk.score(
            eval_results=[sample_eval_result],
            test_id="test-1",
            response=sample_response,
        )

        assert isinstance(result, ScoredTestResult)
        assert result.test_id == "test-1"
        assert result.passed is True
        assert result.score > 0

    def test_score_without_response(
        self,
        sample_eval_result: EvalResult,
    ) -> None:
        """Test scoring without a response."""
        import atp.sdk as sdk

        result = sdk.score(
            eval_results=[sample_eval_result],
            test_id="test-2",
        )

        assert isinstance(result, ScoredTestResult)
        assert result.passed is True

    def test_score_with_failed_checks(self) -> None:
        """Test scoring with failed evaluation checks."""
        import atp.sdk as sdk

        failed_result = EvalResult(
            evaluator="artifact",
            checks=[
                EvalCheck(
                    name="check-fail",
                    passed=False,
                    score=0.0,
                    message="Not found",
                )
            ],
        )

        result = sdk.score(
            eval_results=[failed_result],
            test_id="test-3",
        )

        assert isinstance(result, ScoredTestResult)
        assert result.passed is False


class TestResolveAdapter:
    """Tests for adapter resolution."""

    def test_resolve_adapter_instance(self) -> None:
        """Test that adapter instances are returned as-is."""
        from atp.sdk import _resolve_adapter

        mock_adapter = AsyncMock(spec=AgentAdapter)
        result = _resolve_adapter(mock_adapter)
        assert result is mock_adapter

    def test_resolve_adapter_string(self) -> None:
        """Test that string adapter types are resolved via registry."""
        from atp.sdk import _resolve_adapter

        mock_adapter = AsyncMock()
        with patch(
            "atp.sdk.create_adapter",
            return_value=mock_adapter,
        ) as mock_create:
            result = _resolve_adapter("http", {"base_url": "http://localhost"})
            mock_create.assert_called_once_with(
                "http", {"base_url": "http://localhost"}
            )
            assert result is mock_adapter

    def test_resolve_adapter_string_no_config(self) -> None:
        """Test adapter resolution with no config defaults to empty dict."""
        from atp.sdk import _resolve_adapter

        mock_adapter = AsyncMock()
        with patch(
            "atp.sdk.create_adapter",
            return_value=mock_adapter,
        ) as mock_create:
            _resolve_adapter("cli")
            mock_create.assert_called_once_with("cli", {})


# Helper for creating async iterators from lists
async def _async_iter(items: list):  # type: ignore[type-arg]
    """Create an async iterator from a list."""
    for item in items:
        yield item


# Import AgentAdapter at module level for spec= usage
from atp.adapters.base import AgentAdapter  # noqa: E402
