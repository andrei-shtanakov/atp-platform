"""Unit tests for CompositeEvaluator."""

from unittest.mock import AsyncMock, patch

import pytest

from atp.evaluators.base import EvalCheck, EvalResult
from atp.evaluators.composite import CompositeEvaluator
from atp.loader.models import (
    Assertion,
    Constraints,
    TaskDefinition,
    TestDefinition,
)
from atp.protocol import (
    ArtifactFile,
    ATPResponse,
    ResponseStatus,
)


@pytest.fixture
def evaluator() -> CompositeEvaluator:
    """Create CompositeEvaluator instance."""
    return CompositeEvaluator()


@pytest.fixture
def sample_task() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-001",
        name="Sample Test",
        task=TaskDefinition(description="Test task"),
        constraints=Constraints(),
    )


@pytest.fixture
def sample_response() -> ATPResponse:
    """Create a sample response with artifacts."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactFile(
                path="output.txt",
                content="Hello world",
                content_type="text/plain",
            ),
        ],
    )


@pytest.fixture
def empty_response() -> ATPResponse:
    """Create response with no artifacts."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[],
    )


def _make_result(passed: bool, score: float, name: str = "test") -> EvalResult:
    """Helper to create a mock EvalResult."""
    return EvalResult(
        evaluator=name,
        checks=[
            EvalCheck(
                name=name,
                passed=passed,
                score=score,
                message=f"{'passed' if passed else 'failed'}",
            )
        ],
    )


class TestEvaluatorProperties:
    """Tests for evaluator properties."""

    def test_evaluator_name(self, evaluator: CompositeEvaluator) -> None:
        """Test evaluator name property."""
        assert evaluator.name == "composite"


class TestAndOperator:
    """Tests for AND operator."""

    @pytest.mark.anyio
    async def test_and_all_pass(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """AND: all conditions pass -> composite passes."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "output.txt"},
                    },
                    {
                        "type": "contains",
                        "config": {"pattern": "Hello"},
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is True

    @pytest.mark.anyio
    async def test_and_one_fails(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """AND: one condition fails -> composite fails."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "output.txt"},
                    },
                    {
                        "type": "artifact_exists",
                        "config": {"path": "missing.txt"},
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is False

    @pytest.mark.anyio
    async def test_and_all_fail(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
    ) -> None:
        """AND: all conditions fail -> composite fails."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "a.txt"},
                    },
                    {
                        "type": "artifact_exists",
                        "config": {"path": "b.txt"},
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        assert result.passed is False


class TestOrOperator:
    """Tests for OR operator."""

    @pytest.mark.anyio
    async def test_or_one_passes(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """OR: one condition passes -> composite passes."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "or",
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "output.txt"},
                    },
                    {
                        "type": "artifact_exists",
                        "config": {"path": "missing.txt"},
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is True

    @pytest.mark.anyio
    async def test_or_all_fail(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
    ) -> None:
        """OR: all conditions fail -> composite fails."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "or",
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "a.txt"},
                    },
                    {
                        "type": "artifact_exists",
                        "config": {"path": "b.txt"},
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        assert result.passed is False

    @pytest.mark.anyio
    async def test_or_all_pass(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """OR: all conditions pass -> composite passes."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "or",
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "output.txt"},
                    },
                    {
                        "type": "contains",
                        "config": {"pattern": "Hello"},
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is True


class TestNotOperator:
    """Tests for NOT operator."""

    @pytest.mark.anyio
    async def test_not_inverts_pass_to_fail(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """NOT: passing condition becomes failure."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "operator": "not",
                        "condition": {
                            "type": "artifact_exists",
                            "config": {"path": "output.txt"},
                        },
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is False

    @pytest.mark.anyio
    async def test_not_inverts_fail_to_pass(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
    ) -> None:
        """NOT: failing condition becomes pass."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "operator": "not",
                        "condition": {
                            "type": "artifact_exists",
                            "config": {"path": "missing.txt"},
                        },
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        assert result.passed is True


class TestNestedComposition:
    """Tests for nested composite structures."""

    @pytest.mark.anyio
    async def test_and_with_nested_or(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """AND(artifact_exists, OR(contains, artifact_exists))."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "output.txt"},
                    },
                    {
                        "operator": "or",
                        "conditions": [
                            {
                                "type": "contains",
                                "config": {"pattern": "nonexistent"},
                            },
                            {
                                "type": "contains",
                                "config": {"pattern": "Hello"},
                            },
                        ],
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is True

    @pytest.mark.anyio
    async def test_nested_and_fails_when_inner_or_fails(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """AND(artifact, OR(fail, fail)) -> fails."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "output.txt"},
                    },
                    {
                        "operator": "or",
                        "conditions": [
                            {
                                "type": "artifact_exists",
                                "config": {"path": "a.txt"},
                            },
                            {
                                "type": "artifact_exists",
                                "config": {"path": "b.txt"},
                            },
                        ],
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is False

    @pytest.mark.anyio
    async def test_or_with_nested_not(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
    ) -> None:
        """OR(fail, NOT(fail)) -> passes."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "or",
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "missing.txt"},
                    },
                    {
                        "operator": "not",
                        "condition": {
                            "type": "artifact_exists",
                            "config": {"path": "also_missing.txt"},
                        },
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        assert result.passed is True


class TestThresholdCondition:
    """Tests for threshold conditions."""

    @pytest.mark.anyio
    async def test_threshold_passes(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Threshold: score >= value passes."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "operator": "threshold",
                        "value": 0.8,
                        "comparator": ">=",
                        "condition": {
                            "type": "artifact_exists",
                            "config": {"path": "output.txt"},
                        },
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        # artifact_exists passes -> score=1.0 >= 0.8
        assert result.passed is True

    @pytest.mark.anyio
    async def test_threshold_fails(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
    ) -> None:
        """Threshold: score < value fails."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "operator": "threshold",
                        "value": 0.8,
                        "comparator": ">=",
                        "condition": {
                            "type": "artifact_exists",
                            "config": {"path": "missing.txt"},
                        },
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        # artifact_exists fails -> score=0.0 < 0.8
        assert result.passed is False

    @pytest.mark.anyio
    async def test_threshold_greater_than(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Threshold with > comparator."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "operator": "threshold",
                        "value": 0.5,
                        "comparator": ">",
                        "condition": {
                            "type": "artifact_exists",
                            "config": {"path": "output.txt"},
                        },
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is True

    @pytest.mark.anyio
    async def test_threshold_no_inner_condition(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Threshold without inner condition fails."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "operator": "threshold",
                        "value": 0.8,
                        "comparator": ">=",
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is False


class TestEmptyConditions:
    """Tests for edge cases."""

    @pytest.mark.anyio
    async def test_empty_conditions(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Empty conditions list passes (vacuous truth)."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is True
        assert "vacuous" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_no_conditions_key(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Missing conditions key passes (vacuous truth)."""
        assertion = Assertion(
            type="composite",
            config={"operator": "and"},
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is True


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.anyio
    async def test_unknown_operator(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Unknown operator fails."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "xor",
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "output.txt"},
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is False

    @pytest.mark.anyio
    async def test_missing_type_in_condition(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Condition without type fails."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {"config": {"path": "output.txt"}},
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is False

    @pytest.mark.anyio
    async def test_unknown_assertion_type(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Unknown assertion type in condition fails."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "type": "nonexistent_evaluator",
                        "config": {},
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is False


class TestWithMockedEvaluators:
    """Tests using mocked sub-evaluators."""

    @pytest.mark.anyio
    async def test_and_with_mocked_evaluators(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """AND with mocked evaluators returning specific scores."""
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=_make_result(True, 0.9, "mock")
        )

        with patch("atp.evaluators.registry.get_registry") as mock_registry:
            registry = mock_registry.return_value
            registry.supports_assertion.return_value = True
            registry.create_for_assertion.return_value = mock_evaluator

            assertion = Assertion(
                type="composite",
                config={
                    "operator": "and",
                    "conditions": [
                        {"type": "mock_type", "config": {}},
                        {"type": "mock_type", "config": {}},
                    ],
                },
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )
            assert result.passed is True
            assert mock_evaluator.evaluate.call_count == 2

    @pytest.mark.anyio
    async def test_or_with_mixed_mocked_results(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """OR with one pass and one fail from mocked evaluators."""
        pass_eval = AsyncMock()
        pass_eval.evaluate = AsyncMock(return_value=_make_result(True, 1.0, "pass"))
        fail_eval = AsyncMock()
        fail_eval.evaluate = AsyncMock(return_value=_make_result(False, 0.0, "fail"))

        call_count = 0

        def create_evaluator(assertion_type: str) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fail_eval
            return pass_eval

        with patch("atp.evaluators.registry.get_registry") as mock_registry:
            registry = mock_registry.return_value
            registry.supports_assertion.return_value = True
            registry.create_for_assertion.side_effect = create_evaluator

            assertion = Assertion(
                type="composite",
                config={
                    "operator": "or",
                    "conditions": [
                        {"type": "type_a", "config": {}},
                        {"type": "type_b", "config": {}},
                    ],
                },
            )
            result = await evaluator.evaluate(
                sample_task, sample_response, [], assertion
            )
            assert result.passed is True


class TestDefaultOperator:
    """Test default operator behavior."""

    @pytest.mark.anyio
    async def test_default_operator_is_and(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """Default operator (no operator key) behaves as AND."""
        assertion = Assertion(
            type="composite",
            config={
                "conditions": [
                    {
                        "type": "artifact_exists",
                        "config": {"path": "output.txt"},
                    },
                    {
                        "type": "contains",
                        "config": {"pattern": "Hello"},
                    },
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, sample_response, [], assertion)
        assert result.passed is True


class TestRegistration:
    """Tests for registry integration."""

    def test_composite_registered(self) -> None:
        """Composite evaluator is registered in the registry."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        assert registry.is_registered("composite")

    def test_composite_assertion_supported(self) -> None:
        """Composite assertion type is supported."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        assert registry.supports_assertion("composite")

    def test_create_composite_evaluator(self) -> None:
        """Can create composite evaluator from registry."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        ev = registry.create("composite")
        assert ev.name == "composite"
        assert isinstance(ev, CompositeEvaluator)
