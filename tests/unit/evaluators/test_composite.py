"""Unit tests for CompositeEvaluator."""

from unittest.mock import AsyncMock

import pytest

from atp.evaluation import AssertionUnevaluated
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
    """A composite bound to the full registry — the CLI's trusted context.

    There is no unbound variant that still works: an unbound composite cannot
    resolve anything, by design. That is the fix in ADR-008 track A — the
    global-registry fallback was the policy hole.
    """
    from atp.evaluators.registry import get_registry

    return CompositeEvaluator(get_registry())


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
        """A threshold with nothing to compare is malformed, not failed."""
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
        with pytest.raises(AssertionUnevaluated):
            await evaluator.evaluate(sample_task, sample_response, [], assertion)


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
        with pytest.raises(AssertionUnevaluated):
            await evaluator.evaluate(sample_task, sample_response, [], assertion)

    @pytest.mark.anyio
    async def test_missing_type_in_condition(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """A malformed condition is unmeasured, not measured-and-bad."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {"config": {"path": "output.txt"}},
                ],
            },
        )
        with pytest.raises(AssertionUnevaluated):
            await evaluator.evaluate(sample_task, sample_response, [], assertion)

    @pytest.mark.anyio
    async def test_unknown_assertion_type(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        sample_response: ATPResponse,
    ) -> None:
        """An unresolvable leaf is unknown, so the AND above it is unknown."""
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
        with pytest.raises(AssertionUnevaluated):
            await evaluator.evaluate(sample_task, sample_response, [], assertion)


class StubResolver:
    """Resolves whatever the test says, and records what was asked for."""

    def __init__(self, evaluators: dict[str, object]) -> None:
        self._evaluators = evaluators
        self.asked: list[str] = []

    def create_for_assertion(self, assertion_type: str) -> object:
        self.asked.append(assertion_type)
        if assertion_type not in self._evaluators:
            raise LookupError(f"no evaluator for '{assertion_type}'")
        return self._evaluators[assertion_type]


class TestWithInjectedEvaluators:
    """Sub-evaluators arrive through the injected resolver.

    These used to patch `get_registry`, which is exactly the call the fix
    removes — a test that patches a global is a test that only passes while
    the global is still being consulted.
    """

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

        composite = CompositeEvaluator(StubResolver({"mock_type": mock_evaluator}))

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
        result = await composite.evaluate(sample_task, sample_response, [], assertion)
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

        composite = CompositeEvaluator(
            StubResolver({"type_a": fail_eval, "type_b": pass_eval})
        )

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
        result = await composite.evaluate(sample_task, sample_response, [], assertion)
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


class TestWorkspaceRootReachesLeaves:
    """The resolver rule, applied to the filesystem (ADR-008 track B).

    A composite touches no files itself, so it would be easy to leave the
    grant at the top level — and then every filesystem leaf inside a composite
    would refuse, or worse, be given a root nobody chose. The grant travels
    down exactly as the resolver does.
    """

    @pytest.fixture
    def workspace(self, tmp_path):
        (tmp_path / "report.md").write_text("findings")
        return tmp_path

    @pytest.mark.anyio
    async def test_a_filesystem_leaf_is_granted_the_composites_root(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
        workspace,
    ) -> None:
        evaluator.bind_workspace_root(workspace)
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {"type": "file_exists", "config": {"path": "report.md"}}
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_a_leaf_nested_two_levels_deep_is_granted_it_too(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
        workspace,
    ) -> None:
        """Depth is where a per-level rule quietly stops being applied."""
        evaluator.bind_workspace_root(workspace)
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {
                        "type": "composite",
                        "config": {
                            "operator": "and",
                            "conditions": [
                                {
                                    "type": "file_exists",
                                    "config": {"path": "report.md"},
                                }
                            ],
                        },
                    }
                ],
            },
        )
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        assert result.passed

    @pytest.mark.anyio
    async def test_an_ungranted_composite_leaves_its_filesystem_leaf_unevaluated(
        self,
        evaluator: CompositeEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
    ) -> None:
        """Fail-closed all the way down: unmeasured, not failed."""
        assertion = Assertion(
            type="composite",
            config={
                "operator": "and",
                "conditions": [
                    {"type": "file_exists", "config": {"path": "report.md"}}
                ],
            },
        )
        with pytest.raises(AssertionUnevaluated):
            await evaluator.evaluate(sample_task, empty_response, [], assertion)
