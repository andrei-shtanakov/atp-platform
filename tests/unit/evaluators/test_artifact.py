"""Unit tests for ArtifactEvaluator."""

import pytest

from atp.evaluators.artifact import ArtifactEvaluator
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition
from atp.protocol import (
    ArtifactFile,
    ArtifactStructured,
    ATPResponse,
    ResponseStatus,
)


@pytest.fixture
def evaluator() -> ArtifactEvaluator:
    """Create ArtifactEvaluator instance."""
    return ArtifactEvaluator()


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
def response_with_file_artifacts() -> ATPResponse:
    """Create response with file artifacts."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactFile(
                path="report.md",
                content=(
                    "# Report\n\n## Summary\n\nThis is the summary."
                    "\n\n## Details\n\nSome details."
                ),
                content_type="text/markdown",
            ),
            ArtifactFile(
                path="data.csv",
                content="name,value\nitem1,100\nitem2,200",
                content_type="text/csv",
            ),
        ],
    )


@pytest.fixture
def response_with_structured_artifacts() -> ATPResponse:
    """Create response with structured artifacts."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactStructured(
                name="result",
                data={"status": "success", "count": 42, "items": ["a", "b"]},
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


class TestArtifactExists:
    """Tests for artifact_exists assertion type."""

    @pytest.mark.anyio
    async def test_artifact_exists_pass(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test artifact_exists passes when artifact is found."""
        assertion = Assertion(type="artifact_exists", config={"path": "report.md"})
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is True
        assert result.checks[0].name == "artifact_exists"
        assert "found" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_artifact_exists_fail(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test artifact_exists fails when artifact is not found."""
        assertion = Assertion(type="artifact_exists", config={"path": "missing.txt"})
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False
        assert "not found" in result.checks[0].message.lower()
        assert "available_artifacts" in result.checks[0].details

    @pytest.mark.anyio
    async def test_artifact_exists_empty_response(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
    ) -> None:
        """Test artifact_exists fails on empty response."""
        assertion = Assertion(type="artifact_exists", config={"path": "report.md"})
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        assert result.passed is False

    @pytest.mark.anyio
    async def test_artifact_exists_no_path(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test artifact_exists fails when no path specified."""
        assertion = Assertion(type="artifact_exists", config={})
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False
        assert "no path" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_artifact_exists_structured_by_name(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_structured_artifacts: ATPResponse,
    ) -> None:
        """Test artifact_exists finds structured artifacts by name."""
        assertion = Assertion(type="artifact_exists", config={"path": "result"})
        result = await evaluator.evaluate(
            sample_task, response_with_structured_artifacts, [], assertion
        )
        assert result.passed is True


class TestContains:
    """Tests for contains assertion type."""

    @pytest.mark.anyio
    async def test_contains_plain_text_pass(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test contains passes with plain text match."""
        assertion = Assertion(type="contains", config={"pattern": "Summary"})
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is True
        assert result.checks[0].details["found"] is True

    @pytest.mark.anyio
    async def test_contains_plain_text_fail(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test contains fails when pattern not found."""
        assertion = Assertion(
            type="contains", config={"pattern": "nonexistent_pattern_xyz"}
        )
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False

    @pytest.mark.anyio
    async def test_contains_regex_pass(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test contains passes with regex match."""
        assertion = Assertion(
            type="contains", config={"pattern": r"#\s+Report", "regex": True}
        )
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is True
        assert result.checks[0].details["regex"] is True

    @pytest.mark.anyio
    async def test_contains_regex_fail(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test contains fails with unmatched regex."""
        assertion = Assertion(
            type="contains", config={"pattern": r"^\d{4}-\d{2}-\d{2}$", "regex": True}
        )
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False

    @pytest.mark.anyio
    async def test_contains_invalid_regex(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test contains fails with invalid regex pattern."""
        assertion = Assertion(
            type="contains", config={"pattern": r"[invalid", "regex": True}
        )
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False
        assert "invalid regex" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_contains_specific_artifact(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test contains in specific artifact by path."""
        assertion = Assertion(
            type="contains", config={"path": "data.csv", "pattern": "item1"}
        )
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is True

    @pytest.mark.anyio
    async def test_contains_artifact_not_found(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test contains fails when specified artifact not found."""
        assertion = Assertion(
            type="contains", config={"path": "missing.txt", "pattern": "test"}
        )
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False
        assert "not found" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_contains_no_pattern(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test contains fails when no pattern specified."""
        assertion = Assertion(type="contains", config={})
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False
        assert "no pattern" in result.checks[0].message.lower()


class TestSchema:
    """Tests for schema assertion type."""

    @pytest.mark.anyio
    async def test_schema_validation_pass(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_structured_artifacts: ATPResponse,
    ) -> None:
        """Test schema validation passes with valid data."""
        schema = {
            "type": "object",
            "required": ["status", "count"],
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"},
            },
        }
        assertion = Assertion(
            type="schema", config={"path": "result", "schema": schema}
        )
        result = await evaluator.evaluate(
            sample_task, response_with_structured_artifacts, [], assertion
        )
        assert result.passed is True

    @pytest.mark.anyio
    async def test_schema_validation_missing_required(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_structured_artifacts: ATPResponse,
    ) -> None:
        """Test schema validation fails with missing required field."""
        schema = {
            "type": "object",
            "required": ["missing_field"],
        }
        assertion = Assertion(
            type="schema", config={"path": "result", "schema": schema}
        )
        result = await evaluator.evaluate(
            sample_task, response_with_structured_artifacts, [], assertion
        )
        assert result.passed is False

    @pytest.mark.anyio
    async def test_schema_validation_wrong_type(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_structured_artifacts: ATPResponse,
    ) -> None:
        """Test schema validation fails with wrong type."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "string"},
            },
        }
        assertion = Assertion(
            type="schema", config={"path": "result", "schema": schema}
        )
        result = await evaluator.evaluate(
            sample_task, response_with_structured_artifacts, [], assertion
        )
        assert result.passed is False

    @pytest.mark.anyio
    async def test_schema_no_schema(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_structured_artifacts: ATPResponse,
    ) -> None:
        """Test schema fails when no schema specified."""
        assertion = Assertion(type="schema", config={})
        result = await evaluator.evaluate(
            sample_task, response_with_structured_artifacts, [], assertion
        )
        assert result.passed is False
        assert "no schema" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_schema_no_structured_artifact(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test schema fails when no structured artifact exists."""
        schema = {"type": "object"}
        assertion = Assertion(type="schema", config={"schema": schema})
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False


class TestSections:
    """Tests for sections assertion type."""

    @pytest.mark.anyio
    async def test_sections_all_found(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test sections passes when all sections found."""
        assertion = Assertion(
            type="sections",
            config={"path": "report.md", "sections": ["Summary", "Details"]},
        )
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is True
        assert result.checks[0].details["missing"] == []

    @pytest.mark.anyio
    async def test_sections_some_missing(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test sections fails when some sections missing."""
        assertion = Assertion(
            type="sections",
            config={
                "path": "report.md",
                "sections": ["Summary", "Missing Section"],
            },
        )
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False
        assert "Missing Section" in result.checks[0].details["missing"]

    @pytest.mark.anyio
    async def test_sections_case_insensitive(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test sections matching is case insensitive."""
        assertion = Assertion(
            type="sections",
            config={"path": "report.md", "sections": ["SUMMARY", "details"]},
        )
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is True

    @pytest.mark.anyio
    async def test_sections_no_sections_specified(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test sections fails when no sections specified."""
        assertion = Assertion(type="sections", config={})
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False
        assert "no sections" in result.checks[0].message.lower()


class TestUnknownAssertionType:
    """Tests for unknown assertion types."""

    @pytest.mark.anyio
    async def test_unknown_type(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test unknown assertion type returns failure."""
        assertion = Assertion(type="unknown_type", config={})
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False
        assert "unknown" in result.checks[0].message.lower()


class TestEdgeCases:
    """Edge case tests for ArtifactEvaluator."""

    @pytest.mark.anyio
    async def test_contains_artifact_no_content(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test contains when artifact has no content."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(path="empty.md", content=None),
            ],
        )
        assertion = Assertion(type="contains", config={"pattern": "test"})
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        assert result.passed is False

    @pytest.mark.anyio
    async def test_schema_artifact_not_found(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_structured_artifacts: ATPResponse,
    ) -> None:
        """Test schema fails when specified artifact not found."""
        schema = {"type": "object"}
        assertion = Assertion(
            type="schema", config={"path": "missing", "schema": schema}
        )
        result = await evaluator.evaluate(
            sample_task, response_with_structured_artifacts, [], assertion
        )
        assert result.passed is False
        assert "not found" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_sections_artifact_not_found(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifacts: ATPResponse,
    ) -> None:
        """Test sections fails when specified artifact not found."""
        assertion = Assertion(
            type="sections",
            config={"path": "missing.md", "sections": ["Summary"]},
        )
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifacts, [], assertion
        )
        assert result.passed is False
        assert "not found" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_sections_artifact_no_content(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test sections when artifact has no content."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(path="empty.md", content=None),
            ],
        )
        assertion = Assertion(
            type="sections",
            config={"path": "empty.md", "sections": ["Summary"]},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        assert result.passed is False
        assert "no artifact content" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_sections_found_in_text_not_header(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test sections found in text content, not as markdown header."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(
                    path="doc.md",
                    content="This document discusses the Introduction topic.",
                ),
            ],
        )
        assertion = Assertion(
            type="sections",
            config={"path": "doc.md", "sections": ["Introduction"]},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        assert result.passed is True

    @pytest.mark.anyio
    async def test_contains_in_structured_artifact(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
        response_with_structured_artifacts: ATPResponse,
    ) -> None:
        """Test contains can search in structured artifact JSON."""
        assertion = Assertion(type="contains", config={"pattern": "success"})
        result = await evaluator.evaluate(
            sample_task, response_with_structured_artifacts, [], assertion
        )
        assert result.passed is True

    @pytest.mark.anyio
    async def test_schema_number_type_validation(
        self,
        evaluator: ArtifactEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test schema validation with number type (accepts int or float)."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="data",
                    data={"value": 3.14},
                ),
            ],
        )
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
        }
        assertion = Assertion(type="schema", config={"path": "data", "schema": schema})
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        assert result.passed is True


class TestEvaluatorProperties:
    """Tests for evaluator properties."""

    def test_evaluator_name(self, evaluator: ArtifactEvaluator) -> None:
        """Test evaluator name property."""
        assert evaluator.name == "artifact"
