"""Unit tests for LLMJudgeEvaluator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.evaluators.llm_judge import (
    BUILTIN_CRITERIA,
    LLMJudgeConfig,
    LLMJudgeCost,
    LLMJudgeEvaluator,
    LLMJudgeResponse,
)
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition
from atp.protocol import ArtifactFile, ArtifactStructured, ATPResponse, ResponseStatus


@pytest.fixture
def evaluator() -> LLMJudgeEvaluator:
    """Create LLMJudgeEvaluator instance with default config."""
    return LLMJudgeEvaluator()


@pytest.fixture
def evaluator_with_config() -> LLMJudgeEvaluator:
    """Create LLMJudgeEvaluator instance with custom config."""
    config = LLMJudgeConfig(
        api_key="test-api-key",
        model="claude-sonnet-4-20250514",
        temperature=0.0,
        max_tokens=1024,
        num_runs=1,
        timeout=30.0,
    )
    return LLMJudgeEvaluator(config)


@pytest.fixture
def sample_task() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-001",
        name="Sample Test",
        task=TaskDefinition(description="Write a summary of AI trends"),
        constraints=Constraints(),
    )


@pytest.fixture
def response_with_file_artifact() -> ATPResponse:
    """Create response with file artifact."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactFile(
                path="report.md",
                content="# AI Trends Report\n\nAI is transforming various industries.",
                content_type="text/markdown",
            ),
        ],
    )


@pytest.fixture
def response_with_structured_artifact() -> ATPResponse:
    """Create response with structured artifact."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactStructured(
                name="result",
                data={"summary": "AI trends are evolving", "score": 95},
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


@pytest.fixture
def mock_llm_success_response() -> dict:
    """Create mock successful LLM response."""
    return {
        "score": 0.85,
        "explanation": "The content is well-written and factually accurate.",
        "issues": ["Minor formatting issues"],
        "strengths": ["Clear structure", "Good examples"],
    }


class TestLLMJudgeConfig:
    """Tests for LLMJudgeConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = LLMJudgeConfig()
        assert config.api_key is None
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.num_runs == 1
        assert config.timeout == 60.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = LLMJudgeConfig(
            api_key="test-key",
            model="custom-model",
            temperature=0.5,
            max_tokens=2048,
            num_runs=3,
            timeout=120.0,
        )
        assert config.api_key == "test-key"
        assert config.model == "custom-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.num_runs == 3
        assert config.timeout == 120.0


class TestLLMJudgeCost:
    """Tests for LLMJudgeCost."""

    def test_default_cost(self) -> None:
        """Test default cost values."""
        cost = LLMJudgeCost()
        assert cost.input_tokens == 0
        assert cost.output_tokens == 0
        assert cost.total_calls == 0
        assert cost.estimated_cost_usd == 0.0

    def test_estimated_cost_calculation(self) -> None:
        """Test cost estimation calculation."""
        cost = LLMJudgeCost(
            input_tokens=1000000,
            output_tokens=500000,
            total_calls=10,
        )
        # $3/M input + $15/M output = $3 + $7.5 = $10.5
        expected = 3.0 + 7.5
        assert cost.estimated_cost_usd == expected


class TestLLMJudgeResponse:
    """Tests for LLMJudgeResponse."""

    def test_valid_response(self) -> None:
        """Test valid response creation."""
        response = LLMJudgeResponse(
            score=0.8,
            explanation="Good content",
            issues=["Issue 1"],
            strengths=["Strength 1"],
        )
        assert response.score == 0.8
        assert response.explanation == "Good content"
        assert response.issues == ["Issue 1"]
        assert response.strengths == ["Strength 1"]

    def test_score_bounds(self) -> None:
        """Test score must be between 0 and 1."""
        with pytest.raises(ValueError):
            LLMJudgeResponse(score=1.5, explanation="test")
        with pytest.raises(ValueError):
            LLMJudgeResponse(score=-0.5, explanation="test")

    def test_default_lists(self) -> None:
        """Test default empty lists for issues and strengths."""
        response = LLMJudgeResponse(score=0.5, explanation="test")
        assert response.issues == []
        assert response.strengths == []


class TestLLMJudgeEvaluatorProperties:
    """Tests for LLMJudgeEvaluator properties."""

    def test_evaluator_name(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test evaluator name property."""
        assert evaluator.name == "llm_judge"

    def test_cost_property(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test cost property returns LLMJudgeCost."""
        cost = evaluator.cost
        assert isinstance(cost, LLMJudgeCost)
        assert cost.total_calls == 0

    def test_reset_cost_tracking(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test reset_cost_tracking method."""
        evaluator._total_cost.input_tokens = 1000
        evaluator._total_cost.output_tokens = 500
        evaluator._total_cost.total_calls = 5
        evaluator.reset_cost_tracking()
        assert evaluator.cost.input_tokens == 0
        assert evaluator.cost.output_tokens == 0
        assert evaluator.cost.total_calls == 0

    def test_get_available_criteria(self) -> None:
        """Test get_available_criteria class method."""
        criteria = LLMJudgeEvaluator.get_available_criteria()
        assert "factual_accuracy" in criteria
        assert "completeness" in criteria
        assert "relevance" in criteria
        assert "coherence" in criteria
        assert "clarity" in criteria
        assert "actionability" in criteria


class TestBuiltinCriteria:
    """Tests for built-in criteria."""

    def test_all_builtin_criteria_defined(self) -> None:
        """Test all expected criteria are defined."""
        expected = [
            "factual_accuracy",
            "completeness",
            "relevance",
            "coherence",
            "clarity",
            "actionability",
        ]
        for criteria in expected:
            assert criteria in BUILTIN_CRITERIA

    def test_criteria_have_descriptions(self) -> None:
        """Test all criteria have non-empty descriptions."""
        for criteria, description in BUILTIN_CRITERIA.items():
            assert len(description) > 0, f"Criteria {criteria} has no description"


class TestBuildPrompt:
    """Tests for prompt building."""

    def test_build_prompt_with_criteria(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test building prompt with built-in criteria."""
        prompt = evaluator._build_prompt(
            task_description="Write a report",
            artifact_content="Report content here",
            criteria="factual_accuracy",
        )
        assert "Write a report" in prompt
        assert "Report content here" in prompt
        assert "factual_accuracy" in prompt
        assert "CRITERION" in prompt

    def test_build_prompt_with_custom_prompt(
        self, evaluator: LLMJudgeEvaluator
    ) -> None:
        """Test building prompt with custom prompt."""
        prompt = evaluator._build_prompt(
            task_description="Write a report",
            artifact_content="Report content here",
            custom_prompt="Check for spelling errors",
        )
        assert "Write a report" in prompt
        assert "Report content here" in prompt
        assert "Check for spelling errors" in prompt

    def test_build_prompt_with_both_criteria_and_custom(
        self, evaluator: LLMJudgeEvaluator
    ) -> None:
        """Test building prompt with both criteria and custom prompt."""
        prompt = evaluator._build_prompt(
            task_description="Write a report",
            artifact_content="Report content here",
            criteria="completeness",
            custom_prompt="Also check formatting",
        )
        assert "completeness" in prompt
        assert "Also check formatting" in prompt

    def test_build_prompt_no_criteria_or_custom(
        self, evaluator: LLMJudgeEvaluator
    ) -> None:
        """Test building prompt fails without criteria or custom prompt."""
        with pytest.raises(ValueError) as exc_info:
            evaluator._build_prompt(
                task_description="Write a report",
                artifact_content="Content",
            )
        assert "Either 'criteria' or 'prompt'" in str(exc_info.value)

    def test_build_prompt_unknown_criteria(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test building prompt with unknown criteria fails."""
        with pytest.raises(ValueError) as exc_info:
            evaluator._build_prompt(
                task_description="Write a report",
                artifact_content="Content",
                criteria="unknown_criteria",
            )
        assert "Unknown criteria" in str(exc_info.value)

    def test_build_prompt_truncates_long_content(
        self, evaluator: LLMJudgeEvaluator
    ) -> None:
        """Test prompt truncates very long artifact content."""
        long_content = "x" * 100000
        prompt = evaluator._build_prompt(
            task_description="Test",
            artifact_content=long_content,
            criteria="completeness",
        )
        assert "truncated" in prompt.lower()
        assert len(prompt) < len(long_content) + 5000


class TestParseResponse:
    """Tests for response parsing."""

    def test_parse_valid_json(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test parsing valid JSON response."""
        json_text = """
        {
            "score": 0.85,
            "explanation": "Good content",
            "issues": ["Issue 1"],
            "strengths": ["Strength 1"]
        }
        """
        response = evaluator._parse_response(json_text)
        assert response.score == 0.85
        assert response.explanation == "Good content"
        assert response.issues == ["Issue 1"]
        assert response.strengths == ["Strength 1"]

    def test_parse_json_in_code_block(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        json_text = """```json
        {
            "score": 0.75,
            "explanation": "Decent content",
            "issues": [],
            "strengths": []
        }
        ```"""
        response = evaluator._parse_response(json_text)
        assert response.score == 0.75

    def test_parse_json_with_surrounding_text(
        self, evaluator: LLMJudgeEvaluator
    ) -> None:
        """Test parsing JSON with text around it."""
        text = """Here is my evaluation:
        {
            "score": 0.9,
            "explanation": "Excellent",
            "issues": [],
            "strengths": ["Well done"]
        }
        That's my assessment."""
        response = evaluator._parse_response(text)
        assert response.score == 0.9

    def test_parse_score_only_fallback(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test fallback parsing when only score is extractable."""
        text = "My evaluation: score = 0.7, because the content is good."
        response = evaluator._parse_response(text)
        assert response.score == 0.7

    def test_parse_clamps_score_to_bounds(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test that out-of-bound scores are clamped."""
        json_text = '{"score": 1.5, "explanation": "test"}'
        response = evaluator._parse_response(json_text)
        assert response.score == 1.0

        json_text = '{"score": -0.5, "explanation": "test"}'
        response = evaluator._parse_response(json_text)
        assert response.score == 0.0

    def test_parse_handles_string_score(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test parsing when score is a string."""
        json_text = '{"score": "0.8", "explanation": "test"}'
        response = evaluator._parse_response(json_text)
        assert response.score == 0.8

    def test_parse_invalid_json_fails(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test parsing fails with completely invalid content."""
        text = "This is just text with no score information"
        with pytest.raises(ValueError):
            evaluator._parse_response(text)


class TestGetArtifactContent:
    """Tests for artifact content extraction."""

    def test_get_file_artifact_content(
        self,
        evaluator: LLMJudgeEvaluator,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test extracting content from file artifact."""
        content = evaluator._get_artifact_content(response_with_file_artifact)
        assert content is not None
        assert "AI Trends Report" in content

    def test_get_structured_artifact_content(
        self,
        evaluator: LLMJudgeEvaluator,
        response_with_structured_artifact: ATPResponse,
    ) -> None:
        """Test extracting content from structured artifact."""
        content = evaluator._get_artifact_content(response_with_structured_artifact)
        assert content is not None
        assert "AI trends are evolving" in content

    def test_get_artifact_by_path(
        self,
        evaluator: LLMJudgeEvaluator,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test extracting content from specific artifact by path."""
        content = evaluator._get_artifact_content(
            response_with_file_artifact, path="report.md"
        )
        assert content is not None

    def test_get_artifact_wrong_path(
        self,
        evaluator: LLMJudgeEvaluator,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test returns None when path doesn't exist."""
        content = evaluator._get_artifact_content(
            response_with_file_artifact, path="nonexistent.md"
        )
        assert content is None

    def test_get_artifact_empty_response(
        self,
        evaluator: LLMJudgeEvaluator,
        empty_response: ATPResponse,
    ) -> None:
        """Test returns None for empty response."""
        content = evaluator._get_artifact_content(empty_response)
        assert content is None


class TestEvaluate:
    """Tests for the main evaluate method."""

    @pytest.mark.anyio
    async def test_evaluate_no_artifact(
        self,
        evaluator: LLMJudgeEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
    ) -> None:
        """Test evaluate fails when no artifact found."""
        assertion = Assertion(type="llm_eval", config={"criteria": "factual_accuracy"})
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        assert result.passed is False
        assert "no artifact" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_evaluate_no_criteria_or_prompt(
        self,
        evaluator: LLMJudgeEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test evaluate fails when no criteria or prompt specified."""
        assertion = Assertion(type="llm_eval", config={})
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifact, [], assertion
        )
        assert result.passed is False
        assert "criteria" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_evaluate_invalid_criteria(
        self,
        evaluator: LLMJudgeEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test evaluate fails with invalid criteria."""
        assertion = Assertion(type="llm_eval", config={"criteria": "invalid"})
        result = await evaluator.evaluate(
            sample_task, response_with_file_artifact, [], assertion
        )
        assert result.passed is False
        assert "unknown criteria" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_evaluate_success_above_threshold(
        self,
        evaluator_with_config: LLMJudgeEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
        mock_llm_success_response: dict,
    ) -> None:
        """Test evaluate passes when score exceeds threshold."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"score": 0.85, "explanation": "Good"}')
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        with patch.object(evaluator_with_config, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            assertion = Assertion(
                type="llm_eval",
                config={"criteria": "factual_accuracy", "threshold": 0.7},
            )
            result = await evaluator_with_config.evaluate(
                sample_task, response_with_file_artifact, [], assertion
            )
            assert result.passed is True
            assert result.checks[0].score == 0.85

    @pytest.mark.anyio
    async def test_evaluate_failure_below_threshold(
        self,
        evaluator_with_config: LLMJudgeEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test evaluate fails when score below threshold."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"score": 0.5, "explanation": "Poor"}')
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        with patch.object(evaluator_with_config, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            assertion = Assertion(
                type="llm_eval",
                config={"criteria": "factual_accuracy", "threshold": 0.7},
            )
            result = await evaluator_with_config.evaluate(
                sample_task, response_with_file_artifact, [], assertion
            )
            assert result.passed is False
            assert result.checks[0].score == 0.5

    @pytest.mark.anyio
    async def test_evaluate_with_custom_prompt(
        self,
        evaluator_with_config: LLMJudgeEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test evaluate with custom prompt."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"score": 0.9, "explanation": "Good"}')
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        with patch.object(evaluator_with_config, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            assertion = Assertion(
                type="llm_eval",
                config={"prompt": "Check for spelling errors"},
            )
            result = await evaluator_with_config.evaluate(
                sample_task, response_with_file_artifact, [], assertion
            )
            assert result.passed is True
            assert "custom" in result.checks[0].name

    @pytest.mark.anyio
    async def test_evaluate_tracks_cost(
        self,
        evaluator_with_config: LLMJudgeEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test that evaluation tracks cost."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"score": 0.8, "explanation": "OK"}')]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 200

        with patch.object(evaluator_with_config, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            assertion = Assertion(type="llm_eval", config={"criteria": "completeness"})
            await evaluator_with_config.evaluate(
                sample_task, response_with_file_artifact, [], assertion
            )

            assert evaluator_with_config.cost.input_tokens == 500
            assert evaluator_with_config.cost.output_tokens == 200
            assert evaluator_with_config.cost.total_calls == 1

    @pytest.mark.anyio
    async def test_evaluate_includes_cost_in_details(
        self,
        evaluator_with_config: LLMJudgeEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test that evaluation result includes cost details."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"score": 0.8, "explanation": "OK"}')]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 200

        with patch.object(evaluator_with_config, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            assertion = Assertion(type="llm_eval", config={"criteria": "completeness"})
            result = await evaluator_with_config.evaluate(
                sample_task, response_with_file_artifact, [], assertion
            )

            assert "cost" in result.checks[0].details
            assert "input_tokens" in result.checks[0].details["cost"]


class TestEvaluateWithAveraging:
    """Tests for multi-call averaging."""

    @pytest.mark.anyio
    async def test_averaging_multiple_runs(
        self,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test averaging across multiple runs."""
        config = LLMJudgeConfig(api_key="test", num_runs=3)
        evaluator = LLMJudgeEvaluator(config)

        scores = [0.7, 0.8, 0.9]
        call_count = [0]  # Use list to allow mutation in async closure

        async def create_mock_response(*args, **kwargs) -> MagicMock:
            score = scores[call_count[0] % len(scores)]
            call_count[0] += 1
            mock = MagicMock()
            mock.content = [
                MagicMock(text=f'{{"score": {score}, "explanation": "Test"}}')
            ]
            mock.usage.input_tokens = 100
            mock.usage.output_tokens = 50
            return mock

        with patch.object(evaluator, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = create_mock_response
            mock_get_client.return_value = mock_client

            assertion = Assertion(
                type="llm_eval",
                config={"criteria": "factual_accuracy", "num_runs": 3},
            )
            result = await evaluator.evaluate(
                sample_task, response_with_file_artifact, [], assertion
            )

            # Average of 0.7, 0.8, 0.9 = 0.8
            assert result.checks[0].score == pytest.approx(0.8, abs=0.01)
            assert evaluator.cost.total_calls == 3

    @pytest.mark.anyio
    async def test_averaging_with_some_failures(
        self,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test averaging handles some failed runs."""
        config = LLMJudgeConfig(api_key="test", num_runs=3)
        evaluator = LLMJudgeEvaluator(config)

        call_count = [0]  # Use list to allow mutation in async closure

        async def create_mock_response(*args, **kwargs) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("API Error")
            mock = MagicMock()
            mock.content = [MagicMock(text='{"score": 0.8, "explanation": "Test"}')]
            mock.usage.input_tokens = 100
            mock.usage.output_tokens = 50
            return mock

        with patch.object(evaluator, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = create_mock_response
            mock_get_client.return_value = mock_client

            assertion = Assertion(
                type="llm_eval",
                config={"criteria": "factual_accuracy", "num_runs": 3},
            )
            result = await evaluator.evaluate(
                sample_task, response_with_file_artifact, [], assertion
            )

            # Should succeed with 2 out of 3 runs
            assert result.passed is True


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.anyio
    async def test_handles_api_error(
        self,
        evaluator_with_config: LLMJudgeEvaluator,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test handling of API errors."""
        with patch.object(evaluator_with_config, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
            mock_get_client.return_value = mock_client

            assertion = Assertion(
                type="llm_eval", config={"criteria": "factual_accuracy"}
            )
            result = await evaluator_with_config.evaluate(
                sample_task, response_with_file_artifact, [], assertion
            )
            assert result.passed is False
            assert "failed" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_retries_on_rate_limit(
        self,
        sample_task: TestDefinition,
        response_with_file_artifact: ATPResponse,
    ) -> None:
        """Test retry behavior on rate limit errors."""
        config = LLMJudgeConfig(api_key="test", timeout=1.0)
        evaluator = LLMJudgeEvaluator(config)

        call_count = 0

        async def rate_limit_then_success(*args, **kwargs) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Rate limit exceeded")
            mock = MagicMock()
            mock.content = [MagicMock(text='{"score": 0.8, "explanation": "OK"}')]
            mock.usage.input_tokens = 100
            mock.usage.output_tokens = 50
            return mock

        with patch.object(evaluator, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = rate_limit_then_success
            mock_get_client.return_value = mock_client

            assertion = Assertion(
                type="llm_eval", config={"criteria": "factual_accuracy"}
            )
            result = await evaluator.evaluate(
                sample_task, response_with_file_artifact, [], assertion
            )
            assert result.passed is True
            assert call_count == 3

    def test_missing_anthropic_package(self) -> None:
        """Test error when anthropic package is missing."""
        evaluator = LLMJudgeEvaluator()
        evaluator._client = None  # Reset client

        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(RuntimeError) as exc_info:
                evaluator._get_client()
            assert "anthropic package is required" in str(exc_info.value)


class TestRegistry:
    """Tests for registry integration."""

    def test_llm_judge_in_registry(self) -> None:
        """Test LLMJudgeEvaluator is registered."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        assert registry.is_registered("llm_judge")

    def test_llm_eval_assertion_mapped(self) -> None:
        """Test llm_eval assertion type is mapped."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        assert registry.supports_assertion("llm_eval")

    def test_create_llm_judge_from_registry(self) -> None:
        """Test creating LLMJudgeEvaluator from registry."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        evaluator = registry.create("llm_judge")
        assert isinstance(evaluator, LLMJudgeEvaluator)
