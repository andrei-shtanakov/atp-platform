"""LLM-as-Judge evaluator for semantic evaluation using Anthropic API."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from .base import EvalCheck, EvalResult, Evaluator

if TYPE_CHECKING:
    from atp.analytics.cost import CostTracker

logger = logging.getLogger(__name__)


# Built-in criteria with descriptions
BUILTIN_CRITERIA: dict[str, str] = {
    "factual_accuracy": (
        "Evaluate the factual accuracy of the content. "
        "Check for correct facts, statistics, dates, and verifiable claims. "
        "Score 1.0 if all facts are accurate, 0.0 if mostly inaccurate."
    ),
    "completeness": (
        "Evaluate whether the response completely addresses all aspects of the task. "
        "Check if any required information is missing or skipped. "
        "Score 1.0 if fully complete, 0.0 if major parts are missing."
    ),
    "relevance": (
        "Evaluate how relevant the content is to the given task. "
        "Check if the response stays on topic and addresses the actual request. "
        "Score 1.0 if highly relevant, 0.0 if off-topic."
    ),
    "coherence": (
        "Evaluate the logical flow and coherence of the content. "
        "Check for clear structure, logical connections, and consistent reasoning. "
        "Score 1.0 if perfectly coherent, 0.0 if incoherent or contradictory."
    ),
    "clarity": (
        "Evaluate the clarity of the content. "
        "Check if the writing is clear, understandable, and well-organized. "
        "Score 1.0 if crystal clear, 0.0 if confusing or poorly written."
    ),
    "actionability": (
        "Evaluate the practical applicability of the content. "
        "Check if the response provides actionable guidance or usable information. "
        "Score 1.0 if highly actionable, 0.0 if not actionable at all."
    ),
}


class LLMJudgeResponse(BaseModel):
    """Parsed response from LLM judge."""

    score: float = Field(..., ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    explanation: str = Field(..., description="Reasoning for the score")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    strengths: list[str] = Field(
        default_factory=list, description="List of strengths found"
    )


class LLMJudgeCost(BaseModel):
    """Cost tracking for LLM judge evaluations."""

    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    total_calls: int = Field(default=0, description="Total API calls made")
    model: str = Field(default="", description="Model used for evaluation")

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cost in USD based on token counts.

        Uses approximate pricing (may vary by model).
        Claude 3.5 Sonnet pricing: $3/M input, $15/M output
        """
        input_cost = (self.input_tokens / 1_000_000) * 3.0
        output_cost = (self.output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost


class LLMJudgeConfig(BaseModel):
    """Configuration for LLM Judge evaluator."""

    api_key: str | None = Field(None, description="Anthropic API key")
    model: str = Field(
        "claude-sonnet-4-20250514", description="Model to use for evaluation"
    )
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Model temperature")
    max_tokens: int = Field(1024, ge=1, description="Max tokens for response")
    num_runs: int = Field(1, ge=1, le=10, description="Number of evaluation runs")
    timeout: float = Field(60.0, gt=0, description="Timeout per request in seconds")
    enable_cost_tracking: bool = Field(
        True, description="Enable cost tracking via CostTracker"
    )


class LLMJudgeEvaluator(Evaluator):
    """
    LLM-as-Judge evaluator for semantic evaluation.

    Uses Anthropic's Claude to evaluate agent outputs against various criteria.
    Supports built-in criteria (factual_accuracy, completeness, etc.) and
    custom prompts for specialized evaluation needs.

    Features:
    - Built-in criteria prompts for common evaluation needs
    - Custom prompt support for specialized evaluations
    - Score parsing with explanation extraction
    - Multi-call averaging for more stable results
    - Cost tracking for API usage
    - Error handling with rate limit retry
    """

    def __init__(
        self,
        config: LLMJudgeConfig | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Initialize the LLM Judge evaluator.

        Args:
            config: Optional configuration. If not provided, defaults are used.
            cost_tracker: Optional CostTracker instance. If not provided and
                enable_cost_tracking is True, uses the global tracker.
        """
        self._config = config or LLMJudgeConfig()
        self._client: Any = None
        self._total_cost = LLMJudgeCost(model=self._config.model)
        self._cost_tracker = cost_tracker
        self._test_id: str | None = None
        self._suite_id: str | None = None
        self._agent_name: str | None = None

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "llm_judge"

    @property
    def cost(self) -> LLMJudgeCost:
        """Return cumulative cost tracking."""
        return self._total_cost

    def _get_client(self) -> Any:
        """Get or create Anthropic client (lazy initialization)."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "anthropic package is required for LLMJudgeEvaluator. "
                    "Install it with: uv add anthropic"
                ) from e

            if self._config.api_key:
                self._client = anthropic.AsyncAnthropic(api_key=self._config.api_key)
            else:
                # Uses ANTHROPIC_API_KEY env var by default
                self._client = anthropic.AsyncAnthropic()

        return self._client

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """
        Evaluate agent results using LLM-as-judge.

        Args:
            task: Test definition containing task details.
            response: ATP Response from the agent.
            trace: List of ATP Events (unused for LLM evaluation).
            assertion: Assertion with criteria/prompt configuration.

        Returns:
            EvalResult containing check results.
        """
        # Store context for cost tracking
        self._test_id = task.id
        self._suite_id = getattr(task, "suite_id", None)
        self._agent_name = getattr(task, "agent", None)

        config = assertion.config
        criteria = config.get("criteria")
        custom_prompt = config.get("prompt")
        threshold = config.get("threshold", 0.7)
        artifact_path = config.get("path")
        num_runs = config.get("num_runs", self._config.num_runs)

        # Get artifact content
        artifact_content = self._get_artifact_content(response, artifact_path)
        if artifact_content is None:
            return self._create_result(
                [
                    self._create_check(
                        name="llm_eval",
                        passed=False,
                        message="No artifact content found for evaluation",
                        details={"path": artifact_path},
                    )
                ]
            )

        # Build evaluation prompt
        try:
            eval_prompt = self._build_prompt(
                task_description=task.task.description,
                artifact_content=artifact_content,
                criteria=criteria,
                custom_prompt=custom_prompt,
            )
        except ValueError as e:
            return self._create_result(
                [
                    self._create_check(
                        name="llm_eval",
                        passed=False,
                        message=str(e),
                        details={"criteria": criteria},
                    )
                ]
            )

        # Run evaluation (possibly multiple times for averaging)
        try:
            if num_runs > 1:
                judge_response = await self._evaluate_with_averaging(
                    eval_prompt, num_runs
                )
            else:
                judge_response = await self._call_llm(eval_prompt)
        except Exception as e:
            return self._create_result(
                [
                    self._create_check(
                        name="llm_eval",
                        passed=False,
                        message=f"LLM evaluation failed: {e}",
                        details={"error": str(e), "criteria": criteria},
                    )
                ]
            )

        # Determine pass/fail based on threshold
        passed = judge_response.score >= threshold

        return self._create_result(
            [
                EvalCheck(
                    name=f"llm_eval:{criteria or 'custom'}",
                    passed=passed,
                    score=judge_response.score,
                    message=judge_response.explanation,
                    details={
                        "criteria": criteria,
                        "threshold": threshold,
                        "score": judge_response.score,
                        "issues": judge_response.issues,
                        "strengths": judge_response.strengths,
                        "cost": {
                            "input_tokens": self._total_cost.input_tokens,
                            "output_tokens": self._total_cost.output_tokens,
                            "estimated_usd": self._total_cost.estimated_cost_usd,
                        },
                    },
                )
            ]
        )

    def _get_artifact_content(
        self, response: ATPResponse, path: str | None = None
    ) -> str | None:
        """Extract artifact content for evaluation."""
        target_artifacts = response.artifacts
        if path:
            target_artifacts = [
                a
                for a in response.artifacts
                if (getattr(a, "path", None) or getattr(a, "name", None)) == path
            ]

        for artifact in target_artifacts:
            if hasattr(artifact, "content") and artifact.content:
                return artifact.content
            if hasattr(artifact, "data") and artifact.data:
                return json.dumps(artifact.data, indent=2)

        return None

    def _build_prompt(
        self,
        task_description: str,
        artifact_content: str,
        criteria: str | None = None,
        custom_prompt: str | None = None,
    ) -> str:
        """Build the evaluation prompt.

        Args:
            task_description: The original task description.
            artifact_content: Content of the artifact to evaluate.
            criteria: Built-in criteria name (optional).
            custom_prompt: Custom evaluation prompt (optional).

        Returns:
            Complete prompt for LLM evaluation.

        Raises:
            ValueError: If criteria is invalid or neither criteria nor custom_prompt.
        """
        if not criteria and not custom_prompt:
            raise ValueError(
                "Either 'criteria' or 'prompt' must be specified in assertion config"
            )

        if criteria:
            if criteria not in BUILTIN_CRITERIA:
                available = ", ".join(BUILTIN_CRITERIA.keys())
                raise ValueError(
                    f"Unknown criteria: {criteria}. Available: {available}"
                )
            criteria_description = BUILTIN_CRITERIA[criteria]
        else:
            criteria_description = ""

        # Truncate artifact content if too long
        max_content_length = 50000
        if len(artifact_content) > max_content_length:
            artifact_content = (
                artifact_content[:max_content_length]
                + f"\n\n[Content truncated at {max_content_length} characters]"
            )

        prompt = f"""You are evaluating AI agent output.

TASK:
{task_description}

ARTIFACT:
{artifact_content}

"""

        if criteria:
            prompt += f"""CRITERION: {criteria}
{criteria_description}

"""

        if custom_prompt:
            prompt += f"""{custom_prompt}

"""

        prompt += """Respond ONLY with a valid JSON object in this exact format:
{
  "score": <0.0-1.0>,
  "explanation": "<your reasoning>",
  "issues": ["<list of issues found>"],
  "strengths": ["<list of strengths found>"]
}

Important:
- Score must be a number between 0.0 and 1.0
- Provide clear, specific explanation for your score
- List concrete issues and strengths found in the artifact
- Your response must be valid JSON only, no other text"""

        return prompt

    async def _call_llm(self, prompt: str) -> LLMJudgeResponse:
        """Call the LLM API and parse the response.

        Args:
            prompt: The evaluation prompt.

        Returns:
            Parsed LLMJudgeResponse.

        Raises:
            Exception: If API call fails or response parsing fails.
        """
        client = self._get_client()

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=self._config.model,
                        max_tokens=self._config.max_tokens,
                        temperature=self._config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=self._config.timeout,
                )

                # Update local cost tracking
                self._total_cost.input_tokens += response.usage.input_tokens
                self._total_cost.output_tokens += response.usage.output_tokens
                self._total_cost.total_calls += 1

                # Track cost via CostTracker if enabled
                if self._config.enable_cost_tracking:
                    await self._track_cost(
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    )

                # Parse response
                return self._parse_response(response.content[0].text)

            except TimeoutError:
                if attempt == max_retries - 1:
                    raise TimeoutError(
                        f"LLM call timed out after {self._config.timeout}s"
                    )
                await asyncio.sleep(base_delay * (2**attempt))

            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit errors
                if "rate" in error_str and "limit" in error_str:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2**attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise

        raise RuntimeError("Unexpected: exceeded max retries without returning")

    async def _track_cost(self, input_tokens: int, output_tokens: int) -> None:
        """Track cost via CostTracker.

        Args:
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens used.
        """
        try:
            tracker = self._cost_tracker
            if tracker is None:
                from atp.analytics.cost import get_cost_tracker

                tracker = await get_cost_tracker()

            from datetime import datetime

            from atp.analytics.cost import CostEvent

            await tracker.track(
                CostEvent(
                    timestamp=datetime.now(),
                    provider="anthropic",
                    model=self._config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    test_id=self._test_id,
                    suite_id=self._suite_id,
                    agent_name=self._agent_name,
                    metadata={"evaluator": "llm_judge"},
                )
            )
        except Exception as e:
            logger.warning(f"Failed to track cost: {e}")

    def _parse_response(self, text: str) -> LLMJudgeResponse:
        """Parse LLM response text into structured format.

        Args:
            text: Raw text response from LLM.

        Returns:
            Parsed LLMJudgeResponse.

        Raises:
            ValueError: If response cannot be parsed.
        """
        # Try to extract JSON from response
        text = text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            # Remove code block markers
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Try direct JSON parsing
        try:
            data = json.loads(text)
            return self._validate_parsed_response(data)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_parsed_response(data)
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract score from text
        score_match = re.search(r'"?score"?\s*[=:]\s*([0-9.]+)', text, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            return LLMJudgeResponse(
                score=min(max(score, 0.0), 1.0),
                explanation=f"Extracted from partial response: {text[:200]}...",
                issues=[],
                strengths=[],
            )

        raise ValueError(f"Could not parse LLM response as JSON: {text[:200]}...")

    def _validate_parsed_response(self, data: dict[str, Any]) -> LLMJudgeResponse:
        """Validate and normalize parsed response data.

        Args:
            data: Parsed JSON data.

        Returns:
            Validated LLMJudgeResponse.
        """
        score = data.get("score", 0.5)
        if isinstance(score, str):
            score = float(score)
        score = min(max(score, 0.0), 1.0)

        explanation = data.get("explanation", "No explanation provided")
        if not isinstance(explanation, str):
            explanation = str(explanation)

        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []

        strengths = data.get("strengths", [])
        if not isinstance(strengths, list):
            strengths = [str(strengths)] if strengths else []

        return LLMJudgeResponse(
            score=score,
            explanation=explanation,
            issues=[str(i) for i in issues],
            strengths=[str(s) for s in strengths],
        )

    async def _evaluate_with_averaging(
        self, prompt: str, num_runs: int
    ) -> LLMJudgeResponse:
        """Run multiple evaluations and average the results.

        Args:
            prompt: The evaluation prompt.
            num_runs: Number of runs to perform.

        Returns:
            Averaged LLMJudgeResponse.
        """
        responses: list[LLMJudgeResponse] = []
        errors: list[str] = []

        for _ in range(num_runs):
            try:
                response = await self._call_llm(prompt)
                responses.append(response)
            except Exception as e:
                errors.append(str(e))

        if not responses:
            raise RuntimeError(
                f"All {num_runs} evaluation runs failed. Errors: {errors}"
            )

        # Average the scores
        avg_score = sum(r.score for r in responses) / len(responses)

        # Collect all unique issues and strengths
        all_issues: list[str] = []
        all_strengths: list[str] = []
        for r in responses:
            all_issues.extend(r.issues)
            all_strengths.extend(r.strengths)

        # Deduplicate
        unique_issues = list(dict.fromkeys(all_issues))
        unique_strengths = list(dict.fromkeys(all_strengths))

        # Build combined explanation
        if len(responses) == 1:
            explanation = responses[0].explanation
        else:
            explanation = (
                f"Averaged score from {len(responses)} evaluations. "
                f"Individual scores: {[r.score for r in responses]}. "
                f"Sample explanation: {responses[0].explanation}"
            )

        return LLMJudgeResponse(
            score=avg_score,
            explanation=explanation,
            issues=unique_issues,
            strengths=unique_strengths,
        )

    def reset_cost_tracking(self) -> None:
        """Reset the cumulative cost tracking."""
        self._total_cost = LLMJudgeCost(model=self._config.model)

    @classmethod
    def get_available_criteria(cls) -> dict[str, str]:
        """Get available built-in criteria and their descriptions.

        Returns:
            Dictionary mapping criteria names to descriptions.
        """
        return BUILTIN_CRITERIA.copy()
