"""Evaluator for the agent-eval-case grading model: critical_check then rubric.

Implements the two assertion types the loader emits:

- ``method_critical_check`` — a binary, model-graded trap. The loader marked the
  assertion ``critical=True``, so a failure hard-gates the test (score 0) via the
  core ScoreAggregator. Here we only decide pass/fail.
- ``method_rubric`` — weighted graded criteria. Each criterion is scored 0..1 and
  combined by its weight into a single graded score (the methodology weighting,
  which ATP's unweighted check-averaging would otherwise lose).

The actual model calls are delegated to the platform's ``LLMJudgeEvaluator`` via
synthetic ``llm_eval`` assertions, so provider/model/cost handling is reused. The
judge is injectable for testing.
"""

from __future__ import annotations

from atp.evaluators.base import EvalCheck, EvalResult, Evaluator
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from atp_method.loader import METHOD_CRITICAL_CHECK, METHOD_RUBRIC

# A critical check is binary; treat the judge's 0..1 score as pass at/above this.
CRITICAL_THRESHOLD = 0.5


class AgentEvalCaseEvaluator(Evaluator):
    """Grade agent-eval-case assertions (critical_check and rubric)."""

    def __init__(self, judge: Evaluator | None = None) -> None:
        """Initialize.

        Args:
            judge: Evaluator used for the underlying model calls. Defaults to the
                platform LLM judge; injectable so tests avoid real API calls.
        """
        self._judge = judge

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "agent_eval_case"

    def _get_judge(self) -> Evaluator:
        if self._judge is None:
            from atp.evaluators.llm_judge import LLMJudgeEvaluator

            self._judge = LLMJudgeEvaluator()
        return self._judge

    def _select_text_artifact(
        self, response: ATPResponse, artifact_name: str | None
    ) -> str | None:
        """Select named artifact content, preserving first-content fallback."""
        artifacts_with_content = [
            artifact
            for artifact in response.artifacts
            if getattr(artifact, "content", None)
        ]
        if artifact_name:
            for artifact in artifacts_with_content:
                if getattr(artifact, "name", None) == artifact_name:
                    return artifact.content
                if getattr(artifact, "path", None) == artifact_name:
                    return artifact.content
        return artifacts_with_content[0].content if artifacts_with_content else None

    async def _judge_score(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        prompt: str,
        threshold: float,
    ) -> EvalCheck | None:
        """Run one model-graded judgement via a synthetic llm_eval assertion.

        No artifact ``path`` is set, so the judge grades the agent's primary
        output (the first response artifact with content). agent-eval-case has no
        output-artifact selector; grading a specific one among several is a
        follow-up that would require extending the case format.
        """
        synthetic = Assertion(
            type="llm_eval", config={"prompt": prompt, "threshold": threshold}
        )
        result = await self._get_judge().evaluate(task, response, trace, synthetic)
        return result.checks[0] if result.checks else None

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Dispatch by assertion type to the critical-check or rubric grader."""
        if assertion.type == METHOD_CRITICAL_CHECK:
            return await self._evaluate_critical(task, response, trace, assertion)
        if assertion.type == METHOD_RUBRIC:
            return await self._evaluate_rubric(task, response, trace, assertion)
        return EvalResult(
            evaluator=self.name,
            checks=[
                EvalCheck(
                    name="unsupported",
                    passed=False,
                    score=0.0,
                    message=f"unsupported assertion type: {assertion.type}",
                )
            ],
        )

    async def _evaluate_critical(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        checker_name = assertion.config.get("checker")
        if checker_name:
            from atp.evaluators.checkers import get_checker

            checker = get_checker(checker_name)
            if checker is None:
                return EvalResult(
                    evaluator=self.name,
                    checks=[
                        EvalCheck(
                            name="critical_check",
                            passed=False,
                            score=0.0,
                            message=f"unknown checker: {checker_name}",
                        )
                    ],
                )
            text = self._select_text_artifact(
                response, assertion.config.get("artifact_name")
            )
            verdict = checker(assertion.config, text)
            message = (
                "malformed findings — cannot verify"
                if verdict.malformed
                else f"recall={verdict.recall} fp={verdict.fp_count}"
            )
            return EvalResult(
                evaluator=self.name,
                checks=[
                    EvalCheck(
                        name="critical_check",
                        passed=verdict.critical_pass,
                        score=1.0 if verdict.critical_pass else 0.0,
                        message=message,
                        details=verdict.model_dump(),
                    )
                ],
            )

        check_text = assertion.config.get("check", "")
        failure_mode = assertion.config.get("expected_failure_mode", "")
        gold = assertion.config.get("gold")
        prompt = (
            "Grade whether the agent output passes ONE binary check. "
            "Return score 1.0 if the check holds, 0.0 if it fails.\n"
            f"Check (must pass): {check_text}\n"
            f"Failure mode to detect: {failure_mode}\n"
        )
        # Give the judge the reference answer the case author supplied, if any.
        if gold:
            prompt += f"Reference (gold): {gold}\n"
        check = await self._judge_score(
            task, response, trace, prompt, CRITICAL_THRESHOLD
        )
        if check is None:
            return EvalResult(
                evaluator=self.name,
                checks=[
                    EvalCheck(
                        name="critical_check",
                        passed=False,
                        score=0.0,
                        message="no judge result",
                    )
                ],
            )
        return EvalResult(
            evaluator=self.name,
            checks=[
                EvalCheck(
                    name="critical_check",
                    passed=check.passed,
                    score=check.score,
                    message=check.message,
                )
            ],
        )

    async def _evaluate_rubric(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        rubric = assertion.config.get("rubric", [])
        if not rubric:
            return EvalResult(
                evaluator=self.name,
                checks=[
                    EvalCheck(
                        name="rubric",
                        passed=True,
                        score=1.0,
                        message="no rubric criteria",
                    )
                ],
            )

        weighted = 0.0
        total_weight = 0.0
        items: list[dict[str, float | str | bool]] = []
        for item in rubric:
            criterion = str(item["criterion"])
            weight = float(item["weight"])
            prompt = (
                "Score how well the agent output satisfies this criterion, "
                "from 0.0 (not at all) to 1.0 (fully).\n"
                f"Criterion: {criterion}\n"
            )
            check = await self._judge_score(task, response, trace, prompt, 0.0)
            score = check.score if check is not None else 0.0
            weighted += weight * score
            total_weight += weight
            entry: dict[str, float | str | bool] = {
                "criterion": criterion,
                "weight": weight,
                "score": score,
            }
            if check is None:
                # Surface a judge failure instead of silently scoring 0.
                entry["judge_failed"] = True
            items.append(entry)

        # Normalize by the actual weight sum: the schema does not enforce
        # weights==1.0 (that lives in the authoring linter), so mis-specified
        # weights must not distort the score via raw clamping.
        score = weighted / total_weight if total_weight > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        # Rubric is graded, not a gate: it always "passes" and contributes its
        # weighted score; the hard gate is the separate critical_check.
        return EvalResult(
            evaluator=self.name,
            checks=[
                EvalCheck(
                    name="rubric",
                    passed=True,
                    score=score,
                    message="weighted rubric score",
                    details={"items": items},
                )
            ],
        )
