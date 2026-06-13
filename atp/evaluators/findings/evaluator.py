"""FindingsMatchEvaluator — ATP evaluator for structured code-review findings."""

from typing import Any

from atp.core.results import EvalResult
from atp.evaluators.base import Evaluator
from atp.evaluators.findings.matcher import match_findings, parse_findings
from atp.loader.models import Assertion, TestDefinition
from atp.protocol.models import ATPEvent, ATPResponse


class FindingsMatchEvaluator(Evaluator):
    """Evaluate agent findings against expected defects via deterministic matching."""

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "findings_match"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Match agent findings against expected defects and must-not-flag anchors.

        Reads the first artifact with non-empty content, parses it as a JSON
        findings array, then delegates to :func:`match_findings`. Returns a
        single :class:`EvalCheck` summarising recall / precision / missed /
        false-positives.
        """
        cfg: dict[str, Any] = assertion.config

        content: str | None = next(
            (a.content for a in response.artifacts if getattr(a, "content", None)),
            None,
        )

        if content is None:
            check = self._create_check(
                name="findings_match",
                passed=False,
                message="unparseable findings — cannot verify (expected a JSON array)",
                details={"recall": 0.0, "precision": 0.0},
            )
            return EvalResult(
                evaluator=self.name,
                checks=[check],
                critical=assertion.critical,
            )

        findings = parse_findings(content)
        if findings is None:
            check = self._create_check(
                name="findings_match",
                passed=False,
                message="unparseable findings — cannot verify (expected a JSON array)",
                details={"recall": 0.0, "precision": 0.0},
            )
            return EvalResult(
                evaluator=self.name,
                checks=[check],
                critical=assertion.critical,
            )

        r = match_findings(
            findings,
            cfg.get("expected_findings", []),
            cfg.get("must_not_flag", []),
        )

        missed_str = ", ".join(r.missed) if r.missed else "none"
        fp_str = ", ".join(r.false_positives) if r.false_positives else "none"
        message = (
            f"recall={r.recall:.3f} precision={r.precision:.3f} "
            f"matched={len(r.matched)} missed=[{missed_str}] "
            f"false_positives=[{fp_str}]"
        )

        check = self._create_check(
            name="findings_match",
            passed=r.critical_pass,
            message=message,
            details=r.model_dump(),
        )
        return EvalResult(
            evaluator=self.name,
            checks=[check],
            critical=assertion.critical,
        )
