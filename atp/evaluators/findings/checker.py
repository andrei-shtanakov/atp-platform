"""findings_match checker: grade_findings → uniform CaseVerdict (Phase A-1)."""

from typing import Any

from atp.core.results import CaseVerdict
from atp.evaluators.findings.matcher import grade_findings

FINDINGS_CHECKER_VERSION = "findings_match@1"


def findings_check(config: dict[str, Any], text: str | None) -> CaseVerdict:
    """Run the deterministic findings matcher and map it to a CaseVerdict.

    ``config`` carries ``expected_findings`` / ``must_not_flag`` (ground truth)
    and optional ``schema`` (the output contract). ``text`` is the agent output.
    """
    r = grade_findings(
        text,
        config.get("expected_findings", []),
        config.get("must_not_flag", []),
        schema=config.get("schema"),
    )
    return CaseVerdict(
        critical_pass=r.critical_pass,
        malformed=r.malformed,
        recall=r.recall,
        precision=r.precision,
        fp_count=len(r.false_positives),
        rubric_score=0.0,
        details=r.model_dump(),
        grader_version=FINDINGS_CHECKER_VERSION,
    )
