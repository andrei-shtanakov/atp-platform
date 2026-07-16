"""receipt_chain checker: verify an open-prose run directory's ledger.

Selected via ``grader: {type: programmatic, checker: receipt_chain,
config: {run_dir: <relative path>}}``. The agent-output ``text`` is ignored —
the artifact under test is a run *directory*. ``_case_dir`` is dispatch-owned
(injected by AgentEvalCaseEvaluator, never trusted from user config) and is
the confinement root for ``run_dir`` resolution.
"""

from pathlib import Path
from typing import Any

from atp.core.results import CaseVerdict
from atp.evaluators.openprose_receipts.reader import verify_run

RECEIPT_CHAIN_CHECKER_VERSION = "receipt_chain@1"


def receipt_chain_check(config: dict[str, Any], text: str | None) -> CaseVerdict:
    """Verify the hash-chained ledger under config.run_dir (text is ignored)."""
    run_dir_raw = config.get("run_dir")
    if not isinstance(run_dir_raw, str) or not run_dir_raw.strip():
        return _malformed("grader.config.run_dir is required (relative path)")
    run_dir = Path(run_dir_raw)
    if run_dir.is_absolute():
        return _malformed("run_dir must be relative to the case directory")
    case_dir_raw = config.get("_case_dir")
    if not case_dir_raw:
        return _malformed("no case directory available to resolve run_dir")
    case_dir = Path(str(case_dir_raw)).resolve()
    resolved = (case_dir / run_dir).resolve()
    if not resolved.is_relative_to(case_dir):
        return _malformed("run_dir escapes the case directory")
    receipts_path = resolved / "receipts.jsonl"
    if not receipts_path.is_file():
        return _malformed("receipts.jsonl not found under run_dir")
    if _escapes(receipts_path, case_dir):
        return _malformed("receipts.jsonl escapes the case directory (symlink)")
    manifest_path = resolved / "run.json"
    if manifest_path.is_file() and _escapes(manifest_path, case_dir):
        return _malformed("run.json escapes the case directory (symlink)")
    result = verify_run(resolved)
    return CaseVerdict(
        critical_pass=result.ok,
        rubric_score=1.0 if result.ok else 0.0,
        details={
            "errors": [issue.model_dump() for issue in result.errors],
            "warnings": [issue.model_dump() for issue in result.warnings],
            "receipt_count": result.receipt_count,
        },
        grader_version=RECEIPT_CHAIN_CHECKER_VERSION,
    )


def _escapes(path: Path, case_dir: Path) -> bool:
    """True if path, after following any symlinks, resolves outside case_dir."""
    return not path.resolve().is_relative_to(case_dir)


def _malformed(reason: str) -> CaseVerdict:
    """Not gradeable (bad config / missing artifacts) — distinct from ok=False."""
    return CaseVerdict(
        critical_pass=False,
        malformed=True,
        details={"reason": reason},
        grader_version=RECEIPT_CHAIN_CHECKER_VERSION,
    )
