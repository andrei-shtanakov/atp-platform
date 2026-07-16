"""Load and verify openprose.receipt.v1 ledgers: chain consistency + anchor.

Semantics follow the vendored contract (method/contract/openprose/receipt.md)
and the design spec (docs/superpowers/specs/
2026-07-16-openprose-receipts-evaluation-input-design.md). ``load_ledger``
never raises on bad content — it accumulates line-level issues and returns the
parseable prefix, because a trailing torn line must degrade to a warning while
the prefix still verifies.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

RECEIPT_VERSION = "openprose.receipt.v1"
RUN_VERSION = "openprose.run.v1"

_KINDS = {"session", "parallel_branch", "block_call", "discretion", "control"}
_STATUSES = {"rendered", "skipped", "failed"}
_USAGE_BASES = {"exact", "estimated", "unavailable"}
# Nullable metadata fields (agent, surprise_cause, error, detail, reused_from)
# are not required: consumers must tolerate their absence, per append-frozen
# "ignore unknown, require only what you consume".
_REQUIRED_FIELDS = (
    "v",
    "run_id",
    "statement_id",
    "kind",
    "input_fingerprints",
    "output_fingerprint",
    "status",
    "usage",
    "prev",
    "hash_algorithm",
    "content_hash",
)


class Issue(BaseModel):
    """One verification finding, stable-coded for programmatic consumers."""

    code: str
    line_no: int | None = None  # 1-based ledger line; None for run-level issues
    message: str


class LoadedLedger(BaseModel):
    """Parsed prefix of a ledger plus line-level parse issues."""

    receipts: list[dict[str, Any]]
    errors: list[Issue]
    warnings: list[Issue]


class VerifyResult(BaseModel):
    """Outcome of ``verify_run``: ok iff there are no errors (warnings allowed)."""

    ok: bool
    # Receipts covered by verification — NOT physical line count: the parsed
    # prefix on a torn line, the anchored prefix on a torn manifest.
    receipt_count: int = Field(ge=0)
    errors: list[Issue]
    warnings: list[Issue]


def load_ledger(path: Path) -> LoadedLedger:
    """Parse receipts.jsonl into its valid prefix + issues (never raises)."""
    receipts: list[dict[str, Any]] = []
    errors: list[Issue] = []
    warnings: list[Issue] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    for line_no, line in enumerate(lines, start=1):
        try:
            parsed = json.loads(line)
        except ValueError:
            if line_no == len(lines):
                warnings.append(
                    Issue(
                        code="torn_write_line",
                        line_no=line_no,
                        message=(
                            "torn write: final ledger line is not valid JSON; "
                            f"verifying the {len(receipts)}-receipt prefix"
                        ),
                    )
                )
            else:
                errors.append(
                    Issue(
                        code="invalid_json",
                        line_no=line_no,
                        message=(
                            f"invalid JSON on non-final line {line_no}; the chain "
                            f"is broken here — stopping at the "
                            f"{len(receipts)}-receipt prefix"
                        ),
                    )
                )
            break
        if not isinstance(parsed, dict):
            errors.append(
                Issue(
                    code="invalid_json",
                    line_no=line_no,
                    message=f"line {line_no} is not a JSON object",
                )
            )
            break
        receipts.append(parsed)
    return LoadedLedger(receipts=receipts, errors=errors, warnings=warnings)


def verify_run(run_dir: Path) -> VerifyResult:
    """Verify a run directory: ledger chain consistency + manifest anchor."""
    raise NotImplementedError
