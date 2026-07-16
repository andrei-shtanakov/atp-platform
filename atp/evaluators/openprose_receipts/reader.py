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

from atp.evaluators.openprose_receipts.canonical import content_hash

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


def _receipt_issues(receipt: dict[str, Any], line_no: int) -> list[Issue]:
    """Structural validation of one receipt (in code — no jsonschema)."""
    v = receipt.get("v")
    if v != RECEIPT_VERSION:
        # Contract: consumers MUST refuse unknown versions; no point checking
        # a shape we do not know.
        return [
            Issue(
                code="unknown_version",
                line_no=line_no,
                message=f"unknown receipt version {v!r} at line {line_no} — refusing",
            )
        ]
    issues: list[Issue] = []
    for field in _REQUIRED_FIELDS:
        if field not in receipt:
            issues.append(
                Issue(
                    code="missing_field",
                    line_no=line_no,
                    message=f"missing field '{field}' at line {line_no}",
                )
            )
    if "kind" in receipt and receipt["kind"] not in _KINDS:
        issues.append(
            Issue(
                code="invalid_enum",
                line_no=line_no,
                message=f"invalid kind {receipt['kind']!r} at line {line_no}",
            )
        )
    if "status" in receipt and receipt["status"] not in _STATUSES:
        issues.append(
            Issue(
                code="invalid_enum",
                line_no=line_no,
                message=f"invalid status {receipt['status']!r} at line {line_no}",
            )
        )
    usage = receipt.get("usage")
    if "usage" in receipt and (
        not isinstance(usage, dict) or usage.get("basis") not in _USAGE_BASES
    ):
        issues.append(
            Issue(
                code="invalid_enum",
                line_no=line_no,
                message=f"invalid usage.basis at line {line_no}",
            )
        )
    if "hash_algorithm" in receipt and receipt["hash_algorithm"] != "sha256":
        issues.append(
            Issue(
                code="invalid_enum",
                line_no=line_no,
                message=f"unsupported hash_algorithm at line {line_no}",
            )
        )
    return issues


def verify_run(run_dir: Path) -> VerifyResult:
    """Verify a run directory's ledger: structure → hash → chain → anchor.

    ``ok`` is True iff there are no errors; torn writes (a trailing
    unparseable line, or a manifest trailing the ledger by exactly one
    receipt) degrade to warnings — a crash artifact, not tampering.
    """
    loaded = load_ledger(run_dir / "receipts.jsonl")
    errors = list(loaded.errors)
    warnings = list(loaded.warnings)
    receipts = loaded.receipts
    receipt_count = len(receipts)

    if not receipts and not errors and not warnings:
        errors.append(
            Issue(
                code="empty_ledger",
                message=(
                    "empty ledger: a run always opens with a run_start control receipt"
                ),
            )
        )

    prev_expected: str | None = None
    for line_no, receipt in enumerate(receipts, start=1):
        line_issues = _receipt_issues(receipt, line_no)
        errors.extend(line_issues)
        if not line_issues:
            try:
                recomputed = content_hash(receipt)
            except (TypeError, ValueError):
                errors.append(
                    Issue(
                        code="invalid_number",
                        line_no=line_no,
                        message=(
                            f"non-canonical value at line {line_no} "
                            "(floats/NaN break hash portability)"
                        ),
                    )
                )
            else:
                if recomputed != receipt.get("content_hash"):
                    errors.append(
                        Issue(
                            code="content_hash_mismatch",
                            line_no=line_no,
                            message=f"content_hash mismatch at line {line_no}",
                        )
                    )
        if receipt.get("prev") != prev_expected:
            errors.append(
                Issue(
                    code="chain_break",
                    line_no=line_no,
                    message=f"prev broken at line {line_no}",
                )
            )
        prev_expected = receipt.get("content_hash")

    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        warnings.append(
            Issue(
                code="no_anchor",
                message=(
                    "run.json missing — chain-only verification, "
                    "ledger_head not checked"
                ),
            )
        )
        return VerifyResult(
            ok=not errors,
            receipt_count=receipt_count,
            errors=errors,
            warnings=warnings,
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        manifest = None
    if not isinstance(manifest, dict):
        errors.append(
            Issue(code="invalid_run_json", message="run.json is not a JSON object")
        )
    elif manifest.get("v") != RUN_VERSION:
        errors.append(
            Issue(
                code="unknown_version",
                message=f"unknown run.json version {manifest.get('v')!r} — refusing",
            )
        )
    elif receipts:
        head = manifest.get("ledger_head")
        count = manifest.get("receipt_count")
        last = receipts[-1]
        if head == last.get("content_hash"):
            if count != len(receipts):
                errors.append(
                    Issue(
                        code="receipt_count_mismatch",
                        message=(
                            f"run.json receipt_count={count!r} but the ledger "
                            f"has {len(receipts)} receipts"
                        ),
                    )
                )
        elif (
            head is not None and head == last.get("prev") and count == len(receipts) - 1
        ):
            # Append succeeded, head update did not: the only crash artifact a
            # torn write can produce is trailing by EXACTLY one receipt.
            warnings.append(
                Issue(
                    code="torn_write_manifest",
                    message=(
                        "torn write: manifest trails ledger by exactly one "
                        "receipt; verifying the anchored prefix"
                    ),
                )
            )
            receipt_count = len(receipts) - 1
        else:
            errors.append(
                Issue(
                    code="ledger_head_mismatch",
                    message=(
                        "ledger_head does not match the ledger (a valid but "
                        "unanchored suffix is not a torn write)"
                    ),
                )
            )

    return VerifyResult(
        ok=not errors,
        receipt_count=receipt_count,
        errors=errors,
        warnings=warnings,
    )
