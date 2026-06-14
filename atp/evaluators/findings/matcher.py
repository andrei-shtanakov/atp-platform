"""Deterministic matcher for structured code-review findings (R-07 Phase-1b #1).

Pure functions, no ATP dependencies. Matching keys on a code ANCHOR (a substring of
the offending code, whitespace-normalized) + a synonym set of acceptable rule ids
(case-insensitive) — NOT line numbers, which are too fragile for LLM output.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class Finding(BaseModel):
    """A single structured code-review finding emitted by the agent under test.

    Strict by design (R-07 P3): ``rule_id``/``anchor``/``severity`` are required
    and ``severity`` must be one of the contract's three levels (the prompt
    envelope pins ``critical|major|minor``). Extra keys (``file``, ``line``,
    ``fix``, ...) are ignored. A finding that fails this validation makes the
    *whole* output malformed rather than silently counting as a missed defect —
    a high malformed rate is a real routing fact about the agent, not noise.
    """

    model_config = ConfigDict(extra="ignore")

    rule_id: str
    anchor: str
    severity: Literal["critical", "major", "minor"]


class MatchResult(BaseModel):
    """Outcome of matching agent findings against ground truth."""

    critical_pass: bool
    # Distinct from critical_pass: the output was not a valid findings array
    # (unparseable JSON OR a finding failed strict validation). Unifies the two
    # failure paths so the reporter can aggregate malformed_rate separately from
    # a legitimately missed defect.
    malformed: bool = False
    recall: float
    precision: float
    matched: list[str]
    missed: list[str]
    false_positives: list[str]
    unknown_extras: list[str]


def _norm(s: str) -> str:
    """Whitespace-collapsed, lowercased — the normalization for anchor/id compares."""
    return " ".join(str(s).split()).lower()


def _anchor_overlap(a: str, b: str) -> bool:
    """True if either normalized anchor is a substring of the other (bidirectional:
    the agent may quote a shorter or longer snippet than the ground-truth anchor)."""
    na, nb = _norm(a), _norm(b)
    return bool(na) and bool(nb) and (na in nb or nb in na)


def parse_findings(text: str) -> list[dict[str, Any]] | None:
    """Parse a findings JSON array from the agent's response text, tolerating a
    surrounding markdown code fence. Returns None if it is not a JSON array."""
    if text is None:
        return None
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, list) else None


def validate_findings(parsed: list[Any]) -> list[dict[str, Any]] | None:
    """Strictly validate every parsed finding against :class:`Finding`.

    Returns normalized dicts (``rule_id``/``anchor``/``severity`` only — extras
    truly dropped) if *all* validate, else ``None`` (the output is malformed).
    Normalizing guarantees the declared ``list[dict]`` return regardless of input
    shape, so downstream ``.get`` access in :func:`match_findings` is safe.
    Strict-global: one bad finding malforms the whole output — there is no
    lenient drop-and-continue mode, because a high malformed rate is a signal
    about the agent, not noise to be filtered out.
    """
    validated: list[dict[str, Any]] = []
    for item in parsed:
        try:
            validated.append(Finding.model_validate(item).model_dump())
        except ValidationError:
            return None
    return validated


def grade_findings(
    text: str | None,
    expected: list[dict[str, Any]],
    must_not_flag: list[dict[str, Any]],
) -> MatchResult:
    """Parse, strictly validate, and match agent output in a single pass.

    Collapses the two failure modes into one :class:`MatchResult`:

    - ``malformed=True`` (with ``critical_pass=False``) when the output is not a
      JSON array of valid :class:`Finding` objects — whether it failed to parse
      *or* a finding failed strict validation. This is distinct from a
      legitimately missed defect (``malformed=False, critical_pass=False``).
    - otherwise the usual recall/precision/critical_pass from
      :func:`match_findings`.
    """
    parsed = parse_findings(text) if text is not None else None
    if parsed is None:
        return _malformed_result()
    findings = validate_findings(parsed)
    if findings is None:
        return _malformed_result()
    return match_findings(findings, expected, must_not_flag)


def _malformed_result() -> MatchResult:
    """A :class:`MatchResult` for output that is not a valid findings array."""
    return MatchResult(
        critical_pass=False,
        malformed=True,
        recall=0.0,
        precision=0.0,
        matched=[],
        missed=[],
        false_positives=[],
        unknown_extras=[],
    )


def _finding_matches_expected(
    finding: dict[str, Any], expected: dict[str, Any]
) -> bool:
    rule = _norm(finding.get("rule_id", ""))
    syns = {_norm(r) for r in expected.get("rule_ids", [])}
    if syns and rule not in syns:
        return False
    return _anchor_overlap(finding.get("anchor", ""), expected.get("anchor", ""))


def match_findings(
    findings: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    must_not_flag: list[dict[str, Any]],
) -> MatchResult:
    """Match findings against expected (planted defects) + must_not_flag (compliant
    lines). critical_pass = every critical-severity expected matched AND zero
    must_not_flag hits."""
    matched_keys: list[str] = []
    missed_keys: list[str] = []
    critical_ok = True
    for exp in expected:
        # `or ["?"]` guards an empty rule_ids list (the matcher is a pure function
        # callable with raw dicts where the schema's min_length=1 does not apply).
        key = f"{(exp.get('rule_ids') or ['?'])[0]}@{exp.get('anchor', '')}"
        if any(_finding_matches_expected(f, exp) for f in findings):
            matched_keys.append(key)
        else:
            missed_keys.append(key)
            if exp.get("severity") == "critical":
                critical_ok = False

    # Count false positives PER FINDING: each finding that hits any must_not_flag
    # anchor is one false positive (multiple findings on the same compliant line
    # each count, so precision is not inflated).
    false_positives: list[str] = []
    for f in findings:
        hit = next(
            (
                m
                for m in must_not_flag
                if _anchor_overlap(f.get("anchor", ""), m.get("anchor", ""))
            ),
            None,
        )
        if hit is not None:
            false_positives.append(f"{f.get('rule_id', '?')}@{hit.get('anchor', '')}")

    unknown_extras = [
        f.get("anchor", "")
        for f in findings
        if not any(_finding_matches_expected(f, e) for e in expected)
        and not any(
            _anchor_overlap(f.get("anchor", ""), m.get("anchor", ""))
            for m in must_not_flag
        )
    ]

    tp = len(matched_keys)
    fp = len(false_positives)
    recall = 1.0 if not expected else tp / len(expected)
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    critical_pass = critical_ok and fp == 0
    return MatchResult(
        critical_pass=critical_pass,
        recall=round(recall, 6),
        precision=round(precision, 6),
        matched=matched_keys,
        missed=missed_keys,
        false_positives=false_positives,
        unknown_extras=unknown_extras,
    )
