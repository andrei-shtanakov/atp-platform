"""Deterministic matcher for structured code-review findings (R-07 Phase-1b #1).

Pure functions, no ATP dependencies. Matching keys on a code ANCHOR (a substring of
the offending code, whitespace-normalized) + a synonym set of acceptable rule ids
(case-insensitive) — NOT line numbers, which are too fragile for LLM output.
"""

import json
import re
from typing import Any

from pydantic import BaseModel


class MatchResult(BaseModel):
    """Outcome of matching agent findings against ground truth."""

    critical_pass: bool
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
        key = f"{exp.get('rule_ids', ['?'])[0]}@{exp.get('anchor', '')}"
        if any(_finding_matches_expected(f, exp) for f in findings):
            matched_keys.append(key)
        else:
            missed_keys.append(key)
            if exp.get("severity") == "critical":
                critical_ok = False

    false_positives: list[str] = []
    for mnf in must_not_flag:
        for f in findings:
            if _anchor_overlap(f.get("anchor", ""), mnf.get("anchor", "")):
                fp_key = f"{f.get('rule_id', '?')}@{mnf.get('anchor', '')}"
                false_positives.append(fp_key)
                break

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
