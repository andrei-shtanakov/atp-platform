#!/usr/bin/env python3
"""Deterministic extraction-quality score (no LLM).

Compares the agent's extracted requirements (strict JSON) against a ground-truth
JSON and prints a machine-parseable ratio:

    SCORE passed=<N> total=<M>

where M = number of ground-truth requirements and N = how many the agent got
"right". A requirement counts as right when, at the same position, the agent's
`actor` matches (normalized) AND a non-empty `obligation` is present AND the
`deadline` matches the ground truth's absent/present status. Matching is
positional (the task asks for source order), which keeps the score fully
deterministic — no fuzzy text matching, no model.

The ATP `custom_command` evaluator reads the ratio via
`pattern: "passed=(?P<passed>\\d+) total=(?P<total>\\d+)"` and uses passed/total
as the check score (0..1).

Usage: score_extraction.py <agent_json> <ground_truth_json>
"""

import json
import re
import sys

_ABSENT = {"", "null", "none", "n/a", "na", "not stated", "not specified", "-", "—"}


def _extract_json(text: str) -> object:
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no parseable JSON found")


def _reqs(data: object) -> list[dict]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("requirements", [data])
    else:
        # A JSON scalar (string/number/bool) carries no requirements; score it
        # as zero rather than crashing on .get().
        return []
    if not isinstance(items, list):
        # e.g. {"requirements": 5} — a non-list value isn't iterable.
        return []
    return [i for i in items if isinstance(i, dict)]


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def _norm_actor(value: object) -> str:
    """Normalize an actor for comparison: lowercase + drop a leading article.

    Models phrase the same actor as "vendor" or "the vendor"; treating those as
    different would make a correct extraction score 0. Stripping the article is a
    deliberate, documented tolerance — the deterministic grader's rules are
    explicit, unlike an LLM judge's.
    """
    text = _norm(value)
    for article in ("the ", "a ", "an "):
        if text.startswith(article):
            return text[len(article) :]
    return text


def _absent(value: object) -> bool:
    return value is None or _norm(value) in _ABSENT


def main() -> int:
    try:
        with open(sys.argv[1], encoding="utf-8") as fh:
            agent = _reqs(_extract_json(fh.read()))
        with open(sys.argv[2], encoding="utf-8") as fh:
            truth = _reqs(_extract_json(fh.read()))
    except (OSError, IndexError, ValueError) as e:
        # Unreadable output scores zero, not a crash.
        print(f"SCORE passed=0 total=1  ({e})")
        return 0

    total = len(truth)
    passed = 0
    for i, gt in enumerate(truth):
        if i >= len(agent):
            continue  # requirement not extracted
        got = agent[i]
        actor_ok = _norm_actor(got.get("actor")) == _norm_actor(gt.get("actor"))
        obligation_ok = bool(_norm(got.get("obligation")))
        deadline_ok = _absent(got.get("deadline")) == _absent(gt.get("deadline"))
        if actor_ok and obligation_ok and deadline_ok:
            passed += 1

    print(f"SCORE passed={passed} total={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
