"""Tests for demo-game el_farol_agent.parse_action helper."""

import sys
from pathlib import Path

# Add demo-game to path so `import agents.el_farol_agent` resolves.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "demo-game"))


def test_parse_action_returns_intervals():
    from agents.el_farol_agent import parse_action

    parsed = {"intervals": [[3, 7]]}
    assert parse_action(parsed, num_slots=16) == [[3, 7]]


def test_parse_action_clamps_out_of_range_bounds():
    from agents.el_farol_agent import parse_action

    parsed = {"intervals": [[10, 99]]}
    # End is clamped down to num_slots - 1.
    assert parse_action(parsed, num_slots=16) == [[10, 15]]


def test_parse_action_clamps_negative_start():
    from agents.el_farol_agent import parse_action

    parsed = {"intervals": [[-3, 5]]}
    assert parse_action(parsed, num_slots=16) == [[0, 5]]


def test_parse_action_drops_malformed_pairs():
    from agents.el_farol_agent import parse_action

    parsed = {
        "intervals": [
            "not-a-pair",
            [1, 2, 3],  # wrong length
            [1, "x"],  # non-int
            [4, 6],  # valid
        ]
    }
    assert parse_action(parsed, num_slots=16) == [[4, 6]]


def test_parse_action_falls_back_when_intervals_key_missing():
    from agents.el_farol_agent import parse_action

    parsed: dict = {"reasoning": "no plan"}
    # Midday fallback: mid = num_slots // 4 = 4; end = min(mid+5, 15) = 9.
    assert parse_action(parsed, num_slots=16) == [[4, 9]]


def test_parse_action_falls_back_when_parsed_is_none():
    from agents.el_farol_agent import parse_action

    assert parse_action(None, num_slots=16) == [[4, 9]]
