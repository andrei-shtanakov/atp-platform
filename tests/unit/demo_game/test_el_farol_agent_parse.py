"""Tests for demo-game el_farol_agent.parse_action helper."""

import sys
from pathlib import Path

# Add demo-game to path so `import agents.el_farol_agent` resolves.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "demo-game"))


def test_parse_action_prefers_slots():
    from agents.el_farol_agent import parse_action

    parsed = {"slots": [1, 2, 3], "action": [4, 5, 6]}
    assert parse_action(parsed, num_slots=16) == [1, 2, 3]


def test_parse_action_falls_back_to_action_key():
    from agents.el_farol_agent import parse_action

    parsed = {"action": [0, 15]}
    assert parse_action(parsed, num_slots=16) == [0, 15]


def test_parse_action_filters_out_of_range():
    from agents.el_farol_agent import parse_action

    parsed = {"slots": [-1, 0, 16, 99, 5]}
    assert parse_action(parsed, num_slots=16) == [0, 5]


def test_parse_action_fallback_on_missing():
    from agents.el_farol_agent import parse_action

    result = parse_action(None, num_slots=16)
    assert result == [4, 5, 6, 7, 8, 9]  # mid=4, range(4,10)
