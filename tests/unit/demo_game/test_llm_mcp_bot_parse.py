"""Tests for llm_mcp_bot parser, prompt builder, and random fallback.

Pure-function tests only; no network or MCP adapter is exercised.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "demo-game"))


def _pd_state() -> dict:
    return {
        "game_type": "prisoners_dilemma",
        "round_number": 3,
        "total_rounds": 10,
        "your_history": ["cooperate"],
        "opponent_history": ["defect"],
        "your_cumulative_score": 1.0,
        "your_turn": True,
    }


def _sh_state() -> dict:
    return {
        "game_type": "stag_hunt",
        "round_number": 1,
        "total_rounds": 5,
        "your_history": [],
        "opponent_history": [],
        "your_turn": True,
    }


def _bos_state(preferred: str = "B") -> dict:
    return {
        "game_type": "battle_of_sexes",
        "round_number": 1,
        "total_rounds": 5,
        "your_history": [],
        "opponent_history": [],
        "your_preferred": preferred,
        "your_turn": True,
    }


def _el_farol_state(num_slots: int = 16, max_per_day: int = 8) -> dict:
    return {
        "game_type": "el_farol",
        "round_number": 1,
        "total_rounds": 5,
        "your_history": [],
        "num_slots": num_slots,
        "action_schema": {"max_length": max_per_day},
        "pending_submission": True,
    }


# ---- build_user_prompt ----


def test_prompt_pd_includes_history_from_wire_fields():
    from bots.llm_prompts import build_user_prompt

    state = _pd_state()
    state["your_history"] = ["cooperate", "defect"]
    state["opponent_history"] = ["defect", "defect"]
    prompt = build_user_prompt(state)
    # Wire schema uses parallel your_history + opponent_history lists;
    # the summary must surface them so the LLM has match context.
    assert "you=cooperate opp=defect" in prompt
    assert "you=defect opp=defect" in prompt


def test_prompt_pd_mentions_cooperate_and_defect():
    from bots.llm_prompts import build_user_prompt

    prompt = build_user_prompt(_pd_state())
    assert "cooperate" in prompt.lower()
    assert "defect" in prompt.lower()


def test_prompt_bos_surfaces_your_preferred():
    from bots.llm_prompts import build_user_prompt

    prompt = build_user_prompt(_bos_state("B"))
    # Bot must not assume default "A" — prompt must tell the LLM which
    # outcome this participant prefers so it can play the asymmetric game.
    assert "B" in prompt
    assert "preferred" in prompt.lower() or "prefer" in prompt.lower()


def test_prompt_el_farol_mentions_slot_bounds():
    from bots.llm_prompts import build_user_prompt

    prompt = build_user_prompt(_el_farol_state(num_slots=16, max_per_day=8))
    assert "16" in prompt
    assert "8" in prompt


# ---- parse_llm_response ----


def test_parse_pd_happy():
    from bots.llm_prompts import parse_llm_response

    raw = '{"action": "cooperate", "reasoning": "start nice"}'
    out = parse_llm_response(raw, _pd_state())
    assert out == {"choice": "cooperate", "reasoning": "start nice"}


def test_parse_sh_happy():
    from bots.llm_prompts import parse_llm_response

    raw = '{"action": "stag", "reasoning": "risk-dominant coordination"}'
    out = parse_llm_response(raw, _sh_state())
    assert out == {"choice": "stag", "reasoning": "risk-dominant coordination"}


def test_parse_bos_accepts_either_letter():
    from bots.llm_prompts import parse_llm_response

    raw = '{"action": "A", "reasoning": "yielding to partner"}'
    out = parse_llm_response(raw, _bos_state("B"))
    assert out == {"choice": "A", "reasoning": "yielding to partner"}


def test_parse_el_farol_happy_slots_key():
    from bots.llm_prompts import parse_llm_response

    raw = '{"slots": [0, 3, 7], "reasoning": "off-peak"}'
    out = parse_llm_response(raw, _el_farol_state())
    assert out == {"slots": [0, 3, 7], "reasoning": "off-peak"}


def test_parse_json_inside_markdown_fence():
    from bots.llm_prompts import parse_llm_response

    raw = '```json\n{"action": "defect", "reasoning": "retaliating"}\n```'
    out = parse_llm_response(raw, _pd_state())
    assert out == {"choice": "defect", "reasoning": "retaliating"}


def test_parse_malformed_json_returns_none():
    from bots.llm_prompts import parse_llm_response

    assert parse_llm_response("not json at all", _pd_state()) is None


def test_parse_invalid_action_returns_none():
    from bots.llm_prompts import parse_llm_response

    raw = '{"action": "banana", "reasoning": "yolo"}'
    assert parse_llm_response(raw, _pd_state()) is None


def test_parse_missing_action_returns_none():
    from bots.llm_prompts import parse_llm_response

    raw = '{"reasoning": "I forgot to say what I pick"}'
    assert parse_llm_response(raw, _pd_state()) is None


def test_parse_el_farol_rejects_out_of_range_slot():
    from bots.llm_prompts import parse_llm_response

    raw = '{"slots": [0, 99], "reasoning": "bad slot"}'
    assert parse_llm_response(raw, _el_farol_state(num_slots=16)) is None


def test_parse_el_farol_rejects_duplicate_slots():
    from bots.llm_prompts import parse_llm_response

    raw = '{"slots": [0, 0, 3], "reasoning": "dup"}'
    assert parse_llm_response(raw, _el_farol_state()) is None


def test_parse_el_farol_rejects_too_many_slots():
    from bots.llm_prompts import parse_llm_response

    # max_per_day=8 → 9 slots must be rejected
    raw = '{"slots": [0,1,2,3,4,5,6,7,8], "reasoning": "greedy"}'
    assert parse_llm_response(raw, _el_farol_state(max_per_day=8)) is None


def test_parse_truncates_long_reasoning():
    from bots.llm_prompts import REASONING_MAX_CHARS, parse_llm_response

    long_reason = "x" * (REASONING_MAX_CHARS + 500)
    raw = f'{{"action": "cooperate", "reasoning": "{long_reason}"}}'
    out = parse_llm_response(raw, _pd_state())
    assert out is not None
    assert out["choice"] == "cooperate"
    assert len(out["reasoning"]) <= REASONING_MAX_CHARS


def test_parse_strips_empty_reasoning_to_absent():
    from bots.llm_prompts import parse_llm_response

    raw = '{"action": "cooperate", "reasoning": "   "}'
    out = parse_llm_response(raw, _pd_state())
    assert out == {"choice": "cooperate"}


def test_parse_happy_without_reasoning_field():
    from bots.llm_prompts import parse_llm_response

    # Reasoning is optional — model may omit it entirely.
    raw = '{"action": "cooperate"}'
    out = parse_llm_response(raw, _pd_state())
    assert out == {"choice": "cooperate"}


# ---- random_action fallback ----


def test_random_action_pd_returns_valid_choice():
    from bots.llm_prompts import random_action

    rng = random.Random(0)
    out = random_action(_pd_state(), rng)
    assert out["choice"] in {"cooperate", "defect"}
    assert "reasoning" not in out  # random fallback carries no rationale


def test_random_action_el_farol_respects_bounds():
    from bots.llm_prompts import random_action

    rng = random.Random(42)
    out = random_action(_el_farol_state(num_slots=16, max_per_day=8), rng)
    assert "slots" in out
    assert all(0 <= s < 16 for s in out["slots"])
    assert len(out["slots"]) <= 8
    assert len(set(out["slots"])) == len(out["slots"])  # unique


def test_random_action_unknown_game_type_raises():
    from bots.llm_prompts import random_action

    rng = random.Random(0)
    with pytest.raises(ValueError):
        random_action({"game_type": "chess"}, rng)


# ---- llm_decide_action (async wiring) ----


@pytest.mark.anyio
async def test_llm_decide_falls_back_to_random_on_exception():
    from bots.llm_mcp_bot import llm_decide_action

    async def _boom(system: str, user: str) -> str:
        raise RuntimeError("LLM offline")

    rng = random.Random(0)
    out = await llm_decide_action(_pd_state(), completion_fn=_boom, rng=rng)
    assert out["choice"] in {"cooperate", "defect"}


@pytest.mark.anyio
async def test_llm_decide_uses_parsed_response_when_valid():
    from bots.llm_mcp_bot import llm_decide_action

    async def _ok(system: str, user: str) -> str:
        return '{"action": "defect", "reasoning": "tit for tat"}'

    rng = random.Random(0)
    out = await llm_decide_action(_pd_state(), completion_fn=_ok, rng=rng)
    assert out == {"choice": "defect", "reasoning": "tit for tat"}


@pytest.mark.anyio
async def test_llm_decide_falls_back_on_timeout():
    import asyncio

    from bots.llm_mcp_bot import llm_decide_action

    async def _slow(system: str, user: str) -> str:
        await asyncio.sleep(5)
        return '{"action": "cooperate"}'  # would be valid if it ever returned

    rng = random.Random(0)
    out = await llm_decide_action(
        _pd_state(), completion_fn=_slow, rng=rng, call_timeout_s=0.05
    )
    # Hung LLM → random fallback so the round can still be played in time
    assert out["choice"] in {"cooperate", "defect"}
    assert "reasoning" not in out


def test_extract_error_text_pulls_first_text_block():
    from bots.llm_mcp_bot import _extract_error_text

    raw = {
        "content": [{"type": "text", "text": "user already has an active tournament"}],
        "isError": True,
    }
    assert _extract_error_text(raw) == "user already has an active tournament"


def test_extract_error_text_fallback_when_no_content():
    from bots.llm_mcp_bot import _extract_error_text

    assert _extract_error_text({"isError": True}) == "unknown error"


@pytest.mark.anyio
async def test_llm_decide_falls_back_on_garbage():
    from bots.llm_mcp_bot import llm_decide_action

    async def _garbage(system: str, user: str) -> str:
        return "why would you even think I'm JSON"

    rng = random.Random(0)
    out = await llm_decide_action(_pd_state(), completion_fn=_garbage, rng=rng)
    assert out["choice"] in {"cooperate", "defect"}
    # Fallback has no reasoning key
    assert "reasoning" not in out
