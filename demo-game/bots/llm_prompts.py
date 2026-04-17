"""Prompt templates, response parsers, and random fallbacks per game.

Pure, synchronous helpers the LLM-strategy MCP bot composes. Keeping the
game-specific logic (valid-action sets, bounds checks, fallback picks)
here — and out of ``llm_mcp_bot.py`` — makes it easy to tune prompts and
unit-test the parser without touching the MCP main loop or the LLM SDK.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

REASONING_MAX_CHARS = 1000
"""Client-side cap on reasoning length.

Server accepts up to ``ATP_TOURNAMENT_REASONING_MAX_CHARS`` (default 8000);
we truncate aggressively to keep the wire small and avoid 422 on chatty
models. The LLM can easily produce 2-3 KB of chain-of-thought when
unprompted.
"""

SYSTEM_PROMPT = """\
You are a strategic game-playing AI agent competing in a multi-round \
game-theoretic tournament. Each round you receive the current state and \
must choose a single valid action.

Respond with ONLY a compact JSON object. No prose, no markdown fences. \
Example shape:

    {"action": "<one of the valid options>", "reasoning": "<one sentence>"}

For El Farol, use ``slots`` instead of ``action``:

    {"slots": [0, 3, 7], "reasoning": "<one sentence>"}

Keep "reasoning" under 200 characters. Any non-JSON output, any unknown \
action, or any out-of-range slot will be discarded and replaced with a \
random move — losing you the round.
"""


_PD_CHOICES = {"cooperate", "defect"}
_SH_CHOICES = {"stag", "hare"}
_BOS_CHOICES = {"A", "B"}


def _pair_history_summary(
    your_history: list[Any] | None,
    opponent_history: list[Any] | None,
) -> str:
    """Summarize a 2-player game's parallel move histories.

    Wire schema for PD / SH / BoS uses ``your_history`` + ``opponent_history``
    as parallel lists (same length, aligned by round). We present them as
    ``r1: you=C opp=D; r2: ...`` so the LLM can read the match's arc.
    """
    your = your_history or []
    theirs = opponent_history or []
    if not your and not theirs:
        return "no prior rounds"
    pairs = list(zip(your[-10:], theirs[-10:], strict=False))
    entries = [f"r{i}: you={m} opp={o}" for i, (m, o) in enumerate(pairs, start=1)]
    return "; ".join(entries) if entries else "no prior rounds"


def _el_farol_history_summary(your_history: list[list[int]] | None) -> str:
    """Summarize a player's own El Farol attendance history."""
    hist = your_history or []
    if not hist:
        return "no prior rounds"
    entries = [f"r{i}: slots={slots}" for i, slots in enumerate(hist[-10:], start=1)]
    return "; ".join(entries)


def build_user_prompt(state: dict[str, Any]) -> str:
    """Render a game-specific user prompt from current MCP state."""
    game_type = state.get("game_type")
    your_hist = state.get("your_history")
    opp_hist = state.get("opponent_history")
    if game_type == "prisoners_dilemma":
        return (
            "Game: Prisoner's Dilemma (2 players).\n"
            "Valid actions: cooperate | defect.\n"
            "Payoffs (you, opponent): both C -> (3,3); you C opp D -> (0,5); "
            "you D opp C -> (5,0); both D -> (1,1).\n"
            f"Round {state.get('round_number')} of {state.get('total_rounds')}.\n"
            f"Your cumulative score: {state.get('your_cumulative_score')}.\n"
            f"History: {_pair_history_summary(your_hist, opp_hist)}.\n"
            "Pick an action."
        )
    if game_type == "stag_hunt":
        return (
            "Game: Stag Hunt (2 players).\n"
            "Valid actions: stag | hare.\n"
            "Payoffs: both stag -> (4,4) (high reward, risky); "
            "you stag opp hare -> (0,3); you hare opp stag -> (3,0); "
            "both hare -> (2,2) (safe).\n"
            f"Round {state.get('round_number')} of {state.get('total_rounds')}.\n"
            f"History: {_pair_history_summary(your_hist, opp_hist)}.\n"
            "Pick an action."
        )
    if game_type == "battle_of_sexes":
        preferred = state.get("your_preferred", "A")
        return (
            "Game: Battle of the Sexes (2 players, asymmetric).\n"
            "Valid actions: A | B.\n"
            f"Your preferred outcome is **{preferred}**. "
            "If both pick A: you get 2 if your preferred is A else 1. "
            "If both pick B: you get 2 if your preferred is B else 1. "
            "If choices mismatch: both get 0.\n"
            f"Round {state.get('round_number')} of {state.get('total_rounds')}.\n"
            f"History: {_pair_history_summary(your_hist, opp_hist)}.\n"
            "Pick an action."
        )
    if game_type == "el_farol":
        num_slots = int(state.get("num_slots") or 16)
        max_per_day = int((state.get("action_schema") or {}).get("max_length") or 8)
        return (
            f"Game: El Farol Bar (N players, {num_slots} time slots per day).\n"
            f"You choose up to {max_per_day} slots to attend, as integers in "
            f"[0, {num_slots - 1}], unique. Empty list is valid.\n"
            "Payoff per slot: +1 if attendance <= capacity, -1 if over.\n"
            f"Round {state.get('round_number')} of {state.get('total_rounds')}.\n"
            f"History: {_el_farol_history_summary(your_hist)}.\n"
            f"Respond with a JSON object: "
            f'{{"slots": [ints], "reasoning": "..."}}'
        )
    return (
        f"Unknown game_type={game_type!r}. "
        "Respond with a random-looking valid action anyway."
    )


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction: raw, then markdown-fenced, then brace-span."""
    text = raw.strip()

    # Strip markdown fences.
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    for candidate in (text, _brace_span(text)):
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _brace_span(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    return text[start : end + 1]


def _normalize_reasoning(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > REASONING_MAX_CHARS:
        stripped = stripped[:REASONING_MAX_CHARS]
    return stripped


def parse_llm_response(raw_text: str, state: dict[str, Any]) -> dict[str, Any] | None:
    """Parse LLM output into a canonical action dict for ``make_move``.

    Returns ``None`` on any validation failure; the caller is expected to
    fall back to a random action. Empty/whitespace reasoning is dropped
    (no ``reasoning`` key in the output) so the server stores NULL rather
    than a meaningless string.
    """
    parsed = _extract_json(raw_text)
    if parsed is None:
        return None

    game_type = state.get("game_type")
    reasoning = _normalize_reasoning(parsed.get("reasoning"))

    if game_type == "prisoners_dilemma":
        return _finalize_choice(parsed, _PD_CHOICES, reasoning)
    if game_type == "stag_hunt":
        return _finalize_choice(parsed, _SH_CHOICES, reasoning)
    if game_type == "battle_of_sexes":
        return _finalize_choice(parsed, _BOS_CHOICES, reasoning)
    if game_type == "el_farol":
        return _finalize_slots(parsed, state, reasoning)
    return None


def _finalize_choice(
    parsed: dict[str, Any],
    allowed: set[str],
    reasoning: str | None,
) -> dict[str, Any] | None:
    action = parsed.get("action") or parsed.get("choice")
    if not isinstance(action, str) or action not in allowed:
        return None
    out: dict[str, Any] = {"choice": action}
    if reasoning:
        out["reasoning"] = reasoning
    return out


def _finalize_slots(
    parsed: dict[str, Any],
    state: dict[str, Any],
    reasoning: str | None,
) -> dict[str, Any] | None:
    slots = parsed.get("slots")
    if slots is None and isinstance(parsed.get("action"), list):
        slots = parsed["action"]
    if not isinstance(slots, list):
        return None
    num_slots = int(state.get("num_slots") or 16)
    max_per_day = int((state.get("action_schema") or {}).get("max_length") or 8)

    cleaned: list[int] = []
    for s in slots:
        if not isinstance(s, int) or isinstance(s, bool):
            return None
        if s < 0 or s >= num_slots:
            return None
        cleaned.append(s)
    if len(cleaned) != len(set(cleaned)):
        return None
    if len(cleaned) > max_per_day:
        return None

    out: dict[str, Any] = {"slots": sorted(cleaned)}
    if reasoning:
        out["reasoning"] = reasoning
    return out


def random_action(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Last-resort fallback when the LLM is unreachable or malformed.

    No ``reasoning`` key is attached — an honest "we picked at random"
    is better than a fabricated rationale pretending the bot thought
    about the move.
    """
    game_type = state.get("game_type")
    if game_type == "prisoners_dilemma":
        return {"choice": rng.choice(sorted(_PD_CHOICES))}
    if game_type == "stag_hunt":
        return {"choice": rng.choice(sorted(_SH_CHOICES))}
    if game_type == "battle_of_sexes":
        return {"choice": rng.choice(sorted(_BOS_CHOICES))}
    if game_type == "el_farol":
        num_slots = int(state.get("num_slots") or 16)
        max_per_day = int((state.get("action_schema") or {}).get("max_length") or 8)
        k = rng.randint(0, min(max_per_day, num_slots))
        return {"slots": sorted(rng.sample(range(num_slots), k))}
    raise ValueError(f"unknown game_type={game_type!r}")
