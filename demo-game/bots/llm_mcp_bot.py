"""LLM-strategy MCP bot for tournament play.

Connects to an ATP dashboard MCP endpoint with an ``atp_a_...``
agent-scoped token, joins a pre-created tournament, and plays one
tournament to completion using an LLM to pick each move. Supports all
four supported game_types (prisoners_dilemma, stag_hunt,
battle_of_sexes, el_farol).

This is the canonical reference implementation for "how an agent should
think" on top of MCP — external users are expected to copy its shape
when wiring their own LLM/agent into the platform.

Design:
- LLM parsing + game-specific validation lives in ``llm_prompts.py`` as
  pure functions (easy to unit-test).
- ``llm_decide_action`` is the single async call site; it accepts an
  injected ``completion_fn`` so tests can pass a fake and production
  can pass OpenAI/Anthropic-backed factories.
- On any LLM failure (network error, malformed JSON, invalid action)
  we fall back to a uniformly random valid move. The bot keeps playing;
  the orchestrator/operator sees it in logs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any

from atp.adapters.mcp import MCPAdapter, MCPAdapterConfig
from bots.llm_prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_llm_response,
    random_action,
)

logger = logging.getLogger(__name__)

CompletionFn = Callable[[str, str], Awaitable[str]]
"""Async text completion: ``(system_prompt, user_prompt) -> raw_text``.

Kept provider-agnostic so tests can inject canned strings and production
can swap OpenAI/Anthropic factories without touching the MCP loop.
"""


LLM_CALL_TIMEOUT_S = 15.0
"""Hard ceiling on a single LLM completion call.

Independent of the MCP ``timeout_seconds`` and the tournament's
``round_deadline_s``. Set well below a typical 60s round deadline so
that even a slow LLM leaves room to submit a fallback move in time.
"""


async def llm_decide_action(
    state: dict[str, Any],
    *,
    completion_fn: CompletionFn,
    rng: random.Random,
    call_timeout_s: float = LLM_CALL_TIMEOUT_S,
) -> dict[str, Any]:
    """Return an action dict ready for ``make_move``.

    Calls ``completion_fn`` exactly once, under an ``asyncio.wait_for``
    ceiling. Any timeout, exception, malformed JSON, or out-of-range
    action degrades to a random valid fallback — the bot must never
    crash the tournament over an LLM hiccup, and must never block
    longer than one round on an unresponsive provider.
    """
    try:
        raw = await asyncio.wait_for(
            completion_fn(SYSTEM_PROMPT, build_user_prompt(state)),
            timeout=call_timeout_s,
        )
    except TimeoutError:
        logger.warning(
            "llm completion timed out after %.1fs; using random fallback",
            call_timeout_s,
        )
        return random_action(state, rng)
    except Exception as exc:
        logger.warning("llm completion failed: %s; using random fallback", exc)
        return random_action(state, rng)

    parsed = parse_llm_response(raw, state)
    if parsed is None:
        logger.warning(
            "llm output rejected (malformed/invalid): %r; using random fallback",
            raw[:200],
        )
        return random_action(state, rng)
    return parsed


# ---- Provider factories ---------------------------------------------------


def build_openai_completion_fn(*, model: str, api_key: str) -> CompletionFn:
    """Return a ``CompletionFn`` backed by OpenAI Chat Completions."""
    from openai import AsyncOpenAI

    # Explicit client-level timeout so a stuck connection surfaces quickly.
    # The outer asyncio.wait_for in llm_decide_action is the hard cap.
    client = AsyncOpenAI(api_key=api_key, timeout=LLM_CALL_TIMEOUT_S)

    async def _call(system: str, user: str) -> str:
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=256,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    return _call


def build_anthropic_completion_fn(*, model: str, api_key: str) -> CompletionFn:
    """Return a ``CompletionFn`` backed by Anthropic Messages."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key, timeout=LLM_CALL_TIMEOUT_S)

    async def _call(system: str, user: str) -> str:
        resp = await client.messages.create(
            model=model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [block.text for block in resp.content if getattr(block, "text", None)]
        return "".join(parts).strip()

    return _call


def build_completion_fn_from_env() -> CompletionFn:
    """Construct the configured ``CompletionFn`` from env vars.

    Reads ``LLM_PROVIDER`` (``openai`` | ``anthropic``, default ``openai``)
    and ``LLM_MODEL`` (provider-appropriate default). Raises RuntimeError
    at startup if the required API key is missing — better to fail
    immediately than mid-tournament.
    """
    provider = (os.environ.get("LLM_PROVIDER") or "openai").lower()
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        return build_openai_completion_fn(model=model, api_key=key)
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set"
            )
        model = os.environ.get("LLM_MODEL") or "claude-haiku-4-5-20251001"
        return build_anthropic_completion_fn(model=model, api_key=key)
    raise RuntimeError(f"Unknown LLM_PROVIDER={provider!r}")


# ---- MCP main loop --------------------------------------------------------


async def run_bot(
    *,
    server_url: str,
    token: str,
    tournament_id: int,
    agent_name: str,
    rounds: int,
    seed: int,
    completion_fn: CompletionFn | None = None,
    verify_ssl: bool = True,
    per_round_timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Play one tournament to completion using an LLM strategy.

    Mirrors ``el_farol_mcp_bot.run_bot`` but delegates move selection to
    the LLM via ``completion_fn`` (or, if unset, constructs one from env).

    Args:
        server_url: Base URL of the ATP dashboard; MCP SSE is
            ``{server_url}/mcp/sse``.
        token: ``atp_a_...`` agent-scoped API token.
        tournament_id: Tournament the orchestrator pre-created.
        agent_name: Display name for this participant.
        rounds: Total rounds — used for the overall wall-clock budget.
        seed: Per-bot RNG seed for the random-fallback path.
        completion_fn: Inject a custom ``CompletionFn`` (tests/smoke).
            When ``None``, reads ``LLM_PROVIDER`` / keys from env.
        verify_ssl: Pass ``False`` only for local dev with self-signed
            certs.
        per_round_timeout_s: Per-``make_move`` ceiling; also used to
            compute the overall deadline.
    """
    rng = random.Random(seed)
    completion = completion_fn or build_completion_fn_from_env()

    config = MCPAdapterConfig(
        transport="sse",
        url=f"{server_url.rstrip('/')}/mcp/sse",
        headers={"Authorization": f"Bearer {token}"},
        verify_ssl=verify_ssl,
        startup_timeout=30.0,
        timeout_seconds=per_round_timeout_s,
    )
    adapter = MCPAdapter(config)

    rounds_played = 0
    final_score: float | None = None
    final_status = "unknown"
    last_round_played = 0
    loop = asyncio.get_event_loop()
    deadline = loop.time() + rounds * per_round_timeout_s + 60.0

    try:
        await adapter.initialize()

        join_result = await adapter.call_tool(
            "join_tournament",
            {"tournament_id": tournament_id, "agent_name": agent_name},
        )
        logger.info("[%s] joined: %s", agent_name, join_result)

        while loop.time() < deadline:
            state_raw = await adapter.call_tool(
                "get_current_state", {"tournament_id": tournament_id}
            )
            state = _unwrap_tool_result(state_raw)

            # Wire schemas differ by game: PDRoundState / SHRoundState /
            # BoSRoundState expose ``your_turn`` (bool), while
            # ElFarolRoundState exposes ``pending_submission`` (bool).
            # Accept either so this bot plays every supported game.
            pending = state.get("pending_submission")
            if pending is None:
                pending = state.get("your_turn")
            round_number = int(state.get("round_number") or 0)
            total_rounds = int(state.get("total_rounds") or 0)

            if not pending:
                final_score = state.get("your_cumulative_score")
                if round_number >= total_rounds and last_round_played >= total_rounds:
                    final_status = "completed"
                    break
                await asyncio.sleep(0.3)
                continue

            if round_number == last_round_played:
                await asyncio.sleep(0.3)
                continue

            action = await llm_decide_action(state, completion_fn=completion, rng=rng)
            move = await adapter.call_tool(
                "make_move",
                {"tournament_id": tournament_id, "action": action},
            )
            rounds_played += 1
            last_round_played = round_number
            logger.info(
                "[%s] r%s: action=%s -> %s",
                agent_name,
                round_number,
                _summarize_action(action),
                _unwrap_tool_result(move).get("status"),
            )
        else:
            final_status = "timeout"

    finally:
        try:
            await adapter.disconnect()
        except Exception:
            pass

    return {
        "agent_name": agent_name,
        "final_score": final_score,
        "rounds_played": rounds_played,
        "final_status": final_status,
    }


def _summarize_action(action: dict[str, Any]) -> str:
    """Log-friendly representation of an action dict."""
    if "choice" in action:
        base = f"choice={action['choice']}"
    elif "slots" in action:
        base = f"slots={action['slots']}"
    else:
        base = str(action)
    if "reasoning" in action:
        reason = action["reasoning"]
        return f"{base} reason={reason[:60]!r}"
    return base


def _unwrap_tool_result(raw: dict[str, Any]) -> dict[str, Any]:
    """FastMCP wraps tool returns in ``{"content": [...]}``; unwrap it.

    Mirrors the helper in ``el_farol_mcp_bot`` — some MCP clients surface
    the wrapped form, others pre-unwrap, so we handle both.
    """
    if not isinstance(raw, dict):
        return {}
    if "structuredContent" in raw and isinstance(raw["structuredContent"], dict):
        return raw["structuredContent"]
    if "content" in raw and isinstance(raw["content"], list):
        import json

        for item in raw["content"]:
            if isinstance(item, dict) and item.get("type") == "text":
                try:
                    parsed = json.loads(item.get("text", ""))
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
    return raw
