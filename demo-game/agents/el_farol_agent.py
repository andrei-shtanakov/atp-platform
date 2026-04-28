"""
El Farol Bar — LLM Game Agent (OpenAI GPT-4o-mini).

HTTP agent for the El Farol Bar Problem served via ATP Platform.
Receives a game-state description (attendance history, slots), calls the
OpenAI API, and returns the list of slots to attend.

Run locally:
    export OPENAI_API_KEY=sk-...
    uv run python -m uvicorn demo-game.agents.el_farol_agent:app --port 8011

Run in a container (Podman/Docker):
    podman build -f demo-game/Containerfile.el-farol -t el-farol-agent .
    podman run --rm -p 8011:8011 -e OPENAI_API_KEY=sk-... el-farol-agent
"""

import json
import os
import re
import time

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="El Farol Agent (GPT-4o-mini)")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

MODEL = os.environ.get("MODEL", "gpt-4o-mini")

# Prices per 1M tokens (USD)
INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.60

SYSTEM_PROMPT = """\
You are a strategic agent in the El Farol Bar Problem.

Each day you choose which time slots to attend at a bar.
If too many people attend the same slot, it becomes crowded (bad).
You want to maximize happy (non-crowded) slots and minimize crowded ones.

You will receive:
- Number of slots and capacity threshold
- Attendance history from previous days
- Your cumulative stats (happy vs crowded slots)

Respond with ONLY a valid JSON object:

{"intervals": [[4, 8]],
 "reasoning": "<brief explanation>"}

Rules:
- "intervals" is a list of inclusive [start, end] pairs of 0-based
  slot indices.
- Submit at most 2 intervals covering at most 8 slots in total per day.
- Intervals must be non-overlapping and non-adjacent (at least one
  empty slot between them).
- Use attendance history to predict which slots will be quiet.
- Keep reasoning brief (1-2 sentences).
- Do NOT add any text before or after the JSON.
- ``{"intervals": []}`` means "stay home" (no slots attended).
"""


# --- ATP Protocol models ---


class Task(BaseModel):
    description: str
    input_data: dict | None = None
    expected_artifacts: list[str] | None = None


class Constraints(BaseModel):
    max_steps: int | None = None
    max_tokens: int | None = None
    timeout_seconds: int | None = None
    budget_usd: float | None = None
    response_format: dict | None = None


class ATPRequest(BaseModel):
    version: str = "1.0"
    task_id: str
    task: Task
    constraints: Constraints | dict | None = None
    context: dict | None = None
    metadata: dict | None = None


class StructuredArtifact(BaseModel):
    type: str = "structured"
    name: str = "game_action"
    data: dict


class Metrics(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_steps: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    wall_time_seconds: float = 0
    cost_usd: float = 0


class ATPResponse(BaseModel):
    version: str = "1.0"
    task_id: str
    status: str
    artifacts: list[StructuredArtifact] = []
    metrics: Metrics = Metrics()
    error: str | None = None


# --- Helpers ---


def extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    pattern = r"```(?:json)?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def parse_action(parsed: dict | None, num_slots: int = 16) -> list[list[int]]:
    """Extract intervals from parsed JSON, with a safe fallback.

    Reads the canonical ``intervals`` key only; flat-slot legacy keys
    (``slots`` / ``action``) are no longer accepted. Each interval is
    clamped to ``[0, num_slots - 1]`` and dropped if malformed. Returns
    a single mid-day fallback interval when nothing parseable is found.
    """
    if parsed:
        raw = parsed.get("intervals")
        if isinstance(raw, list):
            cleaned: list[list[int]] = []
            for pair in raw:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                start, end = pair[0], pair[1]
                if not (isinstance(start, int) and isinstance(end, int)):
                    continue
                lo = max(0, int(start))
                hi = min(num_slots - 1, int(end))
                if lo <= hi:
                    cleaned.append([lo, hi])
            return cleaned
    # Fallback: a single midday interval of 6 slots.
    mid = num_slots // 4
    end = min(mid + 5, num_slots - 1)
    if end < mid:
        return []
    return [[mid, end]]


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD."""
    input_cost = input_tokens * INPUT_PRICE_PER_M / 1_000_000
    output_cost = output_tokens * OUTPUT_PRICE_PER_M / 1_000_000
    return round(input_cost + output_cost, 6)


# --- Endpoints ---


@app.post("/")
async def handle_request(request: ATPRequest) -> ATPResponse:
    """Handle ATP game request: choose slots via OpenAI."""
    start = time.monotonic()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=256,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.task.description},
            ],
        )

        raw_text = response.choices[0].message.content or ""
        elapsed = time.monotonic() - start

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        parsed = extract_json(raw_text)
        action = parse_action(parsed)
        reasoning = parsed.get("reasoning", "") if parsed else ""

        return ATPResponse(
            task_id=request.task_id,
            status="completed",
            artifacts=[
                StructuredArtifact(
                    data={
                        "intervals": action,
                        "reasoning": reasoning,
                    },
                )
            ],
            metrics=Metrics(
                total_tokens=input_tokens + output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_steps=1,
                llm_calls=1,
                wall_time_seconds=round(elapsed, 3),
                cost_usd=calculate_cost(input_tokens, output_tokens),
            ),
        )

    except Exception as e:
        elapsed = time.monotonic() - start
        return ATPResponse(
            task_id=request.task_id,
            status="failed",
            error=f"OpenAI API error: {e}",
            metrics=Metrics(wall_time_seconds=round(elapsed, 3)),
        )


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL, "game": "el_farol"}
