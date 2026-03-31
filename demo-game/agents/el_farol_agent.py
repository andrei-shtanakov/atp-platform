"""
El Farol Bar — LLM Game Agent (OpenAI GPT-4o-mini).

HTTP-агент для El Farol Bar Problem через ATP Platform.
Получает описание игровой ситуации (attendance history, slots),
вызывает OpenAI API и возвращает список слотов для посещения.

Запуск локально:
    export OPENAI_API_KEY=sk-...
    uv run python -m uvicorn demo-game.agents.el_farol_agent:app --port 8011

Запуск в контейнере (Podman/Docker):
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

{"action": [4, 5, 6, 7, 8], "reasoning": "<brief explanation>"}

Rules:
- "action" must be a list of integers (slot indices, 0-based).
- Pick 4-8 consecutive slots that you predict will be least crowded.
- Use attendance history to predict which slots will be quiet today.
- Keep reasoning brief (1-2 sentences).
- Do NOT add any text before or after the JSON.
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


def parse_action(parsed: dict | None, num_slots: int = 16) -> list[int]:
    """Extract slot list from parsed JSON, with fallback."""
    if parsed and "action" in parsed:
        action = parsed["action"]
        if isinstance(action, list):
            return [s for s in action if isinstance(s, int) and 0 <= s < num_slots]
    # Fallback: middle slots
    mid = num_slots // 4
    return list(range(mid, mid + 6))


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
                    data={"action": action, "reasoning": reasoning},
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
