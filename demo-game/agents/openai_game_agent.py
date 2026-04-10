"""
Game-Playing Agent — OpenAI GPT-4o-mini.

HTTP agent for game-theoretic testing via ATP Platform. Receives a
game-state description, calls the OpenAI API, and returns a decision
(action) in ATP Protocol format.

Run:
    export OPENAI_API_KEY=sk-...
    uv run python -m uvicorn demo-game.agents.openai_game_agent:app --port 8010

Dependencies:
    uv add fastapi uvicorn openai
"""

import json
import os
import re
import time

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Game Agent (OpenAI GPT-4o-mini)")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

MODEL = "gpt-4o-mini"

# GPT-4o-mini prices (USD per 1M tokens)
INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.60

SYSTEM_PROMPT = """\
You are a strategic game-playing AI agent. You are participating in a
game-theoretic experiment. You will receive a description of the current
game state and must choose an action.

Rules:
1. Analyze the game state, history, and available actions carefully.
2. Think about what strategy maximizes your long-term payoff.
3. Consider your opponent's likely strategy based on their past actions.
4. Respond with ONLY a valid JSON object in this exact format:

{"action": "<your_chosen_action>", "reasoning": "<brief explanation>"}

IMPORTANT:
- The "action" field must be one of the available actions listed.
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
    # Remove markdown fences
    pattern = r"```(?:json)?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try finding JSON in text
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


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD."""
    input_cost = input_tokens * INPUT_PRICE_PER_M / 1_000_000
    output_cost = output_tokens * OUTPUT_PRICE_PER_M / 1_000_000
    return round(input_cost + output_cost, 6)


# --- Endpoints ---


@app.post("/")
async def handle_request(request: ATPRequest) -> ATPResponse:
    """Handle ATP game request: choose action via OpenAI."""
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

        # Parse action from response
        parsed = extract_json(raw_text)
        if parsed and "action" in parsed:
            action_data = parsed
        else:
            # Fallback: try to extract action keyword
            raw_lower = raw_text.lower().strip()
            if "cooperate" in raw_lower:
                action_data = {"action": "cooperate", "reasoning": raw_text}
            elif "defect" in raw_lower:
                action_data = {"action": "defect", "reasoning": raw_text}
            else:
                action_data = {"action": raw_text.strip(), "reasoning": ""}

        return ATPResponse(
            task_id=request.task_id,
            status="completed",
            artifacts=[
                StructuredArtifact(
                    type="structured",
                    format="json",
                    data=action_data,
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
    return {"status": "ok", "model": MODEL, "provider": "openai"}
