"""
Code Writer Agent — OpenAI GPT-4o.

HTTP agent for ATP Platform. Accepts a Python code-writing task, calls the
OpenAI API, and returns the generated file.

Run:
    export OPENAI_API_KEY=sk-...
    uv run uvicorn agents.openai_agent:app --port 8001

Dependencies:
    uv add fastapi uvicorn openai
"""

import os
import re
import time

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Code Writer Agent (OpenAI)")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-4o-mini"

# GPT-4o-mini prices (USD per 1M tokens)
INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.60

SYSTEM_PROMPT = """\
You are a Python code generator. You receive a task description and return ONLY
valid Python code. Rules:

1. Output ONLY the Python code — no markdown fences, no explanations, no comments
   like "here is the code".
2. Include type hints for all function signatures.
3. Include docstrings for all public functions and classes.
4. Follow PEP 8 style (snake_case, 88 char lines).
5. Handle edge cases and raise appropriate exceptions.
6. Use only the libraries specified in the requirements.
"""


# --- Pydantic models (ATP Protocol) ---


class Task(BaseModel):
    description: str
    input_data: dict | None = None
    expected_artifacts: list[str] | None = None


class Constraints(BaseModel):
    max_steps: int | None = None
    max_tokens: int | None = None
    timeout_seconds: int | None = None
    budget_usd: float | None = None


class ATPRequest(BaseModel):
    version: str = "1.0"
    task_id: str
    task: Task
    constraints: Constraints | None = None
    context: dict | None = None
    metadata: dict | None = None


class Artifact(BaseModel):
    type: str = "file"
    path: str | None = None
    content_type: str = "text/x-python"
    content: str | None = None


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
    artifacts: list[Artifact] = []
    metrics: Metrics = Metrics()
    error: str | None = None


# --- Helpers ---


def strip_markdown_fences(text: str) -> str:
    """Strip ```python ... ``` fences if the LLM added them."""
    text = text.strip()
    pattern = r"^```(?:python)?\s*\n(.*?)```\s*$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Compute request cost in USD."""
    input_cost = input_tokens * INPUT_PRICE_PER_M / 1_000_000
    output_cost = output_tokens * OUTPUT_PRICE_PER_M / 1_000_000
    return round(input_cost + output_cost, 6)


# --- Endpoints ---


@app.post("/")
async def handle_request(request: ATPRequest) -> ATPResponse:
    """Handle an ATP request: generate Python code via OpenAI."""
    start = time.monotonic()

    try:
        input_data = request.task.input_data or {}
        requirements = input_data.get("requirements", request.task.description)
        filename = input_data.get("filename", "solution.py")

        max_tokens = 4096
        if request.constraints and request.constraints.max_tokens:
            max_tokens = min(request.constraints.max_tokens, 16384)

        user_message = (
            f"Task: {request.task.description}\n\nRequirements:\n{requirements}"
        )

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        code = strip_markdown_fences(response.choices[0].message.content or "")
        elapsed = time.monotonic() - start

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return ATPResponse(
            task_id=request.task_id,
            status="completed",
            artifacts=[
                Artifact(
                    type="file",
                    path=filename,
                    content_type="text/x-python",
                    content=code,
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
    return {"status": "ok", "model": MODEL, "provider": "openai"}
