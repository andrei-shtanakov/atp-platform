"""Minimal ATP-compatible HTTP agent for the docker-compose demo.

Speaks the ATP HTTP contract directly:
  POST /execute  with an ATPRequest JSON body
  ->             an ATPResponse JSON body

It is deterministic and fully offline (no LLM call), so the on-prem demo runs
green without any API key. The same suite can target a real Bedrock Agent in the
cloud variant simply by swapping the adapter at run time.
"""

import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="ATP Demo HTTP Agent")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by the compose healthcheck."""
    return {"status": "ok"}


@app.post("/execute")
async def execute(request: Request) -> JSONResponse:
    """Handle an ATPRequest and return an ATPResponse.

    The platform's HTTP adapter POSTs ``ATPRequest.model_dump(mode="json")``
    and validates the reply as an ``ATPResponse``.
    """
    start = time.perf_counter()
    payload: dict[str, Any] = await request.json()

    task_id: str = payload.get("task_id", "unknown")
    task: dict[str, Any] = payload.get("task") or {}
    description: str = task.get("description", "")
    expected: list[str] = task.get("expected_artifacts") or ["output.txt"]
    out_path = expected[0]

    # Deterministic "work": summarise the task into the output artifact.
    content = (
        f"Task: {description}\n"
        f"Handled by the ATP demo HTTP agent (on-prem container).\n"
    )

    response = {
        "task_id": task_id,
        "status": "completed",
        "artifacts": [
            {
                "type": "file",
                "path": out_path,
                "content": content,
                "content_type": "text/plain",
            }
        ],
        "metrics": {
            "wall_time_seconds": round(time.perf_counter() - start, 4),
            "total_steps": 1,
            "tool_calls": 0,
            "llm_calls": 0,
        },
    }
    return JSONResponse(response)
