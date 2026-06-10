"""LLM-backed ATP HTTP agent for the methodology demo.

Unlike the deterministic echo agent, this one actually attempts the task by
calling an OpenAI-compatible chat-completions endpoint. That makes the
agent-eval-case *trap* meaningful: a capable model passes the ``clean`` level but
tends to fabricate a value on ``severe`` — the "curve of collapse" the sweep is
built to reveal.

Configuration (env):
  LLM_BASE_URL   OpenAI-compatible base URL (default https://api.openai.com/v1).
                 Point at a local server (Ollama, vLLM, ...) for an air-gapped run.
  LLM_MODEL      model name (default gpt-4o-mini).
  LLM_API_KEY    API key (use any non-empty value for local servers that ignore it).
"""

import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="ATP Methodology Demo Agent")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for the compose healthcheck."""
    return {"status": "ok"}


def _build_prompt(task: dict[str, Any]) -> str:
    """Render only what the agent should legitimately see into the prompt.

    Includes the instruction, each inline artifact's type + content, and the
    constraints. Author-side metadata is deliberately excluded: `note` flags where
    the trap is planted and `distractor` describes the trap's pressure — feeding
    either to the agent would leak the trap and corrupt the evaluation. `path`
    (external artifacts) and `turns` (multi-turn scripts) are out of scope for this
    single-turn demo agent.
    """
    parts = [task.get("description", "")]
    input_data = task.get("input_data") or {}
    for artifact in input_data.get("artifacts", []):
        content = artifact.get("content")
        if content:
            aid = artifact.get("id", "artifact")
            atype = artifact.get("type", "artifact")
            parts.append(f"\n--- {aid} ({atype}) ---\n{content}")
    constraints = input_data.get("constraints") or []
    if constraints:
        parts.append("\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints))
    return "\n".join(parts)


@app.post("/execute")
async def execute(request: Request) -> JSONResponse:
    """Handle an ATPRequest: call the LLM, return its answer as an artifact."""
    start = time.perf_counter()
    payload: dict[str, Any] = await request.json()
    task_id: str = payload.get("task_id", "unknown")
    task: dict[str, Any] = payload.get("task") or {}

    prompt = _build_prompt(task)
    # Omit the auth header when no key is set — some local OpenAI-compatible
    # servers reject an empty Bearer token.
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers=headers,
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            body = resp.json()
            answer = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
    except Exception as e:  # noqa: BLE001 — report the failure as an ATP failure
        return JSONResponse(
            {
                "task_id": task_id,
                "status": "failed",
                "error": f"LLM call failed: {e}",
                "metrics": {"wall_time_seconds": round(time.perf_counter() - start, 4)},
            }
        )

    expected = task.get("expected_artifacts") or ["output.txt"]
    return JSONResponse(
        {
            "task_id": task_id,
            "status": "completed",
            "artifacts": [
                {
                    "type": "file",
                    "path": expected[0],
                    "content": answer,
                    "content_type": "text/plain",
                }
            ],
            "metrics": {
                "wall_time_seconds": round(time.perf_counter() - start, 4),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "llm_calls": 1,
            },
        }
    )
