# Integration Guide

## Обзор

Данный документ описывает, как интегрировать AI-агента с Agent Test Platform. Независимо от используемого фреймворка, агент должен реализовать ATP Protocol для взаимодействия с платформой.

## Способы интеграции

ATP поддерживает несколько способов интеграции, от простых до продвинутых:

| Способ | Сложность | Use Case |
|--------|-----------|----------|
| HTTP Endpoint | Низкая | Агент уже имеет HTTP API |
| Container | Низкая | Агент упакован в Docker |
| CLI Wrapper | Низкая | Агент — скрипт или CLI-утилита |
| Native Adapter | Средняя | Интеграция с фреймворком |
| Direct Integration | Высокая | Максимальный контроль |

---

## 1. HTTP Endpoint Integration

Самый простой способ — агент предоставляет HTTP endpoint, принимающий ATP Request.

### Минимальная реализация (FastAPI)

```python
# agent_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
import uuid
from datetime import datetime, timezone

app = FastAPI()

class Task(BaseModel):
    description: str
    input_data: dict | None = None

class Constraints(BaseModel):
    max_steps: int | None = None
    max_tokens: int | None = None
    timeout_seconds: int = 300
    allowed_tools: list[str] | None = None

class ATPRequest(BaseModel):
    version: str
    task_id: str
    task: Task
    constraints: Constraints | None = None
    context: dict | None = None
    metadata: dict | None = None

class Artifact(BaseModel):
    type: str
    path: str | None = None
    name: str | None = None
    content: str | None = None
    data: dict | None = None

class Metrics(BaseModel):
    total_tokens: int = 0
    total_steps: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    wall_time_seconds: float = 0.0

class ATPResponse(BaseModel):
    version: str = "1.0"
    task_id: str
    status: str
    artifacts: list[Artifact]
    metrics: Metrics
    error: str | None = None

@app.post("/execute", response_model=ATPResponse)
async def execute(request: ATPRequest) -> ATPResponse:
    """Execute agent task."""
    start_time = datetime.now(timezone.utc)

    try:
        # Your agent logic here
        result = await run_your_agent(
            task=request.task.description,
            inputs=request.task.input_data,
            constraints=request.constraints,
        )

        # Build response
        return ATPResponse(
            task_id=request.task_id,
            status="completed",
            artifacts=[
                Artifact(
                    type="file",
                    path="output.md",
                    content=result.output,
                ),
            ],
            metrics=Metrics(
                total_tokens=result.tokens_used,
                total_steps=result.steps,
                tool_calls=result.tool_calls,
                llm_calls=result.llm_calls,
                wall_time_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
            ),
        )

    except Exception as e:
        return ATPResponse(
            task_id=request.task_id,
            status="failed",
            artifacts=[],
            metrics=Metrics(),
            error=str(e),
        )

async def run_your_agent(task: str, inputs: dict | None, constraints: Constraints | None):
    """Replace with your agent implementation."""
    # Your agent code here
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Регистрация в ATP

```yaml
# atp.config.yaml
agents:
  my-agent:
    type: http
    endpoint: "http://localhost:8000"
    timeout: 300
    health_check: "/health"  # Optional
```

### Запуск теста

```bash
# Start your agent
python agent_server.py &

# Run tests
atp test --agent=my-agent --suite=smoke
```

---

## 2. Container Integration

Агент упакован в Docker-образ и взаимодействует через stdin/stdout.

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ATP contract: read request from stdin, write response to stdout
ENTRYPOINT ["python", "agent_cli.py"]
```

### CLI Agent Script

```python
#!/usr/bin/env python3
# agent_cli.py
"""ATP-compatible CLI agent."""

import json
import sys
from datetime import datetime, timezone

def main():
    # Read ATP Request from stdin
    request_json = sys.stdin.read()
    request = json.loads(request_json)

    task_id = request["task_id"]
    task_description = request["task"]["description"]
    constraints = request.get("constraints", {})

    # Emit progress events to stderr (optional)
    emit_event(task_id, 0, "progress", {"message": "Starting", "percentage": 0})

    try:
        # Run your agent
        result = run_agent(task_description, constraints)

        # Build ATP Response
        response = {
            "version": "1.0",
            "task_id": task_id,
            "status": "completed",
            "artifacts": [
                {
                    "type": "file",
                    "path": "output.md",
                    "content": result["output"],
                }
            ],
            "metrics": {
                "total_tokens": result.get("tokens", 0),
                "total_steps": result.get("steps", 0),
                "tool_calls": result.get("tool_calls", 0),
                "llm_calls": result.get("llm_calls", 0),
                "wall_time_seconds": result.get("duration", 0),
            },
        }

    except Exception as e:
        response = {
            "version": "1.0",
            "task_id": task_id,
            "status": "failed",
            "artifacts": [],
            "metrics": {},
            "error": str(e),
        }

    # Write ATP Response to stdout
    print(json.dumps(response))

def emit_event(task_id: str, seq: int, event_type: str, payload: dict):
    """Emit ATP Event to stderr."""
    event = {
        "version": "1.0",
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence": seq,
        "event_type": event_type,
        "payload": payload,
    }
    print(json.dumps(event), file=sys.stderr)

def run_agent(task: str, constraints: dict) -> dict:
    """Your agent implementation."""
    # Replace with actual logic
    return {
        "output": f"Processed: {task}",
        "tokens": 1000,
        "steps": 5,
    }

if __name__ == "__main__":
    main()
```

### Build and Register

```bash
# Build image
docker build -t my-agent:latest .

# Register in ATP config
```

```yaml
# atp.config.yaml
agents:
  my-agent:
    type: container
    image: "my-agent:latest"
    resources:
      memory: "2Gi"
      cpu: "1"
```

---

## 3. Framework Adapters

Для популярных фреймворков ATP предоставляет готовые адаптеры.

### LangGraph Adapter

```python
# agents/my_langgraph_agent.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define your LangGraph agent
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    task: str
    result: str

def research_node(state: AgentState) -> dict:
    # Your logic
    return {"messages": [...], "result": "..."}

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("research", research_node)
    graph.add_edge("research", END)
    graph.set_entry_point("research")
    return graph.compile()

# Export for ATP
agent_graph = build_graph()
```

```yaml
# atp.config.yaml
agents:
  langgraph-research:
    type: langgraph
    module: agents.my_langgraph_agent
    graph: agent_graph
    # Optional: entry point configuration
    config:
      recursion_limit: 50
```

### CrewAI Adapter

```python
# agents/my_crewai_agent.py
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Researcher",
    goal="Find comprehensive information",
    backstory="Expert researcher",
)

analyst = Agent(
    role="Analyst",
    goal="Analyze and summarize findings",
    backstory="Expert analyst",
)

def create_crew(task_description: str) -> Crew:
    research_task = Task(
        description=task_description,
        agent=researcher,
    )

    analysis_task = Task(
        description="Analyze the research findings",
        agent=analyst,
    )

    return Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
    )

# Export for ATP
crew_factory = create_crew
```

```yaml
# atp.config.yaml
agents:
  crewai-research:
    type: crewai
    module: agents.my_crewai_agent
    factory: crew_factory
```

---

## 4. Event Streaming

Для отладки и мониторинга полезно стримить события во время выполнения.

### HTTP SSE Streaming

```python
# agent_server.py (extended)
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

@app.post("/execute/stream")
async def execute_stream(request: ATPRequest):
    """Execute with event streaming."""

    async def event_generator():
        seq = 0

        # Start event
        yield format_sse(create_event(request.task_id, seq, "progress", {
            "message": "Starting task",
            "percentage": 0,
        }))
        seq += 1

        # Run agent with callbacks
        async for event in run_agent_with_events(request):
            yield format_sse(create_event(
                request.task_id, seq, event["type"], event["payload"]
            ))
            seq += 1

        # Final response
        response = await build_response(request.task_id)
        yield format_sse({"type": "response", "data": response.model_dump()})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

def format_sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

def create_event(task_id: str, seq: int, event_type: str, payload: dict) -> dict:
    return {
        "version": "1.0",
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence": seq,
        "event_type": event_type,
        "payload": payload,
    }
```

### Container Event Streaming

В container-режиме события пишутся в stderr:

```python
# agent_cli.py (with events)
import sys
import json

class EventEmitter:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.seq = 0

    def emit(self, event_type: str, payload: dict):
        event = {
            "version": "1.0",
            "task_id": self.task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self.seq,
            "event_type": event_type,
            "payload": payload,
        }
        print(json.dumps(event), file=sys.stderr, flush=True)
        self.seq += 1

    def tool_call(self, tool: str, input: dict, output: dict, duration_ms: int):
        self.emit("tool_call", {
            "tool": tool,
            "input": input,
            "output": output,
            "duration_ms": duration_ms,
            "status": "success",
        })

    def llm_request(self, model: str, input_tokens: int, output_tokens: int):
        self.emit("llm_request", {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

    def progress(self, message: str, percentage: int):
        self.emit("progress", {
            "message": message,
            "percentage": percentage,
        })

# Usage in agent
emitter = EventEmitter(task_id)
emitter.progress("Starting research", 10)
emitter.tool_call("web_search", {"query": "..."}, {"results": [...]}, 1500)
emitter.llm_request("claude-sonnet-4-20250514", 2000, 500)
```

---

## 5. Mock Tools

При тестировании полезно заменять реальные инструменты моками.

### Конфигурация Mock Tools

```yaml
# tests/mocks/web_search.yaml
tool: web_search
responses:
  - match:
      query: "Slack competitors"
    response:
      results:
        - title: "Microsoft Teams vs Slack"
          url: "https://example.com/teams-slack"
          snippet: "Comparison of enterprise communication tools..."
        - title: "Zoom vs Slack"
          url: "https://example.com/zoom-slack"
          snippet: "..."

  - match:
      query: "*"  # Default for any query
    response:
      results: []
      error: null
```

### Использование Mock Endpoint

Когда ATP запускает агента, он может передать `tools_endpoint` в context:

```json
{
  "context": {
    "tools_endpoint": "http://atp-mock-tools:8080/tools"
  }
}
```

Агент должен вызывать инструменты через этот endpoint:

```python
async def call_tool(tool_name: str, input: dict) -> dict:
    tools_endpoint = context.get("tools_endpoint")

    if tools_endpoint:
        # Call mock tools
        response = await httpx.post(
            f"{tools_endpoint}/{tool_name}",
            json={"input": input},
        )
        return response.json()["output"]
    else:
        # Call real tool
        return await real_tool_implementation(tool_name, input)
```

---

## 6. Handling Constraints

Агент должен соблюдать ограничения из ATP Request.

### Проверка ограничений

```python
class ConstraintChecker:
    def __init__(self, constraints: dict):
        self.max_steps = constraints.get("max_steps", float("inf"))
        self.max_tokens = constraints.get("max_tokens", float("inf"))
        self.timeout = constraints.get("timeout_seconds", 300)
        self.allowed_tools = set(constraints.get("allowed_tools") or [])

        self.current_steps = 0
        self.current_tokens = 0
        self.start_time = time.time()

    def check_step(self) -> bool:
        self.current_steps += 1
        return self.current_steps <= self.max_steps

    def check_tokens(self, tokens: int) -> bool:
        self.current_tokens += tokens
        return self.current_tokens <= self.max_tokens

    def check_timeout(self) -> bool:
        elapsed = time.time() - self.start_time
        return elapsed < self.timeout

    def check_tool(self, tool: str) -> bool:
        if not self.allowed_tools:  # Empty = all allowed
            return True
        return tool in self.allowed_tools

    def get_metrics(self) -> dict:
        return {
            "total_steps": self.current_steps,
            "total_tokens": self.current_tokens,
            "wall_time_seconds": time.time() - self.start_time,
        }

# Usage in agent
checker = ConstraintChecker(request.constraints)

for step in agent_loop():
    if not checker.check_step():
        raise ConstraintViolation("Max steps exceeded")

    if not checker.check_timeout():
        raise ConstraintViolation("Timeout exceeded")

    if step.tool_call:
        if not checker.check_tool(step.tool_call.name):
            raise ConstraintViolation(f"Tool {step.tool_call.name} not allowed")
```

---

## 7. Best Practices

### Error Handling

```python
async def execute(request: ATPRequest) -> ATPResponse:
    try:
        result = await run_agent(request)
        return build_success_response(request.task_id, result)

    except ConstraintViolation as e:
        # Agent hit a limit
        return ATPResponse(
            task_id=request.task_id,
            status="partial",
            artifacts=e.partial_results,
            metrics=checker.get_metrics(),
            error=str(e),
        )

    except asyncio.TimeoutError:
        return ATPResponse(
            task_id=request.task_id,
            status="timeout",
            artifacts=[],
            metrics=checker.get_metrics(),
            error="Execution timed out",
        )

    except Exception as e:
        logger.exception("Agent failed")
        return ATPResponse(
            task_id=request.task_id,
            status="failed",
            artifacts=[],
            metrics={},
            error=f"Internal error: {type(e).__name__}: {str(e)}",
        )
```

### Idempotency

Агент должен быть идемпотентным — повторный запуск с тем же task_id должен давать аналогичный результат (с учётом стохастичности LLM).

```python
# DON'T persist state between calls
global_state = {}  # BAD

# DO reset state for each request
def execute(request):
    state = {}  # Fresh state for each execution
    # ...
```

### Graceful Degradation

При недоступности инструментов агент должен адаптироваться:

```python
async def call_tool_with_fallback(tool: str, input: dict) -> dict:
    try:
        return await call_tool(tool, input, timeout=30)
    except ToolUnavailable:
        # Log and try alternative
        logger.warning(f"Tool {tool} unavailable, trying fallback")
        return await fallback_implementation(tool, input)
    except TimeoutError:
        # Tool too slow
        logger.warning(f"Tool {tool} timed out")
        return {"error": "timeout", "partial_result": None}
```

### Logging and Observability

```python
import structlog

logger = structlog.get_logger()

async def execute(request: ATPRequest) -> ATPResponse:
    logger = logger.bind(task_id=request.task_id)
    logger.info("Starting execution", task=request.task.description[:100])

    try:
        result = await run_agent(request)
        logger.info("Execution completed",
                   status="completed",
                   tokens=result.metrics.total_tokens)
        return result

    except Exception as e:
        logger.error("Execution failed", error=str(e), exc_info=True)
        raise
```

---

## 8. Testing Your Integration

### Validate ATP Response

```python
# test_integration.py
import pytest
from jsonschema import validate
import json

ATP_RESPONSE_SCHEMA = {...}  # From docs/04-protocol.md

def test_response_schema():
    """Verify agent produces valid ATP Response."""
    request = build_test_request()
    response = agent.execute(request)

    validate(response.model_dump(), ATP_RESPONSE_SCHEMA)

def test_event_streaming():
    """Verify events are emitted correctly."""
    events = []

    async for event in agent.execute_stream(request):
        events.append(event)

    # Check events are ordered
    sequences = [e["sequence"] for e in events if "sequence" in e]
    assert sequences == sorted(sequences)

    # Check required event types
    event_types = {e["event_type"] for e in events if "event_type" in e}
    assert "progress" in event_types

def test_constraints_respected():
    """Verify agent respects constraints."""
    request = build_test_request(constraints={"max_steps": 5})
    response = agent.execute(request)

    assert response.metrics.total_steps <= 5
```

### Run ATP Smoke Tests

```bash
# Start your agent
python agent_server.py &

# Validate integration
atp validate --agent=my-agent

# Run smoke tests
atp test --agent=my-agent --tags=smoke --verbose
```

---

## 9. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Response validation failed | Missing required fields | Check ATP Response schema |
| Timeout during test | Agent too slow | Increase timeout or optimize |
| Events not captured | Wrong stream format | Use SSE format with `data:` prefix |
| Container exits immediately | Missing entrypoint | Add ENTRYPOINT to Dockerfile |
| Tool calls not recorded | Not using mock endpoint | Check `tools_endpoint` in context |

### Debug Mode

```bash
# Run with verbose output
atp test --agent=my-agent --test=basic --verbose --debug

# Save full trace
atp test --agent=my-agent --test=basic --save-trace=./traces/

# Compare traces
atp diff-trace ./traces/run1.json ./traces/run2.json
```

---

## 10. Migration from Existing Agents

### From AutoGen

```python
# Original AutoGen agent
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config={...})
user_proxy = UserProxyAgent("user", code_execution_config={...})

# Wrap for ATP
class AutoGenATPWrapper:
    def __init__(self, assistant, user_proxy):
        self.assistant = assistant
        self.user_proxy = user_proxy

    async def execute(self, request: ATPRequest) -> ATPResponse:
        # Initiate chat
        self.user_proxy.initiate_chat(
            self.assistant,
            message=request.task.description,
        )

        # Collect results
        messages = self.assistant.chat_messages[self.user_proxy]
        output = messages[-1]["content"] if messages else ""

        return ATPResponse(
            task_id=request.task_id,
            status="completed",
            artifacts=[Artifact(type="structured", name="output", data={"content": output})],
            metrics=Metrics(),  # AutoGen doesn't expose metrics easily
        )
```

### From LangChain

```python
# Original LangChain agent
from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Wrap for ATP
class LangChainATPWrapper:
    def __init__(self, executor: AgentExecutor):
        self.executor = executor

    async def execute(self, request: ATPRequest) -> ATPResponse:
        result = await self.executor.ainvoke({
            "input": request.task.description,
        })

        return ATPResponse(
            task_id=request.task_id,
            status="completed",
            artifacts=[Artifact(
                type="structured",
                name="output",
                data={"content": result["output"]},
            )],
            metrics=Metrics(
                # Extract from callbacks if available
            ),
        )
```
