# Adapter Development Guide

This guide explains how to develop custom adapters for ATP and how to use the built-in framework adapters (LangGraph, CrewAI, AutoGen).

## Overview

Adapters translate between the ATP Protocol and framework-specific APIs. They enable ATP to work with agents implemented using different frameworks without requiring changes to the agent code.

## Built-in Framework Adapters

ATP includes adapters for popular agent frameworks:

| Adapter | Framework | Use Case |
|---------|-----------|----------|
| `LangGraphAdapter` | LangGraph | Stateful agent graphs |
| `CrewAIAdapter` | CrewAI | Multi-agent crews |
| `AutoGenAdapter` | Microsoft AutoGen | Conversational agents |

## LangGraph Adapter

### Configuration

```python
from atp.adapters import LangGraphAdapter, LangGraphAdapterConfig

config = LangGraphAdapterConfig(
    module="my_agent.graph",      # Python module path
    graph="agent_graph",          # Graph variable name in module
    config={                      # Optional LangGraph config
        "recursion_limit": 50,
    },
    input_key="messages",         # Key for input state (default: "messages")
    output_key=None,              # Key to extract output (None = auto-detect)
    timeout_seconds=300.0,        # Execution timeout
)

adapter = LangGraphAdapter(config)
```

### YAML Configuration

```yaml
agents:
  my-langgraph-agent:
    type: langgraph
    module: "agents.research"
    graph: "agent_graph"
    config:
      recursion_limit: 50
    input_key: "messages"
    timeout_seconds: 300
```

### Example Agent Module

```python
# agents/research.py
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

def create_graph():
    # Define your graph here
    workflow = StateGraph(dict)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)

    # Add edges
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)

    workflow.set_entry_point("researcher")

    return workflow.compile()

# Export the compiled graph
agent_graph = create_graph()
```

### How It Works

1. **Graph Loading**: The adapter imports the specified module and retrieves the graph object.

2. **Input Mapping**: ATP Request is converted to LangGraph input state:
   ```python
   # For input_key="messages"
   {
       "messages": [{"role": "user", "content": task_description}],
       **request.task.input_data
   }
   ```

3. **Execution**: The adapter calls `graph.ainvoke()` or `graph.invoke()` with timeout handling.

4. **Event Streaming**: During streaming, node outputs are converted to ATP Events:
   - Assistant messages → `EventType.LLM_REQUEST`
   - Tool messages → `EventType.TOOL_CALL`
   - Other outputs → `EventType.PROGRESS`

5. **Output Extraction**: The final state is processed to extract the response.

## CrewAI Adapter

### Configuration

```python
from atp.adapters import CrewAIAdapter, CrewAIAdapterConfig

config = CrewAIAdapterConfig(
    module="my_agent.crew",       # Python module path
    crew="research_crew",         # Crew variable or factory name
    is_factory=False,             # True if crew is a factory function
    factory_args={},              # Args for factory function
    verbose=False,                # Enable verbose output
    memory=False,                 # Enable crew memory
    timeout_seconds=300.0,        # Execution timeout
)

adapter = CrewAIAdapter(config)
```

### YAML Configuration

```yaml
agents:
  my-crewai-agent:
    type: crewai
    module: "agents.research_crew"
    crew: "create_crew"
    is_factory: true
    factory_args:
      model: "gpt-4"
      verbose: true
    timeout_seconds: 300
```

### Example Crew Module

```python
# agents/research_crew.py
from crewai import Agent, Crew, Task

def create_crew(model: str = "gpt-4", verbose: bool = False) -> Crew:
    """Factory function to create a research crew."""

    researcher = Agent(
        role="Researcher",
        goal="Research and gather information",
        backstory="An expert researcher with deep knowledge",
        model=model,
        verbose=verbose,
    )

    writer = Agent(
        role="Writer",
        goal="Write clear and concise reports",
        backstory="A skilled technical writer",
        model=model,
        verbose=verbose,
    )

    research_task = Task(
        description="{task}",  # Filled from ATP request
        agent=researcher,
    )

    writing_task = Task(
        description="Write a report based on research findings",
        agent=writer,
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
    )

# Or export a pre-created crew
research_crew = create_crew()
```

### How It Works

1. **Crew Loading**: The adapter imports the module and retrieves the crew object (or calls the factory function).

2. **Input Mapping**: ATP Request is converted to crew inputs:
   ```python
   {
       "task": request.task.description,
       "description": request.task.description,
       **request.task.input_data
   }
   ```

3. **Execution**: The adapter calls `crew.kickoff_async()` or `crew.kickoff()`.

4. **Event Streaming**: Pre-execution events include task and agent definitions. Post-execution events include task outputs.

5. **Metrics Extraction**: Token usage and task counts are extracted from the result.

## AutoGen Adapter

### Configuration

```python
from atp.adapters import AutoGenAdapter, AutoGenAdapterConfig

config = AutoGenAdapterConfig(
    module="my_agent.autogen",    # Python module path
    agent="assistant",            # Agent variable or factory name
    is_factory=False,             # True if agent is a factory function
    factory_args={},              # Args for factory function
    user_proxy="user_proxy",      # Optional user proxy agent name
    max_consecutive_auto_reply=10,  # Max auto-replies
    human_input_mode="NEVER",     # NEVER, TERMINATE, or ALWAYS
    timeout_seconds=300.0,        # Execution timeout
)

adapter = AutoGenAdapter(config)
```

### YAML Configuration

```yaml
agents:
  my-autogen-agent:
    type: autogen
    module: "agents.chat_agent"
    agent: "assistant"
    user_proxy: "user_proxy"
    max_consecutive_auto_reply: 5
    human_input_mode: "NEVER"
    timeout_seconds: 300
```

### Example AutoGen Module

```python
# agents/chat_agent.py
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Load LLM config
config_list = config_list_from_json("OAI_CONFIG_LIST")

# Create assistant agent
assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "temperature": 0,
    },
    system_message="You are a helpful AI assistant.",
)

# Create user proxy (optional - adapter creates default if not provided)
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "workspace"},
)
```

### How It Works

1. **Agent Loading**: The adapter imports the module and retrieves the agent (and optional user proxy).

2. **User Proxy**: If no user proxy is configured, a default one is created with the specified settings.

3. **Message Building**: ATP Request is converted to a chat message:
   ```
   Task description

   key1: value1
   key2: value2
   ```

4. **Execution**: The adapter calls `user_proxy.a_initiate_chat()` or `user_proxy.initiate_chat()`.

5. **Chat History**: The full conversation is captured and included in the response.

## Creating Custom Adapters

### Base Interface

All adapters must implement the `AgentAdapter` abstract base class:

```python
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from atp.adapters.base import AdapterConfig, AgentAdapter
from atp.protocol import ATPEvent, ATPRequest, ATPResponse


class MyAdapterConfig(AdapterConfig):
    """Configuration for my custom adapter."""

    my_setting: str
    optional_setting: int = 10


class MyAdapter(AgentAdapter):
    """Custom adapter implementation."""

    def __init__(self, config: MyAdapterConfig) -> None:
        super().__init__(config)
        self._config = config

    @property
    def adapter_type(self) -> str:
        """Return unique adapter identifier."""
        return "my_adapter"

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """Execute task synchronously."""
        # 1. Translate ATP request to agent format
        agent_input = self._translate_request(request)

        # 2. Execute agent
        result = await self._run_agent(agent_input)

        # 3. Build ATP response
        return self._build_response(request, result)

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """Execute task with event streaming."""
        # Yield events during execution
        yield self._create_progress_event(request, "Starting...", 0)

        # Execute and yield events
        result = await self._run_agent(...)

        # Yield final response
        yield self._build_response(request, result)

    async def health_check(self) -> bool:
        """Check if agent is available."""
        try:
            # Verify agent is accessible
            return True
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Release resources."""
        pass
```

### Registering Custom Adapters

```python
from atp.adapters import AdapterRegistry, get_registry

# Get the global registry
registry = get_registry()

# Register your adapter
registry.register("my_adapter", MyAdapter, MyAdapterConfig)

# Now you can create adapters
adapter = registry.create("my_adapter", {"my_setting": "value"})
```

### Error Handling

Use the provided exception classes for consistent error handling:

```python
from atp.adapters.exceptions import (
    AdapterError,
    AdapterTimeoutError,
    AdapterConnectionError,
    AdapterResponseError,
)

class MyAdapter(AgentAdapter):
    async def execute(self, request: ATPRequest) -> ATPResponse:
        try:
            result = await self._run_agent(request)
        except TimeoutError as e:
            raise AdapterTimeoutError(
                f"Execution timed out after {self._config.timeout_seconds}s",
                timeout_seconds=self._config.timeout_seconds,
                adapter_type=self.adapter_type,
            ) from e
        except ConnectionError as e:
            raise AdapterConnectionError(
                f"Cannot connect to agent",
                endpoint=self._endpoint,
                adapter_type=self.adapter_type,
                cause=e,
            ) from e
```

### Event Creation Helper

Create consistent ATP events:

```python
from datetime import datetime
from atp.protocol import ATPEvent, EventType

def _create_event(
    self,
    request: ATPRequest,
    event_type: EventType,
    payload: dict,
    sequence: int,
) -> ATPEvent:
    return ATPEvent(
        task_id=request.task_id,
        timestamp=datetime.now(),
        sequence=sequence,
        event_type=event_type,
        payload=payload,
    )
```

## Testing Adapters

### Unit Testing Pattern

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from atp.adapters import MyAdapter, MyAdapterConfig
from atp.protocol import ATPRequest, ResponseStatus, Task


@pytest.fixture
def config() -> MyAdapterConfig:
    return MyAdapterConfig(my_setting="test")


@pytest.fixture
def request() -> ATPRequest:
    return ATPRequest(
        task_id="test-123",
        task=Task(description="Test task"),
    )


@pytest.mark.anyio
async def test_execute_success(config, request):
    adapter = MyAdapter(config)

    # Mock internal methods
    with patch.object(adapter, "_run_agent") as mock_run:
        mock_run.return_value = {"output": "result"}

        response = await adapter.execute(request)

    assert response.status == ResponseStatus.COMPLETED


@pytest.mark.anyio
async def test_stream_events(config, request):
    adapter = MyAdapter(config)

    events = []
    async for event in adapter.stream_events(request):
        events.append(event)

    # Check events were yielded
    assert len(events) > 0
    # Last item should be response
    assert isinstance(events[-1], ATPResponse)
```

### Integration Testing

Test with real (or realistic mock) agents:

```python
@pytest.mark.integration
@pytest.mark.anyio
async def test_real_agent_execution(request):
    config = MyAdapterConfig(
        my_setting="real_config",
        timeout_seconds=60,
    )
    adapter = MyAdapter(config)

    async with adapter:
        # Health check first
        assert await adapter.health_check() is True

        # Execute
        response = await adapter.execute(request)

        assert response.status == ResponseStatus.COMPLETED
        assert response.metrics is not None
```

## Best Practices

### 1. Always Support Timeout

Handle timeouts gracefully:

```python
import asyncio

async def execute(self, request: ATPRequest) -> ATPResponse:
    timeout = request.constraints.get(
        "timeout_seconds", self._config.timeout_seconds
    )

    try:
        result = await asyncio.wait_for(
            self._run_agent(request),
            timeout=timeout,
        )
    except asyncio.TimeoutError as e:
        raise AdapterTimeoutError(
            f"Timed out after {timeout}s",
            timeout_seconds=timeout,
            adapter_type=self.adapter_type,
        ) from e
```

### 2. Collect Metrics

Always populate metrics when possible:

```python
from atp.protocol import Metrics

metrics = Metrics(
    total_tokens=token_count,
    total_steps=step_count,
    llm_calls=llm_call_count,
    tool_calls=tool_call_count,
    wall_time_seconds=elapsed_time,
)
```

### 3. Use Context Manager Pattern

Support async context manager for resource management:

```python
async def __aenter__(self):
    await self._initialize()
    return self

async def __aexit__(self, *args):
    await self.cleanup()
```

### 4. Handle Both Sync and Async

Support frameworks that may have sync-only APIs:

```python
import asyncio

if hasattr(agent, "ainvoke"):
    result = await agent.ainvoke(input)
elif hasattr(agent, "invoke"):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: agent.invoke(input)
    )
else:
    raise AdapterError("Agent has no invoke method")
```

### 5. Fail Gracefully

Return failed responses instead of raising for recoverable errors:

```python
try:
    result = await self._run_agent(request)
except RecoverableError as e:
    return ATPResponse(
        task_id=request.task_id,
        status=ResponseStatus.FAILED,
        error=str(e),
        metrics=Metrics(wall_time_seconds=elapsed),
    )
```

## See Also

- [Adapter Configuration Reference](../reference/adapters.md)
- [ATP Protocol](../04-protocol.md)
- [Testing Guide](../reference/troubleshooting.md)
