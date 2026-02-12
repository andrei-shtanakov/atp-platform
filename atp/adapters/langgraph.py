"""LangGraph adapter for agents implemented with LangGraph."""

import importlib
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from pydantic import Field

from atp.protocol import (
    ATPEvent,
    ATPRequest,
    ATPResponse,
    EventType,
    Metrics,
    ResponseStatus,
)

from .base import AdapterConfig, AgentAdapter
from .exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterTimeoutError,
)


class LangGraphAdapterConfig(AdapterConfig):
    """Configuration for LangGraph adapter."""

    module: str = Field(..., description="Python module path containing the graph")
    graph: str = Field(..., description="Name of the graph variable in the module")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="LangGraph configuration (e.g., recursion_limit)",
    )
    checkpointer: str | None = Field(
        None, description="Optional checkpointer class path for stateful execution"
    )
    input_key: str = Field(
        "messages", description="Key to use for input in the graph state"
    )
    output_key: str | None = Field(
        None, description="Key to extract output from (None = use full state)"
    )


class LangGraphAdapter(AgentAdapter):
    """
    Adapter for agents implemented with LangGraph.

    Loads LangGraph graphs from Python modules and executes them
    with ATP Protocol translation.
    """

    def __init__(self, config: LangGraphAdapterConfig) -> None:
        """
        Initialize LangGraph adapter.

        Args:
            config: LangGraph adapter configuration.
        """
        super().__init__(config)
        self._config: LangGraphAdapterConfig = config
        self._graph: Any = None
        self._module: Any = None

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "langgraph"

    def _load_graph(self) -> Any:
        """
        Load the LangGraph graph from the configured module.

        Returns:
            The loaded graph object.

        Raises:
            AdapterConnectionError: If module or graph cannot be loaded.
        """
        if self._graph is not None:
            return self._graph

        try:
            self._module = importlib.import_module(self._config.module)
        except ImportError as e:
            raise AdapterConnectionError(
                f"Failed to import module '{self._config.module}': {e}",
                endpoint=self._config.module,
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

        try:
            self._graph = getattr(self._module, self._config.graph)
        except AttributeError as e:
            raise AdapterConnectionError(
                f"Graph '{self._config.graph}' not found in module "
                f"'{self._config.module}'",
                endpoint=f"{self._config.module}.{self._config.graph}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

        return self._graph

    def _build_input_state(self, request: ATPRequest) -> dict[str, Any]:
        """
        Build the input state for the graph from an ATP request.

        Args:
            request: ATP Request with task specification.

        Returns:
            Input state dictionary for the graph.
        """
        # Build the input based on the configured input_key
        input_data = request.task.input_data or {}

        # If there's a task description, include it in the input
        if self._config.input_key == "messages":
            # For message-based graphs, create a message from the task
            return {
                "messages": [{"role": "user", "content": request.task.description}],
                **input_data,
            }
        else:
            # For other input keys, use the task description directly
            return {
                self._config.input_key: request.task.description,
                **input_data,
            }

    def _extract_output(self, state: dict[str, Any]) -> str:
        """
        Extract the output from the final graph state.

        Args:
            state: Final state from graph execution.

        Returns:
            Extracted output as a string.
        """
        if self._config.output_key:
            output = state.get(self._config.output_key, "")
            if isinstance(output, list) and output:
                # Handle list of messages
                last_item = output[-1]
                if isinstance(last_item, dict):
                    return str(last_item.get("content", last_item))
                return str(last_item)
            return str(output)

        # Default: try to get messages or return full state
        if "messages" in state:
            messages = state["messages"]
            if messages and isinstance(messages, list):
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    return str(last_msg.get("content", last_msg))
                # Handle LangChain message objects
                if hasattr(last_msg, "content"):
                    return str(last_msg.content)
                return str(last_msg)

        return str(state)

    def _create_event(
        self,
        request: ATPRequest,
        event_type: EventType,
        payload: dict[str, Any],
        sequence: int,
    ) -> ATPEvent:
        """Create an ATP event."""
        return ATPEvent(
            task_id=request.task_id,
            timestamp=datetime.now(),
            sequence=sequence,
            event_type=event_type,
            payload=payload,
        )

    async def execute(self, request: ATPRequest) -> ATPResponse:
        """
        Execute a task via LangGraph.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse from the graph execution.

        Raises:
            AdapterConnectionError: If graph cannot be loaded.
            AdapterTimeoutError: If execution times out.
            AdapterError: If execution fails.
        """
        import asyncio

        graph = self._load_graph()
        input_state = self._build_input_state(request)

        # Prepare config with recursion limit and other settings
        config = {"configurable": self._config.config.copy()}
        if "recursion_limit" not in config["configurable"]:
            max_steps = request.constraints.get("max_steps")
            if max_steps:
                config["configurable"]["recursion_limit"] = max_steps

        start_time = time.time()
        total_steps = 0
        llm_calls = 0
        tool_calls = 0

        try:
            # Check if graph has async invoke method
            if hasattr(graph, "ainvoke"):
                # Run with timeout
                timeout = request.constraints.get(
                    "timeout_seconds", self._config.timeout_seconds
                )
                try:
                    result = await asyncio.wait_for(
                        graph.ainvoke(input_state, config),
                        timeout=timeout,
                    )
                except TimeoutError as e:
                    raise AdapterTimeoutError(
                        f"LangGraph execution timed out after {timeout}s",
                        timeout_seconds=timeout,
                        adapter_type=self.adapter_type,
                    ) from e
            elif hasattr(graph, "invoke"):
                # Sync invoke - run in executor
                timeout = request.constraints.get(
                    "timeout_seconds", self._config.timeout_seconds
                )
                try:
                    result = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None, lambda: graph.invoke(input_state, config)
                        ),
                        timeout=timeout,
                    )
                except TimeoutError as e:
                    raise AdapterTimeoutError(
                        f"LangGraph execution timed out after {timeout}s",
                        timeout_seconds=timeout,
                        adapter_type=self.adapter_type,
                    ) from e
            else:
                raise AdapterError(
                    "Graph does not have invoke or ainvoke method",
                    adapter_type=self.adapter_type,
                )

            wall_time = time.time() - start_time

            # Extract output
            output = self._extract_output(result)

            # Count steps from messages if available
            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
                if isinstance(messages, list):
                    total_steps = len(messages)
                    # Count LLM calls (assistant messages) and tool calls
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            if role == "assistant":
                                llm_calls += 1
                            elif role == "tool":
                                tool_calls += 1
                        elif hasattr(msg, "type"):
                            if msg.type == "ai":
                                llm_calls += 1
                            elif msg.type == "tool":
                                tool_calls += 1

            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                artifacts=[
                    {
                        "type": "structured",
                        "name": "output",
                        "data": {"content": output, "state": result},
                    }
                ],
                metrics=Metrics(
                    total_steps=total_steps,
                    llm_calls=llm_calls,
                    tool_calls=tool_calls,
                    wall_time_seconds=wall_time,
                ),
            )

        except AdapterTimeoutError:
            raise
        except AdapterError:
            raise
        except Exception as e:
            wall_time = time.time() - start_time
            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(
                    wall_time_seconds=wall_time,
                ),
            )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """
        Execute a task with event streaming via LangGraph.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects during execution.
            Final ATPResponse when complete.

        Raises:
            AdapterConnectionError: If graph cannot be loaded.
            AdapterTimeoutError: If execution times out.
            AdapterError: If execution fails.
        """
        import asyncio

        graph = self._load_graph()
        input_state = self._build_input_state(request)

        # Prepare config
        config = {"configurable": self._config.config.copy()}
        if "recursion_limit" not in config["configurable"]:
            max_steps = request.constraints.get("max_steps")
            if max_steps:
                config["configurable"]["recursion_limit"] = max_steps

        start_time = time.time()
        sequence = 0
        total_steps = 0
        llm_calls = 0
        tool_calls = 0
        final_state: dict[str, Any] = {}
        timeout = request.constraints.get(
            "timeout_seconds", self._config.timeout_seconds
        )
        deadline = time.time() + timeout

        try:
            # Check if graph supports streaming
            if hasattr(graph, "astream"):
                # Async streaming
                async for event in graph.astream(input_state, config):
                    if time.time() > deadline:
                        raise AdapterTimeoutError(
                            f"LangGraph execution timed out after {timeout}s",
                            timeout_seconds=timeout,
                            adapter_type=self.adapter_type,
                        )

                    total_steps += 1
                    final_state = event if isinstance(event, dict) else {}

                    # Extract event type and payload
                    atp_event = self._convert_langgraph_event(request, event, sequence)
                    if atp_event:
                        yield atp_event
                        sequence += 1

                        # Track metrics
                        if atp_event.event_type == EventType.LLM_REQUEST:
                            llm_calls += 1
                        elif atp_event.event_type == EventType.TOOL_CALL:
                            tool_calls += 1

            elif hasattr(graph, "stream"):
                # Sync streaming - wrap in async
                loop = asyncio.get_running_loop()

                def stream_generator():
                    return list(graph.stream(input_state, config))

                events = await asyncio.wait_for(
                    loop.run_in_executor(None, stream_generator),
                    timeout=timeout,
                )

                for event in events:
                    total_steps += 1
                    final_state = event if isinstance(event, dict) else {}

                    atp_event = self._convert_langgraph_event(request, event, sequence)
                    if atp_event:
                        yield atp_event
                        sequence += 1

                        if atp_event.event_type == EventType.LLM_REQUEST:
                            llm_calls += 1
                        elif atp_event.event_type == EventType.TOOL_CALL:
                            tool_calls += 1
            else:
                # No streaming support - fall back to execute
                yield self._create_event(
                    request,
                    EventType.PROGRESS,
                    {"message": "Graph does not support streaming, using execute"},
                    sequence,
                )
                response = await self.execute(request)
                yield response
                return

            wall_time = time.time() - start_time

            # Extract output from final state
            output = self._extract_output(final_state)

            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                artifacts=[
                    {
                        "type": "structured",
                        "name": "output",
                        "data": {"content": output, "state": final_state},
                    }
                ],
                metrics=Metrics(
                    total_steps=total_steps,
                    llm_calls=llm_calls,
                    tool_calls=tool_calls,
                    wall_time_seconds=wall_time,
                ),
            )

        except AdapterTimeoutError:
            raise
        except TimeoutError as e:
            raise AdapterTimeoutError(
                f"LangGraph execution timed out after {timeout}s",
                timeout_seconds=timeout,
                adapter_type=self.adapter_type,
            ) from e
        except Exception as e:
            wall_time = time.time() - start_time
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(
                    wall_time_seconds=wall_time,
                ),
            )

    def _convert_langgraph_event(
        self, request: ATPRequest, event: Any, sequence: int
    ) -> ATPEvent | None:
        """
        Convert a LangGraph event to an ATP event.

        Args:
            request: ATP Request for context.
            event: LangGraph event to convert.
            sequence: Current sequence number.

        Returns:
            ATPEvent or None if event cannot be converted.
        """
        if isinstance(event, dict):
            # Check for common LangGraph event patterns
            # Node output event
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    # Check if it's a messages update
                    if "messages" in node_output:
                        messages = node_output["messages"]
                        if messages and isinstance(messages, list):
                            last_msg = messages[-1]
                            return self._message_to_event(
                                request, last_msg, node_name, sequence
                            )

                    # Generic node output
                    return self._create_event(
                        request,
                        EventType.PROGRESS,
                        {
                            "message": f"Node '{node_name}' completed",
                            "node": node_name,
                            "output": node_output,
                        },
                        sequence,
                    )

        return self._create_event(
            request,
            EventType.PROGRESS,
            {"message": str(event)},
            sequence,
        )

    def _message_to_event(
        self, request: ATPRequest, message: Any, node_name: str, sequence: int
    ) -> ATPEvent:
        """Convert a LangGraph message to an ATP event."""
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "assistant":
                return self._create_event(
                    request,
                    EventType.LLM_REQUEST,
                    {
                        "model": "unknown",
                        "node": node_name,
                        "content": content,
                    },
                    sequence,
                )
            elif role == "tool":
                return self._create_event(
                    request,
                    EventType.TOOL_CALL,
                    {
                        "tool": message.get("name", "unknown"),
                        "output": content,
                        "node": node_name,
                    },
                    sequence,
                )
            else:
                return self._create_event(
                    request,
                    EventType.PROGRESS,
                    {
                        "message": content,
                        "role": role,
                        "node": node_name,
                    },
                    sequence,
                )

        # Handle LangChain message objects
        if hasattr(message, "type"):
            msg_type = message.type
            content = getattr(message, "content", str(message))

            if msg_type == "ai":
                return self._create_event(
                    request,
                    EventType.LLM_REQUEST,
                    {
                        "model": getattr(message, "response_metadata", {}).get(
                            "model", "unknown"
                        ),
                        "node": node_name,
                        "content": content,
                    },
                    sequence,
                )
            elif msg_type == "tool":
                return self._create_event(
                    request,
                    EventType.TOOL_CALL,
                    {
                        "tool": getattr(message, "name", "unknown"),
                        "output": content,
                        "node": node_name,
                    },
                    sequence,
                )

        return self._create_event(
            request,
            EventType.PROGRESS,
            {"message": str(message), "node": node_name},
            sequence,
        )

    async def health_check(self) -> bool:
        """
        Check if the LangGraph graph can be loaded.

        Returns:
            True if graph loads successfully, False otherwise.
        """
        try:
            self._load_graph()
            return True
        except AdapterConnectionError:
            return False

    async def cleanup(self) -> None:
        """Release any resources."""
        self._graph = None
        self._module = None
