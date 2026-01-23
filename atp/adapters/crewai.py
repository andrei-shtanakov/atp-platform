"""CrewAI adapter for agents implemented with CrewAI."""

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


class CrewAIAdapterConfig(AdapterConfig):
    """Configuration for CrewAI adapter."""

    module: str = Field(..., description="Python module path containing the crew")
    crew: str = Field(..., description="Name of the crew variable or factory function")
    is_factory: bool = Field(
        False, description="Whether 'crew' is a factory function that returns a Crew"
    )
    factory_args: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to factory function"
    )
    verbose: bool = Field(False, description="Enable verbose output from CrewAI")
    memory: bool = Field(False, description="Enable memory for the crew")


class CrewAIAdapter(AgentAdapter):
    """
    Adapter for agents implemented with CrewAI.

    Loads CrewAI crews from Python modules and executes them
    with ATP Protocol translation.
    """

    def __init__(self, config: CrewAIAdapterConfig) -> None:
        """
        Initialize CrewAI adapter.

        Args:
            config: CrewAI adapter configuration.
        """
        super().__init__(config)
        self._config: CrewAIAdapterConfig = config
        self._crew: Any = None
        self._module: Any = None

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier."""
        return "crewai"

    def _load_crew(self) -> Any:
        """
        Load the CrewAI crew from the configured module.

        Returns:
            The loaded Crew object.

        Raises:
            AdapterConnectionError: If module or crew cannot be loaded.
        """
        if self._crew is not None:
            return self._crew

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
            crew_or_factory = getattr(self._module, self._config.crew)
        except AttributeError as e:
            raise AdapterConnectionError(
                f"Crew '{self._config.crew}' not found in module "
                f"'{self._config.module}'",
                endpoint=f"{self._config.module}.{self._config.crew}",
                adapter_type=self.adapter_type,
                cause=e,
            ) from e

        if self._config.is_factory:
            try:
                self._crew = crew_or_factory(**self._config.factory_args)
            except Exception as e:
                raise AdapterConnectionError(
                    f"Failed to create crew from factory '{self._config.crew}': {e}",
                    endpoint=f"{self._config.module}.{self._config.crew}",
                    adapter_type=self.adapter_type,
                    cause=e,
                ) from e
        else:
            self._crew = crew_or_factory

        return self._crew

    def _build_inputs(self, request: ATPRequest) -> dict[str, Any]:
        """
        Build inputs for the CrewAI crew from an ATP request.

        Args:
            request: ATP Request with task specification.

        Returns:
            Input dictionary for the crew.
        """
        inputs = request.task.input_data.copy() if request.task.input_data else {}

        # Add task description as a default input
        if "task" not in inputs:
            inputs["task"] = request.task.description
        if "description" not in inputs:
            inputs["description"] = request.task.description

        return inputs

    def _extract_output(self, result: Any) -> tuple[str, dict[str, Any]]:
        """
        Extract output from CrewAI result.

        Args:
            result: Result from crew execution.

        Returns:
            Tuple of (output string, full result data).
        """
        # Handle CrewOutput object
        if hasattr(result, "raw"):
            output = str(result.raw)
            data = {
                "raw": output,
                "tasks_output": [],
            }
            # Extract tasks output if available
            if hasattr(result, "tasks_output"):
                for task_output in result.tasks_output:
                    task_data = {
                        "description": getattr(task_output, "description", ""),
                        "raw": str(getattr(task_output, "raw", "")),
                    }
                    if hasattr(task_output, "agent"):
                        task_data["agent"] = str(task_output.agent)
                    data["tasks_output"].append(task_data)
            return output, data

        # Handle dict result
        if isinstance(result, dict):
            output = result.get("raw", result.get("output", str(result)))
            return str(output), result

        # Fallback to string conversion
        return str(result), {"raw": str(result)}

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
        Execute a task via CrewAI.

        Args:
            request: ATP Request with task specification.

        Returns:
            ATPResponse from the crew execution.

        Raises:
            AdapterConnectionError: If crew cannot be loaded.
            AdapterTimeoutError: If execution times out.
            AdapterError: If execution fails.
        """
        import asyncio

        crew = self._load_crew()
        inputs = self._build_inputs(request)

        start_time = time.time()
        timeout = request.constraints.get(
            "timeout_seconds", self._config.timeout_seconds
        )

        try:
            # Check for async kickoff
            if hasattr(crew, "kickoff_async"):
                try:
                    result = await asyncio.wait_for(
                        crew.kickoff_async(inputs=inputs),
                        timeout=timeout,
                    )
                except TimeoutError as e:
                    raise AdapterTimeoutError(
                        f"CrewAI execution timed out after {timeout}s",
                        timeout_seconds=timeout,
                        adapter_type=self.adapter_type,
                    ) from e
            elif hasattr(crew, "kickoff"):
                # Sync kickoff - run in executor
                loop = asyncio.get_event_loop()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: crew.kickoff(inputs=inputs)),
                        timeout=timeout,
                    )
                except TimeoutError as e:
                    raise AdapterTimeoutError(
                        f"CrewAI execution timed out after {timeout}s",
                        timeout_seconds=timeout,
                        adapter_type=self.adapter_type,
                    ) from e
            else:
                raise AdapterError(
                    "Crew does not have kickoff or kickoff_async method",
                    adapter_type=self.adapter_type,
                )

            wall_time = time.time() - start_time

            # Extract output
            output, data = self._extract_output(result)

            # Extract metrics from crew if available
            metrics = self._extract_metrics(crew, result, wall_time)

            return ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                artifacts=[
                    {
                        "type": "structured",
                        "name": "output",
                        "data": {"content": output, **data},
                    }
                ],
                metrics=metrics,
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
                metrics=Metrics(wall_time_seconds=wall_time),
            )

    def _extract_metrics(self, crew: Any, result: Any, wall_time: float) -> Metrics:
        """
        Extract metrics from CrewAI execution.

        Args:
            crew: The CrewAI crew object.
            result: Result from crew execution.
            wall_time: Wall clock time in seconds.

        Returns:
            Metrics object with extracted data.
        """
        total_tokens = None
        llm_calls = 0
        tool_calls = 0
        total_steps = 0

        # Try to extract token usage from result
        if hasattr(result, "token_usage"):
            usage = result.token_usage
            if hasattr(usage, "total_tokens"):
                total_tokens = usage.total_tokens

        # Count tasks as steps
        if hasattr(result, "tasks_output"):
            total_steps = len(result.tasks_output)
            # Estimate LLM calls based on tasks
            llm_calls = total_steps

        # Try to get metrics from crew
        if hasattr(crew, "tasks"):
            total_steps = len(crew.tasks)

        if hasattr(crew, "agents"):
            # Count agents for potential tool usage
            for agent in crew.agents:
                if hasattr(agent, "tools") and agent.tools:
                    tool_calls += len(agent.tools)

        return Metrics(
            total_tokens=total_tokens,
            total_steps=total_steps,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            wall_time_seconds=wall_time,
        )

    async def stream_events(
        self, request: ATPRequest
    ) -> AsyncIterator[ATPEvent | ATPResponse]:
        """
        Execute a task with event streaming via CrewAI.

        CrewAI doesn't natively support streaming, so this uses
        callback mechanisms to emit events.

        Args:
            request: ATP Request with task specification.

        Yields:
            ATPEvent objects during execution.
            Final ATPResponse when complete.

        Raises:
            AdapterConnectionError: If crew cannot be loaded.
            AdapterTimeoutError: If execution times out.
            AdapterError: If execution fails.
        """
        import asyncio

        crew = self._load_crew()
        inputs = self._build_inputs(request)

        start_time = time.time()
        sequence = 0
        timeout = request.constraints.get(
            "timeout_seconds", self._config.timeout_seconds
        )

        # Emit start event
        yield self._create_event(
            request,
            EventType.PROGRESS,
            {
                "message": "Starting CrewAI execution",
                "current_step": 0,
            },
            sequence,
        )
        sequence += 1

        # Emit events for each task in the crew
        if hasattr(crew, "tasks"):
            for i, task in enumerate(crew.tasks):
                task_desc = getattr(task, "description", f"Task {i + 1}")
                yield self._create_event(
                    request,
                    EventType.PROGRESS,
                    {
                        "message": f"Task defined: {task_desc[:100]}...",
                        "current_step": i + 1,
                        "task_index": i,
                    },
                    sequence,
                )
                sequence += 1

        # Emit events for each agent
        if hasattr(crew, "agents"):
            for agent in crew.agents:
                agent_role = getattr(agent, "role", "Agent")
                yield self._create_event(
                    request,
                    EventType.PROGRESS,
                    {
                        "message": f"Agent ready: {agent_role}",
                        "agent": agent_role,
                    },
                    sequence,
                )
                sequence += 1

        try:
            # Execute the crew
            if hasattr(crew, "kickoff_async"):
                try:
                    result = await asyncio.wait_for(
                        crew.kickoff_async(inputs=inputs),
                        timeout=timeout,
                    )
                except TimeoutError as e:
                    raise AdapterTimeoutError(
                        f"CrewAI execution timed out after {timeout}s",
                        timeout_seconds=timeout,
                        adapter_type=self.adapter_type,
                    ) from e
            elif hasattr(crew, "kickoff"):
                loop = asyncio.get_event_loop()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: crew.kickoff(inputs=inputs)),
                        timeout=timeout,
                    )
                except TimeoutError as e:
                    raise AdapterTimeoutError(
                        f"CrewAI execution timed out after {timeout}s",
                        timeout_seconds=timeout,
                        adapter_type=self.adapter_type,
                    ) from e
            else:
                raise AdapterError(
                    "Crew does not have kickoff or kickoff_async method",
                    adapter_type=self.adapter_type,
                )

            wall_time = time.time() - start_time

            # Emit events for task outputs
            if hasattr(result, "tasks_output"):
                for i, task_output in enumerate(result.tasks_output):
                    task_desc = getattr(task_output, "description", f"Task {i + 1}")
                    agent = str(getattr(task_output, "agent", "unknown"))

                    yield self._create_event(
                        request,
                        EventType.LLM_REQUEST,
                        {
                            "model": "crewai",
                            "task": task_desc[:100],
                            "agent": agent,
                        },
                        sequence,
                    )
                    sequence += 1

            # Extract output
            output, data = self._extract_output(result)
            metrics = self._extract_metrics(crew, result, wall_time)

            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.COMPLETED,
                artifacts=[
                    {
                        "type": "structured",
                        "name": "output",
                        "data": {"content": output, **data},
                    }
                ],
                metrics=metrics,
            )

        except AdapterTimeoutError:
            raise
        except AdapterError:
            raise
        except Exception as e:
            wall_time = time.time() - start_time
            yield ATPResponse(
                task_id=request.task_id,
                status=ResponseStatus.FAILED,
                error=str(e),
                metrics=Metrics(wall_time_seconds=wall_time),
            )

    async def health_check(self) -> bool:
        """
        Check if the CrewAI crew can be loaded.

        Returns:
            True if crew loads successfully, False otherwise.
        """
        try:
            self._load_crew()
            return True
        except AdapterConnectionError:
            return False

    async def cleanup(self) -> None:
        """Release any resources."""
        self._crew = None
        self._module = None
