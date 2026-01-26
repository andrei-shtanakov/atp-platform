"""Sample events data for timeline and comparison tests.

This module provides factory functions for creating ATP events
that can be used in testing timeline visualization and agent comparison features.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Event types matching the ATP protocol."""

    TOOL_CALL = "tool_call"
    LLM_REQUEST = "llm_request"
    REASONING = "reasoning"
    ERROR = "error"
    PROGRESS = "progress"


def create_tool_call_event(
    task_id: str = "task-001",
    sequence: int = 0,
    tool: str = "web_search",
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    status: str = "success",
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Create a tool call event.

    Args:
        task_id: The task identifier.
        sequence: Event sequence number.
        tool: Tool name (e.g., "web_search", "file_read", "code_exec").
        input_data: Tool input parameters.
        output_data: Tool output/result.
        duration_ms: Tool execution duration in milliseconds.
        status: Call status ("started", "success", "error").
        timestamp: Event timestamp (defaults to now).

    Returns:
        Dictionary representing the event.
    """
    if timestamp is None:
        timestamp = datetime.now()

    if input_data is None:
        input_data = {"query": "test query"}

    payload: dict[str, Any] = {
        "tool": tool,
        "input": input_data,
        "status": status,
    }

    if output_data is not None:
        payload["output"] = output_data

    if duration_ms is not None:
        payload["duration_ms"] = duration_ms

    return {
        "version": "1.0",
        "task_id": task_id,
        "timestamp": timestamp.isoformat(),
        "sequence": sequence,
        "event_type": EventType.TOOL_CALL.value,
        "payload": payload,
    }


def create_llm_request_event(
    task_id: str = "task-001",
    sequence: int = 0,
    model: str = "claude-sonnet-4-20250514",
    input_tokens: int | None = 500,
    output_tokens: int | None = 200,
    duration_ms: float | None = 1500.0,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Create an LLM request event.

    Args:
        task_id: The task identifier.
        sequence: Event sequence number.
        model: Model name/identifier.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        duration_ms: Request duration in milliseconds.
        timestamp: Event timestamp (defaults to now).

    Returns:
        Dictionary representing the event.
    """
    if timestamp is None:
        timestamp = datetime.now()

    payload: dict[str, Any] = {
        "model": model,
    }

    if input_tokens is not None:
        payload["input_tokens"] = input_tokens

    if output_tokens is not None:
        payload["output_tokens"] = output_tokens

    if duration_ms is not None:
        payload["duration_ms"] = duration_ms

    return {
        "version": "1.0",
        "task_id": task_id,
        "timestamp": timestamp.isoformat(),
        "sequence": sequence,
        "event_type": EventType.LLM_REQUEST.value,
        "payload": payload,
    }


def create_reasoning_event(
    task_id: str = "task-001",
    sequence: int = 0,
    thought: str | None = None,
    plan: str | None = None,
    step: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Create a reasoning event.

    Args:
        task_id: The task identifier.
        sequence: Event sequence number.
        thought: Agent's thought process.
        plan: Execution plan or strategy.
        step: Current step description.
        timestamp: Event timestamp (defaults to now).

    Returns:
        Dictionary representing the event.
    """
    if timestamp is None:
        timestamp = datetime.now()

    payload: dict[str, Any] = {}
    if thought is not None:
        payload["thought"] = thought
    if plan is not None:
        payload["plan"] = plan
    if step is not None:
        payload["step"] = step

    return {
        "version": "1.0",
        "task_id": task_id,
        "timestamp": timestamp.isoformat(),
        "sequence": sequence,
        "event_type": EventType.REASONING.value,
        "payload": payload,
    }


def create_error_event(
    task_id: str = "task-001",
    sequence: int = 0,
    error_type: str = "RuntimeError",
    message: str = "An error occurred",
    recoverable: bool = True,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Create an error event.

    Args:
        task_id: The task identifier.
        sequence: Event sequence number.
        error_type: Type/class of the error.
        message: Error message.
        recoverable: Whether the error is recoverable.
        timestamp: Event timestamp (defaults to now).

    Returns:
        Dictionary representing the event.
    """
    if timestamp is None:
        timestamp = datetime.now()

    return {
        "version": "1.0",
        "task_id": task_id,
        "timestamp": timestamp.isoformat(),
        "sequence": sequence,
        "event_type": EventType.ERROR.value,
        "payload": {
            "error_type": error_type,
            "message": message,
            "recoverable": recoverable,
        },
    }


def create_progress_event(
    task_id: str = "task-001",
    sequence: int = 0,
    current_step: int | None = None,
    percentage: float | None = None,
    message: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Create a progress event.

    Args:
        task_id: The task identifier.
        sequence: Event sequence number.
        current_step: Current step number.
        percentage: Progress percentage (0-100).
        message: Progress message.
        timestamp: Event timestamp (defaults to now).

    Returns:
        Dictionary representing the event.
    """
    if timestamp is None:
        timestamp = datetime.now()

    payload: dict[str, Any] = {}
    if current_step is not None:
        payload["current_step"] = current_step
    if percentage is not None:
        payload["percentage"] = percentage
    if message is not None:
        payload["message"] = message

    return {
        "version": "1.0",
        "task_id": task_id,
        "timestamp": timestamp.isoformat(),
        "sequence": sequence,
        "event_type": EventType.PROGRESS.value,
        "payload": payload,
    }


def generate_realistic_event_sequence(
    task_id: str = "task-001",
    num_steps: int = 5,
    include_error: bool = False,
    start_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Generate a realistic sequence of events for testing.

    Creates a sequence that mimics typical agent behavior:
    1. Start with progress event
    2. Alternate between reasoning, LLM requests, and tool calls
    3. Optionally include an error
    4. End with completion progress

    Args:
        task_id: The task identifier.
        num_steps: Number of main steps (tool calls).
        include_error: Whether to include a recoverable error.
        start_time: Starting timestamp (defaults to now).

    Returns:
        List of event dictionaries.
    """
    if start_time is None:
        start_time = datetime.now()

    events: list[dict[str, Any]] = []
    sequence = 0
    current_time = start_time

    # Start progress
    events.append(
        create_progress_event(
            task_id=task_id,
            sequence=sequence,
            current_step=0,
            percentage=0.0,
            message="Starting task execution",
            timestamp=current_time,
        )
    )
    sequence += 1
    current_time += timedelta(milliseconds=100)

    # Tool names for variety
    tools = ["web_search", "file_read", "code_exec", "api_call", "write_file"]

    for step in range(num_steps):
        # Reasoning before action
        events.append(
            create_reasoning_event(
                task_id=task_id,
                sequence=sequence,
                thought=f"Analyzing step {step + 1}",
                plan=f"Execute {tools[step % len(tools)]} to gather information",
                step=f"Step {step + 1} of {num_steps}",
                timestamp=current_time,
            )
        )
        sequence += 1
        current_time += timedelta(milliseconds=200)

        # LLM request
        events.append(
            create_llm_request_event(
                task_id=task_id,
                sequence=sequence,
                input_tokens=500 + step * 100,
                output_tokens=200 + step * 50,
                duration_ms=1500.0 + step * 200,
                timestamp=current_time,
            )
        )
        sequence += 1
        current_time += timedelta(milliseconds=1500 + step * 200)

        # Tool call
        tool_name = tools[step % len(tools)]
        events.append(
            create_tool_call_event(
                task_id=task_id,
                sequence=sequence,
                tool=tool_name,
                input_data={"step": step + 1, "action": f"perform_{tool_name}"},
                output_data={"result": "success", "data": f"result_{step + 1}"},
                duration_ms=500.0 + step * 100,
                status="success",
                timestamp=current_time,
            )
        )
        sequence += 1
        current_time += timedelta(milliseconds=500 + step * 100)

        # Progress update
        progress = ((step + 1) / num_steps) * 100
        events.append(
            create_progress_event(
                task_id=task_id,
                sequence=sequence,
                current_step=step + 1,
                percentage=min(progress, 99.0),  # Keep below 100 until done
                message=f"Completed step {step + 1}",
                timestamp=current_time,
            )
        )
        sequence += 1
        current_time += timedelta(milliseconds=50)

        # Optional error (at step 3 if enabled)
        if include_error and step == 2:
            events.append(
                create_error_event(
                    task_id=task_id,
                    sequence=sequence,
                    error_type="RateLimitError",
                    message="API rate limit exceeded, retrying in 5 seconds",
                    recoverable=True,
                    timestamp=current_time,
                )
            )
            sequence += 1
            current_time += timedelta(seconds=5)

    # Final progress
    events.append(
        create_progress_event(
            task_id=task_id,
            sequence=sequence,
            current_step=num_steps,
            percentage=100.0,
            message="Task completed successfully",
            timestamp=current_time,
        )
    )

    return events


# Pre-built sample events for common test scenarios
SAMPLE_EVENTS = {
    "simple_success": generate_realistic_event_sequence(
        task_id="simple-success",
        num_steps=3,
        include_error=False,
    ),
    "with_error": generate_realistic_event_sequence(
        task_id="with-error",
        num_steps=5,
        include_error=True,
    ),
    "long_sequence": generate_realistic_event_sequence(
        task_id="long-sequence",
        num_steps=10,
        include_error=False,
    ),
    "minimal": [
        create_progress_event(
            task_id="minimal",
            sequence=0,
            percentage=0.0,
            message="Start",
        ),
        create_tool_call_event(
            task_id="minimal",
            sequence=1,
            tool="simple_tool",
            status="success",
        ),
        create_progress_event(
            task_id="minimal",
            sequence=2,
            percentage=100.0,
            message="Done",
        ),
    ],
    "error_only": [
        create_progress_event(
            task_id="error-only",
            sequence=0,
            percentage=0.0,
            message="Start",
        ),
        create_error_event(
            task_id="error-only",
            sequence=1,
            error_type="AuthenticationError",
            message="Invalid API key",
            recoverable=False,
        ),
    ],
}
